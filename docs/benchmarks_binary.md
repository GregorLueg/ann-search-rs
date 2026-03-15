## Binarised indices benchmarks and parameter

Binarised indices compress the data stored in the index structure itself via
very aggressive quantisation to (basically) only bits. This has two impacts:

1. Drastic reduction in memory fingerprint of the index itself.
2. Increased query speed in most cases as the bit-wise operations are very
fast on modern CPUs.
3. However, when not using any re-ranking of the top candidates, dramatically
lower recall (less so for RaBitQ, an excellent way of compressing vectors).

The benchmarks below show scenarios with and without re-ranking.

```bash
cargo run --example gridsearch_binary --release --features binary
```

If you wish to run all of the benchmarks, below, you can just run:

```bash
bash ./examples/run_benchmarks.sh --binary
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided.

## Table of Contents

- [Binarisation](#binary-ivf-and-exhaustive)
- [RaBitQ](#rabitq-ivf-and-exhaustive)

### <u>Binary (IVF and exhaustive)</u>

Three binarisations are offered in this crate:

- **SimHash**: Projects vectors onto random hyperplanes and encodes the sign of
  each projection as a bit. The random planes are orthogonalised to improve
  coverage of the vector space.
- **ITQ (Iterative Quantisation)**: Uses PCA to find the axes of maximum
  variance in the data, then iteratively rotates the data to minimise
  quantisation error before binarising. This is more expensive to build but
  tends to yield better recall than random projections pending the data
  structure (it's not a must!).
- **Signed**: Simply encodes the sign of each embedding dimension directly as
  a bit, meaning n_bits is fixed to the number of dimensions. Straightforward
  but only sensible for high-dimensional data; at low dimensionality the recall
  degrades quickly.

These indices have the option to use a VecStore that saves the original data on
disk for fast retrieval and re-ranking. This is recommended if you wish to
maintain reasonable recall. Generally speaking, these indices shine in very
high-dimensional data where memory requirements become constraining.

**Key parameters *(general)*:**

- *n_bits*: Into how many bits to encode the data. The binariser has two
  different options here to generate the bits (more on that later). As one
  can appreciate the higher the number, the better the Recall.
- *binarisation_init*: Three options are provided in the crate. `"random"` that
  generates random planes that are subsequently orthogonalised, `"itq"` that
  leverages PCA to identify axis of maximum variation or `"signed"` that just
  uses the sign of the respective embedding dimensions. In this case, `n_bits`
  is set automatically to `n_dim`. Signed only really makes sense if you have
  a lot of dimensions; otherwise, the performance is not good (at all).
- *reranking*: The Binary indices have the option to store the original vectors
  on disk. Once Hamming distance has been leveraged to identify the most
  interesting potential neighbours, the on-disk vectors are loaded in and the
  results are re-ranked. A key parameter here is the reranking_factor, i.e.,
  how many more vectors are reranked than the desired k. For example 10 means
  that `10 * k vectors` are scored and then re-ranked. The more candidates you
  allow here, the better the Recall.

**Key parameters *(IVF-specific)*:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

The self queries (i.e., kNN generation ) are done with `reranking_factor = 10`.

#### With 32 dimensions

The performance of the binarisation is very dependent on the underlying
data. For some of the datasets we still reach decent Recalls of ≥0.8 in some
configurations; for others not at all and the Recall rapidly drops to ~0.5
and worse.

<details>
<summary><b>Binary - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.16     1_526.64     1_529.79       1.0000     0.000000        18.31
Exhaustive (self)                                          3.16    16_212.62    16_215.78       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_160.46       490.93     1_651.39       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_160.46       552.51     1_712.97       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_160.46       589.50     1_749.96       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_160.46       695.81     1_856.27       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                     1_160.46     5_996.37     7_156.83       0.5461     2.056589         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_362.43       488.94     1_851.36       0.1873          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_362.43       542.32     1_904.75       0.3844     4.171681         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_362.43       585.91     1_948.34       0.5073     2.366457         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_362.43       681.03     2_043.45       0.6439     1.245957         4.61
ExhaustiveBinary-256-itq (self)                        1_362.43     5_872.02     7_234.45       0.5079     2.339769         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_306.26       830.51     3_136.76       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_306.26       864.77     3_171.02       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_306.26       936.20     3_242.46       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_306.26     1_017.12     3_323.37       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                     2_306.26     9_207.21    11_513.47       0.7223     0.828915         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_591.30       818.70     3_410.00       0.2823          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_591.30       863.38     3_454.68       0.5675     1.840282         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_591.30       915.13     3_506.43       0.7062     0.870974         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_591.30     1_036.09     3_627.39       0.8300     0.361878         9.22
ExhaustiveBinary-512-itq (self)                        2_591.30     9_188.83    11_780.13       0.7065     0.858252         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 158.95       613.55       772.50       0.0578          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   158.95       640.66       799.61       0.1270    15.329844         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  158.95       681.63       840.58       0.1824    10.683378         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  158.95       756.74       915.69       0.2652     7.182754         0.58
ExhaustiveBinary-32-signed (self)                        158.95     6_877.04     7_035.99       0.1841    10.412094         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           1_941.44        53.47     1_994.91       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           1_941.44        61.02     2_002.46       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           1_941.44        75.36     2_016.80       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           1_941.44        82.03     2_023.47       0.4237     3.638423         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          1_941.44       111.10     2_052.54       0.5531     2.019567         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          1_941.44       156.22     2_097.66       0.6927     1.019852         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           1_941.44        90.71     2_032.15       0.4219     3.666926         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          1_941.44       119.97     2_061.41       0.5516     2.031950         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          1_941.44       170.18     2_111.62       0.6910     1.025282         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           1_941.44       107.92     2_049.36       0.4194     3.697758         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          1_941.44       139.65     2_081.09       0.5486     2.056576         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          1_941.44       194.23     2_135.67       0.6878     1.040576         5.79
IVF-Binary-256-nl273-random (self)                     1_941.44     1_205.36     3_146.80       0.5511     2.018480         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           2_243.18        53.63     2_296.80       0.2064          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_243.18        66.71     2_309.89       0.2048          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_243.18        80.55     2_323.72       0.4237     3.634957         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_243.18       108.35     2_351.53       0.5534     2.012177         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_243.18       152.38     2_395.56       0.6933     1.015195         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_243.18        94.16     2_337.34       0.4202     3.684317         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_243.18       122.48     2_365.66       0.5495     2.044215         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_243.18       173.52     2_416.70       0.6887     1.035562         5.80
IVF-Binary-256-nl387-random (self)                     2_243.18     1_077.56     3_320.74       0.5532     1.998143         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           2_675.66        51.78     2_727.44       0.2075          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_675.66        55.86     2_731.52       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_675.66        64.58     2_740.24       0.2052          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_675.66        88.32     2_763.98       0.4263     3.598411         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_675.66       103.92     2_779.58       0.5571     1.985476         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_675.66       158.96     2_834.62       0.6973     0.994908         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_675.66        82.74     2_758.40       0.4236     3.632258         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_675.66       112.11     2_787.77       0.5537     2.010509         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_675.66       155.97     2_831.63       0.6941     1.006790         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_675.66        90.05     2_765.71       0.4210     3.674864         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_675.66       119.66     2_795.32       0.5506     2.036845         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_675.66       176.43     2_852.09       0.6901     1.028662         5.82
IVF-Binary-256-nl547-random (self)                     2_675.66     1_040.40     3_716.06       0.5570     1.967753         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_120.51        52.70     2_173.21       0.1905          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_120.51        58.87     2_179.37       0.1893          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_120.51        75.08     2_195.58       0.1883          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_120.51        80.26     2_200.77       0.3914     4.063042         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_120.51       107.31     2_227.81       0.5149     2.303126         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_120.51       153.50     2_274.00       0.6520     1.207540         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_120.51        87.76     2_208.26       0.3888     4.102234         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_120.51       118.20     2_238.71       0.5127     2.322490         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_120.51       165.98     2_286.49       0.6497     1.217388         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_120.51       104.00     2_224.51       0.3863     4.144271         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_120.51       135.39     2_255.89       0.5094     2.347611         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_120.51       186.65     2_307.16       0.6461     1.235595         5.79
IVF-Binary-256-nl273-itq (self)                        2_120.51     1_165.85     3_286.35       0.5137     2.293500         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_445.02        62.42     2_507.44       0.1902          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_445.02        74.94     2_519.96       0.1886          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_445.02        88.56     2_533.58       0.3907     4.063618         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_445.02       114.41     2_559.44       0.5153     2.298229         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_445.02       160.42     2_605.44       0.6520     1.205868         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_445.02       111.61     2_556.63       0.3868     4.129609         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_445.02       142.54     2_587.57       0.5107     2.334748         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_445.02       181.20     2_626.22       0.6474     1.227411         5.80
IVF-Binary-256-nl387-itq (self)                        2_445.02     1_138.55     3_583.57       0.5160     2.270526         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)              2_868.90        52.54     2_921.44       0.1917          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              2_868.90        56.28     2_925.18       0.1903          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              2_868.90        62.47     2_931.37       0.1892          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              2_868.90        78.49     2_947.39       0.3939     4.017812         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             2_868.90       104.57     2_973.47       0.5191     2.263756         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             2_868.90       147.52     3_016.42       0.6568     1.180336         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              2_868.90        82.59     2_951.49       0.3910     4.059910         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             2_868.90       109.02     2_977.92       0.5156     2.292191         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             2_868.90       155.87     3_024.77       0.6528     1.196090         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              2_868.90        89.91     2_958.81       0.3882     4.109983         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             2_868.90       117.48     2_986.38       0.5117     2.327676         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             2_868.90       164.65     3_033.55       0.6488     1.219085         5.82
IVF-Binary-256-nl547-itq (self)                        2_868.90     1_033.00     3_901.90       0.5202     2.235200         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_113.63       102.32     3_215.95       0.2928          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_113.63       112.07     3_225.70       0.2922          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_113.63       141.63     3_255.26       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_113.63       138.23     3_251.86       0.5865     1.777085        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_113.63       173.15     3_286.78       0.7256     0.833652        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_113.63       229.32     3_342.95       0.8431     0.358751        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_113.63       153.93     3_267.56       0.5863     1.774697        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_113.63       191.59     3_305.22       0.7262     0.823205        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_113.63       252.01     3_365.64       0.8454     0.341524        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_113.63       191.22     3_304.85       0.5847     1.785669        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_113.63       229.55     3_343.18       0.7248     0.828478        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_113.63       294.06     3_407.69       0.8446     0.342891        10.40
IVF-Binary-512-nl273-random (self)                     3_113.63     1_911.39     5_025.02       0.7246     0.820201        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)           3_387.79       104.65     3_492.44       0.2927          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_387.79       130.58     3_518.37       0.2916          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_387.79       143.87     3_531.66       0.5870     1.770037        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_387.79       179.34     3_567.13       0.7265     0.828245        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_387.79       231.61     3_619.40       0.8449     0.354020        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_387.79       171.07     3_558.86       0.5852     1.780085        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_387.79       209.61     3_597.40       0.7252     0.826154        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_387.79       271.66     3_659.45       0.8452     0.340889        10.41
IVF-Binary-512-nl387-random (self)                     3_387.79     1_783.56     5_171.35       0.7248     0.826965        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)           3_836.45        96.26     3_932.71       0.2935          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           3_836.45       105.20     3_941.65       0.2928          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           3_836.45       119.35     3_955.80       0.2920          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           3_836.45       131.99     3_968.44       0.5884     1.762465        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          3_836.45       157.69     3_994.14       0.7278     0.824517        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          3_836.45       207.01     4_043.46       0.8457     0.354496        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           3_836.45       138.85     3_975.30       0.5872     1.765331        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          3_836.45       168.97     4_005.42       0.7273     0.819011        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          3_836.45       220.39     4_056.83       0.8465     0.341188        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           3_836.45       165.15     4_001.60       0.5853     1.778278        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          3_836.45       189.39     4_025.84       0.7260     0.822615        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          3_836.45       240.31     4_076.76       0.8456     0.339646        10.43
IVF-Binary-512-nl547-random (self)                     3_836.45     1_576.06     5_412.51       0.7267     0.818703        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_357.61       102.02     3_459.63       0.2843          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_357.61       116.36     3_473.96       0.2839          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_357.61       146.02     3_503.62       0.2830          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_357.61       134.85     3_492.46       0.5703     1.827586        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_357.61       184.99     3_542.60       0.7081     0.873482        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_357.61       217.83     3_575.44       0.8300     0.375367        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_357.61       154.51     3_512.12       0.5702     1.824096        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_357.61       186.73     3_544.34       0.7086     0.862286        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_357.61       240.98     3_598.59       0.8320     0.358562        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_357.61       184.85     3_542.46       0.5688     1.832095        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_357.61       219.86     3_577.47       0.7071     0.866835        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_357.61       279.31     3_636.92       0.8308     0.360078        10.40
IVF-Binary-512-nl273-itq (self)                        3_357.61     1_820.00     5_177.61       0.7092     0.849450        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_674.85       107.19     3_782.04       0.2844          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_674.85       133.74     3_808.59       0.2834          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_674.85       140.05     3_814.91       0.5707     1.824636        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_674.85       170.28     3_845.13       0.7088     0.870642        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_674.85       228.13     3_902.98       0.8313     0.373148        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_674.85       168.38     3_843.24       0.5692     1.829770        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_674.85       203.81     3_878.66       0.7076     0.864124        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_674.85       256.64     3_931.49       0.8311     0.359219        10.41
IVF-Binary-512-nl387-itq (self)                        3_674.85     1_736.16     5_411.01       0.7092     0.858656        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_126.66        98.89     4_225.55       0.2853          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_126.66       107.84     4_234.50       0.2845          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_126.66       121.78     4_248.44       0.2836          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_126.66       130.80     4_257.46       0.5724     1.811736        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_126.66       162.76     4_289.42       0.7108     0.864355        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_126.66       209.47     4_336.13       0.8323     0.372880        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_126.66       141.05     4_267.71       0.5714     1.813983        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_126.66       170.97     4_297.63       0.7102     0.857436        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_126.66       223.63     4_350.29       0.8331     0.359259        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_126.66       155.77     4_282.43       0.5696     1.824932        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_126.66       188.08     4_314.74       0.7083     0.861599        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_126.66       242.65     4_369.31       0.8317     0.357581        10.43
IVF-Binary-512-nl547-itq (self)                        4_126.66     1_600.28     5_726.94       0.7111     0.850696        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)              940.79        40.95       981.74       0.0679          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)              940.79        48.41       989.21       0.0624          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)              940.79        67.08     1_007.87       0.0602          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)              940.79        63.33     1_004.12       0.1447    13.569746         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)             940.79        79.02     1_019.81       0.2059     9.378159         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)             940.79       113.77     1_054.56       0.2962     6.200633         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)              940.79        69.50     1_010.29       0.1366    14.121664         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)             940.79        88.02     1_028.81       0.1960     9.786761         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)             940.79       124.85     1_065.64       0.2849     6.496712         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)              940.79        91.49     1_032.28       0.1323    14.552605         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)             940.79       126.94     1_067.74       0.1900    10.127619         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)             940.79       147.85     1_088.64       0.2759     6.765662         1.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-signed (self)                        940.79       874.63     1_815.42       0.1987     9.540459         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_262.76        42.19     1_304.95       0.0647          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_262.76        55.30     1_318.05       0.0616          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_262.76        60.42     1_323.18       0.1409    13.656099         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_262.76        78.89     1_341.65       0.2038     9.360511         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_262.76       113.02     1_375.77       0.2934     6.165021         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_262.76        74.53     1_337.28       0.1349    14.234154         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_262.76        94.16     1_356.92       0.1944     9.836159         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_262.76       131.40     1_394.16       0.2810     6.532805         1.77
IVF-Binary-32-nl387-signed (self)                      1_262.76       786.24     2_048.99       0.2045     9.174261         1.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_725.98        39.18     1_765.16       0.0670          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_725.98        44.00     1_769.97       0.0650          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_725.98        50.70     1_776.68       0.0627          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_725.98        57.14     1_783.12       0.1458    13.298341         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_725.98        74.99     1_800.97       0.2095     9.093104         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_725.98       108.77     1_834.75       0.3024     5.928558         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_725.98        63.33     1_789.31       0.1413    13.650579         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_725.98        80.77     1_806.75       0.2032     9.392887         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_725.98       114.74     1_840.72       0.2937     6.177874         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_725.98        69.96     1_795.94       0.1373    14.019250         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_725.98        88.72     1_814.69       0.1971     9.710259         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_725.98       123.58     1_849.56       0.2845     6.443362         1.79
IVF-Binary-32-nl547-signed (self)                      1_725.98       744.73     2_470.70       0.2109     8.898138         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.85     1_565.91     1_569.76       1.0000     0.000000        18.88
Exhaustive (self)                                          3.85    16_276.85    16_280.70       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_162.09       497.31     1_659.40       0.2158          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_162.09       548.92     1_711.01       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_162.09       602.59     1_764.69       0.5705     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_162.09       683.36     1_845.45       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                     1_162.09     5_883.59     7_045.68       0.5696     0.001382         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_356.62       482.82     1_839.45       0.1983          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_356.62       531.85     1_888.47       0.4042     0.002874         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_356.62       579.13     1_935.75       0.5292     0.001626         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_356.62       676.90     2_033.53       0.6649     0.000853         4.61
ExhaustiveBinary-256-itq (self)                        1_356.62     5_809.01     7_165.63       0.5290     0.001611         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_307.90       822.29     3_130.19       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_307.90       863.28     3_171.18       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_307.90       914.52     3_222.42       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_307.90     1_028.72     3_336.62       0.8652     0.000203         9.22
ExhaustiveBinary-512-random (self)                     2_307.90     9_153.13    11_461.02       0.7519     0.000503         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_591.70       825.77     3_417.47       0.3035          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_591.70       866.10     3_457.80       0.5988     0.001175         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_591.70       931.87     3_523.57       0.7344     0.000548         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_591.70     1_019.89     3_611.59       0.8497     0.000226         9.22
ExhaustiveBinary-512-itq (self)                        2_591.70     9_161.15    11_752.85       0.7336     0.000546         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 159.35       606.85       766.20       0.0588          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   159.35       641.50       800.85       0.1297     0.011294         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  159.35       681.12       840.47       0.1862     0.007851         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  159.35       756.77       916.13       0.2706     0.005257         0.58
ExhaustiveBinary-32-signed (self)                        159.35     6_799.28     6_958.63       0.1885     0.007670         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           1_899.64        65.85     1_965.49       0.2193          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           1_899.64        68.00     1_967.63       0.2181          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           1_899.64        79.83     1_979.47       0.2170          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           1_899.64        81.79     1_981.43       0.4460     0.002467         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          1_899.64       107.63     2_007.27       0.5773     0.001353         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          1_899.64       154.61     2_054.25       0.7143     0.000676         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           1_899.64        88.57     1_988.21       0.4441     0.002485         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          1_899.64       117.98     2_017.61       0.5757     0.001359         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          1_899.64       167.11     2_066.75       0.7130     0.000677         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           1_899.64       105.43     2_005.07       0.4416     0.002508         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          1_899.64       137.64     2_037.27       0.5727     0.001375         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          1_899.64       191.12     2_090.75       0.7101     0.000688         5.79
IVF-Binary-256-nl273-random (self)                     1_899.64     1_183.25     3_082.88       0.5747     0.001354         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           2_186.86        61.33     2_248.19       0.2185          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)           2_186.86        72.35     2_259.21       0.2168          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)           2_186.86        86.58     2_273.44       0.4459     0.002463         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)          2_186.86       113.06     2_299.92       0.5775     0.001347         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)          2_186.86       158.06     2_344.92       0.7151     0.000670         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)           2_186.86       103.27     2_290.13       0.4423     0.002500         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)          2_186.86       133.09     2_319.95       0.5735     0.001369         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)          2_186.86       183.39     2_370.25       0.7108     0.000685         5.81
IVF-Binary-256-nl387-random (self)                     2_186.86     1_130.92     3_317.78       0.5768     0.001341         5.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           2_598.62        58.77     2_657.39       0.2199          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)           2_598.62        62.94     2_661.56       0.2186          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)           2_598.62        71.13     2_669.75       0.2176          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)           2_598.62        85.52     2_684.14       0.4486     0.002437         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)          2_598.62       109.96     2_708.58       0.5813     0.001326         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)          2_598.62       153.01     2_751.63       0.7190     0.000657         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)           2_598.62        91.03     2_689.65       0.4460     0.002463         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)          2_598.62       117.08     2_715.70       0.5779     0.001344         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)          2_598.62       161.71     2_760.33       0.7158     0.000666         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)           2_598.62       100.15     2_698.77       0.4434     0.002490         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)          2_598.62       127.50     2_726.12       0.5745     0.001364         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)          2_598.62       174.57     2_773.19       0.7121     0.000680         5.83
IVF-Binary-256-nl547-random (self)                     2_598.62     1_104.85     3_703.47       0.5803     0.001321         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_085.87        58.34     2_144.21       0.2019          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_085.87        65.97     2_151.84       0.2009          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_085.87        83.80     2_169.67       0.1996          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_085.87        87.64     2_173.50       0.4111     0.002797         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_085.87       114.81     2_200.68       0.5370     0.001578         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_085.87       163.22     2_249.09       0.6729     0.000826         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_085.87        95.83     2_181.70       0.4089     0.002823         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_085.87       126.39     2_212.26       0.5344     0.001593         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_085.87       178.69     2_264.56       0.6711     0.000830         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_085.87       117.34     2_203.20       0.4063     0.002850         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_085.87       149.05     2_234.92       0.5313     0.001613         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_085.87       207.86     2_293.73       0.6672     0.000844         5.79
IVF-Binary-256-nl273-itq (self)                        2_085.87     1_256.41     3_342.28       0.5347     0.001578         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_377.92        54.61     2_432.54       0.2015          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_377.92        65.35     2_443.27       0.1997          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_377.92        81.50     2_459.43       0.4117     0.002789         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_377.92       107.72     2_485.65       0.5373     0.001574         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_377.92       152.62     2_530.55       0.6736     0.000819         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_377.92        94.95     2_472.88       0.4072     0.002839         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_377.92       123.61     2_501.53       0.5325     0.001604         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_377.92       174.34     2_552.27       0.6684     0.000837         5.81
IVF-Binary-256-nl387-itq (self)                        2_377.92     1_072.24     3_450.17       0.5372     0.001560         5.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)              2_797.87        56.02     2_853.89       0.2032          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)              2_797.87        60.40     2_858.27       0.2015          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)              2_797.87        67.81     2_865.68       0.2002          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)              2_797.87        89.41     2_887.28       0.4146     0.002757         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)             2_797.87       115.78     2_913.66       0.5413     0.001548         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)             2_797.87       159.22     2_957.09       0.6779     0.000803         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)              2_797.87        91.01     2_888.88       0.4116     0.002790         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)             2_797.87       119.66     2_917.53       0.5376     0.001571         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)             2_797.87       168.76     2_966.63       0.6742     0.000815         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)              2_797.87       117.17     2_915.04       0.4085     0.002825         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)             2_797.87       128.13     2_926.00       0.5336     0.001598         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)             2_797.87       179.41     2_977.28       0.6700     0.000832         5.83
IVF-Binary-256-nl547-itq (self)                        2_797.87     1_119.98     3_917.85       0.5410     0.001537         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_040.70       106.20     3_146.89       0.3155          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_040.70       121.90     3_162.60       0.3150          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_040.70       159.23     3_199.93       0.3143          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_040.70       139.45     3_180.15       0.6202     0.001095        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_040.70       169.52     3_210.22       0.7548     0.000507        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_040.70       221.53     3_262.23       0.8638     0.000215        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_040.70       157.58     3_198.28       0.6201     0.001092        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_040.70       192.18     3_232.88       0.7559     0.000497        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_040.70       255.19     3_295.88       0.8665     0.000201        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_040.70       197.85     3_238.55       0.6189     0.001099        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_040.70       233.25     3_273.95       0.7548     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_040.70       295.78     3_336.48       0.8659     0.000201        10.40
IVF-Binary-512-nl273-random (self)                     3_040.70     1_890.19     4_930.89       0.7541     0.000497        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)           3_338.63       113.92     3_452.56       0.3154          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_338.63       138.59     3_477.23       0.3144          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_338.63       149.18     3_487.82       0.6210     0.001090        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_338.63       183.33     3_521.96       0.7559     0.000502        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_338.63       237.61     3_576.24       0.8653     0.000209        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_338.63       185.94     3_524.58       0.6193     0.001096        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_338.63       219.73     3_558.36       0.7552     0.000499        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_338.63       281.46     3_620.10       0.8661     0.000201        10.41
IVF-Binary-512-nl387-random (self)                     3_338.63     1_838.56     5_177.20       0.7539     0.000502        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)           3_759.71       103.30     3_863.00       0.3162          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)           3_759.71       111.01     3_870.72       0.3157          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)           3_759.71       126.62     3_886.32       0.3148          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)           3_759.71       139.72     3_899.43       0.6225     0.001082        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)          3_759.71       185.83     3_945.54       0.7572     0.000498        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)          3_759.71       222.22     3_981.92       0.8660     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)           3_759.71       150.32     3_910.02       0.6214     0.001086        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)          3_759.71       184.59     3_944.29       0.7570     0.000495        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)          3_759.71       238.11     3_997.81       0.8670     0.000202        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)           3_759.71       167.89     3_927.59       0.6198     0.001094        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)          3_759.71       203.80     3_963.50       0.7556     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)          3_759.71       261.29     4_020.99       0.8666     0.000200        10.44
IVF-Binary-512-nl547-random (self)                     3_759.71     1_707.64     5_467.34       0.7554     0.000498        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_316.08       108.77     3_424.85       0.3057          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_316.08       124.35     3_440.43       0.3051          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_316.08       161.87     3_477.95       0.3043          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_316.08       142.50     3_458.58       0.6012     0.001167        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_316.08       172.40     3_488.49       0.7359     0.000550        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_316.08       225.53     3_541.61       0.8491     0.000237        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_316.08       160.49     3_476.57       0.6011     0.001164        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_316.08       193.82     3_509.90       0.7364     0.000542        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_316.08       247.32     3_563.40       0.8513     0.000224        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_316.08       201.06     3_517.15       0.6000     0.001169        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_316.08       236.34     3_552.43       0.7355     0.000544        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_316.08       317.96     3_634.04       0.8503     0.000225        10.40
IVF-Binary-512-nl273-itq (self)                        3_316.08     1_924.78     5_240.86       0.7360     0.000539        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_614.82       113.63     3_728.45       0.3056          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_614.82       143.03     3_757.85       0.3044          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_614.82       147.93     3_762.74       0.6017     0.001163        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_614.82       176.72     3_791.54       0.7367     0.000545        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_614.82       228.26     3_843.08       0.8505     0.000231        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_614.82       179.83     3_794.65       0.6001     0.001168        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_614.82       212.96     3_827.78       0.7357     0.000543        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_614.82       267.44     3_882.26       0.8505     0.000224        10.41
IVF-Binary-512-nl387-itq (self)                        3_614.82     1_783.78     5_398.60       0.7359     0.000544        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_049.58       103.03     4_152.61       0.3063          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_049.58       122.04     4_171.62       0.3054          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_049.58       138.62     4_188.20       0.3045          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_049.58       139.05     4_188.63       0.6033     0.001155        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_049.58       164.60     4_214.18       0.7380     0.000543        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_049.58       230.31     4_279.89       0.8514     0.000231        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_049.58       147.11     4_196.69       0.6023     0.001158        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_049.58       176.95     4_226.53       0.7376     0.000539        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_049.58       229.25     4_278.83       0.8521     0.000223        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_049.58       163.15     4_212.72       0.6008     0.001164        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_049.58       204.81     4_254.39       0.7364     0.000541        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_049.58       270.51     4_320.09       0.8512     0.000223        10.44
IVF-Binary-512-nl547-itq (self)                        4_049.58     1_638.64     5_688.22       0.7374     0.000541        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)              898.65        40.52       939.17       0.0688          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)              898.65        48.58       947.24       0.0632          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)              898.65        64.85       963.50       0.0610          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)              898.65        59.37       958.02       0.1488     0.009993         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)             898.65        79.90       978.55       0.2108     0.006867         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)             898.65       113.42     1_012.08       0.3022     0.004533         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)              898.65        67.24       965.89       0.1404     0.010416         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)             898.65        88.66       987.32       0.2007     0.007189         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)             898.65       124.31     1_022.96       0.2905     0.004741         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)              898.65        85.87       984.52       0.1354     0.010757         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)             898.65       107.88     1_006.53       0.1934     0.007454         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)             898.65       146.30     1_044.95       0.2806     0.004966         1.76
IVF-Binary-32-nl273-signed (self)                        898.65       891.98     1_790.63       0.2029     0.007019         1.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_198.12        42.19     1_240.31       0.0656          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_198.12        56.41     1_254.54       0.0625          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_198.12        60.74     1_258.86       0.1445     0.010057         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_198.12        79.19     1_277.31       0.2081     0.006872         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_198.12       130.90     1_329.03       0.3001     0.004518         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_198.12        75.83     1_273.96       0.1372     0.010530         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_198.12        96.30     1_294.42       0.1979     0.007261         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_198.12       130.37     1_328.50       0.2858     0.004815         1.77
IVF-Binary-32-nl387-signed (self)                      1_198.12       793.26     1_991.39       0.2098     0.006741         1.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_652.29        39.89     1_692.18       0.0677          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_652.29        43.74     1_696.02       0.0655          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_652.29        53.52     1_705.80       0.0635          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_652.29        57.57     1_709.86       0.1486     0.009791         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_652.29        76.50     1_728.79       0.2131     0.006681         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_652.29       109.52     1_761.81       0.3085     0.004339         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_652.29        61.80     1_714.09       0.1443     0.010053         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_652.29        80.08     1_732.37       0.2068     0.006903         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_652.29       114.03     1_766.31       0.2993     0.004526         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_652.29        69.56     1_721.85       0.1397     0.010380         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_652.29        88.14     1_740.43       0.2001     0.007158         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_652.29       123.22     1_775.50       0.2893     0.004724         1.79
IVF-Binary-32-nl547-signed (self)                      1_652.29       744.40     2_396.69       0.2155     0.006553         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         2.96     1_575.98     1_578.94       1.0000     0.000000        18.31
Exhaustive (self)                                          2.96    16_510.17    16_513.13       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_155.97       486.10     1_642.07       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_155.97       539.60     1_695.57       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_155.97       619.33     1_775.31       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_155.97       667.86     1_823.83       0.5614     0.256662         4.61
ExhaustiveBinary-256-random (self)                     1_155.97     5_781.42     6_937.40       0.4294     0.437191         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_359.10       488.36     1_847.47       0.1323          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_359.10       525.00     1_884.11       0.3010     0.745496         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_359.10       571.54     1_930.64       0.4149     0.448765         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_359.10       681.50     2_040.61       0.5534     0.252544         4.61
ExhaustiveBinary-256-itq (self)                        1_359.10     5_694.74     7_053.84       0.4203     0.435859         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_303.82       823.17     3_126.99       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_303.82       864.07     3_167.89       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_303.82       924.08     3_227.89       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_303.82     1_012.54     3_316.35       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                     2_303.82     9_127.62    11_431.44       0.6185     0.184365         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_581.66       824.66     3_406.32       0.2201          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_581.66       873.30     3_454.96       0.4659     0.364676         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_581.66       922.95     3_504.61       0.6031     0.194711         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_581.66     1_016.51     3_598.17       0.7399     0.094579         9.22
ExhaustiveBinary-512-itq (self)                        2_581.66     9_162.83    11_744.49       0.6053     0.192118         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 156.61       616.34       772.95       0.0127          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   156.61       629.52       786.13       0.0523     2.784656         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  156.61       659.58       816.19       0.0945     1.948956         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  156.61       743.79       900.40       0.1639     1.328894         0.58
ExhaustiveBinary-32-signed (self)                        156.61     6_669.36     6_825.97       0.0975     1.917524         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           1_938.78        48.96     1_987.74       0.1441          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           1_938.78        56.45     1_995.23       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           1_938.78        73.57     2_012.35       0.1372          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           1_938.78        73.44     2_012.22       0.3248     0.694129         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          1_938.78        95.48     2_034.26       0.4466     0.414033         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          1_938.78       140.16     2_078.94       0.5872     0.229701         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           1_938.78        82.91     2_021.69       0.3162     0.723258         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          1_938.78       107.64     2_046.42       0.4366     0.432150         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          1_938.78       151.99     2_090.77       0.5767     0.241241         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           1_938.78       104.09     2_042.87       0.3125     0.740161         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          1_938.78       127.66     2_066.44       0.4310     0.444567         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          1_938.78       174.73     2_113.52       0.5701     0.249304         5.79
IVF-Binary-256-nl273-random (self)                     1_938.78     1_074.70     3_013.48       0.4421     0.417017         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           2_224.62        54.08     2_278.70       0.1402          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_224.62        65.46     2_290.08       0.1379          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_224.62        77.00     2_301.61       0.3201     0.708580         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_224.62       100.53     2_325.14       0.4412     0.421108         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_224.62       139.80     2_364.42       0.5822     0.233784         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_224.62        91.59     2_316.21       0.3145     0.731502         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_224.62       117.27     2_341.89       0.4333     0.437798         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_224.62       160.38     2_384.99       0.5721     0.245997         5.80
IVF-Binary-256-nl387-random (self)                     2_224.62     1_005.99     3_230.61       0.4462     0.407720         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           2_666.87        51.18     2_718.06       0.1413          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_666.87        54.19     2_721.06       0.1399          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_666.87        60.48     2_727.35       0.1382          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_666.87        73.14     2_740.01       0.3224     0.696633         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_666.87        94.93     2_761.81       0.4452     0.411720         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_666.87       133.80     2_800.67       0.5868     0.227456         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_666.87        77.57     2_744.44       0.3185     0.710966         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_666.87       101.07     2_767.95       0.4401     0.422530         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_666.87       140.86     2_807.73       0.5800     0.235760         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_666.87        85.34     2_752.22       0.3148     0.727129         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_666.87       116.68     2_783.55       0.4348     0.434719         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_666.87       150.99     2_817.87       0.5737     0.244079         5.82
IVF-Binary-256-nl547-random (self)                     2_666.87       948.86     3_615.74       0.4497     0.399649         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_123.89        52.16     2_176.05       0.1410          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_123.89        62.03     2_185.92       0.1357          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_123.89        76.25     2_200.14       0.1338          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_123.89        76.42     2_200.31       0.3208     0.685541         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_123.89        99.79     2_223.68       0.4396     0.407217         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_123.89       141.05     2_264.94       0.5798     0.225195         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_123.89        85.00     2_208.89       0.3109     0.715357         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_123.89       111.17     2_235.06       0.4280     0.426496         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_123.89       154.87     2_278.76       0.5688     0.236011         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_123.89       104.64     2_228.53       0.3066     0.734420         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_123.89       132.76     2_256.66       0.4225     0.438484         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_123.89       180.25     2_304.14       0.5615     0.245100         5.79
IVF-Binary-256-nl273-itq (self)                        2_123.89     1_111.83     3_235.72       0.4337     0.414139         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_438.92        52.42     2_491.34       0.1377          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_438.92        63.97     2_502.89       0.1352          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_438.92        76.50     2_515.42       0.3141     0.702705         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_438.92       100.75     2_539.67       0.4329     0.416658         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_438.92       139.78     2_578.70       0.5744     0.230631         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_438.92        90.58     2_529.50       0.3081     0.726328         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_438.92       115.93     2_554.85       0.4247     0.432890         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_438.92       159.85     2_598.77       0.5644     0.242053         5.80
IVF-Binary-256-nl387-itq (self)                        2_438.92     1_007.25     3_446.17       0.4379     0.405679         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)              2_896.42        50.82     2_947.24       0.1385          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              2_896.42        56.88     2_953.29       0.1368          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              2_896.42        60.90     2_957.32       0.1350          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              2_896.42        73.41     2_969.82       0.3163     0.694441         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             2_896.42        97.80     2_994.22       0.4354     0.411362         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             2_896.42       140.77     3_037.19       0.5786     0.225544         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              2_896.42        79.79     2_976.21       0.3124     0.710008         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             2_896.42       102.27     2_998.69       0.4299     0.422999         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             2_896.42       141.45     3_037.87       0.5716     0.233715         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              2_896.42        84.53     2_980.94       0.3082     0.726527         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             2_896.42       128.91     3_025.33       0.4246     0.434241         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             2_896.42       164.16     3_060.58       0.5647     0.242161         5.82
IVF-Binary-256-nl547-itq (self)                        2_896.42       953.75     3_850.17       0.4408     0.399864         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_070.89       106.79     3_177.68       0.2301          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_070.89       124.99     3_195.88       0.2277          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_070.89       162.79     3_233.68       0.2271          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_070.89       134.30     3_205.19       0.4858     0.337669        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_070.89       171.28     3_242.17       0.6273     0.176951        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_070.89       206.10     3_276.98       0.7651     0.083714        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_070.89       154.23     3_225.11       0.4811     0.344883        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_070.89       183.43     3_254.32       0.6226     0.181279        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_070.89       233.98     3_304.87       0.7604     0.086426        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_070.89       208.66     3_279.55       0.4790     0.349067        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_070.89       231.59     3_302.48       0.6197     0.184441        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_070.89       296.49     3_367.38       0.7568     0.088730        10.40
IVF-Binary-512-nl273-random (self)                     3_070.89     1_827.29     4_898.18       0.6260     0.177568        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)           3_488.20       113.30     3_601.50       0.2288          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_488.20       141.38     3_629.58       0.2273          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_488.20       141.19     3_629.39       0.4834     0.341959        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_488.20       165.44     3_653.64       0.6246     0.179194        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_488.20       218.45     3_706.65       0.7627     0.085009        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_488.20       171.06     3_659.26       0.4805     0.347662        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_488.20       202.87     3_691.07       0.6204     0.183313        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_488.20       250.40     3_738.60       0.7580     0.087911        10.41
IVF-Binary-512-nl387-random (self)                     3_488.20     1_649.15     5_137.35       0.6279     0.175702        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)           3_832.86       102.66     3_935.52       0.2294          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           3_832.86       116.35     3_949.21       0.2285          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           3_832.86       128.36     3_961.21       0.2275          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           3_832.86       132.17     3_965.03       0.4845     0.339823        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          3_832.86       157.86     3_990.72       0.6260     0.178040        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          3_832.86       200.56     4_033.42       0.7648     0.083955        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           3_832.86       141.88     3_974.73       0.4820     0.344380        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          3_832.86       168.75     4_001.61       0.6235     0.180762        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          3_832.86       235.03     4_067.88       0.7614     0.086087        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           3_832.86       170.40     4_003.26       0.4802     0.347987        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          3_832.86       189.47     4_022.33       0.6207     0.183507        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          3_832.86       237.87     4_070.73       0.7586     0.087878        10.43
IVF-Binary-512-nl547-random (self)                     3_832.86     1_558.16     5_391.02       0.6297     0.174170        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_356.15       107.66     3_463.81       0.2255          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_356.15       124.08     3_480.23       0.2227          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_356.15       162.79     3_518.94       0.2216          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_356.15       135.51     3_491.65       0.4774     0.346615        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_356.15       161.65     3_517.80       0.6162     0.182766        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_356.15       213.00     3_569.15       0.7532     0.087053        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_356.15       155.68     3_511.82       0.4725     0.354320        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_356.15       183.99     3_540.14       0.6109     0.187407        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_356.15       234.12     3_590.27       0.7479     0.090065        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_356.15       195.96     3_552.11       0.4701     0.359322        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_356.15       241.93     3_598.08       0.6079     0.190805        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_356.15       286.03     3_642.18       0.7443     0.092319        10.40
IVF-Binary-512-nl273-itq (self)                        3_356.15     1_840.84     5_196.99       0.6128     0.185261        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_673.20       113.78     3_786.99       0.2241          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_673.20       140.03     3_813.24       0.2227          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_673.20       143.09     3_816.30       0.4748     0.350717        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_673.20       166.35     3_839.56       0.6134     0.184894        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_673.20       220.69     3_893.89       0.7503     0.088493        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_673.20       173.63     3_846.84       0.4712     0.357131        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_673.20       199.09     3_872.29       0.6092     0.189326        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_673.20       250.55     3_923.75       0.7456     0.091400        10.41
IVF-Binary-512-nl387-itq (self)                        3_673.20     1_661.29     5_334.50       0.6151     0.183109        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_114.02       103.83     4_217.86       0.2248          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_114.02       113.55     4_227.57       0.2240          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_114.02       128.62     4_242.65       0.2230          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_114.02       136.69     4_250.71       0.4759     0.348742        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_114.02       157.76     4_271.78       0.6141     0.183927        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_114.02       201.63     4_315.65       0.7529     0.087149        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_114.02       142.71     4_256.73       0.4736     0.353096        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_114.02       171.37     4_285.40       0.6111     0.187066        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_114.02       218.44     4_332.47       0.7488     0.089617        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_114.02       159.20     4_273.22       0.4713     0.357356        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_114.02       188.06     4_302.08       0.6084     0.190077        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_114.02       237.78     4_351.80       0.7457     0.091608        10.43
IVF-Binary-512-nl547-itq (self)                        4_114.02     1_571.34     5_685.36       0.6169     0.181390        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)              933.97        37.31       971.28       0.0234          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)              933.97        45.86       979.83       0.0151          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)              933.97        63.34       997.31       0.0136          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)              933.97        48.08       982.05       0.0890     2.423463         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)             933.97        59.61       993.58       0.1529     1.710396         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)             933.97        83.09     1_017.06       0.2508     1.160225         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)              933.97        55.68       989.65       0.0625     2.883751         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)             933.97        68.28     1_002.25       0.1141     2.056626         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)             933.97        93.62     1_027.59       0.1984     1.404948         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)              933.97        75.32     1_009.29       0.0556     3.127341         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)             933.97        90.96     1_024.93       0.1027     2.252976         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)             933.97       116.46     1_050.43       0.1795     1.556631         1.76
IVF-Binary-32-nl273-signed (self)                        933.97       678.14     1_612.11       0.1187     2.029094         1.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_250.92        39.98     1_290.90       0.0176          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_250.92        52.52     1_303.45       0.0148          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_250.92        50.42     1_301.34       0.0717     2.620430         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_250.92        62.55     1_313.47       0.1283     1.859415         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_250.92        87.80     1_338.73       0.2193     1.240960         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_250.92        64.29     1_315.21       0.0609     2.951507         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_250.92        77.65     1_328.58       0.1114     2.110392         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_250.92       101.90     1_352.83       0.1940     1.419800         1.77
IVF-Binary-32-nl387-signed (self)                      1_250.92       615.63     1_866.55       0.1322     1.832312         1.77
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_702.01        37.70     1_739.71       0.0185          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_702.01        42.35     1_744.36       0.0172          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_702.01        48.87     1_750.88       0.0154          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_702.01        48.92     1_750.92       0.0762     2.550798         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_702.01        59.59     1_761.60       0.1351     1.776800         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_702.01        80.80     1_782.80       0.2283     1.159375         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_702.01        53.12     1_755.13       0.0706     2.699178         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_702.01        64.10     1_766.10       0.1257     1.900296         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_702.01        86.16     1_788.17       0.2132     1.248510         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_702.01        62.12     1_764.13       0.0634     2.959131         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_702.01        70.22     1_772.23       0.1138     2.101924         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_702.01        94.16     1_796.16       0.1957     1.387744         1.79
IVF-Binary-32-nl547-signed (self)                      1_702.01       580.38     2_282.39       0.1380     1.740267         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         2.98     1_563.45     1_566.43       1.0000     0.000000        18.31
Exhaustive (self)                                          2.98    16_480.36    16_483.34       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_159.36       481.62     1_640.98       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_159.36       531.90     1_691.25       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_159.36       581.99     1_741.35       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_159.36       658.35     1_817.71       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                     1_159.36     5_729.25     6_888.61       0.3238     3.208083         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_342.14       478.18     1_820.32       0.0723          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_342.14       521.25     1_863.39       0.1941     5.797375         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_342.14       564.62     1_906.76       0.2870     3.686845         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_342.14       651.70     1_993.84       0.4150     2.209544         4.61
ExhaustiveBinary-256-itq (self)                        1_342.14     5_642.00     6_984.14       0.2901     3.621623         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_305.52       821.56     3_127.07       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_305.52       858.84     3_164.36       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_305.52       936.60     3_242.11       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_305.52     1_015.62     3_321.14       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                     2_305.52     9_060.44    11_365.95       0.4788     1.710526         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_574.24       826.96     3_401.21       0.1559          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_574.24       862.49     3_436.73       0.3458     3.038797         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_574.24       911.44     3_485.68       0.4728     1.776101         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_574.24     1_010.25     3_584.49       0.6194     0.970394         9.22
ExhaustiveBinary-512-itq (self)                        2_574.24     9_104.27    11_678.51       0.4718     1.776907         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 155.26       604.57       759.83       0.0061          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   155.26       626.68       781.94       0.0286    14.192335         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  155.26       654.65       809.91       0.0545    10.228876         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  155.26       720.58       875.84       0.1028     7.206309         0.58
ExhaustiveBinary-32-signed (self)                        155.26     6_566.12     6_721.38       0.0548    10.217732         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           1_947.84        47.57     1_995.41       0.1011          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           1_947.84        54.25     2_002.10       0.0924          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           1_947.84        68.39     2_016.23       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           1_947.84        70.17     2_018.01       0.2469     4.688474         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          1_947.84        92.02     2_039.86       0.3531     2.912185         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          1_947.84       132.14     2_079.98       0.4880     1.701449         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           1_947.84        77.01     2_024.86       0.2317     5.164905         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          1_947.84       101.29     2_049.13       0.3352     3.192027         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          1_947.84       143.82     2_091.66       0.4693     1.864007         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           1_947.84        92.07     2_039.91       0.2275     5.292553         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          1_947.84       118.43     2_066.27       0.3301     3.272207         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          1_947.84       162.94     2_110.78       0.4633     1.914805         5.79
IVF-Binary-256-nl273-random (self)                     1_947.84     1_024.62     2_972.46       0.3386     3.127658         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           2_237.06        49.54     2_286.60       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_237.06        60.88     2_297.94       0.0914          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_237.06        71.78     2_308.84       0.2359     4.925976         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_237.06        92.50     2_329.55       0.3422     3.051455         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_237.06       131.05     2_368.11       0.4774     1.803083         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_237.06        83.30     2_320.36       0.2291     5.147233         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_237.06       106.38     2_343.44       0.3333     3.199914         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_237.06       146.07     2_383.13       0.4671     1.891111         5.80
IVF-Binary-256-nl387-random (self)                     2_237.06       918.68     3_155.74       0.3445     3.009237         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           2_669.15        52.15     2_721.30       0.0947          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_669.15        51.95     2_721.11       0.0929          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_669.15        58.16     2_727.31       0.0910          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_669.15        69.14     2_738.29       0.2389     4.810820         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_669.15        89.09     2_758.25       0.3460     2.991620         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_669.15       127.17     2_796.32       0.4824     1.751270         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_669.15        73.09     2_742.24       0.2347     4.947063         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_669.15        93.94     2_763.10       0.3398     3.083894         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_669.15       132.24     2_801.40       0.4744     1.821608         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_669.15        81.42     2_750.57       0.2300     5.116134         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_669.15       101.93     2_771.09       0.3331     3.201004         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_669.15       140.83     2_809.99       0.4666     1.890853         5.82
IVF-Binary-256-nl547-random (self)                     2_669.15       901.24     3_570.39       0.3486     2.929273         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_105.84        54.84     2_160.69       0.0866          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_105.84        61.82     2_167.67       0.0760          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_105.84        78.53     2_184.37       0.0740          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_105.84        76.44     2_182.28       0.2237     5.261834         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_105.84        98.09     2_203.94       0.3227     3.335121         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_105.84       134.45     2_240.29       0.4538     1.957094         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_105.84        83.70     2_189.55       0.2056     5.858344         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_105.84       107.03     2_212.87       0.3030     3.667372         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_105.84       147.72     2_253.56       0.4353     2.130955         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_105.84       103.26     2_209.11       0.2023     6.011169         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_105.84       127.77     2_233.62       0.2984     3.754848         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_105.84       171.06     2_276.91       0.4298     2.183232         5.79
IVF-Binary-256-nl273-itq (self)                        2_105.84     1_068.37     3_174.21       0.3060     3.592747         5.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_407.59        52.54     2_460.13       0.0789          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_407.59        62.14     2_469.72       0.0754          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_407.59        70.71     2_478.29       0.2110     5.546564         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_407.59        92.09     2_499.68       0.3114     3.466633         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_407.59       130.33     2_537.92       0.4429     2.032685         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_407.59        84.07     2_491.66       0.2040     5.830488         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_407.59       107.01     2_514.60       0.3021     3.637967         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_407.59       145.91     2_553.50       0.4321     2.138410         5.80
IVF-Binary-256-nl387-itq (self)                        2_407.59       916.28     3_323.87       0.3132     3.418433         5.80
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)              2_848.68        50.30     2_898.98       0.0788          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              2_848.68        54.64     2_903.31       0.0768          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              2_848.68        61.42     2_910.10       0.0748          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              2_848.68        68.90     2_917.58       0.2131     5.392806         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             2_848.68        89.84     2_938.52       0.3138     3.376187         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             2_848.68       126.12     2_974.80       0.4476     1.984036         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              2_848.68        73.10     2_921.78       0.2078     5.560815         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             2_848.68        93.34     2_942.02       0.3074     3.486131         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             2_848.68       131.77     2_980.44       0.4397     2.054047         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              2_848.68        80.74     2_929.42       0.2026     5.786136         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             2_848.68       100.76     2_949.44       0.3000     3.634165         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             2_848.68       141.74     2_990.42       0.4312     2.137257         5.82
IVF-Binary-256-nl547-itq (self)                        2_848.68       894.74     3_743.42       0.3171     3.323726         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_083.74       101.59     3_185.33       0.1683          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_083.74       115.70     3_199.45       0.1639          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_083.74       146.97     3_230.72       0.1627          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_083.74       133.94     3_217.69       0.3646     2.767526        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_083.74       154.59     3_238.33       0.4943     1.615565        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_083.74       200.87     3_284.61       0.6426     0.866902        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_083.74       144.79     3_228.53       0.3572     2.873210        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_083.74       172.68     3_256.43       0.4862     1.680947        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_083.74       220.21     3_303.95       0.6347     0.904506        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_083.74       178.81     3_262.55       0.3551     2.904493        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_083.74       209.54     3_293.28       0.4838     1.699523        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_083.74       262.93     3_346.67       0.6316     0.917348        10.40
IVF-Binary-512-nl273-random (self)                     3_083.74     1_713.04     4_796.79       0.4883     1.656119        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)           3_403.96       104.54     3_508.50       0.1646          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_403.96       130.06     3_534.02       0.1629          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_403.96       141.91     3_545.88       0.3606     2.821520        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_403.96       156.94     3_560.90       0.4905     1.641297        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_403.96       212.91     3_616.88       0.6384     0.885257        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_403.96       159.40     3_563.36       0.3565     2.883438        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_403.96       186.76     3_590.72       0.4854     1.680373        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_403.96       234.70     3_638.66       0.6332     0.908750        10.41
IVF-Binary-512-nl387-random (self)                     3_403.96     1_571.57     4_975.53       0.4916     1.624401        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)           3_878.43       100.44     3_978.87       0.1656          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           3_878.43       109.92     3_988.35       0.1644          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           3_878.43       123.40     4_001.83       0.1631          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           3_878.43       126.39     4_004.82       0.3625     2.787397        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          3_878.43       151.43     4_029.86       0.4925     1.617504        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          3_878.43       194.55     4_072.98       0.6421     0.867831        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           3_878.43       136.93     4_015.36       0.3595     2.836286        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          3_878.43       162.09     4_040.52       0.4889     1.650064        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          3_878.43       205.67     4_084.10       0.6374     0.891076        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           3_878.43       150.97     4_029.40       0.3567     2.879432        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          3_878.43       178.29     4_056.72       0.4850     1.679600        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          3_878.43       224.74     4_103.17       0.6330     0.910585        10.43
IVF-Binary-512-nl547-random (self)                     3_878.43     1_508.79     5_387.21       0.4939     1.602424        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_338.05       105.03     3_443.08       0.1641          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_338.05       119.75     3_457.80       0.1594          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_338.05       160.53     3_498.58       0.1586          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_338.05       134.81     3_472.86       0.3619     2.813177        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_338.05       158.48     3_496.53       0.4904     1.636914        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_338.05       202.63     3_540.68       0.6375     0.887765        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_338.05       147.18     3_485.23       0.3546     2.933729        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_338.05       177.55     3_515.60       0.4823     1.711114        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_338.05       223.19     3_561.23       0.6300     0.926617        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_338.05       181.72     3_519.77       0.3528     2.960040        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_338.05       214.06     3_552.10       0.4803     1.727661        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_338.05       264.43     3_602.48       0.6274     0.938000        10.40
IVF-Binary-512-nl273-itq (self)                        3_338.05     1_751.03     5_089.08       0.4820     1.710575        10.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_648.91       112.02     3_760.93       0.1609          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_648.91       131.41     3_780.32       0.1592          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_648.91       135.11     3_784.02       0.3568     2.892556        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_648.91       158.62     3_807.53       0.4854     1.682095        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_648.91       202.30     3_851.21       0.6337     0.905241        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_648.91       160.14     3_809.05       0.3527     2.960534        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_648.91       188.24     3_837.15       0.4808     1.722693        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_648.91       236.80     3_885.71       0.6286     0.930674        10.41
IVF-Binary-512-nl387-itq (self)                        3_648.91     1_572.60     5_221.51       0.4852     1.677564        10.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_079.68       105.64     4_185.31       0.1610          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_079.68       108.11     4_187.78       0.1597          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_079.68       121.97     4_201.64       0.1585          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_079.68       126.66     4_206.33       0.3587     2.845388        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_079.68       150.14     4_229.82       0.4877     1.654173        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_079.68       207.10     4_286.77       0.6368     0.890153        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_079.68       134.64     4_214.31       0.3558     2.897453        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_079.68       161.44     4_241.12       0.4840     1.690117        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_079.68       208.07     4_287.75       0.6324     0.913921        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_079.68       158.65     4_238.33       0.3530     2.949192        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_079.68       176.07     4_255.75       0.4806     1.719996        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_079.68       222.52     4_302.20       0.6284     0.933792        10.43
IVF-Binary-512-nl547-itq (self)                        4_079.68     1_494.62     5_574.29       0.4876     1.653264        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)              926.77        37.67       964.45       0.0129          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)              926.77        43.38       970.16       0.0072          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)              926.77        57.92       984.69       0.0063          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)              926.77        43.49       970.27       0.0586    13.096980         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)             926.77        51.21       977.98       0.1112     9.821555         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)             926.77        68.36       995.13       0.2011     7.183025         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)              926.77        50.08       976.85       0.0348    15.883413         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)             926.77        57.54       984.31       0.0691    12.325441         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)             926.77        75.18     1_001.96       0.1292     9.466448         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)              926.77        66.25       993.03       0.0313    16.844205         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)             926.77        74.72     1_001.50       0.0621    13.207787         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)             926.77        92.13     1_018.91       0.1180    10.184865         1.76
IVF-Binary-32-nl273-signed (self)                        926.77       573.84     1_500.61       0.0698    12.240784         1.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_249.43        38.86     1_288.29       0.0091          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_249.43        51.41     1_300.83       0.0074          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_249.43        46.46     1_295.89       0.0424    14.581671         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_249.43        52.46     1_301.88       0.0819    11.231527         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_249.43        71.26     1_320.69       0.1529     8.266913         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_249.43        57.55     1_306.98       0.0359    16.263112         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_249.43        66.68     1_316.11       0.0683    12.773930         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_249.43        82.75     1_332.18       0.1290     9.520516         1.77
IVF-Binary-32-nl387-signed (self)                      1_249.43       521.82     1_771.25       0.0810    11.197807         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_714.77        36.82     1_751.59       0.0093          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_714.77        41.44     1_756.20       0.0082          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_714.77        47.49     1_762.25       0.0073          NaN         1.79
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_714.77        43.11     1_757.88       0.0435    13.965319         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_714.77        49.93     1_764.69       0.0846    10.644863         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_714.77        66.08     1_780.84       0.1608     7.468921         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_714.77        47.66     1_762.42       0.0394    15.076596         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_714.77        54.01     1_768.77       0.0769    11.641643         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_714.77        69.44     1_784.21       0.1463     8.230163         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_714.77        53.97     1_768.73       0.0352    16.505947         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_714.77        60.82     1_775.58       0.0684    12.933777         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_714.77        76.43     1_791.20       0.1304     9.277859         1.79
IVF-Binary-32-nl547-signed (self)                      1_714.77       493.74     2_208.51       0.0872    10.587026         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With more dimensions

The binary indices shine more with more dimensions in the end; however,
the strong compression still yields much worse Recalls.

<details>
<summary><b>Binary - Euclidean (LowRank - 128 bit)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.49     5_942.66     5_957.15       1.0000     0.000000        73.24
Exhaustive (self)                                         14.49    60_708.87    60_723.36       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_749.88       506.84     4_256.73       0.0770          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_749.88       551.92     4_301.80       0.1853    40.590911         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_749.88       610.01     4_359.89       0.2680    26.716815         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_749.88       705.65     4_455.53       0.3854    16.589779         4.70
ExhaustiveBinary-256-random (self)                     3_749.88     6_451.39    10_201.27       0.2721    26.279507         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 4_523.65       506.95     5_030.60       0.0352          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                   4_523.65       545.31     5_068.97       0.1078    58.311100         4.70
ExhaustiveBinary-256-itq-rf10 (query)                  4_523.65       604.79     5_128.45       0.1716    39.742881         4.70
ExhaustiveBinary-256-itq-rf20 (query)                  4_523.65       691.54     5_215.19       0.2692    25.845190         4.70
ExhaustiveBinary-256-itq (self)                        4_523.65     5_980.26    10_503.92       0.1755    39.408653         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_295.14       880.11     8_175.25       0.1527          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_295.14       926.26     8_221.40       0.3242    22.196757         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_295.14       980.95     8_276.09       0.4419    13.545010         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_295.14     1_095.49     8_390.63       0.5832     7.719977         9.41
ExhaustiveBinary-512-random (self)                     7_295.14     9_801.09    17_096.23       0.4431    13.453984         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 8_298.31       880.78     9_179.09       0.1257          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                   8_298.31       936.70     9_235.01       0.2688    27.688764         9.41
ExhaustiveBinary-512-itq-rf10 (query)                  8_298.31       982.79     9_281.11       0.3728    17.457655         9.41
ExhaustiveBinary-512-itq-rf20 (query)                  8_298.31     1_110.77     9_409.08       0.5062    10.365482         9.41
ExhaustiveBinary-512-itq (self)                        8_298.31     9_823.33    18_121.65       0.3728    17.440968         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            14_571.27     1_486.66    16_057.93       0.2353          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              14_571.27     1_565.00    16_136.27       0.4918    11.183271        18.81
ExhaustiveBinary-1024-random-rf10 (query)             14_571.27     1_629.27    16_200.54       0.6358     5.988378        18.81
ExhaustiveBinary-1024-random-rf20 (query)             14_571.27     1_748.26    16_319.53       0.7779     2.871209        18.81
ExhaustiveBinary-1024-random (self)                   14_571.27    16_043.74    30_615.01       0.6381     5.940314        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq_no_rr (query)               15_812.63     1_491.79    17_304.42       0.2198          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 15_812.63     1_551.49    17_364.13       0.4636    12.595159        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                15_812.63     1_613.83    17_426.46       0.6048     6.884343        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                15_812.63     1_734.47    17_547.10       0.7500     3.387537        18.81
ExhaustiveBinary-1024-itq (self)                      15_812.63    16_148.33    31_960.97       0.6059     6.857783        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-128-signed_no_rr (query)              1_904.95       412.49     2_317.44       0.0275          NaN         2.35
ExhaustiveBinary-128-signed-rf5 (query)                1_904.95       440.47     2_345.41       0.0917    63.471902         2.35
ExhaustiveBinary-128-signed-rf10 (query)               1_904.95       469.95     2_374.89       0.1491    44.012830         2.35
ExhaustiveBinary-128-signed-rf20 (query)               1_904.95       564.91     2_469.86       0.2396    29.121769         2.35
ExhaustiveBinary-128-signed (self)                     1_904.95     4_671.27     6_576.21       0.1538    43.418564         2.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           4_392.27        83.89     4_476.16       0.0902          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           4_392.27        90.05     4_482.32       0.0819          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           4_392.27       102.03     4_494.29       0.0801          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           4_392.27       116.74     4_509.00       0.2091    36.578792         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          4_392.27       149.17     4_541.44       0.2994    23.787256         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          4_392.27       210.22     4_602.49       0.4198    14.719867         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           4_392.27       122.07     4_514.33       0.1953    39.550198         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          4_392.27       157.58     4_549.85       0.2830    25.760295         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          4_392.27       224.26     4_616.53       0.4022    15.936889         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           4_392.27       135.47     4_527.74       0.1914    40.388363         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          4_392.27       170.75     4_563.01       0.2780    26.330895         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          4_392.27       241.77     4_634.03       0.3961    16.314407         5.98
IVF-Binary-256-nl273-random (self)                     4_392.27     1_575.70     5_967.97       0.2863    25.401151         5.98
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           4_752.36        89.63     4_842.00       0.0833          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           4_752.36        98.75     4_851.11       0.0803          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           4_752.36       119.77     4_872.13       0.1988    38.580488         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          4_752.36       152.63     4_904.99       0.2878    25.052774         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          4_752.36       215.03     4_967.40       0.4094    15.361461         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           4_752.36       129.44     4_881.80       0.1924    40.100378         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          4_752.36       164.58     4_916.94       0.2784    26.221115         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          4_752.36       229.77     4_982.14       0.3991    16.078668         6.04
IVF-Binary-256-nl387-random (self)                     4_752.36     1_523.08     6_275.44       0.2922    24.627327         6.04
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           5_264.93        90.65     5_355.58       0.0849          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           5_264.93        96.00     5_360.93       0.0833          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           5_264.93       103.97     5_368.90       0.0816          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           5_264.93       123.68     5_388.61       0.2032    37.491643         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          5_264.93       157.49     5_422.43       0.2944    24.252697         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          5_264.93       215.43     5_480.36       0.4167    14.903314         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           5_264.93       125.04     5_389.97       0.1991    38.388563         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          5_264.93       163.27     5_428.20       0.2887    24.923524         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          5_264.93       222.82     5_487.75       0.4089    15.379880         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           5_264.93       131.62     5_396.55       0.1956    39.282687         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          5_264.93       165.81     5_430.75       0.2833    25.613983         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          5_264.93       231.65     5_496.58       0.4024    15.812546         6.12
IVF-Binary-256-nl547-random (self)                     5_264.93     1_565.66     6_830.59       0.2983    23.870101         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              5_154.01        85.01     5_239.02       0.0487          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)              5_154.01        90.48     5_244.50       0.0384          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)              5_154.01       103.35     5_257.36       0.0368          NaN         5.98
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf5-itq (query)              5_154.01       126.23     5_280.25       0.1401    52.288654         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)             5_154.01       142.59     5_296.60       0.2148    35.122498         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)             5_154.01       201.14     5_355.16       0.3221    22.338869         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)              5_154.01       116.43     5_270.44       0.1189    57.863947         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)             5_154.01       149.24     5_303.25       0.1896    39.050758         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)             5_154.01       212.53     5_366.55       0.2959    24.689005         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)              5_154.01       130.17     5_284.18       0.1141    60.629599         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)             5_154.01       160.21     5_314.22       0.1814    41.279348         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)             5_154.01       224.71     5_378.73       0.2847    26.047475         5.98
IVF-Binary-256-nl273-itq (self)                        5_154.01     1_472.79     6_626.81       0.1935    38.785311         5.98
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)              5_503.44        88.85     5_592.29       0.0406          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)              5_503.44       100.48     5_603.92       0.0376          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)              5_503.44       116.29     5_619.73       0.1245    55.409024         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)             5_503.44       146.62     5_650.07       0.1982    37.188360         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)             5_503.44       205.99     5_709.43       0.3050    23.672227         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)              5_503.44       128.70     5_632.14       0.1162    58.675988         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)             5_503.44       157.74     5_661.18       0.1862    39.515395         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)             5_503.44       220.17     5_723.61       0.2892    25.386678         6.04
IVF-Binary-256-nl387-itq (self)                        5_503.44     1_467.56     6_971.00       0.2014    37.046103         6.04
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)              6_014.52        90.60     6_105.12       0.0422          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)              6_014.52        94.41     6_108.93       0.0405          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)              6_014.52        99.70     6_114.22       0.0388          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)              6_014.52       119.88     6_134.40       0.1299    53.388247         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)             6_014.52       147.93     6_162.45       0.2055    35.710531         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)             6_014.52       203.51     6_218.03       0.3158    22.516698         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)              6_014.52       129.18     6_143.71       0.1248    55.169413         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)             6_014.52       150.74     6_165.26       0.1985    37.017374         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)             6_014.52       208.90     6_223.42       0.3063    23.439718         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)              6_014.52       125.30     6_139.82       0.1199    57.122875         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)             6_014.52       156.23     6_170.75       0.1918    38.438977         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)             6_014.52       217.15     6_231.67       0.2971    24.418490         6.12
IVF-Binary-256-nl547-itq (self)                        6_014.52     1_470.59     7_485.12       0.2087    35.561453         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           8_043.76       173.94     8_217.69       0.1602          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)           8_043.76       188.51     8_232.26       0.1562          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)           8_043.76       220.73     8_264.49       0.1551          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)           8_043.76       226.02     8_269.77       0.3381    20.892325        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)          8_043.76       262.66     8_306.42       0.4580    12.672864        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)          8_043.76       328.29     8_372.05       0.5999     7.188412        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)           8_043.76       242.02     8_285.77       0.3309    21.652949        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)          8_043.76       285.05     8_328.81       0.4504    13.148010        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)          8_043.76       355.04     8_398.80       0.5921     7.463470        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)           8_043.76       275.12     8_318.88       0.3289    21.833934        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)          8_043.76       324.17     8_367.92       0.4476    13.287856        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)          8_043.76       401.17     8_444.93       0.5892     7.551369        10.69
IVF-Binary-512-nl273-random (self)                     8_043.76     2_827.67    10_871.42       0.4516    13.064777        10.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)           8_286.62       179.00     8_465.62       0.1570          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)           8_286.62       204.48     8_491.11       0.1554          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)           8_286.62       230.64     8_517.26       0.3329    21.392194        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)          8_286.62       265.68     8_552.31       0.4539    12.917844        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)          8_286.62       331.21     8_617.84       0.5956     7.339191        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)           8_286.62       256.13     8_542.75       0.3292    21.773974        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)          8_286.62       299.21     8_585.84       0.4491    13.194343        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)          8_286.62       367.96     8_654.59       0.5904     7.519640        10.74
IVF-Binary-512-nl387-random (self)                     8_286.62     2_659.36    10_945.99       0.4548    12.865173        10.74
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)           8_811.79       175.51     8_987.30       0.1583          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)           8_811.79       184.65     8_996.44       0.1572          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)           8_811.79       199.14     9_010.92       0.1564          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)           8_811.79       224.15     9_035.94       0.3359    21.084876        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)          8_811.79       258.27     9_070.06       0.4573    12.732967        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)          8_811.79       322.94     9_134.73       0.6004     7.180296        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)           8_811.79       247.28     9_059.07       0.3331    21.395559        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)          8_811.79       271.17     9_082.96       0.4534    12.958526        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)          8_811.79       332.86     9_144.65       0.5955     7.342621        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)           8_811.79       250.18     9_061.96       0.3305    21.646023        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)          8_811.79       290.34     9_102.13       0.4505    13.114137        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)          8_811.79       355.44     9_167.22       0.5920     7.461133        10.82
IVF-Binary-512-nl547-random (self)                     8_811.79     2_573.02    11_384.81       0.4586    12.646246        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              8_800.82       172.06     8_972.87       0.1340          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)              8_800.82       185.02     8_985.84       0.1293          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)              8_800.82       214.33     9_015.14       0.1278          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)              8_800.82       221.46     9_022.27       0.2839    25.978536        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)             8_800.82       259.13     9_059.95       0.3917    16.248466        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)             8_800.82       320.30     9_121.11       0.5269     9.600056        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)              8_800.82       238.86     9_039.67       0.2761    26.883355        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)             8_800.82       279.31     9_080.13       0.3827    16.844752        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)             8_800.82       346.03     9_146.84       0.5176     9.960214        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)              8_800.82       268.93     9_069.75       0.2734    27.240484        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)             8_800.82       332.80     9_133.62       0.3791    17.094724        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)             8_800.82       390.83     9_191.64       0.5136    10.119578        10.69
IVF-Binary-512-nl273-itq (self)                        8_800.82     2_773.34    11_574.16       0.3830    16.857468        10.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)              9_201.06       183.15     9_384.21       0.1304          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)              9_201.06       209.20     9_410.27       0.1281          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)              9_201.06       252.57     9_453.64       0.2781    26.595084        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)             9_201.06       267.78     9_468.84       0.3865    16.598294        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)             9_201.06       331.29     9_532.36       0.5215     9.795978        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)              9_201.06       272.71     9_473.77       0.2740    27.188689        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)             9_201.06       310.20     9_511.26       0.3808    17.014995        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)             9_201.06       368.19     9_569.26       0.5144    10.085787        10.74
IVF-Binary-512-nl387-itq (self)                        9_201.06     2_660.16    11_861.23       0.3865    16.606584        10.74
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)              9_662.52       177.94     9_840.46       0.1316          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)              9_662.52       187.15     9_849.67       0.1304          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)              9_662.52       201.65     9_864.18       0.1292          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)              9_662.52       239.11     9_901.64       0.2817    26.152205        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)             9_662.52       259.56     9_922.08       0.3899    16.325027        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)             9_662.52       319.64     9_982.16       0.5273     9.576019        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)              9_662.52       234.27     9_896.79       0.2787    26.530614        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)             9_662.52       275.04     9_937.56       0.3857    16.606746        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)             9_662.52       333.74     9_996.26       0.5215     9.793693        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)              9_662.52       252.06     9_914.58       0.2758    26.929552        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)             9_662.52       297.53     9_960.06       0.3822    16.861961        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)             9_662.52       354.65    10_017.17       0.5167     9.985026        10.82
IVF-Binary-512-nl547-itq (self)                        9_662.52     2_583.98    12_246.50       0.3908    16.303195        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         15_064.69       425.55    15_490.23       0.2401          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         15_064.69       470.01    15_534.70       0.2381          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         15_064.69       564.64    15_629.33       0.2375          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         15_064.69       478.86    15_543.55       0.4998    10.816616        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        15_064.69       494.43    15_559.11       0.6434     5.777947        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        15_064.69       549.01    15_613.70       0.7844     2.752281        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         15_064.69       511.21    15_575.89       0.4962    10.985121        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        15_064.69       552.63    15_617.32       0.6403     5.869059        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        15_064.69       604.60    15_669.29       0.7816     2.805109        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         15_064.69       605.14    15_669.83       0.4952    11.037948        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        15_064.69       644.94    15_709.63       0.6392     5.897839        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        15_064.69       717.17    15_781.86       0.7806     2.824663        20.09
IVF-Binary-1024-nl273-random (self)                   15_064.69     5_414.21    20_478.90       0.6427     5.817969        20.09
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-random (query)         15_438.17       440.66    15_878.83       0.2383          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         15_438.17       518.63    15_956.80       0.2373          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         15_438.17       475.54    15_913.71       0.4979    10.898759        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        15_438.17       508.50    15_946.67       0.6416     5.825887        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        15_438.17       569.70    16_007.87       0.7831     2.775808        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         15_438.17       559.00    15_997.17       0.4959    11.003445        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        15_438.17       594.94    16_033.10       0.6392     5.891274        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        15_438.17       660.46    16_098.62       0.7810     2.815588        20.15
IVF-Binary-1024-nl387-random (self)                   15_438.17     5_096.49    20_534.65       0.6440     5.775508        20.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-random (query)         15_937.58       408.73    16_346.30       0.2395          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         15_937.58       440.86    16_378.44       0.2388          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         15_937.58       484.89    16_422.47       0.2382          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         15_937.58       444.35    16_381.93       0.4997    10.806026        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        15_937.58       477.70    16_415.27       0.6440     5.755572        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        15_937.58       531.42    16_469.00       0.7852     2.737681        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         15_937.58       480.31    16_417.89       0.4978    10.904969        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        15_937.58       516.58    16_454.16       0.6417     5.823544        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        15_937.58       566.11    16_503.69       0.7827     2.782061        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         15_937.58       520.78    16_458.35       0.4963    10.979509        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        15_937.58       555.75    16_493.33       0.6400     5.873334        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        15_937.58       619.11    16_556.68       0.7812     2.811811        20.23
IVF-Binary-1024-nl547-random (self)                   15_937.58     4_797.55    20_735.13       0.6464     5.703027        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            16_279.36       432.14    16_711.50       0.2249          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            16_279.36       480.82    16_760.18       0.2228          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            16_279.36       578.75    16_858.11       0.2222          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            16_279.36       474.50    16_753.86       0.4721    12.145417        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           16_279.36       510.74    16_790.10       0.6138     6.617575        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           16_279.36       594.80    16_874.16       0.7576     3.236223        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            16_279.36       545.52    16_824.88       0.4689    12.327458        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           16_279.36       560.01    16_839.37       0.6102     6.730367        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           16_279.36       610.81    16_890.17       0.7543     3.305606        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            16_279.36       612.10    16_891.46       0.4674    12.407278        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           16_279.36       661.56    16_940.92       0.6086     6.782831        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           16_279.36       718.12    16_997.48       0.7526     3.335969        20.09
IVF-Binary-1024-nl273-itq (self)                      16_279.36     5_501.02    21_780.38       0.6108     6.713118        20.09
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-itq (query)            16_752.11       440.06    17_192.17       0.2233          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            16_752.11       522.17    17_274.28       0.2224          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            16_752.11       479.61    17_231.72       0.4697    12.264457        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           16_752.11       508.94    17_261.05       0.6113     6.691400        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           16_752.11       565.81    17_317.92       0.7558     3.274724        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            16_752.11       560.99    17_313.11       0.4674    12.385999        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           16_752.11       595.57    17_347.68       0.6086     6.776923        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           16_752.11       656.08    17_408.19       0.7535     3.321376        20.15
IVF-Binary-1024-nl387-itq (self)                      16_752.11     5_102.81    21_854.92       0.6122     6.665528        20.15
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-itq (query)            17_281.54       416.82    17_698.36       0.2241          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            17_281.54       446.95    17_728.50       0.2232          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            17_281.54       493.05    17_774.59       0.2223          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            17_281.54       461.77    17_743.32       0.4713    12.165897        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           17_281.54       480.92    17_762.46       0.6141     6.602729        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           17_281.54       540.76    17_822.30       0.7583     3.220276        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            17_281.54       482.95    17_764.49       0.4697    12.266341        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           17_281.54       524.17    17_805.71       0.6117     6.681995        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           17_281.54       570.73    17_852.27       0.7556     3.273780        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            17_281.54       529.92    17_811.47       0.4685    12.344726        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           17_281.54       563.23    17_844.78       0.6099     6.739142        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           17_281.54       632.20    17_913.75       0.7539     3.307580        20.23
IVF-Binary-1024-nl547-itq (self)                      17_281.54     4_818.17    22_099.71       0.6147     6.580361        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-128-nl273-np13-rf0-signed (query)           2_541.61        54.59     2_596.20       0.0405          NaN         3.63
IVF-Binary-128-nl273-np16-rf0-signed (query)           2_541.61        59.14     2_600.75       0.0316          NaN         3.63
IVF-Binary-128-nl273-np23-rf0-signed (query)           2_541.61        68.73     2_610.34       0.0298          NaN         3.63
IVF-Binary-128-nl273-np13-rf5-signed (query)           2_541.61        77.80     2_619.42       0.1259    55.863005         3.63
IVF-Binary-128-nl273-np13-rf10-signed (query)          2_541.61       105.13     2_646.75       0.1973    38.135787         3.63
IVF-Binary-128-nl273-np13-rf20-signed (query)          2_541.61       146.69     2_688.30       0.3031    24.545632         3.63
IVF-Binary-128-nl273-np16-rf5-signed (query)           2_541.61        80.95     2_622.56       0.1029    63.684530         3.63
IVF-Binary-128-nl273-np16-rf10-signed (query)          2_541.61       105.46     2_647.08       0.1670    44.091872         3.63
IVF-Binary-128-nl273-np16-rf20-signed (query)          2_541.61       154.01     2_695.63       0.2658    28.752678         3.63
IVF-Binary-128-nl273-np23-rf5-signed (query)           2_541.61        91.50     2_633.11       0.0984    65.659545         3.63
IVF-Binary-128-nl273-np23-rf10-signed (query)          2_541.61       117.22     2_658.83       0.1605    45.502618         3.63
IVF-Binary-128-nl273-np23-rf20-signed (query)          2_541.61       165.25     2_706.86       0.2555    29.909447         3.63
IVF-Binary-128-nl273-signed (self)                     2_541.61     1_060.98     3_602.59       0.1705    43.928033         3.63
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-128-nl387-np19-rf0-signed (query)           2_878.32        59.56     2_937.88       0.0335          NaN         3.69
IVF-Binary-128-nl387-np27-rf0-signed (query)           2_878.32        66.82     2_945.14       0.0311          NaN         3.69
IVF-Binary-128-nl387-np19-rf5-signed (query)           2_878.32        82.81     2_961.13       0.1085    61.081309         3.69
IVF-Binary-128-nl387-np19-rf10-signed (query)          2_878.32       105.00     2_983.32       0.1761    42.104932         3.69
IVF-Binary-128-nl387-np19-rf20-signed (query)          2_878.32       158.06     3_036.37       0.2786    27.252575         3.69
IVF-Binary-128-nl387-np27-rf5-signed (query)           2_878.32        90.30     2_968.62       0.1007    64.610159         3.69
IVF-Binary-128-nl387-np27-rf10-signed (query)          2_878.32       115.33     2_993.65       0.1638    44.938258         3.69
IVF-Binary-128-nl387-np27-rf20-signed (query)          2_878.32       162.47     3_040.78       0.2600    29.469313         3.69
IVF-Binary-128-nl387-signed (self)                     2_878.32     1_045.77     3_924.08       0.1801    41.704632         3.69
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-128-nl547-np23-rf0-signed (query)           3_412.35        61.34     3_473.69       0.0347          NaN         3.77
IVF-Binary-128-nl547-np27-rf0-signed (query)           3_412.35        63.82     3_476.17       0.0332          NaN         3.77
IVF-Binary-128-nl547-np33-rf0-signed (query)           3_412.35        68.87     3_481.22       0.0313          NaN         3.77
IVF-Binary-128-nl547-np23-rf5-signed (query)           3_412.35        83.51     3_495.86       0.1136    58.440812         3.77
IVF-Binary-128-nl547-np23-rf10-signed (query)          3_412.35       104.73     3_517.08       0.1836    39.855237         3.77
IVF-Binary-128-nl547-np23-rf20-signed (query)          3_412.35       151.33     3_563.67       0.2887    25.505688         3.77
IVF-Binary-128-nl547-np27-rf5-signed (query)           3_412.35        84.74     3_497.09       0.1096    60.271032         3.77
IVF-Binary-128-nl547-np27-rf10-signed (query)          3_412.35       108.24     3_520.59       0.1769    41.374801         3.77
IVF-Binary-128-nl547-np27-rf20-signed (query)          3_412.35       154.64     3_566.99       0.2780    26.736117         3.77
IVF-Binary-128-nl547-np33-rf5-signed (query)           3_412.35        89.73     3_502.08       0.1045    62.486372         3.77
IVF-Binary-128-nl547-np33-rf10-signed (query)          3_412.35       113.07     3_525.42       0.1698    43.037431         3.77
IVF-Binary-128-nl547-np33-rf20-signed (query)          3_412.35       159.54     3_571.88       0.2677    27.966006         3.77
IVF-Binary-128-nl547-signed (self)                     3_412.35     1_040.69     4_453.04       0.1881    39.440811         3.77
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

<details>
<summary><b>Binary - Euclidean (LowRank - 256 bit)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 256D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        28.69    14_736.97    14_765.66       1.0000     0.000000       146.48
Exhaustive (self)                                         28.69   148_773.57   148_802.26       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              7_667.15       563.16     8_230.32       0.0848          NaN         4.83
ExhaustiveBinary-256-random-rf5 (query)                7_667.15       639.21     8_306.36       0.1980    84.795844         4.83
ExhaustiveBinary-256-random-rf10 (query)               7_667.15       668.58     8_335.74       0.2831    55.592413         4.83
ExhaustiveBinary-256-random-rf20 (query)               7_667.15       835.28     8_502.43       0.4015    34.646234         4.83
ExhaustiveBinary-256-random (self)                     7_667.15     6_660.58    14_327.73       0.2867    54.828071         4.83
ExhaustiveBinary-256-itq_no_rr (query)                 9_626.20       554.60    10_180.79       0.0027          NaN         4.83
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq-rf5 (query)                   9_626.20       623.26    10_249.46       0.0128   233.324047         4.83
ExhaustiveBinary-256-itq-rf10 (query)                  9_626.20       611.38    10_237.58       0.0254   177.356207         4.83
ExhaustiveBinary-256-itq-rf20 (query)                  9_626.20       685.38    10_311.58       0.0500   134.022852         4.83
ExhaustiveBinary-256-itq (self)                        9_626.20     6_131.39    15_757.58       0.0256   177.818519         4.83
ExhaustiveBinary-512-random_no_rr (query)             15_035.48       965.80    16_001.29       0.1519          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random-rf5 (query)               15_035.48     1_027.89    16_063.37       0.3177    49.573717         9.66
ExhaustiveBinary-512-random-rf10 (query)              15_035.48     1_116.43    16_151.91       0.4333    30.425506         9.66
ExhaustiveBinary-512-random-rf20 (query)              15_035.48     1_242.85    16_278.33       0.5756    17.359891         9.66
ExhaustiveBinary-512-random (self)                    15_035.48    10_943.37    25_978.85       0.4358    30.179063         9.66
ExhaustiveBinary-512-itq_no_rr (query)                17_352.32       972.32    18_324.64       0.0848          NaN         9.66
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq-rf5 (query)                  17_352.32     1_029.61    18_381.93       0.1972    85.264886         9.66
ExhaustiveBinary-512-itq-rf10 (query)                 17_352.32     1_093.79    18_446.11       0.2835    55.811208         9.66
ExhaustiveBinary-512-itq-rf20 (query)                 17_352.32     1_326.05    18_678.37       0.4008    34.735180         9.66
ExhaustiveBinary-512-itq (self)                       17_352.32    10_908.51    28_260.83       0.2855    55.077668         9.66
ExhaustiveBinary-1024-random_no_rr (query)            29_704.49     1_686.75    31_391.24       0.2320          NaN        19.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random-rf5 (query)              29_704.49     1_763.96    31_468.45       0.4863    25.204494        19.31
ExhaustiveBinary-1024-random-rf10 (query)             29_704.49     1_815.50    31_519.99       0.6287    13.571699        19.31
ExhaustiveBinary-1024-random-rf20 (query)             29_704.49     1_963.79    31_668.29       0.7709     6.528124        19.31
ExhaustiveBinary-1024-random (self)                   29_704.49    18_255.05    47_959.54       0.6327    13.347710        19.31
ExhaustiveBinary-1024-itq_no_rr (query)               32_743.65     1_698.01    34_441.66       0.1937          NaN        19.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq-rf5 (query)                 32_743.65     1_767.09    34_510.74       0.4069    34.763449        19.31
ExhaustiveBinary-1024-itq-rf10 (query)                32_743.65     1_836.10    34_579.75       0.5407    19.947772        19.31
ExhaustiveBinary-1024-itq-rf20 (query)                32_743.65     1_963.40    34_707.05       0.6873    10.488234        19.31
ExhaustiveBinary-1024-itq (self)                      32_743.65    18_408.90    51_152.55       0.5428    19.821924        19.31
ExhaustiveBinary-256-signed_no_rr (query)              7_676.77       560.03     8_236.80       0.0848          NaN         4.83
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-signed-rf5 (query)                7_676.77       617.30     8_294.07       0.1980    84.795844         4.83
ExhaustiveBinary-256-signed-rf10 (query)               7_676.77       667.57     8_344.34       0.2831    55.592413         4.83
ExhaustiveBinary-256-signed-rf20 (query)               7_676.77       798.01     8_474.78       0.4015    34.646234         4.83
ExhaustiveBinary-256-signed (self)                     7_676.77     6_663.51    14_340.28       0.2867    54.828071         4.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           8_733.23       141.95     8_875.18       0.0975          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-random (query)           8_733.23       147.46     8_880.70       0.0903          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-random (query)           8_733.23       161.81     8_895.04       0.0874          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-random (query)           8_733.23       195.83     8_929.07       0.2202    76.986041         6.24
IVF-Binary-256-nl273-np13-rf10-random (query)          8_733.23       244.16     8_977.39       0.3106    49.904755         6.24
IVF-Binary-256-nl273-np13-rf20-random (query)          8_733.23       332.14     9_065.37       0.4342    30.708365         6.24
IVF-Binary-256-nl273-np16-rf5-random (query)           8_733.23       199.29     8_932.52       0.2092    81.570143         6.24
IVF-Binary-256-nl273-np16-rf10-random (query)          8_733.23       254.04     8_987.28       0.2985    52.787271         6.24
IVF-Binary-256-nl273-np16-rf20-random (query)          8_733.23       341.30     9_074.53       0.4205    32.516565         6.24
IVF-Binary-256-nl273-np23-rf5-random (query)           8_733.23       211.01     8_944.24       0.2036    84.117063         6.24
IVF-Binary-256-nl273-np23-rf10-random (query)          8_733.23       270.92     9_004.15       0.2912    54.726832         6.24
IVF-Binary-256-nl273-np23-rf20-random (query)          8_733.23       367.58     9_100.81       0.4117    33.856600         6.24
IVF-Binary-256-nl273-random (self)                     8_733.23     2_530.27    11_263.50       0.3019    52.280744         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)           9_320.74       151.71     9_472.45       0.0917          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-random (query)           9_320.74       163.32     9_484.06       0.0890          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-random (query)           9_320.74       209.04     9_529.78       0.2131    79.271693         6.35
IVF-Binary-256-nl387-np19-rf10-random (query)          9_320.74       255.65     9_576.39       0.3030    51.248429         6.35
IVF-Binary-256-nl387-np19-rf20-random (query)          9_320.74       338.42     9_659.16       0.4279    31.362689         6.35
IVF-Binary-256-nl387-np27-rf5-random (query)           9_320.74       216.44     9_537.18       0.2067    82.316705         6.35
IVF-Binary-256-nl387-np27-rf10-random (query)          9_320.74       274.78     9_595.53       0.2951    53.352131         6.35
IVF-Binary-256-nl387-np27-rf20-random (query)          9_320.74       362.95     9_683.69       0.4171    32.933454         6.35
IVF-Binary-256-nl387-random (self)                     9_320.74     2_548.94    11_869.69       0.3066    50.676822         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)           9_943.66       160.83    10_104.48       0.0929          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-random (query)           9_943.66       165.33    10_108.99       0.0907          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-random (query)           9_943.66       172.81    10_116.47       0.0888          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-random (query)           9_943.66       219.97    10_163.63       0.2167    76.983647         6.51
IVF-Binary-256-nl547-np23-rf10-random (query)          9_943.66       282.61    10_226.27       0.3093    49.456854         6.51
IVF-Binary-256-nl547-np23-rf20-random (query)          9_943.66       342.28    10_285.94       0.4348    30.219562         6.51
IVF-Binary-256-nl547-np27-rf5-random (query)           9_943.66       220.82    10_164.48       0.2121    79.100710         6.51
IVF-Binary-256-nl547-np27-rf10-random (query)          9_943.66       270.06    10_213.71       0.3024    51.043113         6.51
IVF-Binary-256-nl547-np27-rf20-random (query)          9_943.66       351.69    10_295.34       0.4263    31.325323         6.51
IVF-Binary-256-nl547-np33-rf5-random (query)           9_943.66       230.67    10_174.32       0.2083    81.042925         6.51
IVF-Binary-256-nl547-np33-rf10-random (query)          9_943.66       280.97    10_224.63       0.2966    52.484571         6.51
IVF-Binary-256-nl547-np33-rf20-random (query)          9_943.66       369.77    10_313.42       0.4193    32.278388         6.51
IVF-Binary-256-nl547-random (self)                     9_943.66     2_624.89    12_568.54       0.3132    48.780140         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)             10_615.21       149.18    10_764.39       0.0066          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-itq (query)             10_615.21       156.35    10_771.56       0.0040          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-itq (query)             10_615.21       171.56    10_786.77       0.0034          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-itq (query)             10_615.21       179.04    10_794.26       0.0302   204.860825         6.24
IVF-Binary-256-nl273-np13-rf10-itq (query)            10_615.21       211.25    10_826.46       0.0611   156.677726         6.24
IVF-Binary-256-nl273-np13-rf20-itq (query)            10_615.21       272.85    10_888.06       0.1215   118.466106         6.24
IVF-Binary-256-nl273-np16-rf5-itq (query)             10_615.21       185.71    10_800.92       0.0187   229.874198         6.24
IVF-Binary-256-nl273-np16-rf10-itq (query)            10_615.21       216.93    10_832.15       0.0380   180.668158         6.24
IVF-Binary-256-nl273-np16-rf20-itq (query)            10_615.21       275.70    10_890.91       0.0756   142.404774         6.24
IVF-Binary-256-nl273-np23-rf5-itq (query)             10_615.21       198.33    10_813.54       0.0157   244.018146         6.24
IVF-Binary-256-nl273-np23-rf10-itq (query)            10_615.21       230.47    10_845.68       0.0315   194.560815         6.24
IVF-Binary-256-nl273-np23-rf20-itq (query)            10_615.21       294.89    10_910.10       0.0626   156.283445         6.24
IVF-Binary-256-nl273-itq (self)                       10_615.21     2_147.85    12_763.06       0.0388   180.605269         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)             11_058.63       151.27    11_209.90       0.0044          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-itq (query)             11_058.63       162.70    11_221.33       0.0036          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-itq (query)             11_058.63       182.97    11_241.59       0.0223   221.454963         6.35
IVF-Binary-256-nl387-np19-rf10-itq (query)            11_058.63       214.11    11_272.74       0.0442   174.213070         6.35
IVF-Binary-256-nl387-np19-rf20-itq (query)            11_058.63       274.41    11_333.04       0.0881   136.442481         6.35
IVF-Binary-256-nl387-np27-rf5-itq (query)             11_058.63       198.69    11_257.31       0.0182   237.005507         6.35
IVF-Binary-256-nl387-np27-rf10-itq (query)            11_058.63       223.73    11_282.36       0.0359   188.916676         6.35
IVF-Binary-256-nl387-np27-rf20-itq (query)            11_058.63       304.25    11_362.87       0.0728   149.971472         6.35
IVF-Binary-256-nl387-itq (self)                       11_058.63     2_259.38    13_318.01       0.0443   173.705136         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_858.97       160.80    12_019.77       0.0048          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_858.97       164.31    12_023.29       0.0042          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_858.97       181.01    12_039.99       0.0037          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_858.97       196.68    12_055.65       0.0241   213.678718         6.51
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_858.97       225.42    12_084.39       0.0476   168.381951         6.51
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_858.97       285.28    12_144.25       0.0962   127.673372         6.51
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_858.97       193.73    12_052.70       0.0208   226.070632         6.51
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_858.97       228.78    12_087.75       0.0412   180.096713         6.51
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_858.97       284.17    12_143.14       0.0825   138.553616         6.51
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_858.97       197.92    12_056.90       0.0182   237.496159         6.51
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_858.97       232.57    12_091.55       0.0363   190.644118         6.51
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_858.97       292.85    12_151.83       0.0723   147.504525         6.51
IVF-Binary-256-nl547-itq (self)                       11_858.97     2_241.62    14_100.59       0.0491   168.890799         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          16_062.60       277.02    16_339.62       0.1585          NaN        11.07
IVF-Binary-512-nl273-np16-rf0-random (query)          16_062.60       292.29    16_354.89       0.1556          NaN        11.07
IVF-Binary-512-nl273-np23-rf0-random (query)          16_062.60       335.45    16_398.05       0.1542          NaN        11.07
IVF-Binary-512-nl273-np13-rf5-random (query)          16_062.60       370.06    16_432.65       0.3312    46.806070        11.07
IVF-Binary-512-nl273-np13-rf10-random (query)         16_062.60       396.79    16_459.39       0.4499    28.530589        11.07
IVF-Binary-512-nl273-np13-rf20-random (query)         16_062.60       479.80    16_542.39       0.5930    16.162221        11.07
IVF-Binary-512-nl273-np16-rf5-random (query)          16_062.60       367.36    16_429.95       0.3258    47.985698        11.07
IVF-Binary-512-nl273-np16-rf10-random (query)         16_062.60       421.01    16_483.60       0.4430    29.305110        11.07
IVF-Binary-512-nl273-np16-rf20-random (query)         16_062.60       548.38    16_610.97       0.5866    16.621724        11.07
IVF-Binary-512-nl273-np23-rf5-random (query)          16_062.60       409.01    16_471.60       0.3225    48.701864        11.07
IVF-Binary-512-nl273-np23-rf10-random (query)         16_062.60       491.89    16_554.49       0.4390    29.816257        11.07
IVF-Binary-512-nl273-np23-rf20-random (query)         16_062.60       595.82    16_658.41       0.5818    16.971071        11.07
IVF-Binary-512-nl273-random (self)                    16_062.60     4_199.00    20_261.60       0.4456    29.092256        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)          16_521.23       292.40    16_813.63       0.1562          NaN        11.18
IVF-Binary-512-nl387-np27-rf0-random (query)          16_521.23       323.13    16_844.36       0.1542          NaN        11.18
IVF-Binary-512-nl387-np19-rf5-random (query)          16_521.23       366.17    16_887.41       0.3276    47.541240        11.18
IVF-Binary-512-nl387-np19-rf10-random (query)         16_521.23       411.03    16_932.26       0.4461    28.906371        11.18
IVF-Binary-512-nl387-np19-rf20-random (query)         16_521.23       495.06    17_016.29       0.5893    16.401606        11.18
IVF-Binary-512-nl387-np27-rf5-random (query)          16_521.23       399.30    16_920.53       0.3238    48.474272        11.18
IVF-Binary-512-nl387-np27-rf10-random (query)         16_521.23       455.10    16_976.33       0.4408    29.588087        11.18
IVF-Binary-512-nl387-np27-rf20-random (query)         16_521.23       548.16    17_069.39       0.5835    16.840666        11.18
IVF-Binary-512-nl387-random (self)                    16_521.23     4_096.35    20_617.58       0.4480    28.751399        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)          17_411.93       295.45    17_707.38       0.1574          NaN        11.34
IVF-Binary-512-nl547-np27-rf0-random (query)          17_411.93       307.35    17_719.28       0.1559          NaN        11.34
IVF-Binary-512-nl547-np33-rf0-random (query)          17_411.93       324.86    17_736.80       0.1546          NaN        11.34
IVF-Binary-512-nl547-np23-rf5-random (query)          17_411.93       365.49    17_777.42       0.3306    46.796057        11.34
IVF-Binary-512-nl547-np23-rf10-random (query)         17_411.93       408.79    17_820.73       0.4498    28.416225        11.34
IVF-Binary-512-nl547-np23-rf20-random (query)         17_411.93       489.24    17_901.17       0.5937    16.066491        11.34
IVF-Binary-512-nl547-np27-rf5-random (query)          17_411.93       375.37    17_787.30       0.3268    47.630979        11.34
IVF-Binary-512-nl547-np27-rf10-random (query)         17_411.93       425.47    17_837.41       0.4450    29.018144        11.34
IVF-Binary-512-nl547-np27-rf20-random (query)         17_411.93       528.28    17_940.21       0.5879    16.487216        11.34
IVF-Binary-512-nl547-np33-rf5-random (query)          17_411.93       406.87    17_818.81       0.3243    48.228451        11.34
IVF-Binary-512-nl547-np33-rf10-random (query)         17_411.93       458.30    17_870.23       0.4414    29.460866        11.34
IVF-Binary-512-nl547-np33-rf20-random (query)         17_411.93       544.04    17_955.97       0.5840    16.783482        11.34
IVF-Binary-512-nl547-random (self)                    17_411.93     4_101.46    21_513.39       0.4521    28.251501        11.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)             18_306.47       282.04    18_588.52       0.0976          NaN        11.07
IVF-Binary-512-nl273-np16-rf0-itq (query)             18_306.47       296.84    18_603.31       0.0903          NaN        11.07
IVF-Binary-512-nl273-np23-rf0-itq (query)             18_306.47       330.84    18_637.32       0.0879          NaN        11.07
IVF-Binary-512-nl273-np13-rf5-itq (query)             18_306.47       354.79    18_661.26       0.2222    75.897669        11.07
IVF-Binary-512-nl273-np13-rf10-itq (query)            18_306.47       400.26    18_706.73       0.3147    49.114348        11.07
IVF-Binary-512-nl273-np13-rf20-itq (query)            18_306.47       486.66    18_793.14       0.4362    30.338010        11.07
IVF-Binary-512-nl273-np16-rf5-itq (query)             18_306.47       371.86    18_678.33       0.2100    81.526193        11.07
IVF-Binary-512-nl273-np16-rf10-itq (query)            18_306.47       423.34    18_729.81       0.2999    52.867249        11.07
IVF-Binary-512-nl273-np16-rf20-itq (query)            18_306.47       511.06    18_817.53       0.4209    32.567948        11.07
IVF-Binary-512-nl273-np23-rf5-itq (query)             18_306.47       412.09    18_718.57       0.2050    83.770981        11.07
IVF-Binary-512-nl273-np23-rf10-itq (query)            18_306.47       471.23    18_777.70       0.2931    54.494866        11.07
IVF-Binary-512-nl273-np23-rf20-itq (query)            18_306.47       568.03    18_874.50       0.4129    33.630415        11.07
IVF-Binary-512-nl273-itq (self)                       18_306.47     4_218.91    22_525.38       0.3010    52.355476        11.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)             18_785.44       297.02    19_082.46       0.0918          NaN        11.18
IVF-Binary-512-nl387-np27-rf0-itq (query)             18_785.44       335.22    19_120.67       0.0892          NaN        11.18
IVF-Binary-512-nl387-np19-rf5-itq (query)             18_785.44       370.18    19_155.62       0.2125    79.962584        11.18
IVF-Binary-512-nl387-np19-rf10-itq (query)            18_785.44       413.31    19_198.75       0.3035    51.779923        11.18
IVF-Binary-512-nl387-np19-rf20-itq (query)            18_785.44       494.82    19_280.27       0.4252    31.882383        11.18
IVF-Binary-512-nl387-np27-rf5-itq (query)             18_785.44       408.06    19_193.50       0.2066    82.508429        11.18
IVF-Binary-512-nl387-np27-rf10-itq (query)            18_785.44       455.30    19_240.74       0.2953    53.862960        11.18
IVF-Binary-512-nl387-np27-rf20-itq (query)            18_785.44       543.16    19_328.60       0.4156    33.232527        11.18
IVF-Binary-512-nl387-itq (self)                       18_785.44     4_108.65    22_894.10       0.3054    51.114799        11.18
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)             19_593.87       299.24    19_893.11       0.0928          NaN        11.34
IVF-Binary-512-nl547-np27-rf0-itq (query)             19_593.87       332.69    19_926.56       0.0906          NaN        11.34
IVF-Binary-512-nl547-np33-rf0-itq (query)             19_593.87       326.38    19_920.25       0.0887          NaN        11.34
IVF-Binary-512-nl547-np23-rf5-itq (query)             19_593.87       367.18    19_961.05       0.2163    77.815923        11.34
IVF-Binary-512-nl547-np23-rf10-itq (query)            19_593.87       410.76    20_004.63       0.3094    50.225444        11.34
IVF-Binary-512-nl547-np23-rf20-itq (query)            19_593.87       501.22    20_095.09       0.4325    30.827118        11.34
IVF-Binary-512-nl547-np27-rf5-itq (query)             19_593.87       378.05    19_971.92       0.2113    79.816910        11.34
IVF-Binary-512-nl547-np27-rf10-itq (query)            19_593.87       426.45    20_020.32       0.3024    51.793071        11.34
IVF-Binary-512-nl547-np27-rf20-itq (query)            19_593.87       516.29    20_110.16       0.4241    31.830457        11.34
IVF-Binary-512-nl547-np33-rf5-itq (query)             19_593.87       398.41    19_992.28       0.2071    82.030970        11.34
IVF-Binary-512-nl547-np33-rf10-itq (query)            19_593.87       450.20    20_044.06       0.2963    53.421778        11.34
IVF-Binary-512-nl547-np33-rf20-itq (query)            19_593.87       534.04    20_127.91       0.4162    33.056877        11.34
IVF-Binary-512-nl547-itq (self)                       19_593.87     4_109.56    23_703.43       0.3107    49.702907        11.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         30_759.15       629.96    31_389.11       0.2364          NaN        20.72
IVF-Binary-1024-nl273-np16-rf0-random (query)         30_759.15       693.14    31_452.29       0.2347          NaN        20.72
IVF-Binary-1024-nl273-np23-rf0-random (query)         30_759.15       805.79    31_564.94       0.2338          NaN        20.72
IVF-Binary-1024-nl273-np13-rf5-random (query)         30_759.15       678.99    31_438.14       0.4942    24.374124        20.72
IVF-Binary-1024-nl273-np13-rf10-random (query)        30_759.15       726.46    31_485.61       0.6366    13.052042        20.72
IVF-Binary-1024-nl273-np13-rf20-random (query)        30_759.15       792.42    31_551.57       0.7781     6.233594        20.72
IVF-Binary-1024-nl273-np16-rf5-random (query)         30_759.15       730.61    31_489.76       0.4912    24.690673        20.72
IVF-Binary-1024-nl273-np16-rf10-random (query)        30_759.15       774.66    31_533.81       0.6340    13.219695        20.72
IVF-Binary-1024-nl273-np16-rf20-random (query)        30_759.15       855.58    31_614.73       0.7754     6.342967        20.72
IVF-Binary-1024-nl273-np23-rf5-random (query)         30_759.15       860.57    31_619.72       0.4896    24.875110        20.72
IVF-Binary-1024-nl273-np23-rf10-random (query)        30_759.15       902.54    31_661.69       0.6322    13.342830        20.72
IVF-Binary-1024-nl273-np23-rf20-random (query)        30_759.15       983.99    31_743.14       0.7737     6.421981        20.72
IVF-Binary-1024-nl273-random (self)                   30_759.15     7_724.12    38_483.27       0.6379    13.027924        20.72
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-random (query)         31_299.56       646.07    31_945.64       0.2353          NaN        20.84
IVF-Binary-1024-nl387-np27-rf0-random (query)         31_299.56       753.52    32_053.08       0.2342          NaN        20.84
IVF-Binary-1024-nl387-np19-rf5-random (query)         31_299.56       696.26    31_995.82       0.4923    24.565964        20.84
IVF-Binary-1024-nl387-np19-rf10-random (query)        31_299.56       744.28    32_043.84       0.6351    13.165115        20.84
IVF-Binary-1024-nl387-np19-rf20-random (query)        31_299.56       814.22    32_113.78       0.7765     6.301286        20.84
IVF-Binary-1024-nl387-np27-rf5-random (query)         31_299.56       795.88    32_095.45       0.4903    24.788303        20.84
IVF-Binary-1024-nl387-np27-rf10-random (query)        31_299.56       842.65    32_142.21       0.6328    13.317639        20.84
IVF-Binary-1024-nl387-np27-rf20-random (query)        31_299.56       909.55    32_209.11       0.7741     6.396815        20.84
IVF-Binary-1024-nl387-random (self)                   31_299.56     7_310.45    38_610.02       0.6389    12.968449        20.84
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-random (query)         32_213.86       633.91    32_847.78       0.2363          NaN        20.99
IVF-Binary-1024-nl547-np27-rf0-random (query)         32_213.86       660.19    32_874.05       0.2353          NaN        20.99
IVF-Binary-1024-nl547-np33-rf0-random (query)         32_213.86       723.65    32_937.51       0.2345          NaN        20.99
IVF-Binary-1024-nl547-np23-rf5-random (query)         32_213.86       671.98    32_885.85       0.4942    24.350197        20.99
IVF-Binary-1024-nl547-np23-rf10-random (query)        32_213.86       708.66    32_922.52       0.6371    13.012765        20.99
IVF-Binary-1024-nl547-np23-rf20-random (query)        32_213.86       792.47    33_006.34       0.7791     6.185553        20.99
IVF-Binary-1024-nl547-np27-rf5-random (query)         32_213.86       724.83    32_938.70       0.4918    24.619173        20.99
IVF-Binary-1024-nl547-np27-rf10-random (query)        32_213.86       755.04    32_968.90       0.6343    13.205686        20.99
IVF-Binary-1024-nl547-np27-rf20-random (query)        32_213.86       820.01    33_033.87       0.7760     6.320139        20.99
IVF-Binary-1024-nl547-np33-rf5-random (query)         32_213.86       759.63    32_973.49       0.4902    24.792155        20.99
IVF-Binary-1024-nl547-np33-rf10-random (query)        32_213.86       801.43    33_015.29       0.6327    13.309994        20.99
IVF-Binary-1024-nl547-np33-rf20-random (query)        32_213.86       871.85    33_085.71       0.7744     6.391359        20.99
IVF-Binary-1024-nl547-random (self)                   32_213.86     7_102.34    39_316.20       0.6414    12.802013        20.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            33_983.35       647.51    34_630.85       0.1995          NaN        20.72
IVF-Binary-1024-nl273-np16-rf0-itq (query)            33_983.35       701.14    34_684.49       0.1972          NaN        20.72
IVF-Binary-1024-nl273-np23-rf0-itq (query)            33_983.35       821.46    34_804.81       0.1959          NaN        20.72
IVF-Binary-1024-nl273-np13-rf5-itq (query)            33_983.35       704.19    34_687.54       0.4172    33.291521        20.72
IVF-Binary-1024-nl273-np13-rf10-itq (query)           33_983.35       741.11    34_724.46       0.5520    18.978683        20.72
IVF-Binary-1024-nl273-np13-rf20-itq (query)           33_983.35       800.06    34_783.40       0.6985     9.908255        20.72
IVF-Binary-1024-nl273-np16-rf5-itq (query)            33_983.35       760.12    34_743.46       0.4133    33.910620        20.72
IVF-Binary-1024-nl273-np16-rf10-itq (query)           33_983.35       791.30    34_774.64       0.5474    19.391421        20.72
IVF-Binary-1024-nl273-np16-rf20-itq (query)           33_983.35       873.37    34_856.72       0.6940    10.152299        20.72
IVF-Binary-1024-nl273-np23-rf5-itq (query)            33_983.35       875.76    34_859.10       0.4111    34.226334        20.72
IVF-Binary-1024-nl273-np23-rf10-itq (query)           33_983.35       914.52    34_897.87       0.5446    19.648231        20.72
IVF-Binary-1024-nl273-np23-rf20-itq (query)           33_983.35       994.40    34_977.75       0.6914    10.289526        20.72
IVF-Binary-1024-nl273-itq (self)                      33_983.35     7_893.41    41_876.76       0.5497    19.249374        20.72
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-itq (query)            34_346.03       661.07    35_007.10       0.1976          NaN        20.84
IVF-Binary-1024-nl387-np27-rf0-itq (query)            34_346.03       761.21    35_107.24       0.1963          NaN        20.84
IVF-Binary-1024-nl387-np19-rf5-itq (query)            34_346.03       708.20    35_054.23       0.4150    33.656789        20.84
IVF-Binary-1024-nl387-np19-rf10-itq (query)           34_346.03       747.89    35_093.92       0.5488    19.299001        20.84
IVF-Binary-1024-nl387-np19-rf20-itq (query)           34_346.03       824.24    35_170.28       0.6954    10.074423        20.84
IVF-Binary-1024-nl387-np27-rf5-itq (query)            34_346.03       813.89    35_159.92       0.4122    34.084996        20.84
IVF-Binary-1024-nl387-np27-rf10-itq (query)           34_346.03       862.12    35_208.15       0.5453    19.602657        20.84
IVF-Binary-1024-nl387-np27-rf20-itq (query)           34_346.03       927.61    35_273.64       0.6919    10.253922        20.84
IVF-Binary-1024-nl387-itq (self)                      34_346.03     7_473.72    41_819.75       0.5514    19.121436        20.84
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-itq (query)            35_328.16       645.24    35_973.40       0.1981          NaN        20.99
IVF-Binary-1024-nl547-np27-rf0-itq (query)            35_328.16       681.34    36_009.50       0.1970          NaN        20.99
IVF-Binary-1024-nl547-np33-rf0-itq (query)            35_328.16       734.10    36_062.26       0.1960          NaN        20.99
IVF-Binary-1024-nl547-np23-rf5-itq (query)            35_328.16       691.15    36_019.30       0.4170    33.326480        20.99
IVF-Binary-1024-nl547-np23-rf10-itq (query)           35_328.16       734.47    36_062.62       0.5521    19.006299        20.99
IVF-Binary-1024-nl547-np23-rf20-itq (query)           35_328.16       788.34    36_116.50       0.6991     9.899350        20.99
IVF-Binary-1024-nl547-np27-rf5-itq (query)            35_328.16       720.03    36_048.19       0.4138    33.792146        20.99
IVF-Binary-1024-nl547-np27-rf10-itq (query)           35_328.16       761.00    36_089.15       0.5483    19.321586        20.99
IVF-Binary-1024-nl547-np27-rf20-itq (query)           35_328.16       836.96    36_165.12       0.6949    10.116089        20.99
IVF-Binary-1024-nl547-np33-rf5-itq (query)            35_328.16       786.53    36_114.69       0.4118    34.123305        20.99
IVF-Binary-1024-nl547-np33-rf10-itq (query)           35_328.16       837.07    36_165.22       0.5459    19.535544        20.99
IVF-Binary-1024-nl547-np33-rf20-itq (query)           35_328.16       891.53    36_219.68       0.6924    10.247352        20.99
IVF-Binary-1024-nl547-itq (self)                      35_328.16     7_243.78    42_571.94       0.5543    18.871955        20.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-signed (query)           8_770.17       142.43     8_912.59       0.0975          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-signed (query)           8_770.17       148.88     8_919.05       0.0903          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-signed (query)           8_770.17       166.23     8_936.39       0.0874          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-signed (query)           8_770.17       195.62     8_965.79       0.2202    76.986041         6.24
IVF-Binary-256-nl273-np13-rf10-signed (query)          8_770.17       244.49     9_014.66       0.3106    49.904755         6.24
IVF-Binary-256-nl273-np13-rf20-signed (query)          8_770.17       327.81     9_097.97       0.4342    30.708365         6.24
IVF-Binary-256-nl273-np16-rf5-signed (query)           8_770.17       197.42     8_967.58       0.2092    81.570143         6.24
IVF-Binary-256-nl273-np16-rf10-signed (query)          8_770.17       252.94     9_023.11       0.2985    52.787271         6.24
IVF-Binary-256-nl273-np16-rf20-signed (query)          8_770.17       346.11     9_116.28       0.4205    32.516565         6.24
IVF-Binary-256-nl273-np23-rf5-signed (query)           8_770.17       210.90     8_981.06       0.2036    84.117063         6.24
IVF-Binary-256-nl273-np23-rf10-signed (query)          8_770.17       268.52     9_038.69       0.2912    54.726832         6.24
IVF-Binary-256-nl273-np23-rf20-signed (query)          8_770.17       366.85     9_137.02       0.4117    33.856600         6.24
IVF-Binary-256-nl273-signed (self)                     8_770.17     2_517.97    11_288.14       0.3019    52.280744         6.24
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-signed (query)           9_221.73       153.09     9_374.82       0.0917          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-signed (query)           9_221.73       165.07     9_386.80       0.0890          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-signed (query)           9_221.73       208.19     9_429.92       0.2131    79.271693         6.35
IVF-Binary-256-nl387-np19-rf10-signed (query)          9_221.73       255.05     9_476.78       0.3030    51.248429         6.35
IVF-Binary-256-nl387-np19-rf20-signed (query)          9_221.73       338.91     9_560.64       0.4279    31.362689         6.35
IVF-Binary-256-nl387-np27-rf5-signed (query)           9_221.73       221.05     9_442.78       0.2067    82.316705         6.35
IVF-Binary-256-nl387-np27-rf10-signed (query)          9_221.73       283.53     9_505.26       0.2951    53.352131         6.35
IVF-Binary-256-nl387-np27-rf20-signed (query)          9_221.73       373.42     9_595.15       0.4171    32.933454         6.35
IVF-Binary-256-nl387-signed (self)                     9_221.73     2_551.75    11_773.48       0.3066    50.676822         6.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-signed (query)          10_237.07       160.28    10_397.35       0.0929          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-signed (query)          10_237.07       164.64    10_401.71       0.0907          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-signed (query)          10_237.07       171.41    10_408.48       0.0888          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-signed (query)          10_237.07       217.69    10_454.76       0.2167    76.983647         6.51
IVF-Binary-256-nl547-np23-rf10-signed (query)         10_237.07       262.03    10_499.10       0.3093    49.456854         6.51
IVF-Binary-256-nl547-np23-rf20-signed (query)         10_237.07       340.79    10_577.86       0.4348    30.219562         6.51
IVF-Binary-256-nl547-np27-rf5-signed (query)          10_237.07       219.28    10_456.35       0.2121    79.100710         6.51
IVF-Binary-256-nl547-np27-rf10-signed (query)         10_237.07       268.48    10_505.55       0.3024    51.043113         6.51
IVF-Binary-256-nl547-np27-rf20-signed (query)         10_237.07       359.36    10_596.43       0.4263    31.325323         6.51
IVF-Binary-256-nl547-np33-rf5-signed (query)          10_237.07       231.25    10_468.32       0.2083    81.042925         6.51
IVF-Binary-256-nl547-np33-rf10-signed (query)         10_237.07       277.18    10_514.25       0.2966    52.484571         6.51
IVF-Binary-256-nl547-np33-rf20-signed (query)         10_237.07       363.07    10_600.14       0.4193    32.278388         6.51
IVF-Binary-256-nl547-signed (self)                    10_237.07     2_597.51    12_834.58       0.3132    48.780140         6.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

<details>
<summary><b>Binary - Euclidean (LowRank - 512 bit)</b>:</summary>
</br>
================================================================================================================================
Benchmark: 150k cells, 512D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        62.55    32_889.81    32_952.36       1.0000     0.000000       292.97
Exhaustive (self)                                         62.55   326_664.49   326_727.04       1.0000     0.000000       292.97
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)             16_883.13       669.32    17_552.46       0.0845          NaN         5.08
ExhaustiveBinary-256-random-rf5 (query)               16_883.13       729.62    17_612.75       0.1952   176.518712         5.08
ExhaustiveBinary-256-random-rf10 (query)              16_883.13       821.94    17_705.07       0.2797   115.500230         5.08
ExhaustiveBinary-256-random-rf20 (query)              16_883.13       959.79    17_842.93       0.3971    71.994085         5.08
ExhaustiveBinary-256-random (self)                    16_883.13     7_992.01    24_875.14       0.2816   114.414527         5.08
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                19_465.16       662.88    20_128.04       0.0028          NaN         5.08
ExhaustiveBinary-256-itq-rf5 (query)                  19_465.16       698.99    20_164.15       0.0128   483.900180         5.08
ExhaustiveBinary-256-itq-rf10 (query)                 19_465.16       739.32    20_204.48       0.0254   368.460394         5.08
ExhaustiveBinary-256-itq-rf20 (query)                 19_465.16       841.17    20_306.33       0.0504   278.818201         5.08
ExhaustiveBinary-256-itq (self)                       19_465.16     7_425.42    26_890.58       0.0254   369.278263         5.08
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)             33_025.83     1_190.17    34_216.00       0.1540          NaN        10.16
ExhaustiveBinary-512-random-rf5 (query)               33_025.83     1_275.87    34_301.70       0.3232   101.322531        10.16
ExhaustiveBinary-512-random-rf10 (query)              33_025.83     1_337.76    34_363.59       0.4401    61.958812        10.16
ExhaustiveBinary-512-random-rf20 (query)              33_025.83     1_512.67    34_538.50       0.5831    35.248181        10.16
ExhaustiveBinary-512-random (self)                    33_025.83    13_345.18    46_371.01       0.4408    61.967741        10.16
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                38_392.24     1_188.02    39_580.26       0.0030          NaN        10.16
ExhaustiveBinary-512-itq-rf5 (query)                  38_392.24     1_235.99    39_628.23       0.0134   481.784629        10.16
ExhaustiveBinary-512-itq-rf10 (query)                 38_392.24     1_277.04    39_669.27       0.0262   366.971659        10.16
ExhaustiveBinary-512-itq-rf20 (query)                 38_392.24     1_398.68    39_790.92       0.0517   277.301568        10.16
ExhaustiveBinary-512-itq (self)                       38_392.24    12_783.76    51_175.99       0.0262   367.742681        10.16
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            65_569.46     2_078.17    67_647.63       0.2347          NaN        20.31
ExhaustiveBinary-1024-random-rf5 (query)              65_569.46     2_308.52    67_877.97       0.4901    51.452302        20.31
ExhaustiveBinary-1024-random-rf10 (query)             65_569.46     2_272.11    67_841.57       0.6349    27.349367        20.31
ExhaustiveBinary-1024-random-rf20 (query)             65_569.46     2_474.30    68_043.75       0.7761    13.107840        20.31
ExhaustiveBinary-1024-random (self)                   65_569.46    22_657.04    88_226.49       0.6349    27.516157        20.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq_no_rr (query)               72_296.58     2_112.94    74_409.53       0.1537          NaN        20.31
ExhaustiveBinary-1024-itq-rf5 (query)                 72_296.58     2_194.96    74_491.55       0.3240   102.060707        20.31
ExhaustiveBinary-1024-itq-rf10 (query)                72_296.58     2_283.66    74_580.24       0.4416    62.332213        20.31
ExhaustiveBinary-1024-itq-rf20 (query)                72_296.58     2_497.93    74_794.51       0.5822    35.683979        20.31
ExhaustiveBinary-1024-itq (self)                      72_296.58    22_852.22    95_148.81       0.4440    61.950177        20.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-signed_no_rr (query)             33_142.13     1_210.16    34_352.29       0.1540          NaN        10.16
ExhaustiveBinary-512-signed-rf5 (query)               33_142.13     1_284.19    34_426.32       0.3232   101.322531        10.16
ExhaustiveBinary-512-signed-rf10 (query)              33_142.13     1_362.80    34_504.93       0.4401    61.958812        10.16
ExhaustiveBinary-512-signed-rf20 (query)              33_142.13     1_519.27    34_661.40       0.5831    35.248181        10.16
ExhaustiveBinary-512-signed (self)                    33_142.13    13_696.25    46_838.37       0.4408    61.967741        10.16
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)          18_852.14       271.38    19_123.53       0.0957          NaN         6.76
IVF-Binary-256-nl273-np16-rf0-random (query)          18_852.14       296.83    19_148.98       0.0892          NaN         6.76
IVF-Binary-256-nl273-np23-rf0-random (query)          18_852.14       294.31    19_146.45       0.0867          NaN         6.76
IVF-Binary-256-nl273-np13-rf5-random (query)          18_852.14       356.20    19_208.35       0.2167   160.665291         6.76
IVF-Binary-256-nl273-np13-rf10-random (query)         18_852.14       414.79    19_266.94       0.3057   104.612323         6.76
IVF-Binary-256-nl273-np13-rf20-random (query)         18_852.14       550.82    19_402.96       0.4278    64.592741         6.76
IVF-Binary-256-nl273-np16-rf5-random (query)          18_852.14       365.78    19_217.93       0.2062   169.437515         6.76
IVF-Binary-256-nl273-np16-rf10-random (query)         18_852.14       438.56    19_290.70       0.2940   110.114643         6.76
IVF-Binary-256-nl273-np16-rf20-random (query)         18_852.14       556.87    19_409.01       0.4150    67.983527         6.76
IVF-Binary-256-nl273-np23-rf5-random (query)          18_852.14       377.16    19_229.31       0.2015   173.833155         6.76
IVF-Binary-256-nl273-np23-rf10-random (query)         18_852.14       473.27    19_325.41       0.2882   113.201701         6.76
IVF-Binary-256-nl273-np23-rf20-random (query)         18_852.14       582.49    19_434.64       0.4078    70.076568         6.76
IVF-Binary-256-nl273-random (self)                    18_852.14     4_262.12    23_114.27       0.2968   108.971982         6.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-random (query)          19_515.78       297.44    19_813.21       0.0905          NaN         6.98
IVF-Binary-256-nl387-np27-rf0-random (query)          19_515.78       339.38    19_855.16       0.0875          NaN         6.98
IVF-Binary-256-nl387-np19-rf5-random (query)          19_515.78       382.62    19_898.39       0.2082   167.908692         6.98
IVF-Binary-256-nl387-np19-rf10-random (query)         19_515.78       448.89    19_964.67       0.2972   108.942427         6.98
IVF-Binary-256-nl387-np19-rf20-random (query)         19_515.78       559.26    20_075.04       0.4179    67.501987         6.98
IVF-Binary-256-nl387-np27-rf5-random (query)          19_515.78       396.76    19_912.54       0.2023   173.626664         6.98
IVF-Binary-256-nl387-np27-rf10-random (query)         19_515.78       464.38    19_980.16       0.2892   113.016527         6.98
IVF-Binary-256-nl387-np27-rf20-random (query)         19_515.78       593.41    20_109.19       0.4081    70.109372         6.98
IVF-Binary-256-nl387-random (self)                    19_515.78     4_347.39    23_863.16       0.2995   108.124364         6.98
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-random (query)          21_028.79       310.52    21_339.31       0.0929          NaN         7.29
IVF-Binary-256-nl547-np27-rf0-random (query)          21_028.79       315.11    21_343.89       0.0907          NaN         7.29
IVF-Binary-256-nl547-np33-rf0-random (query)          21_028.79       323.60    21_352.39       0.0884          NaN         7.29
IVF-Binary-256-nl547-np23-rf5-random (query)          21_028.79       420.69    21_449.47       0.2138   161.993936         7.29
IVF-Binary-256-nl547-np23-rf10-random (query)         21_028.79       451.78    21_480.57       0.3047   104.830137         7.29
IVF-Binary-256-nl547-np23-rf20-random (query)         21_028.79       563.19    21_591.98       0.4280    64.333361         7.29
IVF-Binary-256-nl547-np27-rf5-random (query)          21_028.79       396.36    21_425.15       0.2085   166.628896         7.29
IVF-Binary-256-nl547-np27-rf10-random (query)         21_028.79       463.45    21_492.23       0.2973   108.206512         7.29
IVF-Binary-256-nl547-np27-rf20-random (query)         21_028.79       574.27    21_603.06       0.4193    66.592847         7.29
IVF-Binary-256-nl547-np33-rf5-random (query)          21_028.79       407.31    21_436.10       0.2039   170.793158         7.29
IVF-Binary-256-nl547-np33-rf10-random (query)         21_028.79       474.30    21_503.09       0.2916   111.064499         7.29
IVF-Binary-256-nl547-np33-rf20-random (query)         21_028.79       592.60    21_621.39       0.4120    68.556604         7.29
IVF-Binary-256-nl547-random (self)                    21_028.79     4_502.26    25_531.05       0.3064   103.792337         7.29
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)             21_594.93       274.11    21_869.04       0.0061          NaN         6.76
IVF-Binary-256-nl273-np16-rf0-itq (query)             21_594.93       280.27    21_875.20       0.0040          NaN         6.76
IVF-Binary-256-nl273-np23-rf0-itq (query)             21_594.93       302.65    21_897.58       0.0033          NaN         6.76
IVF-Binary-256-nl273-np13-rf5-itq (query)             21_594.93       339.70    21_934.64       0.0291   427.889671         6.76
IVF-Binary-256-nl273-np13-rf10-itq (query)            21_594.93       377.26    21_972.20       0.0589   326.853795         6.76
IVF-Binary-256-nl273-np13-rf20-itq (query)            21_594.93       474.40    22_069.34       0.1182   244.496794         6.76
IVF-Binary-256-nl273-np16-rf5-itq (query)             21_594.93       341.51    21_936.44       0.0189   473.816716         6.76
IVF-Binary-256-nl273-np16-rf10-itq (query)            21_594.93       385.62    21_980.56       0.0381   371.111667         6.76
IVF-Binary-256-nl273-np16-rf20-itq (query)            21_594.93       491.12    22_086.05       0.0767   288.546859         6.76
IVF-Binary-256-nl273-np23-rf5-itq (query)             21_594.93       348.00    21_942.93       0.0163   495.201920         6.76
IVF-Binary-256-nl273-np23-rf10-itq (query)            21_594.93       406.17    22_001.11       0.0329   391.192628         6.76
IVF-Binary-256-nl273-np23-rf20-itq (query)            21_594.93       508.00    22_102.94       0.0659   308.783928         6.76
IVF-Binary-256-nl273-itq (self)                       21_594.93     3_840.77    25_435.71       0.0379   371.738968         6.76
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl387-np19-rf0-itq (query)             22_420.86       293.24    22_714.11       0.0043          NaN         6.98
IVF-Binary-256-nl387-np27-rf0-itq (query)             22_420.86       313.31    22_734.17       0.0034          NaN         6.98
IVF-Binary-256-nl387-np19-rf5-itq (query)             22_420.86       349.23    22_770.10       0.0211   461.235087         6.98
IVF-Binary-256-nl387-np19-rf10-itq (query)            22_420.86       398.06    22_818.92       0.0418   362.314752         6.98
IVF-Binary-256-nl387-np19-rf20-itq (query)            22_420.86       492.47    22_913.33       0.0836   281.793643         6.98
IVF-Binary-256-nl387-np27-rf5-itq (query)             22_420.86       360.67    22_781.54       0.0169   497.976409         6.98
IVF-Binary-256-nl387-np27-rf10-itq (query)            22_420.86       424.08    22_844.94       0.0333   396.379403         6.98
IVF-Binary-256-nl387-np27-rf20-itq (query)            22_420.86       515.81    22_936.68       0.0673   313.223385         6.98
IVF-Binary-256-nl387-itq (self)                       22_420.86     3_928.64    26_349.50       0.0424   362.147177         6.98
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl547-np23-rf0-itq (query)             23_757.89       311.61    24_069.50       0.0051          NaN         7.30
IVF-Binary-256-nl547-np27-rf0-itq (query)             23_757.89       316.15    24_074.04       0.0044          NaN         7.30
IVF-Binary-256-nl547-np33-rf0-itq (query)             23_757.89       323.42    24_081.31       0.0037          NaN         7.30
IVF-Binary-256-nl547-np23-rf5-itq (query)             23_757.89       372.78    24_130.67       0.0247   440.615430         7.30
IVF-Binary-256-nl547-np23-rf10-itq (query)            23_757.89       424.25    24_182.14       0.0489   345.732137         7.30
IVF-Binary-256-nl547-np23-rf20-itq (query)            23_757.89       511.19    24_269.08       0.0968   261.935671         7.30
IVF-Binary-256-nl547-np27-rf5-itq (query)             23_757.89       383.22    24_141.11       0.0216   462.825801         7.30
IVF-Binary-256-nl547-np27-rf10-itq (query)            23_757.89       428.22    24_186.11       0.0427   367.074711         7.30
IVF-Binary-256-nl547-np27-rf20-itq (query)            23_757.89       518.05    24_275.94       0.0846   281.454411         7.30
IVF-Binary-256-nl547-np33-rf5-itq (query)             23_757.89       383.34    24_141.23       0.0186   484.778469         7.30
IVF-Binary-256-nl547-np33-rf10-itq (query)            23_757.89       434.62    24_192.51       0.0367   388.753131         7.30
IVF-Binary-256-nl547-np33-rf20-itq (query)            23_757.89       529.75    24_287.64       0.0724   300.718784         7.30
IVF-Binary-256-nl547-itq (self)                       23_757.89     4_152.41    27_910.30       0.0486   346.169069         7.30
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          35_074.87       526.65    35_601.51       0.1599          NaN        11.84
IVF-Binary-512-nl273-np16-rf0-random (query)          35_074.87       550.33    35_625.20       0.1571          NaN        11.84
IVF-Binary-512-nl273-np23-rf0-random (query)          35_074.87       594.40    35_669.27       0.1560          NaN        11.84
IVF-Binary-512-nl273-np13-rf5-random (query)          35_074.87       638.49    35_713.36       0.3353    96.351106        11.84
IVF-Binary-512-nl273-np13-rf10-random (query)         35_074.87       689.88    35_764.75       0.4555    58.420827        11.84
IVF-Binary-512-nl273-np13-rf20-random (query)         35_074.87       795.21    35_870.08       0.5993    32.991403        11.84
IVF-Binary-512-nl273-np16-rf5-random (query)          35_074.87       640.96    35_715.83       0.3303    98.570883        11.84
IVF-Binary-512-nl273-np16-rf10-random (query)         35_074.87       713.82    35_788.69       0.4492    59.882774        11.84
IVF-Binary-512-nl273-np16-rf20-random (query)         35_074.87       827.74    35_902.61       0.5929    33.882468        11.84
IVF-Binary-512-nl273-np23-rf5-random (query)          35_074.87       696.60    35_771.47       0.3275    99.795531        11.84
IVF-Binary-512-nl273-np23-rf10-random (query)         35_074.87       770.89    35_845.76       0.4458    60.780937        11.84
IVF-Binary-512-nl273-np23-rf20-random (query)         35_074.87       900.35    35_975.22       0.5891    34.439524        11.84
IVF-Binary-512-nl273-random (self)                    35_074.87     7_097.29    42_172.16       0.4505    59.843277        11.84
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-random (query)          36_182.09       551.58    36_733.67       0.1577          NaN        12.06
IVF-Binary-512-nl387-np27-rf0-random (query)          36_182.09       587.98    36_770.07       0.1560          NaN        12.06
IVF-Binary-512-nl387-np19-rf5-random (query)          36_182.09       644.79    36_826.87       0.3323    97.801190        12.06
IVF-Binary-512-nl387-np19-rf10-random (query)         36_182.09       707.82    36_889.91       0.4506    59.645195        12.06
IVF-Binary-512-nl387-np19-rf20-random (query)         36_182.09       815.93    36_998.02       0.5948    33.653650        12.06
IVF-Binary-512-nl387-np27-rf5-random (query)          36_182.09       682.80    36_864.89       0.3287    99.396340        12.06
IVF-Binary-512-nl387-np27-rf10-random (query)         36_182.09       754.53    36_936.62       0.4458    60.810695        12.06
IVF-Binary-512-nl387-np27-rf20-random (query)         36_182.09       881.18    37_063.27       0.5898    34.355937        12.06
IVF-Binary-512-nl387-random (self)                    36_182.09     6_948.09    43_130.18       0.4521    59.458274        12.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-random (query)          37_379.06       563.71    37_942.77       0.1595          NaN        12.37
IVF-Binary-512-nl547-np27-rf0-random (query)          37_379.06       577.99    37_957.04       0.1580          NaN        12.37
IVF-Binary-512-nl547-np33-rf0-random (query)          37_379.06       599.97    37_979.03       0.1568          NaN        12.37
IVF-Binary-512-nl547-np23-rf5-random (query)          37_379.06       647.82    38_026.88       0.3360    96.088527        12.37
IVF-Binary-512-nl547-np23-rf10-random (query)         37_379.06       706.38    38_085.44       0.4559    58.273753        12.37
IVF-Binary-512-nl547-np23-rf20-random (query)         37_379.06       815.84    38_194.90       0.6010    32.712228        12.37
IVF-Binary-512-nl547-np27-rf5-random (query)          37_379.06       661.03    38_040.09       0.3326    97.548526        12.37
IVF-Binary-512-nl547-np27-rf10-random (query)         37_379.06       723.89    38_102.95       0.4509    59.474107        12.37
IVF-Binary-512-nl547-np27-rf20-random (query)         37_379.06       844.38    38_223.44       0.5953    33.508355        12.37
IVF-Binary-512-nl547-np33-rf5-random (query)          37_379.06       697.18    38_076.24       0.3300    98.653713        12.37
IVF-Binary-512-nl547-np33-rf10-random (query)         37_379.06       757.74    38_136.80       0.4475    60.241849        12.37
IVF-Binary-512-nl547-np33-rf20-random (query)         37_379.06       882.08    38_261.14       0.5916    34.043211        12.37
IVF-Binary-512-nl547-random (self)                    37_379.06     7_122.84    44_501.90       0.4567    58.235755        12.37
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)             40_687.75       636.71    41_324.46       0.0064          NaN        11.84
IVF-Binary-512-nl273-np16-rf0-itq (query)             40_687.75       563.85    41_251.60       0.0042          NaN        11.84
IVF-Binary-512-nl273-np23-rf0-itq (query)             40_687.75       600.30    41_288.05       0.0036          NaN        11.84
IVF-Binary-512-nl273-np13-rf5-itq (query)             40_687.75       595.59    41_283.34       0.0299   426.096978        11.84
IVF-Binary-512-nl273-np13-rf10-itq (query)            40_687.75       651.14    41_338.89       0.0599   325.315319        11.84
IVF-Binary-512-nl273-np13-rf20-itq (query)            40_687.75       735.32    41_423.07       0.1200   242.983863        11.84
IVF-Binary-512-nl273-np16-rf5-itq (query)             40_687.75       617.67    41_305.42       0.0196   471.936710        11.84
IVF-Binary-512-nl273-np16-rf10-itq (query)            40_687.75       662.31    41_350.06       0.0391   369.472730        11.84
IVF-Binary-512-nl273-np16-rf20-itq (query)            40_687.75       767.92    41_455.67       0.0784   286.942014        11.84
IVF-Binary-512-nl273-np23-rf5-itq (query)             40_687.75       674.67    41_362.42       0.0168   493.358953        11.84
IVF-Binary-512-nl273-np23-rf10-itq (query)            40_687.75       719.80    41_407.55       0.0337   389.546798        11.84
IVF-Binary-512-nl273-np23-rf20-itq (query)            40_687.75       818.64    41_506.39       0.0674   307.026485        11.84
IVF-Binary-512-nl273-itq (self)                       40_687.75     6_613.64    47_301.39       0.0390   369.958255        11.84
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-itq (query)             41_455.31       567.07    42_022.38       0.0044          NaN        12.06
IVF-Binary-512-nl387-np27-rf0-itq (query)             41_455.31       605.45    42_060.76       0.0035          NaN        12.06
IVF-Binary-512-nl387-np19-rf5-itq (query)             41_455.31       622.48    42_077.79       0.0218   459.819374        12.06
IVF-Binary-512-nl387-np19-rf10-itq (query)            41_455.31       661.16    42_116.47       0.0426   361.116021        12.06
IVF-Binary-512-nl387-np19-rf20-itq (query)            41_455.31       753.59    42_208.89       0.0849   280.515789        12.06
IVF-Binary-512-nl387-np27-rf5-itq (query)             41_455.31       658.32    42_113.62       0.0175   496.470791        12.06
IVF-Binary-512-nl387-np27-rf10-itq (query)            41_455.31       707.34    42_162.64       0.0341   394.989471        12.06
IVF-Binary-512-nl387-np27-rf20-itq (query)            41_455.31       819.70    42_275.01       0.0687   311.735574        12.06
IVF-Binary-512-nl387-itq (self)                       41_455.31     6_582.04    48_037.35       0.0433   360.817949        12.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-itq (query)             42_783.00       576.24    43_359.24       0.0053          NaN        12.37
IVF-Binary-512-nl547-np27-rf0-itq (query)             42_783.00       594.28    43_377.28       0.0046          NaN        12.37
IVF-Binary-512-nl547-np33-rf0-itq (query)             42_783.00       609.20    43_392.20       0.0038          NaN        12.37
IVF-Binary-512-nl547-np23-rf5-itq (query)             42_783.00       639.61    43_422.61       0.0253   439.140137        12.37
IVF-Binary-512-nl547-np23-rf10-itq (query)            42_783.00       692.18    43_475.18       0.0499   344.140967        12.37
IVF-Binary-512-nl547-np23-rf20-itq (query)            42_783.00       765.93    43_548.93       0.0982   260.462666        12.37
IVF-Binary-512-nl547-np27-rf5-itq (query)             42_783.00       645.00    43_428.00       0.0222   461.372143        12.37
IVF-Binary-512-nl547-np27-rf10-itq (query)            42_783.00       691.46    43_474.46       0.0437   365.311496        12.37
IVF-Binary-512-nl547-np27-rf20-itq (query)            42_783.00       793.02    43_576.02       0.0859   279.780281        12.37
IVF-Binary-512-nl547-np33-rf5-itq (query)             42_783.00       682.22    43_465.22       0.0191   483.242519        12.37
IVF-Binary-512-nl547-np33-rf10-itq (query)            42_783.00       730.75    43_513.75       0.0376   386.724513        12.37
IVF-Binary-512-nl547-np33-rf20-itq (query)            42_783.00       808.84    43_591.84       0.0738   298.620954        12.37
IVF-Binary-512-nl547-itq (self)                       42_783.00     6_724.27    49_507.27       0.0496   344.599554        12.37
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         67_804.63     1_094.06    68_898.69       0.2390          NaN        21.99
IVF-Binary-1024-nl273-np16-rf0-random (query)         67_804.63     1_158.36    68_962.98       0.2374          NaN        21.99
IVF-Binary-1024-nl273-np23-rf0-random (query)         67_804.63     1_285.37    69_089.99       0.2364          NaN        21.99
IVF-Binary-1024-nl273-np13-rf5-random (query)         67_804.63     1_157.98    68_962.61       0.4975    49.846542        21.99
IVF-Binary-1024-nl273-np13-rf10-random (query)        67_804.63     1_200.45    69_005.08       0.6423    26.412121        21.99
IVF-Binary-1024-nl273-np13-rf20-random (query)        67_804.63     1_309.84    69_114.47       0.7822    12.585593        21.99
IVF-Binary-1024-nl273-np16-rf5-random (query)         67_804.63     1_215.38    69_020.00       0.4952    50.384320        21.99
IVF-Binary-1024-nl273-np16-rf10-random (query)        67_804.63     1_261.66    69_066.29       0.6395    26.778764        21.99
IVF-Binary-1024-nl273-np16-rf20-random (query)        67_804.63     1_400.17    69_204.80       0.7797    12.789591        21.99
IVF-Binary-1024-nl273-np23-rf5-random (query)         67_804.63     1_393.78    69_198.40       0.4935    50.734683        21.99
IVF-Binary-1024-nl273-np23-rf10-random (query)        67_804.63     1_416.76    69_221.38       0.6378    27.000161        21.99
IVF-Binary-1024-nl273-np23-rf20-random (query)        67_804.63     1_525.20    69_329.82       0.7785    12.894928        21.99
IVF-Binary-1024-nl273-random (self)                   67_804.63    12_496.45    80_301.08       0.6396    26.902952        21.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-random (query)         68_732.65     1_102.98    69_835.62       0.2378          NaN        22.21
IVF-Binary-1024-nl387-np27-rf0-random (query)         68_732.65     1_229.96    69_962.61       0.2368          NaN        22.21
IVF-Binary-1024-nl387-np19-rf5-random (query)         68_732.65     1_161.72    69_894.37       0.4954    50.312337        22.21
IVF-Binary-1024-nl387-np19-rf10-random (query)        68_732.65     1_216.20    69_948.85       0.6402    26.683907        22.21
IVF-Binary-1024-nl387-np19-rf20-random (query)        68_732.65     1_320.44    70_053.09       0.7806    12.730999        22.21
IVF-Binary-1024-nl387-np27-rf5-random (query)         68_732.65     1_275.27    70_007.92       0.4932    50.850925        22.21
IVF-Binary-1024-nl387-np27-rf10-random (query)        68_732.65     1_324.42    70_057.07       0.6377    27.044311        22.21
IVF-Binary-1024-nl387-np27-rf20-random (query)        68_732.65     1_438.92    70_171.56       0.7780    12.949822        22.21
IVF-Binary-1024-nl387-random (self)                   68_732.65    12_074.31    80_806.95       0.6404    26.811227        22.21
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-random (query)         69_792.95     1_111.01    70_903.96       0.2389          NaN        22.53
IVF-Binary-1024-nl547-np27-rf0-random (query)         69_792.95     1_146.73    70_939.68       0.2380          NaN        22.53
IVF-Binary-1024-nl547-np33-rf0-random (query)         69_792.95     1_204.69    70_997.64       0.2373          NaN        22.53
IVF-Binary-1024-nl547-np23-rf5-random (query)         69_792.95     1_159.13    70_952.08       0.4979    49.773250        22.53
IVF-Binary-1024-nl547-np23-rf10-random (query)        69_792.95     1_207.48    71_000.43       0.6430    26.336457        22.53
IVF-Binary-1024-nl547-np23-rf20-random (query)        69_792.95     1_320.80    71_113.75       0.7833    12.497196        22.53
IVF-Binary-1024-nl547-np27-rf5-random (query)         69_792.95     1_226.50    71_019.45       0.4956    50.305517        22.53
IVF-Binary-1024-nl547-np27-rf10-random (query)        69_792.95     1_336.94    71_129.89       0.6401    26.703161        22.53
IVF-Binary-1024-nl547-np27-rf20-random (query)        69_792.95     1_361.54    71_154.48       0.7804    12.758650        22.53
IVF-Binary-1024-nl547-np33-rf5-random (query)         69_792.95     1_268.68    71_061.63       0.4942    50.593966        22.53
IVF-Binary-1024-nl547-np33-rf10-random (query)        69_792.95     1_311.00    71_103.94       0.6385    26.897202        22.53
IVF-Binary-1024-nl547-np33-rf20-random (query)        69_792.95     1_428.95    71_221.90       0.7787    12.888226        22.53
IVF-Binary-1024-nl547-random (self)                   69_792.95    12_090.27    81_883.21       0.6432    26.439123        22.53
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            74_505.62     1_186.25    75_691.86       0.1608          NaN        21.99
IVF-Binary-1024-nl273-np16-rf0-itq (query)            74_505.62     1_167.31    75_672.93       0.1581          NaN        21.99
IVF-Binary-1024-nl273-np23-rf0-itq (query)            74_505.62     1_294.38    75_799.99       0.1567          NaN        21.99
IVF-Binary-1024-nl273-np13-rf5-itq (query)            74_505.62     1_167.34    75_672.95       0.3365    96.493855        21.99
IVF-Binary-1024-nl273-np13-rf10-itq (query)           74_505.62     1_210.39    75_716.00       0.4561    58.720492        21.99
IVF-Binary-1024-nl273-np13-rf20-itq (query)           74_505.62     1_314.24    75_819.85       0.5983    33.309435        21.99
IVF-Binary-1024-nl273-np16-rf5-itq (query)            74_505.62     1_219.48    75_725.09       0.3315    98.500862        21.99
IVF-Binary-1024-nl273-np16-rf10-itq (query)           74_505.62     1_271.75    75_777.36       0.4505    60.011650        21.99
IVF-Binary-1024-nl273-np16-rf20-itq (query)           74_505.62     1_395.60    75_901.21       0.5928    34.089455        21.99
IVF-Binary-1024-nl273-np23-rf5-itq (query)            74_505.62     1_352.66    75_858.27       0.3285   100.035358        21.99
IVF-Binary-1024-nl273-np23-rf10-itq (query)           74_505.62     1_407.33    75_912.95       0.4467    61.078614        21.99
IVF-Binary-1024-nl273-np23-rf20-itq (query)           74_505.62     1_519.50    76_025.12       0.5884    34.811803        21.99
IVF-Binary-1024-nl273-itq (self)                      74_505.62    12_693.24    87_198.86       0.4538    59.503677        21.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl387-np19-rf0-itq (query)            75_291.49     1_144.98    76_436.46       0.1576          NaN        22.22
IVF-Binary-1024-nl387-np27-rf0-itq (query)            75_291.49     1_237.21    76_528.70       0.1561          NaN        22.22
IVF-Binary-1024-nl387-np19-rf5-itq (query)            75_291.49     1_278.55    76_570.03       0.3330    98.183018        22.22
IVF-Binary-1024-nl387-np19-rf10-itq (query)           75_291.49     1_230.27    76_521.75       0.4522    59.742709        22.22
IVF-Binary-1024-nl387-np19-rf20-itq (query)           75_291.49     1_335.72    76_627.21       0.5933    34.082845        22.22
IVF-Binary-1024-nl387-np27-rf5-itq (query)            75_291.49     1_295.03    76_586.52       0.3290   100.157986        22.22
IVF-Binary-1024-nl387-np27-rf10-itq (query)           75_291.49     1_343.22    76_634.71       0.4471    61.128957        22.22
IVF-Binary-1024-nl387-np27-rf20-itq (query)           75_291.49     1_468.58    76_760.07       0.5884    34.883688        22.22
IVF-Binary-1024-nl387-itq (self)                      75_291.49    12_274.70    87_566.18       0.4552    59.254810        22.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl547-np23-rf0-itq (query)            76_613.79     1_124.81    77_738.60       0.1598          NaN        22.53
IVF-Binary-1024-nl547-np27-rf0-itq (query)            76_613.79     1_192.20    77_805.99       0.1581          NaN        22.53
IVF-Binary-1024-nl547-np33-rf0-itq (query)            76_613.79     1_248.33    77_862.12       0.1568          NaN        22.53
IVF-Binary-1024-nl547-np23-rf5-itq (query)            76_613.79     1_208.83    77_822.62       0.3361    96.368810        22.53
IVF-Binary-1024-nl547-np23-rf10-itq (query)           76_613.79     1_268.07    77_881.86       0.4574    58.222166        22.53
IVF-Binary-1024-nl547-np23-rf20-itq (query)           76_613.79     1_330.42    77_944.21       0.5999    32.952157        22.53
IVF-Binary-1024-nl547-np27-rf5-itq (query)            76_613.79     1_208.84    77_822.63       0.3324    98.133030        22.53
IVF-Binary-1024-nl547-np27-rf10-itq (query)           76_613.79     1_255.64    77_869.44       0.4526    59.450642        22.53
IVF-Binary-1024-nl547-np27-rf20-itq (query)           76_613.79     1_372.80    77_986.59       0.5938    33.883582        22.53
IVF-Binary-1024-nl547-np33-rf5-itq (query)            76_613.79     1_269.72    77_883.51       0.3299    99.313165        22.53
IVF-Binary-1024-nl547-np33-rf10-itq (query)           76_613.79     1_317.24    77_931.03       0.4492    60.336405        22.53
IVF-Binary-1024-nl547-np33-rf20-itq (query)           76_613.79     1_431.65    78_045.44       0.5897    34.526797        22.53
IVF-Binary-1024-nl547-itq (self)                      76_613.79    12_155.69    88_769.48       0.4601    57.883100        22.53
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-signed (query)          35_142.26       529.25    35_671.51       0.1599          NaN        11.84
IVF-Binary-512-nl273-np16-rf0-signed (query)          35_142.26       548.80    35_691.06       0.1571          NaN        11.84
IVF-Binary-512-nl273-np23-rf0-signed (query)          35_142.26       592.45    35_734.71       0.1560          NaN        11.84
IVF-Binary-512-nl273-np13-rf5-signed (query)          35_142.26       625.89    35_768.15       0.3353    96.351106        11.84
IVF-Binary-512-nl273-np13-rf10-signed (query)         35_142.26       688.32    35_830.58       0.4555    58.420827        11.84
IVF-Binary-512-nl273-np13-rf20-signed (query)         35_142.26       794.19    35_936.45       0.5993    32.991403        11.84
IVF-Binary-512-nl273-np16-rf5-signed (query)          35_142.26       643.10    35_785.36       0.3303    98.570883        11.84
IVF-Binary-512-nl273-np16-rf10-signed (query)         35_142.26       713.28    35_855.54       0.4492    59.882774        11.84
IVF-Binary-512-nl273-np16-rf20-signed (query)         35_142.26       831.94    35_974.20       0.5929    33.882468        11.84
IVF-Binary-512-nl273-np23-rf5-signed (query)          35_142.26       698.83    35_841.09       0.3275    99.795531        11.84
IVF-Binary-512-nl273-np23-rf10-signed (query)         35_142.26       772.23    35_914.49       0.4458    60.780937        11.84
IVF-Binary-512-nl273-np23-rf20-signed (query)         35_142.26       901.34    36_043.60       0.5891    34.439524        11.84
IVF-Binary-512-nl273-signed (self)                    35_142.26     7_080.64    42_222.90       0.4505    59.843277        11.84
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl387-np19-rf0-signed (query)          35_949.17       549.60    36_498.77       0.1577          NaN        12.06
IVF-Binary-512-nl387-np27-rf0-signed (query)          35_949.17       587.28    36_536.45       0.1560          NaN        12.06
IVF-Binary-512-nl387-np19-rf5-signed (query)          35_949.17       692.83    36_642.00       0.3323    97.801190        12.06
IVF-Binary-512-nl387-np19-rf10-signed (query)         35_949.17       704.06    36_653.23       0.4506    59.645195        12.06
IVF-Binary-512-nl387-np19-rf20-signed (query)         35_949.17       819.65    36_768.82       0.5948    33.653650        12.06
IVF-Binary-512-nl387-np27-rf5-signed (query)          35_949.17       689.38    36_638.55       0.3287    99.396340        12.06
IVF-Binary-512-nl387-np27-rf10-signed (query)         35_949.17       753.14    36_702.31       0.4458    60.810695        12.06
IVF-Binary-512-nl387-np27-rf20-signed (query)         35_949.17       886.13    36_835.30       0.5898    34.355937        12.06
IVF-Binary-512-nl387-signed (self)                    35_949.17     7_039.13    42_988.30       0.4521    59.458274        12.06
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl547-np23-rf0-signed (query)          37_477.07       562.24    38_039.31       0.1595          NaN        12.37
IVF-Binary-512-nl547-np27-rf0-signed (query)          37_477.07       582.66    38_059.73       0.1580          NaN        12.37
IVF-Binary-512-nl547-np33-rf0-signed (query)          37_477.07       602.56    38_079.63       0.1568          NaN        12.37
IVF-Binary-512-nl547-np23-rf5-signed (query)          37_477.07       649.71    38_126.78       0.3360    96.088527        12.37
IVF-Binary-512-nl547-np23-rf10-signed (query)         37_477.07       720.16    38_197.23       0.4559    58.273753        12.37
IVF-Binary-512-nl547-np23-rf20-signed (query)         37_477.07       818.48    38_295.55       0.6010    32.712228        12.37
IVF-Binary-512-nl547-np27-rf5-signed (query)          37_477.07       661.88    38_138.96       0.3326    97.548526        12.37
IVF-Binary-512-nl547-np27-rf10-signed (query)         37_477.07       728.84    38_205.91       0.4509    59.474107        12.37
IVF-Binary-512-nl547-np27-rf20-signed (query)         37_477.07       861.75    38_338.83       0.5953    33.508355        12.37
IVF-Binary-512-nl547-np33-rf5-signed (query)          37_477.07       687.22    38_164.30       0.3300    98.653713        12.37
IVF-Binary-512-nl547-np33-rf10-signed (query)         37_477.07       752.91    38_229.98       0.4475    60.241849        12.37
IVF-Binary-512-nl547-np33-rf20-signed (query)         37_477.07       869.98    38_347.05       0.5916    34.043211        12.37
IVF-Binary-512-nl547-signed (self)                    37_477.07     7_528.54    45_005.61       0.4567    58.235755        12.37
--------------------------------------------------------------------------------------------------------------------------------
<pre><code>

</code></pre>
</details>

### <u>RaBitQ (IVF and exhaustive)</u>

[RaBitQ](https://arxiv.org/abs/2405.12497) is an very powerful
quantisation that combines strong compression with excellent Recalls (even
without re-ranking!). It works better on higher dimensions. In the case of the
`ExhaustiveRaBitQ`, the quantiser itself generates a smaller number of centroids
for quantisation purposes (`sqrt(n)` centroids in this case). On the other
hand for the `IVF-RaBitQ` index, the IVF centroids are directly used for
centroid calculations in the quantiser. The only disadvantage over the binary
quantiser is the reduced query speed due to the more complex approximate
distance calculation.

**Key parameters *(RaBitQ)*:**

- *reranking*: The RaBitQ indices have the option to store the original vectors
  on disk. Once the RaBitQ-specific approximated distance has been leveraged to
  identify the most interesting potential neighbours, the on-disk vectors are
  loaded in and the results are re-ranked. A key parameter here is the
  reranking_factor, i.e., how many more vectors are reranked than the desired k.
  For example 10 means that `10 * k vectors` are scored and then re-ranked. The
  more candidates you allow here, the better the Recall.

**Key parameters *(IVF-specific)*:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

#### With 32 dimensions

<details>
<summary><b>RaBitQ - Euclidean (Gaussian)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.73     1_520.85     1_524.58       1.0000     0.000000        18.31
Exhaustive (self)                                          3.73    16_168.30    16_172.03       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_305.07       494.54     1_799.61       0.3144    41.590284         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_305.07       592.97     1_898.03       0.6776     1.428163         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_305.07       648.97     1_954.03       0.8321     0.534643         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_305.07       781.12     2_086.19       0.9366     0.161174         3.46
ExhaustiveRaBitQ (self)                                1_305.07     6_608.70     7_913.77       0.8324     0.527768         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        854.92       115.23       970.15       0.3195    41.573323         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        854.92       137.89       992.81       0.3185    41.580439         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        854.92       191.43     1_046.35       0.3175    41.584033         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        854.92       166.35     1_021.27       0.6876     1.444602         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       854.92       215.63     1_070.55       0.8368     0.555261         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       854.92       295.96     1_150.88       0.9353     0.178942         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        854.92       193.08     1_048.00       0.6874     1.447090         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       854.92       242.43     1_097.35       0.8389     0.545002         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       854.92       334.27     1_189.19       0.9398     0.160040         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        854.92       248.18     1_103.10       0.6858     1.457594         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       854.92       303.19     1_158.11       0.8386     0.547127         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       854.92       408.66     1_263.58       0.9408     0.158475         3.47
IVF-RaBitQ-nl273 (self)                                  854.92     3_967.98     4_822.90       0.9410     0.156568         3.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_162.08       121.18     1_283.26       0.3214    41.563286         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_162.08       164.65     1_326.73       0.3190    41.572722         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_162.08       171.96     1_334.04       0.6913     1.395216         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_162.08       220.21     1_382.29       0.8430     0.517782         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_162.08       301.82     1_463.90       0.9401     0.165874         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_162.08       224.69     1_386.77       0.6884     1.418525         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_162.08       271.15     1_433.23       0.8426     0.516410         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_162.08       358.09     1_520.17       0.9429     0.146813         3.49
IVF-RaBitQ-nl387 (self)                                1_162.08     3_589.14     4_751.22       0.9430     0.146811         3.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_603.47       110.44     1_713.92       0.3271    41.546231         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_603.47       127.62     1_731.09       0.3249    41.554688         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_603.47       151.76     1_755.23       0.3234    41.560268         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_603.47       161.34     1_764.81       0.6964     1.352767         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_603.47       205.98     1_809.45       0.8458     0.508759         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_603.47       288.46     1_891.93       0.9405     0.166157         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_603.47       176.44     1_779.91       0.6948     1.361829         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_603.47       235.22     1_838.69       0.8463     0.500727         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_603.47       307.59     1_911.06       0.9444     0.146654         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_603.47       203.59     1_807.06       0.6928     1.379005         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_603.47       251.94     1_855.41       0.8451     0.505865         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_603.47       339.16     1_942.63       0.9452     0.141182         3.51
IVF-RaBitQ-nl547 (self)                                1_603.47     3_405.42     5_008.90       0.9454     0.139824         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.75     1_565.53     1_569.28       1.0000     0.000000        18.88
Exhaustive (self)                                          3.75    16_441.54    16_445.29       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_291.33       438.05     1_729.39       0.3169     0.168415         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_291.33       522.25     1_813.58       0.6840     0.001052         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_291.33       589.89     1_881.22       0.8364     0.000396         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_291.33       706.54     1_997.88       0.9404     0.000112         3.46
ExhaustiveRaBitQ (self)                                1_291.33     5_880.05     7_171.38       0.8380     0.000388         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        842.42       116.27       958.69       0.3168     0.167977         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        842.42       139.06       981.48       0.3153     0.167681         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        842.42       198.07     1_040.50       0.3142     0.167536         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        842.42       169.09     1_011.51       0.6846     0.001130         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       842.42       213.36     1_055.78       0.8364     0.000437         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       842.42       295.66     1_138.08       0.9351     0.000141         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        842.42       209.90     1_052.32       0.6842     0.001136         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       842.42       243.27     1_085.69       0.8383     0.000429         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       842.42       331.33     1_173.75       0.9395     0.000128         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        842.42       251.97     1_094.40       0.6826     0.001146         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       842.42       307.31     1_149.73       0.8379     0.000432         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       842.42       417.16     1_259.59       0.9406     0.000126         3.47
IVF-RaBitQ-nl273 (self)                                  842.42     4_033.22     4_875.64       0.9412     0.000125         3.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_140.67       127.97     1_268.64       0.3200     0.168476         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_140.67       169.98     1_310.65       0.3178     0.168112         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_140.67       173.84     1_314.51       0.6890     0.001097         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_140.67       219.39     1_360.06       0.8411     0.000416         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_140.67       311.47     1_452.15       0.9397     0.000129         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_140.67       221.34     1_362.01       0.6862     0.001116         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_140.67       271.82     1_412.49       0.8402     0.000420         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_140.67       359.75     1_500.42       0.9420     0.000120         3.49
IVF-RaBitQ-nl387 (self)                                1_140.67     3_593.27     4_733.94       0.9431     0.000118         3.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_544.06       117.23     1_661.29       0.3235     0.169070         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_544.06       127.15     1_671.21       0.3212     0.168758         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_544.06       150.78     1_694.84       0.3196     0.168559         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_544.06       164.58     1_708.65       0.6951     0.001062         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_544.06       204.88     1_748.94       0.8449     0.000403         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_544.06       285.05     1_829.12       0.9403     0.000129         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_544.06       176.59     1_720.66       0.6927     0.001073         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_544.06       226.01     1_770.08       0.8450     0.000402         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_544.06       307.51     1_851.58       0.9434     0.000118         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_544.06       203.33     1_747.40       0.6904     0.001088         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_544.06       252.41     1_796.47       0.8441     0.000406         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_544.06       338.85     1_882.91       0.9442     0.000116         3.51
IVF-RaBitQ-nl547 (self)                                1_544.06     3_393.95     4_938.01       0.9453     0.000112         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         2.96     1_529.34     1_532.30       1.0000     0.000000        18.31
Exhaustive (self)                                          2.96    15_872.84    15_875.79       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_292.92       386.08     1_678.99       0.4436     1.402695         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_292.92       464.12     1_757.03       0.8785     0.030801         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_292.92       527.20     1_820.11       0.9683     0.005703         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_292.92       636.27     1_929.19       0.9954     0.000689         3.46
ExhaustiveRaBitQ (self)                                1_292.92     5_327.88     6_620.80       0.9697     0.005504         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        846.45       115.73       962.18       0.4623     1.393377         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        846.45       141.73       988.18       0.4621     1.393535         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        846.45       195.89     1_042.33       0.4619     1.393594         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        846.45       162.67     1_009.12       0.8930     0.027876         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       846.45       205.48     1_051.92       0.9748     0.004928         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       846.45       278.58     1_125.03       0.9967     0.000541         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        846.45       192.35     1_038.80       0.8928     0.027953         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       846.45       240.82     1_087.26       0.9747     0.004953         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       846.45       331.84     1_178.28       0.9967     0.000530         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        846.45       264.55     1_110.99       0.8926     0.028029         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       846.45       311.35     1_157.79       0.9746     0.004969         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       846.45       398.43     1_244.88       0.9967     0.000530         3.47
IVF-RaBitQ-nl273 (self)                                  846.45     4_027.71     4_874.16       0.9972     0.000459         3.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_157.19       119.00     1_276.19       0.4728     1.382032         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_157.19       173.22     1_330.41       0.4727     1.382090         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_157.19       169.30     1_326.49       0.9029     0.024285         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_157.19       214.13     1_371.32       0.9780     0.004021         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_157.19       287.50     1_444.69       0.9976     0.000377         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_157.19       218.89     1_376.08       0.9027     0.024327         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_157.19       267.38     1_424.57       0.9779     0.004030         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_157.19       355.34     1_512.54       0.9976     0.000374         3.49
IVF-RaBitQ-nl387 (self)                                1_157.19     3_487.56     4_644.75       0.9978     0.000353         3.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_603.63       110.34     1_713.98       0.4827     1.373328         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_603.63       136.30     1_739.93       0.4825     1.373426         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_603.63       151.77     1_755.40       0.4824     1.373469         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_603.63       160.44     1_764.07       0.9112     0.021537         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_603.63       200.50     1_804.14       0.9814     0.003302         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_603.63       280.61     1_884.25       0.9981     0.000292         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_603.63       176.07     1_779.70       0.9109     0.021633         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_603.63       222.11     1_825.75       0.9813     0.003336         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_603.63       295.77     1_899.40       0.9981     0.000296         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_603.63       211.85     1_815.48       0.9107     0.021672         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_603.63       248.39     1_852.02       0.9813     0.003336         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_603.63       329.01     1_932.64       0.9981     0.000292         3.51
IVF-RaBitQ-nl547 (self)                                1_603.63     3_315.35     4_918.98       0.9982     0.000273         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.13     1_524.26     1_527.39       1.0000     0.000000        18.31
Exhaustive (self)                                          3.13    15_868.61    15_871.75       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_284.40       407.44     1_691.84       0.4785     7.157863         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_284.40       485.15     1_769.55       0.9138     0.090004         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_284.40       539.03     1_823.43       0.9834     0.012928         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_284.40       653.86     1_938.26       0.9984     0.001130         3.46
ExhaustiveRaBitQ (self)                                1_284.40     5_413.57     6_697.97       0.9836     0.012962         3.46
IVF-RaBitQ-nl273-np13-rf0 (query)                        856.24       110.68       966.93       0.4889     7.133483         3.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np16-rf0 (query)                        856.24       135.93       992.17       0.4888     7.133529         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        856.24       184.83     1_041.08       0.4888     7.133535         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        856.24       160.19     1_016.43       0.9213     0.078670         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       856.24       203.55     1_059.80       0.9859     0.010581         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       856.24       275.45     1_131.69       0.9988     0.000820         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        856.24       204.06     1_060.30       0.9212     0.078723         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       856.24       238.27     1_094.51       0.9859     0.010572         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       856.24       317.09     1_173.33       0.9988     0.000807         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        856.24       241.09     1_097.33       0.9212     0.078726         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       856.24       295.63     1_151.87       0.9859     0.010574         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       856.24       380.79     1_237.03       0.9988     0.000807         3.47
IVF-RaBitQ-nl273 (self)                                  856.24     3_820.57     4_676.81       0.9989     0.000730         3.47
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_154.29       117.93     1_272.22       0.5001     7.113013         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_154.29       162.18     1_316.47       0.5001     7.113015         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_154.29       169.20     1_323.49       0.9291     0.067027         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_154.29       213.01     1_367.30       0.9875     0.009238         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_154.29       286.25     1_440.54       0.9992     0.000493         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_154.29       217.15     1_371.44       0.9291     0.067041         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_154.29       263.42     1_417.70       0.9875     0.009244         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_154.29       354.78     1_509.07       0.9992     0.000493         3.49
IVF-RaBitQ-nl387 (self)                                1_154.29     3_428.05     4_582.34       0.9991     0.000590         3.49
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_586.38       110.42     1_696.80       0.5119     7.089816         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_586.38       127.13     1_713.51       0.5119     7.089833         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_586.38       161.08     1_747.46       0.5119     7.089837         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_586.38       170.28     1_756.66       0.9356     0.059028         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_586.38       202.04     1_788.42       0.9895     0.007504         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_586.38       275.96     1_862.33       0.9993     0.000410         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_586.38       176.08     1_762.46       0.9355     0.059066         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_586.38       221.30     1_807.68       0.9895     0.007519         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_586.38       293.97     1_880.35       0.9993     0.000415         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_586.38       204.16     1_790.54       0.9355     0.059062         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_586.38       252.53     1_838.91       0.9895     0.007516         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_586.38       328.12     1_914.49       0.9993     0.000412         3.51
IVF-RaBitQ-nl547 (self)                                1_586.38     3_258.39     4_844.77       0.9993     0.000391         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With more dimensions

RaBitQ particularly shines with higher dimensionality in the data. The
interesting property of RaBitQ is that its own query mechanism becomes
better and better, the higher the dimensionality of the data.

<details>
<summary><b>RaBitQ - Euclidean (LowRank - 128 dimensions)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.03     5_894.30     5_908.33       1.0000     0.000000        73.24
Exhaustive (self)                                         14.03    59_997.84    60_011.87       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           2_793.51       856.46     3_649.97       0.7023    73.762204         5.31
ExhaustiveRaBitQ-rf5 (query)                           2_793.51       930.40     3_723.90       0.9950     0.019100         5.31
ExhaustiveRaBitQ-rf10 (query)                          2_793.51     1_005.24     3_798.75       0.9999     0.000396         5.31
ExhaustiveRaBitQ-rf20 (query)                          2_793.51     1_125.41     3_918.92       1.0000     0.000000         5.31
ExhaustiveRaBitQ (self)                                2_793.51    10_052.31    12_845.82       0.9999     0.000444         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      1_090.96       248.59     1_339.56       0.7067    73.750627         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      1_090.96       287.71     1_378.68       0.7067    73.750827         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      1_090.96       399.89     1_490.86       0.7067    73.750840         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      1_090.96       307.08     1_398.04       0.9953     0.016873         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     1_090.96       345.75     1_436.71       0.9996     0.002258         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     1_090.96       436.94     1_527.91       0.9997     0.002042         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      1_090.96       349.37     1_440.33       0.9956     0.014900         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     1_090.96       408.08     1_499.05       0.9999     0.000269         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     1_090.96       502.80     1_593.77       1.0000     0.000053         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      1_090.96       460.72     1_551.69       0.9956     0.014855         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     1_090.96       521.67     1_612.63       0.9999     0.000216         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     1_090.96       621.94     1_712.90       1.0000     0.000000         5.35
IVF-RaBitQ-nl273 (self)                                1_090.96     6_218.64     7_309.61       1.0000     0.000000         5.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_443.14       280.76     1_723.90       0.7134    73.741226         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_443.14       375.00     1_818.14       0.7133    73.741243         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_443.14       337.61     1_780.75       0.9960     0.013943         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_443.14       393.88     1_837.01       0.9999     0.000359         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_443.14       484.98     1_928.12       1.0000     0.000192         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_443.14       441.40     1_884.54       0.9961     0.013746         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_443.14       497.75     1_940.89       0.9999     0.000167         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_443.14       594.96     2_038.10       1.0000     0.000000         5.41
IVF-RaBitQ-nl387 (self)                                1_443.14     5_906.09     7_349.23       1.0000     0.000004         5.41
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_917.53       271.86     2_189.38       0.7169    73.731426         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_917.53       316.42     2_233.95       0.7169    73.731517         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_917.53       401.25     2_318.78       0.7169    73.731534         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_917.53       355.42     2_272.94       0.9962     0.012944         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_917.53       376.81     2_294.34       0.9998     0.000864         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_917.53       465.34     2_382.87       0.9999     0.000586         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_917.53       368.37     2_285.89       0.9963     0.012548         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_917.53       421.00     2_338.53       0.9999     0.000390         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_917.53       528.32     2_445.84       1.0000     0.000112         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_917.53       431.89     2_349.41       0.9964     0.012439         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_917.53       487.44     2_404.96       0.9999     0.000281         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_917.53       583.00     2_500.53       1.0000     0.000003         5.49
IVF-RaBitQ-nl547 (self)                                1_917.53     5_786.43     7_703.96       1.0000     0.000002         5.49
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>


<details>
<summary><b>RaBitQ - Euclidean (LowRank - 256 dimensions)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 256D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        28.81    14_896.25    14_925.06       1.0000     0.000000       146.48
Exhaustive (self)                                         28.81   150_809.13   150_837.94       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           6_779.01     1_820.35     8_599.35       0.7812   172.768307         7.88
ExhaustiveRaBitQ-rf5 (query)                           6_779.01     1_911.38     8_690.38       0.9995     0.002595         7.88
ExhaustiveRaBitQ-rf10 (query)                          6_779.01     1_975.61     8_754.62       1.0000     0.000000         7.88
ExhaustiveRaBitQ-rf20 (query)                          6_779.01     2_128.76     8_907.77       1.0000     0.000000         7.88
ExhaustiveRaBitQ (self)                                6_779.01    19_753.59    26_532.59       1.0000     0.000002         7.88
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      2_884.30       537.36     3_421.66       0.7838   172.759184         7.96
IVF-RaBitQ-nl273-np16-rf0 (query)                      2_884.30       657.38     3_541.68       0.7839   172.759408         7.96
IVF-RaBitQ-nl273-np23-rf0 (query)                      2_884.30       912.78     3_797.07       0.7839   172.759416         7.96
IVF-RaBitQ-nl273-np13-rf5 (query)                      2_884.30       602.08     3_486.38       0.9994     0.005565         7.96
IVF-RaBitQ-nl273-np13-rf10 (query)                     2_884.30       656.60     3_540.90       0.9997     0.004006         7.96
IVF-RaBitQ-nl273-np13-rf20 (query)                     2_884.30       775.56     3_659.86       0.9997     0.004006         7.96
IVF-RaBitQ-nl273-np16-rf5 (query)                      2_884.30       716.37     3_600.67       0.9997     0.001772         7.96
IVF-RaBitQ-nl273-np16-rf10 (query)                     2_884.30       784.13     3_668.42       1.0000     0.000213         7.96
IVF-RaBitQ-nl273-np16-rf20 (query)                     2_884.30       894.45     3_778.74       1.0000     0.000213         7.96
IVF-RaBitQ-nl273-np23-rf5 (query)                      2_884.30       993.49     3_877.79       0.9997     0.001559         7.96
IVF-RaBitQ-nl273-np23-rf10 (query)                     2_884.30     1_132.97     4_017.26       1.0000     0.000000         7.96
IVF-RaBitQ-nl273-np23-rf20 (query)                     2_884.30     1_166.69     4_050.98       1.0000     0.000000         7.96
IVF-RaBitQ-nl273 (self)                                2_884.30    11_656.62    14_540.92       1.0000     0.000000         7.96
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      3_315.64       662.77     3_978.41       0.7874   172.749734         8.07
IVF-RaBitQ-nl387-np27-rf0 (query)                      3_315.64       927.71     4_243.35       0.7874   172.749752         8.07
IVF-RaBitQ-nl387-np19-rf5 (query)                      3_315.64       732.28     4_047.92       0.9996     0.002304         8.07
IVF-RaBitQ-nl387-np19-rf10 (query)                     3_315.64       791.88     4_107.53       1.0000     0.000365         8.07
IVF-RaBitQ-nl387-np19-rf20 (query)                     3_315.64       902.98     4_218.62       1.0000     0.000365         8.07
IVF-RaBitQ-nl387-np27-rf5 (query)                      3_315.64       994.95     4_310.60       0.9997     0.001966         8.07
IVF-RaBitQ-nl387-np27-rf10 (query)                     3_315.64     1_061.21     4_376.86       1.0000     0.000000         8.07
IVF-RaBitQ-nl387-np27-rf20 (query)                     3_315.64     1_179.23     4_494.87       1.0000     0.000000         8.07
IVF-RaBitQ-nl387 (self)                                3_315.64    11_776.55    15_092.20       1.0000     0.000000         8.07
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                      4_188.88       717.04     4_905.92       0.7904   172.743150         8.23
IVF-RaBitQ-nl547-np27-rf0 (query)                      4_188.88       832.38     5_021.26       0.7904   172.743304         8.23
IVF-RaBitQ-nl547-np33-rf0 (query)                      4_188.88       998.04     5_186.93       0.7904   172.743314         8.23
IVF-RaBitQ-nl547-np23-rf5 (query)                      4_188.88       807.59     4_996.47       0.9995     0.004037         8.23
IVF-RaBitQ-nl547-np23-rf10 (query)                     4_188.88       836.60     5_025.49       0.9998     0.002925         8.23
IVF-RaBitQ-nl547-np23-rf20 (query)                     4_188.88       955.47     5_144.36       0.9998     0.002925         8.23
IVF-RaBitQ-nl547-np27-rf5 (query)                      4_188.88       912.33     5_101.22       0.9997     0.001330         8.23
IVF-RaBitQ-nl547-np27-rf10 (query)                     4_188.88       966.74     5_155.62       1.0000     0.000231         8.23
IVF-RaBitQ-nl547-np27-rf20 (query)                     4_188.88     1_069.72     5_258.61       1.0000     0.000231         8.23
IVF-RaBitQ-nl547-np33-rf5 (query)                      4_188.88     1_073.98     5_262.87       0.9998     0.001099         8.23
IVF-RaBitQ-nl547-np33-rf10 (query)                     4_188.88     1_141.32     5_330.21       1.0000     0.000000         8.23
IVF-RaBitQ-nl547-np33-rf20 (query)                     4_188.88     1_255.74     5_444.62       1.0000     0.000000         8.23
IVF-RaBitQ-nl547 (self)                                4_188.88    12_556.71    16_745.59       1.0000     0.000000         8.23
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

<details>
<summary><b>RaBitQ - Euclidean (LowRank - 512 dimensions)</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 512D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        60.66    32_462.03    32_522.69       1.0000     0.000000       292.97
Exhaustive (self)                                         60.66   327_346.00   327_406.66       1.0000     0.000000       292.97
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                          17_292.61     4_810.76    22_103.37       0.8405   370.189430        13.40
ExhaustiveRaBitQ-rf5 (query)                          17_292.61     4_910.64    22_203.25       0.9994     0.084579        13.40
ExhaustiveRaBitQ-rf10 (query)                         17_292.61     4_968.70    22_261.30       0.9995     0.052237        13.40
ExhaustiveRaBitQ-rf20 (query)                         17_292.61     5_132.64    22_425.25       0.9996     0.031282        13.40
ExhaustiveRaBitQ (self)                               17_292.61    49_652.31    66_944.92       0.9995     0.059264        13.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      8_819.66     1_560.97    10_380.63       0.8406   370.186223        13.55
IVF-RaBitQ-nl273-np16-rf0 (query)                      8_819.66     1_872.32    10_691.98       0.8408   370.186601        13.55
IVF-RaBitQ-nl273-np23-rf0 (query)                      8_819.66     2_657.77    11_477.43       0.8408   370.186621        13.55
IVF-RaBitQ-nl273-np13-rf5 (query)                      8_819.66     1_609.84    10_429.49       0.9991     0.090643        13.55
IVF-RaBitQ-nl273-np13-rf10 (query)                     8_819.66     1_694.91    10_514.57       0.9992     0.059367        13.55
IVF-RaBitQ-nl273-np13-rf20 (query)                     8_819.66     1_811.06    10_630.71       0.9993     0.046711        13.55
IVF-RaBitQ-nl273-np16-rf5 (query)                      8_819.66     1_949.34    10_769.00       0.9995     0.075727        13.55
IVF-RaBitQ-nl273-np16-rf10 (query)                     8_819.66     2_026.21    10_845.87       0.9996     0.044451        13.55
IVF-RaBitQ-nl273-np16-rf20 (query)                     8_819.66     2_169.46    10_989.12       0.9996     0.031795        13.55
IVF-RaBitQ-nl273-np23-rf5 (query)                      8_819.66     2_736.52    11_556.18       0.9995     0.075348        13.55
IVF-RaBitQ-nl273-np23-rf10 (query)                     8_819.66     2_826.38    11_646.04       0.9996     0.044072        13.55
IVF-RaBitQ-nl273-np23-rf20 (query)                     8_819.66     2_986.04    11_805.69       0.9996     0.031417        13.55
IVF-RaBitQ-nl273 (self)                                8_819.66    29_695.70    38_515.36       0.9996     0.034065        13.55
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl387-np19-rf0 (query)                      9_632.05     2_035.32    11_667.37       0.8444   370.183354        13.78
IVF-RaBitQ-nl387-np27-rf0 (query)                      9_632.05     3_028.45    12_660.50       0.8444   370.183376        13.78
IVF-RaBitQ-nl387-np19-rf5 (query)                      9_632.05     2_098.09    11_730.14       0.9995     0.060023        13.78
IVF-RaBitQ-nl387-np19-rf10 (query)                     9_632.05     2_174.22    11_806.27       0.9995     0.050791        13.78
IVF-RaBitQ-nl387-np19-rf20 (query)                     9_632.05     2_326.69    11_958.74       0.9996     0.034409        13.78
IVF-RaBitQ-nl387-np27-rf5 (query)                      9_632.05     2_939.82    12_571.87       0.9995     0.058974        13.78
IVF-RaBitQ-nl387-np27-rf10 (query)                     9_632.05     2_996.05    12_628.10       0.9996     0.049741        13.78
IVF-RaBitQ-nl387-np27-rf20 (query)                     9_632.05     3_192.48    12_824.53       0.9996     0.033360        13.78
IVF-RaBitQ-nl387 (self)                                9_632.05    31_825.19    41_457.24       0.9996     0.026435        13.78
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl547-np23-rf0 (query)                     11_011.65     2_280.29    13_291.94       0.8468   370.175057        14.09
IVF-RaBitQ-nl547-np27-rf0 (query)                     11_011.65     2_656.68    13_668.33       0.8468   370.175243        14.09
IVF-RaBitQ-nl547-np33-rf0 (query)                     11_011.65     3_258.48    14_270.12       0.8468   370.175269        14.09
IVF-RaBitQ-nl547-np23-rf5 (query)                     11_011.65     2_317.28    13_328.93       0.9993     0.044230        14.09
IVF-RaBitQ-nl547-np23-rf10 (query)                    11_011.65     2_368.10    13_379.75       0.9994     0.040785        14.09
IVF-RaBitQ-nl547-np23-rf20 (query)                    11_011.65     2_522.87    13_534.52       0.9994     0.029099        14.09
IVF-RaBitQ-nl547-np27-rf5 (query)                     11_011.65     2_670.34    13_681.98       0.9996     0.038662        14.09
IVF-RaBitQ-nl547-np27-rf10 (query)                    11_011.65     2_772.34    13_783.99       0.9996     0.035217        14.09
IVF-RaBitQ-nl547-np27-rf20 (query)                    11_011.65     2_908.61    13_920.26       0.9997     0.023531        14.09
IVF-RaBitQ-nl547-np33-rf5 (query)                     11_011.65     3_266.69    14_278.34       0.9996     0.038161        14.09
IVF-RaBitQ-nl547-np33-rf10 (query)                    11_011.65     3_313.36    14_325.01       0.9996     0.034715        14.09
IVF-RaBitQ-nl547-np33-rf20 (query)                    11_011.65     3_425.85    14_437.50       0.9997     0.023029        14.09
IVF-RaBitQ-nl547 (self)                               11_011.65    34_320.38    45_332.03       0.9996     0.025413        14.09
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*Last update: 2026/03/15 with version **0.2.5***
