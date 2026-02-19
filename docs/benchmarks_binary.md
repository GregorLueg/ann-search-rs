## Binarised indices benchmarks and parameter

Binarised indices compress the data stored in the index structure itself via
very aggressive quantisation to basically only bits. This has two impacts:

1. Drastic reduction in memory usage.
2. Increased query speed in most cases as the bit-wise operations are very
fast.
3. However, when not using any re-ranking of the top candidates, dramatically
lower recall (less so for RaBitQ).

These indices can be nonetheless quite useful in memory-constrained scenarios.
In both cases, there is an option to do re-ranking against the original vectors.
However, these are stored on disk and accessed via rapid access. The benchmarks
below show scenarios with and without re-ranking.

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

These indices uses binarisation via either SimHash or iterative quantisation
via PCA rotations. They both have the option to use a VecStore that saves the
original data on disk for fast retrieval and re-ranking. This is recommended if
you wish to maintain some Recall. Generally speaking, these indices shine in
very high-dimensional data where memory requirements becomes very constraining.

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
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.33     1_520.29     1_523.61       1.0000     0.000000        18.31
Exhaustive (self)                                          3.33    16_485.28    16_488.61       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_167.41       491.18     1_658.59       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_167.41       706.21     1_873.62       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_167.41       604.53     1_771.94       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_167.41       712.35     1_879.76       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                     1_167.41     5_965.36     7_132.77       0.5461     2.056589         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_364.83       495.93     1_860.76       0.1873          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_364.83       554.48     1_919.31       0.3844     4.171681         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_364.83       585.30     1_950.13       0.5073     2.366457         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_364.83       686.39     2_051.22       0.6439     1.245957         4.61
ExhaustiveBinary-256-itq (self)                        1_364.83     5_851.36     7_216.19       0.5079     2.339769         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_307.21       839.02     3_146.22       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_307.21       886.79     3_193.99       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_307.21       928.33     3_235.54       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_307.21     1_055.87     3_363.08       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                     2_307.21     9_560.05    11_867.25       0.7223     0.828915         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_587.57       830.10     3_417.67       0.2823          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_587.57       876.10     3_463.67       0.5675     1.840282         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_587.57       929.45     3_517.02       0.7062     0.870974         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_587.57     1_039.16     3_626.73       0.8300     0.361878         9.22
ExhaustiveBinary-512-itq (self)                        2_587.57     9_326.51    11_914.08       0.7065     0.858252         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 162.64       614.27       776.91       0.0578          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   162.64       647.82       810.46       0.1270    15.329844         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  162.64       696.32       858.97       0.1824    10.683378         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  162.64       762.76       925.40       0.2652     7.182754         0.58
ExhaustiveBinary-32-signed (self)                        162.64     6_859.85     7_022.49       0.1841    10.412094         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_076.45        54.59     2_131.05       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_076.45        60.12     2_136.57       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_076.45        75.63     2_152.08       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_076.45        82.59     2_159.05       0.4239     3.637415         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_076.45       109.52     2_185.97       0.5531     2.021145         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_076.45       156.32     2_232.77       0.6927     1.020296         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_076.45        91.26     2_167.71       0.4223     3.662012         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_076.45       119.67     2_196.12       0.5515     2.033363         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_076.45       169.63     2_246.08       0.6913     1.025118         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_076.45       112.51     2_188.97       0.4198     3.692295         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_076.45       145.71     2_222.17       0.5484     2.057241         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_076.45       192.26     2_268.71       0.6879     1.040613         5.79
IVF-Binary-256-nl273-random (self)                     2_076.45     1_207.02     3_283.47       0.5511     2.018819         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_366.50        56.30     2_422.80       0.2061          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_366.50        67.39     2_433.90       0.2047          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_366.50        84.07     2_450.58       0.4238     3.634205         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_366.50       111.35     2_477.85       0.5532     2.014085         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_366.50       157.33     2_523.83       0.6934     1.014780         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_366.50        97.80     2_464.30       0.4203     3.683698         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_366.50       127.52     2_494.02       0.5495     2.045078         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_366.50       176.71     2_543.22       0.6888     1.036117         5.80
IVF-Binary-256-nl387-random (self)                     2_366.50     1_107.03     3_473.53       0.5531     1.998615         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_890.98        52.86     2_943.84       0.2076          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_890.98        56.37     2_947.35       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_890.98        62.69     2_953.67       0.2050          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_890.98        78.82     2_969.80       0.4265     3.596401         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_890.98       104.35     2_995.33       0.5576     1.983148         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_890.98       149.33     3_040.31       0.6977     0.994532         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_890.98        82.91     2_973.90       0.4238     3.631225         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_890.98       110.09     3_001.07       0.5539     2.008651         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_890.98       157.15     3_048.13       0.6941     1.007218         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_890.98        92.38     2_983.36       0.4211     3.673312         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_890.98       122.36     3_013.34       0.5506     2.035196         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_890.98       175.08     3_066.06       0.6900     1.029156         5.82
IVF-Binary-256-nl547-random (self)                     2_890.98     1_041.83     3_932.82       0.5570     1.967408         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_223.50        52.64     2_276.15       0.1904          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_223.50        58.84     2_282.35       0.1893          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_223.50        73.98     2_297.49       0.1883          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_223.50        82.05     2_305.55       0.3916     4.060777         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_223.50       110.02     2_333.52       0.5151     2.302263         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_223.50       158.84     2_382.34       0.6522     1.207268         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_223.50        89.24     2_312.75       0.3890     4.101804         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_223.50       118.72     2_342.23       0.5128     2.322152         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_223.50       168.47     2_391.97       0.6499     1.216464         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_223.50       106.37     2_329.87       0.3864     4.140148         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_223.50       139.26     2_362.76       0.5094     2.347524         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_223.50       190.43     2_413.94       0.6461     1.235485         5.79
IVF-Binary-256-nl273-itq (self)                        2_223.50     1_184.34     3_407.85       0.5137     2.293966         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_553.60        55.64     2_609.23       0.1903          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_553.60        66.14     2_619.73       0.1888          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_553.60        83.68     2_637.28       0.3911     4.060723         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_553.60       109.86     2_663.46       0.5150     2.299160         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_553.60       155.13     2_708.72       0.6522     1.205072         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_553.60        95.11     2_648.71       0.3872     4.125703         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_553.60       124.06     2_677.66       0.5105     2.336554         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_553.60       173.11     2_726.70       0.6473     1.227071         5.80
IVF-Binary-256-nl387-itq (self)                        2_553.60     1_090.85     3_644.44       0.5159     2.270944         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_068.10        55.03     3_123.13       0.1915          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_068.10        57.80     3_125.89       0.1904          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_068.10        64.45     3_132.55       0.1892          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_068.10        80.51     3_148.61       0.3943     4.017566         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_068.10       106.57     3_174.66       0.5192     2.262953         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_068.10       151.25     3_219.35       0.6571     1.179721         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_068.10        85.90     3_154.00       0.3912     4.062240         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_068.10       112.94     3_181.04       0.5152     2.293847         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_068.10       157.47     3_225.57       0.6530     1.195484         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_068.10        93.28     3_161.37       0.3883     4.110804         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_068.10       120.38     3_188.48       0.5114     2.329036         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_068.10       169.11     3_237.21       0.6489     1.218997         5.82
IVF-Binary-256-nl547-itq (self)                        3_068.10     1_064.48     4_132.58       0.5201     2.235387         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_175.09        98.95     3_274.04       0.2929          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_175.09       112.64     3_287.72       0.2924          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_175.09       143.60     3_318.68       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_175.09       132.47     3_307.55       0.5865     1.777894        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_175.09       165.03     3_340.12       0.7255     0.834278        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_175.09       218.38     3_393.47       0.8433     0.358489        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_175.09       148.47     3_323.56       0.5863     1.774378        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_175.09       181.79     3_356.88       0.7262     0.823265        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_175.09       239.23     3_414.32       0.8456     0.341312        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_175.09       181.27     3_356.36       0.5849     1.784210        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_175.09       217.48     3_392.56       0.7249     0.828604        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_175.09       279.77     3_454.86       0.8447     0.342678        10.40
IVF-Binary-512-nl273-random (self)                     3_175.09     1_807.08     4_982.16       0.7247     0.820119        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_510.97       103.05     3_614.02       0.2925          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_510.97       130.52     3_641.48       0.2915          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_510.97       135.99     3_646.96       0.5871     1.768931        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_510.97       167.39     3_678.36       0.7264     0.828199        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_510.97       221.02     3_731.99       0.8449     0.354058        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_510.97       163.33     3_674.30       0.5851     1.780633        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_510.97       197.00     3_707.97       0.7251     0.826723        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_510.97       255.37     3_766.34       0.8452     0.341036        10.41
IVF-Binary-512-nl387-random (self)                     3_510.97     1_781.41     5_292.38       0.7248     0.827215        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           3_996.79        95.96     4_092.75       0.2936          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           3_996.79       103.61     4_100.40       0.2929          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           3_996.79       117.03     4_113.82       0.2921          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           3_996.79       127.85     4_124.64       0.5886     1.763517        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          3_996.79       158.18     4_154.97       0.7281     0.824131        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          3_996.79       208.20     4_204.99       0.8461     0.353593        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           3_996.79       137.31     4_134.10       0.5871     1.766939        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          3_996.79       169.47     4_166.26       0.7271     0.820926        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          3_996.79       222.32     4_219.11       0.8464     0.341065        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           3_996.79       151.36     4_148.15       0.5855     1.778571        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          3_996.79       195.08     4_191.87       0.7259     0.824482        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          3_996.79       253.18     4_249.97       0.8456     0.338961        10.43
IVF-Binary-512-nl547-random (self)                     3_996.79     1_664.97     5_661.76       0.7268     0.818374        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_503.71       102.50     3_606.22       0.2844          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_503.71       116.83     3_620.55       0.2840          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_503.71       147.14     3_650.85       0.2830          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_503.71       152.09     3_655.80       0.5702     1.829008        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_503.71       174.32     3_678.03       0.7080     0.874265        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_503.71       219.27     3_722.99       0.8300     0.375344        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_503.71       152.80     3_656.52       0.5701     1.823803        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_503.71       183.73     3_687.45       0.7087     0.861913        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_503.71       242.69     3_746.40       0.8321     0.358097        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_503.71       185.71     3_689.42       0.5686     1.832962        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_503.71       224.93     3_728.65       0.7071     0.866957        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_503.71       280.87     3_784.58       0.8307     0.359946        10.40
IVF-Binary-512-nl273-itq (self)                        3_503.71     1_831.98     5_335.70       0.7091     0.849525        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_820.22       106.24     3_926.46       0.2842          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_820.22       131.45     3_951.67       0.2832          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_820.22       144.67     3_964.89       0.5704     1.826286        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_820.22       169.04     3_989.26       0.7087     0.871520        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_820.22       222.09     4_042.31       0.8314     0.372721        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_820.22       166.30     3_986.52       0.5690     1.830102        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_820.22       199.98     4_020.20       0.7075     0.865294        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_820.22       257.30     4_077.52       0.8312     0.358956        10.41
IVF-Binary-512-nl387-itq (self)                        3_820.22     1_686.20     5_506.42       0.7092     0.858300        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_277.56        98.87     4_376.43       0.2851          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_277.56       109.20     4_386.76       0.2843          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_277.56       122.15     4_399.71       0.2835          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_277.56       134.38     4_411.95       0.5724     1.810842        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_277.56       166.53     4_444.09       0.7109     0.863964        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_277.56       221.19     4_498.76       0.8324     0.372531        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_277.56       141.08     4_418.65       0.5712     1.814069        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_277.56       174.77     4_452.34       0.7099     0.858586        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_277.56       228.59     4_506.16       0.8329     0.359564        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_277.56       158.99     4_436.55       0.5696     1.824876        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_277.56       190.59     4_468.15       0.7082     0.861870        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_277.56       244.90     4_522.47       0.8316     0.357369        10.43
IVF-Binary-512-nl547-itq (self)                        4_277.56     1_618.91     5_896.48       0.7111     0.850607        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_026.63        46.10     1_072.73       0.0678          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_026.63        52.62     1_079.25       0.0622          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_026.63        71.24     1_097.87       0.0598          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_026.63        63.99     1_090.62       0.1443    13.596398         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_026.63        82.57     1_109.20       0.2064     9.381052         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_026.63       117.31     1_143.94       0.2965     6.196962         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_026.63        77.36     1_104.00       0.1362    14.167816         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_026.63        92.93     1_119.57       0.1967     9.780414         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_026.63       128.55     1_155.18       0.2850     6.504469         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_026.63        92.16     1_118.79       0.1318    14.570114         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_026.63       114.03     1_140.67       0.1904    10.133050         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_026.63       153.28     1_179.91       0.2760     6.758148         1.76
IVF-Binary-32-nl273-signed (self)                      1_026.63       924.96     1_951.59       0.1988     9.537432         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_374.85        48.17     1_423.02       0.0647          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_374.85        61.54     1_436.39       0.0617          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_374.85        65.79     1_440.64       0.1411    13.649253         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_374.85        83.23     1_458.08       0.2036     9.376952         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_374.85       117.05     1_491.89       0.2932     6.166358         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_374.85        80.37     1_455.22       0.1351    14.226639         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_374.85       100.95     1_475.80       0.1945     9.827848         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_374.85       135.68     1_510.53       0.2809     6.537042         1.77
IVF-Binary-32-nl387-signed (self)                      1_374.85       843.94     2_218.79       0.2046     9.170273         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_905.52        47.17     1_952.69       0.0664          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_905.52        49.96     1_955.48       0.0644          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_905.52        57.07     1_962.59       0.0624          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_905.52        61.74     1_967.26       0.1456    13.304912         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_905.52        81.48     1_987.00       0.2094     9.098909         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_905.52       117.56     2_023.08       0.3026     5.932205         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_905.52        66.71     1_972.23       0.1413    13.664483         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_905.52        89.95     1_995.47       0.2031     9.410317         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_905.52       119.33     2_024.86       0.2934     6.188357         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_905.52        74.82     1_980.34       0.1369    14.047945         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_905.52        94.45     1_999.97       0.1971     9.719177         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_905.52       130.38     2_035.90       0.2843     6.447317         1.79
IVF-Binary-32-nl547-signed (self)                      1_905.52       785.76     2_691.28       0.2110     8.898960         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.22     1_598.39     1_602.61       1.0000     0.000000        18.88
Exhaustive (self)                                          4.22    16_424.03    16_428.25       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_166.45       494.86     1_661.31       0.2158          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_166.45       552.55     1_719.00       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_166.45       594.85     1_761.30       0.5705     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_166.45       692.96     1_859.41       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                     1_166.45     6_440.02     7_606.47       0.5696     0.001382         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_417.50       594.92     2_012.42       0.1983          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_417.50       594.44     2_011.94       0.4042     0.002874         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_417.50       602.04     2_019.54       0.5292     0.001626         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_417.50       707.36     2_124.86       0.6649     0.000853         4.61
ExhaustiveBinary-256-itq (self)                        1_417.50     5_896.33     7_313.84       0.5290     0.001611         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_313.53       823.56     3_137.08       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_313.53       866.32     3_179.84       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_313.53       918.22     3_231.74       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_313.53     1_027.45     3_340.98       0.8652     0.000203         9.22
ExhaustiveBinary-512-random (self)                     2_313.53     9_187.98    11_501.51       0.7519     0.000503         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_588.84       826.59     3_415.43       0.3035          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_588.84       876.96     3_465.79       0.5988     0.001175         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_588.84       921.15     3_509.99       0.7344     0.000548         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_588.84     1_038.07     3_626.91       0.8497     0.000226         9.22
ExhaustiveBinary-512-itq (self)                        2_588.84     9_225.27    11_814.11       0.7336     0.000546         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 162.86       609.32       772.18       0.0588          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   162.86       650.61       813.47       0.1297     0.011294         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  162.86       682.29       845.15       0.1862     0.007851         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  162.86       771.73       934.59       0.2706     0.005257         0.58
ExhaustiveBinary-32-signed (self)                        162.86     6_843.82     7_006.68       0.1885     0.007670         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_006.22        52.15     2_058.37       0.2192          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_006.22        59.28     2_065.49       0.2181          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_006.22        73.77     2_079.99       0.2170          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_006.22        80.60     2_086.81       0.4458     0.002466         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_006.22       108.53     2_114.75       0.5774     0.001352         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_006.22       155.49     2_161.71       0.7146     0.000675         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_006.22        88.77     2_094.99       0.4440     0.002486         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_006.22       118.52     2_124.74       0.5758     0.001359         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_006.22       167.88     2_174.10       0.7130     0.000678         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_006.22       106.60     2_112.82       0.4418     0.002508         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_006.22       138.31     2_144.53       0.5730     0.001375         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_006.22       191.98     2_198.19       0.7099     0.000688         5.79
IVF-Binary-256-nl273-random (self)                     2_006.22     1_184.62     3_190.83       0.5747     0.001354         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_322.67        57.50     2_380.16       0.2188          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)           2_322.67        71.07     2_393.74       0.2170          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)           2_322.67        82.26     2_404.93       0.4458     0.002464         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)          2_322.67       108.43     2_431.10       0.5772     0.001349         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)          2_322.67       155.13     2_477.79       0.7147     0.000672         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)           2_322.67        95.47     2_418.13       0.4424     0.002499         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)          2_322.67       125.52     2_448.19       0.5735     0.001369         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)          2_322.67       174.12     2_496.78       0.7105     0.000686         5.81
IVF-Binary-256-nl387-random (self)                     2_322.67     1_083.24     3_405.90       0.5767     0.001342         5.81
IVF-Binary-256-nl547-np23-rf0-random (query)           2_801.20        51.85     2_853.06       0.2201          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)           2_801.20        59.39     2_860.59       0.2185          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)           2_801.20        61.78     2_862.98       0.2175          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)           2_801.20        83.77     2_884.97       0.4488     0.002436         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)          2_801.20       108.00     2_909.20       0.5813     0.001327         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)          2_801.20       151.55     2_952.76       0.7190     0.000657         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)           2_801.20        85.45     2_886.65       0.4461     0.002462         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)          2_801.20       112.47     2_913.68       0.5778     0.001344         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)          2_801.20       159.06     2_960.27       0.7157     0.000666         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)           2_801.20        92.90     2_894.10       0.4433     0.002492         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)          2_801.20       121.78     2_922.99       0.5744     0.001365         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)          2_801.20       170.26     2_971.47       0.7120     0.000680         5.83
IVF-Binary-256-nl547-random (self)                     2_801.20     1_066.06     3_867.26       0.5805     0.001321         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_212.98        53.58     2_266.56       0.2016          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_212.98        59.18     2_272.16       0.2007          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_212.98        79.75     2_292.73       0.1995          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_212.98        81.67     2_294.65       0.4112     0.002797         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_212.98       112.75     2_325.74       0.5371     0.001579         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_212.98       155.39     2_368.38       0.6731     0.000826         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_212.98        89.60     2_302.58       0.4087     0.002824         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_212.98       120.07     2_333.05       0.5346     0.001592         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_212.98       168.62     2_381.61       0.6712     0.000831         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_212.98       108.81     2_321.79       0.4061     0.002852         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_212.98       144.36     2_357.34       0.5316     0.001611         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_212.98       191.83     2_404.82       0.6674     0.000844         5.79
IVF-Binary-256-nl273-itq (self)                        2_212.98     1_188.59     3_401.57       0.5347     0.001578         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_527.53        54.73     2_582.26       0.2017          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_527.53        65.79     2_593.32       0.1998          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_527.53        82.63     2_610.16       0.4114     0.002791         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_527.53       108.85     2_636.38       0.5375     0.001573         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_527.53       155.03     2_682.56       0.6732     0.000820         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_527.53       100.29     2_627.82       0.4074     0.002837         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_527.53       124.39     2_651.92       0.5324     0.001604         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_527.53       172.17     2_699.70       0.6681     0.000839         5.81
IVF-Binary-256-nl387-itq (self)                        2_527.53     1_082.04     3_609.57       0.5371     0.001561         5.81
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_011.85        51.91     3_063.76       0.2029          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_011.85        55.89     3_067.74       0.2015          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_011.85        61.94     3_073.80       0.2005          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_011.85        79.56     3_091.41       0.4147     0.002759         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_011.85       104.95     3_116.80       0.5413     0.001548         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_011.85       149.24     3_161.10       0.6777     0.000804         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_011.85        83.74     3_095.60       0.4114     0.002794         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_011.85       111.54     3_123.39       0.5372     0.001572         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_011.85       156.95     3_168.80       0.6739     0.000816         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_011.85        91.00     3_102.86       0.4086     0.002826         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_011.85       118.75     3_130.61       0.5334     0.001597         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_011.85       166.87     3_178.73       0.6693     0.000835         5.83
IVF-Binary-256-nl547-itq (self)                        3_011.85     1_042.12     4_053.97       0.5411     0.001537         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_152.16       109.62     3_261.78       0.3157          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_152.16       120.86     3_273.02       0.3151          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_152.16       157.63     3_309.80       0.3143          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_152.16       139.11     3_291.27       0.6204     0.001094        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_152.16       169.53     3_321.70       0.7550     0.000506        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_152.16       221.51     3_373.67       0.8638     0.000215        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_152.16       157.30     3_309.46       0.6204     0.001092        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_152.16       189.24     3_341.41       0.7559     0.000498        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_152.16       247.64     3_399.81       0.8663     0.000202        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_152.16       196.62     3_348.79       0.6189     0.001099        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_152.16       231.69     3_383.85       0.7547     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_152.16       296.74     3_448.90       0.8658     0.000201        10.40
IVF-Binary-512-nl273-random (self)                     3_152.16     1_882.38     5_034.54       0.7540     0.000497        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_486.19       108.07     3_594.26       0.3154          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_486.19       136.02     3_622.21       0.3146          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_486.19       142.24     3_628.43       0.6207     0.001091        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_486.19       173.67     3_659.86       0.7556     0.000503        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_486.19       234.99     3_721.17       0.8650     0.000210        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_486.19       183.66     3_669.85       0.6194     0.001096        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_486.19       215.05     3_701.24       0.7552     0.000499        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_486.19       263.80     3_749.98       0.8661     0.000201        10.41
IVF-Binary-512-nl387-random (self)                     3_486.19     1_730.08     5_216.26       0.7539     0.000502        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           3_975.22        99.38     4_074.61       0.3163          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)           3_975.22       109.85     4_085.07       0.3155          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)           3_975.22       124.99     4_100.21       0.3146          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)           3_975.22       135.60     4_110.82       0.6226     0.001082        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)          3_975.22       175.82     4_151.04       0.7571     0.000498        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)          3_975.22       214.93     4_190.16       0.8660     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)           3_975.22       143.95     4_119.18       0.6213     0.001086        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)          3_975.22       175.92     4_151.14       0.7570     0.000494        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)          3_975.22       228.87     4_204.09       0.8670     0.000201        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)           3_975.22       161.44     4_136.66       0.6196     0.001094        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)          3_975.22       193.42     4_168.64       0.7556     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)          3_975.22       248.79     4_224.01       0.8664     0.000200        10.44
IVF-Binary-512-nl547-random (self)                     3_975.22     1_626.36     5_601.58       0.7555     0.000498        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_438.69       107.08     3_545.77       0.3056          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_438.69       122.91     3_561.60       0.3049          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_438.69       158.62     3_597.31       0.3041          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_438.69       140.67     3_579.35       0.6014     0.001168        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_438.69       171.67     3_610.36       0.7359     0.000550        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_438.69       224.42     3_663.10       0.8491     0.000237        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_438.69       160.16     3_598.84       0.6011     0.001165        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_438.69       192.13     3_630.82       0.7363     0.000543        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_438.69       247.83     3_686.52       0.8513     0.000225        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_438.69       197.35     3_636.03       0.6000     0.001169        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_438.69       233.23     3_671.92       0.7352     0.000545        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_438.69       295.46     3_734.14       0.8505     0.000225        10.40
IVF-Binary-512-nl273-itq (self)                        3_438.69     1_905.40     5_344.09       0.7360     0.000540        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_765.26       110.63     3_875.89       0.3059          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_765.26       137.97     3_903.23       0.3046          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_765.26       142.57     3_907.83       0.6017     0.001164        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_765.26       173.91     3_939.17       0.7365     0.000546        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_765.26       225.95     3_991.21       0.8502     0.000232        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_765.26       174.30     3_939.56       0.6002     0.001168        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_765.26       207.80     3_973.06       0.7357     0.000543        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_765.26       265.01     4_030.27       0.8505     0.000224        10.41
IVF-Binary-512-nl387-itq (self)                        3_765.26     1_723.71     5_488.97       0.7359     0.000544        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_264.25       101.55     4_365.80       0.3065          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_264.25       112.95     4_377.20       0.3056          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_264.25       126.86     4_391.11       0.3048          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_264.25       136.14     4_400.39       0.6033     0.001155        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_264.25       163.59     4_427.84       0.7382     0.000542        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_264.25       213.57     4_477.82       0.8515     0.000231        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_264.25       149.45     4_413.70       0.6024     0.001157        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_264.25       175.72     4_439.97       0.7377     0.000539        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_264.25       229.73     4_493.98       0.8521     0.000223        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_264.25       162.85     4_427.10       0.6009     0.001164        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_264.25       193.22     4_457.48       0.7364     0.000541        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_264.25       250.55     4_514.80       0.8513     0.000223        10.44
IVF-Binary-512-nl547-itq (self)                        4_264.25     1_630.75     5_895.00       0.7375     0.000541        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_001.16        44.53     1_045.70       0.0691          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_001.16        52.66     1_053.82       0.0633          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_001.16        72.14     1_073.30       0.0609          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_001.16        63.72     1_064.89       0.1490     0.009976         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_001.16        82.53     1_083.70       0.2109     0.006864         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_001.16       116.78     1_117.94       0.3020     0.004535         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_001.16        73.68     1_074.84       0.1407     0.010419         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_001.16        92.30     1_093.47       0.2006     0.007202         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_001.16       130.04     1_131.20       0.2906     0.004758         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_001.16        93.21     1_094.38       0.1359     0.010741         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_001.16       134.35     1_135.52       0.1937     0.007453         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_001.16       163.12     1_164.28       0.2806     0.004968         1.76
IVF-Binary-32-nl273-signed (self)                      1_001.16       920.85     1_922.02       0.2029     0.007019         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_341.87        46.40     1_388.27       0.0657          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_341.87        61.27     1_403.15       0.0623          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_341.87        65.15     1_407.02       0.1447     0.010046         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_341.87        83.29     1_425.16       0.2074     0.006884         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_341.87       116.76     1_458.63       0.2999     0.004519         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_341.87        81.76     1_423.63       0.1372     0.010529         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_341.87       100.76     1_442.63       0.1972     0.007266         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_341.87       137.13     1_479.01       0.2853     0.004824         1.77
IVF-Binary-32-nl387-signed (self)                      1_341.87       823.09     2_164.96       0.2095     0.006747         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_838.33        42.36     1_880.69       0.0671          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_838.33        48.11     1_886.44       0.0652          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_838.33        55.31     1_893.64       0.0632          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_838.33        60.31     1_898.64       0.1488     0.009792         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_838.33        78.12     1_916.46       0.2135     0.006671         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_838.33       110.93     1_949.26       0.3087     0.004335         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_838.33        68.84     1_907.17       0.1442     0.010066         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_838.33        85.97     1_924.31       0.2069     0.006904         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_838.33       116.98     1_955.31       0.2995     0.004521         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_838.33        74.00     1_912.34       0.1397     0.010375         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_838.33        92.79     1_931.12       0.2002     0.007154         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_838.33       127.47     1_965.80       0.2895     0.004722         1.79
IVF-Binary-32-nl547-signed (self)                      1_838.33       772.81     2_611.14       0.2153     0.006559         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (Correlated)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.37     1_581.39     1_584.76       1.0000     0.000000        18.31
Exhaustive (self)                                          3.37    16_608.50    16_611.87       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_167.44       501.49     1_668.93       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_167.44       533.88     1_701.33       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_167.44       614.02     1_781.46       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_167.44       681.93     1_849.37       0.5614     0.256662         4.61
ExhaustiveBinary-256-random (self)                     1_167.44     5_819.26     6_986.70       0.4294     0.437191         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_362.98       482.39     1_845.38       0.1323          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_362.98       529.01     1_891.99       0.3010     0.745496         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_362.98       576.88     1_939.86       0.4149     0.448765         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_362.98       666.79     2_029.77       0.5534     0.252544         4.61
ExhaustiveBinary-256-itq (self)                        1_362.98     5_727.68     7_090.66       0.4203     0.435859         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_310.58       821.79     3_132.37       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_310.58       931.65     3_242.22       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_310.58       913.16     3_223.74       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_310.58     1_022.87     3_333.44       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                     2_310.58     9_181.34    11_491.92       0.6185     0.184365         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_600.15       844.39     3_444.53       0.2201          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_600.15       869.12     3_469.26       0.4659     0.364676         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_600.15       921.30     3_521.45       0.6031     0.194711         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_600.15     1_016.67     3_616.82       0.7399     0.094579         9.22
ExhaustiveBinary-512-itq (self)                        2_600.15     9_160.99    11_761.13       0.6053     0.192118         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 165.82       608.90       774.72       0.0127          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   165.82       630.92       796.74       0.0523     2.784656         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  165.82       679.19       845.01       0.0945     1.948956         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  165.82       745.58       911.41       0.1639     1.328894         0.58
ExhaustiveBinary-32-signed (self)                        165.82     6_621.90     6_787.72       0.0975     1.917524         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_029.16        49.03     2_078.19       0.1443          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_029.16        56.08     2_085.24       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_029.16        72.48     2_101.64       0.1371          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_029.16        75.79     2_104.95       0.3253     0.693086         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_029.16        95.44     2_124.60       0.4471     0.412061         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_029.16       136.09     2_165.25       0.5876     0.229141         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_029.16       103.56     2_132.72       0.3165     0.721914         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_029.16       109.03     2_138.19       0.4368     0.431051         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_029.16       149.91     2_179.07       0.5764     0.241195         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_029.16        99.96     2_129.12       0.3123     0.739522         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_029.16       128.53     2_157.69       0.4312     0.443199         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_029.16       174.50     2_203.66       0.5697     0.249326         5.79
IVF-Binary-256-nl273-random (self)                     2_029.16     1_065.43     3_094.60       0.4422     0.416770         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_369.32        51.78     2_421.09       0.1403          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_369.32        63.24     2_432.55       0.1380          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_369.32        76.16     2_445.48       0.3203     0.707603         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_369.32        99.12     2_468.44       0.4414     0.420884         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_369.32       139.09     2_508.41       0.5825     0.233595         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_369.32        88.86     2_458.18       0.3145     0.730875         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_369.32       121.17     2_490.48       0.4334     0.437761         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_369.32       158.80     2_528.12       0.5726     0.245420         5.80
IVF-Binary-256-nl387-random (self)                     2_369.32       982.25     3_351.57       0.4467     0.406730         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_840.47        50.36     2_890.83       0.1414          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_840.47        54.10     2_894.57       0.1400          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_840.47        59.43     2_899.90       0.1386          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_840.47        73.17     2_913.64       0.3225     0.697036         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_840.47        94.27     2_934.74       0.4446     0.412525         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_840.47       133.73     2_974.20       0.5866     0.227694         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_840.47        76.58     2_917.04       0.3186     0.712264         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_840.47        99.59     2_940.05       0.4393     0.423758         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_840.47       140.05     2_980.51       0.5799     0.235815         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_840.47        84.55     2_925.01       0.3149     0.727559         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_840.47       107.76     2_948.22       0.4343     0.435358         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_840.47       149.81     2_990.28       0.5738     0.243782         5.82
IVF-Binary-256-nl547-random (self)                     2_840.47       938.08     3_778.54       0.4498     0.399469         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_211.59        50.17     2_261.76       0.1419          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_211.59        57.18     2_268.77       0.1360          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_211.59        74.61     2_286.20       0.1341          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_211.59        75.54     2_287.13       0.3212     0.685010         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_211.59        98.07     2_309.66       0.4400     0.406661         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_211.59       139.13     2_350.72       0.5808     0.224650         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_211.59        82.82     2_294.41       0.3108     0.716015         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_211.59       110.30     2_321.88       0.4281     0.426268         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_211.59       153.00     2_364.59       0.5689     0.236183         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_211.59       101.47     2_313.06       0.3065     0.734144         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_211.59       133.06     2_344.65       0.4227     0.437877         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_211.59       176.70     2_388.29       0.5619     0.244814         5.79
IVF-Binary-256-nl273-itq (self)                        2_211.59     1_082.97     3_294.56       0.4336     0.414147         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_550.23        52.76     2_602.99       0.1381          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_550.23        64.19     2_614.42       0.1355          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_550.23        76.56     2_626.79       0.3138     0.702306         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_550.23        99.39     2_649.62       0.4326     0.417460         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_550.23       138.63     2_688.86       0.5748     0.230083         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_550.23        89.55     2_639.77       0.3080     0.725553         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_550.23       114.61     2_664.84       0.4251     0.432819         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_550.23       157.35     2_707.57       0.5646     0.241740         5.80
IVF-Binary-256-nl387-itq (self)                        2_550.23     1_001.14     3_551.36       0.4384     0.405037         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_049.61        50.68     3_100.28       0.1387          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_049.61        54.34     3_103.95       0.1368          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_049.61        60.85     3_110.45       0.1349          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_049.61        73.46     3_123.07       0.3163     0.695063         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_049.61        95.15     3_144.75       0.4354     0.411472         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_049.61       134.29     3_183.90       0.5784     0.225638         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_049.61        89.93     3_139.53       0.3122     0.711002         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_049.61       101.23     3_150.84       0.4295     0.423325         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_049.61       141.69     3_191.30       0.5713     0.233789         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_049.61        86.28     3_135.88       0.3082     0.727060         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_049.61       108.62     3_158.23       0.4242     0.434190         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_049.61       151.51     3_201.11       0.5647     0.241780         5.82
IVF-Binary-256-nl547-itq (self)                        3_049.61       948.71     3_998.32       0.4406     0.400147         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_166.79        98.54     3_265.32       0.2304          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_166.79       112.84     3_279.63       0.2278          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_166.79       146.59     3_313.37       0.2270          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_166.79       128.14     3_294.93       0.4860     0.337755        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_166.79       156.73     3_323.52       0.6273     0.176939        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_166.79       238.04     3_404.82       0.7652     0.083691        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_166.79       145.39     3_312.17       0.4812     0.344920        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_166.79       174.82     3_341.61       0.6224     0.181309        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_166.79       226.75     3_393.54       0.7602     0.086526        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_166.79       183.09     3_349.87       0.4791     0.349097        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_166.79       216.76     3_383.55       0.6194     0.184459        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_166.79       275.98     3_442.76       0.7568     0.088670        10.40
IVF-Binary-512-nl273-random (self)                     3_166.79     1_757.74     4_924.53       0.6260     0.177548        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_506.20       106.80     3_613.00       0.2289          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_506.20       127.52     3_633.72       0.2275          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_506.20       134.38     3_640.58       0.4839     0.341409        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_506.20       160.72     3_666.92       0.6254     0.178808        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_506.20       213.43     3_719.63       0.7626     0.084991        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_506.20       160.63     3_666.83       0.4809     0.347106        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_506.20       190.10     3_696.30       0.6210     0.183001        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_506.20       244.02     3_750.22       0.7578     0.087878        10.41
IVF-Binary-512-nl387-random (self)                     3_506.20     1_583.29     5_089.49       0.6282     0.175442        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           3_985.77        95.39     4_081.17       0.2295          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           3_985.77       104.86     4_090.63       0.2286          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           3_985.77       118.86     4_104.64       0.2277          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           3_985.77       125.38     4_111.15       0.4843     0.340155        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          3_985.77       151.18     4_136.96       0.6262     0.177887        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          3_985.77       199.63     4_185.40       0.7649     0.083909        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           3_985.77       135.79     4_121.57       0.4822     0.344454        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          3_985.77       163.47     4_149.24       0.6234     0.180818        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          3_985.77       210.80     4_196.58       0.7614     0.086094        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           3_985.77       150.56     4_136.34       0.4802     0.348120        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          3_985.77       179.64     4_165.41       0.6210     0.183288        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          3_985.77       232.88     4_218.65       0.7585     0.087932        10.43
IVF-Binary-512-nl547-random (self)                     3_985.77     1_501.69     5_487.46       0.6297     0.174185        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_498.40       109.73     3_608.13       0.2257          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_498.40       124.50     3_622.90       0.2228          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_498.40       165.43     3_663.83       0.2217          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_498.40       131.22     3_629.62       0.4776     0.346424        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_498.40       190.63     3_689.03       0.6165     0.182585        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_498.40       218.62     3_717.02       0.7533     0.086960        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_498.40       152.20     3_650.60       0.4724     0.354595        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_498.40       188.96     3_687.36       0.6109     0.187437        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_498.40       232.54     3_730.94       0.7479     0.089948        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_498.40       185.51     3_683.91       0.4700     0.359308        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_498.40       220.69     3_719.09       0.6079     0.190682        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_498.40       288.64     3_787.04       0.7442     0.092299        10.40
IVF-Binary-512-nl273-itq (self)                        3_498.40     1_790.46     5_288.86       0.6129     0.185172        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_942.78       106.80     4_049.58       0.2241          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_942.78       134.75     4_077.53       0.2228          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_942.78       138.06     4_080.84       0.4746     0.350633        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_942.78       175.65     4_118.42       0.6132     0.185038        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_942.78       248.41     4_191.19       0.7506     0.088404        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_942.78       171.00     4_113.78       0.4712     0.356842        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_942.78       195.51     4_138.29       0.6089     0.189400        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_942.78       254.48     4_197.26       0.7457     0.091460        10.41
IVF-Binary-512-nl387-itq (self)                        3_942.78     1_663.24     5_606.02       0.6153     0.182902        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_385.03        99.86     4_484.90       0.2247          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_385.03       110.90     4_495.93       0.2239          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_385.03       123.98     4_509.02       0.2230          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_385.03       150.38     4_535.42       0.4758     0.348893        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_385.03       175.61     4_560.64       0.6143     0.183855        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_385.03       205.57     4_590.60       0.7530     0.087132        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_385.03       148.88     4_533.91       0.4732     0.353735        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_385.03       176.99     4_562.02       0.6109     0.187362        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_385.03       228.61     4_613.64       0.7491     0.089423        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_385.03       174.14     4_559.17       0.4711     0.357554        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_385.03       184.40     4_569.43       0.6083     0.190193        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_385.03       246.78     4_631.81       0.7458     0.091576        10.43
IVF-Binary-512-nl547-itq (self)                        4_385.03     1_572.56     5_957.60       0.6169     0.181414        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_191.24        52.23     1_243.47       0.0236          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_191.24        60.68     1_251.92       0.0154          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_191.24        71.59     1_262.83       0.0138          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_191.24        63.72     1_254.96       0.0900     2.410446         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_191.24        84.48     1_275.71       0.1545     1.700610         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_191.24        95.86     1_287.10       0.2536     1.152475         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_191.24        73.01     1_264.24       0.0637     2.873877         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_191.24       107.63     1_298.87       0.1149     2.056942         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_191.24        99.47     1_290.71       0.1984     1.414400         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_191.24        83.67     1_274.91       0.0565     3.117689         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_191.24        99.08     1_290.31       0.1034     2.251510         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_191.24       124.60     1_315.84       0.1800     1.559523         1.76
IVF-Binary-32-nl273-signed (self)                      1_191.24       738.94     1_930.18       0.1190     2.023478         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_449.08        44.24     1_493.32       0.0178          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_449.08        58.88     1_507.96       0.0149          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_449.08        55.63     1_504.71       0.0722     2.606487         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_449.08        66.77     1_515.85       0.1288     1.851199         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_449.08        92.71     1_541.79       0.2194     1.233441         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_449.08        70.14     1_519.22       0.0616     2.927243         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_449.08        90.92     1_540.01       0.1123     2.092484         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_449.08       111.89     1_560.97       0.1941     1.417090         1.77
IVF-Binary-32-nl387-signed (self)                      1_449.08       662.98     2_112.07       0.1327     1.822713         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_955.51        44.95     2_000.46       0.0186          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_955.51        46.27     2_001.78       0.0172          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_955.51        53.87     2_009.39       0.0153          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_955.51        51.68     2_007.19       0.0764     2.534943         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_955.51        63.90     2_019.41       0.1357     1.769596         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_955.51        95.79     2_051.31       0.2278     1.159882         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_955.51        58.93     2_014.44       0.0706     2.697489         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_955.51        76.94     2_032.46       0.1260     1.897793         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_955.51        99.50     2_055.02       0.2128     1.252096         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_955.51        65.77     2_021.29       0.0642     2.948452         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_955.51        76.88     2_032.39       0.1152     2.093163         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_955.51       113.36     2_068.87       0.1966     1.385020         1.79
IVF-Binary-32-nl547-signed (self)                      1_955.51       644.13     2_599.64       0.1380     1.737336         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.22     1_616.94     1_620.15       1.0000     0.000000        18.31
Exhaustive (self)                                          3.22    16_982.18    16_985.40       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_171.97       499.72     1_671.69       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_171.97       544.85     1_716.82       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_171.97       609.79     1_781.76       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_171.97       723.14     1_895.11       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                     1_171.97     5_926.21     7_098.19       0.3238     3.208083         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_353.20       495.63     1_848.83       0.0723          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_353.20       538.39     1_891.59       0.1941     5.797375         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_353.20       581.87     1_935.07       0.2870     3.686845         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_353.20       673.19     2_026.39       0.4150     2.209544         4.61
ExhaustiveBinary-256-itq (self)                        1_353.20     5_855.65     7_208.85       0.2901     3.621623         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_333.20       856.99     3_190.19       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_333.20       887.63     3_220.84       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_333.20       944.27     3_277.47       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_333.20     1_034.01     3_367.22       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                     2_333.20     9_332.61    11_665.81       0.4788     1.710526         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_569.82       872.49     3_442.31       0.1559          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_569.82     1_028.67     3_598.50       0.3458     3.038797         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_569.82       983.34     3_553.16       0.4728     1.776101         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_569.82     1_107.33     3_677.16       0.6194     0.970394         9.22
ExhaustiveBinary-512-itq (self)                        2_569.82     9_670.85    12_240.67       0.4718     1.776907         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 159.18       654.73       813.91       0.0061          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   159.18       674.33       833.50       0.0286    14.192335         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  159.18       692.66       851.84       0.0545    10.228876         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  159.18       764.02       923.19       0.1028     7.206309         0.58
ExhaustiveBinary-32-signed (self)                        159.18     6_866.60     7_025.78       0.0548    10.217732         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_164.53        50.95     2_215.47       0.1015          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_164.53        57.19     2_221.72       0.0926          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_164.53        72.09     2_236.62       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_164.53        73.41     2_237.94       0.2466     4.682597         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_164.53        96.54     2_261.07       0.3527     2.916537         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_164.53       134.57     2_299.09       0.4886     1.698168         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_164.53        80.89     2_245.42       0.2318     5.166092         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_164.53       105.20     2_269.73       0.3354     3.190525         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_164.53       147.07     2_311.60       0.4700     1.862358         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_164.53        97.04     2_261.57       0.2280     5.296145         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_164.53       126.01     2_290.54       0.3304     3.274036         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_164.53       169.40     2_333.93       0.4638     1.915926         5.79
IVF-Binary-256-nl273-random (self)                     2_164.53     1_046.44     3_210.96       0.3387     3.130099         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_404.81        51.07     2_455.88       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_404.81        62.09     2_466.90       0.0919          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_404.81        72.92     2_477.73       0.2358     4.932324         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_404.81        95.47     2_500.28       0.3421     3.052908         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_404.81       137.40     2_542.21       0.4777     1.801955         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_404.81        86.28     2_491.09       0.2298     5.158334         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_404.81       111.32     2_516.13       0.3340     3.196587         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_404.81       149.75     2_554.56       0.4679     1.887948         5.80
IVF-Binary-256-nl387-random (self)                     2_404.81       946.83     3_351.64       0.3448     3.003033         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_899.27        63.14     2_962.41       0.0953          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_899.27        67.48     2_966.75       0.0934          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_899.27        76.57     2_975.84       0.0917          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_899.27        84.27     2_983.54       0.2397     4.790051         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_899.27       107.95     3_007.22       0.3461     2.989410         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_899.27       146.30     3_045.57       0.4824     1.757249         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_899.27        88.50     2_987.77       0.2354     4.924561         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_899.27       110.40     3_009.67       0.3401     3.083253         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_899.27       147.66     3_046.93       0.4749     1.823375         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_899.27        98.07     2_997.34       0.2310     5.076465         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_899.27       120.49     3_019.76       0.3340     3.185517         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_899.27       160.27     3_059.54       0.4677     1.889603         5.82
IVF-Binary-256-nl547-random (self)                     2_899.27     1_037.95     3_937.22       0.3482     2.939525         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_223.69        48.51     2_272.20       0.0868          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_223.69        54.88     2_278.57       0.0764          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_223.69        70.35     2_294.04       0.0743          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_223.69        71.21     2_294.90       0.2240     5.242254         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_223.69       102.21     2_325.90       0.3228     3.332816         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_223.69       130.69     2_354.38       0.4540     1.955591         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_223.69        80.06     2_303.75       0.2059     5.837070         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_223.69       104.50     2_328.19       0.3031     3.665059         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_223.69       162.71     2_386.40       0.4354     2.128142         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_223.69        99.57     2_323.26       0.2024     5.999445         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_223.69       128.41     2_352.10       0.2984     3.750497         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_223.69       164.17     2_387.86       0.4299     2.178631         5.79
IVF-Binary-256-nl273-itq (self)                        2_223.69     1_040.84     3_264.53       0.3062     3.592545         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_631.83        61.69     2_693.53       0.0787          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_631.83        76.87     2_708.70       0.0755          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_631.83        82.46     2_714.29       0.2106     5.559691         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_631.83       105.04     2_736.87       0.3103     3.474337         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_631.83       149.58     2_781.41       0.4434     2.030065         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_631.83        98.91     2_730.74       0.2045     5.815075         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_631.83       121.98     2_753.81       0.3019     3.634756         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_631.83       162.36     2_794.20       0.4335     2.126122         5.80
IVF-Binary-256-nl387-itq (self)                        2_631.83     1_023.90     3_655.74       0.3133     3.418430         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_043.08        49.90     3_092.97       0.0794          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_043.08        55.64     3_098.72       0.0772          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_043.08        59.75     3_102.83       0.0754          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_043.08        70.28     3_113.35       0.2134     5.388027         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_043.08        90.50     3_133.58       0.3145     3.374689         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_043.08       126.93     3_170.01       0.4484     1.978852         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_043.08        73.46     3_116.53       0.2078     5.557946         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_043.08        94.38     3_137.46       0.3074     3.489575         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_043.08       132.38     3_175.46       0.4400     2.053468         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_043.08        79.78     3_122.86       0.2034     5.753020         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_043.08       101.82     3_144.90       0.3013     3.612860         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_043.08       140.18     3_183.25       0.4325     2.124859         5.82
IVF-Binary-256-nl547-itq (self)                        3_043.08       889.47     3_932.55       0.3172     3.322422         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_200.64       104.13     3_304.77       0.1685          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_200.64       117.79     3_318.43       0.1642          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_200.64       148.11     3_348.75       0.1629          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_200.64       131.22     3_331.86       0.3648     2.762852        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_200.64       157.15     3_357.79       0.4946     1.613586        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_200.64       208.00     3_408.64       0.6428     0.865216        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_200.64       156.03     3_356.67       0.3574     2.870953        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_200.64       175.00     3_375.64       0.4866     1.678608        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_200.64       225.08     3_425.72       0.6348     0.903671        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_200.64       181.33     3_381.98       0.3551     2.903769        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_200.64       215.51     3_416.16       0.4839     1.699072        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_200.64       273.17     3_473.81       0.6317     0.917169        10.40
IVF-Binary-512-nl273-random (self)                     3_200.64     1_740.04     4_940.69       0.4885     1.655177        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_940.42       114.37     4_054.79       0.1647          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_940.42       135.31     4_075.73       0.1631          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_940.42       135.44     4_075.86       0.3603     2.826319        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_940.42       162.39     4_102.81       0.4905     1.642435        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_940.42       210.71     4_151.12       0.6385     0.885603        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_940.42       164.75     4_105.17       0.3567     2.886554        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_940.42       194.19     4_134.61       0.4856     1.679814        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_940.42       241.06     4_181.48       0.6332     0.909694        10.41
IVF-Binary-512-nl387-random (self)                     3_940.42     1_612.27     5_552.68       0.4917     1.624178        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_043.64       103.29     4_146.93       0.1660          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_043.64       114.09     4_157.73       0.1648          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_043.64       127.06     4_170.70       0.1637          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_043.64       129.59     4_173.23       0.3623     2.788844        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_043.64       155.18     4_198.82       0.4927     1.618053        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_043.64       199.37     4_243.00       0.6419     0.867394        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_043.64       139.92     4_183.56       0.3595     2.836677        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_043.64       166.59     4_210.22       0.4890     1.650040        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_043.64       215.49     4_259.13       0.6371     0.891036        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_043.64       161.65     4_205.29       0.3568     2.876142        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_043.64       183.51     4_227.15       0.4854     1.678098        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_043.64       231.24     4_274.88       0.6333     0.908540        10.43
IVF-Binary-512-nl547-random (self)                     4_043.64     1_545.31     5_588.95       0.4940     1.602595        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_462.55       104.98     3_567.53       0.1641          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_462.55       117.24     3_579.79       0.1594          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_462.55       149.74     3_612.29       0.1586          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_462.55       131.45     3_594.00       0.3622     2.801495        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_462.55       160.32     3_622.87       0.4908     1.632770        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_462.55       202.94     3_665.49       0.6380     0.884673        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_462.55       147.72     3_610.27       0.3548     2.928605        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_462.55       176.51     3_639.06       0.4828     1.707191        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_462.55       237.31     3_699.86       0.6304     0.924631        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_462.55       185.79     3_648.34       0.3529     2.955746        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_462.55       215.08     3_677.63       0.4806     1.725365        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_462.55       269.10     3_731.65       0.6277     0.936508        10.40
IVF-Binary-512-nl273-itq (self)                        3_462.55     1_792.67     5_255.22       0.4822     1.709519        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_825.26       110.93     3_936.19       0.1606          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_825.26       138.54     3_963.79       0.1589          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_825.26       140.56     3_965.82       0.3567     2.893509        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_825.26       163.10     3_988.35       0.4852     1.682718        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_825.26       208.66     4_033.91       0.6339     0.904804        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_825.26       167.00     3_992.26       0.3529     2.952913        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_825.26       196.67     4_021.93       0.4807     1.720456        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_825.26       243.37     4_068.63       0.6290     0.928580        10.41
IVF-Binary-512-nl387-itq (self)                        3_825.26     1_628.40     5_453.66       0.4854     1.676904        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_297.38       114.99     4_412.37       0.1614          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_297.38       124.48     4_421.86       0.1601          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_297.38       130.37     4_427.75       0.1590          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_297.38       133.26     4_430.64       0.3586     2.846504        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_297.38       156.37     4_453.75       0.4877     1.655088        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_297.38       201.19     4_498.57       0.6368     0.889086        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_297.38       145.92     4_443.30       0.3555     2.901350        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_297.38       166.54     4_463.92       0.4839     1.690289        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_297.38       210.89     4_508.27       0.6324     0.913441        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_297.38       154.76     4_452.15       0.3531     2.945969        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_297.38       181.25     4_478.63       0.4810     1.718298        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_297.38       230.37     4_527.75       0.6288     0.931787        10.43
IVF-Binary-512-nl547-itq (self)                        4_297.38     1_532.08     5_829.46       0.4875     1.654300        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_030.96        41.00     1_071.97       0.0130          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_030.96        48.57     1_079.54       0.0074          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_030.96        64.55     1_095.52       0.0064          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_030.96        51.75     1_082.71       0.0591    13.000291         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_030.96        55.78     1_086.74       0.1116     9.719921         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_030.96        72.38     1_103.34       0.2026     7.085233         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_030.96        55.05     1_086.01       0.0353    15.842858         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_030.96        62.47     1_093.43       0.0694    12.250771         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_030.96        80.27     1_111.23       0.1301     9.401566         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_030.96        72.71     1_103.67       0.0315    16.892835         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_030.96        81.64     1_112.60       0.0622    13.196258         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_030.96        99.76     1_130.72       0.1188    10.155905         1.76
IVF-Binary-32-nl273-signed (self)                      1_030.96       619.76     1_650.73       0.0700    12.239843         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_379.35        43.62     1_422.97       0.0090          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_379.35        57.98     1_437.34       0.0074          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_379.35        52.88     1_432.24       0.0422    14.669474         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_379.35        57.43     1_436.78       0.0814    11.360889         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_379.35        74.86     1_454.21       0.1521     8.329832         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_379.35        66.43     1_445.78       0.0362    16.328615         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_379.35        72.26     1_451.61       0.0693    12.840094         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_379.35        89.14     1_468.50       0.1306     9.503298         1.77
IVF-Binary-32-nl387-signed (self)                      1_379.35       564.14     1_943.49       0.0809    11.278042         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_868.94        41.11     1_910.05       0.0094          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_868.94        45.88     1_914.83       0.0084          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_868.94        52.97     1_921.91       0.0075          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_868.94        46.85     1_915.80       0.0449    13.934437         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_868.94        54.40     1_923.34       0.0868    10.610329         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_868.94        70.32     1_939.26       0.1631     7.402378         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_868.94        51.49     1_920.44       0.0404    15.027103         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_868.94        59.11     1_928.05       0.0786    11.592403         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_868.94        74.54     1_943.49       0.1486     8.130423         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_868.94        59.35     1_928.29       0.0364    16.431882         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_868.94        67.21     1_936.15       0.0711    12.849049         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_868.94        82.45     1_951.39       0.1343     9.208698         1.79
IVF-Binary-32-nl547-signed (self)                      1_868.94       551.36     2_420.31       0.0883    10.543500         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With more dimensions

The binary indices shine more with more dimensions in the end; however,
the strong compression still yields much worse Recalls.

<details>
<summary><b>Binary - Euclidean (LowRank - 128 bit)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.52     6_142.75     6_158.26       1.0000     0.000000        73.24
Exhaustive (self)                                         15.52    62_488.85    62_504.36       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_749.79       514.42     4_264.21       0.0770          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_749.79       556.12     4_305.90       0.1853    40.590911         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_749.79       606.03     4_355.81       0.2680    26.716815         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_749.79       713.47     4_463.26       0.3854    16.589779         4.70
ExhaustiveBinary-256-random (self)                     3_749.79     6_057.62     9_807.40       0.2721    26.279507         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 4_555.77       520.79     5_076.56       0.0352          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                   4_555.77       551.09     5_106.86       0.1078    58.311100         4.70
ExhaustiveBinary-256-itq-rf10 (query)                  4_555.77       618.74     5_174.51       0.1716    39.742881         4.70
ExhaustiveBinary-256-itq-rf20 (query)                  4_555.77       700.98     5_256.75       0.2692    25.845190         4.70
ExhaustiveBinary-256-itq (self)                        4_555.77     6_014.38    10_570.15       0.1755    39.408653         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_344.83       885.77     8_230.59       0.1527          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_344.83       937.41     8_282.23       0.3242    22.196757         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_344.83       987.37     8_332.19       0.4419    13.545010         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_344.83     1_101.61     8_446.43       0.5832     7.719977         9.41
ExhaustiveBinary-512-random (self)                     7_344.83     9_887.48    17_232.31       0.4431    13.453984         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 8_249.18       884.72     9_133.90       0.1257          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                   8_249.18       933.99     9_183.17       0.2688    27.688764         9.41
ExhaustiveBinary-512-itq-rf10 (query)                  8_249.18       988.76     9_237.94       0.3728    17.457655         9.41
ExhaustiveBinary-512-itq-rf20 (query)                  8_249.18     1_101.53     9_350.71       0.5062    10.365482         9.41
ExhaustiveBinary-512-itq (self)                        8_249.18     9_865.67    18_114.85       0.3728    17.440968         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            14_576.99     1_502.40    16_079.40       0.2353          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              14_576.99     1_555.85    16_132.84       0.4918    11.183271        18.81
ExhaustiveBinary-1024-random-rf10 (query)             14_576.99     1_628.03    16_205.02       0.6358     5.988378        18.81
ExhaustiveBinary-1024-random-rf20 (query)             14_576.99     1_827.22    16_404.21       0.7779     2.871209        18.81
ExhaustiveBinary-1024-random (self)                   14_576.99    16_370.50    30_947.49       0.6381     5.940314        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               15_848.90     1_521.19    17_370.09       0.2198          NaN        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq-rf5 (query)                 15_848.90     1_561.86    17_410.76       0.4636    12.595159        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                15_848.90     1_637.68    17_486.58       0.6048     6.884343        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                15_848.90     1_756.29    17_605.19       0.7500     3.387537        18.81
ExhaustiveBinary-1024-itq (self)                      15_848.90    16_370.59    32_219.49       0.6059     6.857783        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-128-signed_no_rr (query)              1_909.80       388.43     2_298.23       0.0275          NaN         2.35
ExhaustiveBinary-128-signed-rf5 (query)                1_909.80       423.24     2_333.04       0.0917    63.471902         2.35
ExhaustiveBinary-128-signed-rf10 (query)               1_909.80       465.57     2_375.37       0.1491    44.012830         2.35
ExhaustiveBinary-128-signed-rf20 (query)               1_909.80       561.09     2_470.89       0.2396    29.121769         2.35
ExhaustiveBinary-128-signed (self)                     1_909.80     4_642.74     6_552.54       0.1538    43.418564         2.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           6_553.22        87.31     6_640.53       0.0894          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           6_553.22        89.94     6_643.16       0.0819          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           6_553.22       103.36     6_656.58       0.0801          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           6_553.22       117.75     6_670.97       0.2082    37.004660         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          6_553.22       150.30     6_703.51       0.2984    23.990935         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          6_553.22       212.35     6_765.57       0.4184    14.853320         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           6_553.22       122.49     6_675.71       0.1959    39.545046         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          6_553.22       157.09     6_710.31       0.2835    25.724251         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          6_553.22       223.77     6_776.99       0.4031    15.890861         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           6_553.22       135.53     6_688.74       0.1927    40.260869         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          6_553.22       170.84     6_724.05       0.2789    26.258619         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          6_553.22       239.79     6_793.00       0.3972    16.273115         5.98
IVF-Binary-256-nl273-random (self)                     6_553.22     1_572.90     8_126.12       0.2865    25.392015         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)           7_696.90        88.68     7_785.57       0.0832          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           7_696.90       101.06     7_797.96       0.0801          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           7_696.90       121.63     7_818.52       0.1995    38.528842         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          7_696.90       158.28     7_855.17       0.2884    25.014672         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          7_696.90       217.17     7_914.07       0.4104    15.315480         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           7_696.90       131.90     7_828.80       0.1929    40.071462         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          7_696.90       172.31     7_869.21       0.2798    26.100169         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          7_696.90       232.00     7_928.89       0.4005    15.973244         6.04
IVF-Binary-256-nl387-random (self)                     7_696.90     1_535.78     9_232.67       0.2930    24.576691         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)           9_307.63        91.72     9_399.35       0.0847          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           9_307.63        94.12     9_401.75       0.0829          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           9_307.63       100.55     9_408.18       0.0815          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           9_307.63       122.99     9_430.62       0.2025    37.568610         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          9_307.63       156.92     9_464.55       0.2934    24.313937         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          9_307.63       214.63     9_522.26       0.4162    14.904961         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           9_307.63       126.00     9_433.63       0.1984    38.517340         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          9_307.63       161.08     9_468.71       0.2875    25.024912         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          9_307.63       222.84     9_530.47       0.4083    15.400206         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           9_307.63       132.20     9_439.83       0.1947    39.414704         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          9_307.63       165.83     9_473.46       0.2822    25.707970         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          9_307.63       229.68     9_537.31       0.4019    15.826438         6.12
IVF-Binary-256-nl547-random (self)                     9_307.63     1_549.43    10_857.06       0.2978    23.892058         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              7_403.47        85.52     7_488.99       0.0482          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)              7_403.47        91.51     7_494.99       0.0390          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)              7_403.47       103.75     7_507.22       0.0374          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)              7_403.47       115.31     7_518.78       0.1397    52.463938         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)             7_403.47       148.90     7_552.37       0.2145    35.197972         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)             7_403.47       203.35     7_606.82       0.3213    22.347273         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)              7_403.47       118.80     7_522.28       0.1199    57.928141         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)             7_403.47       154.46     7_557.94       0.1903    39.114020         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)             7_403.47       215.06     7_618.53       0.2962    24.655666         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)              7_403.47       142.79     7_546.26       0.1149    60.513426         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)             7_403.47       172.60     7_576.07       0.1829    41.080971         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)             7_403.47       230.29     7_633.76       0.2865    25.849148         5.98
IVF-Binary-256-nl273-itq (self)                        7_403.47     1_495.52     8_898.99       0.1938    38.884114         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)              8_563.02        90.70     8_653.72       0.0404          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)              8_563.02        99.91     8_662.93       0.0378          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)              8_563.02       123.51     8_686.53       0.1243    55.642534         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)             8_563.02       147.59     8_710.60       0.1978    37.321941         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)             8_563.02       206.26     8_769.28       0.3054    23.656456         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)              8_563.02       126.43     8_689.44       0.1163    58.801648         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)             8_563.02       157.77     8_720.79       0.1864    39.622310         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)             8_563.02       220.03     8_783.05       0.2896    25.336476         6.04
IVF-Binary-256-nl387-itq (self)                        8_563.02     1_470.16    10_033.18       0.2017    36.990422         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             10_014.02        92.12    10_106.14       0.0416          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             10_014.02        94.68    10_108.70       0.0403          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             10_014.02       100.29    10_114.31       0.0385          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             10_014.02       118.48    10_132.50       0.1289    53.785071         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            10_014.02       147.42    10_161.44       0.2041    36.103561         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            10_014.02       205.02    10_219.04       0.3149    22.628192         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             10_014.02       120.17    10_134.19       0.1243    55.479762         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            10_014.02       151.72    10_165.75       0.1975    37.316953         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            10_014.02       210.69    10_224.71       0.3057    23.490532         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             10_014.02       126.22    10_140.24       0.1194    57.266975         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            10_014.02       157.18    10_171.20       0.1907    38.580131         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            10_014.02       218.41    10_232.43       0.2973    24.382722         6.12
IVF-Binary-256-nl547-itq (self)                       10_014.02     1_470.62    11_484.64       0.2073    35.790167         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          10_202.08       173.23    10_375.31       0.1596          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          10_202.08       185.50    10_387.58       0.1563          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          10_202.08       216.57    10_418.64       0.1554          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          10_202.08       225.00    10_427.07       0.3370    20.987751        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         10_202.08       263.15    10_465.23       0.4570    12.738091        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         10_202.08       334.37    10_536.44       0.5992     7.207613        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          10_202.08       243.50    10_445.58       0.3308    21.630921        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         10_202.08       279.99    10_482.06       0.4506    13.121710        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         10_202.08       349.69    10_551.77       0.5923     7.447657        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          10_202.08       269.89    10_471.96       0.3289    21.801439        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         10_202.08       346.22    10_548.30       0.4477    13.266981        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         10_202.08       445.36    10_647.44       0.5898     7.527176        10.69
IVF-Binary-512-nl273-random (self)                    10_202.08     2_788.61    12_990.69       0.4518    13.046860        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          11_365.16       176.68    11_541.84       0.1571          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          11_365.16       200.12    11_565.29       0.1557          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          11_365.16       227.57    11_592.73       0.3331    21.366472        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         11_365.16       265.75    11_630.91       0.4543    12.892701        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         11_365.16       330.95    11_696.11       0.5962     7.323848        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          11_365.16       253.10    11_618.26       0.3294    21.746609        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         11_365.16       299.62    11_664.79       0.4493    13.172260        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         11_365.16       369.16    11_734.32       0.5909     7.498725        10.74
IVF-Binary-512-nl387-random (self)                    11_365.16     2_646.37    14_011.53       0.4552    12.836098        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          12_950.15       175.68    13_125.83       0.1579          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          12_950.15       182.78    13_132.94       0.1569          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          12_950.15       197.49    13_147.64       0.1560          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          12_950.15       224.88    13_175.03       0.3354    21.117508        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         12_950.15       259.39    13_209.55       0.4570    12.738653        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         12_950.15       321.08    13_271.23       0.5999     7.182206        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          12_950.15       235.25    13_185.40       0.3326    21.426735        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         12_950.15       271.60    13_221.75       0.4531    12.960609        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         12_950.15       336.32    13_286.48       0.5949     7.349314        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          12_950.15       255.67    13_205.83       0.3301    21.676819        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         12_950.15       290.43    13_240.58       0.4501    13.125100        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         12_950.15       356.37    13_306.52       0.5916     7.462558        10.82
IVF-Binary-512-nl547-random (self)                    12_950.15     2_589.66    15_539.81       0.4583    12.660445        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_124.08       183.03    11_307.11       0.1335          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_124.08       187.68    11_311.76       0.1292          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_124.08       230.37    11_354.45       0.1280          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_124.08       223.68    11_347.76       0.2833    26.028741        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_124.08       260.07    11_384.15       0.3914    16.282029        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_124.08       323.83    11_447.90       0.5268     9.605272        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_124.08       238.57    11_362.64       0.2760    26.895798        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_124.08       285.41    11_409.48       0.3833    16.818853        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_124.08       348.31    11_472.38       0.5181     9.956129        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_124.08       269.11    11_393.19       0.2736    27.247896        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_124.08       319.57    11_443.65       0.3797    17.080475        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_124.08       390.25    11_514.33       0.5142    10.116339        10.69
IVF-Binary-512-nl273-itq (self)                       11_124.08     2_806.88    13_930.96       0.3832    16.843347        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_250.31       187.04    12_437.35       0.1306          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_250.31       207.77    12_458.08       0.1284          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_250.31       232.32    12_482.63       0.2787    26.548786        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_250.31       273.59    12_523.90       0.3872    16.550673        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_250.31       343.81    12_594.12       0.5220     9.796127        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_250.31       260.57    12_510.88       0.2742    27.159052        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_250.31       304.33    12_554.64       0.3814    16.980048        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_250.31       377.48    12_627.79       0.5151    10.076268        10.74
IVF-Binary-512-nl387-itq (self)                       12_250.31     2_682.46    14_932.77       0.3869    16.582669        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             13_807.67       177.78    13_985.45       0.1309          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             13_807.67       186.90    13_994.57       0.1297          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             13_807.67       201.91    14_009.58       0.1287          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             13_807.67       227.85    14_035.53       0.2813    26.218914        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            13_807.67       284.17    14_091.85       0.3898    16.347559        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            13_807.67       324.25    14_131.92       0.5268     9.607822        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             13_807.67       242.49    14_050.16       0.2781    26.609800        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            13_807.67       274.50    14_082.17       0.3853    16.644434        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            13_807.67       337.25    14_144.92       0.5211     9.822816        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             13_807.67       255.82    14_063.49       0.2755    26.964158        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            13_807.67       294.00    14_101.67       0.3817    16.897180        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            13_807.67       365.93    14_173.60       0.5166    10.003392        10.82
IVF-Binary-512-nl547-itq (self)                       13_807.67     2_613.18    16_420.85       0.3903    16.342942        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         17_260.34       426.36    17_686.70       0.2399          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         17_260.34       497.90    17_758.24       0.2382          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         17_260.34       563.96    17_824.29       0.2377          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         17_260.34       464.72    17_725.05       0.4992    10.834117        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        17_260.34       495.88    17_756.22       0.6434     5.779240        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        17_260.34       554.60    17_814.93       0.7845     2.747805        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         17_260.34       511.64    17_771.98       0.4962    10.981356        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        17_260.34       544.76    17_805.10       0.6406     5.859089        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        17_260.34       652.82    17_913.16       0.7820     2.797675        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         17_260.34       606.53    17_866.87       0.4952    11.037131        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        17_260.34       648.70    17_909.04       0.6394     5.893505        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        17_260.34       711.59    17_971.92       0.7808     2.820922        20.09
IVF-Binary-1024-nl273-random (self)                   17_260.34     5_455.06    22_715.40       0.6428     5.813738        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         18_556.76       467.15    19_023.92       0.2387          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         18_556.76       539.74    19_096.50       0.2378          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         18_556.76       529.34    19_086.11       0.4978    10.900723        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        18_556.76       556.65    19_113.41       0.6417     5.823667        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        18_556.76       647.05    19_203.81       0.7830     2.778159        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         18_556.76       656.37    19_213.13       0.4959    11.002308        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        18_556.76       624.46    19_181.22       0.6394     5.889963        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        18_556.76       675.92    19_232.68       0.7809     2.816231        20.15
IVF-Binary-1024-nl387-random (self)                   18_556.76     5_133.59    23_690.35       0.6441     5.772927        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         20_220.69       424.95    20_645.64       0.2393          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         20_220.69       467.89    20_688.58       0.2386          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         20_220.69       496.07    20_716.76       0.2379          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         20_220.69       458.65    20_679.33       0.4993    10.819022        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        20_220.69       494.51    20_715.20       0.6438     5.766379        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        20_220.69       560.84    20_781.52       0.7852     2.739656        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         20_220.69       496.99    20_717.68       0.4976    10.906062        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        20_220.69       538.82    20_759.51       0.6415     5.829342        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        20_220.69       587.82    20_808.51       0.7827     2.782993        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         20_220.69       555.09    20_775.78       0.4962    10.977444        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        20_220.69       598.51    20_819.20       0.6400     5.874175        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        20_220.69       641.94    20_862.63       0.7814     2.810928        20.23
IVF-Binary-1024-nl547-random (self)                   20_220.69     4_885.86    25_106.54       0.6462     5.710744        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            19_353.91       468.70    19_822.61       0.2242          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            19_353.91       512.14    19_866.05       0.2223          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            19_353.91       618.32    19_972.23       0.2220          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            19_353.91       500.83    19_854.74       0.4719    12.157504        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           19_353.91       510.43    19_864.34       0.6134     6.627209        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           19_353.91       574.62    19_928.53       0.7573     3.242512        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            19_353.91       523.59    19_877.51       0.4688    12.329625        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           19_353.91       559.78    19_913.69       0.6100     6.735050        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           19_353.91       641.62    19_995.53       0.7541     3.309755        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            19_353.91       621.52    19_975.43       0.4675    12.399111        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           19_353.91       661.65    20_015.56       0.6087     6.782239        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           19_353.91       731.06    20_084.97       0.7528     3.336675        20.09
IVF-Binary-1024-nl273-itq (self)                      19_353.91     5_575.77    24_929.68       0.6108     6.714377        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            20_078.40       444.56    20_522.97       0.2238          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            20_078.40       545.75    20_624.15       0.2227          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            20_078.40       487.18    20_565.58       0.4700    12.245717        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           20_078.40       516.95    20_595.36       0.6113     6.696701        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           20_078.40       585.30    20_663.71       0.7560     3.268036        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            20_078.40       567.56    20_645.96       0.4677    12.372991        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           20_078.40       603.66    20_682.07       0.6089     6.769585        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           20_078.40       665.25    20_743.65       0.7537     3.313465        20.15
IVF-Binary-1024-nl387-itq (self)                      20_078.40     5_170.04    25_248.44       0.6124     6.660010        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            21_302.56       417.89    21_720.45       0.2236          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            21_302.56       447.89    21_750.45       0.2227          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            21_302.56       498.78    21_801.34       0.2221          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            21_302.56       456.69    21_759.25       0.4714    12.177027        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           21_302.56       486.68    21_789.23       0.6140     6.605088        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           21_302.56       541.24    21_843.80       0.7582     3.223254        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            21_302.56       498.37    21_800.93       0.4697    12.275342        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           21_302.56       522.24    21_824.79       0.6115     6.686787        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           21_302.56       582.24    21_884.80       0.7556     3.275393        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            21_302.56       532.80    21_835.36       0.4683    12.346531        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           21_302.56       572.10    21_874.66       0.6098     6.740111        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           21_302.56       633.12    21_935.68       0.7541     3.306943        20.23
IVF-Binary-1024-nl547-itq (self)                      21_302.56     4_846.16    26_148.71       0.6145     6.587174        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-128-nl273-np13-rf0-signed (query)           4_760.15        56.51     4_816.67       0.0399          NaN         3.63
IVF-Binary-128-nl273-np16-rf0-signed (query)           4_760.15        61.24     4_821.39       0.0317          NaN         3.63
IVF-Binary-128-nl273-np23-rf0-signed (query)           4_760.15        73.59     4_833.74       0.0301          NaN         3.63
IVF-Binary-128-nl273-np13-rf5-signed (query)           4_760.15        79.72     4_839.87       0.1240    56.688958         3.63
IVF-Binary-128-nl273-np13-rf10-signed (query)          4_760.15       101.10     4_861.25       0.1943    38.799744         3.63
IVF-Binary-128-nl273-np13-rf20-signed (query)          4_760.15       149.45     4_909.60       0.3001    25.014773         3.63
IVF-Binary-128-nl273-np16-rf5-signed (query)           4_760.15        84.09     4_844.25       0.1043    63.227946         3.63
IVF-Binary-128-nl273-np16-rf10-signed (query)          4_760.15       108.08     4_868.24       0.1684    43.819553         3.63
IVF-Binary-128-nl273-np16-rf20-signed (query)          4_760.15       157.15     4_917.31       0.2676    28.594712         3.63
IVF-Binary-128-nl273-np23-rf5-signed (query)           4_760.15        93.71     4_853.86       0.1001    65.245199         3.63
IVF-Binary-128-nl273-np23-rf10-signed (query)          4_760.15       118.90     4_879.06       0.1629    45.202372         3.63
IVF-Binary-128-nl273-np23-rf20-signed (query)          4_760.15       169.14     4_929.29       0.2585    29.695205         3.63
IVF-Binary-128-nl273-signed (self)                     4_760.15     1_071.02     5_831.17       0.1715    43.671082         3.63
IVF-Binary-128-nl387-np19-rf0-signed (query)           5_916.38        60.81     5_977.19       0.0330          NaN         3.69
IVF-Binary-128-nl387-np27-rf0-signed (query)           5_916.38        68.69     5_985.07       0.0307          NaN         3.69
IVF-Binary-128-nl387-np19-rf5-signed (query)           5_916.38        83.38     5_999.76       0.1088    60.824698         3.69
IVF-Binary-128-nl387-np19-rf10-signed (query)          5_916.38       105.54     6_021.92       0.1766    41.772363         3.69
IVF-Binary-128-nl387-np19-rf20-signed (query)          5_916.38       153.14     6_069.52       0.2795    27.136534         3.69
IVF-Binary-128-nl387-np27-rf5-signed (query)           5_916.38        90.88     6_007.26       0.1009    64.418088         3.69
IVF-Binary-128-nl387-np27-rf10-signed (query)          5_916.38       115.31     6_031.69       0.1645    44.718063         3.69
IVF-Binary-128-nl387-np27-rf20-signed (query)          5_916.38       163.06     6_079.44       0.2616    29.290649         3.69
IVF-Binary-128-nl387-signed (self)                     5_916.38     1_048.30     6_964.68       0.1814    41.284592         3.69
IVF-Binary-128-nl547-np23-rf0-signed (query)           7_529.04        65.04     7_594.08       0.0337          NaN         3.77
IVF-Binary-128-nl547-np27-rf0-signed (query)           7_529.04        69.05     7_598.09       0.0323          NaN         3.77
IVF-Binary-128-nl547-np33-rf0-signed (query)           7_529.04        70.75     7_599.79       0.0309          NaN         3.77
IVF-Binary-128-nl547-np23-rf5-signed (query)           7_529.04        87.68     7_616.72       0.1121    58.734937         3.77
IVF-Binary-128-nl547-np23-rf10-signed (query)          7_529.04       108.47     7_637.51       0.1812    40.085235         3.77
IVF-Binary-128-nl547-np23-rf20-signed (query)          7_529.04       156.77     7_685.81       0.2874    25.551875         3.77
IVF-Binary-128-nl547-np27-rf5-signed (query)           7_529.04        89.29     7_618.33       0.1078    60.612006         3.77
IVF-Binary-128-nl547-np27-rf10-signed (query)          7_529.04       111.88     7_640.92       0.1744    41.625537         3.77
IVF-Binary-128-nl547-np27-rf20-signed (query)          7_529.04       158.61     7_687.65       0.2772    26.742213         3.77
IVF-Binary-128-nl547-np33-rf5-signed (query)           7_529.04        93.95     7_622.99       0.1032    62.857123         3.77
IVF-Binary-128-nl547-np33-rf10-signed (query)          7_529.04       115.93     7_644.97       0.1678    43.287777         3.77
IVF-Binary-128-nl547-np33-rf20-signed (query)          7_529.04       163.08     7_692.12       0.2670    27.968410         3.77
IVF-Binary-128-nl547-signed (self)                     7_529.04     1_064.45     8_593.49       0.1869    39.527632         3.77
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

<details>
<summary><b>Binary - Euclidean (LowRank - 256 bit)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 256D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        30.24    14_849.60    14_879.84       1.0000     0.000000       146.48
Exhaustive (self)                                         30.24   153_520.20   153_550.44       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              7_702.93       573.75     8_276.68       0.0848          NaN         4.83
ExhaustiveBinary-256-random-rf5 (query)                7_702.93       626.26     8_329.19       0.1980    84.795844         4.83
ExhaustiveBinary-256-random-rf10 (query)               7_702.93       771.05     8_473.98       0.2831    55.592413         4.83
ExhaustiveBinary-256-random-rf20 (query)               7_702.93       834.93     8_537.86       0.4015    34.646234         4.83
ExhaustiveBinary-256-random (self)                     7_702.93     6_852.40    14_555.33       0.2867    54.828071         4.83
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 9_928.98       574.98    10_503.96       0.0027          NaN         4.83
ExhaustiveBinary-256-itq-rf5 (query)                   9_928.98       599.37    10_528.35       0.0128   233.324047         4.83
ExhaustiveBinary-256-itq-rf10 (query)                  9_928.98       626.76    10_555.74       0.0254   177.356207         4.83
ExhaustiveBinary-256-itq-rf20 (query)                  9_928.98       713.33    10_642.31       0.0500   134.022852         4.83
ExhaustiveBinary-256-itq (self)                        9_928.98     6_311.93    16_240.92       0.0256   177.818519         4.83
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)             15_351.73       995.42    16_347.15       0.1519          NaN         9.66
ExhaustiveBinary-512-random-rf5 (query)               15_351.73     1_056.94    16_408.67       0.3177    49.573717         9.66
ExhaustiveBinary-512-random-rf10 (query)              15_351.73     1_145.86    16_497.59       0.4333    30.425506         9.66
ExhaustiveBinary-512-random-rf20 (query)              15_351.73     1_295.02    16_646.75       0.5756    17.359891         9.66
ExhaustiveBinary-512-random (self)                    15_351.73    11_271.37    26_623.10       0.4358    30.179063         9.66
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                17_526.72       996.55    18_523.27       0.0848          NaN         9.66
ExhaustiveBinary-512-itq-rf5 (query)                  17_526.72     1_066.38    18_593.09       0.1972    85.264886         9.66
ExhaustiveBinary-512-itq-rf10 (query)                 17_526.72     1_122.13    18_648.85       0.2835    55.811208         9.66
ExhaustiveBinary-512-itq-rf20 (query)                 17_526.72     1_271.32    18_798.03       0.4008    34.735180         9.66
ExhaustiveBinary-512-itq (self)                       17_526.72    11_243.01    28_769.73       0.2855    55.077668         9.66
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            30_018.77     1_724.66    31_743.44       0.2320          NaN        19.31
ExhaustiveBinary-1024-random-rf5 (query)              30_018.77     1_793.52    31_812.30       0.4863    25.204494        19.31
ExhaustiveBinary-1024-random-rf10 (query)             30_018.77     1_863.17    31_881.94       0.6287    13.571699        19.31
ExhaustiveBinary-1024-random-rf20 (query)             30_018.77     2_005.07    32_023.85       0.7709     6.528124        19.31
ExhaustiveBinary-1024-random (self)                   30_018.77    18_640.81    48_659.58       0.6327    13.347710        19.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq_no_rr (query)               33_076.11     1_749.48    34_825.60       0.1937          NaN        19.31
ExhaustiveBinary-1024-itq-rf5 (query)                 33_076.11     1_808.67    34_884.78       0.4069    34.763449        19.31
ExhaustiveBinary-1024-itq-rf10 (query)                33_076.11     1_868.99    34_945.10       0.5407    19.947772        19.31
ExhaustiveBinary-1024-itq-rf20 (query)                33_076.11     2_022.26    35_098.37       0.6873    10.488234        19.31
ExhaustiveBinary-1024-itq (self)                      33_076.11    19_297.82    52_373.93       0.5428    19.821924        19.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-signed_no_rr (query)              7_758.48       582.64     8_341.12       0.0848          NaN         4.83
ExhaustiveBinary-256-signed-rf5 (query)                7_758.48       629.49     8_387.97       0.1980    84.795844         4.83
ExhaustiveBinary-256-signed-rf10 (query)               7_758.48       694.44     8_452.92       0.2831    55.592413         4.83
ExhaustiveBinary-256-signed-rf20 (query)               7_758.48       828.15     8_586.63       0.4015    34.646234         4.83
ExhaustiveBinary-256-signed (self)                     7_758.48     6_878.15    14_636.63       0.2867    54.828071         4.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)          14_159.53       151.80    14_311.33       0.0972          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-random (query)          14_159.53       151.69    14_311.22       0.0901          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-random (query)          14_159.53       166.58    14_326.11       0.0878          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-random (query)          14_159.53       199.98    14_359.51       0.2201    76.967321         6.24
IVF-Binary-256-nl273-np13-rf10-random (query)         14_159.53       258.00    14_417.53       0.3105    49.885639         6.24
IVF-Binary-256-nl273-np13-rf20-random (query)         14_159.53       348.63    14_508.16       0.4337    30.695358         6.24
IVF-Binary-256-nl273-np16-rf5-random (query)          14_159.53       221.93    14_381.46       0.2094    81.402596         6.24
IVF-Binary-256-nl273-np16-rf10-random (query)         14_159.53       279.17    14_438.70       0.2989    52.738843         6.24
IVF-Binary-256-nl273-np16-rf20-random (query)         14_159.53       372.54    14_532.07       0.4208    32.487636         6.24
IVF-Binary-256-nl273-np23-rf5-random (query)          14_159.53       231.70    14_391.23       0.2042    83.447553         6.24
IVF-Binary-256-nl273-np23-rf10-random (query)         14_159.53       284.55    14_444.08       0.2917    54.370157         6.24
IVF-Binary-256-nl273-np23-rf20-random (query)         14_159.53       385.06    14_544.59       0.4128    33.552199         6.24
IVF-Binary-256-nl273-random (self)                    14_159.53     2_609.91    16_769.44       0.3025    52.077444         6.24
IVF-Binary-256-nl387-np19-rf0-random (query)          16_735.43       154.90    16_890.33       0.0917          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-random (query)          16_735.43       164.90    16_900.33       0.0887          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-random (query)          16_735.43       213.07    16_948.49       0.2128    79.553056         6.35
IVF-Binary-256-nl387-np19-rf10-random (query)         16_735.43       259.87    16_995.30       0.3029    51.315014         6.35
IVF-Binary-256-nl387-np19-rf20-random (query)         16_735.43       349.80    17_085.22       0.4264    31.551908         6.35
IVF-Binary-256-nl387-np27-rf5-random (query)          16_735.43       217.12    16_952.54       0.2070    82.152281         6.35
IVF-Binary-256-nl387-np27-rf10-random (query)         16_735.43       274.87    17_010.30       0.2953    53.172257         6.35
IVF-Binary-256-nl387-np27-rf20-random (query)         16_735.43       378.74    17_114.17       0.4166    32.862712         6.35
IVF-Binary-256-nl387-random (self)                    16_735.43     2_604.10    19_339.53       0.3058    50.835729         6.35
IVF-Binary-256-nl547-np23-rf0-random (query)          20_587.93       162.18    20_750.10       0.0929          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-random (query)          20_587.93       167.95    20_755.87       0.0909          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-random (query)          20_587.93       181.91    20_769.83       0.0888          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-random (query)          20_587.93       241.38    20_829.31       0.2173    76.797898         6.51
IVF-Binary-256-nl547-np23-rf10-random (query)         20_587.93       269.20    20_857.12       0.3098    49.340023         6.51
IVF-Binary-256-nl547-np23-rf20-random (query)         20_587.93       371.02    20_958.94       0.4358    30.124481         6.51
IVF-Binary-256-nl547-np27-rf5-random (query)          20_587.93       222.07    20_809.99       0.2128    78.810235         6.51
IVF-Binary-256-nl547-np27-rf10-random (query)         20_587.93       281.32    20_869.25       0.3033    50.850227         6.51
IVF-Binary-256-nl547-np27-rf20-random (query)         20_587.93       364.69    20_952.61       0.4273    31.222115         6.51
IVF-Binary-256-nl547-np33-rf5-random (query)          20_587.93       229.36    20_817.29       0.2084    80.866425         6.51
IVF-Binary-256-nl547-np33-rf10-random (query)         20_587.93       283.91    20_871.84       0.2972    52.390333         6.51
IVF-Binary-256-nl547-np33-rf20-random (query)         20_587.93       388.20    20_976.12       0.4199    32.257029         6.51
IVF-Binary-256-nl547-random (self)                    20_587.93     2_681.51    23_269.44       0.3136    48.731489         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)             16_021.39       149.30    16_170.69       0.0064          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-itq (query)             16_021.39       162.96    16_184.35       0.0040          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-itq (query)             16_021.39       168.41    16_189.80       0.0033          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-itq (query)             16_021.39       180.61    16_202.00       0.0299   204.577361         6.24
IVF-Binary-256-nl273-np13-rf10-itq (query)            16_021.39       213.13    16_234.52       0.0604   156.418921         6.24
IVF-Binary-256-nl273-np13-rf20-itq (query)            16_021.39       282.78    16_304.17       0.1207   117.956687         6.24
IVF-Binary-256-nl273-np16-rf5-itq (query)             16_021.39       180.60    16_201.99       0.0190   228.679028         6.24
IVF-Binary-256-nl273-np16-rf10-itq (query)            16_021.39       215.92    16_237.31       0.0383   179.719098         6.24
IVF-Binary-256-nl273-np16-rf20-itq (query)            16_021.39       284.49    16_305.88       0.0764   141.441457         6.24
IVF-Binary-256-nl273-np23-rf5-itq (query)             16_021.39       192.64    16_214.03       0.0159   242.077967         6.24
IVF-Binary-256-nl273-np23-rf10-itq (query)            16_021.39       227.07    16_248.46       0.0320   192.815504         6.24
IVF-Binary-256-nl273-np23-rf20-itq (query)            16_021.39       300.60    16_321.99       0.0638   154.346052         6.24
IVF-Binary-256-nl273-itq (self)                       16_021.39     2_142.16    18_163.55       0.0392   179.706019         6.24
IVF-Binary-256-nl387-np19-rf0-itq (query)             18_841.38       154.97    18_996.35       0.0043          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-itq (query)             18_841.38       175.18    19_016.55       0.0035          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-itq (query)             18_841.38       185.56    19_026.94       0.0212   223.155040         6.35
IVF-Binary-256-nl387-np19-rf10-itq (query)            18_841.38       218.92    19_060.29       0.0430   175.424823         6.35
IVF-Binary-256-nl387-np19-rf20-itq (query)            18_841.38       292.06    19_133.43       0.0863   137.549631         6.35
IVF-Binary-256-nl387-np27-rf5-itq (query)             18_841.38       194.89    19_036.27       0.0173   239.541955         6.35
IVF-Binary-256-nl387-np27-rf10-itq (query)            18_841.38       233.03    19_074.40       0.0350   190.893745         6.35
IVF-Binary-256-nl387-np27-rf20-itq (query)            18_841.38       299.48    19_140.85       0.0709   151.197763         6.35
IVF-Binary-256-nl387-itq (self)                       18_841.38     2_170.84    21_012.21       0.0434   175.155890         6.35
IVF-Binary-256-nl547-np23-rf0-itq (query)             22_805.23       168.55    22_973.78       0.0048          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-itq (query)             22_805.23       167.60    22_972.83       0.0042          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-itq (query)             22_805.23       174.96    22_980.19       0.0037          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-itq (query)             22_805.23       194.81    23_000.04       0.0244   214.046800         6.51
IVF-Binary-256-nl547-np23-rf10-itq (query)            22_805.23       231.74    23_036.97       0.0483   168.659369         6.51
IVF-Binary-256-nl547-np23-rf20-itq (query)            22_805.23       295.89    23_101.12       0.0962   128.203333         6.51
IVF-Binary-256-nl547-np27-rf5-itq (query)             22_805.23       210.38    23_015.61       0.0211   225.710414         6.51
IVF-Binary-256-nl547-np27-rf10-itq (query)            22_805.23       244.45    23_049.68       0.0424   179.766988         6.51
IVF-Binary-256-nl547-np27-rf20-itq (query)            22_805.23       295.58    23_100.81       0.0850   138.022925         6.51
IVF-Binary-256-nl547-np33-rf5-itq (query)             22_805.23       200.08    23_005.30       0.0181   237.733483         6.51
IVF-Binary-256-nl547-np33-rf10-itq (query)            22_805.23       242.66    23_047.89       0.0364   191.385293         6.51
IVF-Binary-256-nl547-np33-rf20-itq (query)            22_805.23       305.59    23_110.82       0.0730   148.146894         6.51
IVF-Binary-256-nl547-itq (self)                       22_805.23     2_465.38    25_270.61       0.0490   168.725251         6.51
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          21_628.52       289.83    21_918.35       0.1582          NaN        11.07
IVF-Binary-512-nl273-np16-rf0-random (query)          21_628.52       310.59    21_939.11       0.1554          NaN        11.07
IVF-Binary-512-nl273-np23-rf0-random (query)          21_628.52       343.66    21_972.18       0.1540          NaN        11.07
IVF-Binary-512-nl273-np13-rf5-random (query)          21_628.52       362.97    21_991.49       0.3309    46.907985        11.07
IVF-Binary-512-nl273-np13-rf10-random (query)         21_628.52       414.14    22_042.66       0.4495    28.572622        11.07
IVF-Binary-512-nl273-np13-rf20-random (query)         21_628.52       513.86    22_142.38       0.5929    16.174214        11.07
IVF-Binary-512-nl273-np16-rf5-random (query)          21_628.52       413.32    22_041.84       0.3259    47.990736        11.07
IVF-Binary-512-nl273-np16-rf10-random (query)         21_628.52       446.60    22_075.12       0.4433    29.276411        11.07
IVF-Binary-512-nl273-np16-rf20-random (query)         21_628.52       534.56    22_163.07       0.5864    16.636090        11.07
IVF-Binary-512-nl273-np23-rf5-random (query)          21_628.52       428.69    22_057.21       0.3222    48.699082        11.07
IVF-Binary-512-nl273-np23-rf10-random (query)         21_628.52       503.76    22_132.28       0.4395    29.761800        11.07
IVF-Binary-512-nl273-np23-rf20-random (query)         21_628.52       591.11    22_219.63       0.5820    16.949358        11.07
IVF-Binary-512-nl273-random (self)                    21_628.52     4_367.70    25_996.22       0.4458    29.056660        11.07
IVF-Binary-512-nl387-np19-rf0-random (query)          24_579.46       334.74    24_914.20       0.1558          NaN        11.18
IVF-Binary-512-nl387-np27-rf0-random (query)          24_579.46       355.88    24_935.35       0.1542          NaN        11.18
IVF-Binary-512-nl387-np19-rf5-random (query)          24_579.46       379.79    24_959.25       0.3275    47.594899        11.18
IVF-Binary-512-nl387-np19-rf10-random (query)         24_579.46       570.91    25_150.38       0.4457    28.972403        11.18
IVF-Binary-512-nl387-np19-rf20-random (query)         24_579.46       580.33    25_159.80       0.5887    16.437354        11.18
IVF-Binary-512-nl387-np27-rf5-random (query)          24_579.46       447.84    25_027.31       0.3237    48.459293        11.18
IVF-Binary-512-nl387-np27-rf10-random (query)         24_579.46       500.62    25_080.09       0.4408    29.561736        11.18
IVF-Binary-512-nl387-np27-rf20-random (query)         24_579.46       620.18    25_199.65       0.5837    16.823553        11.18
IVF-Binary-512-nl387-random (self)                    24_579.46     4_545.43    29_124.90       0.4478    28.785657        11.18
IVF-Binary-512-nl547-np23-rf0-random (query)          28_363.90       302.74    28_666.64       0.1574          NaN        11.34
IVF-Binary-512-nl547-np27-rf0-random (query)          28_363.90       311.37    28_675.28       0.1560          NaN        11.34
IVF-Binary-512-nl547-np33-rf0-random (query)          28_363.90       330.89    28_694.79       0.1546          NaN        11.34
IVF-Binary-512-nl547-np23-rf5-random (query)          28_363.90       373.82    28_737.73       0.3308    46.776326        11.34
IVF-Binary-512-nl547-np23-rf10-random (query)         28_363.90       418.57    28_782.47       0.4501    28.398732        11.34
IVF-Binary-512-nl547-np23-rf20-random (query)         28_363.90       504.72    28_868.63       0.5943    16.017686        11.34
IVF-Binary-512-nl547-np27-rf5-random (query)          28_363.90       393.17    28_757.07       0.3273    47.563707        11.34
IVF-Binary-512-nl547-np27-rf10-random (query)         28_363.90       462.13    28_826.03       0.4453    29.005000        11.34
IVF-Binary-512-nl547-np27-rf20-random (query)         28_363.90       526.77    28_890.67       0.5884    16.440292        11.34
IVF-Binary-512-nl547-np33-rf5-random (query)          28_363.90       404.95    28_768.86       0.3244    48.206128        11.34
IVF-Binary-512-nl547-np33-rf10-random (query)         28_363.90       457.45    28_821.36       0.4414    29.472939        11.34
IVF-Binary-512-nl547-np33-rf20-random (query)         28_363.90       551.26    28_915.17       0.5846    16.731692        11.34
IVF-Binary-512-nl547-random (self)                    28_363.90     4_147.67    32_511.58       0.4523    28.218805        11.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)             23_810.72       288.05    24_098.77       0.0969          NaN        11.07
IVF-Binary-512-nl273-np16-rf0-itq (query)             23_810.72       303.07    24_113.78       0.0905          NaN        11.07
IVF-Binary-512-nl273-np23-rf0-itq (query)             23_810.72       345.72    24_156.44       0.0879          NaN        11.07
IVF-Binary-512-nl273-np13-rf5-itq (query)             23_810.72       364.65    24_175.37       0.2216    75.977283        11.07
IVF-Binary-512-nl273-np13-rf10-itq (query)            23_810.72       435.26    24_245.97       0.3151    48.939698        11.07
IVF-Binary-512-nl273-np13-rf20-itq (query)            23_810.72       492.33    24_303.04       0.4355    30.332467        11.07
IVF-Binary-512-nl273-np16-rf5-itq (query)             23_810.72       381.23    24_191.95       0.2100    81.371759        11.07
IVF-Binary-512-nl273-np16-rf10-itq (query)            23_810.72       437.67    24_248.39       0.3007    52.697160        11.07
IVF-Binary-512-nl273-np16-rf20-itq (query)            23_810.72       521.19    24_331.90       0.4206    32.536295        11.07
IVF-Binary-512-nl273-np23-rf5-itq (query)             23_810.72       426.72    24_237.43       0.2054    83.522924        11.07
IVF-Binary-512-nl273-np23-rf10-itq (query)            23_810.72       483.34    24_294.05       0.2940    54.299775        11.07
IVF-Binary-512-nl273-np23-rf20-itq (query)            23_810.72       581.41    24_392.12       0.4131    33.558718        11.07
IVF-Binary-512-nl273-itq (self)                       23_810.72     4_292.64    28_103.36       0.3016    52.203990        11.07
IVF-Binary-512-nl387-np19-rf0-itq (query)             26_486.04       299.80    26_785.84       0.0917          NaN        11.18
IVF-Binary-512-nl387-np27-rf0-itq (query)             26_486.04       329.86    26_815.90       0.0889          NaN        11.18
IVF-Binary-512-nl387-np19-rf5-itq (query)             26_486.04       374.15    26_860.19       0.2115    80.344208        11.18
IVF-Binary-512-nl387-np19-rf10-itq (query)            26_486.04       417.78    26_903.83       0.3029    51.998946        11.18
IVF-Binary-512-nl387-np19-rf20-itq (query)            26_486.04       498.51    26_984.55       0.4253    31.955492        11.18
IVF-Binary-512-nl387-np27-rf5-itq (query)             26_486.04       406.80    26_892.84       0.2059    83.213638        11.18
IVF-Binary-512-nl387-np27-rf10-itq (query)            26_486.04       460.14    26_946.18       0.2951    54.021537        11.18
IVF-Binary-512-nl387-np27-rf20-itq (query)            26_486.04       548.32    27_034.37       0.4150    33.377394        11.18
IVF-Binary-512-nl387-itq (self)                       26_486.04     4_150.16    30_636.20       0.3045    51.338199        11.18
IVF-Binary-512-nl547-np23-rf0-itq (query)             30_334.24       305.96    30_640.20       0.0930          NaN        11.34
IVF-Binary-512-nl547-np27-rf0-itq (query)             30_334.24       320.04    30_654.29       0.0909          NaN        11.34
IVF-Binary-512-nl547-np33-rf0-itq (query)             30_334.24       346.04    30_680.28       0.0891          NaN        11.34
IVF-Binary-512-nl547-np23-rf5-itq (query)             30_334.24       373.02    30_707.26       0.2164    77.762562        11.34
IVF-Binary-512-nl547-np23-rf10-itq (query)            30_334.24       416.75    30_750.99       0.3094    50.174437        11.34
IVF-Binary-512-nl547-np23-rf20-itq (query)            30_334.24       503.10    30_837.34       0.4326    30.843173        11.34
IVF-Binary-512-nl547-np27-rf5-itq (query)             30_334.24       383.74    30_717.98       0.2116    79.780695        11.34
IVF-Binary-512-nl547-np27-rf10-itq (query)            30_334.24       430.21    30_764.45       0.3028    51.693459        11.34
IVF-Binary-512-nl547-np27-rf20-itq (query)            30_334.24       517.89    30_852.14       0.4245    31.885676        11.34
IVF-Binary-512-nl547-np33-rf5-itq (query)             30_334.24       405.74    30_739.98       0.2075    81.744406        11.34
IVF-Binary-512-nl547-np33-rf10-itq (query)            30_334.24       457.57    30_791.81       0.2964    53.264663        11.34
IVF-Binary-512-nl547-np33-rf20-itq (query)            30_334.24       553.81    30_888.05       0.4165    32.980185        11.34
IVF-Binary-512-nl547-itq (self)                       30_334.24     4_131.67    34_465.91       0.3110    49.597960        11.34
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         36_201.55       649.89    36_851.45       0.2364          NaN        20.72
IVF-Binary-1024-nl273-np16-rf0-random (query)         36_201.55       693.34    36_894.89       0.2348          NaN        20.72
IVF-Binary-1024-nl273-np23-rf0-random (query)         36_201.55       813.99    37_015.55       0.2338          NaN        20.72
IVF-Binary-1024-nl273-np13-rf5-random (query)         36_201.55       704.11    36_905.66       0.4941    24.376234        20.72
IVF-Binary-1024-nl273-np13-rf10-random (query)        36_201.55       728.44    36_929.99       0.6364    13.069484        20.72
IVF-Binary-1024-nl273-np13-rf20-random (query)        36_201.55       804.35    37_005.90       0.7779     6.235714        20.72
IVF-Binary-1024-nl273-np16-rf5-random (query)         36_201.55       747.80    36_949.36       0.4913    24.671035        20.72
IVF-Binary-1024-nl273-np16-rf10-random (query)        36_201.55       782.72    36_984.28       0.6337    13.232376        20.72
IVF-Binary-1024-nl273-np16-rf20-random (query)        36_201.55       872.82    37_074.38       0.7754     6.338185        20.72
IVF-Binary-1024-nl273-np23-rf5-random (query)         36_201.55       869.33    37_070.88       0.4896    24.880623        20.72
IVF-Binary-1024-nl273-np23-rf10-random (query)        36_201.55       924.42    37_125.98       0.6320    13.344735        20.72
IVF-Binary-1024-nl273-np23-rf20-random (query)        36_201.55     1_000.34    37_201.90       0.7737     6.415676        20.72
IVF-Binary-1024-nl273-random (self)                   36_201.55     7_793.93    43_995.49       0.6380    13.022033        20.72
IVF-Binary-1024-nl387-np19-rf0-random (query)         38_749.72       651.28    39_401.00       0.2351          NaN        20.84
IVF-Binary-1024-nl387-np27-rf0-random (query)         38_749.72       747.83    39_497.55       0.2342          NaN        20.84
IVF-Binary-1024-nl387-np19-rf5-random (query)         38_749.72       701.35    39_451.07       0.4920    24.593836        20.84
IVF-Binary-1024-nl387-np19-rf10-random (query)        38_749.72       728.84    39_478.56       0.6347    13.179466        20.84
IVF-Binary-1024-nl387-np19-rf20-random (query)        38_749.72       799.31    39_549.03       0.7762     6.307003        20.84
IVF-Binary-1024-nl387-np27-rf5-random (query)         38_749.72       801.94    39_551.66       0.4900    24.815567        20.84
IVF-Binary-1024-nl387-np27-rf10-random (query)        38_749.72       842.64    39_592.36       0.6326    13.322874        20.84
IVF-Binary-1024-nl387-np27-rf20-random (query)        38_749.72       917.80    39_667.52       0.7744     6.387057        20.84
IVF-Binary-1024-nl387-random (self)                   38_749.72     7_279.30    46_029.02       0.6387    12.985396        20.84
IVF-Binary-1024-nl547-np23-rf0-random (query)         42_559.30       627.65    43_186.95       0.2365          NaN        20.99
IVF-Binary-1024-nl547-np27-rf0-random (query)         42_559.30       668.06    43_227.36       0.2356          NaN        20.99
IVF-Binary-1024-nl547-np33-rf0-random (query)         42_559.30       743.90    43_303.20       0.2349          NaN        20.99
IVF-Binary-1024-nl547-np23-rf5-random (query)         42_559.30       680.14    43_239.44       0.4940    24.368460        20.99
IVF-Binary-1024-nl547-np23-rf10-random (query)        42_559.30       707.56    43_266.86       0.6373    13.013508        20.99
IVF-Binary-1024-nl547-np23-rf20-random (query)        42_559.30       774.60    43_333.90       0.7792     6.184769        20.99
IVF-Binary-1024-nl547-np27-rf5-random (query)         42_559.30       717.03    43_276.32       0.4916    24.630557        20.99
IVF-Binary-1024-nl547-np27-rf10-random (query)        42_559.30       745.20    43_304.50       0.6342    13.211604        20.99
IVF-Binary-1024-nl547-np27-rf20-random (query)        42_559.30       813.10    43_372.40       0.7760     6.320150        20.99
IVF-Binary-1024-nl547-np33-rf5-random (query)         42_559.30       768.49    43_327.79       0.4901    24.798033        20.99
IVF-Binary-1024-nl547-np33-rf10-random (query)        42_559.30       805.94    43_365.24       0.6325    13.325582        20.99
IVF-Binary-1024-nl547-np33-rf20-random (query)        42_559.30       873.61    43_432.91       0.7745     6.385337        20.99
IVF-Binary-1024-nl547-random (self)                   42_559.30     7_058.98    49_618.28       0.6414    12.801849        20.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            39_004.27       650.01    39_654.28       0.1992          NaN        20.72
IVF-Binary-1024-nl273-np16-rf0-itq (query)            39_004.27       713.04    39_717.31       0.1972          NaN        20.72
IVF-Binary-1024-nl273-np23-rf0-itq (query)            39_004.27       823.08    39_827.35       0.1958          NaN        20.72
IVF-Binary-1024-nl273-np13-rf5-itq (query)            39_004.27       712.64    39_716.91       0.4171    33.314613        20.72
IVF-Binary-1024-nl273-np13-rf10-itq (query)           39_004.27       731.82    39_736.09       0.5520    18.999806        20.72
IVF-Binary-1024-nl273-np13-rf20-itq (query)           39_004.27       816.69    39_820.96       0.6986     9.904431        20.72
IVF-Binary-1024-nl273-np16-rf5-itq (query)            39_004.27       783.71    39_787.98       0.4135    33.868030        20.72
IVF-Binary-1024-nl273-np16-rf10-itq (query)           39_004.27       800.27    39_804.55       0.5477    19.389505        20.72
IVF-Binary-1024-nl273-np16-rf20-itq (query)           39_004.27       863.84    39_868.11       0.6941    10.153080        20.72
IVF-Binary-1024-nl273-np23-rf5-itq (query)            39_004.27       878.34    39_882.61       0.4113    34.210900        20.72
IVF-Binary-1024-nl273-np23-rf10-itq (query)           39_004.27       921.88    39_926.15       0.5450    19.622469        20.72
IVF-Binary-1024-nl273-np23-rf20-itq (query)           39_004.27     1_007.90    40_012.17       0.6914    10.293050        20.72
IVF-Binary-1024-nl273-itq (self)                      39_004.27     7_930.18    46_934.45       0.5500    19.224281        20.72
IVF-Binary-1024-nl387-np19-rf0-itq (query)            41_820.95       666.12    42_487.08       0.1974          NaN        20.84
IVF-Binary-1024-nl387-np27-rf0-itq (query)            41_820.95       770.57    42_591.53       0.1963          NaN        20.84
IVF-Binary-1024-nl387-np19-rf5-itq (query)            41_820.95       716.38    42_537.34       0.4146    33.714378        20.84
IVF-Binary-1024-nl387-np19-rf10-itq (query)           41_820.95       746.71    42_567.66       0.5489    19.281358        20.84
IVF-Binary-1024-nl387-np19-rf20-itq (query)           41_820.95       813.59    42_634.54       0.6950    10.097759        20.84
IVF-Binary-1024-nl387-np27-rf5-itq (query)            41_820.95       833.45    42_654.40       0.4121    34.132991        20.84
IVF-Binary-1024-nl387-np27-rf10-itq (query)           41_820.95       859.43    42_680.39       0.5457    19.578459        20.84
IVF-Binary-1024-nl387-np27-rf20-itq (query)           41_820.95       926.34    42_747.30       0.6916    10.274577        20.84
IVF-Binary-1024-nl387-itq (self)                      41_820.95     7_464.16    49_285.12       0.5511    19.133173        20.84
IVF-Binary-1024-nl547-np23-rf0-itq (query)            45_717.78       665.21    46_382.99       0.1980          NaN        20.99
IVF-Binary-1024-nl547-np27-rf0-itq (query)            45_717.78       703.04    46_420.82       0.1969          NaN        20.99
IVF-Binary-1024-nl547-np33-rf0-itq (query)            45_717.78       768.24    46_486.02       0.1960          NaN        20.99
IVF-Binary-1024-nl547-np23-rf5-itq (query)            45_717.78       736.25    46_454.03       0.4166    33.377499        20.99
IVF-Binary-1024-nl547-np23-rf10-itq (query)           45_717.78       758.09    46_475.87       0.5522    18.992405        20.99
IVF-Binary-1024-nl547-np23-rf20-itq (query)           45_717.78       833.88    46_551.66       0.6993     9.885410        20.99
IVF-Binary-1024-nl547-np27-rf5-itq (query)            45_717.78       797.73    46_515.51       0.4139    33.788404        20.99
IVF-Binary-1024-nl547-np27-rf10-itq (query)           45_717.78       788.17    46_505.95       0.5483    19.330194        20.99
IVF-Binary-1024-nl547-np27-rf20-itq (query)           45_717.78       859.55    46_577.34       0.6949    10.110196        20.99
IVF-Binary-1024-nl547-np33-rf5-itq (query)            45_717.78       819.63    46_537.41       0.4120    34.089375        20.99
IVF-Binary-1024-nl547-np33-rf10-itq (query)           45_717.78       851.50    46_569.29       0.5457    19.543360        20.99
IVF-Binary-1024-nl547-np33-rf20-itq (query)           45_717.78       912.73    46_630.52       0.6922    10.251824        20.99
IVF-Binary-1024-nl547-itq (self)                      45_717.78     7_400.97    53_118.75       0.5544    18.862208        20.99
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-signed (query)          14_359.40       146.29    14_505.69       0.0972          NaN         6.24
IVF-Binary-256-nl273-np16-rf0-signed (query)          14_359.40       162.37    14_521.76       0.0901          NaN         6.24
IVF-Binary-256-nl273-np23-rf0-signed (query)          14_359.40       180.29    14_539.68       0.0878          NaN         6.24
IVF-Binary-256-nl273-np13-rf5-signed (query)          14_359.40       204.45    14_563.85       0.2201    76.967321         6.24
IVF-Binary-256-nl273-np13-rf10-signed (query)         14_359.40       253.49    14_612.88       0.3105    49.885639         6.24
IVF-Binary-256-nl273-np13-rf20-signed (query)         14_359.40       340.01    14_699.41       0.4337    30.695358         6.24
IVF-Binary-256-nl273-np16-rf5-signed (query)          14_359.40       213.15    14_572.55       0.2094    81.402596         6.24
IVF-Binary-256-nl273-np16-rf10-signed (query)         14_359.40       264.67    14_624.07       0.2989    52.738843         6.24
IVF-Binary-256-nl273-np16-rf20-signed (query)         14_359.40       365.28    14_724.67       0.4208    32.487636         6.24
IVF-Binary-256-nl273-np23-rf5-signed (query)          14_359.40       245.78    14_605.17       0.2042    83.447553         6.24
IVF-Binary-256-nl273-np23-rf10-signed (query)         14_359.40       281.80    14_641.20       0.2917    54.370157         6.24
IVF-Binary-256-nl273-np23-rf20-signed (query)         14_359.40       385.96    14_745.35       0.4128    33.552199         6.24
IVF-Binary-256-nl273-signed (self)                    14_359.40     2_653.07    17_012.47       0.3025    52.077464         6.24
IVF-Binary-256-nl387-np19-rf0-signed (query)          17_044.76       155.81    17_200.57       0.0916          NaN         6.35
IVF-Binary-256-nl387-np27-rf0-signed (query)          17_044.76       167.92    17_212.69       0.0886          NaN         6.35
IVF-Binary-256-nl387-np19-rf5-signed (query)          17_044.76       216.66    17_261.42       0.2128    79.556601         6.35
IVF-Binary-256-nl387-np19-rf10-signed (query)         17_044.76       271.43    17_316.19       0.3028    51.326124         6.35
IVF-Binary-256-nl387-np19-rf20-signed (query)         17_044.76       354.36    17_399.12       0.4263    31.560848         6.35
IVF-Binary-256-nl387-np27-rf5-signed (query)          17_044.76       218.98    17_263.74       0.2070    82.135294         6.35
IVF-Binary-256-nl387-np27-rf10-signed (query)         17_044.76       277.88    17_322.64       0.2953    53.161692         6.35
IVF-Binary-256-nl387-np27-rf20-signed (query)         17_044.76       374.44    17_419.21       0.4166    32.853596         6.35
IVF-Binary-256-nl387-signed (self)                    17_044.76     2_616.82    19_661.58       0.3058    50.839344         6.35
IVF-Binary-256-nl547-np23-rf0-signed (query)          21_171.99       164.45    21_336.44       0.0929          NaN         6.51
IVF-Binary-256-nl547-np27-rf0-signed (query)          21_171.99       169.11    21_341.10       0.0909          NaN         6.51
IVF-Binary-256-nl547-np33-rf0-signed (query)          21_171.99       181.44    21_353.42       0.0888          NaN         6.51
IVF-Binary-256-nl547-np23-rf5-signed (query)          21_171.99       223.41    21_395.39       0.2173    76.797431         6.51
IVF-Binary-256-nl547-np23-rf10-signed (query)         21_171.99       281.94    21_453.92       0.3098    49.339898         6.51
IVF-Binary-256-nl547-np23-rf20-signed (query)         21_171.99       359.40    21_531.39       0.4358    30.124653         6.51
IVF-Binary-256-nl547-np27-rf5-signed (query)          21_171.99       224.22    21_396.21       0.2128    78.810235         6.51
IVF-Binary-256-nl547-np27-rf10-signed (query)         21_171.99       285.67    21_457.66       0.3033    50.850227         6.51
IVF-Binary-256-nl547-np27-rf20-signed (query)         21_171.99       372.10    21_544.09       0.4273    31.222115         6.51
IVF-Binary-256-nl547-np33-rf5-signed (query)          21_171.99       227.32    21_399.30       0.2084    80.866425         6.51
IVF-Binary-256-nl547-np33-rf10-signed (query)         21_171.99       286.29    21_458.28       0.2972    52.390333         6.51
IVF-Binary-256-nl547-np33-rf20-signed (query)         21_171.99       383.56    21_555.54       0.4199    32.257029         6.51
IVF-Binary-256-nl547-signed (self)                    21_171.99     2_677.20    23_849.19       0.3136    48.731391         6.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

<details>
<summary><b>Binary - Euclidean (LowRank - 512 bit)</b>:</summary>
<pre><code>

</code></pre>
</details>

### <u>RaBitQ (IVF and exhaustive)</u>

[RaBitQ](https://arxiv.org/abs/2405.12497) is an incredibly powerful
quantisation that combines strong compression with excellent Recalls (even
without re-ranking). It works better on higher dimensions. In the case of the
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
Exhaustive (query)                                         3.05     1_460.42     1_463.46       1.0000     0.000000        18.31
Exhaustive (self)                                          3.05    15_256.07    15_259.12       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_612.46       498.80     2_111.27       0.3141    41.590412         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_612.46       582.49     2_194.96       0.6781     1.424540         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_612.46       653.35     2_265.82       0.8317     0.540476         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_612.46       784.15     2_396.61       0.9366     0.159663         3.46
ExhaustiveRaBitQ (self)                                1_612.46     6_572.60     8_185.06       0.8323     0.527515         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        887.36       116.23     1_003.60       0.3203    41.573351         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        887.36       138.65     1_026.01       0.3193    41.580391         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        887.36       196.67     1_084.03       0.3181    41.584016         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        887.36       169.16     1_056.52       0.6869     1.446372         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       887.36       214.60     1_101.96       0.8370     0.558086         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       887.36       298.68     1_186.04       0.9352     0.179393         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        887.36       193.43     1_080.79       0.6866     1.449363         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       887.36       245.23     1_132.59       0.8392     0.547526         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       887.36       334.16     1_221.53       0.9397     0.160331         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        887.36       251.11     1_138.47       0.6853     1.459546         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       887.36       305.64     1_193.00       0.8388     0.549881         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       887.36       399.35     1_286.71       0.9407     0.158858         3.47
IVF-RaBitQ-nl273 (self)                                  887.36     4_003.74     4_891.11       0.9411     0.156272         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_235.16       122.05     1_357.21       0.3213    41.562971         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_235.16       165.15     1_400.31       0.3189    41.572592         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_235.16       172.86     1_408.02       0.6916     1.395349         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_235.16       221.18     1_456.34       0.8430     0.518563         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_235.16       304.57     1_539.73       0.9397     0.166007         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_235.16       219.22     1_454.38       0.6888     1.418668         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_235.16       290.95     1_526.11       0.8425     0.516610         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_235.16       370.35     1_605.51       0.9428     0.146618         3.49
IVF-RaBitQ-nl387 (self)                                1_235.16     3_600.06     4_835.22       0.9429     0.146906         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_678.98       110.48     1_789.46       0.3271    41.545554         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_678.98       130.05     1_809.03       0.3252    41.553698         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_678.98       151.52     1_830.50       0.3237    41.559146         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_678.98       164.03     1_843.02       0.6983     1.345744         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_678.98       206.69     1_885.67       0.8470     0.505449         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_678.98       287.76     1_966.74       0.9405     0.167264         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_678.98       177.56     1_856.54       0.6962     1.356471         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_678.98       226.92     1_905.90       0.8474     0.498092         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_678.98       335.99     2_014.97       0.9440     0.148206         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_678.98       204.10     1_883.08       0.6943     1.372707         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_678.98       253.13     1_932.11       0.8465     0.502418         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_678.98       342.24     2_021.22       0.9451     0.142254         3.51
IVF-RaBitQ-nl547 (self)                                1_678.98     3_418.41     5_097.39       0.9453     0.139848         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Cosine (Gaussian)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         4.02     1_568.91     1_572.93       1.0000     0.000000        18.88
Exhaustive (self)                                          4.02    16_397.88    16_401.90       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_645.64       439.88     2_085.52       0.3171     0.168426         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_645.64       517.42     2_163.06       0.6841     0.001054         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_645.64       584.27     2_229.92       0.8360     0.000399         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_645.64       710.54     2_356.18       0.9403     0.000113         3.46
ExhaustiveRaBitQ (self)                                1_645.64     5_871.03     7_516.67       0.8379     0.000387         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        935.32       116.11     1_051.43       0.3166     0.167983         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        935.32       140.01     1_075.33       0.3153     0.167689         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        935.32       196.61     1_131.93       0.3141     0.167541         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        935.32       170.96     1_106.28       0.6843     0.001130         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       935.32       233.99     1_169.31       0.8363     0.000436         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       935.32       299.23     1_234.55       0.9348     0.000143         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        935.32       197.48     1_132.81       0.6839     0.001136         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       935.32       245.74     1_181.06       0.8379     0.000431         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       935.32       335.92     1_271.24       0.9392     0.000129         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        935.32       252.77     1_188.10       0.6825     0.001145         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       935.32       314.12     1_249.44       0.8374     0.000432         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       935.32       406.36     1_341.68       0.9404     0.000127         3.47
IVF-RaBitQ-nl273 (self)                                  935.32     4_044.64     4_979.96       0.9409     0.000126         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_267.88       121.01     1_388.89       0.3204     0.168497         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_267.88       169.98     1_437.86       0.3182     0.168130         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_267.88       175.56     1_443.44       0.6892     0.001096         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_267.88       223.26     1_491.14       0.8406     0.000417         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_267.88       307.41     1_575.29       0.9393     0.000130         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_267.88       221.62     1_489.50       0.6865     0.001114         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_267.88       273.52     1_541.40       0.8400     0.000420         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_267.88       364.50     1_632.38       0.9418     0.000121         3.49
IVF-RaBitQ-nl387 (self)                                1_267.88     3_635.97     4_903.85       0.9432     0.000118         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_742.07       110.70     1_852.77       0.3221     0.169055         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_742.07       127.90     1_869.97       0.3198     0.168735         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_742.07       161.46     1_903.53       0.3183     0.168529         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_742.07       161.28     1_903.35       0.6944     0.001058         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_742.07       205.95     1_948.02       0.8444     0.000403         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_742.07       288.58     2_030.65       0.9408     0.000129         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_742.07       178.61     1_920.68       0.6921     0.001072         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_742.07       227.26     1_969.33       0.8444     0.000402         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_742.07       324.51     2_066.58       0.9435     0.000120         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_742.07       205.32     1_947.39       0.6894     0.001088         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_742.07       254.88     1_996.95       0.8431     0.000407         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_742.07       344.08     2_086.15       0.9442     0.000118         3.51
IVF-RaBitQ-nl547 (self)                                1_742.07     3_424.67     5_166.74       0.9452     0.000112         3.51
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
Exhaustive (query)                                         3.16     1_493.38     1_496.54       1.0000     0.000000        18.31
Exhaustive (self)                                          3.16    17_122.25    17_125.41       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_788.69       400.86     2_189.56       0.4425     1.402784         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_788.69       481.77     2_270.46       0.8784     0.030819         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_788.69       546.86     2_335.55       0.9681     0.005683         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_788.69       648.44     2_437.13       0.9954     0.000661         3.46
ExhaustiveRaBitQ (self)                                1_788.69     5_382.75     7_171.44       0.9697     0.005495         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        956.66       115.10     1_071.75       0.4624     1.393407         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        956.66       149.41     1_106.06       0.4620     1.393562         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        956.66       202.11     1_158.77       0.4619     1.393619         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        956.66       166.48     1_123.13       0.8931     0.028100         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       956.66       206.97     1_163.62       0.9745     0.004925         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       956.66       284.28     1_240.94       0.9968     0.000521         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        956.66       197.12     1_153.78       0.8928     0.028226         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       956.66       255.80     1_212.45       0.9744     0.004949         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       956.66       327.38     1_284.04       0.9969     0.000502         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        956.66       259.61     1_216.26       0.8926     0.028290         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       956.66       316.65     1_273.31       0.9743     0.004969         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       956.66       406.39     1_363.05       0.9969     0.000504         3.47
IVF-RaBitQ-nl273 (self)                                  956.66     4_098.80     5_055.46       0.9973     0.000448         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_309.58       119.94     1_429.52       0.4729     1.382070         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_309.58       165.25     1_474.84       0.4728     1.382133         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_309.58       172.55     1_482.13       0.9030     0.024206         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_309.58       217.71     1_527.29       0.9778     0.004107         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_309.58       293.45     1_603.03       0.9976     0.000386         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_309.58       222.04     1_531.62       0.9028     0.024248         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_309.58       280.01     1_589.59       0.9778     0.004118         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_309.58       361.20     1_670.78       0.9976     0.000376         3.49
IVF-RaBitQ-nl387 (self)                                1_309.58     3_540.95     4_850.54       0.9978     0.000345         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_734.49       111.32     1_845.81       0.4837     1.373040         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_734.49       128.36     1_862.85       0.4835     1.373146         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_734.49       151.27     1_885.76       0.4834     1.373184         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_734.49       162.96     1_897.45       0.9115     0.021036         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_734.49       214.15     1_948.64       0.9815     0.003284         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_734.49       278.84     2_013.33       0.9980     0.000305         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_734.49       178.02     1_912.51       0.9112     0.021118         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_734.49       225.60     1_960.09       0.9813     0.003309         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_734.49       300.62     2_035.11       0.9980     0.000300         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_734.49       205.80     1_940.29       0.9111     0.021140         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_734.49       254.37     1_988.86       0.9813     0.003314         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_734.49       340.91     2_075.41       0.9980     0.000294         3.51
IVF-RaBitQ-nl547 (self)                                1_734.49     3_396.52     5_131.01       0.9982     0.000269         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.36     1_499.93     1_503.29       1.0000     0.000000        18.31
Exhaustive (self)                                          3.36    16_808.46    16_811.82       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_595.82       408.42     2_004.24       0.4782     7.158297         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_595.82       489.01     2_084.83       0.9142     0.089854         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_595.82       540.89     2_136.71       0.9836     0.012777         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_595.82       643.06     2_238.88       0.9985     0.001012         3.46
ExhaustiveRaBitQ (self)                                1_595.82     5_440.57     7_036.39       0.9834     0.013140         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        883.65       114.36       998.01       0.4887     7.133549         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        883.65       132.64     1_016.28       0.4887     7.133580         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        883.65       185.89     1_069.54       0.4887     7.133584         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        883.65       161.79     1_045.44       0.9216     0.078073         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       883.65       204.27     1_087.91       0.9858     0.010544         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       883.65       276.96     1_160.61       0.9988     0.000755         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        883.65       188.53     1_072.18       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       883.65       237.69     1_121.34       0.9857     0.010553         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       883.65       317.10     1_200.75       0.9988     0.000741         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        883.65       243.22     1_126.87       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       883.65       317.42     1_201.07       0.9857     0.010558         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       883.65       381.59     1_265.24       0.9988     0.000741         3.47
IVF-RaBitQ-nl273 (self)                                  883.65     3_847.28     4_730.93       0.9988     0.000729         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_204.05       117.80     1_321.86       0.5004     7.112936         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_204.05       162.78     1_366.83       0.5004     7.112938         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_204.05       169.52     1_373.57       0.9290     0.068039         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_204.05       224.19     1_428.24       0.9875     0.009140         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_204.05       293.85     1_497.90       0.9991     0.000516         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_204.05       220.68     1_424.73       0.9290     0.068061         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_204.05       266.04     1_470.09       0.9875     0.009147         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_204.05       365.03     1_569.08       0.9991     0.000516         3.49
IVF-RaBitQ-nl387 (self)                                1_204.05     3_472.04     4_676.10       0.9991     0.000578         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_772.14       111.53     1_883.67       0.5107     7.089673         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_772.14       126.61     1_898.75       0.5107     7.089686         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_772.14       160.62     1_932.76       0.5107     7.089690         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_772.14       160.58     1_932.72       0.9363     0.058698         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_772.14       200.78     1_972.92       0.9896     0.007452         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_772.14       273.22     2_045.36       0.9994     0.000388         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_772.14       185.45     1_957.59       0.9362     0.058740         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_772.14       220.50     1_992.64       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_772.14       294.24     2_066.38       0.9994     0.000393         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_772.14       204.14     1_976.28       0.9362     0.058740         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_772.14       248.25     2_020.39       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_772.14       329.76     2_101.90       0.9994     0.000393         3.51
IVF-RaBitQ-nl547 (self)                                1_772.14     3_270.37     5_042.51       0.9993     0.000388         3.51
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
Exhaustive (query)                                        14.76     6_086.26     6_101.01       1.0000     0.000000        73.24
Exhaustive (self)                                         14.76    62_538.97    62_553.72       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           5_063.39       848.80     5_912.20       0.7024    73.760064         5.31
ExhaustiveRaBitQ-rf5 (query)                           5_063.39       933.67     5_997.06       0.9952     0.017379         5.31
ExhaustiveRaBitQ-rf10 (query)                          5_063.39     1_087.12     6_150.51       0.9999     0.000387         5.31
ExhaustiveRaBitQ-rf20 (query)                          5_063.39     1_139.36     6_202.76       1.0000     0.000000         5.31
ExhaustiveRaBitQ (self)                                5_063.39    10_085.51    15_148.90       0.9999     0.000415         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      3_190.64       241.77     3_432.41       0.7066    73.750602         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      3_190.64       291.55     3_482.19       0.7066    73.750778         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      3_190.64       404.23     3_594.87       0.7066    73.750785         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      3_190.64       303.68     3_494.32       0.9954     0.017287         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     3_190.64       347.65     3_538.29       0.9996     0.002368         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     3_190.64       440.57     3_631.21       0.9997     0.002034         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      3_190.64       351.21     3_541.85       0.9956     0.015465         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     3_190.64       406.61     3_597.25       0.9999     0.000515         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     3_190.64       503.68     3_694.32       1.0000     0.000182         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      3_190.64       461.88     3_652.52       0.9956     0.015283         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     3_190.64       533.07     3_723.71       0.9999     0.000333         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     3_190.64       626.84     3_817.48       1.0000     0.000000         5.35
IVF-RaBitQ-nl273 (self)                                3_190.64     6_294.48     9_485.12       1.0000     0.000001         5.35
IVF-RaBitQ-nl387-np19-rf0 (query)                      4_318.07       273.92     4_591.99       0.7122    73.742349         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      4_318.07       374.87     4_692.94       0.7122    73.742370         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      4_318.07       337.30     4_655.37       0.9959     0.014264         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     4_318.07       385.36     4_703.43       0.9999     0.000347         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     4_318.07       475.86     4_793.93       1.0000     0.000111         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      4_318.07       437.93     4_756.01       0.9959     0.014164         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     4_318.07       491.34     4_809.41       0.9999     0.000236         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     4_318.07       586.65     4_904.72       1.0000     0.000000         5.41
IVF-RaBitQ-nl387 (self)                                4_318.07     5_868.83    10_186.91       1.0000     0.000004         5.41
IVF-RaBitQ-nl547-np23-rf0 (query)                      5_865.14       272.97     6_138.11       0.7162    73.731250         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      5_865.14       329.32     6_194.46       0.7162    73.731339         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      5_865.14       376.39     6_241.53       0.7162    73.731351         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      5_865.14       329.59     6_194.73       0.9962     0.013968         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     5_865.14       381.39     6_246.53       0.9998     0.000954         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     5_865.14       469.31     6_334.45       0.9999     0.000615         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      5_865.14       370.34     6_235.48       0.9963     0.013478         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     5_865.14       422.94     6_288.08       0.9999     0.000400         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     5_865.14       513.08     6_378.22       1.0000     0.000062         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      5_865.14       439.64     6_304.78       0.9963     0.013416         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     5_865.14       488.71     6_353.85       0.9999     0.000339         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     5_865.14       582.03     6_447.17       1.0000     0.000000         5.49
IVF-RaBitQ-nl547 (self)                                5_865.14     5_809.21    11_674.35       1.0000     0.000001         5.49
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
Exhaustive (query)                                        30.19    15_393.70    15_423.89       1.0000     0.000000       146.48
Exhaustive (self)                                         30.19   155_528.47   155_558.66       1.0000     0.000000       146.48
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                          12_299.08     1_851.72    14_150.80       0.7810   172.768130         7.88
ExhaustiveRaBitQ-rf5 (query)                          12_299.08     1_945.22    14_244.30       0.9996     0.002470         7.88
ExhaustiveRaBitQ-rf10 (query)                         12_299.08     2_007.95    14_307.03       1.0000     0.000000         7.88
ExhaustiveRaBitQ-rf20 (query)                         12_299.08     2_215.94    14_515.02       1.0000     0.000000         7.88
ExhaustiveRaBitQ (self)                               12_299.08    20_221.47    32_520.55       1.0000     0.000001         7.88
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      8_127.95       557.40     8_685.35       0.7846   172.759323         7.96
IVF-RaBitQ-nl273-np16-rf0 (query)                      8_127.95       676.08     8_804.03       0.7846   172.759636         7.96
IVF-RaBitQ-nl273-np23-rf0 (query)                      8_127.95       945.70     9_073.65       0.7846   172.759648         7.96
IVF-RaBitQ-nl273-np13-rf5 (query)                      8_127.95       627.62     8_755.57       0.9994     0.007643         7.96
IVF-RaBitQ-nl273-np13-rf10 (query)                     8_127.95       665.29     8_793.24       0.9997     0.006296         7.96
IVF-RaBitQ-nl273-np13-rf20 (query)                     8_127.95       776.39     8_904.33       0.9997     0.006296         7.96
IVF-RaBitQ-nl273-np16-rf5 (query)                      8_127.95       721.62     8_849.57       0.9997     0.001407         7.96
IVF-RaBitQ-nl273-np16-rf10 (query)                     8_127.95       786.16     8_914.11       1.0000     0.000060         7.96
IVF-RaBitQ-nl273-np16-rf20 (query)                     8_127.95       903.84     9_031.79       1.0000     0.000060         7.96
IVF-RaBitQ-nl273-np23-rf5 (query)                      8_127.95     1_007.17     9_135.12       0.9997     0.001346         7.96
IVF-RaBitQ-nl273-np23-rf10 (query)                     8_127.95     1_056.48     9_184.42       1.0000     0.000000         7.96
IVF-RaBitQ-nl273-np23-rf20 (query)                     8_127.95     1_178.94     9_306.89       1.0000     0.000000         7.96
IVF-RaBitQ-nl273 (self)                                8_127.95    11_740.28    19_868.23       1.0000     0.000000         7.96
IVF-RaBitQ-nl387-np19-rf0 (query)                     10_662.97       661.53    11_324.49       0.7875   172.751088         8.07
IVF-RaBitQ-nl387-np27-rf0 (query)                     10_662.97       935.85    11_598.82       0.7875   172.751111         8.07
IVF-RaBitQ-nl387-np19-rf5 (query)                     10_662.97       780.94    11_443.91       0.9997     0.002281         8.07
IVF-RaBitQ-nl387-np19-rf10 (query)                    10_662.97       789.81    11_452.78       1.0000     0.000194         8.07
IVF-RaBitQ-nl387-np19-rf20 (query)                    10_662.97       902.74    11_565.71       1.0000     0.000194         8.07
IVF-RaBitQ-nl387-np27-rf5 (query)                     10_662.97       989.10    11_652.07       0.9997     0.002087         8.07
IVF-RaBitQ-nl387-np27-rf10 (query)                    10_662.97     1_059.78    11_722.75       1.0000     0.000000         8.07
IVF-RaBitQ-nl387-np27-rf20 (query)                    10_662.97     1_176.44    11_839.41       1.0000     0.000000         8.07
IVF-RaBitQ-nl387 (self)                               10_662.97    11_843.23    22_506.20       1.0000     0.000000         8.07
IVF-RaBitQ-nl547-np23-rf0 (query)                     14_256.51       714.90    14_971.42       0.7907   172.743226         8.23
IVF-RaBitQ-nl547-np27-rf0 (query)                     14_256.51       840.80    15_097.31       0.7907   172.743359         8.23
IVF-RaBitQ-nl547-np33-rf0 (query)                     14_256.51       996.51    15_253.03       0.7907   172.743373         8.23
IVF-RaBitQ-nl547-np23-rf5 (query)                     14_256.51       779.45    15_035.96       0.9996     0.003517         8.23
IVF-RaBitQ-nl547-np23-rf10 (query)                    14_256.51       842.98    15_099.50       0.9998     0.002294         8.23
IVF-RaBitQ-nl547-np23-rf20 (query)                    14_256.51       950.55    15_207.06       0.9998     0.002294         8.23
IVF-RaBitQ-nl547-np27-rf5 (query)                     14_256.51       886.51    15_143.02       0.9997     0.001378         8.23
IVF-RaBitQ-nl547-np27-rf10 (query)                    14_256.51       948.97    15_205.49       1.0000     0.000155         8.23
IVF-RaBitQ-nl547-np27-rf20 (query)                    14_256.51     1_066.23    15_322.74       1.0000     0.000155         8.23
IVF-RaBitQ-nl547-np33-rf5 (query)                     14_256.51     1_067.66    15_324.17       0.9998     0.001223         8.23
IVF-RaBitQ-nl547-np33-rf10 (query)                    14_256.51     1_125.31    15_381.82       1.0000     0.000000         8.23
IVF-RaBitQ-nl547-np33-rf20 (query)                    14_256.51     1_238.80    15_495.31       1.0000     0.000000         8.23
IVF-RaBitQ-nl547 (self)                               14_256.51    12_548.18    26_804.69       1.0000     0.000000         8.23
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
Exhaustive (query)                                        63.22    33_090.29    33_153.51       1.0000     0.000000       292.97
Exhaustive (self)                                         63.22   337_093.46   337_156.68       1.0000     0.000000       292.97
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                          28_405.84     4_876.94    33_282.78       0.8406   370.192195        13.40
ExhaustiveRaBitQ-rf5 (query)                          28_405.84     4_931.65    33_337.49       0.9994     0.089933        13.40
ExhaustiveRaBitQ-rf10 (query)                         28_405.84     5_083.22    33_489.06       0.9995     0.054206        13.40
ExhaustiveRaBitQ-rf20 (query)                         28_405.84     5_182.29    33_588.13       0.9995     0.040006        13.40
ExhaustiveRaBitQ (self)                               28_405.84    50_010.16    78_416.00       0.9995     0.061672        13.40
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                     19_975.15     1_519.17    21_494.32       0.8399   370.187701        13.55
IVF-RaBitQ-nl273-np16-rf0 (query)                     19_975.15     1_848.80    21_823.95       0.8400   370.188073        13.55
IVF-RaBitQ-nl273-np23-rf0 (query)                     19_975.15     2_716.44    22_691.59       0.8401   370.188098        13.55
IVF-RaBitQ-nl273-np13-rf5 (query)                     19_975.15     1_633.29    21_608.43       0.9991     0.080732        13.55
IVF-RaBitQ-nl273-np13-rf10 (query)                    19_975.15     1_777.59    21_752.73       0.9992     0.062207        13.55
IVF-RaBitQ-nl273-np13-rf20 (query)                    19_975.15     1_837.18    21_812.33       0.9992     0.048362        13.55
IVF-RaBitQ-nl273-np16-rf5 (query)                     19_975.15     1_968.52    21_943.67       0.9995     0.066882        13.55
IVF-RaBitQ-nl273-np16-rf10 (query)                    19_975.15     2_041.14    22_016.29       0.9995     0.048358        13.55
IVF-RaBitQ-nl273-np16-rf20 (query)                    19_975.15     2_159.63    22_134.78       0.9996     0.034512        13.55
IVF-RaBitQ-nl273-np23-rf5 (query)                     19_975.15     2_736.44    22_711.59       0.9995     0.065903        13.55
IVF-RaBitQ-nl273-np23-rf10 (query)                    19_975.15     2_786.70    22_761.84       0.9996     0.047379        13.55
IVF-RaBitQ-nl273-np23-rf20 (query)                    19_975.15     2_955.18    22_930.33       0.9996     0.033533        13.55
IVF-RaBitQ-nl273 (self)                               19_975.15    29_372.17    49_347.31       0.9996     0.033674        13.55
IVF-RaBitQ-nl387-np19-rf0 (query)                     25_737.90     1_993.58    27_731.48       0.8445   370.182610        13.78
IVF-RaBitQ-nl387-np27-rf0 (query)                     25_737.90     2_867.06    28_604.96       0.8446   370.182645        13.78
IVF-RaBitQ-nl387-np19-rf5 (query)                     25_737.90     2_052.67    27_790.57       0.9996     0.060151        13.78
IVF-RaBitQ-nl387-np19-rf10 (query)                    25_737.90     2_123.14    27_861.04       0.9996     0.034682        13.78
IVF-RaBitQ-nl387-np19-rf20 (query)                    25_737.90     2_262.61    28_000.51       0.9997     0.028196        13.78
IVF-RaBitQ-nl387-np27-rf5 (query)                     25_737.90     2_854.45    28_592.35       0.9996     0.058819        13.78
IVF-RaBitQ-nl387-np27-rf10 (query)                    25_737.90     2_933.21    28_671.11       0.9997     0.033350        13.78
IVF-RaBitQ-nl387-np27-rf20 (query)                    25_737.90     3_085.46    28_823.36       0.9997     0.026864        13.78
IVF-RaBitQ-nl387 (self)                               25_737.90    30_735.87    56_473.77       0.9996     0.029224        13.78
IVF-RaBitQ-nl547-np23-rf0 (query)                     33_520.85     2_256.21    35_777.06       0.8466   370.174142        14.09
IVF-RaBitQ-nl547-np27-rf0 (query)                     33_520.85     2_665.21    36_186.06       0.8466   370.174303        14.09
IVF-RaBitQ-nl547-np33-rf0 (query)                     33_520.85     3_186.38    36_707.23       0.8466   370.174316        14.09
IVF-RaBitQ-nl547-np23-rf5 (query)                     33_520.85     2_356.48    35_877.33       0.9993     0.047254        14.09
IVF-RaBitQ-nl547-np23-rf10 (query)                    33_520.85     2_388.92    35_909.77       0.9994     0.036903        14.09
IVF-RaBitQ-nl547-np23-rf20 (query)                    33_520.85     2_514.16    36_035.01       0.9995     0.029689        14.09
IVF-RaBitQ-nl547-np27-rf5 (query)                     33_520.85     2_677.80    36_198.65       0.9996     0.042023        14.09
IVF-RaBitQ-nl547-np27-rf10 (query)                    33_520.85     2_758.57    36_279.42       0.9996     0.031672        14.09
IVF-RaBitQ-nl547-np27-rf20 (query)                    33_520.85     2_890.11    36_410.96       0.9997     0.024458        14.09
IVF-RaBitQ-nl547-np33-rf5 (query)                     33_520.85     3_241.51    36_762.37       0.9996     0.041716        14.09
IVF-RaBitQ-nl547-np33-rf10 (query)                    33_520.85     3_312.64    36_833.49       0.9996     0.031366        14.09
IVF-RaBitQ-nl547-np33-rf20 (query)                    33_520.85     3_463.37    36_984.23       0.9997     0.024151        14.09
IVF-RaBitQ-nl547 (self)                               33_520.85    34_473.09    67_993.95       0.9997     0.024876        14.09
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>


Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls.

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
