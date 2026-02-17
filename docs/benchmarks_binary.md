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
Exhaustive (query)                                         3.07     1_514.32     1_517.39       1.0000     0.000000        18.31
Exhaustive (self)                                          3.07    15_327.17    15_330.23       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_162.91       503.56     1_666.47       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_162.91       536.39     1_699.30       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_162.91       596.22     1_759.13       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_162.91       690.72     1_853.63       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                     1_162.91     5_935.32     7_098.23       0.5461     2.056589         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_365.25       501.12     1_866.37       0.1873          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_365.25       545.08     1_910.33       0.3844     4.171681         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_365.25       599.88     1_965.13       0.5073     2.366457         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_365.25       703.54     2_068.79       0.6439     1.245957         4.61
ExhaustiveBinary-256-itq (self)                        1_365.25     5_940.91     7_306.15       0.5079     2.339769         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_318.82       839.58     3_158.40       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_318.82       965.75     3_284.57       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_318.82       938.35     3_257.16       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_318.82     1_039.42     3_358.23       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                     2_318.82     9_272.00    11_590.82       0.7223     0.828915         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_600.58       830.05     3_430.62       0.2823          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_600.58       877.29     3_477.86       0.5675     1.840282         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_600.58       948.19     3_548.77       0.7062     0.870974         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_600.58     1_048.17     3_648.75       0.8300     0.361878         9.22
ExhaustiveBinary-512-itq (self)                        2_600.58     9_415.92    12_016.49       0.7065     0.858252         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 160.92       613.79       774.71       0.0578          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   160.92       655.44       816.36       0.1270    15.329844         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  160.92       693.21       854.13       0.1824    10.683378         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  160.92       768.35       929.26       0.2652     7.182754         0.58
ExhaustiveBinary-32-signed (self)                        160.92     6_935.26     7_096.17       0.1841    10.412094         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_131.95        57.59     2_189.53       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_131.95        66.46     2_198.41       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_131.95        81.00     2_212.94       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_131.95        90.99     2_222.94       0.4239     3.637501         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_131.95       119.33     2_251.28       0.5531     2.021353         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_131.95       179.26     2_311.21       0.6927     1.020556         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_131.95       102.55     2_234.50       0.4223     3.661802         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_131.95       124.14     2_256.09       0.5514     2.033747         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_131.95       176.94     2_308.89       0.6913     1.025397         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_131.95       109.26     2_241.21       0.4198     3.692131         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_131.95       140.66     2_272.60       0.5484     2.057348         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_131.95       204.18     2_336.12       0.6879     1.040413         5.79
IVF-Binary-256-nl273-random (self)                     2_131.95     1_217.26     3_349.21       0.5511     2.018663         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_465.52        55.09     2_520.62       0.2061          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_465.52        66.55     2_532.07       0.2047          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_465.52        84.83     2_550.35       0.4238     3.634205         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_465.52       114.15     2_579.67       0.5532     2.014085         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_465.52       161.44     2_626.96       0.6934     1.014780         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_465.52       102.34     2_567.86       0.4203     3.683698         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_465.52       136.76     2_602.28       0.5495     2.045078         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_465.52       185.59     2_651.11       0.6888     1.036117         5.80
IVF-Binary-256-nl387-random (self)                     2_465.52     1_141.62     3_607.15       0.5531     1.998615         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_933.36        52.47     2_985.83       0.2076          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_933.36        55.67     2_989.04       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_933.36        62.80     2_996.16       0.2050          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_933.36        78.91     3_012.28       0.4265     3.596401         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_933.36       105.18     3_038.54       0.5576     1.983148         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_933.36       150.09     3_083.45       0.6977     0.994532         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_933.36        83.65     3_017.02       0.4238     3.631225         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_933.36       114.00     3_047.36       0.5539     2.008651         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_933.36       162.17     3_095.54       0.6941     1.007218         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_933.36        94.16     3_027.53       0.4211     3.673312         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_933.36       137.82     3_071.18       0.5506     2.035196         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_933.36       174.94     3_108.30       0.6900     1.029156         5.82
IVF-Binary-256-nl547-random (self)                     2_933.36     1_212.81     4_146.18       0.5570     1.967407         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_521.18        54.35     2_575.52       0.1904          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_521.18        63.88     2_585.05       0.1893          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_521.18        76.95     2_598.13       0.1883          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_521.18        89.58     2_610.76       0.3916     4.060917         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_521.18       129.82     2_651.00       0.5151     2.302227         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_521.18       192.03     2_713.20       0.6522     1.207171         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_521.18       108.74     2_629.91       0.3891     4.101545         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_521.18       162.56     2_683.73       0.5128     2.321976         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_521.18       212.99     2_734.17       0.6499     1.216265         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_521.18       131.35     2_652.53       0.3864     4.140029         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_521.18       177.32     2_698.50       0.5095     2.347326         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_521.18       250.92     2_772.09       0.6461     1.235365         5.79
IVF-Binary-256-nl273-itq (self)                        2_521.18     1_374.79     3_895.96       0.5137     2.294061         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_778.53        67.62     2_846.15       0.1903          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_778.53        80.52     2_859.05       0.1888          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_778.53       101.02     2_879.55       0.3911     4.060725         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_778.53       127.89     2_906.42       0.5150     2.299208         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_778.53       178.39     2_956.92       0.6521     1.205093         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_778.53       110.36     2_888.89       0.3871     4.126425         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_778.53       146.15     2_924.68       0.5105     2.336230         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_778.53       205.25     2_983.78       0.6472     1.227264         5.80
IVF-Binary-256-nl387-itq (self)                        2_778.53     1_237.32     4_015.85       0.5159     2.270768         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_084.80        56.95     3_141.75       0.1915          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_084.80        61.41     3_146.21       0.1904          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_084.80        67.48     3_152.28       0.1892          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_084.80        86.33     3_171.13       0.3943     4.017566         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_084.80       113.54     3_198.34       0.5192     2.262953         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_084.80       163.12     3_247.92       0.6571     1.179721         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_084.80        85.71     3_170.51       0.3912     4.062240         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_084.80       113.13     3_197.93       0.5152     2.293847         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_084.80       162.53     3_247.33       0.6530     1.195484         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_084.80        92.48     3_177.28       0.3883     4.110804         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_084.80       120.92     3_205.72       0.5114     2.329036         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_084.80       171.29     3_256.09       0.6489     1.218997         5.82
IVF-Binary-256-nl547-itq (self)                        3_084.80     1_049.80     4_134.60       0.5201     2.235387         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_187.20       106.35     3_293.54       0.2929          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_187.20       116.22     3_303.41       0.2924          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_187.20       148.61     3_335.80       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_187.20       138.91     3_326.11       0.5865     1.777894        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_187.20       169.04     3_356.24       0.7255     0.834278        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_187.20       226.88     3_414.07       0.8433     0.358489        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_187.20       155.03     3_342.23       0.5863     1.774378        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_187.20       184.64     3_371.84       0.7262     0.823265        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_187.20       240.71     3_427.90       0.8456     0.341312        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_187.20       183.40     3_370.59       0.5849     1.784210        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_187.20       218.77     3_405.97       0.7249     0.828604        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_187.20       282.97     3_470.17       0.8447     0.342678        10.40
IVF-Binary-512-nl273-random (self)                     3_187.20     1_872.07     5_059.27       0.7247     0.820119        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_523.71       106.90     3_630.61       0.2925          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_523.71       134.93     3_658.64       0.2915          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_523.71       140.09     3_663.80       0.5871     1.768931        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_523.71       170.06     3_693.77       0.7264     0.828199        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_523.71       223.45     3_747.16       0.8449     0.354058        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_523.71       169.28     3_692.99       0.5851     1.780633        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_523.71       200.58     3_724.29       0.7251     0.826723        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_523.71       259.22     3_782.93       0.8452     0.341036        10.41
IVF-Binary-512-nl387-random (self)                     3_523.71     1_818.17     5_341.88       0.7248     0.827215        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_088.46       101.40     4_189.85       0.2936          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_088.46       110.66     4_199.11       0.2929          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_088.46       126.91     4_215.36       0.2921          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_088.46       135.05     4_223.50       0.5886     1.763517        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_088.46       164.86     4_253.31       0.7281     0.824131        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_088.46       210.38     4_298.84       0.8461     0.353593        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_088.46       141.61     4_230.06       0.5871     1.766939        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_088.46       178.06     4_266.52       0.7271     0.820926        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_088.46       225.31     4_313.77       0.8464     0.341065        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_088.46       172.48     4_260.94       0.5855     1.778571        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_088.46       204.73     4_293.19       0.7259     0.824482        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_088.46       287.13     4_375.59       0.8456     0.338961        10.43
IVF-Binary-512-nl547-random (self)                     4_088.46     1_726.69     5_815.14       0.7268     0.818374        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_470.99       103.91     3_574.90       0.2844          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_470.99       120.84     3_591.83       0.2840          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_470.99       150.03     3_621.02       0.2830          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_470.99       145.40     3_616.39       0.5702     1.829008        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_470.99       186.92     3_657.91       0.7080     0.874265        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_470.99       224.57     3_695.57       0.8300     0.375344        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_470.99       159.06     3_630.05       0.5701     1.823803        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_470.99       194.85     3_665.84       0.7087     0.861913        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_470.99       253.68     3_724.67       0.8321     0.358097        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_470.99       203.44     3_674.43       0.5686     1.832962        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_470.99       230.91     3_701.90       0.7071     0.866957        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_470.99       297.64     3_768.63       0.8307     0.359946        10.40
IVF-Binary-512-nl273-itq (self)                        3_470.99     1_884.90     5_355.89       0.7091     0.849525        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_789.43       105.13     3_894.57       0.2842          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_789.43       130.54     3_919.98       0.2832          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_789.43       140.63     3_930.06       0.5704     1.826086        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_789.43       169.26     3_958.69       0.7088     0.871419        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_789.43       224.07     4_013.51       0.8314     0.372787        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_789.43       167.10     3_956.54       0.5689     1.830080        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_789.43       200.19     3_989.62       0.7076     0.864892        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_789.43       259.64     4_049.07       0.8312     0.359127        10.41
IVF-Binary-512-nl387-itq (self)                        3_789.43     1_685.92     5_475.36       0.7092     0.858401        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_277.75       100.94     4_378.69       0.2851          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_277.75       109.76     4_387.51       0.2843          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_277.75       123.07     4_400.82       0.2835          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_277.75       131.43     4_409.18       0.5724     1.810842        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_277.75       161.79     4_439.54       0.7109     0.863964        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_277.75       216.12     4_493.88       0.8324     0.372531        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_277.75       143.65     4_421.40       0.5712     1.814069        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_277.75       172.96     4_450.72       0.7099     0.858586        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_277.75       225.44     4_503.19       0.8329     0.359564        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_277.75       158.12     4_435.87       0.5696     1.824876        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_277.75       190.19     4_467.94       0.7082     0.861870        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_277.75       244.85     4_522.61       0.8316     0.357369        10.43
IVF-Binary-512-nl547-itq (self)                        4_277.75     1_597.99     5_875.74       0.7111     0.850607        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_018.48        45.56     1_064.04       0.0678          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_018.48        53.29     1_071.77       0.0623          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_018.48        76.45     1_094.93       0.0599          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_018.48        66.84     1_085.32       0.1445    13.594418         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_018.48        89.69     1_108.17       0.2066     9.378090         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_018.48       118.38     1_136.86       0.2966     6.195382         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_018.48        74.26     1_092.74       0.1363    14.167233         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_018.48        93.42     1_111.90       0.1967     9.779402         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_018.48       130.60     1_149.08       0.2850     6.504299         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_018.48        92.10     1_110.58       0.1319    14.569742         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_018.48       113.94     1_132.41       0.1905    10.136059         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_018.48       151.48     1_169.96       0.2759     6.759263         1.76
IVF-Binary-32-nl273-signed (self)                      1_018.48       919.45     1_937.93       0.1988     9.537055         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_355.05        46.56     1_401.60       0.0647          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_355.05        61.66     1_416.71       0.0617          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_355.05        65.02     1_420.07       0.1411    13.649253         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_355.05        85.09     1_440.14       0.2036     9.376952         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_355.05       116.81     1_471.86       0.2932     6.166358         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_355.05        80.14     1_435.18       0.1351    14.226639         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_355.05       100.10     1_455.14       0.1945     9.827848         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_355.05       141.44     1_496.49       0.2809     6.537042         1.77
IVF-Binary-32-nl387-signed (self)                      1_355.05       830.83     2_185.88       0.2046     9.170278         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_858.15        43.17     1_901.31       0.0664          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_858.15        48.71     1_906.86       0.0644          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_858.15        56.14     1_914.28       0.0624          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_858.15        60.88     1_919.02       0.1456    13.304912         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_858.15        77.91     1_936.06       0.2094     9.098909         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_858.15       110.97     1_969.12       0.3026     5.932205         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_858.15        65.68     1_923.83       0.1413    13.664483         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_858.15        84.09     1_942.24       0.2031     9.410317         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_858.15       118.38     1_976.53       0.2934     6.188357         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_858.15        74.26     1_932.41       0.1369    14.047945         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_858.15        94.93     1_953.08       0.1971     9.719177         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_858.15       128.24     1_986.39       0.2843     6.447317         1.79
IVF-Binary-32-nl547-signed (self)                      1_858.15       777.45     2_635.60       0.2110     8.898953         1.79
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
Exhaustive (query)                                         4.46     1_555.99     1_560.45       1.0000     0.000000        18.88
Exhaustive (self)                                          4.46    16_670.00    16_674.45       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_176.12       504.68     1_680.80       0.2158          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_176.12       567.57     1_743.69       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_176.12       615.19     1_791.32       0.5705     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_176.12       697.04     1_873.16       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                     1_176.12     6_252.77     7_428.90       0.5696     0.001382         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_384.79       515.43     1_900.22       0.1983          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_384.79       578.84     1_963.62       0.4042     0.002874         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_384.79       621.09     2_005.88       0.5292     0.001626         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_384.79       745.44     2_130.22       0.6649     0.000853         4.61
ExhaustiveBinary-256-itq (self)                        1_384.79     6_218.67     7_603.46       0.5290     0.001611         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_331.05       858.00     3_189.05       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_331.05       957.72     3_288.76       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_331.05     1_005.93     3_336.98       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_331.05     1_074.94     3_405.99       0.8652     0.000203         9.22
ExhaustiveBinary-512-random (self)                     2_331.05     9_879.71    12_210.76       0.7519     0.000503         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_619.42       856.63     3_476.05       0.3035          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_619.42       875.71     3_495.13       0.5988     0.001175         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_619.42       994.54     3_613.97       0.7344     0.000548         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_619.42     1_123.51     3_742.93       0.8497     0.000226         9.22
ExhaustiveBinary-512-itq (self)                        2_619.42     9_870.12    12_489.54       0.7336     0.000546         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 162.64       632.56       795.20       0.0588          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   162.64       666.06       828.69       0.1297     0.011294         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  162.64       707.42       870.06       0.1862     0.007851         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  162.64       814.31       976.95       0.2706     0.005257         0.58
ExhaustiveBinary-32-signed (self)                        162.64     6_992.01     7_154.64       0.1885     0.007670         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_018.91        54.59     2_073.51       0.2192          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_018.91        60.09     2_079.01       0.2180          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_018.91        76.39     2_095.30       0.2171          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_018.91        84.35     2_103.26       0.4457     0.002467         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_018.91       109.41     2_128.33       0.5774     0.001353         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_018.91       156.76     2_175.67       0.7147     0.000675         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_018.91        90.94     2_109.85       0.4441     0.002485         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_018.91       120.86     2_139.78       0.5760     0.001359         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_018.91       172.04     2_190.95       0.7130     0.000678         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_018.91       109.23     2_128.14       0.4417     0.002507         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_018.91       157.54     2_176.45       0.5730     0.001375         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_018.91       211.90     2_230.82       0.7100     0.000688         5.79
IVF-Binary-256-nl273-random (self)                     2_018.91     1_359.73     3_378.64       0.5747     0.001354         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_447.48        56.88     2_504.37       0.2187          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)           2_447.48        67.35     2_514.84       0.2167          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)           2_447.48        86.74     2_534.23       0.4457     0.002464         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)          2_447.48       117.29     2_564.77       0.5774     0.001348         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)          2_447.48       165.33     2_612.82       0.7148     0.000672         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)           2_447.48       101.34     2_548.83       0.4423     0.002500         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)          2_447.48       147.65     2_595.14       0.5734     0.001369         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)          2_447.48       204.29     2_651.78       0.7106     0.000686         5.81
IVF-Binary-256-nl387-random (self)                     2_447.48     1_173.20     3_620.69       0.5768     0.001341         5.81
IVF-Binary-256-nl547-np23-rf0-random (query)           2_952.94        57.38     3_010.32       0.2200          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)           2_952.94        62.32     3_015.26       0.2186          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)           2_952.94        68.02     3_020.96       0.2175          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)           2_952.94        89.22     3_042.16       0.4487     0.002436         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)          2_952.94       119.11     3_072.05       0.5812     0.001327         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)          2_952.94       168.90     3_121.84       0.7189     0.000657         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)           2_952.94        93.40     3_046.34       0.4461     0.002462         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)          2_952.94       115.78     3_068.72       0.5777     0.001344         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)          2_952.94       163.23     3_116.17       0.7157     0.000666         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)           2_952.94        94.09     3_047.03       0.4433     0.002492         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)          2_952.94       136.24     3_089.18       0.5745     0.001364         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)          2_952.94       176.23     3_129.17       0.7119     0.000680         5.83
IVF-Binary-256-nl547-random (self)                     2_952.94     1_081.55     4_034.49       0.5805     0.001321         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_201.01        54.60     2_255.61       0.2017          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_201.01        59.60     2_260.61       0.2007          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_201.01        75.17     2_276.18       0.1994          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_201.01        81.80     2_282.81       0.4113     0.002795         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_201.01       109.06     2_310.08       0.5371     0.001579         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_201.01       157.00     2_358.01       0.6728     0.000827         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_201.01        89.20     2_290.21       0.4087     0.002823         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_201.01       119.05     2_320.06       0.5345     0.001593         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_201.01       168.59     2_369.60       0.6710     0.000831         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_201.01       107.86     2_308.87       0.4062     0.002850         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_201.01       138.40     2_339.41       0.5314     0.001613         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_201.01       191.38     2_392.39       0.6672     0.000844         5.79
IVF-Binary-256-nl273-itq (self)                        2_201.01     1_202.31     3_403.32       0.5347     0.001578         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_554.53        59.76     2_614.30       0.2017          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_554.53        70.32     2_624.85       0.1999          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_554.53        89.60     2_644.14       0.4114     0.002790         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_554.53       116.12     2_670.65       0.5375     0.001573         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_554.53       162.44     2_716.97       0.6734     0.000819         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_554.53        98.66     2_653.19       0.4074     0.002838         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_554.53       129.63     2_684.16       0.5323     0.001604         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_554.53       196.80     2_751.33       0.6682     0.000838         5.81
IVF-Binary-256-nl387-itq (self)                        2_554.53     1_196.43     3_750.96       0.5371     0.001561         5.81
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_035.25        52.17     3_087.42       0.2029          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_035.25        55.78     3_091.02       0.2016          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_035.25        62.07     3_097.32       0.2005          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_035.25        81.08     3_116.33       0.4146     0.002761         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_035.25       105.31     3_140.56       0.5413     0.001549         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_035.25       151.02     3_186.26       0.6778     0.000804         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_035.25        84.00     3_119.25       0.4114     0.002795         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_035.25       111.64     3_146.89       0.5373     0.001572         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_035.25       158.89     3_194.14       0.6741     0.000816         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_035.25        90.93     3_126.18       0.4086     0.002828         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_035.25       120.90     3_156.15       0.5338     0.001596         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_035.25       167.92     3_203.17       0.6698     0.000833         5.83
IVF-Binary-256-nl547-itq (self)                        3_035.25     1_051.23     4_086.48       0.5410     0.001537         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_176.87       118.47     3_295.34       0.3156          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_176.87       141.66     3_318.54       0.3150          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_176.87       165.42     3_342.30       0.3144          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_176.87       145.61     3_322.49       0.6203     0.001095        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_176.87       198.28     3_375.15       0.7549     0.000507        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_176.87       260.76     3_437.63       0.8636     0.000215        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_176.87       187.77     3_364.64       0.6205     0.001091        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_176.87       228.34     3_405.22       0.7560     0.000498        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_176.87       289.67     3_466.54       0.8664     0.000202        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_176.87       223.03     3_399.90       0.6192     0.001098        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_176.87       278.03     3_454.90       0.7548     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_176.87       337.67     3_514.54       0.8658     0.000202        10.40
IVF-Binary-512-nl273-random (self)                     3_176.87     2_199.16     5_376.03       0.7540     0.000497        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_623.20       130.32     3_753.51       0.3154          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_623.20       164.03     3_787.22       0.3145          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_623.20       174.23     3_797.43       0.6208     0.001090        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_623.20       203.60     3_826.79       0.7556     0.000503        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_623.20       268.03     3_891.23       0.8650     0.000210        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_623.20       208.11     3_831.30       0.6191     0.001097        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_623.20       235.39     3_858.59       0.7551     0.000499        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_623.20       312.31     3_935.51       0.8661     0.000201        10.41
IVF-Binary-512-nl387-random (self)                     3_623.20     1_839.99     5_463.19       0.7539     0.000502        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_157.22       125.69     4_282.91       0.3164          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)           4_157.22       133.60     4_290.82       0.3157          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)           4_157.22       147.93     4_305.14       0.3147          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)           4_157.22       161.94     4_319.16       0.6225     0.001082        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)          4_157.22       191.75     4_348.97       0.7572     0.000497        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)          4_157.22       250.31     4_407.53       0.8660     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)           4_157.22       171.91     4_329.13       0.6212     0.001086        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)          4_157.22       205.80     4_363.01       0.7571     0.000493        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)          4_157.22       268.61     4_425.82       0.8670     0.000201        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)           4_157.22       184.63     4_341.85       0.6195     0.001095        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)          4_157.22       233.02     4_390.24       0.7557     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)          4_157.22       266.96     4_424.18       0.8665     0.000200        10.44
IVF-Binary-512-nl547-random (self)                     4_157.22     1_719.94     5_877.16       0.7555     0.000498        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_468.83       108.83     3_577.67       0.3055          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_468.83       134.44     3_603.28       0.3049          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_468.83       170.81     3_639.65       0.3041          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_468.83       146.96     3_615.79       0.6015     0.001167        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_468.83       180.27     3_649.10       0.7360     0.000550        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_468.83       231.86     3_700.69       0.8492     0.000237        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_468.83       163.98     3_632.81       0.6012     0.001164        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_468.83       198.09     3_666.93       0.7364     0.000542        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_468.83       287.76     3_756.59       0.8514     0.000225        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_468.83       232.49     3_701.33       0.6000     0.001169        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_468.83       280.33     3_749.16       0.7351     0.000545        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_468.83       366.60     3_835.43       0.8505     0.000225        10.40
IVF-Binary-512-nl273-itq (self)                        3_468.83     2_069.88     5_538.71       0.7361     0.000539        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_987.75       118.26     4_106.01       0.3058          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_987.75       142.28     4_130.03       0.3044          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_987.75       158.83     4_146.58       0.6018     0.001163        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_987.75       198.42     4_186.18       0.7365     0.000546        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_987.75       238.69     4_226.44       0.8502     0.000232        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_987.75       181.60     4_169.35       0.6002     0.001168        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_987.75       216.17     4_203.92       0.7356     0.000544        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_987.75       271.90     4_259.65       0.8505     0.000224        10.41
IVF-Binary-512-nl387-itq (self)                        3_987.75     1_946.43     5_934.18       0.7359     0.000544        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_360.49       111.30     4_471.78       0.3067          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_360.49       120.72     4_481.21       0.3058          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_360.49       138.08     4_498.57       0.3050          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_360.49       149.55     4_510.04       0.6031     0.001155        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_360.49       177.84     4_538.32       0.7381     0.000542        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_360.49       224.41     4_584.90       0.8514     0.000231        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_360.49       149.40     4_509.89       0.6022     0.001158        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_360.49       195.10     4_555.58       0.7376     0.000539        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_360.49       250.69     4_611.18       0.8520     0.000223        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_360.49       172.40     4_532.89       0.6009     0.001164        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_360.49       219.07     4_579.55       0.7362     0.000542        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_360.49       264.74     4_625.23       0.8512     0.000223        10.44
IVF-Binary-512-nl547-itq (self)                        4_360.49     1_750.76     6_111.25       0.7374     0.000541        10.44
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_047.79        45.48     1_093.28       0.0691          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_047.79        55.58     1_103.37       0.0632          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_047.79        75.49     1_123.29       0.0608          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_047.79        66.38     1_114.18       0.1490     0.009980         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_047.79        90.25     1_138.05       0.2112     0.006858         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_047.79       121.79     1_169.59       0.3020     0.004530         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_047.79        78.00     1_125.80       0.1407     0.010412         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_047.79        98.04     1_145.84       0.2004     0.007200         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_047.79       130.47     1_178.26       0.2905     0.004744         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_047.79        96.93     1_144.73       0.1359     0.010732         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_047.79       118.14     1_165.94       0.1935     0.007454         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_047.79       154.69     1_202.49       0.2806     0.004964         1.76
IVF-Binary-32-nl273-signed (self)                      1_047.79       949.79     1_997.58       0.2029     0.007018         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_431.09        46.26     1_477.36       0.0656          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_431.09        62.83     1_493.93       0.0623          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_431.09        65.76     1_496.85       0.1446     0.010049         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_431.09        86.26     1_517.36       0.2076     0.006882         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_431.09       122.97     1_554.06       0.3001     0.004515         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_431.09        85.23     1_516.32       0.1372     0.010527         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_431.09       105.88     1_536.97       0.1973     0.007264         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_431.09       146.19     1_577.29       0.2856     0.004812         1.77
IVF-Binary-32-nl387-signed (self)                      1_431.09       880.85     2_311.94       0.2095     0.006749         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_936.11        44.08     1_980.19       0.0673          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_936.11        47.81     1_983.92       0.0653          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_936.11        55.84     1_991.95       0.0632          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_936.11        60.92     1_997.03       0.1488     0.009793         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_936.11        77.94     2_014.05       0.2134     0.006672         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_936.11       112.47     2_048.58       0.3088     0.004337         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_936.11        65.55     2_001.66       0.1441     0.010066         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_936.11        94.11     2_030.22       0.2070     0.006903         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_936.11       139.62     2_075.73       0.2999     0.004522         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_936.11        74.84     2_010.95       0.1393     0.010368         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_936.11        94.10     2_030.21       0.2001     0.007146         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_936.11       134.31     2_070.43       0.2896     0.004723         1.79
IVF-Binary-32-nl547-signed (self)                      1_936.11       823.99     2_760.10       0.2153     0.006558         1.79
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
Exhaustive (query)                                         3.41     1_754.51     1_757.92       1.0000     0.000000        18.31
Exhaustive (self)                                          3.41    18_017.92    18_021.33       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_162.27       527.46     1_689.72       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_162.27       585.33     1_747.59       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_162.27       647.09     1_809.36       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_162.27       726.62     1_888.88       0.5614     0.256662         4.61
ExhaustiveBinary-256-random (self)                     1_162.27     6_190.95     7_353.21       0.4294     0.437191         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_363.36       505.78     1_869.14       0.1323          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_363.36       554.19     1_917.55       0.3010     0.745496         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_363.36       601.43     1_964.79       0.4149     0.448765         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_363.36       700.21     2_063.57       0.5534     0.252544         4.61
ExhaustiveBinary-256-itq (self)                        1_363.36     6_064.81     7_428.17       0.4203     0.435859         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_340.72       912.97     3_253.69       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_340.72       945.03     3_285.75       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_340.72       961.06     3_301.78       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_340.72     1_066.70     3_407.41       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                     2_340.72     9_762.18    12_102.90       0.6185     0.184365         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_616.70       893.86     3_510.56       0.2201          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_616.70       971.03     3_587.73       0.4659     0.364676         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_616.70     1_041.21     3_657.91       0.6031     0.194711         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_616.70     1_115.74     3_732.44       0.7399     0.094579         9.22
ExhaustiveBinary-512-itq (self)                        2_616.70    10_176.43    12_793.13       0.6053     0.192118         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 162.20       631.57       793.77       0.0127          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   162.20       682.68       844.89       0.0523     2.784656         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  162.20       750.27       912.47       0.0945     1.948956         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  162.20       748.21       910.41       0.1639     1.328894         0.58
ExhaustiveBinary-32-signed (self)                        162.20     7_143.23     7_305.43       0.0975     1.917524         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_059.27        49.22     2_108.49       0.1443          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_059.27        56.06     2_115.33       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_059.27        73.05     2_132.31       0.1371          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_059.27        75.57     2_134.83       0.3253     0.693086         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_059.27       100.24     2_159.50       0.4471     0.412061         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_059.27       146.18     2_205.44       0.5876     0.229141         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_059.27        83.36     2_142.62       0.3165     0.721914         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_059.27       108.29     2_167.55       0.4368     0.431051         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_059.27       152.81     2_212.07       0.5764     0.241195         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_059.27       102.27     2_161.54       0.3123     0.739522         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_059.27       130.26     2_189.52       0.4312     0.443199         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_059.27       176.63     2_235.90       0.5697     0.249326         5.79
IVF-Binary-256-nl273-random (self)                     2_059.27     1_081.92     3_141.19       0.4422     0.416770         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_375.47        53.33     2_428.80       0.1403          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_375.47        65.54     2_441.02       0.1380          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_375.47        78.15     2_453.63       0.3203     0.707597         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_375.47       101.58     2_477.05       0.4414     0.420873         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_375.47       140.67     2_516.14       0.5825     0.233592         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_375.47        92.06     2_467.53       0.3145     0.730872         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_375.47       118.41     2_493.89       0.4334     0.437743         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_375.47       160.85     2_536.32       0.5726     0.245415         5.80
IVF-Binary-256-nl387-random (self)                     2_375.47     1_036.80     3_412.27       0.4467     0.406727         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_935.90        62.10     2_998.00       0.1414          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_935.90        61.94     2_997.83       0.1400          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_935.90        69.26     3_005.15       0.1386          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_935.90        80.17     3_016.07       0.3224     0.696960         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_935.90       104.20     3_040.10       0.4446     0.412561         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_935.90       147.88     3_083.78       0.5867     0.227688         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_935.90        89.43     3_025.33       0.3186     0.712150         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_935.90       116.92     3_052.82       0.4393     0.423692         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_935.90       163.31     3_099.21       0.5799     0.235828         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_935.90        96.66     3_032.56       0.3148     0.727776         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_935.90       126.93     3_062.83       0.4343     0.435360         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_935.90       170.85     3_106.75       0.5738     0.243852         5.82
IVF-Binary-256-nl547-random (self)                     2_935.90     1_087.32     4_023.22       0.4498     0.399471         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_295.73        56.48     2_352.22       0.1419          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_295.73        60.10     2_355.83       0.1360          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_295.73        80.12     2_375.85       0.1341          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_295.73        84.19     2_379.92       0.3212     0.685010         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_295.73       112.93     2_408.66       0.4400     0.406661         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_295.73       154.77     2_450.51       0.5808     0.224650         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_295.73        91.93     2_387.66       0.3108     0.716015         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_295.73       125.65     2_421.39       0.4281     0.426268         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_295.73       191.91     2_487.64       0.5689     0.236183         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_295.73       112.66     2_408.39       0.3065     0.734144         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_295.73       133.77     2_429.50       0.4227     0.437877         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_295.73       179.41     2_475.14       0.5619     0.244814         5.79
IVF-Binary-256-nl273-itq (self)                        2_295.73     1_213.82     3_509.55       0.4336     0.414147         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_654.92        58.41     2_713.33       0.1381          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_654.92        69.21     2_724.14       0.1355          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_654.92        81.84     2_736.76       0.3138     0.702433         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_654.92       105.24     2_760.16       0.4326     0.417412         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_654.92       145.81     2_800.73       0.5747     0.230125         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_654.92        98.85     2_753.77       0.3080     0.725600         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_654.92       122.75     2_777.68       0.4251     0.432796         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_654.92       163.90     2_818.83       0.5646     0.241725         5.80
IVF-Binary-256-nl387-itq (self)                        2_654.92     1_033.09     3_688.02       0.4384     0.405036         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_053.44        58.13     3_111.56       0.1387          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_053.44        60.89     3_114.33       0.1368          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_053.44        67.80     3_121.24       0.1349          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_053.44        82.73     3_136.17       0.3163     0.695063         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_053.44       103.18     3_156.61       0.4354     0.411472         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_053.44       141.44     3_194.88       0.5784     0.225638         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_053.44        82.46     3_135.90       0.3122     0.711002         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_053.44       102.80     3_156.23       0.4295     0.423325         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_053.44       143.87     3_197.31       0.5713     0.233789         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_053.44        88.00     3_141.43       0.3082     0.727060         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_053.44       110.84     3_164.28       0.4242     0.434190         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_053.44       153.67     3_207.11       0.5647     0.241780         5.82
IVF-Binary-256-nl547-itq (self)                        3_053.44       996.01     4_049.44       0.4406     0.400147         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_228.18       109.29     3_337.48       0.2304          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_228.18       122.90     3_351.08       0.2278          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_228.18       159.04     3_387.23       0.2270          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_228.18       133.40     3_361.59       0.4860     0.337755        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_228.18       160.91     3_389.09       0.6273     0.176939        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_228.18       205.90     3_434.08       0.7652     0.083691        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_228.18       152.27     3_380.45       0.4812     0.344920        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_228.18       190.12     3_418.31       0.6224     0.181309        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_228.18       233.16     3_461.35       0.7602     0.086526        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_228.18       195.99     3_424.17       0.4791     0.349097        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_228.18       229.22     3_457.41       0.6194     0.184459        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_228.18       304.94     3_533.12       0.7568     0.088670        10.40
IVF-Binary-512-nl273-random (self)                     3_228.18     2_059.89     5_288.07       0.6260     0.177548        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_584.88       110.74     3_695.62       0.2289          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_584.88       145.07     3_729.95       0.2274          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_584.88       153.89     3_738.77       0.4838     0.341431        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_584.88       186.99     3_771.87       0.6253     0.178808        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_584.88       220.53     3_805.41       0.7626     0.084984        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_584.88       174.26     3_759.14       0.4809     0.347136        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_584.88       204.67     3_789.55       0.6210     0.183003        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_584.88       259.67     3_844.55       0.7578     0.087874        10.41
IVF-Binary-512-nl387-random (self)                     3_584.88     1_759.44     5_344.32       0.6282     0.175444        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_147.16       106.32     4_253.48       0.2295          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_147.16       120.53     4_267.69       0.2286          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_147.16       137.82     4_284.98       0.2277          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_147.16       141.81     4_288.97       0.4844     0.340126        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_147.16       167.73     4_314.89       0.6262     0.177884        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_147.16       215.89     4_363.05       0.7650     0.083907        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_147.16       148.73     4_295.89       0.4822     0.344431        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_147.16       180.74     4_327.91       0.6234     0.180818        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_147.16       235.49     4_382.65       0.7614     0.086089        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_147.16       195.82     4_342.99       0.4802     0.348094        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_147.16       212.24     4_359.41       0.6210     0.183289        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_147.16       291.06     4_438.23       0.7585     0.087937        10.43
IVF-Binary-512-nl547-random (self)                     4_147.16     1_846.52     5_993.69       0.6297     0.174184        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_531.13       105.36     3_636.49       0.2257          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_531.13       121.81     3_652.94       0.2228          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_531.13       157.37     3_688.50       0.2217          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_531.13       141.22     3_672.35       0.4776     0.346424        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_531.13       171.13     3_702.26       0.6165     0.182585        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_531.13       227.24     3_758.37       0.7533     0.086960        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_531.13       160.56     3_691.69       0.4724     0.354595        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_531.13       199.04     3_730.17       0.6109     0.187437        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_531.13       284.32     3_815.45       0.7479     0.089948        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_531.13       205.55     3_736.68       0.4700     0.359308        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_531.13       256.07     3_787.20       0.6079     0.190682        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_531.13       342.45     3_873.58       0.7442     0.092299        10.40
IVF-Binary-512-nl273-itq (self)                        3_531.13     2_141.90     5_673.04       0.6129     0.185172        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_952.86       112.29     4_065.15       0.2241          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_952.86       139.40     4_092.26       0.2228          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_952.86       140.81     4_093.67       0.4746     0.350633        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_952.86       170.21     4_123.07       0.6132     0.185038        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_952.86       215.14     4_168.00       0.7506     0.088404        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_952.86       171.03     4_123.89       0.4712     0.356842        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_952.86       202.25     4_155.11       0.6089     0.189400        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_952.86       256.10     4_208.95       0.7457     0.091460        10.41
IVF-Binary-512-nl387-itq (self)                        3_952.86     1_687.97     5_640.83       0.6153     0.182902        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_284.18       104.06     4_388.24       0.2247          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_284.18       113.01     4_397.19       0.2239          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_284.18       129.13     4_413.31       0.2230          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_284.18       137.81     4_421.99       0.4758     0.348893        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_284.18       167.46     4_451.64       0.6143     0.183855        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_284.18       214.42     4_498.60       0.7530     0.087132        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_284.18       147.66     4_431.84       0.4732     0.353735        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_284.18       183.50     4_467.68       0.6109     0.187362        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_284.18       228.22     4_512.39       0.7491     0.089423        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_284.18       163.99     4_448.17       0.4711     0.357554        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_284.18       198.17     4_482.34       0.6083     0.190193        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_284.18       250.58     4_534.76       0.7458     0.091576        10.43
IVF-Binary-512-nl547-itq (self)                        4_284.18     1_677.49     5_961.67       0.6169     0.181414        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_017.84        41.29     1_059.12       0.0235          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_017.84        52.07     1_069.90       0.0154          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_017.84        70.87     1_088.71       0.0137          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_017.84        51.79     1_069.63       0.0897     2.416169         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_017.84        64.87     1_082.71       0.1544     1.703588         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_017.84        86.63     1_104.47       0.2533     1.154367         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_017.84        61.43     1_079.27       0.0635     2.876777         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_017.84        74.86     1_092.70       0.1149     2.058055         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_017.84        97.60     1_115.44       0.1983     1.415134         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_017.84        83.27     1_101.11       0.0562     3.122823         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_017.84        96.87     1_114.70       0.1031     2.254940         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_017.84       122.66     1_140.50       0.1797     1.561374         1.76
IVF-Binary-32-nl273-signed (self)                      1_017.84       738.56     1_756.40       0.1191     2.022036         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_366.40        43.84     1_410.25       0.0178          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_366.40        58.31     1_424.71       0.0149          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_366.40        54.64     1_421.05       0.0722     2.606529         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_366.40        66.07     1_432.48       0.1288     1.851287         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_366.40        89.80     1_456.20       0.2194     1.233486         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_366.40        69.29     1_435.70       0.0616     2.927427         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_366.40        82.31     1_448.72       0.1123     2.092720         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_366.40       108.19     1_474.59       0.1942     1.417277         1.77
IVF-Binary-32-nl387-signed (self)                      1_366.40       656.57     2_022.97       0.1327     1.822754         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_875.07        40.84     1_915.91       0.0186          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_875.07        45.96     1_921.04       0.0172          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_875.07        56.54     1_931.61       0.0153          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_875.07        51.98     1_927.05       0.0765     2.534380         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_875.07        62.57     1_937.64       0.1357     1.769089         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_875.07        84.84     1_959.91       0.2279     1.159732         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_875.07        55.75     1_930.82       0.0706     2.697398         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_875.07        67.93     1_943.00       0.1261     1.897787         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_875.07        89.76     1_964.83       0.2128     1.252451         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_875.07        63.33     1_938.40       0.0643     2.947101         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_875.07        78.40     1_953.47       0.1153     2.091844         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_875.07       100.37     1_975.44       0.1967     1.384908         1.79
IVF-Binary-32-nl547-signed (self)                      1_875.07       663.14     2_538.21       0.1379     1.737439         1.79
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
Exhaustive (query)                                         3.24     1_636.76     1_640.00       1.0000     0.000000        18.31
Exhaustive (self)                                          3.24    17_757.38    17_760.62       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_194.33       535.98     1_730.31       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_194.33       593.07     1_787.40       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_194.33       649.75     1_844.08       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_194.33       745.67     1_940.00       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                     1_194.33     6_058.38     7_252.71       0.3238     3.208083         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 1_359.87       540.90     1_900.77       0.0723          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   1_359.87       569.55     1_929.43       0.1941     5.797375         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  1_359.87       623.24     1_983.11       0.2870     3.686845         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  1_359.87       697.72     2_057.59       0.4150     2.209544         4.61
ExhaustiveBinary-256-itq (self)                        1_359.87     6_101.85     7_461.72       0.2901     3.621623         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_337.40       878.70     3_216.11       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_337.40       958.04     3_295.44       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_337.40     1_031.18     3_368.59       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_337.40     1_119.96     3_457.36       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                     2_337.40     9_747.18    12_084.58       0.4788     1.710526         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 2_604.68       872.24     3_476.92       0.1559          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                   2_604.68       901.94     3_506.62       0.3458     3.038797         9.22
ExhaustiveBinary-512-itq-rf10 (query)                  2_604.68     1_008.84     3_613.52       0.4728     1.776101         9.22
ExhaustiveBinary-512-itq-rf20 (query)                  2_604.68     1_106.63     3_711.31       0.6194     0.970394         9.22
ExhaustiveBinary-512-itq (self)                        2_604.68     9_730.52    12_335.20       0.4718     1.776907         9.22
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-32-signed_no_rr (query)                 160.33       654.85       815.18       0.0061          NaN         0.58
ExhaustiveBinary-32-signed-rf5 (query)                   160.33       671.74       832.08       0.0286    14.192335         0.58
ExhaustiveBinary-32-signed-rf10 (query)                  160.33       689.17       849.51       0.0545    10.228876         0.58
ExhaustiveBinary-32-signed-rf20 (query)                  160.33       756.48       916.81       0.1028     7.206309         0.58
ExhaustiveBinary-32-signed (self)                        160.33     6_879.77     7_040.10       0.0548    10.217732         0.58
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_087.09        53.97     2_141.06       0.1015          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_087.09        57.55     2_144.63       0.0926          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_087.09        69.07     2_156.15       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_087.09        71.71     2_158.80       0.2466     4.682597         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_087.09        96.72     2_183.81       0.3527     2.916537         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_087.09       135.81     2_222.90       0.4886     1.698168         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_087.09        80.31     2_167.39       0.2318     5.166092         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_087.09       102.99     2_190.08       0.3354     3.190525         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_087.09       148.53     2_235.62       0.4700     1.862358         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_087.09        93.61     2_180.69       0.2280     5.296145         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_087.09       121.98     2_209.06       0.3304     3.274036         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_087.09       180.82     2_267.91       0.4638     1.915926         5.79
IVF-Binary-256-nl273-random (self)                     2_087.09     1_091.75     3_178.83       0.3387     3.130111         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_424.71        50.80     2_475.51       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_424.71        67.70     2_492.41       0.0919          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_424.71        82.56     2_507.27       0.2358     4.931507         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_424.71       106.86     2_531.57       0.3421     3.052683         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_424.71       145.70     2_570.40       0.4777     1.802324         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_424.71        88.37     2_513.08       0.2299     5.156744         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_424.71       110.41     2_535.11       0.3340     3.196200         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_424.71       152.94     2_577.64       0.4679     1.888188         5.80
IVF-Binary-256-nl387-random (self)                     2_424.71     1_062.15     3_486.86       0.3448     3.003180         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_927.62        50.52     2_978.14       0.0954          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_927.62        61.28     2_988.90       0.0934          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_927.62        60.41     2_988.04       0.0917          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_927.62        78.34     3_005.96       0.2398     4.791733         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_927.62       100.26     3_027.88       0.3460     2.991093         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_927.62       134.05     3_061.67       0.4824     1.757338         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_927.62       101.10     3_028.72       0.2354     4.928101         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_927.62       122.39     3_050.01       0.3400     3.085964         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_927.62       167.83     3_095.45       0.4749     1.823784         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_927.62        96.30     3_023.92       0.2309     5.078371         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_927.62       104.64     3_032.26       0.3339     3.187089         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_927.62       170.42     3_098.04       0.4677     1.889983         5.82
IVF-Binary-256-nl547-random (self)                     2_927.62       960.12     3_887.74       0.3483     2.939438         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              2_361.60        54.06     2_415.66       0.0868          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              2_361.60        61.24     2_422.84       0.0764          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              2_361.60        75.69     2_437.29       0.0743          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              2_361.60        78.53     2_440.13       0.2240     5.242254         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             2_361.60       111.46     2_473.06       0.3228     3.332816         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             2_361.60       143.53     2_505.13       0.4540     1.955591         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              2_361.60        78.97     2_440.57       0.2059     5.837070         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             2_361.60       139.09     2_500.69       0.3031     3.665059         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             2_361.60       141.18     2_502.78       0.4354     2.128142         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              2_361.60        92.13     2_453.73       0.2024     5.999445         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             2_361.60       119.10     2_480.70       0.2984     3.750497         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             2_361.60       161.28     2_522.88       0.4299     2.178631         5.79
IVF-Binary-256-nl273-itq (self)                        2_361.60     1_020.07     3_381.67       0.3062     3.592544         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)              2_559.21        52.59     2_611.80       0.0787          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)              2_559.21        64.31     2_623.53       0.0755          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)              2_559.21        76.00     2_635.21       0.2105     5.561993         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)             2_559.21       106.62     2_665.84       0.3103     3.476108         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)             2_559.21       139.19     2_698.40       0.4433     2.030233         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)              2_559.21        86.56     2_645.78       0.2044     5.813640         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)             2_559.21       118.10     2_677.31       0.3018     3.634383         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)             2_559.21       155.67     2_714.89       0.4335     2.125014         5.80
IVF-Binary-256-nl387-itq (self)                        2_559.21       979.00     3_538.21       0.3133     3.418173         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)              3_107.29        51.22     3_158.51       0.0794          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)              3_107.29        53.95     3_161.24       0.0772          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)              3_107.29        61.12     3_168.41       0.0754          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)              3_107.29        73.22     3_180.51       0.2134     5.388027         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)             3_107.29        94.58     3_201.87       0.3145     3.374689         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)             3_107.29       142.49     3_249.78       0.4484     1.978852         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)              3_107.29        77.37     3_184.66       0.2078     5.557946         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)             3_107.29        98.26     3_205.55       0.3074     3.489575         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)             3_107.29       148.12     3_255.42       0.4400     2.053468         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)              3_107.29        93.42     3_200.71       0.2034     5.753020         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)             3_107.29       106.83     3_214.12       0.3013     3.612860         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)             3_107.29       147.67     3_254.96       0.4325     2.124859         5.82
IVF-Binary-256-nl547-itq (self)                        3_107.29       965.73     4_073.02       0.3172     3.322422         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_232.62       104.85     3_337.47       0.1685          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_232.62       125.15     3_357.77       0.1642          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_232.62       157.62     3_390.24       0.1629          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_232.62       140.37     3_372.99       0.3648     2.762852        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_232.62       165.35     3_397.96       0.4946     1.613586        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_232.62       211.11     3_443.73       0.6428     0.865216        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_232.62       150.84     3_383.46       0.3574     2.870953        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_232.62       177.69     3_410.31       0.4866     1.678608        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_232.62       238.15     3_470.76       0.6348     0.903671        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_232.62       190.63     3_423.24       0.3551     2.903769        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_232.62       224.49     3_457.11       0.4839     1.699072        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_232.62       284.84     3_517.45       0.6317     0.917169        10.40
IVF-Binary-512-nl273-random (self)                     3_232.62     1_818.15     5_050.77       0.4885     1.655176        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_559.07       108.02     3_667.09       0.1647          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_559.07       157.06     3_716.12       0.1631          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_559.07       143.55     3_702.61       0.3603     2.826319        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_559.07       161.41     3_720.48       0.4905     1.642435        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_559.07       226.79     3_785.86       0.6385     0.885603        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_559.07       169.91     3_728.97       0.3567     2.886554        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_559.07       191.74     3_750.81       0.4856     1.679814        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_559.07       245.95     3_805.02       0.6332     0.909694        10.41
IVF-Binary-512-nl387-random (self)                     3_559.07     1_660.70     5_219.76       0.4917     1.624177        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_159.22       111.39     4_270.61       0.1660          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_159.22       128.76     4_287.98       0.1648          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_159.22       148.84     4_308.06       0.1637          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_159.22       147.99     4_307.21       0.3623     2.788844        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_159.22       169.16     4_328.38       0.4927     1.618053        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_159.22       227.45     4_386.67       0.6419     0.867394        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_159.22       151.91     4_311.13       0.3595     2.836677        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_159.22       179.68     4_338.90       0.4890     1.650040        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_159.22       230.79     4_390.01       0.6371     0.891036        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_159.22       162.21     4_321.43       0.3568     2.876142        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_159.22       199.77     4_358.99       0.4854     1.678098        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_159.22       269.65     4_428.87       0.6333     0.908540        10.43
IVF-Binary-512-nl547-random (self)                     4_159.22     1_664.81     5_824.03       0.4940     1.602595        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)              3_477.32       104.02     3_581.34       0.1641          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)              3_477.32       117.57     3_594.89       0.1594          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)              3_477.32       151.48     3_628.80       0.1586          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)              3_477.32       137.79     3_615.11       0.3622     2.801495        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)             3_477.32       160.41     3_637.73       0.4908     1.632770        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)             3_477.32       202.89     3_680.21       0.6380     0.884673        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)              3_477.32       163.52     3_640.84       0.3548     2.928605        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)             3_477.32       178.49     3_655.81       0.4828     1.707191        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)             3_477.32       238.49     3_715.81       0.6304     0.924631        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)              3_477.32       213.43     3_690.75       0.3529     2.955746        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)             3_477.32       269.68     3_747.00       0.4806     1.725365        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)             3_477.32       286.14     3_763.46       0.6277     0.936508        10.40
IVF-Binary-512-nl273-itq (self)                        3_477.32     1_976.33     5_453.65       0.4822     1.709519        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)              3_874.08       110.41     3_984.49       0.1606          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)              3_874.08       137.97     4_012.05       0.1589          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)              3_874.08       137.60     4_011.68       0.3567     2.893509        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)             3_874.08       164.85     4_038.93       0.4852     1.682718        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)             3_874.08       210.74     4_084.82       0.6339     0.904804        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)              3_874.08       171.77     4_045.85       0.3529     2.952913        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)             3_874.08       212.43     4_086.51       0.4807     1.720456        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)             3_874.08       257.77     4_131.85       0.6290     0.928580        10.41
IVF-Binary-512-nl387-itq (self)                        3_874.08     1_683.75     5_557.83       0.4854     1.676904        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)              4_416.42       107.20     4_523.61       0.1614          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)              4_416.42       113.84     4_530.26       0.1600          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)              4_416.42       134.98     4_551.40       0.1590          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)              4_416.42       136.54     4_552.95       0.3587     2.846465        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)             4_416.42       159.11     4_575.53       0.4877     1.654850        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)             4_416.42       215.25     4_631.67       0.6368     0.889093        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)              4_416.42       148.97     4_565.38       0.3555     2.901752        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)             4_416.42       182.74     4_599.16       0.4839     1.690145        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)             4_416.42       243.38     4_659.80       0.6325     0.913422        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)              4_416.42       189.62     4_606.04       0.3531     2.946528        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)             4_416.42       218.35     4_634.77       0.4811     1.718104        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)             4_416.42       256.95     4_673.37       0.6288     0.931682        10.43
IVF-Binary-512-nl547-itq (self)                        4_416.42     1_672.16     6_088.58       0.4875     1.654260        10.43
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-32-nl273-np13-rf0-signed (query)            1_080.79        41.08     1_121.87       0.0130          NaN         1.76
IVF-Binary-32-nl273-np16-rf0-signed (query)            1_080.79        50.19     1_130.98       0.0074          NaN         1.76
IVF-Binary-32-nl273-np23-rf0-signed (query)            1_080.79        67.41     1_148.20       0.0064          NaN         1.76
IVF-Binary-32-nl273-np13-rf5-signed (query)            1_080.79        48.68     1_129.47       0.0591    13.000291         1.76
IVF-Binary-32-nl273-np13-rf10-signed (query)           1_080.79        56.44     1_137.23       0.1116     9.719921         1.76
IVF-Binary-32-nl273-np13-rf20-signed (query)           1_080.79        74.46     1_155.25       0.2026     7.085233         1.76
IVF-Binary-32-nl273-np16-rf5-signed (query)            1_080.79        55.50     1_136.29       0.0353    15.842858         1.76
IVF-Binary-32-nl273-np16-rf10-signed (query)           1_080.79        66.36     1_147.15       0.0694    12.250771         1.76
IVF-Binary-32-nl273-np16-rf20-signed (query)           1_080.79        85.05     1_165.84       0.1301     9.401566         1.76
IVF-Binary-32-nl273-np23-rf5-signed (query)            1_080.79        74.50     1_155.29       0.0315    16.892835         1.76
IVF-Binary-32-nl273-np23-rf10-signed (query)           1_080.79        85.09     1_165.88       0.0622    13.196258         1.76
IVF-Binary-32-nl273-np23-rf20-signed (query)           1_080.79       110.01     1_190.80       0.1188    10.155905         1.76
IVF-Binary-32-nl273-signed (self)                      1_080.79       635.25     1_716.04       0.0700    12.239707         1.76
IVF-Binary-32-nl387-np19-rf0-signed (query)            1_450.81        44.71     1_495.52       0.0089          NaN         1.77
IVF-Binary-32-nl387-np27-rf0-signed (query)            1_450.81        61.03     1_511.84       0.0074          NaN         1.77
IVF-Binary-32-nl387-np19-rf5-signed (query)            1_450.81        52.06     1_502.86       0.0421    14.676558         1.77
IVF-Binary-32-nl387-np19-rf10-signed (query)           1_450.81        59.00     1_509.81       0.0814    11.363510         1.77
IVF-Binary-32-nl387-np19-rf20-signed (query)           1_450.81        85.93     1_536.73       0.1520     8.332591         1.77
IVF-Binary-32-nl387-np27-rf5-signed (query)            1_450.81        72.77     1_523.58       0.0361    16.320991         1.77
IVF-Binary-32-nl387-np27-rf10-signed (query)           1_450.81        80.65     1_531.46       0.0692    12.831872         1.77
IVF-Binary-32-nl387-np27-rf20-signed (query)           1_450.81        95.50     1_546.31       0.1306     9.495885         1.77
IVF-Binary-32-nl387-signed (self)                      1_450.81       595.25     2_046.05       0.0809    11.280660         1.77
IVF-Binary-32-nl547-np23-rf0-signed (query)            1_982.25        40.55     2_022.80       0.0094          NaN         1.79
IVF-Binary-32-nl547-np27-rf0-signed (query)            1_982.25        46.29     2_028.55       0.0084          NaN         1.79
IVF-Binary-32-nl547-np33-rf0-signed (query)            1_982.25        54.34     2_036.59       0.0075          NaN         1.79
IVF-Binary-32-nl547-np23-rf5-signed (query)            1_982.25        46.63     2_028.88       0.0449    13.934437         1.79
IVF-Binary-32-nl547-np23-rf10-signed (query)           1_982.25        54.97     2_037.22       0.0868    10.610329         1.79
IVF-Binary-32-nl547-np23-rf20-signed (query)           1_982.25        73.41     2_055.66       0.1631     7.402378         1.79
IVF-Binary-32-nl547-np27-rf5-signed (query)            1_982.25        51.26     2_033.52       0.0404    15.027103         1.79
IVF-Binary-32-nl547-np27-rf10-signed (query)           1_982.25        63.98     2_046.24       0.0786    11.592403         1.79
IVF-Binary-32-nl547-np27-rf20-signed (query)           1_982.25        78.24     2_060.50       0.1486     8.130423         1.79
IVF-Binary-32-nl547-np33-rf5-signed (query)            1_982.25        59.15     2_041.40       0.0364    16.431882         1.79
IVF-Binary-32-nl547-np33-rf10-signed (query)           1_982.25        70.00     2_052.25       0.0711    12.849049         1.79
IVF-Binary-32-nl547-np33-rf20-signed (query)           1_982.25        85.03     2_067.28       0.1343     9.208698         1.79
IVF-Binary-32-nl547-signed (self)                      1_982.25       580.82     2_563.07       0.0883    10.543611         1.79
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 128 dimensions

The binary indices shine clearly more with higher dimensions (especially
the signed version).

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        16.03     6_842.00     6_858.03       1.0000     0.000000        73.24
Exhaustive (self)                                         16.03    69_415.59    69_431.62       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_789.07       529.77     4_318.84       0.0770          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_789.07       589.85     4_378.92       0.1853    40.590911         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_789.07       667.63     4_456.71       0.2680    26.716815         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_789.07       775.03     4_564.11       0.3854    16.589779         4.70
ExhaustiveBinary-256-random (self)                     3_789.07     6_373.74    10_162.81       0.2721    26.279507         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-itq_no_rr (query)                 4_623.20       536.21     5_159.41       0.0352          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                   4_623.20       571.59     5_194.79       0.1078    58.311100         4.70
ExhaustiveBinary-256-itq-rf10 (query)                  4_623.20       613.13     5_236.33       0.1716    39.742881         4.70
ExhaustiveBinary-256-itq-rf20 (query)                  4_623.20       736.15     5_359.35       0.2692    25.845190         4.70
ExhaustiveBinary-256-itq (self)                        4_623.20     6_329.08    10_952.28       0.1755    39.408653         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_512.85       926.70     8_439.55       0.1527          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_512.85     1_033.43     8_546.28       0.3242    22.196757         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_512.85     1_078.57     8_591.42       0.4419    13.545010         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_512.85     1_188.24     8_701.09       0.5832     7.719977         9.41
ExhaustiveBinary-512-random (self)                     7_512.85    10_310.19    17_823.04       0.4431    13.453984         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-itq_no_rr (query)                 8_337.69       949.46     9_287.15       0.1257          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                   8_337.69     1_027.23     9_364.91       0.2688    27.688764         9.41
ExhaustiveBinary-512-itq-rf10 (query)                  8_337.69     1_100.29     9_437.98       0.3728    17.457655         9.41
ExhaustiveBinary-512-itq-rf20 (query)                  8_337.69     1_235.09     9_572.77       0.5062    10.365482         9.41
ExhaustiveBinary-512-itq (self)                        8_337.69    10_894.89    19_232.58       0.3728    17.440968         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            14_838.60     1_556.99    16_395.59       0.2353          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              14_838.60     1_697.52    16_536.12       0.4918    11.183271        18.81
ExhaustiveBinary-1024-random-rf10 (query)             14_838.60     1_651.67    16_490.27       0.6358     5.988378        18.81
ExhaustiveBinary-1024-random-rf20 (query)             14_838.60     1_828.24    16_666.84       0.7779     2.871209        18.81
ExhaustiveBinary-1024-random (self)                   14_838.60    18_092.69    32_931.29       0.6381     5.940314        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-itq_no_rr (query)               16_014.47     1_555.35    17_569.82       0.2198          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 16_014.47     1_726.00    17_740.47       0.4636    12.595159        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                16_014.47     1_855.88    17_870.34       0.6048     6.884343        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                16_014.47     1_862.73    17_877.20       0.7500     3.387537        18.81
ExhaustiveBinary-1024-itq (self)                      16_014.47    18_099.44    34_113.91       0.6059     6.857783        18.81
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-128-signed_no_rr (query)              1_999.22       406.04     2_405.26       0.0275          NaN         2.35
ExhaustiveBinary-128-signed-rf5 (query)                1_999.22       466.58     2_465.80       0.0917    63.471902         2.35
ExhaustiveBinary-128-signed-rf10 (query)               1_999.22       503.46     2_502.68       0.1491    44.012830         2.35
ExhaustiveBinary-128-signed-rf20 (query)               1_999.22       672.00     2_671.22       0.2396    29.121769         2.35
ExhaustiveBinary-128-signed (self)                     1_999.22     5_097.73     7_096.95       0.1538    43.418564         2.35
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           6_876.14        85.66     6_961.80       0.0894          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           6_876.14        91.21     6_967.35       0.0819          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           6_876.14       104.98     6_981.12       0.0801          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           6_876.14       118.48     6_994.62       0.2082    37.003584         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          6_876.14       151.41     7_027.55       0.2984    23.995106         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          6_876.14       231.46     7_107.60       0.4184    14.855149         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           6_876.14       143.14     7_019.28       0.1960    39.530049         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          6_876.14       183.94     7_060.08       0.2836    25.718898         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          6_876.14       260.65     7_136.79       0.4031    15.887879         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           6_876.14       157.55     7_033.69       0.1927    40.256023         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          6_876.14       199.56     7_075.70       0.2789    26.258727         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          6_876.14       286.57     7_162.71       0.3972    16.273250         5.98
IVF-Binary-256-nl273-random (self)                     6_876.14     1_748.94     8_625.08       0.2865    25.390857         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)           8_165.70       100.14     8_265.84       0.0832          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           8_165.70       112.30     8_278.00       0.0801          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           8_165.70       139.00     8_304.70       0.1995    38.528842         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          8_165.70       171.31     8_337.01       0.2884    25.014672         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          8_165.70       231.34     8_397.04       0.4104    15.315480         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           8_165.70       148.34     8_314.04       0.1929    40.071462         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          8_165.70       194.45     8_360.14       0.2798    26.100169         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          8_165.70       276.79     8_442.49       0.4005    15.973244         6.04
IVF-Binary-256-nl387-random (self)                     8_165.70     1_724.73     9_890.43       0.2930    24.576702         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)           9_709.22       101.61     9_810.84       0.0847          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           9_709.22       105.46     9_814.68       0.0829          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           9_709.22       107.84     9_817.06       0.0815          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           9_709.22       141.71     9_850.94       0.2026    37.589199         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          9_709.22       180.33     9_889.55       0.2933    24.331515         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          9_709.22       245.81     9_955.03       0.4160    14.924539         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           9_709.22       143.43     9_852.66       0.1985    38.518369         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          9_709.22       181.20     9_890.42       0.2874    25.037356         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          9_709.22       254.18     9_963.41       0.4081    15.411046         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           9_709.22       160.44     9_869.67       0.1948    39.402901         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          9_709.22       196.95     9_906.17       0.2821    25.705501         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          9_709.22       286.97     9_996.20       0.4018    15.833779         6.12
IVF-Binary-256-nl547-random (self)                     9_709.22     1_743.68    11_452.90       0.2978    23.888895         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-itq (query)              7_565.42        96.06     7_661.47       0.0482          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)              7_565.42       104.06     7_669.48       0.0390          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)              7_565.42       127.47     7_692.89       0.0374          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)              7_565.42       128.66     7_694.08       0.1397    52.463938         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)             7_565.42       159.97     7_725.39       0.2145    35.197972         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)             7_565.42       215.14     7_780.56       0.3213    22.347273         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)              7_565.42       138.61     7_704.03       0.1199    57.928141         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)             7_565.42       168.86     7_734.28       0.1903    39.114020         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)             7_565.42       227.89     7_793.31       0.2962    24.655666         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)              7_565.42       157.63     7_723.04       0.1149    60.513426         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)             7_565.42       186.85     7_752.27       0.1829    41.080971         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)             7_565.42       257.81     7_823.23       0.2865    25.849148         5.98
IVF-Binary-256-nl273-itq (self)                        7_565.42     1_671.02     9_236.43       0.1938    38.884251         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)              8_522.13       112.13     8_634.26       0.0404          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)              8_522.13       108.69     8_630.83       0.0378          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)              8_522.13       125.84     8_647.97       0.1244    55.640731         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)             8_522.13       156.29     8_678.42       0.1978    37.320039         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)             8_522.13       215.74     8_737.87       0.3054    23.656529         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)              8_522.13       137.69     8_659.82       0.1163    58.802592         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)             8_522.13       168.54     8_690.67       0.1864    39.620553         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)             8_522.13       228.44     8_750.57       0.2896    25.336545         6.04
IVF-Binary-256-nl387-itq (self)                        8_522.13     1_551.05    10_073.19       0.2017    36.989675         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             10_212.00        91.62    10_303.62       0.0417          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             10_212.00        94.62    10_306.62       0.0403          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             10_212.00       101.16    10_313.17       0.0386          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             10_212.00       119.75    10_331.75       0.1290    53.767635         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            10_212.00       154.62    10_366.62       0.2041    36.081452         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            10_212.00       205.78    10_417.79       0.3149    22.618775         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             10_212.00       121.11    10_333.11       0.1243    55.474048         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            10_212.00       154.04    10_366.04       0.1974    37.310195         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            10_212.00       210.67    10_422.67       0.3058    23.486295         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             10_212.00       127.72    10_339.73       0.1195    57.248562         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            10_212.00       158.90    10_370.90       0.1906    38.568551         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            10_212.00       219.00    10_431.00       0.2973    24.372360         6.12
IVF-Binary-256-nl547-itq (self)                       10_212.00     1_481.62    11_693.62       0.2073    35.779817         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          10_278.22       174.97    10_453.19       0.1596          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          10_278.22       188.42    10_466.64       0.1563          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          10_278.22       219.04    10_497.26       0.1554          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          10_278.22       224.11    10_502.33       0.3370    20.987751        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         10_278.22       261.85    10_540.07       0.4570    12.738091        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         10_278.22       325.45    10_603.67       0.5992     7.207613        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          10_278.22       240.29    10_518.51       0.3308    21.630921        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         10_278.22       287.66    10_565.88       0.4506    13.121710        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         10_278.22       352.86    10_631.08       0.5923     7.447657        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          10_278.22       270.95    10_549.17       0.3289    21.801439        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         10_278.22       322.43    10_600.66       0.4477    13.266981        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         10_278.22       394.55    10_672.77       0.5898     7.527176        10.69
IVF-Binary-512-nl273-random (self)                    10_278.22     2_860.04    13_138.27       0.4518    13.046860        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          12_080.31       181.02    12_261.33       0.1571          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          12_080.31       204.22    12_284.53       0.1557          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          12_080.31       230.74    12_311.05       0.3331    21.366472        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         12_080.31       270.34    12_350.65       0.4543    12.892701        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         12_080.31       332.85    12_413.16       0.5962     7.323848        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          12_080.31       257.23    12_337.54       0.3294    21.746609        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         12_080.31       300.90    12_381.21       0.4493    13.172260        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         12_080.31       371.95    12_452.26       0.5909     7.498725        10.74
IVF-Binary-512-nl387-random (self)                    12_080.31     2_671.17    14_751.48       0.4552    12.836096        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          13_078.70       177.30    13_256.00       0.1579          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          13_078.70       187.31    13_266.02       0.1569          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          13_078.70       200.54    13_279.25       0.1560          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          13_078.70       226.64    13_305.34       0.3355    21.117416        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         13_078.70       262.37    13_341.08       0.4570    12.738653        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         13_078.70       322.41    13_401.12       0.5999     7.182206        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          13_078.70       238.54    13_317.24       0.3326    21.426644        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         13_078.70       305.10    13_383.80       0.4531    12.960609        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         13_078.70       353.17    13_431.87       0.5949     7.349314        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          13_078.70       252.25    13_330.95       0.3301    21.676728        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         13_078.70       306.93    13_385.64       0.4501    13.125100        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         13_078.70       360.11    13_438.82       0.5916     7.462558        10.82
IVF-Binary-512-nl547-random (self)                    13_078.70     2_613.26    15_691.96       0.4583    12.660440        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_113.55       178.37    11_291.91       0.1335          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_113.55       192.62    11_306.16       0.1292          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_113.55       224.79    11_338.33       0.1279          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_113.55       226.15    11_339.69       0.2833    26.029419        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_113.55       265.40    11_378.95       0.3914    16.279525        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_113.55       328.20    11_441.75       0.5269     9.602523        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_113.55       243.26    11_356.81       0.2760    26.900039        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_113.55       285.20    11_398.75       0.3833    16.817419        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_113.55       354.68    11_468.22       0.5181     9.955689        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_113.55       276.67    11_390.22       0.2736    27.251520        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_113.55       333.71    11_447.26       0.3797    17.080878        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_113.55       399.70    11_513.24       0.5142    10.116242        10.69
IVF-Binary-512-nl273-itq (self)                       11_113.55     2_850.32    13_963.87       0.3833    16.842147        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_414.51       187.63    12_602.14       0.1306          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_414.51       215.41    12_629.91       0.1284          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_414.51       234.97    12_649.48       0.2787    26.548786        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_414.51       278.53    12_693.04       0.3872    16.550673        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_414.51       333.99    12_748.50       0.5220     9.796127        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_414.51       262.30    12_676.81       0.2742    27.159052        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_414.51       305.59    12_720.10       0.3814    16.980048        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_414.51       373.69    12_788.20       0.5151    10.076268        10.74
IVF-Binary-512-nl387-itq (self)                       12_414.51     2_694.76    15_109.27       0.3869    16.582669        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             14_202.75       182.45    14_385.20       0.1309          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             14_202.75       192.29    14_395.04       0.1297          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             14_202.75       205.27    14_408.02       0.1287          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             14_202.75       246.12    14_448.87       0.2813    26.218914        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            14_202.75       281.02    14_483.77       0.3898    16.347559        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            14_202.75       326.94    14_529.69       0.5268     9.607873        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             14_202.75       237.73    14_440.47       0.2781    26.609800        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            14_202.75       276.85    14_479.60       0.3853    16.644434        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            14_202.75       338.86    14_541.61       0.5211     9.822816        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             14_202.75       255.40    14_458.15       0.2755    26.964158        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            14_202.75       296.02    14_498.77       0.3817    16.897180        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            14_202.75       364.86    14_567.61       0.5166    10.003392        10.82
IVF-Binary-512-nl547-itq (self)                       14_202.75     2_803.19    17_005.94       0.3903    16.342942        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         17_817.17       428.79    18_245.96       0.2399          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         17_817.17       525.16    18_342.33       0.2382          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         17_817.17       618.04    18_435.22       0.2377          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         17_817.17       485.57    18_302.74       0.4992    10.834117        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        17_817.17       528.05    18_345.23       0.6434     5.779240        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        17_817.17       601.02    18_418.20       0.7845     2.747805        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         17_817.17       548.78    18_365.96       0.4962    10.981356        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        17_817.17       584.07    18_401.25       0.6406     5.859089        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        17_817.17       636.03    18_453.20       0.7820     2.797675        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         17_817.17       613.54    18_430.72       0.4952    11.037131        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        17_817.17       660.91    18_478.08       0.6394     5.893505        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        17_817.17       726.03    18_543.21       0.7808     2.820922        20.09
IVF-Binary-1024-nl273-random (self)                   17_817.17     5_700.92    23_518.10       0.6428     5.813738        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         19_292.48       465.28    19_757.77       0.2387          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         19_292.48       564.72    19_857.21       0.2378          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         19_292.48       499.33    19_791.82       0.4978    10.900723        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        19_292.48       554.42    19_846.90       0.6417     5.823667        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        19_292.48       611.23    19_903.72       0.7830     2.778159        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         19_292.48       594.46    19_886.95       0.4959    11.002308        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        19_292.48       633.62    19_926.11       0.6394     5.889963        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        19_292.48       706.85    19_999.33       0.7809     2.816231        20.15
IVF-Binary-1024-nl387-random (self)                   19_292.48     5_256.57    24_549.06       0.6441     5.772927        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         20_880.86       413.14    21_294.00       0.2393          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         20_880.86       473.18    21_354.04       0.2386          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         20_880.86       503.92    21_384.77       0.2379          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         20_880.86       463.62    21_344.48       0.4993    10.819022        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        20_880.86       485.77    21_366.63       0.6438     5.766379        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        20_880.86       563.96    21_444.82       0.7852     2.739656        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         20_880.86       524.16    21_405.02       0.4976    10.906062        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        20_880.86       569.47    21_450.32       0.6415     5.829342        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        20_880.86       628.21    21_509.07       0.7827     2.782993        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         20_880.86       571.17    21_452.03       0.4962    10.977444        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        20_880.86       579.23    21_460.09       0.6400     5.874175        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        20_880.86       630.25    21_511.11       0.7814     2.810928        20.23
IVF-Binary-1024-nl547-random (self)                   20_880.86     4_880.50    25_761.35       0.6462     5.710744        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-itq (query)            18_926.18       438.28    19_364.46       0.2242          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            18_926.18       482.12    19_408.30       0.2223          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            18_926.18       576.91    19_503.09       0.2220          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            18_926.18       488.09    19_414.27       0.4719    12.157504        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           18_926.18       506.08    19_432.26       0.6134     6.627209        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           18_926.18       563.93    19_490.11       0.7573     3.242512        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            18_926.18       523.24    19_449.42       0.4688    12.329625        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           18_926.18       573.13    19_499.31       0.6100     6.735050        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           18_926.18       636.23    19_562.41       0.7541     3.309755        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            18_926.18       619.11    19_545.29       0.4675    12.399111        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           18_926.18       658.22    19_584.40       0.6087     6.782239        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           18_926.18       726.14    19_652.32       0.7528     3.336675        20.09
IVF-Binary-1024-nl273-itq (self)                      18_926.18     5_557.75    24_483.93       0.6108     6.714386        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            19_965.90       444.58    20_410.48       0.2238          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            19_965.90       528.74    20_494.64       0.2227          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            19_965.90       486.90    20_452.80       0.4700    12.245717        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           19_965.90       516.58    20_482.48       0.6113     6.696701        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           19_965.90       577.14    20_543.03       0.7560     3.268036        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            19_965.90       566.03    20_531.93       0.4677    12.372991        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           19_965.90       599.56    20_565.46       0.6089     6.769585        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           19_965.90       663.77    20_629.67       0.7537     3.313465        20.15
IVF-Binary-1024-nl387-itq (self)                      19_965.90     5_117.53    25_083.43       0.6124     6.660010        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            21_713.54       476.81    22_190.34       0.2236          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            21_713.54       499.62    22_213.15       0.2227          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            21_713.54       542.75    22_256.29       0.2221          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            21_713.54       498.84    22_212.38       0.4714    12.177027        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           21_713.54       536.94    22_250.48       0.6140     6.605088        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           21_713.54       615.73    22_329.27       0.7582     3.223254        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            21_713.54       525.76    22_239.30       0.4697    12.275342        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           21_713.54       547.66    22_261.20       0.6115     6.686787        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           21_713.54       620.15    22_333.69       0.7556     3.275393        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            21_713.54       572.49    22_286.03       0.4683    12.346531        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           21_713.54       600.16    22_313.70       0.6098     6.740111        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           21_713.54       658.50    22_372.04       0.7541     3.306943        20.23
IVF-Binary-1024-nl547-itq (self)                      21_713.54     5_481.52    27_195.06       0.6145     6.587174        20.23
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-128-nl273-np13-rf0-signed (query)           5_173.39        61.81     5_235.20       0.0399          NaN         3.63
IVF-Binary-128-nl273-np16-rf0-signed (query)           5_173.39        65.73     5_239.12       0.0317          NaN         3.63
IVF-Binary-128-nl273-np23-rf0-signed (query)           5_173.39        76.60     5_249.99       0.0301          NaN         3.63
IVF-Binary-128-nl273-np13-rf5-signed (query)           5_173.39        96.14     5_269.53       0.1239    56.710380         3.63
IVF-Binary-128-nl273-np13-rf10-signed (query)          5_173.39       122.11     5_295.51       0.1943    38.807787         3.63
IVF-Binary-128-nl273-np13-rf20-signed (query)          5_173.39       171.42     5_344.81       0.3000    25.020530         3.63
IVF-Binary-128-nl273-np16-rf5-signed (query)           5_173.39        94.02     5_267.41       0.1042    63.252727         3.63
IVF-Binary-128-nl273-np16-rf10-signed (query)          5_173.39       116.26     5_289.65       0.1684    43.835064         3.63
IVF-Binary-128-nl273-np16-rf20-signed (query)          5_173.39       168.14     5_341.53       0.2676    28.606538         3.63
IVF-Binary-128-nl273-np23-rf5-signed (query)           5_173.39        96.47     5_269.86       0.1001    65.256932         3.63
IVF-Binary-128-nl273-np23-rf10-signed (query)          5_173.39       135.00     5_308.39       0.1629    45.210498         3.63
IVF-Binary-128-nl273-np23-rf20-signed (query)          5_173.39       174.25     5_347.64       0.2585    29.701483         3.63
IVF-Binary-128-nl273-signed (self)                     5_173.39     1_111.15     6_284.54       0.1715    43.677138         3.63
IVF-Binary-128-nl387-np19-rf0-signed (query)           6_226.19        61.56     6_287.75       0.0330          NaN         3.69
IVF-Binary-128-nl387-np27-rf0-signed (query)           6_226.19        80.00     6_306.19       0.0307          NaN         3.69
IVF-Binary-128-nl387-np19-rf5-signed (query)           6_226.19        91.32     6_317.50       0.1089    60.815050         3.69
IVF-Binary-128-nl387-np19-rf10-signed (query)          6_226.19       106.19     6_332.38       0.1766    41.768239         3.69
IVF-Binary-128-nl387-np19-rf20-signed (query)          6_226.19       157.88     6_384.07       0.2794    27.140112         3.69
IVF-Binary-128-nl387-np27-rf5-signed (query)           6_226.19        92.12     6_318.31       0.1009    64.411662         3.69
IVF-Binary-128-nl387-np27-rf10-signed (query)          6_226.19       116.33     6_342.51       0.1645    44.719881         3.69
IVF-Binary-128-nl387-np27-rf20-signed (query)          6_226.19       162.46     6_388.65       0.2616    29.296424         3.69
IVF-Binary-128-nl387-signed (self)                     6_226.19     1_060.74     7_286.92       0.1815    41.279689         3.69
IVF-Binary-128-nl547-np23-rf0-signed (query)           7_813.72        64.28     7_878.00       0.0337          NaN         3.77
IVF-Binary-128-nl547-np27-rf0-signed (query)           7_813.72        68.65     7_882.37       0.0323          NaN         3.77
IVF-Binary-128-nl547-np33-rf0-signed (query)           7_813.72        70.86     7_884.58       0.0308          NaN         3.77
IVF-Binary-128-nl547-np23-rf5-signed (query)           7_813.72        89.01     7_902.72       0.1120    58.838025         3.77
IVF-Binary-128-nl547-np23-rf10-signed (query)          7_813.72       116.02     7_929.74       0.1811    40.180082         3.77
IVF-Binary-128-nl547-np23-rf20-signed (query)          7_813.72       155.68     7_969.40       0.2872    25.597596         3.77
IVF-Binary-128-nl547-np27-rf5-signed (query)           7_813.72        92.91     7_906.63       0.1078    60.660689         3.77
IVF-Binary-128-nl547-np27-rf10-signed (query)          7_813.72       111.51     7_925.22       0.1744    41.668459         3.77
IVF-Binary-128-nl547-np27-rf20-signed (query)          7_813.72       158.21     7_971.93       0.2770    26.760298         3.77
IVF-Binary-128-nl547-np33-rf5-signed (query)           7_813.72        92.36     7_906.08       0.1031    62.912931         3.77
IVF-Binary-128-nl547-np33-rf10-signed (query)          7_813.72       117.99     7_931.71       0.1677    43.335357         3.77
IVF-Binary-128-nl547-np33-rf20-signed (query)          7_813.72       164.21     7_977.92       0.2668    27.981886         3.77
IVF-Binary-128-nl547-signed (self)                     7_813.72     1_074.47     8_888.19       0.1868    39.559219         3.77
--------------------------------------------------------------------------------------------------------------------------------
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
<pre><code>

================================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         3.41     1_581.78     1_585.19       1.0000     0.000000        18.31
Exhaustive (self)                                          3.41    17_116.60    17_120.01       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_700.16       524.81     2_224.97       0.3141    41.590414         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_700.16       599.34     2_299.50       0.6781     1.424540         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_700.16       679.06     2_379.22       0.8317     0.540476         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_700.16       793.98     2_494.14       0.9366     0.159663         3.46
ExhaustiveRaBitQ (self)                                1_700.16     6_662.39     8_362.54       0.8323     0.527515         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        962.89       116.35     1_079.24       0.3203    41.573350         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        962.89       138.35     1_101.25       0.3193    41.580391         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        962.89       193.83     1_156.72       0.3181    41.584016         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        962.89       168.14     1_131.04       0.6869     1.446368         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       962.89       215.71     1_178.61       0.8370     0.558086         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       962.89       300.01     1_262.90       0.9352     0.179393         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        962.89       193.51     1_156.41       0.6866     1.449359         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       962.89       245.43     1_208.32       0.8392     0.547526         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       962.89       344.19     1_307.08       0.9397     0.160331         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        962.89       260.83     1_223.72       0.6853     1.459542         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       962.89       315.07     1_277.96       0.8388     0.549881         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       962.89       420.29     1_383.18       0.9407     0.158858         3.47
IVF-RaBitQ-nl273 (self)                                  962.89     4_161.89     5_124.78       0.9411     0.156273         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_272.25       122.11     1_394.36       0.3211    41.562649         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_272.25       168.68     1_440.94       0.3187    41.572303         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_272.25       175.66     1_447.92       0.6916     1.395271         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_272.25       232.17     1_504.42       0.8428     0.518946         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_272.25       324.25     1_596.51       0.9399     0.165851         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_272.25       257.14     1_529.40       0.6887     1.418885         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_272.25       288.29     1_560.55       0.8423     0.516875         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_272.25       378.90     1_651.16       0.9428     0.146575         3.49
IVF-RaBitQ-nl387 (self)                                1_272.25     3_774.46     5_046.71       0.9429     0.146880         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_853.73       142.33     1_996.05       0.3271    41.545554         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_853.73       166.57     2_020.30       0.3252    41.553698         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_853.73       202.68     2_056.41       0.3237    41.559146         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_853.73       193.56     2_047.29       0.6983     1.345744         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_853.73       231.34     2_085.06       0.8470     0.505449         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_853.73       319.22     2_172.94       0.9405     0.167264         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_853.73       194.65     2_048.38       0.6962     1.356471         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_853.73       246.77     2_100.49       0.8474     0.498092         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_853.73       342.86     2_196.59       0.9440     0.148206         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_853.73       222.35     2_076.08       0.6943     1.372707         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_853.73       264.41     2_118.13       0.8465     0.502418         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_853.73       362.55     2_216.27       0.9451     0.142254         3.51
IVF-RaBitQ-nl547 (self)                                1_853.73     3_564.01     5_417.74       0.9453     0.139848         3.51
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
Exhaustive (query)                                         4.37     1_658.15     1_662.52       1.0000     0.000000        18.88
Exhaustive (self)                                          4.37    16_818.92    16_823.29       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_780.40       454.56     2_234.96       0.3169     0.168465         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_780.40       534.36     2_314.76       0.6843     0.001053         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_780.40       613.48     2_393.88       0.8362     0.000399         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_780.40       778.33     2_558.73       0.9403     0.000112         3.46
ExhaustiveRaBitQ (self)                                1_780.40     6_131.16     7_911.56       0.8381     0.000387         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        994.10       123.65     1_117.75       0.3168     0.167971         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        994.10       180.02     1_174.11       0.3154     0.167676         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        994.10       220.45     1_214.55       0.3143     0.167526         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        994.10       180.44     1_174.54       0.6845     0.001130         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       994.10       247.28     1_241.37       0.8365     0.000437         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       994.10       333.53     1_327.63       0.9348     0.000143         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        994.10       205.98     1_200.08       0.6841     0.001136         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       994.10       259.33     1_253.43       0.8380     0.000432         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       994.10       354.44     1_348.54       0.9391     0.000130         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        994.10       279.94     1_274.04       0.6826     0.001145         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       994.10       329.06     1_323.15       0.8376     0.000434         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       994.10       432.18     1_426.28       0.9403     0.000127         3.47
IVF-RaBitQ-nl273 (self)                                  994.10     4_203.06     5_197.16       0.9409     0.000126         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_347.66       123.89     1_471.55       0.3204     0.168494         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_347.66       170.01     1_517.67       0.3183     0.168127         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_347.66       190.57     1_538.23       0.6895     0.001095         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_347.66       239.39     1_587.06       0.8408     0.000417         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_347.66       315.39     1_663.05       0.9394     0.000130         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_347.66       226.61     1_574.27       0.6868     0.001114         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_347.66       305.67     1_653.34       0.8401     0.000420         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_347.66       378.85     1_726.51       0.9420     0.000121         3.49
IVF-RaBitQ-nl387 (self)                                1_347.66     3_702.78     5_050.44       0.9432     0.000118         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_856.90       112.11     1_969.02       0.3220     0.169047         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_856.90       130.00     1_986.90       0.3198     0.168728         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_856.90       152.76     2_009.66       0.3184     0.168524         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_856.90       169.60     2_026.51       0.6948     0.001059         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_856.90       218.23     2_075.14       0.8446     0.000404         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_856.90       317.26     2_174.16       0.9406     0.000129         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_856.90       197.28     2_054.18       0.6923     0.001073         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_856.90       236.26     2_093.16       0.8445     0.000403         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_856.90       339.62     2_196.53       0.9434     0.000120         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_856.90       219.74     2_076.64       0.6898     0.001088         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_856.90       274.37     2_131.27       0.8431     0.000408         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_856.90       352.85     2_209.76       0.9441     0.000118         3.51
IVF-RaBitQ-nl547 (self)                                1_856.90     3_579.07     5_435.97       0.9452     0.000112         3.51
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
Exhaustive (query)                                         3.37     1_608.06     1_611.43       1.0000     0.000000        18.31
Exhaustive (self)                                          3.37    16_716.48    16_719.85       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_695.62       399.36     2_094.98       0.4426     1.402651         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_695.62       478.36     2_173.99       0.8785     0.030781         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_695.62       559.25     2_254.88       0.9682     0.005679         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_695.62       679.23     2_374.86       0.9954     0.000665         3.46
ExhaustiveRaBitQ (self)                                1_695.62     5_569.67     7_265.29       0.9696     0.005520         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        992.33       119.95     1_112.29       0.4624     1.393408         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        992.33       150.61     1_142.95       0.4620     1.393563         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        992.33       210.99     1_203.32       0.4619     1.393619         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        992.33       172.97     1_165.31       0.8931     0.028100         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       992.33       212.99     1_205.32       0.9745     0.004925         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       992.33       293.94     1_286.28       0.9968     0.000521         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        992.33       207.15     1_199.49       0.8928     0.028226         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       992.33       273.50     1_265.84       0.9744     0.004949         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       992.33       336.06     1_328.40       0.9969     0.000502         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        992.33       262.58     1_254.91       0.8926     0.028290         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       992.33       327.52     1_319.86       0.9743     0.004969         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       992.33       444.94     1_437.27       0.9969     0.000504         3.47
IVF-RaBitQ-nl273 (self)                                  992.33     4_077.63     5_069.97       0.9973     0.000448         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_211.15       119.74     1_330.89       0.4729     1.382070         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_211.15       166.22     1_377.37       0.4728     1.382133         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_211.15       171.74     1_382.89       0.9030     0.024206         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_211.15       216.55     1_427.70       0.9778     0.004107         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_211.15       306.11     1_517.26       0.9976     0.000386         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_211.15       229.85     1_441.00       0.9028     0.024248         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_211.15       274.88     1_486.03       0.9778     0.004118         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_211.15       354.39     1_565.54       0.9976     0.000376         3.49
IVF-RaBitQ-nl387 (self)                                1_211.15     3_545.90     4_757.05       0.9978     0.000345         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_730.69       111.79     1_842.48       0.4838     1.373109         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_730.69       132.51     1_863.20       0.4836     1.373215         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_730.69       151.91     1_882.60       0.4835     1.373254         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_730.69       162.84     1_893.53       0.9114     0.021068         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_730.69       206.28     1_936.97       0.9814     0.003289         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_730.69       278.84     2_009.53       0.9980     0.000298         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_730.69       180.40     1_911.09       0.9111     0.021149         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_730.69       225.90     1_956.60       0.9813     0.003312         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_730.69       298.73     2_029.42       0.9980     0.000293         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_730.69       205.41     1_936.10       0.9110     0.021171         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_730.69       251.56     1_982.26       0.9812     0.003318         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_730.69       333.86     2_064.55       0.9980     0.000286         3.51
IVF-RaBitQ-nl547 (self)                                1_730.69     3_412.99     5_143.68       0.9982     0.000270         3.51
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
Exhaustive (query)                                         3.41     1_590.90     1_594.31       1.0000     0.000000        18.31
Exhaustive (self)                                          3.41    17_005.52    17_008.93       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_685.20       413.42     2_098.62       0.4782     7.158330         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_685.20       509.33     2_194.53       0.9141     0.089870         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_685.20       556.42     2_241.62       0.9835     0.012778         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_685.20       661.40     2_346.60       0.9985     0.001011         3.46
ExhaustiveRaBitQ (self)                                1_685.20     5_711.19     7_396.39       0.9834     0.013136         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        881.48       111.32       992.80       0.4887     7.133550         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        881.48       134.83     1_016.31       0.4887     7.133580         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        881.48       191.20     1_072.68       0.4887     7.133584         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        881.48       161.57     1_043.05       0.9216     0.078073         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       881.48       214.33     1_095.81       0.9858     0.010544         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       881.48       286.60     1_168.08       0.9988     0.000755         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        881.48       189.38     1_070.86       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       881.48       237.60     1_119.08       0.9857     0.010553         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       881.48       319.57     1_201.05       0.9988     0.000741         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        881.48       244.09     1_125.57       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       881.48       301.18     1_182.66       0.9857     0.010558         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       881.48       403.75     1_285.23       0.9988     0.000741         3.47
IVF-RaBitQ-nl273 (self)                                  881.48     4_083.56     4_965.04       0.9988     0.000729         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_257.87       125.79     1_383.66       0.5004     7.112936         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_257.87       166.52     1_424.40       0.5004     7.112938         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_257.87       175.30     1_433.18       0.9290     0.068039         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_257.87       242.11     1_499.98       0.9875     0.009140         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_257.87       303.82     1_561.69       0.9991     0.000516         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_257.87       232.05     1_489.92       0.9290     0.068061         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_257.87       275.21     1_533.08       0.9875     0.009147         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_257.87       354.48     1_612.36       0.9991     0.000516         3.49
IVF-RaBitQ-nl387 (self)                                1_257.87     3_539.68     4_797.56       0.9991     0.000578         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_744.47       112.41     1_856.88       0.5107     7.089673         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_744.47       130.00     1_874.47       0.5107     7.089687         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_744.47       154.72     1_899.19       0.5107     7.089690         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_744.47       176.14     1_920.61       0.9363     0.058694         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_744.47       222.38     1_966.85       0.9896     0.007452         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_744.47       295.69     2_040.15       0.9994     0.000388         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_744.47       190.47     1_934.94       0.9362     0.058736         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_744.47       228.96     1_973.43       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_744.47       313.41     2_057.87       0.9994     0.000393         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_744.47       221.66     1_966.13       0.9362     0.058736         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_744.47       280.32     2_024.78       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_744.47       366.00     2_110.47       0.9994     0.000393         3.51
IVF-RaBitQ-nl547 (self)                                1_744.47     3_432.27     5_176.74       0.9993     0.000388         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With 128 dimensions

RaBitQ particularly shines with higher dimensionality in the data.

<details>
<summary><b>RaBitQ - Euclidean (Gaussian - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.40     6_489.29     6_504.69       1.0000     0.000000        73.24
Exhaustive (self)                                         15.40    68_561.75    68_577.15       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           5_309.33       901.62     6_210.95       0.7024    73.760102         5.31
ExhaustiveRaBitQ-rf5 (query)                           5_309.33       965.55     6_274.89       0.9952     0.017331         5.31
ExhaustiveRaBitQ-rf10 (query)                          5_309.33     1_027.96     6_337.29       0.9999     0.000453         5.31
ExhaustiveRaBitQ-rf20 (query)                          5_309.33     1_174.93     6_484.27       1.0000     0.000000         5.31
ExhaustiveRaBitQ (self)                                5_309.33    10_415.00    15_724.34       0.9999     0.000420         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      3_391.18       242.14     3_633.32       0.7066    73.750604         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      3_391.18       298.25     3_689.44       0.7066    73.750780         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      3_391.18       399.25     3_790.43       0.7066    73.750786         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      3_391.18       299.89     3_691.07       0.9954     0.017287         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     3_391.18       349.26     3_740.44       0.9996     0.002368         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     3_391.18       442.71     3_833.90       0.9997     0.002034         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      3_391.18       351.46     3_742.64       0.9956     0.015465         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     3_391.18       412.77     3_803.95       0.9999     0.000515         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     3_391.18       515.89     3_907.08       1.0000     0.000182         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      3_391.18       472.00     3_863.18       0.9956     0.015283         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     3_391.18       526.11     3_917.29       0.9999     0.000333         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     3_391.18       625.50     4_016.68       1.0000     0.000000         5.35
IVF-RaBitQ-nl273 (self)                                3_391.18     6_263.37     9_654.55       1.0000     0.000001         5.35
IVF-RaBitQ-nl387-np19-rf0 (query)                      4_340.54       275.30     4_615.84       0.7122    73.742352         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      4_340.54       378.94     4_719.48       0.7122    73.742372         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      4_340.54       333.75     4_674.29       0.9959     0.014264         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     4_340.54       388.75     4_729.29       0.9999     0.000347         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     4_340.54       479.65     4_820.19       1.0000     0.000111         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      4_340.54       438.69     4_779.23       0.9959     0.014164         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     4_340.54       492.17     4_832.72       0.9999     0.000236         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     4_340.54       591.21     4_931.75       1.0000     0.000000         5.41
IVF-RaBitQ-nl387 (self)                                4_340.54     5_876.28    10_216.82       1.0000     0.000004         5.41
IVF-RaBitQ-nl547-np23-rf0 (query)                      5_913.93       274.92     6_188.85       0.7164    73.731255         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      5_913.93       320.73     6_234.66       0.7164    73.731344         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      5_913.93       378.76     6_292.69       0.7164    73.731356         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      5_913.93       332.10     6_246.03       0.9962     0.013859         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     5_913.93       404.34     6_318.28       0.9998     0.000942         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     5_913.93       473.53     6_387.47       0.9999     0.000615         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      5_913.93       373.82     6_287.76       0.9963     0.013368         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     5_913.93       428.51     6_342.44       0.9999     0.000388         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     5_913.93       519.34     6_433.27       1.0000     0.000062         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      5_913.93       439.56     6_353.50       0.9963     0.013307         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     5_913.93       490.40     6_404.33       0.9999     0.000326         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     5_913.93       632.44     6_546.38       1.0000     0.000000         5.49
IVF-RaBitQ-nl547 (self)                                5_913.93     5_863.34    11_777.28       1.0000     0.000001         5.49
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>


Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls.

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
