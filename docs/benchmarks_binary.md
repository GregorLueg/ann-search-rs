## Binarised indices benchmarks and parameter 

Binarised indices compress the data stored in the index structure itself via
very aggressive quantisation to basically only bits. This has two impacts:

1. Drastic reduction in memory usage.
2. Increased query speed in most cases because the bit-wise operations are very
fast.
3. When not using any re-ranking of the top candidates, dramatically lower
Recall.

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
- *binarisation_init*: Two options are provided in the crate. `"random"` that
  generates random planes that are subsequently orthogonalised or a `"itq"` that
  leverages PCA to identify axis of maximum variation. As can seen below the
  latter however does not perform with few dimensions. In this case, `"random"`
  is a better choice. 
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
Exhaustive (query)                                         3.04     1_518.37     1_521.40       1.0000     0.000000        18.31
Exhaustive (self)                                          3.04    15_268.94    15_271.98       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_244.97       489.63     1_734.59       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_244.97       534.99     1_779.96       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_244.97       594.70     1_839.67       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_244.97       715.16     1_960.13       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                     1_244.97     5_837.77     7_082.74       0.5461     2.056589         4.61
ExhaustiveBinary-256-itq_no_rr (query)                13_446.11       489.78    13_935.89       0.1962          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                  13_446.11       532.96    13_979.07       0.4008     3.839819         4.61
ExhaustiveBinary-256-itq-rf10 (query)                 13_446.11       582.84    14_028.95       0.5266     2.152408         4.61
ExhaustiveBinary-256-itq-rf20 (query)                 13_446.11       682.84    14_128.95       0.6623     1.116198         4.61
ExhaustiveBinary-256-itq (self)                       13_446.11     5_818.02    19_264.13       0.5242     2.138272         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_458.00       841.15     3_299.16       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_458.00       892.40     3_350.40       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_458.00       939.93     3_397.93       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_458.00     1_043.17     3_501.17       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                     2_458.00     9_382.58    11_840.59       0.7223     0.828915         9.22
ExhaustiveBinary-512-itq_no_rr (query)                10_418.05       838.71    11_256.76       0.2923          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  10_418.05       876.79    11_294.84       0.5833     1.686259         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 10_418.05       929.07    11_347.12       0.7204     0.786147         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 10_418.05     1_047.21    11_465.26       0.8387     0.321334         9.22
ExhaustiveBinary-512-itq (self)                       10_418.05     9_336.06    19_754.11       0.7197     0.773467         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_137.53        52.76     2_190.29       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_137.53        60.60     2_198.14       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_137.53        75.21     2_212.74       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_137.53        87.80     2_225.33       0.4239     3.637415         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_137.53       110.30     2_247.83       0.5531     2.021145         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_137.53       157.15     2_294.68       0.6927     1.020296         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_137.53        91.79     2_229.32       0.4223     3.662012         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_137.53       120.35     2_257.88       0.5515     2.033363         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_137.53       169.78     2_307.31       0.6913     1.025118         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_137.53       106.62     2_244.16       0.4198     3.692295         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_137.53       138.39     2_275.92       0.5484     2.057241         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_137.53       192.17     2_329.70       0.6879     1.040613         5.79
IVF-Binary-256-nl273-random (self)                     2_137.53     1_189.42     3_326.95       0.5511     2.018819         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_419.95        55.76     2_475.71       0.2060          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_419.95        67.12     2_487.07       0.2047          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_419.95        83.47     2_503.43       0.4238     3.633534         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_419.95       111.67     2_531.62       0.5533     2.013383         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_419.95       158.02     2_577.97       0.6934     1.014371         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_419.95        96.88     2_516.84       0.4202     3.684350         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_419.95       143.22     2_563.18       0.5496     2.044808         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_419.95       204.81     2_624.76       0.6888     1.035978         5.80
IVF-Binary-256-nl387-random (self)                     2_419.95     1_156.47     3_576.43       0.5532     1.998682         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_922.20        52.99     2_975.19       0.2076          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_922.20        56.99     2_979.19       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_922.20        63.29     2_985.48       0.2050          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_922.20        80.00     3_002.20       0.4265     3.596401         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_922.20       106.37     3_028.57       0.5576     1.983148         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_922.20       153.67     3_075.86       0.6977     0.994532         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_922.20        85.10     3_007.30       0.4238     3.631225         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_922.20       113.28     3_035.48       0.5539     2.008651         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_922.20       158.38     3_080.57       0.6941     1.007218         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_922.20        92.04     3_014.24       0.4211     3.673312         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_922.20       120.13     3_042.33       0.5506     2.035196         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_922.20       169.61     3_091.81       0.6900     1.029156         5.82
IVF-Binary-256-nl547-random (self)                     2_922.20     1_053.49     3_975.69       0.5570     1.967407         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)              9_980.18        53.53    10_033.70       0.1999          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)              9_980.18        59.16    10_039.34       0.1986          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)              9_980.18        73.90    10_054.08       0.1973          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)              9_980.18        81.86    10_062.04       0.4075     3.755238         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)             9_980.18       108.50    10_088.68       0.5343     2.102086         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)             9_980.18       163.93    10_144.11       0.6698     1.087633         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)              9_980.18        90.05    10_070.23       0.4055     3.776912         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)             9_980.18       118.48    10_098.65       0.5322     2.113827         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)             9_980.18       167.67    10_147.84       0.6683     1.087712         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)              9_980.18       106.81    10_086.98       0.4027     3.813394         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)             9_980.18       138.83    10_119.00       0.5288     2.137900         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)             9_980.18       190.00    10_170.18       0.6646     1.105484         5.79
IVF-Binary-256-nl273-itq (self)                        9_980.18     1_183.95    11_164.12       0.5300     2.096063         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             10_483.29        55.11    10_538.40       0.1994          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             10_483.29        66.28    10_549.57       0.1977          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             10_483.29        84.80    10_568.09       0.4075     3.750131         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            10_483.29       109.47    10_592.76       0.5346     2.090951         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            10_483.29       156.60    10_639.89       0.6704     1.081536         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             10_483.29        97.02    10_580.31       0.4037     3.798110         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            10_483.29       128.05    10_611.34       0.5298     2.127606         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            10_483.29       173.66    10_656.95       0.6656     1.098403         5.80
IVF-Binary-256-nl387-itq (self)                       10_483.29     1_098.98    11_582.28       0.5320     2.081287         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             10_647.50        52.85    10_700.35       0.2007          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             10_647.50        56.99    10_704.49       0.1994          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             10_647.50        63.20    10_710.70       0.1982          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             10_647.50        80.66    10_728.16       0.4103     3.710174         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            10_647.50       105.70    10_753.20       0.5382     2.062483         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            10_647.50       150.46    10_797.96       0.6755     1.058102         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             10_647.50        84.67    10_732.17       0.4076     3.744299         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            10_647.50       111.52    10_759.02       0.5348     2.086236         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            10_647.50       157.56    10_805.06       0.6714     1.072275         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             10_647.50        91.47    10_738.97       0.4048     3.783199         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            10_647.50       122.40    10_769.90       0.5312     2.116777         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            10_647.50       168.90    10_816.40       0.6674     1.090905         5.82
IVF-Binary-256-nl547-itq (self)                       10_647.50     1_088.48    11_735.98       0.5363     2.045271         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_289.69       106.28     3_395.97       0.2929          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_289.69       121.15     3_410.85       0.2924          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_289.69       152.65     3_442.34       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_289.69       139.13     3_428.82       0.5866     1.777778        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_289.69       172.95     3_462.64       0.7255     0.834322        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_289.69       223.71     3_513.41       0.8433     0.358432        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_289.69       156.99     3_446.68       0.5864     1.774118        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_289.69       192.89     3_482.58       0.7262     0.823307        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_289.69       245.27     3_534.97       0.8455     0.341275        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_289.69       193.86     3_483.55       0.5849     1.784087        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_289.69       228.25     3_517.95       0.7249     0.828403        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_289.69       287.48     3_577.17       0.8447     0.342656        10.40
IVF-Binary-512-nl273-random (self)                     3_289.69     1_880.86     5_170.55       0.7247     0.820090        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_653.13       113.39     3_766.52       0.2924          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_653.13       138.97     3_792.10       0.2915          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_653.13       144.02     3_797.15       0.5871     1.768998        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_653.13       176.92     3_830.05       0.7264     0.828266        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_653.13       227.63     3_880.76       0.8449     0.354107        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_653.13       185.60     3_838.73       0.5850     1.780764        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_653.13       210.30     3_863.43       0.7251     0.826669        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_653.13       264.38     3_917.52       0.8452     0.340960        10.41
IVF-Binary-512-nl387-random (self)                     3_653.13     1_741.20     5_394.34       0.7248     0.827156        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_115.84       104.83     4_220.68       0.2936          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_115.84       114.11     4_229.95       0.2929          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_115.84       130.36     4_246.20       0.2921          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_115.84       135.43     4_251.27       0.5886     1.763517        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_115.84       165.94     4_281.78       0.7281     0.824131        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_115.84       214.59     4_330.44       0.8461     0.353593        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_115.84       148.21     4_264.05       0.5871     1.766939        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_115.84       182.14     4_297.98       0.7271     0.820926        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_115.84       231.56     4_347.40       0.8464     0.341065        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_115.84       162.77     4_278.61       0.5855     1.778571        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_115.84       196.86     4_312.71       0.7259     0.824482        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_115.84       249.50     4_365.34       0.8456     0.338961        10.43
IVF-Binary-512-nl547-random (self)                     4_115.84     1_635.77     5_751.61       0.7268     0.818374        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_108.57       110.22    11_218.79       0.2946          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_108.57       123.00    11_231.58       0.2938          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_108.57       165.15    11_273.73       0.2928          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_108.57       143.66    11_252.24       0.5861     1.674212        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_108.57       171.73    11_280.31       0.7222     0.788901        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_108.57       224.73    11_333.31       0.8380     0.337141        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_108.57       158.88    11_267.46       0.5858     1.669777        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_108.57       193.31    11_301.89       0.7231     0.777003        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_108.57       246.19    11_354.77       0.8404     0.318308        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_108.57       193.98    11_302.55       0.5845     1.678209        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_108.57       234.99    11_343.57       0.7216     0.781877        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_108.57       289.90    11_398.48       0.8395     0.319796        10.40
IVF-Binary-512-nl273-itq (self)                       11_108.57     1_899.48    13_008.06       0.7222     0.764851        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             11_509.08       113.11    11_622.19       0.2945          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             11_509.08       142.07    11_651.15       0.2931          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             11_509.08       152.12    11_661.20       0.5865     1.671161        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            11_509.08       175.05    11_684.13       0.7230     0.784775        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            11_509.08       237.57    11_746.65       0.8395     0.333336        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             11_509.08       185.68    11_694.76       0.5851     1.673364        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            11_509.08       209.47    11_718.55       0.7223     0.779235        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            11_509.08       265.96    11_775.04       0.8401     0.317695        10.41
IVF-Binary-512-nl387-itq (self)                       11_509.08     1_744.55    13_253.63       0.7220     0.775487        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             11_986.12       104.75    12_090.86       0.2950          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             11_986.12       113.98    12_100.09       0.2942          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             11_986.12       128.30    12_114.41       0.2936          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             11_986.12       136.93    12_123.05       0.5882     1.660796        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            11_986.12       164.56    12_150.67       0.7249     0.779454        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            11_986.12       213.46    12_199.58       0.8409     0.333185        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             11_986.12       145.74    12_131.85       0.5872     1.660440        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            11_986.12       175.68    12_161.79       0.7242     0.773961        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            11_986.12       227.30    12_213.42       0.8415     0.319869        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             11_986.12       162.01    12_148.13       0.5857     1.670277        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            11_986.12       194.59    12_180.71       0.7229     0.777236        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            11_986.12       248.02    12_234.13       0.8406     0.316972        10.43
IVF-Binary-512-nl547-itq (self)                       11_986.12     1_643.18    13_629.30       0.7241     0.767452        10.43
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
Exhaustive (query)                                         4.33     1_489.44     1_493.77       1.0000     0.000000        18.88
Exhaustive (self)                                          4.33    15_914.91    15_919.23       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_247.60       493.63     1_741.22       0.2159          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_247.60       545.65     1_793.25       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_247.60       605.91     1_853.51       0.5705     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_247.60       699.91     1_947.50       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                     1_247.60     6_137.71     7_385.31       0.5696     0.001382         4.61
ExhaustiveBinary-256-itq_no_rr (query)                 9_712.09       527.17    10_239.26       0.2068          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   9_712.09       553.79    10_265.88       0.4203     0.002652         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  9_712.09       610.63    10_322.72       0.5468     0.001484         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  9_712.09       712.34    10_424.44       0.6809     0.000767         4.61
ExhaustiveBinary-256-itq (self)                        9_712.09     6_388.25    16_100.34       0.5456     0.001474         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_532.56       851.45     3_384.01       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_532.56     1_026.54     3_559.11       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_532.56     1_138.07     3_670.63       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_532.56     1_083.54     3_616.10       0.8651     0.000202         9.22
ExhaustiveBinary-512-random (self)                     2_532.56    10_283.86    12_816.42       0.7519     0.000503         9.22
ExhaustiveBinary-512-itq_no_rr (query)                11_094.82       889.89    11_984.71       0.3144          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  11_094.82       895.22    11_990.04       0.6144     0.001065         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 11_094.82       938.44    12_033.26       0.7478     0.000491         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 11_094.82     1_042.25    12_137.07       0.8576     0.000201         9.22
ExhaustiveBinary-512-itq (self)                       11_094.82    10_086.80    21_181.62       0.7466     0.000485         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_129.49        53.00     2_182.50       0.2189          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_129.49        63.16     2_192.66       0.2178          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_129.49        78.20     2_207.69       0.2168          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_129.49        83.07     2_212.56       0.4458     0.002466         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_129.49       111.00     2_240.50       0.5771     0.001354         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_129.49       158.28     2_287.78       0.7144     0.000676         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_129.49        90.79     2_220.29       0.4440     0.002486         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_129.49       120.99     2_250.48       0.5757     0.001360         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_129.49       170.03     2_299.52       0.7130     0.000678         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_129.49       108.38     2_237.87       0.4416     0.002508         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_129.49       140.26     2_269.75       0.5727     0.001376         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_129.49       193.63     2_323.13       0.7099     0.000689         5.79
IVF-Binary-256-nl273-random (self)                     2_129.49     1_218.07     3_347.56       0.5746     0.001354         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_471.79        55.68     2_527.48       0.2189          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)           2_471.79        68.55     2_540.34       0.2173          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)           2_471.79        86.55     2_558.34       0.4455     0.002465         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)          2_471.79       111.96     2_583.76       0.5775     0.001347         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)          2_471.79       159.02     2_630.81       0.7149     0.000670         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)           2_471.79       100.04     2_571.83       0.4420     0.002502         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)          2_471.79       130.65     2_602.44       0.5735     0.001369         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)          2_471.79       179.00     2_650.79       0.7106     0.000685         5.81
IVF-Binary-256-nl387-random (self)                     2_471.79     1_116.18     3_587.97       0.5767     0.001341         5.81
IVF-Binary-256-nl547-np23-rf0-random (query)           2_969.11        53.78     3_022.89       0.2198          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)           2_969.11        58.46     3_027.57       0.2187          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)           2_969.11        65.35     3_034.45       0.2175          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)           2_969.11        83.44     3_052.55       0.4486     0.002436         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)          2_969.11       106.86     3_075.97       0.5810     0.001327         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)          2_969.11       152.34     3_121.44       0.7187     0.000658         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)           2_969.11        87.64     3_056.75       0.4459     0.002461         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)          2_969.11       113.47     3_082.57       0.5779     0.001343         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)          2_969.11       159.83     3_128.94       0.7158     0.000666         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)           2_969.11        94.48     3_063.58       0.4430     0.002491         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)          2_969.11       125.83     3_094.94       0.5745     0.001364         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)          2_969.11       170.88     3_139.99       0.7119     0.000680         5.83
IVF-Binary-256-nl547-random (self)                     2_969.11     1_071.27     4_040.38       0.5804     0.001321         5.83
IVF-Binary-256-nl273-np13-rf0-itq (query)             10_048.68        52.09    10_100.77       0.2104          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             10_048.68        58.64    10_107.32       0.2090          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             10_048.68        74.93    10_123.61       0.2078          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             10_048.68        82.76    10_131.44       0.4272     0.002589         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            10_048.68       108.01    10_156.68       0.5545     0.001445         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            10_048.68       154.74    10_203.42       0.6884     0.000746         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             10_048.68        88.47    10_137.15       0.4250     0.002610         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            10_048.68       117.83    10_166.50       0.5524     0.001455         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            10_048.68       166.72    10_215.40       0.6867     0.000749         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             10_048.68       106.05    10_154.73       0.4224     0.002635         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            10_048.68       138.82    10_187.50       0.5488     0.001474         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            10_048.68       190.70    10_239.38       0.6829     0.000762         5.79
IVF-Binary-256-nl273-itq (self)                       10_048.68     1_175.80    11_224.48       0.5511     0.001444         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             10_364.80        55.46    10_420.25       0.2102          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)             10_364.80        66.08    10_430.87       0.2084          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)             10_364.80        84.47    10_449.27       0.4271     0.002587         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)            10_364.80       109.89    10_474.69       0.5547     0.001439         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)            10_364.80       154.98    10_519.77       0.6888     0.000742         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)             10_364.80        97.16    10_461.96       0.4233     0.002623         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)            10_364.80       127.84    10_492.63       0.5501     0.001464         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)            10_364.80       174.23    10_539.03       0.6845     0.000754         5.81
IVF-Binary-256-nl387-itq (self)                       10_364.80     1_111.56    11_476.36       0.5535     0.001430         5.81
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_038.25        60.98    11_099.23       0.2114          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_038.25        60.52    11_098.77       0.2099          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_038.25        67.81    11_106.06       0.2087          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_038.25        83.84    11_122.09       0.4301     0.002556         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_038.25       108.78    11_147.04       0.5584     0.001417         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_038.25       152.85    11_191.10       0.6935     0.000724         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_038.25        90.22    11_128.47       0.4274     0.002583         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_038.25       116.17    11_154.43       0.5550     0.001435         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_038.25       160.98    11_199.23       0.6898     0.000734         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_038.25        97.27    11_135.52       0.4245     0.002612         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_038.25       125.46    11_163.71       0.5513     0.001457         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_038.25       173.16    11_211.42       0.6856     0.000750         5.83
IVF-Binary-256-nl547-itq (self)                       11_038.25     1_090.67    12_128.93       0.5576     0.001407         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_314.79       104.15     3_418.94       0.3155          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_314.79       126.01     3_440.80       0.3152          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_314.79       153.30     3_468.09       0.3144          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_314.79       139.52     3_454.31       0.6203     0.001095        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_314.79       172.68     3_487.48       0.7548     0.000506        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_314.79       232.00     3_546.79       0.8636     0.000215        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_314.79       156.19     3_470.98       0.6205     0.001091        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_314.79       192.13     3_506.92       0.7560     0.000497        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_314.79       259.12     3_573.91       0.8664     0.000202        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_314.79       195.75     3_510.54       0.6191     0.001099        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_314.79       232.52     3_547.31       0.7548     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_314.79       297.49     3_612.28       0.8659     0.000201        10.40
IVF-Binary-512-nl273-random (self)                     3_314.79     1_907.50     5_222.29       0.7540     0.000497        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_668.80       108.95     3_777.75       0.3156          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_668.80       136.30     3_805.09       0.3145          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_668.80       143.51     3_812.31       0.6206     0.001092        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_668.80       182.05     3_850.85       0.7557     0.000502        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_668.80       235.00     3_903.79       0.8651     0.000210        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_668.80       174.33     3_843.13       0.6191     0.001097        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_668.80       209.18     3_877.98       0.7553     0.000498        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_668.80       273.38     3_942.18       0.8660     0.000201        10.41
IVF-Binary-512-nl387-random (self)                     3_668.80     1_768.74     5_437.54       0.7538     0.000502        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_173.48       101.53     4_275.01       0.3162          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)           4_173.48       111.27     4_284.75       0.3154          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)           4_173.48       126.30     4_299.77       0.3145          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)           4_173.48       136.32     4_309.79       0.6227     0.001081        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)          4_173.48       168.97     4_342.45       0.7573     0.000497        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)          4_173.48       222.42     4_395.90       0.8660     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)           4_173.48       147.07     4_320.55       0.6215     0.001085        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)          4_173.48       180.69     4_354.17       0.7570     0.000493        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)          4_173.48       238.42     4_411.89       0.8670     0.000201        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)           4_173.48       174.77     4_348.25       0.6199     0.001093        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)          4_173.48       198.45     4_371.93       0.7556     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)          4_173.48       259.09     4_432.56       0.8665     0.000200        10.44
IVF-Binary-512-nl547-random (self)                     4_173.48     1_690.04     5_863.52       0.7555     0.000498        10.44
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_237.76       106.63    11_344.38       0.3164          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_237.76       119.38    11_357.13       0.3158          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_237.76       154.69    11_392.44       0.3148          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_237.76       140.52    11_378.28       0.6167     0.001060        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_237.76       174.10    11_411.86       0.7491     0.000495        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_237.76       238.40    11_476.15       0.8565     0.000213        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_237.76       156.87    11_394.63       0.6167     0.001056        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_237.76       193.11    11_430.86       0.7504     0.000485        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_237.76       256.76    11_494.51       0.8594     0.000199        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_237.76       198.05    11_435.81       0.6154     0.001061        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_237.76       234.29    11_472.04       0.7489     0.000488        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_237.76       301.49    11_539.24       0.8585     0.000199        10.40
IVF-Binary-512-nl273-itq (self)                       11_237.76     1_944.83    13_182.58       0.7488     0.000479        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             11_638.53       106.39    11_744.92       0.3165          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             11_638.53       133.72    11_772.25       0.3152          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             11_638.53       142.86    11_781.39       0.6171     0.001056        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            11_638.53       175.88    11_814.41       0.7498     0.000491        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            11_638.53       233.87    11_872.40       0.8580     0.000208        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             11_638.53       179.94    11_818.47       0.6161     0.001057        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            11_638.53       208.39    11_846.92       0.7493     0.000487        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            11_638.53       273.38    11_911.91       0.8588     0.000199        10.41
IVF-Binary-512-nl387-itq (self)                       11_638.53     1_888.92    13_527.45       0.7486     0.000485        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             12_238.69       101.20    12_339.89       0.3173          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)             12_238.69       111.04    12_349.73       0.3165          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)             12_238.69       125.71    12_364.40       0.3156          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)             12_238.69       134.65    12_373.34       0.6189     0.001047        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)            12_238.69       168.10    12_406.79       0.7517     0.000487        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)            12_238.69       221.96    12_460.65       0.8590     0.000208        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)             12_238.69       145.33    12_384.02       0.6181     0.001048        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)            12_238.69       180.04    12_418.73       0.7513     0.000483        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)            12_238.69       236.10    12_474.79       0.8601     0.000199        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)             12_238.69       164.83    12_403.52       0.6166     0.001055        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)            12_238.69       196.57    12_435.26       0.7498     0.000486        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)            12_238.69       257.88    12_496.56       0.8594     0.000198        10.44
IVF-Binary-512-nl547-itq (self)                       12_238.69     1_676.83    13_915.52       0.7503     0.000482        10.44
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
Exhaustive (query)                                         3.00     1_608.44     1_611.44       1.0000     0.000000        18.31
Exhaustive (self)                                          3.00    16_495.26    16_498.26       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_249.33       492.81     1_742.14       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_249.33       544.47     1_793.79       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_249.33       592.25     1_841.57       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_249.33       684.82     1_934.15       0.5614     0.256662         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random (self)                     1_249.33     5_907.29     7_156.62       0.4294     0.437191         4.61
ExhaustiveBinary-256-itq_no_rr (query)                 9_174.21       491.52     9_665.73       0.1197          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   9_174.21       527.58     9_701.79       0.2765     0.831692         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  9_174.21       582.43     9_756.65       0.3861     0.506502         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  9_174.21       668.76     9_842.97       0.5230     0.288077         4.61
ExhaustiveBinary-256-itq (self)                        9_174.21     5_753.64    14_927.86       0.3927     0.490943         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_452.85       833.52     3_286.36       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_452.85       876.28     3_329.13       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_452.85       928.48     3_381.33       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_452.85     1_038.71     3_491.56       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                     2_452.85     9_382.31    11_835.16       0.6185     0.184365         9.22
ExhaustiveBinary-512-itq_no_rr (query)                10_342.76       828.20    11_170.96       0.2230          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  10_342.76       992.07    11_334.83       0.4733     0.353636         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 10_342.76     1_139.92    11_482.68       0.6117     0.187548         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 10_342.76     1_220.50    11_563.26       0.7467     0.091104         9.22
ExhaustiveBinary-512-itq (self)                       10_342.76    10_616.56    20_959.32       0.6142     0.183153         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_241.20        51.25     2_292.45       0.1443          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_241.20        59.64     2_300.84       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_241.20        85.90     2_327.10       0.1371          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_241.20        77.77     2_318.97       0.3253     0.693086         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_241.20       119.28     2_360.49       0.4471     0.412061         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_241.20       159.33     2_400.54       0.5876     0.229141         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_241.20        97.06     2_338.26       0.3165     0.721914         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_241.20       113.18     2_354.38       0.4368     0.431051         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_241.20       155.61     2_396.81       0.5764     0.241195         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_241.20       107.01     2_348.22       0.3123     0.739522         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_241.20       138.21     2_379.41       0.4312     0.443199         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_241.20       193.29     2_434.50       0.5697     0.249326         5.79
IVF-Binary-256-nl273-random (self)                     2_241.20     1_094.17     3_335.37       0.4422     0.416770         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_691.12        60.02     2_751.14       0.1403          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_691.12        66.98     2_758.10       0.1380          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_691.12        86.43     2_777.55       0.3203     0.707597         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_691.12       129.82     2_820.94       0.4414     0.420873         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_691.12       186.54     2_877.66       0.5825     0.233592         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_691.12       102.96     2_794.08       0.3145     0.730872         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_691.12       140.02     2_831.14       0.4334     0.437743         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_691.12       200.44     2_891.56       0.5726     0.245415         5.80
IVF-Binary-256-nl387-random (self)                     2_691.12     1_166.61     3_857.73       0.4467     0.406727         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           3_052.93        59.75     3_112.68       0.1414          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           3_052.93        66.29     3_119.22       0.1400          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           3_052.93        71.91     3_124.85       0.1386          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           3_052.93        88.89     3_141.83       0.3224     0.697059         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          3_052.93       116.13     3_169.06       0.4446     0.412600         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          3_052.93       150.80     3_203.74       0.5866     0.227694         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           3_052.93        84.13     3_137.07       0.3186     0.712202         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          3_052.93       107.16     3_160.10       0.4393     0.423748         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          3_052.93       144.36     3_197.29       0.5799     0.235834         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           3_052.93        88.34     3_141.27       0.3148     0.727786         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          3_052.93       114.47     3_167.40       0.4343     0.435387         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          3_052.93       161.92     3_214.85       0.5738     0.243853         5.82
IVF-Binary-256-nl547-random (self)                     3_052.93       983.23     4_036.16       0.4498     0.399477         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)             16_093.08        53.10    16_146.19       0.1299          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             16_093.08        58.50    16_151.59       0.1239          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             16_093.08        77.68    16_170.76       0.1220          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             16_093.08        83.11    16_176.19       0.2980     0.760619         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            16_093.08        96.96    16_190.05       0.4143     0.455526         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            16_093.08       138.56    16_231.64       0.5536     0.256381         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             16_093.08        81.89    16_174.98       0.2866     0.799417         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            16_093.08       108.06    16_201.14       0.4006     0.481416         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            16_093.08       162.28    16_255.36       0.5395     0.272093         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             16_093.08       105.11    16_198.20       0.2825     0.819698         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            16_093.08       134.93    16_228.02       0.3941     0.496934         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            16_093.08       198.88    16_291.96       0.5322     0.282010         5.79
IVF-Binary-256-nl273-itq (self)                       16_093.08     1_178.02    17_271.11       0.4069     0.467068         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             10_828.90        52.39    10_881.28       0.1259          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             10_828.90        63.44    10_892.34       0.1230          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             10_828.90        75.52    10_904.42       0.2906     0.783300         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            10_828.90       103.51    10_932.41       0.4060     0.469739         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            10_828.90       142.77    10_971.66       0.5456     0.264006         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             10_828.90        89.00    10_917.90       0.2835     0.815136         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            10_828.90       115.94    10_944.84       0.3962     0.492780         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            10_828.90       164.36    10_993.25       0.5336     0.279398         5.80
IVF-Binary-256-nl387-itq (self)                       10_828.90     1_071.78    11_900.68       0.4117     0.456742         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_633.78        63.86    11_697.64       0.1260          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_633.78        73.22    11_707.00       0.1239          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_633.78        81.41    11_715.19       0.1224          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_633.78       101.44    11_735.22       0.2928     0.771548         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_633.78       125.92    11_759.70       0.4099     0.459383         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_633.78       167.12    11_800.90       0.5505     0.256651         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_633.78       101.21    11_734.99       0.2882     0.792145         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_633.78       131.70    11_765.48       0.4034     0.473440         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_633.78       185.86    11_819.64       0.5420     0.266865         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_633.78       103.27    11_737.05       0.2837     0.810164         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_633.78       128.97    11_762.74       0.3981     0.485662         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_633.78       196.28    11_830.06       0.5354     0.274889         5.82
IVF-Binary-256-nl547-itq (self)                       11_633.78     1_095.49    12_729.27       0.4154     0.447606         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_431.23       129.95     3_561.18       0.2304          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_431.23       159.41     3_590.64       0.2278          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_431.23       153.37     3_584.60       0.2270          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_431.23       128.22     3_559.45       0.4860     0.337755        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_431.23       197.68     3_628.91       0.6273     0.176939        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_431.23       220.09     3_651.33       0.7652     0.083691        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_431.23       142.98     3_574.22       0.4812     0.344920        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_431.23       186.98     3_618.21       0.6224     0.181309        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_431.23       249.40     3_680.63       0.7602     0.086526        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_431.23       181.10     3_612.33       0.4791     0.349097        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_431.23       221.43     3_652.66       0.6194     0.184459        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_431.23       304.76     3_735.99       0.7568     0.088670        10.40
IVF-Binary-512-nl273-random (self)                     3_431.23     1_811.30     5_242.53       0.6260     0.177548        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_701.01       103.08     3_804.09       0.2289          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_701.01       146.64     3_847.65       0.2274          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_701.01       162.05     3_863.06       0.4838     0.341431        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_701.01       232.66     3_933.68       0.6253     0.178808        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_701.01       245.30     3_946.31       0.7626     0.084984        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_701.01       175.17     3_876.18       0.4809     0.347136        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_701.01       197.43     3_898.44       0.6210     0.183003        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_701.01       251.07     3_952.08       0.7578     0.087874        10.41
IVF-Binary-512-nl387-random (self)                     3_701.01     1_654.96     5_355.98       0.6282     0.175444        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_209.01        95.54     4_304.55       0.2295          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_209.01       104.21     4_313.22       0.2286          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_209.01       116.18     4_325.19       0.2277          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_209.01       126.43     4_335.44       0.4844     0.340126        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_209.01       154.04     4_363.06       0.6262     0.177884        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_209.01       200.36     4_409.37       0.7650     0.083907        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_209.01       134.32     4_343.34       0.4822     0.344431        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_209.01       163.62     4_372.63       0.6234     0.180818        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_209.01       212.20     4_421.21       0.7614     0.086089        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_209.01       152.53     4_361.54       0.4802     0.348094        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_209.01       179.94     4_388.95       0.6210     0.183289        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_209.01       237.60     4_446.61       0.7585     0.087937        10.43
IVF-Binary-512-nl547-random (self)                     4_209.01     1_519.85     5_728.86       0.6297     0.174184        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             13_457.90       120.86    13_578.76       0.2291          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             13_457.90       128.07    13_585.97       0.2262          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             13_457.90       153.88    13_611.79       0.2253          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             13_457.90       144.00    13_601.90       0.4843     0.336509        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            13_457.90       182.42    13_640.32       0.6241     0.176585        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            13_457.90       235.63    13_693.53       0.7591     0.084287        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             13_457.90       152.88    13_610.78       0.4795     0.344009        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            13_457.90       196.00    13_653.90       0.6188     0.181286        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            13_457.90       257.48    13_715.38       0.7539     0.087383        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             13_457.90       202.38    13_660.28       0.4771     0.348313        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            13_457.90       233.00    13_690.90       0.6158     0.184229        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            13_457.90       298.59    13_756.50       0.7506     0.089206        10.40
IVF-Binary-512-nl273-itq (self)                       13_457.90     1_861.28    15_319.18       0.6214     0.177187        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_420.04       104.38    12_524.42       0.2270          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_420.04       133.04    12_553.08       0.2254          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_420.04       146.90    12_566.94       0.4814     0.341319        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_420.04       168.29    12_588.33       0.6208     0.179146        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_420.04       213.23    12_633.27       0.7560     0.085918        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_420.04       162.21    12_582.25       0.4779     0.347485        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_420.04       192.86    12_612.90       0.6167     0.183252        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_420.04       245.88    12_665.92       0.7515     0.088753        10.41
IVF-Binary-512-nl387-itq (self)                       12_420.04     1_705.24    14_125.28       0.6234     0.175207        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             14_503.15        96.91    14_600.07       0.2275          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             14_503.15       104.80    14_607.95       0.2265          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             14_503.15       117.87    14_621.02       0.2256          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             14_503.15       127.80    14_630.96       0.4828     0.338622        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            14_503.15       152.97    14_656.13       0.6224     0.177249        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            14_503.15       201.33    14_704.49       0.7590     0.084001        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             14_503.15       136.31    14_639.46       0.4801     0.343195        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            14_503.15       164.37    14_667.53       0.6191     0.180399        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            14_503.15       212.50    14_715.66       0.7552     0.086330        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             14_503.15       149.99    14_653.14       0.4780     0.346884        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            14_503.15       180.42    14_683.58       0.6168     0.182693        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            14_503.15       232.69    14_735.84       0.7527     0.087837        10.43
IVF-Binary-512-nl547-itq (self)                       14_503.15     1_535.90    16_039.05       0.6257     0.173003        10.43
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
Exhaustive (query)                                         3.15     1_581.16     1_584.31       1.0000     0.000000        18.31
Exhaustive (self)                                          3.15    16_590.94    16_594.09       1.0000     0.000000        18.31
ExhaustiveBinary-256-random_no_rr (query)              1_259.18       512.08     1_771.26       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_259.18       541.39     1_800.57       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_259.18       587.02     1_846.20       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_259.18       696.71     1_955.89       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                     1_259.18     5_893.75     7_152.93       0.3238     3.208083         4.61
ExhaustiveBinary-256-itq_no_rr (query)                 9_358.84       488.10     9_846.94       0.0734          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   9_358.84       525.41     9_884.25       0.1955     5.709191         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  9_358.84       636.19     9_995.04       0.2902     3.600771         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  9_358.84       724.61    10_083.45       0.4157     2.167224         4.61
ExhaustiveBinary-256-itq (self)                        9_358.84     6_267.49    15_626.33       0.2960     3.485945         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_522.42       877.43     3_399.85       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_522.42       947.26     3_469.68       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_522.42     1_106.54     3_628.96       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_522.42     1_186.28     3_708.70       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                     2_522.42     9_603.13    12_125.54       0.4788     1.710526         9.22
ExhaustiveBinary-512-itq_no_rr (query)                11_969.28       846.09    12_815.37       0.1518          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  11_969.28       920.90    12_890.18       0.3364     3.092957         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 11_969.28       933.71    12_902.99       0.4608     1.824577         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 11_969.28     1_033.10    13_002.38       0.6062     1.008706         9.22
ExhaustiveBinary-512-itq (self)                       11_969.28     9_739.25    21_708.53       0.4642     1.796793         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_152.93        51.99     2_204.92       0.1015          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_152.93        58.41     2_211.34       0.0925          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_152.93        72.84     2_225.77       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_152.93        76.35     2_229.28       0.2466     4.683539         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_152.93        94.99     2_247.92       0.3527     2.916659         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_152.93       135.23     2_288.16       0.4886     1.698542         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_152.93        81.32     2_234.25       0.2317     5.167780         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_152.93       104.80     2_257.73       0.3353     3.191328         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_152.93       149.10     2_302.03       0.4700     1.862975         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_152.93       150.66     2_303.60       0.2280     5.297650         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_152.93       155.81     2_308.75       0.3304     3.274626         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_152.93       202.72     2_355.65       0.4637     1.916709         5.79
IVF-Binary-256-nl273-random (self)                     2_152.93     1_161.65     3_314.58       0.3387     3.130514         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           2_643.44        57.12     2_700.57       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           2_643.44        69.54     2_712.98       0.0919          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           2_643.44        78.67     2_722.11       0.2358     4.931507         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          2_643.44       109.76     2_753.21       0.3421     3.052683         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          2_643.44       210.30     2_853.75       0.4777     1.802324         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           2_643.44       116.28     2_759.72       0.2299     5.156744         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          2_643.44       119.23     2_762.67       0.3340     3.196200         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          2_643.44       162.85     2_806.29       0.4679     1.888188         5.80
IVF-Binary-256-nl387-random (self)                     2_643.44     1_007.51     3_650.95       0.3448     3.003180         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           2_960.53        51.55     3_012.08       0.0954          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           2_960.53        53.58     3_014.11       0.0934          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           2_960.53        63.29     3_023.82       0.0917          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           2_960.53        71.51     3_032.04       0.2397     4.792322         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          2_960.53        96.27     3_056.80       0.3460     2.991438         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          2_960.53       133.19     3_093.72       0.4824     1.757612         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           2_960.53        75.24     3_035.77       0.2354     4.927104         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          2_960.53        98.07     3_058.60       0.3399     3.085668         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          2_960.53       138.25     3_098.78       0.4749     1.823328         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           2_960.53        82.01     3_042.54       0.2310     5.077596         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          2_960.53       105.78     3_066.31       0.3339     3.187068         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          2_960.53       146.22     3_106.75       0.4678     1.889583         5.82
IVF-Binary-256-nl547-random (self)                     2_960.53       931.13     3_891.66       0.3483     2.939393         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)             13_702.51        51.80    13_754.31       0.0873          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             13_702.51        54.39    13_756.90       0.0772          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             13_702.51        68.45    13_770.96       0.0756          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             13_702.51        70.55    13_773.05       0.2216     5.282755         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            13_702.51        93.96    13_796.47       0.3219     3.305650         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            13_702.51       131.40    13_833.91       0.4512     1.955763         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             13_702.51        78.90    13_781.40       0.2066     5.679316         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            13_702.51       101.93    13_804.44       0.3048     3.544723         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            13_702.51       143.87    13_846.38       0.4341     2.092938         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             13_702.51        92.94    13_795.45       0.2033     5.811737         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            13_702.51       123.76    13_826.26       0.3004     3.632149         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            13_702.51       164.87    13_867.37       0.4284     2.147688         5.79
IVF-Binary-256-nl273-itq (self)                       13_702.51     1_025.72    14_728.23       0.3111     3.432551         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             10_292.38        52.79    10_345.16       0.0798          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             10_292.38        62.67    10_355.05       0.0767          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             10_292.38        71.61    10_363.99       0.2117     5.502615         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            10_292.38        93.22    10_385.60       0.3111     3.415891         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            10_292.38       132.12    10_424.50       0.4426     2.008934         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             10_292.38        84.15    10_376.53       0.2050     5.728015         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            10_292.38       108.19    10_400.57       0.3029     3.552823         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            10_292.38       148.26    10_440.64       0.4328     2.095211         5.80
IVF-Binary-256-nl387-itq (self)                       10_292.38       932.28    11_224.66       0.3167     3.315809         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             10_782.21        49.55    10_831.76       0.0812          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             10_782.21        54.02    10_836.23       0.0790          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             10_782.21        59.53    10_841.74       0.0767          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             10_782.21        71.17    10_853.38       0.2158     5.309238         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            10_782.21        91.71    10_873.91       0.3164     3.309382         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            10_782.21       131.21    10_913.41       0.4486     1.951064         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             10_782.21        73.90    10_856.11       0.2107     5.492567         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            10_782.21       101.83    10_884.04       0.3095     3.427927         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            10_782.21       143.66    10_925.87       0.4404     2.024259         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             10_782.21        82.10    10_864.31       0.2062     5.639763         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            10_782.21       106.62    10_888.83       0.3036     3.522671         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            10_782.21       142.95    10_925.16       0.4332     2.084999         5.82
IVF-Binary-256-nl547-itq (self)                       10_782.21       968.59    11_750.80       0.3219     3.220474         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_290.57       100.25     3_390.82       0.1685          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_290.57       112.61     3_403.18       0.1642          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_290.57       143.35     3_433.92       0.1629          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_290.57       128.25     3_418.82       0.3648     2.762852        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_290.57       154.87     3_445.44       0.4946     1.613586        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_290.57       200.78     3_491.35       0.6428     0.865216        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_290.57       144.01     3_434.58       0.3574     2.870953        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_290.57       172.69     3_463.26       0.4866     1.678608        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_290.57       222.24     3_512.81       0.6348     0.903671        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_290.57       175.80     3_466.37       0.3551     2.903769        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_290.57       209.09     3_499.66       0.4839     1.699072        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_290.57       263.00     3_553.57       0.6317     0.917169        10.40
IVF-Binary-512-nl273-random (self)                     3_290.57     1_728.98     5_019.55       0.4885     1.655176        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           3_633.60       104.19     3_737.79       0.1647          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           3_633.60       129.37     3_762.97       0.1631          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           3_633.60       132.83     3_766.43       0.3603     2.826319        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          3_633.60       160.72     3_794.32       0.4905     1.642435        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          3_633.60       206.17     3_839.77       0.6385     0.885603        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           3_633.60       159.50     3_793.09       0.3567     2.886554        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          3_633.60       188.98     3_822.58       0.4856     1.679814        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          3_633.60       237.94     3_871.54       0.6332     0.909694        10.41
IVF-Binary-512-nl387-random (self)                     3_633.60     1_597.78     5_231.38       0.4917     1.624176        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           4_137.92       100.79     4_238.71       0.1660          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           4_137.92       108.59     4_246.51       0.1648          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           4_137.92       122.52     4_260.44       0.1637          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           4_137.92       127.71     4_265.63       0.3623     2.788844        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          4_137.92       152.38     4_290.30       0.4927     1.618053        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          4_137.92       196.09     4_334.01       0.6419     0.867394        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           4_137.92       137.04     4_274.96       0.3595     2.836677        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          4_137.92       180.45     4_318.36       0.4890     1.650040        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          4_137.92       209.17     4_347.09       0.6371     0.891042        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           4_137.92       151.76     4_289.68       0.3568     2.876216        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          4_137.92       178.91     4_316.83       0.4854     1.678098        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          4_137.92       226.46     4_364.38       0.6333     0.908544        10.43
IVF-Binary-512-nl547-random (self)                     4_137.92     1_517.15     5_655.07       0.4940     1.602593        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_352.13        99.19    11_451.32       0.1599          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_352.13       110.21    11_462.34       0.1554          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_352.13       142.13    11_494.27       0.1545          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_352.13       127.00    11_479.13       0.3516     2.904447        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_352.13       153.17    11_505.30       0.4794     1.698773        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_352.13       198.71    11_550.84       0.6251     0.933239        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_352.13       140.06    11_492.19       0.3442     3.008673        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_352.13       170.02    11_522.15       0.4709     1.762526        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_352.13       219.58    11_571.71       0.6173     0.967870        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_352.13       172.08    11_524.21       0.3421     3.040991        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_352.13       203.98    11_556.12       0.4680     1.783083        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_352.13       265.70    11_617.83       0.6141     0.981829        10.40
IVF-Binary-512-nl273-itq (self)                       11_352.13     1_694.65    13_046.78       0.4743     1.736368        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             11_372.36       106.26    11_478.62       0.1565          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             11_372.36       130.09    11_502.45       0.1547          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             11_372.36       133.75    11_506.10       0.3474     2.963602        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            11_372.36       160.00    11_532.35       0.4743     1.734979        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            11_372.36       206.34    11_578.70       0.6202     0.953618        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             11_372.36       160.21    11_532.57       0.3441     3.017253        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            11_372.36       189.23    11_561.58       0.4700     1.772640        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            11_372.36       238.05    11_610.41       0.6155     0.976692        10.41
IVF-Binary-512-nl387-itq (self)                       11_372.36     1_595.14    12_967.49       0.4776     1.709669        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             11_985.80       100.97    12_086.78       0.1576          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             11_985.80       118.48    12_104.29       0.1561          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             11_985.80       121.67    12_107.47       0.1550          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             11_985.80       126.93    12_112.74       0.3489     2.929000        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            11_985.80       151.77    12_137.58       0.4768     1.713785        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            11_985.80       197.07    12_182.87       0.6237     0.934706        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             11_985.80       136.23    12_122.03       0.3460     2.980107        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            11_985.80       167.87    12_153.68       0.4725     1.751391        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            11_985.80       207.51    12_193.31       0.6191     0.956698        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             11_985.80       150.32    12_136.12       0.3438     3.014642        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            11_985.80       177.39    12_163.19       0.4696     1.775743        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            11_985.80       226.16    12_211.96       0.6157     0.972174        10.43
IVF-Binary-512-nl547-itq (self)                       11_985.80     1_514.43    13_500.23       0.4800     1.688610        10.43
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With higher dimensionality

The compression performance increases; however, the Recall clearly suffers
with higher dimensions. Potentially this can be mitigated with providing more
bits.

<details>
<summary><b>Binary - Euclidean (Gaussian - higher dimensionality)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.10     6_220.39     6_234.49       1.0000     0.000000        73.24
Exhaustive (self)                                         14.10    65_083.57    65_097.67       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_921.15       541.49     4_462.64       0.1084          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_921.15       626.67     4_547.82       0.1994    32.157948         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_921.15       755.48     4_676.63       0.2740    21.742093         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_921.15       905.33     4_826.48       0.3781    14.028739         4.70
ExhaustiveBinary-256-random (self)                     3_921.15     6_839.61    10_760.76       0.2729    21.873275         4.70
ExhaustiveBinary-256-itq_no_rr (query)                21_329.30       530.03    21_859.33       0.0945          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  21_329.30       574.26    21_903.56       0.1625    37.472052         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 21_329.30       648.32    21_977.62       0.2221    26.042140         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 21_329.30       789.57    22_118.87       0.3085    17.458979         4.70
ExhaustiveBinary-256-itq (self)                       21_329.30     6_765.21    28_094.51       0.2220    26.221286         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_734.60       904.19     8_638.79       0.1437          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_734.60       960.39     8_694.98       0.2864    21.279597         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_734.60     1_012.96     8_747.56       0.3868    13.379880         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_734.60     1_129.52     8_864.12       0.5137     7.900976         9.41
ExhaustiveBinary-512-random (self)                     7_734.60    10_112.55    17_847.15       0.3866    13.445004         9.41
ExhaustiveBinary-512-itq_no_rr (query)                21_780.10       898.33    22_678.43       0.1367          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  21_780.10       946.85    22_726.95       0.2673    22.229428         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 21_780.10     1_003.81    22_783.90       0.3636    14.093214         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 21_780.10     1_133.39    22_913.49       0.4852     8.407298         9.41
ExhaustiveBinary-512-itq (self)                       21_780.10    10_266.74    32_046.84       0.3631    14.195195         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_212.40     1_517.52    16_729.92       0.2051          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_212.40     1_604.55    16_816.96       0.4198    11.866665        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_212.40     1_901.96    17_114.36       0.5478     6.608533        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_212.40     1_876.36    17_088.76       0.6844     3.360088        18.81
ExhaustiveBinary-1024-random (self)                   15_212.40    16_703.55    31_915.96       0.5467     6.665345        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               29_500.80     1_716.92    31_217.72       0.2061          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 29_500.80     1_749.40    31_250.20       0.4209    11.394439        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                29_500.80     1_726.90    31_227.70       0.5491     6.304691        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                29_500.80     1_855.63    31_356.43       0.6837     3.180706        18.81
ExhaustiveBinary-1024-itq (self)                      29_500.80    17_392.05    46_892.84       0.5479     6.374041        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           6_905.42        93.26     6_998.68       0.1096          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           6_905.42        95.63     7_001.05       0.1090          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           6_905.42       110.38     7_015.80       0.1084          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           6_905.42       134.30     7_039.72       0.2030    31.751737         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          6_905.42       180.97     7_086.40       0.2783    21.451970         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          6_905.42       248.56     7_153.99       0.3831    13.845393         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           6_905.42       146.04     7_051.47       0.2018    31.841557         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          6_905.42       176.57     7_082.00       0.2772    21.495384         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          6_905.42       249.06     7_154.49       0.3822    13.865485         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           6_905.42       157.87     7_063.29       0.2002    32.034661         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          6_905.42       218.42     7_123.84       0.2754    21.644438         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          6_905.42       284.39     7_189.81       0.3797    13.961364         5.98
IVF-Binary-256-nl273-random (self)                     6_905.42     1_795.37     8_700.80       0.2761    21.639918         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)           7_831.69        94.42     7_926.11       0.1094          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           7_831.69       109.13     7_940.82       0.1087          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           7_831.69       132.50     7_964.20       0.2028    31.696102         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          7_831.69       169.68     8_001.38       0.2786    21.406275         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          7_831.69       239.57     8_071.27       0.3833    13.805239         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           7_831.69       143.24     7_974.93       0.2007    31.963751         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          7_831.69       182.47     8_014.16       0.2760    21.607746         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          7_831.69       258.98     8_090.68       0.3804    13.920033         6.04
IVF-Binary-256-nl387-random (self)                     7_831.69     1_693.94     9_525.64       0.2773    21.534373         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)           9_379.59        95.40     9_474.99       0.1100          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           9_379.59       100.09     9_479.68       0.1096          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           9_379.59       109.30     9_488.89       0.1093          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           9_379.59       133.36     9_512.95       0.2038    31.573320         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          9_379.59       170.38     9_549.97       0.2797    21.303754         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          9_379.59       236.48     9_616.07       0.3853    13.729825         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           9_379.59       136.08     9_515.67       0.2028    31.729254         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          9_379.59       175.79     9_555.38       0.2788    21.394742         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          9_379.59       244.75     9_624.34       0.3841    13.777109         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           9_379.59       143.13     9_522.72       0.2016    31.877463         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          9_379.59       184.03     9_563.62       0.2771    21.517933         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          9_379.59       256.91     9_636.50       0.3815    13.885112         6.12
IVF-Binary-256-nl547-random (self)                     9_379.59     1_709.47    11_089.06       0.2790    21.421506         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             29_780.77        95.01    29_875.78       0.0965          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             29_780.77        97.77    29_878.54       0.0958          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             29_780.77       117.73    29_898.50       0.0950          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             29_780.77       126.94    29_907.71       0.1665    36.739101         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            29_780.77       164.71    29_945.48       0.2275    25.483066         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            29_780.77       244.46    30_025.23       0.3162    17.032533         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             29_780.77       137.25    29_918.02       0.1650    36.979770         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            29_780.77       187.16    29_967.93       0.2257    25.658442         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            29_780.77       286.64    30_067.41       0.3139    17.150469         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             29_780.77       174.86    29_955.63       0.1636    37.284541         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            29_780.77       191.72    29_972.49       0.2234    25.907187         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            29_780.77       283.13    30_063.90       0.3105    17.343394         5.98
IVF-Binary-256-nl273-itq (self)                       29_780.77     2_046.52    31_827.28       0.2259    25.817802         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             35_598.79        96.33    35_695.12       0.0961          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             35_598.79       105.74    35_704.53       0.0953          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             35_598.79       139.43    35_738.23       0.1664    36.733791         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            35_598.79       170.11    35_768.90       0.2270    25.500114         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            35_598.79       235.19    35_833.98       0.3156    17.018354         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             35_598.79       140.58    35_739.38       0.1643    37.153489         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            35_598.79       178.78    35_777.58       0.2241    25.807327         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            35_598.79       253.00    35_851.80       0.3116    17.249116         6.04
IVF-Binary-256-nl387-itq (self)                       35_598.79     1_656.26    37_255.05       0.2273    25.650711         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             26_355.10       106.23    26_461.33       0.0968          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             26_355.10       109.46    26_464.55       0.0961          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             26_355.10       106.11    26_461.21       0.0956          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             26_355.10       135.76    26_490.85       0.1675    36.528420         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            26_355.10       191.41    26_546.51       0.2290    25.330151         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            26_355.10       308.46    26_663.56       0.3183    16.888824         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             26_355.10       142.18    26_497.28       0.1663    36.754181         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            26_355.10       174.68    26_529.78       0.2274    25.501755         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            26_355.10       295.39    26_650.48       0.3161    17.009611         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             26_355.10       167.00    26_522.09       0.1648    37.012561         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            26_355.10       224.82    26_579.91       0.2251    25.731574         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            26_355.10       276.56    26_631.66       0.3130    17.178192         6.12
IVF-Binary-256-nl547-itq (self)                       26_355.10     1_729.34    28_084.44       0.2292    25.464976         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          10_657.71       180.37    10_838.08       0.1448          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          10_657.71       196.79    10_854.50       0.1444          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          10_657.71       232.69    10_890.40       0.1441          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          10_657.71       234.22    10_891.93       0.2872    21.249805        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         10_657.71       275.24    10_932.94       0.3875    13.437040        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         10_657.71       345.23    11_002.94       0.5122     8.034049        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          10_657.71       250.75    10_908.45       0.2873    21.194718        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         10_657.71       297.36    10_955.07       0.3888    13.330334        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         10_657.71       372.33    11_030.03       0.5146     7.902241        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          10_657.71       289.24    10_946.94       0.2867    21.236842        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         10_657.71       341.41    10_999.12       0.3878    13.344013        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         10_657.71       424.83    11_082.54       0.5144     7.878926        10.69
IVF-Binary-512-nl273-random (self)                    10_657.71     2_971.85    13_629.56       0.3881    13.397683        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          11_604.74       186.65    11_791.39       0.1448          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          11_604.74       219.07    11_823.81       0.1443          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          11_604.74       252.05    11_856.79       0.2876    21.174211        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         11_604.74       288.82    11_893.56       0.3887    13.364617        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         11_604.74       363.48    11_968.22       0.5143     7.941332        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          11_604.74       283.55    11_888.28       0.2869    21.212750        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         11_604.74       326.92    11_931.66       0.3884    13.326710        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         11_604.74       461.05    12_065.79       0.5150     7.862487        10.74
IVF-Binary-512-nl387-random (self)                    11_604.74     2_876.27    14_481.01       0.3882    13.418317        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          13_276.55       182.63    13_459.17       0.1452          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          13_276.55       193.41    13_469.95       0.1449          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          13_276.55       211.53    13_488.07       0.1444          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          13_276.55       246.17    13_522.71       0.2885    21.132808        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         13_276.55       273.76    13_550.31       0.3893    13.351620        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         13_276.55       351.07    13_627.61       0.5152     7.947935        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          13_276.55       246.60    13_523.14       0.2886    21.124299        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         13_276.55       288.70    13_565.25       0.3898    13.287882        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         13_276.55       385.93    13_662.48       0.5167     7.854387        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          13_276.55       274.01    13_550.56       0.2879    21.166019        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         13_276.55       313.28    13_589.82       0.3891    13.290361        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         13_276.55       384.67    13_661.21       0.5161     7.842132        10.82
IVF-Binary-512-nl547-random (self)                    13_276.55     2_727.98    16_004.52       0.3891    13.391642        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             28_875.72       182.57    29_058.29       0.1378          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             28_875.72       196.04    29_071.76       0.1375          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             28_875.72       231.46    29_107.17       0.1371          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             28_875.72       232.59    29_108.31       0.2691    22.120983        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            28_875.72       275.99    29_151.70       0.3649    14.114843        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            28_875.72       342.67    29_218.39       0.4846     8.519714        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             28_875.72       251.42    29_127.14       0.2692    22.090805        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            28_875.72       294.57    29_170.29       0.3655    14.032002        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            28_875.72       370.36    29_246.08       0.4867     8.393640        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             28_875.72       298.73    29_174.45       0.2681    22.167927        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            28_875.72       341.41    29_217.13       0.3645    14.059004        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            28_875.72       422.25    29_297.96       0.4863     8.381342        10.69
IVF-Binary-512-nl273-itq (self)                       28_875.72     2_958.33    31_834.05       0.3649    14.124327        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             26_148.24       187.81    26_336.05       0.1376          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             26_148.24       216.03    26_364.27       0.1372          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             26_148.24       241.30    26_389.54       0.2696    22.060949        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            26_148.24       281.35    26_429.59       0.3656    14.048951        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            26_148.24       350.59    26_498.83       0.4860     8.446059        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             26_148.24       272.71    26_420.95       0.2688    22.108863        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            26_148.24       319.57    26_467.81       0.3651    14.029173        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            26_148.24       407.22    26_555.45       0.4867     8.367913        10.74
IVF-Binary-512-nl387-itq (self)                       26_148.24     2_817.74    28_965.98       0.3652    14.135805        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             33_655.11       184.25    33_839.36       0.1380          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             33_655.11       194.23    33_849.34       0.1377          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             33_655.11       210.57    33_865.68       0.1371          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             33_655.11       234.21    33_889.32       0.2705    21.995421        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            33_655.11       273.05    33_928.16       0.3669    14.000707        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            33_655.11       338.96    33_994.06       0.4870     8.442677        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             33_655.11       257.41    33_912.51       0.2701    22.003758        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            33_655.11       285.82    33_940.92       0.3669    13.977794        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            33_655.11       356.99    34_012.10       0.4880     8.361194        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             33_655.11       275.14    33_930.25       0.2692    22.079813        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            33_655.11       321.02    33_976.13       0.3660    13.994225        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            33_655.11       385.15    34_040.26       0.4876     8.341396        10.82
IVF-Binary-512-nl547-itq (self)                       33_655.11     2_713.80    36_368.91       0.3662    14.101446        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         18_078.21       456.05    18_534.25       0.2048          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         18_078.21       515.45    18_593.66       0.2054          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         18_078.21       628.73    18_706.94       0.2054          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         18_078.21       509.51    18_587.72       0.4160    12.114475        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        18_078.21       556.52    18_634.73       0.5398     6.941152        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        18_078.21       632.38    18_710.59       0.6698     3.762648        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         18_078.21       596.57    18_674.78       0.4192    11.921203        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        18_078.21       660.91    18_739.12       0.5460     6.695390        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        18_078.21       677.44    18_755.65       0.6798     3.482393        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         18_078.21       683.26    18_761.47       0.4202    11.849902        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        18_078.21       732.48    18_810.69       0.5482     6.602384        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        18_078.21       821.79    18_900.00       0.6848     3.353218        20.09
IVF-Binary-1024-nl273-random (self)                   18_078.21     6_077.12    24_155.33       0.5447     6.754030        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         19_759.23       480.90    20_240.13       0.2052          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         19_759.23       653.80    20_413.04       0.2054          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         19_759.23       543.16    20_302.39       0.4177    12.030220        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        19_759.23       589.05    20_348.29       0.5427     6.838746        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        19_759.23       631.65    20_390.88       0.6749     3.649398        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         19_759.23       630.88    20_390.11       0.4201    11.858151        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        19_759.23       678.21    20_437.45       0.5480     6.612012        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        19_759.23       756.35    20_515.59       0.6840     3.378677        20.15
IVF-Binary-1024-nl387-random (self)                   19_759.23     5_469.66    25_228.90       0.5424     6.876210        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         20_687.81       427.42    21_115.23       0.2055          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         20_687.81       474.36    21_162.17       0.2059          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         20_687.81       518.05    21_205.86       0.2056          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         20_687.81       465.35    21_153.16       0.4177    12.044578        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        20_687.81       496.51    21_184.33       0.5419     6.884795        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        20_687.81       552.60    21_240.42       0.6725     3.716157        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         20_687.81       523.75    21_211.57       0.4202    11.892305        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        20_687.81       534.37    21_222.19       0.5467     6.686514        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        20_687.81       595.86    21_283.68       0.6804     3.485439        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         20_687.81       554.74    21_242.56       0.4207    11.834813        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        20_687.81       588.76    21_276.58       0.5484     6.603220        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        20_687.81       652.72    21_340.53       0.6844     3.375118        20.23
IVF-Binary-1024-nl547-random (self)                   20_687.81     5_019.46    25_707.27       0.5413     6.922030        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            32_429.69       447.94    32_877.64       0.2059          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            32_429.69       509.27    32_938.96       0.2064          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            32_429.69       611.85    33_041.55       0.2063          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            32_429.69       486.40    32_916.10       0.4172    11.648506        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           32_429.69       526.53    32_956.22       0.5412     6.632954        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           32_429.69       577.84    33_007.53       0.6689     3.595940        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            32_429.69       539.26    32_968.95       0.4203    11.446920        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           32_429.69       573.55    33_003.24       0.5473     6.391086        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           32_429.69       640.83    33_070.53       0.6790     3.316168        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            32_429.69       658.39    33_088.08       0.4213    11.381911        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           32_429.69       693.71    33_123.40       0.5496     6.291720        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           32_429.69       765.13    33_194.82       0.6840     3.175858        20.09
IVF-Binary-1024-nl273-itq (self)                      32_429.69     5_730.78    38_160.47       0.5457     6.467127        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            33_115.22       454.83    33_570.06       0.2064          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            33_115.22       556.88    33_672.11       0.2065          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            33_115.22       494.54    33_609.76       0.4190    11.543580        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           33_115.22       532.00    33_647.23       0.5446     6.517360        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           33_115.22       594.92    33_710.15       0.6740     3.483711        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            33_115.22       594.86    33_710.09       0.4216    11.375903        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           33_115.22       639.40    33_754.62       0.5495     6.305210        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           33_115.22       720.10    33_835.32       0.6837     3.195098        20.15
IVF-Binary-1024-nl387-itq (self)                      33_115.22     5_329.92    38_445.14       0.5431     6.597160        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            34_810.33       425.61    35_235.94       0.2064          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            34_810.33       463.77    35_274.10       0.2066          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            34_810.33       514.75    35_325.07       0.2066          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            34_810.33       464.28    35_274.61       0.4190    11.576384        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           34_810.33       496.83    35_307.16       0.5432     6.578658        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           34_810.33       554.36    35_364.69       0.6715     3.550405        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            34_810.33       499.52    35_309.84       0.4210    11.442230        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           34_810.33       538.57    35_348.90       0.5476     6.396562        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           34_810.33       599.86    35_410.19       0.6793     3.321167        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            34_810.33       554.26    35_364.58       0.4218    11.369714        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           34_810.33       585.74    35_396.06       0.5498     6.302854        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           34_810.33       651.64    35_461.97       0.6837     3.199255        20.23
IVF-Binary-1024-nl547-itq (self)                      34_810.33     4_951.93    39_762.25       0.5421     6.640109        20.23
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (Correlated - higher dimensionality)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.15     6_176.60     6_190.75       1.0000     0.000000        73.24
Exhaustive (self)                                         14.15    63_405.06    63_419.21       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_870.09       518.82     4_388.92       0.0835          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_870.09       561.86     4_431.95       0.1599    21.111789         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_870.09       614.86     4_484.95       0.2242    14.518074         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_870.09       724.93     4_595.03       0.3173     9.567601         4.70
ExhaustiveBinary-256-random (self)                     3_870.09     6_153.95    10_024.05       0.2257    14.498282         4.70
ExhaustiveBinary-256-itq_no_rr (query)                18_049.25       549.93    18_599.19       0.0522          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  18_049.25       645.86    18_695.11       0.1084    27.956034         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 18_049.25       705.36    18_754.62       0.1563    19.790083         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 18_049.25       789.14    18_838.39       0.2301    13.538437         4.70
ExhaustiveBinary-256-itq (self)                       18_049.25     6_209.70    24_258.96       0.1607    19.585886         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_657.30       923.31     8_580.61       0.1247          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_657.30       965.16     8_622.46       0.2451    13.652649         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_657.30     1_030.56     8_687.87       0.3362     8.805138         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_657.30     1_151.04     8_808.34       0.4554     5.389016         9.41
ExhaustiveBinary-512-random (self)                     7_657.30    10_380.28    18_037.58       0.3357     8.840667         9.41
ExhaustiveBinary-512-itq_no_rr (query)                21_808.51       912.67    22_721.18       0.1097          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  21_808.51       955.48    22_763.99       0.2108    16.038374         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 21_808.51     1_020.99    22_829.50       0.2904    10.670775         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 21_808.51     1_119.40    22_927.91       0.3995     6.762765         9.41
ExhaustiveBinary-512-itq (self)                       21_808.51    10_053.60    31_862.11       0.2902    10.669535         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_204.75     1_560.94    16_765.69       0.1760          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_204.75     1_581.36    16_786.11       0.3642     7.961838        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_204.75     1_665.44    16_870.19       0.4844     4.696347        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_204.75     1_762.43    16_967.18       0.6214     2.539505        18.81
ExhaustiveBinary-1024-random (self)                   15_204.75    16_652.07    31_856.83       0.4849     4.684572        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               29_707.15     1_539.70    31_246.85       0.1669          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 29_707.15     1_605.81    31_312.96       0.3450     8.646287        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                29_707.15     1_650.62    31_357.76       0.4610     5.152251        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                29_707.15     1_762.62    31_469.76       0.5965     2.860810        18.81
ExhaustiveBinary-1024-itq (self)                      29_707.15    16_626.51    46_333.65       0.4605     5.175618        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           6_759.00        89.17     6_848.17       0.0903          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           6_759.00        94.74     6_853.73       0.0879          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           6_759.00       109.40     6_868.39       0.0860          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           6_759.00       122.45     6_881.45       0.1734    19.725447         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          6_759.00       159.84     6_918.84       0.2422    13.453975         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          6_759.00       223.36     6_982.36       0.3420     8.768326         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           6_759.00       128.53     6_887.52       0.1691    20.136222         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          6_759.00       164.86     6_923.85       0.2364    13.778082         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          6_759.00       233.17     6_992.17       0.3340     9.023211         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           6_759.00       145.13     6_904.13       0.1644    20.702271         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          6_759.00       182.83     6_941.83       0.2299    14.222320         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          6_759.00       256.07     7_015.07       0.3249     9.347148         5.98
IVF-Binary-256-nl273-random (self)                     6_759.00     1_641.47     8_400.46       0.2375    13.776009         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)           7_831.26        99.20     7_930.46       0.0888          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           7_831.26       111.97     7_943.23       0.0866          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           7_831.26       133.49     7_964.74       0.1715    19.887155         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          7_831.26       166.73     7_997.98       0.2400    13.567203         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          7_831.26       229.33     8_060.59       0.3390     8.855966         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           7_831.26       145.46     7_976.72       0.1663    20.504704         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          7_831.26       185.15     8_016.40       0.2322    14.061194         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          7_831.26       254.15     8_085.41       0.3270     9.257091         6.04
IVF-Binary-256-nl387-random (self)                     7_831.26     1_651.21     9_482.47       0.2411    13.572372         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)           9_418.11        95.99     9_514.10       0.0902          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           9_418.11       100.07     9_518.18       0.0888          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           9_418.11       105.93     9_524.04       0.0873          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           9_418.11       130.86     9_548.97       0.1747    19.559525         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          9_418.11       165.81     9_583.91       0.2447    13.313891         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          9_418.11       237.25     9_655.35       0.3462     8.638841         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           9_418.11       136.31     9_554.42       0.1712    19.932302         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          9_418.11       171.58     9_589.69       0.2396    13.616890         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          9_418.11       236.08     9_654.19       0.3384     8.894881         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           9_418.11       143.52     9_561.63       0.1673    20.376846         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          9_418.11       179.67     9_597.77       0.2337    13.976421         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          9_418.11       247.50     9_665.61       0.3300     9.176761         6.12
IVF-Binary-256-nl547-random (self)                     9_418.11     1_651.07    11_069.18       0.2458    13.305652         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             22_209.29        92.20    22_301.48       0.0615          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             22_209.29        94.85    22_304.13       0.0573          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             22_209.29       108.87    22_318.16       0.0547          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             22_209.29       119.33    22_328.62       0.1267    25.425690         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            22_209.29       153.53    22_362.81       0.1816    17.807267         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            22_209.29       217.34    22_426.63       0.2647    11.990270         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             22_209.29       124.47    22_333.76       0.1194    26.395455         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            22_209.29       159.61    22_368.89       0.1718    18.570497         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            22_209.29       225.67    22_434.96       0.2520    12.544219         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             22_209.29       141.76    22_351.05       0.1141    27.395747         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            22_209.29       177.33    22_386.61       0.1640    19.381750         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            22_209.29       247.89    22_457.18       0.2403    13.210360         5.98
IVF-Binary-256-nl273-itq (self)                       22_209.29     1_605.47    23_814.76       0.1756    18.397103         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             21_675.64        92.65    21_768.28       0.0579          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             21_675.64       103.55    21_779.19       0.0555          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             21_675.64       124.56    21_800.20       0.1215    25.943423         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            21_675.64       158.23    21_833.86       0.1750    18.201397         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            21_675.64       222.05    21_897.69       0.2581    12.231315         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             21_675.64       136.65    21_812.28       0.1154    26.965163         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            21_675.64       172.97    21_848.60       0.1661    19.018778         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            21_675.64       239.97    21_915.61       0.2452    12.865624         6.04
IVF-Binary-256-nl387-itq (self)                       21_675.64     1_562.94    23_238.58       0.1795    18.011183         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             28_809.77       101.28    28_911.05       0.0597          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             28_809.77       100.80    28_910.58       0.0580          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             28_809.77       107.01    28_916.79       0.0562          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             28_809.77       126.58    28_936.35       0.1253    25.285566         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            28_809.77       160.32    28_970.09       0.1805    17.659946         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            28_809.77       282.71    29_092.49       0.2675    11.756794         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             28_809.77       131.14    28_940.91       0.1215    25.927407         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            28_809.77       179.40    28_989.18       0.1749    18.152284         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            28_809.77       245.17    29_054.94       0.2586    12.178429         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             28_809.77       140.08    28_949.85       0.1171    26.676971         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            28_809.77       173.98    28_983.76       0.1685    18.765645         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            28_809.77       244.96    29_054.74       0.2484    12.671178         6.12
IVF-Binary-256-nl547-itq (self)                       28_809.77     1_694.15    30_503.92       0.1855    17.466254         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          10_562.96       175.90    10_738.86       0.1288          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          10_562.96       198.73    10_761.69       0.1274          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          10_562.96       225.40    10_788.36       0.1263          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          10_562.96       229.79    10_792.75       0.2554    13.056066        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         10_562.96       270.18    10_833.14       0.3500     8.367575        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         10_562.96       339.98    10_902.94       0.4731     5.065112        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          10_562.96       246.94    10_809.90       0.2518    13.257603        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         10_562.96       304.15    10_867.12       0.3445     8.532496        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         10_562.96       361.85    10_924.82       0.4662     5.187718        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          10_562.96       284.24    10_847.20       0.2483    13.488515        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         10_562.96       332.97    10_895.93       0.3396     8.703304        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         10_562.96       417.17    10_980.13       0.4594     5.320285        10.69
IVF-Binary-512-nl273-random (self)                    10_562.96     2_884.05    13_447.01       0.3446     8.551848        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          11_666.04       184.58    11_850.62       0.1280          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          11_666.04       214.91    11_880.95       0.1262          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          11_666.04       239.62    11_905.66       0.2546    13.095870        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         11_666.04       275.58    11_941.62       0.3488     8.395705        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         11_666.04       342.76    12_008.80       0.4720     5.083484        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          11_666.04       268.31    11_934.35       0.2498    13.398829        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         11_666.04       314.24    11_980.28       0.3414     8.645265        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         11_666.04       390.10    12_056.14       0.4617     5.276591        10.74
IVF-Binary-512-nl387-random (self)                    11_666.04     2_773.07    14_439.11       0.3485     8.424727        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          13_232.56       180.35    13_412.91       0.1292          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          13_232.56       190.90    13_423.46       0.1281          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          13_232.56       207.13    13_439.69       0.1268          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          13_232.56       230.63    13_463.18       0.2571    12.933708        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         13_232.56       268.42    13_500.97       0.3534     8.252243        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         13_232.56       330.93    13_563.49       0.4784     4.975109        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          13_232.56       241.33    13_473.89       0.2538    13.135266        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         13_232.56       281.50    13_514.05       0.3483     8.421781        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         13_232.56       347.18    13_579.74       0.4715     5.105947        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          13_232.56       259.26    13_491.82       0.2505    13.346677        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         13_232.56       301.31    13_533.86       0.3429     8.595584        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         13_232.56       372.27    13_604.83       0.4639     5.244616        10.82
IVF-Binary-512-nl547-random (self)                    13_232.56     2_678.86    15_911.42       0.3531     8.282934        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             28_507.24       177.50    28_684.74       0.1142          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             28_507.24       192.03    28_699.27       0.1125          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             28_507.24       226.51    28_733.75       0.1112          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             28_507.24       251.53    28_758.77       0.2214    15.286978        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            28_507.24       281.09    28_788.33       0.3055    10.075350        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            28_507.24       345.08    28_852.33       0.4203     6.294868        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             28_507.24       243.82    28_751.06       0.2175    15.558026        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            28_507.24       292.51    28_799.75       0.3002    10.284617        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            28_507.24       358.94    28_866.18       0.4127     6.466201        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             28_507.24       282.50    28_789.74       0.2139    15.849197        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            28_507.24       340.23    28_847.47       0.2945    10.522865        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            28_507.24       413.37    28_920.61       0.4049     6.651241        10.69
IVF-Binary-512-nl273-itq (self)                       28_507.24     2_868.98    31_376.22       0.3000    10.282058        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             26_922.96       182.92    27_105.88       0.1134          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             26_922.96       220.06    27_143.02       0.1117          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             26_922.96       236.50    27_159.46       0.2198    15.377204        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            26_922.96       272.94    27_195.90       0.3039    10.116887        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            26_922.96       338.69    27_261.65       0.4180     6.330257        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             26_922.96       267.05    27_190.01       0.2151    15.728427        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            26_922.96       311.36    27_234.32       0.2964    10.421052        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            26_922.96       384.53    27_307.49       0.4077     6.573184        10.74
IVF-Binary-512-nl387-itq (self)                       26_922.96     2_727.45    29_650.41       0.3039    10.116424        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             30_784.49       194.68    30_979.16       0.1145          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             30_784.49       192.09    30_976.57       0.1133          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             30_784.49       215.21    30_999.70       0.1121          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             30_784.49       231.18    31_015.67       0.2226    15.146158        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            30_784.49       286.27    31_070.76       0.3085     9.939220        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            30_784.49       347.31    31_131.79       0.4253     6.178097        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             30_784.49       257.00    31_041.48       0.2193    15.392335        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            30_784.49       280.35    31_064.84       0.3034    10.142749        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            30_784.49       344.67    31_129.15       0.4178     6.352959        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             30_784.49       282.97    31_067.46       0.2158    15.671654        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            30_784.49       318.79    31_103.28       0.2982    10.359584        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            30_784.49       391.59    31_176.08       0.4104     6.531168        10.82
IVF-Binary-512-nl547-itq (self)                       30_784.49     2_657.25    33_441.74       0.3085     9.934329        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         18_058.43       444.39    18_502.81       0.1798          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         18_058.43       508.92    18_567.35       0.1786          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         18_058.43       614.23    18_672.66       0.1774          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         18_058.43       484.98    18_543.40       0.3726     7.705916        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        18_058.43       543.54    18_601.96       0.4955     4.506521        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        18_058.43       587.58    18_646.01       0.6350     2.398236        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         18_058.43       542.94    18_601.37       0.3694     7.804042        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        18_058.43       573.02    18_631.44       0.4909     4.584533        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        18_058.43       635.61    18_694.03       0.6292     2.456994        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         18_058.43       683.41    18_741.84       0.3668     7.887871        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        18_058.43       694.60    18_753.03       0.4870     4.652723        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        18_058.43       785.14    18_843.57       0.6243     2.507896        20.09
IVF-Binary-1024-nl273-random (self)                   18_058.43     5_871.34    23_929.77       0.4913     4.572337        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         19_194.72       460.65    19_655.37       0.1792          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         19_194.72       555.91    19_750.64       0.1776          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         19_194.72       532.99    19_727.71       0.3715     7.728045        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        19_194.72       526.62    19_721.35       0.4949     4.514208        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        19_194.72       584.12    19_778.85       0.6340     2.405332        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         19_194.72       596.84    19_791.56       0.3673     7.867217        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        19_194.72       627.88    19_822.61       0.4883     4.630894        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        19_194.72       696.65    19_891.38       0.6260     2.490097        20.15
IVF-Binary-1024-nl387-random (self)                   19_194.72     5_360.71    24_555.44       0.4950     4.507483        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         20_914.33       460.10    21_374.43       0.1803          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         20_914.33       486.34    21_400.67       0.1791          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         20_914.33       612.08    21_526.41       0.1778          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         20_914.33       499.44    21_413.77       0.3746     7.642280        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        20_914.33       553.53    21_467.86       0.4990     4.437106        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        20_914.33       579.84    21_494.17       0.6395     2.348751        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         20_914.33       516.25    21_430.58       0.3710     7.749561        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        20_914.33       573.50    21_487.83       0.4944     4.520635        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        20_914.33       624.95    21_539.28       0.6335     2.410795        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         20_914.33       568.88    21_483.21       0.3683     7.840859        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        20_914.33       617.76    21_532.09       0.4899     4.600186        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        20_914.33       706.62    21_620.95       0.6274     2.474668        20.23
IVF-Binary-1024-nl547-random (self)                   20_914.33     5_283.48    26_197.81       0.4992     4.434766        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            40_397.40       567.84    40_965.25       0.1705          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            40_397.40       547.54    40_944.94       0.1692          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            40_397.40       689.65    41_087.05       0.1682          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            40_397.40       519.43    40_916.83       0.3537     8.360736        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           40_397.40       581.32    40_978.72       0.4717     4.958550        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           40_397.40       761.17    41_158.57       0.6104     2.715850        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            40_397.40       683.22    41_080.62       0.3506     8.464984        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           40_397.40       687.12    41_084.53       0.4675     5.033830        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           40_397.40       712.15    41_109.55       0.6047     2.775386        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            40_397.40       768.64    41_166.04       0.3476     8.564604        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           40_397.40       696.62    41_094.02       0.4636     5.109604        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           40_397.40       776.02    41_173.42       0.5998     2.826663        20.09
IVF-Binary-1024-nl273-itq (self)                      40_397.40     6_398.73    46_796.14       0.4673     5.052001        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            35_382.22       541.09    35_923.31       0.1702          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            35_382.22       552.43    35_934.65       0.1686          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            35_382.22       544.36    35_926.58       0.3527     8.388790        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           35_382.22       630.33    36_012.55       0.4707     4.971031        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           35_382.22       734.59    36_116.81       0.6094     2.721655        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            35_382.22       654.34    36_036.56       0.3484     8.533778        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           35_382.22       695.18    36_077.40       0.4647     5.084729        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           35_382.22       755.51    36_137.73       0.6008     2.812394        20.15
IVF-Binary-1024-nl387-itq (self)                      35_382.22     5_579.95    40_962.17       0.4709     4.986071        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            38_370.49       430.72    38_801.21       0.1707          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            38_370.49       467.87    38_838.36       0.1697          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            38_370.49       515.32    38_885.82       0.1687          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            38_370.49       462.05    38_832.54       0.3557     8.297828        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           38_370.49       494.75    38_865.25       0.4748     4.900112        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           38_370.49       553.18    38_923.67       0.6148     2.662016        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            38_370.49       499.77    38_870.26       0.3526     8.401633        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           38_370.49       539.18    38_909.68       0.4703     4.984595        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           38_370.49       628.67    38_999.16       0.6087     2.729952        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            38_370.49       641.67    39_012.16       0.3496     8.501827        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           38_370.49       661.79    39_032.28       0.4662     5.059001        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           38_370.49       724.41    39_094.90       0.6031     2.789781        20.23
IVF-Binary-1024-nl547-itq (self)                      38_370.49     5_017.11    43_387.60       0.4749     4.913051        20.23
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (Correlated - higher dimensionality)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.34     6_164.32     6_178.66       1.0000     0.000000        73.24
Exhaustive (self)                                         14.34    64_351.70    64_366.05       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_896.60       528.19     4_424.79       0.0770          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_896.60       589.37     4_485.98       0.1853    40.590911         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_896.60       664.76     4_561.36       0.2680    26.716815         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_896.60       765.21     4_661.81       0.3854    16.589779         4.70
ExhaustiveBinary-256-random (self)                     3_896.60     6_413.52    10_310.12       0.2721    26.279507         4.70
ExhaustiveBinary-256-itq_no_rr (query)                18_493.79       591.11    19_084.89       0.0334          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  18_493.79       634.32    19_128.10       0.1047    59.774198         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 18_493.79       657.81    19_151.59       0.1682    40.784107         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 18_493.79       724.79    19_218.58       0.2648    26.557562         4.70
ExhaustiveBinary-256-itq (self)                       18_493.79     6_620.06    25_113.84       0.1711    40.460557         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_783.07       921.22     8_704.28       0.1527          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_783.07     1_183.89     8_966.96       0.3242    22.196757         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_783.07     1_043.02     8_826.08       0.4419    13.545010         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_783.07     1_252.83     9_035.89       0.5832     7.719977         9.41
ExhaustiveBinary-512-random (self)                     7_783.07    10_551.71    18_334.78       0.4431    13.453984         9.41
ExhaustiveBinary-512-itq_no_rr (query)                22_438.56       905.31    23_343.87       0.1302          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  22_438.56       949.46    23_388.01       0.2779    26.538562         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 22_438.56     1_017.52    23_456.08       0.3835    16.663114         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 22_438.56     1_124.92    23_563.48       0.5194     9.834430         9.41
ExhaustiveBinary-512-itq (self)                       22_438.56    10_505.61    32_944.16       0.3863    16.548400         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_207.87     1_525.68    16_733.55       0.2353          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_207.87     1_579.63    16_787.50       0.4918    11.183271        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_207.87     1_627.91    16_835.78       0.6358     5.988378        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_207.87     1_746.83    16_954.70       0.7779     2.871209        18.81
ExhaustiveBinary-1024-random (self)                   15_207.87    16_600.74    31_808.61       0.6381     5.940314        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               29_369.77     1_678.66    31_048.43       0.2191          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 29_369.77     1_754.71    31_124.48       0.4611    12.755652        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                29_369.77     1_850.45    31_220.22       0.6023     7.013961        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                29_369.77     2_047.43    31_417.21       0.7468     3.481342        18.81
ExhaustiveBinary-1024-itq (self)                      29_369.77    18_733.00    48_102.78       0.6045     6.950550        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           7_474.67        96.46     7_571.13       0.0894          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)           7_474.67       103.21     7_577.88       0.0819          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)           7_474.67       128.74     7_603.41       0.0801          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)           7_474.67       148.38     7_623.05       0.2082    37.003584         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)          7_474.67       184.61     7_659.28       0.2984    23.995106         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)          7_474.67       257.96     7_732.63       0.4184    14.855149         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)           7_474.67       130.06     7_604.73       0.1960    39.530049         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)          7_474.67       169.10     7_643.77       0.2836    25.718898         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)          7_474.67       235.34     7_710.01       0.4031    15.887879         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)           7_474.67       145.05     7_619.72       0.1927    40.256023         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)          7_474.67       188.19     7_662.86       0.2789    26.258727         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)          7_474.67       257.05     7_731.72       0.3972    16.273250         5.98
IVF-Binary-256-nl273-random (self)                     7_474.67     1_711.90     9_186.57       0.2865    25.390828         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)           8_202.75        91.47     8_294.22       0.0832          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)           8_202.75       101.96     8_304.71       0.0801          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)           8_202.75       123.07     8_325.82       0.1995    38.528842         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)          8_202.75       158.14     8_360.89       0.2884    25.014672         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)          8_202.75       227.48     8_430.22       0.4104    15.315480         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)           8_202.75       135.36     8_338.11       0.1929    40.071462         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)          8_202.75       168.48     8_371.23       0.2798    26.100169         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)          8_202.75       237.48     8_440.23       0.4005    15.973244         6.04
IVF-Binary-256-nl387-random (self)                     8_202.75     1_637.50     9_840.25       0.2930    24.576691         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)           9_693.66        98.74     9_792.40       0.0847          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)           9_693.66       101.27     9_794.94       0.0829          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)           9_693.66       108.41     9_802.08       0.0815          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)           9_693.66       130.13     9_823.79       0.2026    37.589199         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)          9_693.66       162.54     9_856.20       0.2933    24.331515         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)          9_693.66       235.24     9_928.90       0.4160    14.924581         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)           9_693.66       137.01     9_830.67       0.1985    38.518369         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)          9_693.66       170.19     9_863.85       0.2874    25.037356         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)          9_693.66       258.99     9_952.65       0.4081    15.411088         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)           9_693.66       167.14     9_860.80       0.1948    39.402901         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)          9_693.66       192.97     9_886.63       0.2821    25.705501         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)          9_693.66       271.21     9_964.88       0.4018    15.833821         6.12
IVF-Binary-256-nl547-random (self)                     9_693.66     1_673.77    11_367.43       0.2978    23.888874         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             27_036.22        88.35    27_124.57       0.0462          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             27_036.22        90.97    27_127.19       0.0370          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             27_036.22       110.77    27_146.99       0.0355          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             27_036.22       112.75    27_148.97       0.1367    53.764831         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            27_036.22       145.96    27_182.18       0.2112    36.381584         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            27_036.22       201.41    27_237.63       0.3175    23.358034         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             27_036.22       117.08    27_153.30       0.1173    59.670220         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            27_036.22       151.46    27_187.68       0.1872    40.770059         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            27_036.22       218.95    27_255.18       0.2906    26.159204         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             27_036.22       131.86    27_168.09       0.1132    61.886937         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            27_036.22       162.85    27_199.07       0.1805    42.819528         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            27_036.22       228.63    27_264.86       0.2805    27.530959         5.98
IVF-Binary-256-nl273-itq (self)                       27_036.22     1_493.78    28_530.00       0.1891    40.811783         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             22_256.67        89.36    22_346.03       0.0385          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             22_256.67        99.21    22_355.87       0.0357          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             22_256.67       118.02    22_374.68       0.1233    57.094275         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            22_256.67       162.17    22_418.84       0.1952    38.608729         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            22_256.67       232.27    22_488.94       0.3019    24.521453         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             22_256.67       127.27    22_383.94       0.1149    60.313468         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            22_256.67       164.97    22_421.64       0.1836    41.058709         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            22_256.67       223.07    22_479.74       0.2869    26.115333         6.04
IVF-Binary-256-nl387-itq (self)                       22_256.67     1_499.31    23_755.98       0.1984    38.246832         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             23_663.28        91.05    23_754.32       0.0404          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             23_663.28        95.06    23_758.34       0.0389          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             23_663.28       100.39    23_763.66       0.0372          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             23_663.28       119.59    23_782.86       0.1270    55.047860         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            23_663.28       150.45    23_813.73       0.2012    37.295712         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            23_663.28       203.84    23_867.12       0.3089    23.748919         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             23_663.28       121.69    23_784.97       0.1227    56.668382         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            23_663.28       154.53    23_817.81       0.1944    38.564183         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            23_663.28       223.13    23_886.41       0.3002    24.578001         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             23_663.28       128.36    23_791.64       0.1177    58.839482         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            23_663.28       163.20    23_826.48       0.1868    40.208023         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            23_663.28       229.30    23_892.57       0.2904    25.678650         6.12
IVF-Binary-256-nl547-itq (self)                       23_663.28     1_511.28    25_174.56       0.2043    37.026983         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          10_578.18       176.13    10_754.32       0.1596          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          10_578.18       200.58    10_778.77       0.1563          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          10_578.18       221.40    10_799.59       0.1554          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          10_578.18       226.86    10_805.04       0.3370    20.987751        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         10_578.18       265.49    10_843.67       0.4570    12.738091        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         10_578.18       327.91    10_906.10       0.5992     7.207613        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          10_578.18       245.17    10_823.35       0.3308    21.630921        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         10_578.18       286.74    10_864.93       0.4506    13.121710        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         10_578.18       361.50    10_939.69       0.5923     7.447657        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          10_578.18       275.16    10_853.34       0.3289    21.801439        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         10_578.18       322.06    10_900.25       0.4477    13.266981        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         10_578.18       405.45    10_983.63       0.5898     7.527176        10.69
IVF-Binary-512-nl273-random (self)                    10_578.18     2_890.17    13_468.35       0.4518    13.046860        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          11_639.29       184.37    11_823.65       0.1571          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          11_639.29       210.43    11_849.72       0.1557          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          11_639.29       234.94    11_874.22       0.3331    21.366472        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         11_639.29       271.77    11_911.06       0.4543    12.892701        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         11_639.29       338.43    11_977.71       0.5962     7.323848        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          11_639.29       290.26    11_929.55       0.3294    21.746609        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         11_639.29       304.90    11_944.19       0.4493    13.172260        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         11_639.29       378.04    12_017.32       0.5909     7.498725        10.74
IVF-Binary-512-nl387-random (self)                    11_639.29     2_768.68    14_407.97       0.4552    12.836096        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          13_369.42       183.58    13_552.99       0.1579          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          13_369.42       198.67    13_568.09       0.1569          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          13_369.42       207.35    13_576.76       0.1560          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          13_369.42       231.14    13_600.56       0.3354    21.117508        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         13_369.42       306.02    13_675.44       0.4570    12.738653        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         13_369.42       334.19    13_703.60       0.5999     7.182206        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          13_369.42       241.91    13_611.33       0.3326    21.426735        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         13_369.42       288.66    13_658.07       0.4531    12.960609        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         13_369.42       341.95    13_711.37       0.5949     7.349314        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          13_369.42       257.00    13_626.41       0.3301    21.676819        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         13_369.42       306.20    13_675.62       0.4501    13.125100        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         13_369.42       365.42    13_734.83       0.5916     7.462558        10.82
IVF-Binary-512-nl547-random (self)                    13_369.42     2_642.86    16_012.28       0.4583    12.660445        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             26_056.03       180.30    26_236.33       0.1373          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             26_056.03       199.24    26_255.26       0.1336          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             26_056.03       227.01    26_283.04       0.1327          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             26_056.03       254.26    26_310.29       0.2921    25.040561        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            26_056.03       265.23    26_321.26       0.4014    15.571823        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            26_056.03       336.63    26_392.66       0.5382     9.170711        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             26_056.03       240.72    26_296.75       0.2858    25.784569        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            26_056.03       321.98    26_378.01       0.3942    16.060633        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            26_056.03       361.80    26_417.82       0.5303     9.478394        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             26_056.03       274.56    26_330.58       0.2834    26.075631        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            26_056.03       329.77    26_385.80       0.3910    16.268067        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            26_056.03       395.65    26_451.68       0.5267     9.612246        10.69
IVF-Binary-512-nl273-itq (self)                       26_056.03     2_839.80    28_895.83       0.3965    15.986403        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             34_822.04       187.80    35_009.85       0.1354          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             34_822.04       214.31    35_036.36       0.1337          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             34_822.04       235.65    35_057.70       0.2879    25.438761        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            34_822.04       277.68    35_099.73       0.3968    15.873447        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            34_822.04       336.40    35_158.44       0.5338     9.322012        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             34_822.04       263.02    35_085.06       0.2837    25.980320        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            34_822.04       303.53    35_125.57       0.3911    16.260259        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            34_822.04       373.71    35_195.76       0.5272     9.577323        10.74
IVF-Binary-512-nl387-itq (self)                       34_822.04     2_716.36    37_538.41       0.4000    15.738281        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             27_961.44       180.66    28_142.11       0.1359          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             27_961.44       192.45    28_153.89       0.1346          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             27_961.44       205.96    28_167.40       0.1335          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             27_961.44       228.46    28_189.91       0.2909    25.061655        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            27_961.44       268.89    28_230.33       0.4003    15.624347        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            27_961.44       325.30    28_286.74       0.5386     9.142311        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             27_961.44       239.80    28_201.24       0.2880    25.409514        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            27_961.44       276.71    28_238.16       0.3963    15.878398        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            27_961.44       340.56    28_302.00       0.5335     9.325783        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             27_961.44       258.84    28_220.29       0.2856    25.723061        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            27_961.44       305.56    28_267.00       0.3927    16.122518        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            27_961.44       363.05    28_324.49       0.5294     9.482738        10.82
IVF-Binary-512-nl547-itq (self)                       27_961.44     2_716.66    30_678.11       0.4033    15.508734        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         18_304.93       444.82    18_749.75       0.2399          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         18_304.93       483.64    18_788.57       0.2382          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         18_304.93       612.39    18_917.32       0.2377          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         18_304.93       474.40    18_779.33       0.4992    10.834117        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        18_304.93       509.85    18_814.78       0.6434     5.779240        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        18_304.93       565.67    18_870.60       0.7845     2.747805        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         18_304.93       542.14    18_847.07       0.4962    10.981356        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        18_304.93       561.84    18_866.77       0.6406     5.859089        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        18_304.93       621.97    18_926.90       0.7820     2.797675        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         18_304.93       619.96    18_924.89       0.4952    11.037131        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        18_304.93       661.24    18_966.17       0.6394     5.893505        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        18_304.93       727.22    19_032.15       0.7808     2.820922        20.09
IVF-Binary-1024-nl273-random (self)                   18_304.93     5_590.87    23_895.80       0.6428     5.813738        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         19_124.66       445.09    19_569.75       0.2387          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         19_124.66       525.57    19_650.23       0.2378          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         19_124.66       481.31    19_605.96       0.4978    10.900723        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        19_124.66       522.00    19_646.66       0.6417     5.823667        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        19_124.66       580.87    19_705.53       0.7830     2.778159        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         19_124.66       567.42    19_692.08       0.4959    11.002308        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        19_124.66       602.63    19_727.29       0.6394     5.889963        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        19_124.66       676.03    19_800.69       0.7809     2.816231        20.15
IVF-Binary-1024-nl387-random (self)                   19_124.66     5_135.01    24_259.66       0.6441     5.772927        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         20_753.34       430.42    21_183.76       0.2394          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         20_753.34       458.88    21_212.22       0.2386          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         20_753.34       504.69    21_258.03       0.2379          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         20_753.34       463.67    21_217.01       0.4993    10.821170        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        20_753.34       491.38    21_244.72       0.6438     5.766783        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        20_753.34       546.76    21_300.10       0.7852     2.739547        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         20_753.34       488.96    21_242.30       0.4976    10.906894        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        20_753.34       525.67    21_279.01       0.6415     5.829356        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        20_753.34       581.03    21_334.37       0.7828     2.783124        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         20_753.34       547.79    21_301.13       0.4962    10.976622        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        20_753.34       577.05    21_330.40       0.6400     5.874798        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        20_753.34       635.21    21_388.56       0.7814     2.811286        20.23
IVF-Binary-1024-nl547-random (self)                   20_753.34     4_900.86    25_654.20       0.6462     5.710279        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            37_836.14       440.99    38_277.13       0.2233          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            37_836.14       482.76    38_318.91       0.2215          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            37_836.14       579.85    38_415.99       0.2211          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            37_836.14       484.55    38_320.69       0.4696    12.292956        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           37_836.14       507.70    38_343.84       0.6111     6.715975        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           37_836.14       569.40    38_405.55       0.7546     3.314411        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            37_836.14       522.56    38_358.70       0.4663    12.481269        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           37_836.14       558.27    38_394.41       0.6075     6.838615        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           37_836.14       620.31    38_456.45       0.7513     3.382404        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            37_836.14       631.23    38_467.37       0.4651    12.562426        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           37_836.14       659.10    38_495.24       0.6060     6.888351        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           37_836.14       728.57    38_564.71       0.7499     3.413899        20.09
IVF-Binary-1024-nl273-itq (self)                      37_836.14     5_560.78    43_396.93       0.6101     6.773542        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            33_872.73       444.26    34_316.99       0.2222          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            33_872.73       528.74    34_401.47       0.2210          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            33_872.73       484.46    34_357.19       0.4677    12.397357        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           33_872.73       515.82    34_388.54       0.6087     6.790945        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           33_872.73       582.80    34_455.53       0.7530     3.347757        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            33_872.73       567.27    34_440.00       0.4649    12.568854        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           33_872.73       608.05    34_480.78       0.6058     6.889699        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           33_872.73       667.65    34_540.37       0.7503     3.409586        20.15
IVF-Binary-1024-nl387-itq (self)                      33_872.73     5_192.00    39_064.73       0.6116     6.726584        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            37_554.65       420.77    37_975.42       0.2230          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            37_554.65       451.74    38_006.40       0.2223          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            37_554.65       497.82    38_052.48       0.2217          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            37_554.65       459.87    38_014.53       0.4689    12.314975        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           37_554.65       488.32    38_042.97       0.6108     6.717628        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           37_554.65       543.63    38_098.28       0.7554     3.300256        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            37_554.65       487.91    38_042.57       0.4670    12.424460        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           37_554.65       520.83    38_075.49       0.6084     6.799578        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           37_554.65       587.22    38_141.87       0.7527     3.359085        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            37_554.65       538.25    38_092.91       0.4659    12.496578        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           37_554.65       574.97    38_129.62       0.6067     6.854108        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           37_554.65       631.85    38_186.51       0.7510     3.394109        20.23
IVF-Binary-1024-nl547-itq (self)                      37_554.65     4_953.25    42_507.90       0.6138     6.650776        20.23
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
Exhaustive (query)                                         3.15     1_731.83     1_734.98       1.0000     0.000000        18.31
Exhaustive (self)                                          3.15    17_066.93    17_070.08       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_723.09       498.88     2_221.97       0.3141    41.590414         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_723.09       581.89     2_304.98       0.6781     1.424540         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_723.09       654.06     2_377.15       0.8317     0.540476         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_723.09       788.72     2_511.81       0.9366     0.159663         3.46
ExhaustiveRaBitQ (self)                                1_723.09     6_577.00     8_300.09       0.8323     0.527516         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        984.49       117.39     1_101.87       0.3202    41.573392         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        984.49       140.51     1_125.00       0.3192    41.580432         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        984.49       193.91     1_178.39       0.3180    41.584057         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        984.49       170.53     1_155.02       0.6871     1.446173         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       984.49       217.49     1_201.98       0.8371     0.557811         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       984.49       301.43     1_285.91       0.9353     0.179328         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        984.49       195.03     1_179.52       0.6868     1.449165         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       984.49       247.12     1_231.61       0.8393     0.547250         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       984.49       339.57     1_324.06       0.9397     0.160266         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        984.49       252.74     1_237.23       0.6854     1.459348         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       984.49       310.78     1_295.27       0.8389     0.549605         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       984.49       403.33     1_387.81       0.9407     0.158794         3.47
IVF-RaBitQ-nl273 (self)                                  984.49     4_012.58     4_997.07       0.9410     0.156272         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_360.44       125.89     1_486.32       0.3212    41.562779         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_360.44       168.48     1_528.92       0.3189    41.572400         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_360.44       189.82     1_550.26       0.6915     1.395785         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_360.44       225.17     1_585.61       0.8427     0.519210         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_360.44       307.68     1_668.12       0.9398     0.165810         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_360.44       222.17     1_582.61       0.6887     1.419301         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_360.44       275.36     1_635.80       0.8422     0.517173         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_360.44       362.90     1_723.34       0.9428     0.146533         3.49
IVF-RaBitQ-nl387 (self)                                1_360.44     3_654.02     5_014.46       0.9429     0.146867         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_861.81       115.24     1_977.05       0.3271    41.545557         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_861.81       131.69     1_993.49       0.3252    41.553699         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_861.81       154.62     2_016.43       0.3237    41.559148         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_861.81       165.76     2_027.57       0.6983     1.345744         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_861.81       209.67     2_071.48       0.8470     0.505474         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_861.81       291.97     2_153.77       0.9405     0.167264         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_861.81       186.56     2_048.37       0.6962     1.356471         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_861.81       230.54     2_092.35       0.8474     0.498117         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_861.81       313.40     2_175.21       0.9440     0.148206         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_861.81       210.03     2_071.84       0.6943     1.372707         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_861.81       257.34     2_119.14       0.8465     0.502442         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_861.81       348.22     2_210.02       0.9451     0.142254         3.51
IVF-RaBitQ-nl547 (self)                                1_861.81     3_457.95     5_319.76       0.9453     0.139847         3.51
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
Exhaustive (query)                                         3.74     1_564.85     1_568.59       1.0000     0.000000        18.88
Exhaustive (self)                                          3.74    16_416.31    16_420.05       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_654.56       444.40     2_098.97       0.3182     0.168383         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_654.56       525.17     2_179.73       0.6833     0.001047         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_654.56       588.14     2_242.70       0.8370     0.000387         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_654.56       713.22     2_367.78       0.9399     0.000109         3.46
ExhaustiveRaBitQ (self)                                1_654.56     5_930.34     7_584.91       0.8377     0.000386         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        933.51       117.69     1_051.20       0.3157     0.167945         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        933.51       140.85     1_074.36       0.3143     0.167649         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        933.51       200.01     1_133.53       0.3131     0.167498         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        933.51       169.39     1_102.91       0.6843     0.001126         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       933.51       217.26     1_150.77       0.8358     0.000439         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       933.51       301.61     1_235.13       0.9346     0.000143         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        933.51       195.98     1_129.50       0.6837     0.001134         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       933.51       247.56     1_181.08       0.8375     0.000434         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       933.51       337.23     1_270.74       0.9395     0.000129         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        933.51       255.15     1_188.67       0.6821     0.001143         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       933.51       310.98     1_244.49       0.8367     0.000436         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       933.51       405.68     1_339.20       0.9402     0.000127         3.47
IVF-RaBitQ-nl273 (self)                                  933.51     4_078.41     5_011.93       0.9408     0.000126         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_284.61       138.30     1_422.91       0.3194     0.168487         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_284.61       172.07     1_456.68       0.3171     0.168118         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_284.61       176.57     1_461.18       0.6886     0.001098         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_284.61       224.28     1_508.89       0.8409     0.000418         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_284.61       307.47     1_592.08       0.9395     0.000129         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_284.61       229.58     1_514.19       0.6860     0.001116         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_284.61       302.61     1_587.22       0.8402     0.000421         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_284.61       365.69     1_650.30       0.9420     0.000121         3.49
IVF-RaBitQ-nl387 (self)                                1_284.61     3_647.40     4_932.02       0.9429     0.000118         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_742.84       113.87     1_856.71       0.3232     0.169083         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_742.84       130.89     1_873.73       0.3210     0.168769         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_742.84       154.14     1_896.98       0.3192     0.168561         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_742.84       164.50     1_907.35       0.6951     0.001058         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_742.84       216.33     1_959.17       0.8454     0.000401         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_742.84       300.81     2_043.65       0.9409     0.000129         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_742.84       190.73     1_933.58       0.6928     0.001070         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_742.84       267.31     2_010.16       0.8455     0.000400         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_742.84       319.69     2_062.53       0.9440     0.000119         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_742.84       210.14     1_952.98       0.6907     0.001084         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_742.84       256.78     1_999.63       0.8443     0.000403         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_742.84       344.72     2_087.56       0.9446     0.000117         3.51
IVF-RaBitQ-nl547 (self)                                1_742.84     3_464.86     5_207.70       0.9451     0.000112         3.51
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
Exhaustive (query)                                         3.19     1_605.50     1_608.69       1.0000     0.000000        18.31
Exhaustive (self)                                          3.19    16_515.02    16_518.21       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_677.44       391.67     2_069.12       0.4426     1.402652         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_677.44       468.64     2_146.09       0.8785     0.030781         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_677.44       533.63     2_211.07       0.9682     0.005679         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_677.44       642.43     2_319.87       0.9954     0.000665         3.46
ExhaustiveRaBitQ (self)                                1_677.44     5_359.04     7_036.49       0.9696     0.005520         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        971.74       113.76     1_085.50       0.4624     1.393408         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        971.74       140.20     1_111.94       0.4620     1.393563         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        971.74       199.08     1_170.83       0.4619     1.393619         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        971.74       162.38     1_134.13       0.8931     0.028100         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       971.74       214.88     1_186.62       0.9745     0.004925         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       971.74       280.93     1_252.68       0.9968     0.000521         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        971.74       200.01     1_171.76       0.8928     0.028226         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       971.74       243.32     1_215.07       0.9744     0.004949         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       971.74       327.40     1_299.15       0.9969     0.000502         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        971.74       258.92     1_230.66       0.8926     0.028290         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       971.74       316.25     1_288.00       0.9743     0.004969         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       971.74       403.45     1_375.20       0.9969     0.000504         3.47
IVF-RaBitQ-nl273 (self)                                  971.74     4_021.93     4_993.68       0.9973     0.000448         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_328.76       121.63     1_450.39       0.4729     1.382072         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_328.76       166.96     1_495.72       0.4728     1.382135         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_328.76       177.21     1_505.97       0.9030     0.024211         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_328.76       219.72     1_548.48       0.9778     0.004106         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_328.76       302.85     1_631.62       0.9976     0.000386         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_328.76       223.01     1_551.77       0.9028     0.024253         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_328.76       271.79     1_600.55       0.9778     0.004117         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_328.76       354.58     1_683.34       0.9976     0.000376         3.49
IVF-RaBitQ-nl387 (self)                                1_328.76     3_540.53     4_869.29       0.9978     0.000345         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_838.34       115.14     1_953.49       0.4837     1.373041         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_838.34       134.00     1_972.34       0.4835     1.373146         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_838.34       153.87     1_992.22       0.4834     1.373185         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_838.34       165.83     2_004.18       0.9115     0.021036         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_838.34       206.67     2_045.01       0.9815     0.003284         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_838.34       279.61     2_117.95       0.9980     0.000305         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_838.34       192.25     2_030.59       0.9112     0.021118         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_838.34       227.42     2_065.76       0.9813     0.003309         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_838.34       309.88     2_148.23       0.9980     0.000300         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_838.34       213.54     2_051.89       0.9111     0.021140         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_838.34       257.56     2_095.91       0.9813     0.003314         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_838.34       335.90     2_174.25       0.9980     0.000294         3.51
IVF-RaBitQ-nl547 (self)                                1_838.34     3_374.00     5_212.34       0.9982     0.000269         3.51
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
Exhaustive (query)                                         3.06     1_664.46     1_667.52       1.0000     0.000000        18.31
Exhaustive (self)                                          3.06    16_863.45    16_866.51       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           1_673.53       412.25     2_085.78       0.4782     7.158251         3.46
ExhaustiveRaBitQ-rf5 (query)                           1_673.53       489.28     2_162.82       0.9142     0.089866         3.46
ExhaustiveRaBitQ-rf10 (query)                          1_673.53       560.81     2_234.34       0.9836     0.012771         3.46
ExhaustiveRaBitQ-rf20 (query)                          1_673.53       647.12     2_320.65       0.9985     0.001016         3.46
ExhaustiveRaBitQ (self)                                1_673.53     5_467.32     7_140.85       0.9834     0.013139         3.46
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                        978.15       113.35     1_091.50       0.4887     7.133549         3.47
IVF-RaBitQ-nl273-np16-rf0 (query)                        978.15       135.37     1_113.53       0.4887     7.133579         3.47
IVF-RaBitQ-nl273-np23-rf0 (query)                        978.15       185.17     1_163.32       0.4887     7.133583         3.47
IVF-RaBitQ-nl273-np13-rf5 (query)                        978.15       161.78     1_139.93       0.9216     0.078073         3.47
IVF-RaBitQ-nl273-np13-rf10 (query)                       978.15       206.22     1_184.38       0.9858     0.010544         3.47
IVF-RaBitQ-nl273-np13-rf20 (query)                       978.15       282.89     1_261.04       0.9988     0.000755         3.47
IVF-RaBitQ-nl273-np16-rf5 (query)                        978.15       190.79     1_168.94       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np16-rf10 (query)                       978.15       243.42     1_221.58       0.9857     0.010553         3.47
IVF-RaBitQ-nl273-np16-rf20 (query)                       978.15       320.25     1_298.41       0.9988     0.000741         3.47
IVF-RaBitQ-nl273-np23-rf5 (query)                        978.15       245.81     1_223.97       0.9216     0.078134         3.47
IVF-RaBitQ-nl273-np23-rf10 (query)                       978.15       299.41     1_277.57       0.9857     0.010558         3.47
IVF-RaBitQ-nl273-np23-rf20 (query)                       978.15       385.61     1_363.76       0.9988     0.000741         3.47
IVF-RaBitQ-nl273 (self)                                  978.15     3_858.46     4_836.61       0.9988     0.000729         3.47
IVF-RaBitQ-nl387-np19-rf0 (query)                      1_319.66       120.63     1_440.29       0.5004     7.112884         3.49
IVF-RaBitQ-nl387-np27-rf0 (query)                      1_319.66       164.96     1_484.62       0.5004     7.112886         3.49
IVF-RaBitQ-nl387-np19-rf5 (query)                      1_319.66       172.21     1_491.87       0.9290     0.068019         3.49
IVF-RaBitQ-nl387-np19-rf10 (query)                     1_319.66       219.44     1_539.10       0.9875     0.009165         3.49
IVF-RaBitQ-nl387-np19-rf20 (query)                     1_319.66       291.36     1_611.02       0.9992     0.000510         3.49
IVF-RaBitQ-nl387-np27-rf5 (query)                      1_319.66       220.60     1_540.26       0.9289     0.068041         3.49
IVF-RaBitQ-nl387-np27-rf10 (query)                     1_319.66       269.59     1_589.25       0.9874     0.009171         3.49
IVF-RaBitQ-nl387-np27-rf20 (query)                     1_319.66       350.38     1_670.04       0.9992     0.000510         3.49
IVF-RaBitQ-nl387 (self)                                1_319.66     3_498.15     4_817.81       0.9991     0.000580         3.49
IVF-RaBitQ-nl547-np23-rf0 (query)                      1_837.86       117.55     1_955.40       0.5107     7.089671         3.51
IVF-RaBitQ-nl547-np27-rf0 (query)                      1_837.86       130.64     1_968.50       0.5107     7.089685         3.51
IVF-RaBitQ-nl547-np33-rf0 (query)                      1_837.86       153.57     1_991.43       0.5107     7.089688         3.51
IVF-RaBitQ-nl547-np23-rf5 (query)                      1_837.86       163.87     2_001.72       0.9363     0.058700         3.51
IVF-RaBitQ-nl547-np23-rf10 (query)                     1_837.86       206.57     2_044.43       0.9896     0.007452         3.51
IVF-RaBitQ-nl547-np23-rf20 (query)                     1_837.86       278.10     2_115.96       0.9994     0.000388         3.51
IVF-RaBitQ-nl547-np27-rf5 (query)                      1_837.86       181.28     2_019.13       0.9362     0.058742         3.51
IVF-RaBitQ-nl547-np27-rf10 (query)                     1_837.86       226.48     2_064.33       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np27-rf20 (query)                     1_837.86       320.32     2_158.18       0.9994     0.000393         3.51
IVF-RaBitQ-nl547-np33-rf5 (query)                      1_837.86       208.50     2_046.36       0.9362     0.058742         3.51
IVF-RaBitQ-nl547-np33-rf10 (query)                     1_837.86       261.89     2_099.75       0.9895     0.007453         3.51
IVF-RaBitQ-nl547-np33-rf20 (query)                     1_837.86       332.79     2_170.64       0.9994     0.000393         3.51
IVF-RaBitQ-nl547 (self)                                1_837.86     3_317.85     5_155.70       0.9993     0.000388         3.51
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With higher dimensionality

RaBitQ particularly shines with higher dimensionality in the data.

<details>
<summary><b>RaBitQ - Euclidean (Gaussian - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.19     6_118.83     6_133.01       1.0000     0.000000        73.24
Exhaustive (self)                                         14.19    69_404.57    69_418.76       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           5_300.49       999.33     6_299.82       0.3639   349.112289         5.31
ExhaustiveRaBitQ-rf5 (query)                           5_300.49     1_099.73     6_400.23       0.7183     3.897594         5.31
ExhaustiveRaBitQ-rf10 (query)                          5_300.49     1_261.72     6_562.21       0.8545     1.514929         5.31
ExhaustiveRaBitQ-rf20 (query)                          5_300.49     1_292.67     6_593.16       0.9452     0.461817         5.31
ExhaustiveRaBitQ (self)                                5_300.49    11_416.41    16_716.90       0.8556     1.514067         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      3_419.99       273.40     3_693.39       0.3589   349.079531         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      3_419.99       326.43     3_746.43       0.3616   349.097123         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      3_419.99       449.03     3_869.03       0.3629   349.109287         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      3_419.99       333.68     3_753.67       0.6970     4.355675         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     3_419.99       393.06     3_813.05       0.8213     2.049424         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     3_419.99       506.10     3_926.10       0.8983     1.070961         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      3_419.99       428.72     3_848.71       0.7108     4.051565         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     3_419.99       476.62     3_896.61       0.8427     1.695339         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     3_419.99       680.58     4_100.58       0.9271     0.681385         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      3_419.99       547.13     3_967.13       0.7189     3.877703         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     3_419.99       593.92     4_013.92       0.8557     1.498582         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     3_419.99       773.59     4_193.58       0.9453     0.462298         5.35
IVF-RaBitQ-nl273 (self)                                3_419.99     7_345.25    10_765.25       0.9460     0.461372         5.35
IVF-RaBitQ-nl387-np19-rf0 (query)                      4_552.30       313.21     4_865.51       0.3614   349.087561         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      4_552.30       483.79     5_036.09       0.3638   349.107939         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      4_552.30       449.10     5_001.40       0.7069     4.172279         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     4_552.30       500.96     5_053.26       0.8332     1.888360         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     4_552.30       556.62     5_108.93       0.9146     0.882773         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      4_552.30       481.68     5_033.98       0.7208     3.855511         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     4_552.30       549.28     5_101.58       0.8551     1.518584         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     4_552.30       658.49     5_210.79       0.9449     0.470040         5.41
IVF-RaBitQ-nl387 (self)                                4_552.30     7_716.52    12_268.82       0.9440     0.495864         5.41
IVF-RaBitQ-nl547-np23-rf0 (query)                      6_288.84       360.51     6_649.36       0.3606   349.077719         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      6_288.84       415.89     6_704.73       0.3624   349.092489         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      6_288.84       447.46     6_736.30       0.3640   349.103934         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      6_288.84       437.28     6_726.12       0.7003     4.298619         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     6_288.84       466.66     6_755.51       0.8238     2.019129         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     6_288.84       566.36     6_855.20       0.9017     1.026178         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      6_288.84       448.87     6_737.71       0.7108     4.043362         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     6_288.84       503.23     6_792.07       0.8402     1.724642         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     6_288.84       618.33     6_907.17       0.9242     0.706346         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      6_288.84       504.92     6_793.76       0.7185     3.898642         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     6_288.84       562.91     6_851.75       0.8526     1.550835         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     6_288.84       672.18     6_961.02       0.9419     0.506257         5.49
IVF-RaBitQ-nl547 (self)                                6_288.84     6_880.57    13_169.42       0.9415     0.506658         5.49
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (Correlated - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        15.07     7_213.49     7_228.56       1.0000     0.000000        73.24
Exhaustive (self)                                         15.07    69_382.77    69_397.84       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           5_351.05       924.57     6_275.62       0.5388    88.769533         5.31
ExhaustiveRaBitQ-rf5 (query)                           5_351.05       997.24     6_348.29       0.9253     0.242421         5.31
ExhaustiveRaBitQ-rf10 (query)                          5_351.05     1_055.35     6_406.40       0.9836     0.038503         5.31
ExhaustiveRaBitQ-rf20 (query)                          5_351.05     1_184.38     6_535.43       0.9982     0.003158         5.31
ExhaustiveRaBitQ (self)                                5_351.05    10_548.49    15_899.54       0.9837     0.038445         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      3_245.08       274.18     3_519.26       0.5446    88.760310         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      3_245.08       323.17     3_568.25       0.5445    88.762307         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      3_245.08       443.97     3_689.04       0.5444    88.763081         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      3_245.08       333.54     3_578.62       0.9268     0.257748         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     3_245.08       383.01     3_628.09       0.9796     0.062082         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     3_245.08       482.70     3_727.77       0.9921     0.028068         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      3_245.08       397.25     3_642.32       0.9294     0.243259         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     3_245.08       439.16     3_684.23       0.9838     0.044760         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     3_245.08       538.05     3_783.12       0.9969     0.009793         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      3_245.08       516.42     3_761.49       0.9301     0.238085         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     3_245.08       573.95     3_819.03       0.9850     0.038312         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     3_245.08       679.92     3_924.99       0.9984     0.003068         5.35
IVF-RaBitQ-nl273 (self)                                3_245.08     6_802.99    10_048.06       0.9985     0.002875         5.35
IVF-RaBitQ-nl387-np19-rf0 (query)                      4_336.23       314.74     4_650.97       0.5474    88.755538         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      4_336.23       423.60     4_759.83       0.5471    88.757297         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      4_336.23       372.51     4_708.74       0.9301     0.240722         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     4_336.23       424.12     4_760.35       0.9826     0.049320         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     4_336.23       517.51     4_853.74       0.9944     0.017985         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      4_336.23       486.38     4_822.61       0.9320     0.229739         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     4_336.23       541.79     4_878.02       0.9859     0.035760         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     4_336.23       642.73     4_978.96       0.9985     0.003076         5.41
IVF-RaBitQ-nl387 (self)                                4_336.23     6_438.07    10_774.30       0.9985     0.003013         5.41
IVF-RaBitQ-nl547-np23-rf0 (query)                      5_921.62       324.11     6_245.73       0.5509    88.748225         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      5_921.62       366.63     6_288.25       0.5509    88.750112         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      5_921.62       431.74     6_353.36       0.5508    88.751159         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      5_921.62       380.09     6_301.71       0.9305     0.234073         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     5_921.62       427.99     6_349.61       0.9807     0.054097         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     5_921.62       521.11     6_442.73       0.9917     0.025312         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      5_921.62       420.34     6_341.97       0.9331     0.221993         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     5_921.62       474.78     6_396.40       0.9845     0.039788         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     5_921.62       570.93     6_492.55       0.9961     0.010073         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      5_921.62       486.66     6_408.28       0.9342     0.216583         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     5_921.62       542.90     6_464.52       0.9864     0.033124         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     5_921.62       641.44     6_563.06       0.9985     0.002691         5.49
IVF-RaBitQ-nl547 (self)                                5_921.62     6_395.71    12_317.34       0.9983     0.003533         5.49
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
================================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        14.79     6_380.16     6_394.95       1.0000     0.000000        73.24
Exhaustive (self)                                         14.79    64_528.65    64_543.44       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                           5_116.58       898.26     6_014.84       0.7024    73.760101         5.31
ExhaustiveRaBitQ-rf5 (query)                           5_116.58       978.58     6_095.16       0.9952     0.017331         5.31
ExhaustiveRaBitQ-rf10 (query)                          5_116.58     1_068.81     6_185.40       0.9999     0.000453         5.31
ExhaustiveRaBitQ-rf20 (query)                          5_116.58     1_175.04     6_291.62       1.0000     0.000000         5.31
ExhaustiveRaBitQ (self)                                5_116.58    10_306.07    15_422.65       0.9999     0.000420         5.31
--------------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                      3_261.52       263.67     3_525.19       0.7066    73.750602         5.35
IVF-RaBitQ-nl273-np16-rf0 (query)                      3_261.52       313.63     3_575.15       0.7066    73.750778         5.35
IVF-RaBitQ-nl273-np23-rf0 (query)                      3_261.52       437.52     3_699.04       0.7066    73.750784         5.35
IVF-RaBitQ-nl273-np13-rf5 (query)                      3_261.52       331.32     3_592.84       0.9954     0.017287         5.35
IVF-RaBitQ-nl273-np13-rf10 (query)                     3_261.52       374.16     3_635.68       0.9996     0.002368         5.35
IVF-RaBitQ-nl273-np13-rf20 (query)                     3_261.52       466.58     3_728.10       0.9997     0.002034         5.35
IVF-RaBitQ-nl273-np16-rf5 (query)                      3_261.52       372.96     3_634.48       0.9956     0.015465         5.35
IVF-RaBitQ-nl273-np16-rf10 (query)                     3_261.52       431.17     3_692.69       0.9999     0.000515         5.35
IVF-RaBitQ-nl273-np16-rf20 (query)                     3_261.52       539.32     3_800.84       1.0000     0.000182         5.35
IVF-RaBitQ-nl273-np23-rf5 (query)                      3_261.52       484.98     3_746.50       0.9956     0.015283         5.35
IVF-RaBitQ-nl273-np23-rf10 (query)                     3_261.52       545.22     3_806.74       0.9999     0.000333         5.35
IVF-RaBitQ-nl273-np23-rf20 (query)                     3_261.52       650.61     3_912.13       1.0000     0.000000         5.35
IVF-RaBitQ-nl273 (self)                                3_261.52     6_457.20     9_718.72       1.0000     0.000001         5.35
IVF-RaBitQ-nl387-np19-rf0 (query)                      4_368.28       307.46     4_675.74       0.7122    73.742352         5.41
IVF-RaBitQ-nl387-np27-rf0 (query)                      4_368.28       409.26     4_777.54       0.7122    73.742373         5.41
IVF-RaBitQ-nl387-np19-rf5 (query)                      4_368.28       366.35     4_734.63       0.9959     0.014264         5.41
IVF-RaBitQ-nl387-np19-rf10 (query)                     4_368.28       417.35     4_785.63       0.9999     0.000347         5.41
IVF-RaBitQ-nl387-np19-rf20 (query)                     4_368.28       509.61     4_877.90       1.0000     0.000111         5.41
IVF-RaBitQ-nl387-np27-rf5 (query)                      4_368.28       467.55     4_835.83       0.9959     0.014164         5.41
IVF-RaBitQ-nl387-np27-rf10 (query)                     4_368.28       524.69     4_892.98       0.9999     0.000236         5.41
IVF-RaBitQ-nl387-np27-rf20 (query)                     4_368.28       625.25     4_993.53       1.0000     0.000000         5.41
IVF-RaBitQ-nl387 (self)                                4_368.28     6_190.39    10_558.67       1.0000     0.000004         5.41
IVF-RaBitQ-nl547-np23-rf0 (query)                      5_939.23       319.57     6_258.80       0.7163    73.731249         5.49
IVF-RaBitQ-nl547-np27-rf0 (query)                      5_939.23       361.85     6_301.08       0.7162    73.731338         5.49
IVF-RaBitQ-nl547-np33-rf0 (query)                      5_939.23       423.12     6_362.35       0.7162    73.731350         5.49
IVF-RaBitQ-nl547-np23-rf5 (query)                      5_939.23       376.22     6_315.45       0.9962     0.013968         5.49
IVF-RaBitQ-nl547-np23-rf10 (query)                     5_939.23       425.83     6_365.07       0.9998     0.000954         5.49
IVF-RaBitQ-nl547-np23-rf20 (query)                     5_939.23       546.03     6_485.26       0.9999     0.000615         5.49
IVF-RaBitQ-nl547-np27-rf5 (query)                      5_939.23       417.21     6_356.44       0.9963     0.013478         5.49
IVF-RaBitQ-nl547-np27-rf10 (query)                     5_939.23       470.33     6_409.57       0.9999     0.000400         5.49
IVF-RaBitQ-nl547-np27-rf20 (query)                     5_939.23       569.91     6_509.14       1.0000     0.000062         5.49
IVF-RaBitQ-nl547-np33-rf5 (query)                      5_939.23       480.26     6_419.49       0.9963     0.013416         5.49
IVF-RaBitQ-nl547-np33-rf10 (query)                     5_939.23       537.11     6_476.34       0.9999     0.000339         5.49
IVF-RaBitQ-nl547-np33-rf20 (query)                     5_939.23       628.76     6_567.99       1.0000     0.000000         5.49
IVF-RaBitQ-nl547 (self)                                5_939.23     6_539.54    12_478.77       1.0000     0.000001         5.49
--------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls.

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*