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
- [RaBitQ](#sq8-ivf)

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
Exhaustive (query)                                         3.06     2_420.66     2_423.72       1.0000     0.000000        18.31
Exhaustive (self)                                          3.06    23_831.07    23_834.14       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_259.27     1_019.74     2_279.01       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_259.27     1_025.05     2_284.32       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_259.27     1_074.52     2_333.79       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_259.27     1_163.57     2_422.84       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                     1_259.27    10_723.52    11_982.79       0.5461     2.056589         4.61
ExhaustiveBinary-256-itq_no_rr (query)                16_410.63       969.80    17_380.43       0.1962          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                  16_410.63     1_017.92    17_428.55       0.4008     3.839819         4.61
ExhaustiveBinary-256-itq-rf10 (query)                 16_410.63     1_082.57    17_493.19       0.5266     2.152408         4.61
ExhaustiveBinary-256-itq-rf20 (query)                 16_410.63     1_166.67    17_577.30       0.6623     1.116198         4.61
ExhaustiveBinary-256-itq (self)                       16_410.63    10_658.97    27_069.60       0.5242     2.138272         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_493.10     1_472.41     3_965.50       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_493.10     1_555.49     4_048.59       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_493.10     1_619.93     4_113.03       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_493.10     1_688.76     4_181.86       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                     2_493.10    16_266.62    18_759.72       0.7223     0.828915         9.22
ExhaustiveBinary-512-itq_no_rr (query)                14_919.80     1_503.68    16_423.48       0.2923          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  14_919.80     1_546.50    16_466.30       0.5833     1.686259         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 14_919.80     1_584.18    16_503.98       0.7204     0.786147         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 14_919.80     1_694.20    16_614.00       0.8387     0.321334         9.22
ExhaustiveBinary-512-itq (self)                       14_919.80    16_325.02    31_244.82       0.7197     0.773467         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_582.59        84.48     2_667.07       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_582.59        94.00     2_676.59       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_582.59       119.26     2_701.85       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_582.59       108.54     2_691.13       0.4238     3.637807         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_582.59       136.17     2_718.77       0.5532     2.020523         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_582.59       186.08     2_768.67       0.6927     1.020427         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_582.59       121.74     2_704.33       0.4222     3.662764         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_582.59       151.62     2_734.21       0.5515     2.033556         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_582.59       203.66     2_786.25       0.6913     1.025018         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_582.59       151.18     2_733.78       0.4198     3.692510         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_582.59       183.57     2_766.16       0.5484     2.057232         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_582.59       242.52     2_825.11       0.6879     1.040725         5.79
IVF-Binary-256-nl273-random (self)                     2_582.59     1_513.42     4_096.01       0.5511     2.018621         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           3_088.75        81.74     3_170.49       0.2061          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           3_088.75       105.76     3_194.51       0.2047          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           3_088.75       109.83     3_198.59       0.4236     3.636869         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          3_088.75       142.27     3_231.03       0.5533     2.013370         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          3_088.75       188.19     3_276.94       0.6933     1.015182         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           3_088.75       132.92     3_221.67       0.4203     3.684047         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          3_088.75       161.67     3_250.42       0.5495     2.044872         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          3_088.75       214.76     3_303.51       0.6888     1.036169         5.80
IVF-Binary-256-nl387-random (self)                     3_088.75     1_381.59     4_470.34       0.5532     1.998144         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           3_850.06        75.68     3_925.75       0.2077          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           3_850.06        83.46     3_933.53       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           3_850.06        94.71     3_944.77       0.2051          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           3_850.06       105.93     3_956.00       0.4267     3.592322         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          3_850.06       131.42     3_981.48       0.5577     1.983086         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          3_850.06       176.86     4_026.93       0.6978     0.993692         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           3_850.06       111.68     3_961.74       0.4239     3.627604         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          3_850.06       139.50     3_989.57       0.5539     2.008636         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          3_850.06       187.63     4_037.69       0.6943     1.006723         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           3_850.06       124.10     3_974.17       0.4211     3.671793         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          3_850.06       152.82     4_002.89       0.5506     2.035867         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          3_850.06       205.86     4_055.92       0.6901     1.028793         5.82
IVF-Binary-256-nl547-random (self)                     3_850.06     1_406.20     5_256.27       0.5570     1.967723         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)             11_509.67        80.14    11_589.81       0.1998          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             11_509.67        88.79    11_598.45       0.1985          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             11_509.67       119.74    11_629.41       0.1972          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             11_509.67       114.22    11_623.89       0.4073     3.756173         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            11_509.67       132.98    11_642.65       0.5343     2.102188         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            11_509.67       181.23    11_690.90       0.6698     1.087921         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             11_509.67       120.52    11_630.19       0.4054     3.777077         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            11_509.67       150.62    11_660.29       0.5321     2.114178         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            11_509.67       201.25    11_710.92       0.6682     1.087966         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             11_509.67       148.49    11_658.16       0.4026     3.813004         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            11_509.67       181.69    11_691.36       0.5289     2.137595         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            11_509.67       231.52    11_741.19       0.6647     1.105353         5.79
IVF-Binary-256-nl273-itq (self)                       11_509.67     1_461.51    12_971.18       0.5300     2.095999         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             11_327.11        80.05    11_407.16       0.1994          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             11_327.11       100.91    11_428.02       0.1977          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             11_327.11       109.31    11_436.42       0.4073     3.753834         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            11_327.11       134.44    11_461.55       0.5346     2.090557         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            11_327.11       182.96    11_510.07       0.6705     1.081527         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             11_327.11       131.81    11_458.92       0.4038     3.798721         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            11_327.11       160.24    11_487.35       0.5299     2.126262         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            11_327.11       208.27    11_535.38       0.6657     1.098482         5.80
IVF-Binary-256-nl387-itq (self)                       11_327.11     1_338.49    12_665.60       0.5320     2.081036         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_929.50        76.47    12_005.97       0.2009          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_929.50        83.43    12_012.93       0.1996          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_929.50        94.00    12_023.50       0.1984          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_929.50       102.90    12_032.40       0.4104     3.709476         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_929.50       127.74    12_057.24       0.5382     2.061970         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_929.50       174.10    12_103.60       0.6754     1.058384         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_929.50       109.86    12_039.36       0.4076     3.744712         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_929.50       136.93    12_066.43       0.5348     2.086013         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_929.50       185.49    12_114.99       0.6714     1.072192         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_929.50       123.75    12_053.25       0.4049     3.783797         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_929.50       150.55    12_080.04       0.5311     2.116576         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_929.50       198.90    12_128.40       0.6675     1.090212         5.82
IVF-Binary-256-nl547-itq (self)                       11_929.50     1_268.62    13_198.12       0.5364     2.045439         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_752.05       141.53     3_893.58       0.2931          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_752.05       163.45     3_915.50       0.2924          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_752.05       213.44     3_965.49       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_752.05       182.78     3_934.84       0.5865     1.777992        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_752.05       214.93     3_966.98       0.7255     0.834135        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_752.05       273.02     4_025.07       0.8433     0.358399        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_752.05       206.80     3_958.85       0.5862     1.774695        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_752.05       243.11     3_995.16       0.7262     0.823354        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_752.05       306.53     4_058.58       0.8455     0.341344        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_752.05       257.20     4_009.25       0.5848     1.784378        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_752.05       294.00     4_046.05       0.7249     0.828663        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_752.05       363.64     4_115.69       0.8447     0.342612        10.40
IVF-Binary-512-nl273-random (self)                     3_752.05     2_382.97     6_135.02       0.7247     0.820023        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           4_313.07       152.94     4_466.01       0.2927          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           4_313.07       193.34     4_506.41       0.2917          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           4_313.07       185.23     4_498.30       0.5872     1.769201        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          4_313.07       220.73     4_533.80       0.7264     0.828231        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          4_313.07       279.84     4_592.91       0.8448     0.354222        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           4_313.07       230.38     4_543.45       0.5853     1.779511        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          4_313.07       266.57     4_579.64       0.7252     0.826392        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          4_313.07       333.94     4_647.01       0.8453     0.340730        10.41
IVF-Binary-512-nl387-random (self)                     4_313.07     2_239.44     6_552.51       0.7248     0.827287        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           5_048.62       138.69     5_187.31       0.2937          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           5_048.62       156.00     5_204.62       0.2930          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           5_048.62       176.46     5_225.08       0.2922          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           5_048.62       175.08     5_223.70       0.5886     1.763029        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          5_048.62       208.74     5_257.36       0.7282     0.824085        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          5_048.62       267.95     5_316.57       0.8461     0.353746        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           5_048.62       201.94     5_250.56       0.5871     1.766634        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          5_048.62       231.07     5_279.69       0.7272     0.820294        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          5_048.62       282.30     5_330.92       0.8464     0.341244        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           5_048.62       215.13     5_263.75       0.5855     1.778317        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          5_048.62       249.06     5_297.68       0.7259     0.823829        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          5_048.62       314.62     5_363.24       0.8456     0.339064        10.43
IVF-Binary-512-nl547-random (self)                     5_048.62     2_068.45     7_117.07       0.7268     0.818373        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_962.96       144.04    12_107.00       0.2945          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_962.96       166.50    12_129.46       0.2938          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_962.96       216.57    12_179.53       0.2928          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_962.96       180.66    12_143.62       0.5860     1.674360        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_962.96       215.08    12_178.04       0.7221     0.788952        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_962.96       283.92    12_246.88       0.8380     0.337175        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_962.96       205.71    12_168.67       0.5857     1.670209        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_962.96       244.03    12_206.99       0.7230     0.777354        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_962.96       307.92    12_270.88       0.8404     0.318569        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_962.96       256.95    12_219.91       0.5844     1.678646        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_962.96       297.26    12_260.22       0.7216     0.782084        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_962.96       365.14    12_328.10       0.8395     0.319643        10.40
IVF-Binary-512-nl273-itq (self)                       11_962.96     2_400.04    14_362.99       0.7222     0.764999        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_272.70       151.36    12_424.06       0.2946          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_272.70       192.83    12_465.54       0.2932          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_272.70       194.57    12_467.27       0.5863     1.671970        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_272.70       226.46    12_499.16       0.7230     0.785252        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_272.70       278.42    12_551.12       0.8395     0.333258        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_272.70       231.87    12_504.57       0.5851     1.673288        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_272.70       267.41    12_540.12       0.7222     0.779467        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_272.70       330.22    12_602.92       0.8400     0.317734        10.41
IVF-Binary-512-nl387-itq (self)                       12_272.70     2_331.59    14_604.29       0.7220     0.775595        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             15_783.98       144.08    15_928.06       0.2950          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             15_783.98       154.13    15_938.11       0.2942          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             15_783.98       189.36    15_973.35       0.2937          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             15_783.98       187.16    15_971.14       0.5881     1.660943        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            15_783.98       335.88    16_119.86       0.7250     0.779199        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            15_783.98       285.31    16_069.29       0.8409     0.333310        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             15_783.98       196.23    15_980.21       0.5872     1.661009        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            15_783.98       234.02    16_018.00       0.7243     0.773708        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            15_783.98       297.90    16_081.88       0.8415     0.319924        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             15_783.98       213.88    15_997.87       0.5855     1.671083        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            15_783.98       289.41    16_073.39       0.7230     0.776685        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            15_783.98       310.82    16_094.81       0.8406     0.316908        10.43
IVF-Binary-512-nl547-itq (self)                       15_783.98     2_059.37    17_843.36       0.7241     0.767564        10.43
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
Exhaustive (query)                                         4.15     2_427.61     2_431.76       1.0000     0.000000        18.88
Exhaustive (self)                                          4.15    24_229.64    24_233.79       1.0000     0.000000        18.88
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_247.25       985.93     2_233.18       0.2159          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_247.25     1_032.29     2_279.54       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_247.25     1_081.46     2_328.71       0.5704     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_247.25     1_165.82     2_413.07       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                     1_247.25    10_719.26    11_966.51       0.5696     0.001382         4.61
ExhaustiveBinary-256-itq_no_rr (query)                14_591.28     1_043.02    15_634.30       0.2068          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                  14_591.28     1_062.70    15_653.98       0.4203     0.002652         4.61
ExhaustiveBinary-256-itq-rf10 (query)                 14_591.28     1_081.76    15_673.04       0.5468     0.001484         4.61
ExhaustiveBinary-256-itq-rf20 (query)                 14_591.28     1_188.03    15_779.31       0.6809     0.000767         4.61
ExhaustiveBinary-256-itq (self)                       14_591.28    10_810.28    25_401.56       0.5456     0.001474         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_459.88     1_568.24     4_028.13       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_459.88     1_577.14     4_037.02       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_459.88     1_594.30     4_054.19       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_459.88     1_700.76     4_160.64       0.8651     0.000202         9.22
ExhaustiveBinary-512-random (self)                     2_459.88    15_882.10    18_341.98       0.7519     0.000503         9.22
ExhaustiveBinary-512-itq_no_rr (query)                10_453.15     1_494.69    11_947.83       0.3144          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  10_453.15     1_548.85    12_002.00       0.6144     0.001065         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 10_453.15     1_578.45    12_031.60       0.7478     0.000491         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 10_453.15     1_684.37    12_137.51       0.8576     0.000201         9.22
ExhaustiveBinary-512-itq (self)                       10_453.15    16_676.35    27_129.49       0.7466     0.000485         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_653.98        79.00     2_732.98       0.2191          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_653.98        94.67     2_748.65       0.2180          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_653.98       124.14     2_778.12       0.2172          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_653.98       109.12     2_763.10       0.4457     0.002467         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_653.98       155.99     2_809.97       0.5771     0.001354         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_653.98       188.74     2_842.72       0.7144     0.000676         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_653.98       122.33     2_776.31       0.4441     0.002486         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_653.98       157.54     2_811.52       0.5756     0.001360         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_653.98       205.99     2_859.97       0.7131     0.000678         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_653.98       153.44     2_807.42       0.4418     0.002507         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_653.98       185.10     2_839.08       0.5727     0.001376         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_653.98       254.68     2_908.66       0.7100     0.000689         5.79
IVF-Binary-256-nl273-random (self)                     2_653.98     1_524.12     4_178.10       0.5747     0.001354         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           3_373.17        90.11     3_463.28       0.2187          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)           3_373.17       105.66     3_478.83       0.2171          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)           3_373.17       115.92     3_489.09       0.4457     0.002464         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)          3_373.17       139.16     3_512.34       0.5776     0.001347         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)          3_373.17       199.20     3_572.37       0.7151     0.000670         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)           3_373.17       136.05     3_509.22       0.4423     0.002501         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)          3_373.17       174.78     3_547.95       0.5736     0.001369         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)          3_373.17       217.89     3_591.06       0.7108     0.000685         5.81
IVF-Binary-256-nl387-random (self)                     3_373.17     1_412.73     4_785.90       0.5768     0.001341         5.81
IVF-Binary-256-nl547-np23-rf0-random (query)           4_065.10        89.95     4_155.05       0.2198          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)           4_065.10        85.03     4_150.13       0.2184          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)           4_065.10       115.73     4_180.83       0.2175          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)           4_065.10       107.72     4_172.82       0.4486     0.002436         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)          4_065.10       152.21     4_217.31       0.5811     0.001327         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)          4_065.10       193.95     4_259.05       0.7191     0.000657         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)           4_065.10       116.37     4_181.47       0.4457     0.002464         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)          4_065.10       145.56     4_210.66       0.5778     0.001344         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)          4_065.10       201.09     4_266.19       0.7159     0.000666         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)           4_065.10       133.63     4_198.73       0.4429     0.002492         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)          4_065.10       177.09     4_242.19       0.5744     0.001365         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)          4_065.10       208.66     4_273.76       0.7120     0.000680         5.83
IVF-Binary-256-nl547-random (self)                     4_065.10     1_366.88     5_431.98       0.5805     0.001320         5.83
IVF-Binary-256-nl273-np13-rf0-itq (query)             15_113.69        83.91    15_197.61       0.2105          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             15_113.69        93.45    15_207.14       0.2090          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             15_113.69       119.57    15_233.26       0.2078          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             15_113.69       107.75    15_221.44       0.4269     0.002592         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            15_113.69       140.48    15_254.17       0.5547     0.001443         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            15_113.69       186.41    15_300.11       0.6882     0.000747         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             15_113.69       141.43    15_255.12       0.4249     0.002611         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            15_113.69       152.09    15_265.78       0.5524     0.001454         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            15_113.69       213.61    15_327.30       0.6867     0.000748         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             15_113.69       154.61    15_268.31       0.4223     0.002633         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            15_113.69       184.01    15_297.70       0.5490     0.001473         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            15_113.69       243.35    15_357.04       0.6831     0.000761         5.79
IVF-Binary-256-nl273-itq (self)                       15_113.69     1_523.59    16_637.28       0.5513     0.001444         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             11_635.23        89.36    11_724.60       0.2103          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)             11_635.23       127.78    11_763.02       0.2084          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)             11_635.23       118.66    11_753.89       0.4266     0.002589         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)            11_635.23       148.20    11_783.44       0.5548     0.001439         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)            11_635.23       204.46    11_839.70       0.6889     0.000742         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)             11_635.23       148.40    11_783.63       0.4231     0.002624         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)            11_635.23       188.94    11_824.17       0.5501     0.001464         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)            11_635.23       220.37    11_855.60       0.6844     0.000755         5.81
IVF-Binary-256-nl387-itq (self)                       11_635.23     1_384.89    13_020.12       0.5536     0.001430         5.81
IVF-Binary-256-nl547-np23-rf0-itq (query)             13_924.42        77.67    14_002.09       0.2111          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)             13_924.42       120.94    14_045.37       0.2100          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)             13_924.42        99.83    14_024.26       0.2086          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)             13_924.42       115.87    14_040.29       0.4302     0.002554         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)            13_924.42       169.17    14_093.60       0.5584     0.001417         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)            13_924.42       193.46    14_117.89       0.6934     0.000724         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)             13_924.42       117.00    14_041.43       0.4274     0.002581         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)            13_924.42       177.92    14_102.34       0.5550     0.001434         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)            13_924.42       264.03    14_188.46       0.6898     0.000735         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)             13_924.42       134.99    14_059.41       0.4247     0.002610         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)            13_924.42       157.11    14_081.54       0.5512     0.001458         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)            13_924.42       207.08    14_131.50       0.6857     0.000750         5.83
IVF-Binary-256-nl547-itq (self)                       13_924.42     1_321.40    15_245.83       0.5574     0.001407         5.83
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_860.51       148.33     4_008.84       0.3154          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_860.51       176.82     4_037.34       0.3149          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_860.51       229.60     4_090.11       0.3142          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_860.51       186.34     4_046.85       0.6202     0.001095        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_860.51       245.51     4_106.03       0.7548     0.000506        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_860.51       307.12     4_167.64       0.8637     0.000214        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_860.51       231.36     4_091.87       0.6203     0.001091        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_860.51       255.82     4_116.33       0.7559     0.000498        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_860.51       325.07     4_185.58       0.8663     0.000202        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_860.51       272.48     4_133.00       0.6191     0.001098        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_860.51       313.21     4_173.73       0.7549     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_860.51       404.80     4_265.31       0.8659     0.000201        10.40
IVF-Binary-512-nl273-random (self)                     3_860.51     2_551.30     6_411.81       0.7541     0.000497        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           4_436.90       152.26     4_589.16       0.3157          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           4_436.90       194.84     4_631.74       0.3146          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           4_436.90       190.05     4_626.95       0.6208     0.001091        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          4_436.90       235.48     4_672.38       0.7558     0.000502        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          4_436.90       288.28     4_725.18       0.8653     0.000209        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           4_436.90       243.53     4_680.42       0.6194     0.001096        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          4_436.90       277.63     4_714.52       0.7551     0.000499        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          4_436.90       343.84     4_780.73       0.8661     0.000201        10.41
IVF-Binary-512-nl387-random (self)                     4_436.90     2_307.20     6_744.09       0.7539     0.000502        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           5_230.40       136.15     5_366.55       0.3165          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)           5_230.40       151.15     5_381.55       0.3157          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)           5_230.40       175.10     5_405.49       0.3148          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)           5_230.40       186.42     5_416.82       0.6226     0.001082        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)          5_230.40       213.79     5_444.19       0.7570     0.000498        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)          5_230.40       270.30     5_500.70       0.8660     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)           5_230.40       196.35     5_426.75       0.6214     0.001086        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)          5_230.40       248.23     5_478.63       0.7567     0.000494        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)          5_230.40       294.15     5_524.55       0.8671     0.000201        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)           5_230.40       223.62     5_454.02       0.6198     0.001094        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)          5_230.40       264.75     5_495.15       0.7557     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)          5_230.40       331.39     5_561.79       0.8664     0.000200        10.44
IVF-Binary-512-nl547-random (self)                     5_230.40     2_149.18     7_379.58       0.7554     0.000498        10.44
IVF-Binary-512-nl273-np13-rf0-itq (query)             12_252.40       153.80    12_406.21       0.3162          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             12_252.40       175.19    12_427.59       0.3159          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             12_252.40       236.55    12_488.96       0.3148          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             12_252.40       208.91    12_461.31       0.6165     0.001061        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            12_252.40       226.15    12_478.56       0.7492     0.000494        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            12_252.40       294.49    12_546.89       0.8566     0.000213        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             12_252.40       215.06    12_467.47       0.6168     0.001055        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            12_252.40       271.97    12_524.38       0.7505     0.000485        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            12_252.40       328.65    12_581.06       0.8593     0.000199        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             12_252.40       293.08    12_545.48       0.6156     0.001061        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            12_252.40       321.70    12_574.10       0.7490     0.000488        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            12_252.40       409.16    12_661.56       0.8584     0.000199        10.40
IVF-Binary-512-nl273-itq (self)                       12_252.40     2_553.59    14_806.00       0.7489     0.000479        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_758.95       152.94    12_911.89       0.3163          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_758.95       199.63    12_958.58       0.3151          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_758.95       195.60    12_954.55       0.6174     0.001054        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_758.95       227.85    12_986.80       0.7500     0.000490        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_758.95       289.66    13_048.61       0.8582     0.000208        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_758.95       254.53    13_013.48       0.6160     0.001057        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_758.95       284.68    13_043.63       0.7494     0.000487        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_758.95       346.23    13_105.18       0.8587     0.000199        10.41
IVF-Binary-512-nl387-itq (self)                       12_758.95     2_310.67    15_069.62       0.7486     0.000485        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             13_549.82       140.11    13_689.92       0.3175          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)             13_549.82       155.81    13_705.62       0.3167          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)             13_549.82       189.73    13_739.54       0.3158          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)             13_549.82       189.28    13_739.09       0.6189     0.001047        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)            13_549.82       214.47    13_764.28       0.7517     0.000486        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)            13_549.82       271.13    13_820.95       0.8589     0.000208        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)             13_549.82       217.77    13_767.59       0.6181     0.001048        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)            13_549.82       234.33    13_784.15       0.7511     0.000483        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)            13_549.82       300.66    13_850.47       0.8600     0.000199        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)             13_549.82       224.00    13_773.82       0.6165     0.001055        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)            13_549.82       259.76    13_809.58       0.7497     0.000486        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)            13_549.82       336.94    13_886.75       0.8592     0.000198        10.44
IVF-Binary-512-nl547-itq (self)                       13_549.82     2_200.91    15_750.72       0.7501     0.000482        10.44
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
Exhaustive (query)                                         3.84     2_421.69     2_425.53       1.0000     0.000000        18.31
Exhaustive (self)                                          3.84    24_134.28    24_138.12       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_266.05     1_046.51     2_312.55       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_266.05     1_052.73     2_318.78       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_266.05     1_102.06     2_368.10       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_266.05     1_202.68     2_468.73       0.5614     0.256662         4.61
ExhaustiveBinary-256-random (self)                     1_266.05    10_950.16    12_216.20       0.4294     0.437191         4.61
ExhaustiveBinary-256-itq_no_rr (query)                 9_820.15     1_012.87    10_833.02       0.1197          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                   9_820.15     1_041.63    10_861.78       0.2765     0.831692         4.61
ExhaustiveBinary-256-itq-rf10 (query)                  9_820.15     1_092.87    10_913.02       0.3861     0.506502         4.61
ExhaustiveBinary-256-itq-rf20 (query)                  9_820.15     1_188.97    11_009.12       0.5230     0.288077         4.61
ExhaustiveBinary-256-itq (self)                        9_820.15    10_909.58    20_729.73       0.3927     0.490943         4.61
ExhaustiveBinary-512-random_no_rr (query)              2_465.03     1_535.14     4_000.18       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_465.03     1_564.39     4_029.42       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_465.03     1_672.85     4_137.88       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_465.03     1_722.46     4_187.50       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                     2_465.03    16_269.28    18_734.32       0.6185     0.184365         9.22
ExhaustiveBinary-512-itq_no_rr (query)                10_523.94     1_534.24    12_058.18       0.2230          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  10_523.94     1_573.10    12_097.03       0.4733     0.353636         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 10_523.94     1_619.13    12_143.07       0.6117     0.187548         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 10_523.94     1_726.05    12_249.99       0.7467     0.091104         9.22
ExhaustiveBinary-512-itq (self)                       10_523.94    16_204.71    26_728.65       0.6142     0.183153         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_641.56        79.62     2_721.18       0.1443          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_641.56        91.87     2_733.43       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_641.56       122.64     2_764.20       0.1371          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_641.56       102.47     2_744.03       0.3253     0.693095         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_641.56       126.46     2_768.02       0.4471     0.412064         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_641.56       174.99     2_816.54       0.5876     0.229141         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_641.56       116.20     2_757.76       0.3165     0.721918         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_641.56       144.96     2_786.52       0.4368     0.431054         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_641.56       193.56     2_835.12       0.5764     0.241196         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_641.56       151.83     2_793.39       0.3123     0.739525         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_641.56       185.54     2_827.10       0.4312     0.443202         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_641.56       236.69     2_878.25       0.5697     0.249327         5.79
IVF-Binary-256-nl273-random (self)                     2_641.56     1_448.47     4_090.03       0.4422     0.416770         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           3_182.73        79.47     3_262.20       0.1403          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           3_182.73       103.12     3_285.85       0.1380          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           3_182.73       104.78     3_287.51       0.3203     0.707597         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          3_182.73       130.32     3_313.05       0.4414     0.420873         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          3_182.73       180.43     3_363.16       0.5825     0.233592         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           3_182.73       127.94     3_310.67       0.3145     0.730872         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          3_182.73       164.39     3_347.12       0.4334     0.437743         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          3_182.73       217.13     3_399.86       0.5726     0.245415         5.80
IVF-Binary-256-nl387-random (self)                     3_182.73     1_302.33     4_485.06       0.4467     0.406727         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           3_949.49        75.29     4_024.78       0.1414          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           3_949.49        84.86     4_034.35       0.1400          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           3_949.49        93.07     4_042.56       0.1386          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           3_949.49       100.04     4_049.53       0.3225     0.696837         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          3_949.49       127.21     4_076.70       0.4444     0.412753         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          3_949.49       172.08     4_121.57       0.5867     0.227736         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           3_949.49       107.67     4_057.16       0.3186     0.711788         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          3_949.49       136.02     4_085.51       0.4391     0.423820         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          3_949.49       178.22     4_127.71       0.5799     0.235998         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           3_949.49       125.54     4_075.03       0.3149     0.727205         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          3_949.49       146.67     4_096.16       0.4341     0.435352         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          3_949.49       200.57     4_150.06       0.5737     0.244012         5.82
IVF-Binary-256-nl547-random (self)                     3_949.49     1_245.36     5_194.85       0.4498     0.399539         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)             10_804.73        77.16    10_881.89       0.1299          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             10_804.73        88.16    10_892.89       0.1239          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             10_804.73       120.04    10_924.77       0.1220          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             10_804.73       101.68    10_906.41       0.2980     0.760619         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            10_804.73       131.49    10_936.22       0.4143     0.455528         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            10_804.73       172.80    10_977.53       0.5536     0.256387         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             10_804.73       117.03    10_921.76       0.2866     0.799417         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            10_804.73       140.77    10_945.50       0.4006     0.481417         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            10_804.73       189.19    10_993.92       0.5395     0.272100         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             10_804.73       147.21    10_951.94       0.2825     0.819698         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            10_804.73       177.96    10_982.69       0.3941     0.496936         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            10_804.73       230.48    11_035.21       0.5322     0.282016         5.79
IVF-Binary-256-nl273-itq (self)                       10_804.73     1_400.83    12_205.56       0.4069     0.467064         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             11_322.51        78.59    11_401.10       0.1259          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             11_322.51       109.86    11_432.37       0.1230          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             11_322.51       104.17    11_426.68       0.2906     0.783290         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            11_322.51       128.14    11_450.64       0.4060     0.469751         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            11_322.51       178.91    11_501.42       0.5456     0.264013         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             11_322.51       128.40    11_450.91       0.2835     0.815126         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            11_322.51       156.99    11_479.50       0.3962     0.492780         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            11_322.51       208.17    11_530.68       0.5336     0.279398         5.80
IVF-Binary-256-nl387-itq (self)                       11_322.51     1_274.16    12_596.67       0.4117     0.456745         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_971.14        74.74    12_045.88       0.1261          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_971.14        83.10    12_054.24       0.1240          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_971.14        94.17    12_065.30       0.1224          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_971.14       103.79    12_074.93       0.2929     0.771193         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_971.14       124.46    12_095.60       0.4098     0.459379         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_971.14       167.50    12_138.64       0.5504     0.256678         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_971.14       106.01    12_077.15       0.2884     0.791809         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_971.14       130.50    12_101.64       0.4034     0.473372         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_971.14       174.49    12_145.63       0.5420     0.266932         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_971.14       122.99    12_094.13       0.2839     0.809761         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_971.14       143.43    12_114.57       0.3981     0.485766         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_971.14       191.27    12_162.41       0.5353     0.275106         5.82
IVF-Binary-256-nl547-itq (self)                       11_971.14     1_222.37    13_193.51       0.4155     0.447615         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_805.66       137.13     3_942.79       0.2304          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_805.66       162.91     3_968.56       0.2278          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_805.66       217.62     4_023.27       0.2270          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_805.66       174.86     3_980.51       0.4860     0.337760        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_805.66       204.94     4_010.60       0.6273     0.176939        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_805.66       275.75     4_081.41       0.7652     0.083695        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_805.66       216.75     4_022.41       0.4812     0.344924        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_805.66       233.71     4_039.37       0.6224     0.181309        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_805.66       295.00     4_100.66       0.7602     0.086530        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_805.66       257.33     4_062.98       0.4791     0.349101        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_805.66       295.44     4_101.10       0.6194     0.184459        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_805.66       361.72     4_167.38       0.7568     0.088673        10.40
IVF-Binary-512-nl273-random (self)                     3_805.66     2_337.87     6_143.53       0.6260     0.177547        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           4_375.40       142.99     4_518.40       0.2289          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           4_375.40       186.27     4_561.67       0.2274          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           4_375.40       186.55     4_561.95       0.4839     0.341409        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          4_375.40       212.88     4_588.28       0.6254     0.178808        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          4_375.40       307.08     4_682.48       0.7626     0.084991        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           4_375.40       228.63     4_604.03       0.4809     0.347106        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          4_375.40       303.20     4_678.61       0.6210     0.183001        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          4_375.40       324.55     4_699.96       0.7578     0.087878        10.41
IVF-Binary-512-nl387-random (self)                     4_375.40     2_126.97     6_502.37       0.6282     0.175444        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           5_150.00       129.35     5_279.35       0.2295          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           5_150.00       141.47     5_291.47       0.2286          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           5_150.00       174.59     5_324.59       0.2277          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           5_150.00       165.08     5_315.08       0.4844     0.340121        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          5_150.00       205.36     5_355.35       0.6262     0.177882        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          5_150.00       252.83     5_402.83       0.7650     0.083906        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           5_150.00       179.84     5_329.84       0.4822     0.344427        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          5_150.00       214.58     5_364.58       0.6234     0.180818        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          5_150.00       272.43     5_422.42       0.7614     0.086088        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           5_150.00       203.92     5_353.92       0.4802     0.348092        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          5_150.00       243.25     5_393.25       0.6210     0.183289        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          5_150.00       299.68     5_449.68       0.7585     0.087937        10.43
IVF-Binary-512-nl547-random (self)                     5_150.00     1_984.35     7_134.35       0.6297     0.174184        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_917.85       135.38    12_053.23       0.2291          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_917.85       158.37    12_076.21       0.2262          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_917.85       215.52    12_133.37       0.2253          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_917.85       170.33    12_088.18       0.4843     0.336509        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_917.85       204.04    12_121.88       0.6241     0.176585        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_917.85       263.38    12_181.23       0.7591     0.084287        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_917.85       195.41    12_113.25       0.4795     0.344009        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_917.85       232.02    12_149.87       0.6188     0.181286        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_917.85       303.15    12_221.00       0.7539     0.087383        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_917.85       256.70    12_174.55       0.4771     0.348313        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_917.85       295.60    12_213.45       0.6158     0.184229        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_917.85       359.30    12_277.15       0.7506     0.089206        10.40
IVF-Binary-512-nl273-itq (self)                       11_917.85     2_334.42    14_252.27       0.6214     0.177186        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             12_546.61       149.03    12_695.64       0.2270          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             12_546.61       182.45    12_729.06       0.2254          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             12_546.61       190.09    12_736.70       0.4814     0.341313        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            12_546.61       218.87    12_765.48       0.6208     0.179148        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            12_546.61       267.59    12_814.21       0.7560     0.085923        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             12_546.61       226.33    12_772.95       0.4779     0.347472        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            12_546.61       258.32    12_804.93       0.6167     0.183253        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            12_546.61       318.93    12_865.54       0.7515     0.088756        10.41
IVF-Binary-512-nl387-itq (self)                       12_546.61     2_129.71    14_676.32       0.6235     0.175207        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             13_313.93       130.35    13_444.28       0.2275          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             13_313.93       151.70    13_465.63       0.2265          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             13_313.93       165.41    13_479.34       0.2256          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             13_313.93       192.51    13_506.44       0.4828     0.338669        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            13_313.93       198.12    13_512.05       0.6225     0.177185        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            13_313.93       259.31    13_573.24       0.7591     0.084005        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             13_313.93       182.15    13_496.08       0.4801     0.343207        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            13_313.93       216.69    13_530.62       0.6191     0.180333        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            13_313.93       276.42    13_590.35       0.7553     0.086346        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             13_313.93       203.40    13_517.33       0.4781     0.346763        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            13_313.93       237.92    13_551.85       0.6167     0.182639        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            13_313.93       298.30    13_612.23       0.7527     0.087851        10.43
IVF-Binary-512-nl547-itq (self)                       13_313.93     2_000.33    15_314.26       0.6257     0.173019        10.43
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
Exhaustive (query)                                         3.07     2_424.58     2_427.65       1.0000     0.000000        18.31
Exhaustive (self)                                          3.07    23_995.82    23_998.89       1.0000     0.000000        18.31
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              1_253.05     1_018.70     2_271.75       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)                1_253.05     1_058.61     2_311.66       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)               1_253.05     1_087.82     2_340.86       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)               1_253.05     1_182.19     2_435.24       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                     1_253.05    11_158.52    12_411.56       0.3238     3.208083         4.61
ExhaustiveBinary-256-itq_no_rr (query)                12_225.78       978.33    13_204.11       0.0734          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)                  12_225.78     1_028.57    13_254.35       0.1955     5.709191         4.61
ExhaustiveBinary-256-itq-rf10 (query)                 12_225.78     1_080.35    13_306.12       0.2902     3.600771         4.61
ExhaustiveBinary-256-itq-rf20 (query)                 12_225.78     1_562.87    13_788.64       0.4157     2.167224         4.61
ExhaustiveBinary-256-itq (self)                       12_225.78    11_185.99    23_411.77       0.2960     3.485945         4.61
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              2_529.37     1_484.66     4_014.02       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)                2_529.37     1_533.47     4_062.84       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)               2_529.37     1_570.96     4_100.33       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)               2_529.37     1_670.30     4_199.67       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                     2_529.37    15_739.58    18_268.94       0.4788     1.710526         9.22
ExhaustiveBinary-512-itq_no_rr (query)                15_400.94     1_483.62    16_884.56       0.1518          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)                  15_400.94     1_519.48    16_920.42       0.3364     3.092957         9.22
ExhaustiveBinary-512-itq-rf10 (query)                 15_400.94     1_580.49    16_981.43       0.4608     1.824577         9.22
ExhaustiveBinary-512-itq-rf20 (query)                 15_400.94     1_679.07    17_080.01       0.6062     1.008706         9.22
ExhaustiveBinary-512-itq (self)                       15_400.94    15_716.55    31_117.49       0.4642     1.796793         9.22
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)           2_606.83        74.05     2_680.88       0.1015          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)           2_606.83        82.74     2_689.57       0.0926          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)           2_606.83       106.96     2_713.80       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)           2_606.83        93.73     2_700.56       0.2466     4.682597         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)          2_606.83       115.70     2_722.54       0.3527     2.916537         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)          2_606.83       156.41     2_763.25       0.4886     1.698168         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)           2_606.83       105.68     2_712.51       0.2318     5.166092         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)          2_606.83       129.85     2_736.69       0.3354     3.190525         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)          2_606.83       172.18     2_779.01       0.4700     1.862358         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)           2_606.83       131.66     2_738.49       0.2280     5.296145         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)          2_606.83       157.55     2_764.38       0.3304     3.274036         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)          2_606.83       202.69     2_809.53       0.4638     1.915926         5.79
IVF-Binary-256-nl273-random (self)                     2_606.83     1_301.96     3_908.79       0.3387     3.130112         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)           3_050.28        77.76     3_128.04       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)           3_050.28        96.22     3_146.50       0.0919          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)           3_050.28        97.14     3_147.42       0.2358     4.931507         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)          3_050.28       118.56     3_168.84       0.3421     3.052683         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)          3_050.28       158.78     3_209.05       0.4777     1.802324         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)           3_050.28       118.64     3_168.92       0.2299     5.156744         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)          3_050.28       142.26     3_192.54       0.3340     3.196200         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)          3_050.28       187.04     3_237.31       0.4679     1.888188         5.80
IVF-Binary-256-nl387-random (self)                     3_050.28     1_184.34     4_234.62       0.3448     3.003189         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)           3_809.85        75.17     3_885.02       0.0954          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)           3_809.85        79.78     3_889.63       0.0934          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)           3_809.85        90.85     3_900.69       0.0917          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)           3_809.85        95.09     3_904.93       0.2397     4.792297         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)          3_809.85       115.65     3_925.50       0.3460     2.991253         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)          3_809.85       155.76     3_965.60       0.4824     1.757604         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)           3_809.85       101.84     3_911.69       0.2354     4.927078         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)          3_809.85       124.21     3_934.06       0.3399     3.085620         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)          3_809.85       163.56     3_973.41       0.4749     1.823326         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)           3_809.85       115.67     3_925.52       0.2310     5.077708         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)          3_809.85       137.53     3_947.38       0.3339     3.187174         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)          3_809.85       178.27     3_988.12       0.4678     1.889622         5.82
IVF-Binary-256-nl547-random (self)                     3_809.85     1_151.29     4_961.14       0.3483     2.939409         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)             10_544.27        71.77    10_616.04       0.0872          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)             10_544.27        83.37    10_627.65       0.0772          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)             10_544.27       106.69    10_650.97       0.0756          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)             10_544.27        94.11    10_638.38       0.2216     5.283134         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)            10_544.27       114.73    10_659.01       0.3219     3.306410         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)            10_544.27       153.80    10_698.08       0.4512     1.955544         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)             10_544.27       110.12    10_654.39       0.2066     5.680359         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)            10_544.27       129.97    10_674.24       0.3048     3.546397         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)            10_544.27       171.95    10_716.22       0.4341     2.093396         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)             10_544.27       130.33    10_674.61       0.2032     5.812621         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)            10_544.27       156.68    10_700.96       0.3004     3.633989         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)            10_544.27       202.89    10_747.17       0.4284     2.148126         5.79
IVF-Binary-256-nl273-itq (self)                       10_544.27     1_287.69    11_831.96       0.3111     3.431917         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)             10_842.00        75.57    10_917.57       0.0798          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)             10_842.00        96.21    10_938.21       0.0767          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)             10_842.00        96.96    10_938.96       0.2117     5.502605         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)            10_842.00       118.84    10_960.84       0.3111     3.415891         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)            10_842.00       172.27    11_014.27       0.4426     2.008915         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)             10_842.00       121.19    10_963.19       0.2050     5.728006         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)            10_842.00       142.12    10_984.12       0.3029     3.552823         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)            10_842.00       183.47    11_025.47       0.4328     2.095192         5.80
IVF-Binary-256-nl387-itq (self)                       10_842.00     1_181.53    12_023.54       0.3167     3.315819         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)             11_558.09        72.55    11_630.64       0.0812          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)             11_558.09        78.88    11_636.97       0.0789          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)             11_558.09        89.98    11_648.07       0.0767          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)             11_558.09        93.35    11_651.44       0.2157     5.312823         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)            11_558.09       112.71    11_670.80       0.3164     3.311091         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)            11_558.09       151.20    11_709.29       0.4485     1.952288         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)             11_558.09        99.83    11_657.92       0.2106     5.496314         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)            11_558.09       121.67    11_679.76       0.3094     3.429665         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)            11_558.09       160.31    11_718.40       0.4403     2.025056         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)             11_558.09       111.52    11_669.61       0.2060     5.642876         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)            11_558.09       133.43    11_691.53       0.3036     3.523895         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)            11_558.09       173.89    11_731.99       0.4331     2.085692         5.82
IVF-Binary-256-nl547-itq (self)                       11_558.09     1_133.15    12_691.24       0.3219     3.220849         5.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)           3_750.32       157.62     3_907.94       0.1685          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)           3_750.32       179.25     3_929.57       0.1642          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)           3_750.32       235.53     3_985.85       0.1629          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)           3_750.32       182.68     3_933.00       0.3648     2.762852        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)          3_750.32       211.56     3_961.88       0.4946     1.613586        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)          3_750.32       261.80     4_012.12       0.6428     0.865216        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)           3_750.32       208.59     3_958.91       0.3574     2.870953        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)          3_750.32       246.44     3_996.76       0.4866     1.678608        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)          3_750.32       294.01     4_044.32       0.6348     0.903671        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)           3_750.32       268.13     4_018.45       0.3551     2.903769        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)          3_750.32       300.88     4_051.20       0.4839     1.699072        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)          3_750.32       363.09     4_113.41       0.6317     0.917169        10.40
IVF-Binary-512-nl273-random (self)                     3_750.32     2_408.53     6_158.85       0.4885     1.655177        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)           4_269.39       161.68     4_431.07       0.1647          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)           4_269.39       208.97     4_478.36       0.1631          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)           4_269.39       189.37     4_458.76       0.3603     2.826319        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)          4_269.39       220.77     4_490.16       0.4905     1.642435        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)          4_269.39       269.90     4_539.29       0.6385     0.885603        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)           4_269.39       238.69     4_508.08       0.3567     2.886554        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)          4_269.39       272.20     4_541.59       0.4856     1.679814        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)          4_269.39       323.37     4_592.76       0.6332     0.909694        10.41
IVF-Binary-512-nl387-random (self)                     4_269.39     2_195.29     6_464.68       0.4917     1.624170        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)           5_027.10       152.13     5_179.23       0.1660          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)           5_027.10       167.23     5_194.33       0.1648          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)           5_027.10       193.17     5_220.27       0.1637          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)           5_027.10       178.01     5_205.11       0.3623     2.788789        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)          5_027.10       207.55     5_234.65       0.4927     1.618030        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)          5_027.10       252.40     5_279.50       0.6419     0.867389        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)           5_027.10       194.50     5_221.60       0.3595     2.836677        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)          5_027.10       224.04     5_251.14       0.4890     1.650040        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)          5_027.10       273.07     5_300.17       0.6371     0.891042        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)           5_027.10       224.06     5_251.16       0.3568     2.876216        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)          5_027.10       251.40     5_278.50       0.4854     1.678082        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)          5_027.10       341.00     5_368.10       0.6333     0.908544        10.43
IVF-Binary-512-nl547-random (self)                     5_027.10     2_071.41     7_098.52       0.4940     1.602597        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)             11_562.14       154.02    11_716.16       0.1599          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)             11_562.14       178.09    11_740.23       0.1554          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)             11_562.14       234.19    11_796.33       0.1545          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)             11_562.14       181.93    11_744.07       0.3516     2.904603        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)            11_562.14       213.44    11_775.57       0.4793     1.699096        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)            11_562.14       258.99    11_821.13       0.6251     0.933238        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)             11_562.14       207.61    11_769.75       0.3443     3.008004        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)            11_562.14       240.10    11_802.24       0.4709     1.762575        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)            11_562.14       291.61    11_853.75       0.6173     0.967726        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)             11_562.14       271.90    11_834.04       0.3421     3.040560        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)            11_562.14       316.54    11_878.68       0.4680     1.783032        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)            11_562.14       363.19    11_925.32       0.6141     0.981597        10.40
IVF-Binary-512-nl273-itq (self)                       11_562.14     2_383.21    13_945.35       0.4743     1.736637        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)             11_995.77       161.22    12_156.98       0.1565          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)             11_995.77       208.61    12_204.37       0.1547          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)             11_995.77       189.61    12_185.37       0.3474     2.963429        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)            11_995.77       219.59    12_215.36       0.4744     1.734850        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)            11_995.77       268.44    12_264.21       0.6202     0.953625        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)             11_995.77       238.83    12_234.60       0.3441     3.017131        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)            11_995.77       270.80    12_266.57       0.4700     1.772576        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)            11_995.77       323.77    12_319.54       0.6155     0.976781        10.41
IVF-Binary-512-nl387-itq (self)                       11_995.77     2_207.31    14_203.08       0.4776     1.709690        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)             12_814.39       151.42    12_965.80       0.1576          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)             12_814.39       166.57    12_980.96       0.1562          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)             12_814.39       191.70    13_006.09       0.1550          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)             12_814.39       178.29    12_992.68       0.3490     2.929320        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)            12_814.39       206.05    13_020.43       0.4769     1.713367        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)            12_814.39       252.21    13_066.60       0.6237     0.934473        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)             12_814.39       194.64    13_009.02       0.3460     2.980738        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)            12_814.39       226.16    13_040.55       0.4726     1.751190        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)            12_814.39       272.14    13_086.52       0.6192     0.956581        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)             12_814.39       222.37    13_036.76       0.3438     3.015335        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)            12_814.39       251.05    13_065.44       0.4696     1.775805        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)            12_814.39       301.58    13_115.97       0.6157     0.972254        10.43
IVF-Binary-512-nl547-itq (self)                       12_814.39     2_060.84    14_875.23       0.4800     1.688462        10.43
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
Exhaustive (query)                                        14.47    16_843.53    16_858.00       1.0000     0.000000        73.24
Exhaustive (self)                                         14.47   168_614.93   168_629.39       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_891.85       992.86     4_884.71       0.1084          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_891.85     1_044.93     4_936.77       0.1994    32.157948         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_891.85     1_105.75     4_997.60       0.2740    21.742093         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_891.85     1_233.24     5_125.09       0.3781    14.028739         4.70
ExhaustiveBinary-256-random (self)                     3_891.85    11_039.29    14_931.14       0.2729    21.873275         4.70
ExhaustiveBinary-256-itq_no_rr (query)                18_233.02       992.61    19_225.62       0.0945          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  18_233.02     1_042.44    19_275.46       0.1625    37.472052         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 18_233.02     1_099.52    19_332.54       0.2221    26.042140         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 18_233.02     1_219.15    19_452.16       0.3085    17.458979         4.70
ExhaustiveBinary-256-itq (self)                       18_233.02    11_013.22    29_246.24       0.2220    26.221286         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_667.67     1_534.58     9_202.25       0.1437          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_667.67     1_589.23     9_256.90       0.2864    21.279597         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_667.67     1_663.86     9_331.52       0.3868    13.379880         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_667.67     1_818.70     9_486.37       0.5137     7.900976         9.41
ExhaustiveBinary-512-random (self)                     7_667.67    16_477.87    24_145.54       0.3866    13.445005         9.41
ExhaustiveBinary-512-itq_no_rr (query)                21_544.68     1_544.08    23_088.76       0.1367          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  21_544.68     1_580.86    23_125.54       0.2673    22.229428         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 21_544.68     1_644.32    23_189.00       0.3636    14.093214         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 21_544.68     1_787.31    23_331.99       0.4852     8.407298         9.41
ExhaustiveBinary-512-itq (self)                       21_544.68    16_507.04    38_051.72       0.3631    14.195195         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_234.09     2_308.17    17_542.26       0.2051          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_234.09     2_358.68    17_592.77       0.4198    11.866665        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_234.09     2_436.72    17_670.81       0.5478     6.608533        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_234.09     2_570.02    17_804.11       0.6844     3.360088        18.81
ExhaustiveBinary-1024-random (self)                   15_234.09    25_548.07    40_782.16       0.5467     6.665345        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               34_933.78     2_340.98    37_274.77       0.2061          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 34_933.78     2_371.36    37_305.15       0.4209    11.394439        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                34_933.78     2_466.10    37_399.89       0.5491     6.304691        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                34_933.78     2_579.93    37_513.72       0.6837     3.180706        18.81
ExhaustiveBinary-1024-itq (self)                      34_933.78    24_710.11    59_643.89       0.5479     6.374041        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)          13_427.57       139.94    13_567.51       0.1097          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)          13_427.57       145.81    13_573.38       0.1090          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)          13_427.57       172.80    13_600.37       0.1085          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)          13_427.57       179.87    13_607.44       0.2030    31.743771         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)         13_427.57       226.39    13_653.96       0.2784    21.447327         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)         13_427.57       313.61    13_741.18       0.3832    13.844052         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)          13_427.57       192.50    13_620.08       0.2018    31.833632         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)         13_427.57       240.13    13_667.71       0.2772    21.496638         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)         13_427.57       332.23    13_759.80       0.3823    13.865423         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)          13_427.57       218.37    13_645.94       0.2002    32.027661         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)         13_427.57       267.21    13_694.78       0.2755    21.639995         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)         13_427.57       367.14    13_794.71       0.3797    13.964629         5.98
IVF-Binary-256-nl273-random (self)                    13_427.57     2_381.58    15_809.15       0.2761    21.638410         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)          17_580.33       164.22    17_744.55       0.1094          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)          17_580.33       191.31    17_771.64       0.1088          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)          17_580.33       202.83    17_783.16       0.2030    31.686432         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)         17_580.33       251.20    17_831.53       0.2785    21.409301         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)         17_580.33       330.37    17_910.70       0.3834    13.804727         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)          17_580.33       213.92    17_794.25       0.2008    31.973757         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)         17_580.33       262.04    17_842.37       0.2760    21.609178         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)         17_580.33       378.15    17_958.48       0.3806    13.918561         6.04
IVF-Binary-256-nl387-random (self)                    17_580.33     2_413.76    19_994.10       0.2773    21.532545         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)          23_174.05       158.70    23_332.75       0.1100          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)          23_174.05       166.22    23_340.27       0.1095          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)          23_174.05       179.07    23_353.12       0.1092          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)          23_174.05       210.81    23_384.86       0.2038    31.581264         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)         23_174.05       266.19    23_440.24       0.2798    21.302956         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)         23_174.05       335.50    23_509.56       0.3853    13.734412         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)          23_174.05       215.24    23_389.30       0.2029    31.732906         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)         23_174.05       265.92    23_439.98       0.2788    21.399255         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)         23_174.05       348.99    23_523.04       0.3842    13.782027         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)          23_174.05       236.14    23_410.20       0.2016    31.882203         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)         23_174.05       277.95    23_452.00       0.2771    21.523568         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)         23_174.05       366.23    23_540.28       0.3815    13.886903         6.12
IVF-Binary-256-nl547-random (self)                    23_174.05     2_553.14    25_727.19       0.2790    21.422311         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             34_977.64       143.35    35_120.99       0.0966          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             34_977.64       146.42    35_124.06       0.0959          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             34_977.64       172.16    35_149.79       0.0951          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             34_977.64       181.33    35_158.96       0.1664    36.759230         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            34_977.64       241.07    35_218.71       0.2275    25.481935         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            34_977.64       309.96    35_287.60       0.3161    17.034223         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             34_977.64       188.27    35_165.91       0.1651    36.994630         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            34_977.64       235.52    35_213.16       0.2257    25.664647         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            34_977.64       330.24    35_307.87       0.3138    17.150663         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             34_977.64       215.51    35_193.14       0.1637    37.280506         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            34_977.64       262.30    35_239.93       0.2235    25.902081         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            34_977.64       358.32    35_335.96       0.3104    17.347528         5.98
IVF-Binary-256-nl273-itq (self)                       34_977.64     2_334.85    37_312.48       0.2259    25.819028         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             31_468.51       147.75    31_616.25       0.0960          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             31_468.51       175.39    31_643.90       0.0953          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             31_468.51       192.39    31_660.90       0.1663    36.742577         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            31_468.51       239.60    31_708.11       0.2270    25.511118         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            31_468.51       324.88    31_793.39       0.3158    17.016395         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             31_468.51       225.92    31_694.43       0.1642    37.163672         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            31_468.51       264.51    31_733.01       0.2242    25.806826         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            31_468.51       355.45    31_823.96       0.3117    17.254061         6.04
IVF-Binary-256-nl387-itq (self)                       31_468.51     2_385.87    33_854.38       0.2273    25.650791         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             41_869.88       158.38    42_028.26       0.0967          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             41_869.88       166.46    42_036.34       0.0961          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             41_869.88       177.72    42_047.60       0.0956          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             41_869.88       202.62    42_072.49       0.1676    36.527492         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            41_869.88       249.67    42_119.55       0.2290    25.332950         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            41_869.88       339.69    42_209.57       0.3183    16.891285         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             41_869.88       210.38    42_080.26       0.1665    36.755341         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            41_869.88       257.35    42_127.22       0.2274    25.498502         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            41_869.88       344.05    42_213.92       0.3162    17.014368         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             41_869.88       220.63    42_090.51       0.1650    37.009645         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            41_869.88       267.55    42_137.43       0.2251    25.732539         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            41_869.88       359.31    42_229.18       0.3131    17.179985         6.12
IVF-Binary-256-nl547-itq (self)                       41_869.88     2_497.38    44_367.26       0.2293    25.462401         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          17_340.39       236.25    17_576.63       0.1447          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          17_340.39       259.05    17_599.43       0.1444          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          17_340.39       311.70    17_652.09       0.1441          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          17_340.39       311.36    17_651.74       0.2872    21.248375        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         17_340.39       359.27    17_699.66       0.3875    13.433080        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         17_340.39       450.69    17_791.08       0.5124     8.025175        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          17_340.39       334.04    17_674.43       0.2873    21.197474        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         17_340.39       401.54    17_741.93       0.3889    13.325453        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         17_340.39       486.47    17_826.86       0.5147     7.899483        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          17_340.39       398.29    17_738.68       0.2867    21.239673        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         17_340.39       453.52    17_793.91       0.3878    13.345985        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         17_340.39       568.55    17_908.94       0.5144     7.880219        10.69
IVF-Binary-512-nl273-random (self)                    17_340.39     3_966.42    21_306.81       0.3881    13.398134        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          21_617.65       255.62    21_873.27       0.1447          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          21_617.65       303.01    21_920.66       0.1443          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          21_617.65       326.37    21_944.02       0.2874    21.182772        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         21_617.65       382.05    21_999.69       0.3886    13.366817        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         21_617.65       474.09    22_091.74       0.5143     7.941023        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          21_617.65       383.39    22_001.04       0.2869    21.215303        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         21_617.65       432.30    22_049.95       0.3883    13.329017        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         21_617.65       540.17    22_157.82       0.5150     7.864684        10.74
IVF-Binary-512-nl387-random (self)                    21_617.65     3_764.29    25_381.94       0.3882    13.417563        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          27_050.36       264.96    27_315.32       0.1452          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          27_050.36       280.80    27_331.15       0.1449          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          27_050.36       305.76    27_356.12       0.1443          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          27_050.36       344.72    27_395.08       0.2884    21.139001        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         27_050.36       377.18    27_427.54       0.3893    13.347819        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         27_050.36       463.40    27_513.76       0.5152     7.947954        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          27_050.36       363.86    27_414.22       0.2885    21.130755        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         27_050.36       398.67    27_449.03       0.3899    13.285486        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         27_050.36       493.25    27_543.61       0.5166     7.855421        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          27_050.36       373.77    27_424.13       0.2878    21.170903        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         27_050.36       431.21    27_481.56       0.3891    13.290583        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         27_050.36       520.53    27_570.89       0.5161     7.840722        10.82
IVF-Binary-512-nl547-random (self)                    27_050.36     3_799.47    30_849.83       0.3891    13.391547        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             31_478.49       237.90    31_716.39       0.1378          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             31_478.49       261.79    31_740.28       0.1375          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             31_478.49       315.32    31_793.81       0.1371          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             31_478.49       306.24    31_784.73       0.2691    22.116362        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            31_478.49       362.82    31_841.31       0.3649    14.114416        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            31_478.49       452.94    31_931.43       0.4846     8.518373        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             31_478.49       334.57    31_813.06       0.2691    22.092687        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            31_478.49       392.83    31_871.32       0.3655    14.031377        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            31_478.49       487.45    31_965.94       0.4867     8.393051        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             31_478.49       390.31    31_868.80       0.2681    22.167268        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            31_478.49       457.06    31_935.55       0.3646    14.053989        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            31_478.49       558.24    32_036.73       0.4862     8.380192        10.69
IVF-Binary-512-nl273-itq (self)                       31_478.49     3_917.51    35_395.99       0.3649    14.125485        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             35_609.31       250.30    35_859.60       0.1377          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             35_609.31       292.17    35_901.47       0.1373          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             35_609.31       321.25    35_930.55       0.2694    22.074021        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            35_609.31       377.42    35_986.73       0.3654    14.055778        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            35_609.31       463.37    36_072.67       0.4859     8.447220        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             35_609.31       366.04    35_975.35       0.2688    22.112510        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            35_609.31       431.62    36_040.93       0.3649    14.033436        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            35_609.31       539.20    36_148.50       0.4868     8.366758        10.74
IVF-Binary-512-nl387-itq (self)                       35_609.31     3_769.77    39_379.07       0.3651    14.137686        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             41_192.18       258.67    41_450.85       0.1380          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             41_192.18       274.29    41_466.47       0.1378          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             41_192.18       299.16    41_491.34       0.1373          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             41_192.18       331.15    41_523.33       0.2705    21.996131        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            41_192.18       380.71    41_572.89       0.3667    14.005606        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            41_192.18       464.42    41_656.60       0.4870     8.443653        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             41_192.18       343.91    41_536.09       0.2701    22.004878        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            41_192.18       403.21    41_595.39       0.3668    13.979437        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            41_192.18       485.74    41_677.92       0.4880     8.361779        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             41_192.18       374.26    41_566.44       0.2692    22.076003        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            41_192.18       427.85    41_620.03       0.3659    13.999844        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            41_192.18       521.14    41_713.32       0.4877     8.341366        10.82
IVF-Binary-512-nl547-itq (self)                       41_192.18     3_776.19    44_968.37       0.3662    14.101167        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         24_885.24       583.68    25_468.92       0.2049          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         24_885.24       662.53    25_547.77       0.2054          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         24_885.24       833.26    25_718.50       0.2054          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         24_885.24       674.35    25_559.59       0.4160    12.111825        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        24_885.24       675.84    25_561.08       0.5399     6.939206        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        24_885.24       758.10    25_643.34       0.6699     3.761954        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         24_885.24       712.13    25_597.37       0.4193    11.917585        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        24_885.24       758.62    25_643.86       0.5460     6.695219        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        24_885.24       835.63    25_720.87       0.6799     3.482442        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         24_885.24       886.77    25_772.01       0.4202    11.846247        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        24_885.24       932.90    25_818.14       0.5483     6.600881        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        24_885.24     1_032.91    25_918.15       0.6849     3.352257        20.09
IVF-Binary-1024-nl273-random (self)                   24_885.24     7_546.05    32_431.29       0.5447     6.754891        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         28_997.81       607.78    29_605.58       0.2050          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         28_997.81       747.97    29_745.77       0.2054          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         28_997.81       658.38    29_656.19       0.4178    12.025323        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        28_997.81       700.96    29_698.76       0.5426     6.838950        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        28_997.81       773.27    29_771.07       0.6748     3.648652        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         28_997.81       806.13    29_803.93       0.4200    11.856041        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        28_997.81       853.01    29_850.81       0.5480     6.614650        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        28_997.81       925.86    29_923.67       0.6839     3.379144        20.15
IVF-Binary-1024-nl387-random (self)                   28_997.81     7_048.29    36_046.09       0.5424     6.876075        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         34_444.01       595.27    35_039.28       0.2055          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         34_444.01       630.06    35_074.07       0.2058          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         34_444.01       712.70    35_156.71       0.2056          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         34_444.01       634.61    35_078.62       0.4178    12.043258        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        34_444.01       678.03    35_122.04       0.5418     6.887299        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        34_444.01       779.30    35_223.31       0.6725     3.716269        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         34_444.01       693.52    35_137.53       0.4201    11.893644        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        34_444.01       726.37    35_170.38       0.5467     6.688950        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        34_444.01       805.68    35_249.69       0.6804     3.485819        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         34_444.01       765.77    35_209.78       0.4207    11.835178        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        34_444.01       815.33    35_259.34       0.5484     6.605357        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        34_444.01       895.91    35_339.92       0.6844     3.375293        20.23
IVF-Binary-1024-nl547-random (self)                   34_444.01     6_663.68    41_107.69       0.5413     6.921843        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            39_460.88       588.94    40_049.82       0.2059          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            39_460.88       660.26    40_121.14       0.2064          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            39_460.88       832.02    40_292.90       0.2063          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            39_460.88       632.55    40_093.44       0.4173    11.644655        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           39_460.88       674.01    40_134.89       0.5413     6.631262        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           39_460.88       762.73    40_223.61       0.6689     3.594260        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            39_460.88       711.48    40_172.36       0.4204    11.446032        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           39_460.88       759.88    40_220.77       0.5473     6.391702        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           39_460.88       839.68    40_300.56       0.6790     3.315706        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            39_460.88       894.06    40_354.94       0.4214    11.380923        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           39_460.88       939.75    40_400.63       0.5496     6.292590        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           39_460.88     1_021.90    40_482.79       0.6840     3.175916        20.09
IVF-Binary-1024-nl273-itq (self)                      39_460.88     7_551.95    47_012.84       0.5457     6.467422        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            43_557.23       606.58    44_163.80       0.2064          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            43_557.23       743.87    44_301.09       0.2066          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            43_557.23       665.00    44_222.23       0.4191    11.541817        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           43_557.23       701.89    44_259.11       0.5446     6.518400        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           43_557.23       773.77    44_330.99       0.6741     3.483272        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            43_557.23       801.99    44_359.22       0.4216    11.373955        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           43_557.23       873.08    44_430.30       0.5494     6.305025        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           43_557.23       934.95    44_492.17       0.6837     3.195305        20.15
IVF-Binary-1024-nl387-itq (self)                      43_557.23     6_935.42    50_492.64       0.5431     6.596732        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            49_969.74       575.88    50_545.62       0.2064          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            49_969.74       634.19    50_603.93       0.2065          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            49_969.74       718.90    50_688.64       0.2065          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            49_969.74       626.72    50_596.46       0.4190    11.574350        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           49_969.74       673.95    50_643.69       0.5431     6.579860        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           49_969.74       746.02    50_715.76       0.6715     3.550569        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            49_969.74       685.40    50_655.13       0.4209    11.443855        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           49_969.74       724.82    50_694.56       0.5477     6.396043        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           49_969.74       809.31    50_779.05       0.6793     3.321845        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            49_969.74       762.92    50_732.65       0.4217    11.373315        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           49_969.74       819.31    50_789.05       0.5497     6.304111        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           49_969.74       889.22    50_858.95       0.6837     3.200391        20.23
IVF-Binary-1024-nl547-itq (self)                      49_969.74     6_668.15    56_637.88       0.5422     6.639879        20.23
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
Exhaustive (query)                                        13.99    16_897.52    16_911.51       1.0000     0.000000        73.24
Exhaustive (self)                                         13.99   171_287.26   171_301.25       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_899.92     1_002.19     4_902.11       0.0835          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_899.92     1_046.13     4_946.05       0.1599    21.111788         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_899.92     1_115.50     5_015.42       0.2242    14.518074         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_899.92     1_246.00     5_145.92       0.3173     9.567601         4.70
ExhaustiveBinary-256-random (self)                     3_899.92    11_003.08    14_902.99       0.2257    14.498282         4.70
ExhaustiveBinary-256-itq_no_rr (query)                26_073.43     1_010.18    27_083.61       0.0522          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  26_073.43     1_051.44    27_124.87       0.1084    27.956034         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 26_073.43     1_094.41    27_167.83       0.1563    19.790083         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 26_073.43     1_222.04    27_295.47       0.2301    13.538437         4.70
ExhaustiveBinary-256-itq (self)                       26_073.43    11_028.70    37_102.12       0.1607    19.585886         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_700.41     1_555.32     9_255.73       0.1247          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_700.41     1_600.17     9_300.58       0.2451    13.652649         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_700.41     1_654.72     9_355.14       0.3362     8.805138         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_700.41     1_791.03     9_491.45       0.4554     5.389016         9.41
ExhaustiveBinary-512-random (self)                     7_700.41    16_575.53    24_275.95       0.3357     8.840667         9.41
ExhaustiveBinary-512-itq_no_rr (query)                22_208.71     1_549.95    23_758.66       0.1097          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  22_208.71     1_589.83    23_798.54       0.2108    16.038374         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 22_208.71     1_667.04    23_875.75       0.2904    10.670775         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 22_208.71     1_768.62    23_977.33       0.3995     6.762765         9.41
ExhaustiveBinary-512-itq (self)                       22_208.71    16_719.98    38_928.69       0.2902    10.669535         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_280.98     2_333.79    17_614.77       0.1760          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_280.98     2_383.85    17_664.83       0.3642     7.961838        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_280.98     2_449.77    17_730.75       0.4844     4.696347        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_280.98     2_582.07    17_863.05       0.6214     2.539505        18.81
ExhaustiveBinary-1024-random (self)                   15_280.98    24_627.67    39_908.65       0.4849     4.684572        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               30_096.89     2_329.16    32_426.05       0.1669          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 30_096.89     2_393.15    32_490.04       0.3450     8.646287        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                30_096.89     2_453.29    32_550.18       0.4610     5.152251        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                30_096.89     2_617.13    32_714.03       0.5965     2.860810        18.81
ExhaustiveBinary-1024-itq (self)                      30_096.89    24_528.14    54_625.03       0.4605     5.175618        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)          13_557.19       135.02    13_692.21       0.0903          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)          13_557.19       145.12    13_702.31       0.0879          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)          13_557.19       173.37    13_730.56       0.0859          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)          13_557.19       186.74    13_743.93       0.1734    19.725002         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)         13_557.19       231.04    13_788.23       0.2421    13.455504         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)         13_557.19       314.31    13_871.50       0.3420     8.766258         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)          13_557.19       184.92    13_742.11       0.1691    20.139718         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)         13_557.19       232.66    13_789.85       0.2363    13.781701         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)         13_557.19       328.36    13_885.55       0.3339     9.022980         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)          13_557.19       222.48    13_779.67       0.1644    20.704767         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)         13_557.19       260.56    13_817.75       0.2299    14.223111         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)         13_557.19       352.98    13_910.17       0.3248     9.348626         5.98
IVF-Binary-256-nl273-random (self)                    13_557.19     2_299.79    15_856.98       0.2375    13.776331         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)          17_559.02       154.25    17_713.26       0.0888          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)          17_559.02       170.56    17_729.58       0.0866          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)          17_559.02       192.00    17_751.02       0.1716    19.890437         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)         17_559.02       238.84    17_797.85       0.2400    13.569855         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)         17_559.02       322.06    17_881.08       0.3391     8.853607         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)          17_559.02       211.61    17_770.63       0.1663    20.504317         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)         17_559.02       268.42    17_827.43       0.2322    14.064814         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)         17_559.02       359.33    17_918.35       0.3270     9.257133         6.04
IVF-Binary-256-nl387-random (self)                    17_559.02     2_347.89    19_906.90       0.2411    13.572676         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)          23_073.90       157.56    23_231.45       0.0903          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)          23_073.90       167.92    23_241.82       0.0889          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)          23_073.90       179.98    23_253.88       0.0874          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)          23_073.90       199.24    23_273.14       0.1745    19.563695         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)         23_073.90       244.96    23_318.86       0.2447    13.314886         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)         23_073.90       324.01    23_397.91       0.3464     8.635701         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)          23_073.90       204.22    23_278.12       0.1710    19.942114         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)         23_073.90       251.88    23_325.77       0.2396    13.618292         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)         23_073.90       344.83    23_418.73       0.3386     8.893535         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)          23_073.90       216.71    23_290.61       0.1672    20.379981         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)         23_073.90       261.54    23_335.43       0.2338    13.974265         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)         23_073.90       356.40    23_430.30       0.3303     9.170442         6.12
IVF-Binary-256-nl547-random (self)                    23_073.90     2_444.49    25_518.39       0.2459    13.303660         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             31_341.40       132.94    31_474.34       0.0615          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             31_341.40       144.78    31_486.17       0.0572          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             31_341.40       174.71    31_516.11       0.0546          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             31_341.40       171.35    31_512.74       0.1266    25.426539         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            31_341.40       215.79    31_557.19       0.1816    17.802360         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            31_341.40       302.32    31_643.72       0.2647    11.985532         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             31_341.40       180.81    31_522.20       0.1193    26.407393         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            31_341.40       223.17    31_564.56       0.1716    18.579437         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            31_341.40       314.89    31_656.28       0.2519    12.547047         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             31_341.40       209.99    31_551.39       0.1140    27.404944         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            31_341.40       263.08    31_604.48       0.1638    19.397621         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            31_341.40       340.32    31_681.71       0.2401    13.218752         5.98
IVF-Binary-256-nl273-itq (self)                       31_341.40     2_258.97    33_600.37       0.1756    18.400737         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             32_023.95       152.69    32_176.63       0.0579          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             32_023.95       174.51    32_198.46       0.0555          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             32_023.95       197.19    32_221.14       0.1215    25.940842         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            32_023.95       236.79    32_260.73       0.1751    18.197586         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            32_023.95       314.58    32_338.53       0.2582    12.227120         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             32_023.95       214.96    32_238.91       0.1155    26.961915         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            32_023.95       258.26    32_282.21       0.1662    19.018492         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            32_023.95       346.86    32_370.81       0.2452    12.862590         6.04
IVF-Binary-256-nl387-itq (self)                       32_023.95     2_343.45    34_367.40       0.1795    18.012376         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             37_373.77       163.08    37_536.84       0.0598          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             37_373.77       169.57    37_543.34       0.0581          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             37_373.77       181.00    37_554.77       0.0563          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             37_373.77       201.23    37_575.00       0.1254    25.283926         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            37_373.77       244.27    37_618.04       0.1806    17.653699         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            37_373.77       319.13    37_692.89       0.2677    11.758868         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             37_373.77       206.92    37_580.68       0.1214    25.935991         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            37_373.77       253.24    37_627.01       0.1748    18.162552         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            37_373.77       331.52    37_705.29       0.2583    12.192179         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             37_373.77       219.12    37_592.89       0.1173    26.667182         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            37_373.77       263.37    37_637.14       0.1687    18.758287         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            37_373.77       346.35    37_720.11       0.2485    12.673975         6.12
IVF-Binary-256-nl547-itq (self)                       37_373.77     2_416.43    39_790.19       0.1855    17.465350         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          17_220.24       240.62    17_460.86       0.1288          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          17_220.24       264.97    17_485.21       0.1273          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          17_220.24       321.02    17_541.26       0.1262          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          17_220.24       304.99    17_525.23       0.2555    13.060243        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         17_220.24       355.05    17_575.29       0.3499     8.371400        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         17_220.24       446.89    17_667.13       0.4731     5.066306        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          17_220.24       331.79    17_552.03       0.2519    13.261266        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         17_220.24       397.84    17_618.09       0.3445     8.536596        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         17_220.24       489.41    17_709.66       0.4662     5.188372        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          17_220.24       419.63    17_639.88       0.2483    13.488315        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         17_220.24       481.21    17_701.45       0.3396     8.707879        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         17_220.24       555.10    17_775.34       0.4595     5.318688        10.69
IVF-Binary-512-nl273-random (self)                    17_220.24     3_851.05    21_071.29       0.3446     8.552099        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          21_366.22       261.45    21_627.67       0.1281          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          21_366.22       307.87    21_674.09       0.1262          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          21_366.22       328.76    21_694.98       0.2547    13.096200        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         21_366.22       377.54    21_743.75       0.3488     8.395253        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         21_366.22       482.01    21_848.22       0.4720     5.083655        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          21_366.22       389.68    21_755.89       0.2498    13.398130        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         21_366.22       439.36    21_805.58       0.3414     8.644376        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         21_366.22       527.89    21_894.11       0.4617     5.276687        10.74
IVF-Binary-512-nl387-random (self)                    21_366.22     3_750.04    25_116.26       0.3485     8.424860        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          26_916.78       291.05    27_207.83       0.1292          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          26_916.78       279.27    27_196.05       0.1281          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          26_916.78       304.96    27_221.74       0.1268          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          26_916.78       327.25    27_244.03       0.2572    12.934168        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         26_916.78       381.86    27_298.64       0.3535     8.250671        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         26_916.78       454.92    27_371.70       0.4785     4.974693        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          26_916.78       344.45    27_261.23       0.2538    13.138030        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         26_916.78       395.66    27_312.44       0.3483     8.423305        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         26_916.78       478.30    27_395.08       0.4714     5.106781        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          26_916.78       370.81    27_287.59       0.2506    13.342603        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         26_916.78       422.18    27_338.95       0.3431     8.595270        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         26_916.78       512.61    27_429.38       0.4639     5.243082        10.82
IVF-Binary-512-nl547-random (self)                    26_916.78     3_737.53    30_654.31       0.3531     8.282651        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             31_640.09       235.36    31_875.45       0.1142          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             31_640.09       259.37    31_899.46       0.1125          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             31_640.09       311.75    31_951.84       0.1112          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             31_640.09       315.29    31_955.37       0.2214    15.286778        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            31_640.09       354.11    31_994.20       0.3055    10.075887        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            31_640.09       450.29    32_090.37       0.4203     6.294904        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             31_640.09       329.03    31_969.12       0.2174    15.561100        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            31_640.09       384.11    32_024.20       0.3002    10.284219        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            31_640.09       484.17    32_124.26       0.4126     6.467727        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             31_640.09       382.91    32_023.00       0.2138    15.853041        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            31_640.09       446.98    32_087.07       0.2945    10.524342        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            31_640.09       546.60    32_186.69       0.4049     6.651100        10.69
IVF-Binary-512-nl273-itq (self)                       31_640.09     3_827.13    35_467.22       0.3000    10.281698        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             39_551.53       263.57    39_815.10       0.1134          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             39_551.53       304.79    39_856.33       0.1117          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             39_551.53       327.86    39_879.39       0.2199    15.376718        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            39_551.53       375.06    39_926.60       0.3039    10.118706        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            39_551.53       489.27    40_040.80       0.4181     6.328834        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             39_551.53       376.97    39_928.50       0.2152    15.727693        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            39_551.53       429.46    39_980.99       0.2964    10.421741        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            39_551.53       528.69    40_080.22       0.4077     6.572021        10.74
IVF-Binary-512-nl387-itq (self)                       39_551.53     3_753.60    43_305.13       0.3040    10.115923        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             41_498.71       260.46    41_759.17       0.1144          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             41_498.71       269.06    41_767.77       0.1132          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             41_498.71       292.25    41_790.96       0.1120          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             41_498.71       323.37    41_822.08       0.2225    15.149505        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            41_498.71       374.96    41_873.67       0.3082     9.944341        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            41_498.71       450.99    41_949.70       0.4252     6.178678        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             41_498.71       337.33    41_836.04       0.2193    15.396813        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            41_498.71       389.41    41_888.12       0.3032    10.148389        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            41_498.71       474.27    41_972.98       0.4178     6.353260        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             41_498.71       372.47    41_871.18       0.2158    15.675278        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            41_498.71       418.11    41_916.82       0.2980    10.363909        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            41_498.71       533.01    42_031.72       0.4105     6.530200        10.82
IVF-Binary-512-nl547-itq (self)                       41_498.71     3_764.37    45_263.08       0.3085     9.934030        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         24_875.05       589.47    25_464.52       0.1798          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         24_875.05       660.98    25_536.04       0.1786          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         24_875.05       832.24    25_707.30       0.1774          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         24_875.05       637.86    25_512.92       0.3726     7.706509        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        24_875.05       676.11    25_551.17       0.4955     4.506730        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        24_875.05       748.56    25_623.61       0.6350     2.398049        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         24_875.05       710.63    25_585.68       0.3694     7.803869        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        24_875.05       756.10    25_631.16       0.4909     4.584735        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        24_875.05       836.46    25_711.52       0.6292     2.457159        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         24_875.05       888.26    25_763.31       0.3668     7.888842        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        24_875.05       934.69    25_809.74       0.4870     4.652445        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        24_875.05     1_021.21    25_896.27       0.6243     2.508230        20.09
IVF-Binary-1024-nl273-random (self)                   24_875.05     7_555.27    32_430.32       0.4913     4.572525        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         28_794.65       606.43    29_401.09       0.1792          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         28_794.65       756.88    29_551.53       0.1776          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         28_794.65       656.18    29_450.84       0.3715     7.730908        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        28_794.65       715.33    29_509.98       0.4947     4.515710        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        28_794.65       766.75    29_561.40       0.6341     2.405584        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         28_794.65       799.45    29_594.11       0.3672     7.866060        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        28_794.65       846.31    29_640.97       0.4882     4.630645        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        28_794.65       924.49    29_719.14       0.6260     2.490177        20.15
IVF-Binary-1024-nl387-random (self)                   28_794.65     6_991.49    35_786.15       0.4950     4.507398        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         34_525.08       638.69    35_163.76       0.1802          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         34_525.08       632.39    35_157.47       0.1791          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         34_525.08       707.15    35_232.23       0.1779          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         34_525.08       627.25    35_152.32       0.3746     7.640733        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        34_525.08       663.56    35_188.64       0.4991     4.437135        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        34_525.08       734.02    35_259.10       0.6396     2.348286        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         34_525.08       681.30    35_206.38       0.3710     7.750665        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        34_525.08       721.63    35_246.71       0.4945     4.520106        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        34_525.08       792.34    35_317.42       0.6334     2.411331        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         34_525.08       757.31    35_282.38       0.3682     7.842813        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        34_525.08       798.56    35_323.64       0.4900     4.598772        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        34_525.08       871.74    35_396.81       0.6274     2.474746        20.23
IVF-Binary-1024-nl547-random (self)                   34_525.08     6_640.89    41_165.97       0.4993     4.433988        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            46_299.22       592.56    46_891.77       0.1705          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            46_299.22       661.63    46_960.85       0.1692          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            46_299.22       829.59    47_128.81       0.1682          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            46_299.22       634.04    46_933.25       0.3537     8.360462        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           46_299.22       678.24    46_977.45       0.4718     4.957939        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           46_299.22       749.50    47_048.72       0.6103     2.715929        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            46_299.22       712.68    47_011.89       0.3505     8.465343        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           46_299.22       761.21    47_060.43       0.4675     5.033550        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           46_299.22       840.41    47_139.63       0.6046     2.775845        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            46_299.22       895.39    47_194.60       0.3476     8.566102        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           46_299.22       941.45    47_240.67       0.4636     5.109638        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           46_299.22     1_021.58    47_320.80       0.5998     2.826582        20.09
IVF-Binary-1024-nl273-itq (self)                      46_299.22     7_525.71    53_824.93       0.4673     5.052061        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            47_760.05       610.06    48_370.11       0.1702          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            47_760.05       748.65    48_508.70       0.1686          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            47_760.05       657.85    48_417.90       0.3527     8.389518        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           47_760.05       702.21    48_462.25       0.4707     4.970856        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           47_760.05       770.21    48_530.26       0.6094     2.721540        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            47_760.05       804.36    48_564.41       0.3484     8.534430        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           47_760.05       848.44    48_608.48       0.4647     5.084909        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           47_760.05       927.94    48_687.99       0.6008     2.812096        20.15
IVF-Binary-1024-nl387-itq (self)                      47_760.05     6_965.57    54_725.62       0.4709     4.986155        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            51_653.09       606.96    52_260.05       0.1707          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            51_653.09       630.41    52_283.50       0.1697          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            51_653.09       700.83    52_353.91       0.1687          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            51_653.09       617.33    52_270.42       0.3558     8.297208        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           51_653.09       659.57    52_312.66       0.4746     4.901860        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           51_653.09       728.17    52_381.26       0.6147     2.662573        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            51_653.09       675.05    52_328.13       0.3526     8.401470        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           51_653.09       724.59    52_377.68       0.4702     4.986301        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           51_653.09       794.58    52_447.67       0.6085     2.731754        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            51_653.09       749.28    52_402.37       0.3495     8.501591        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           51_653.09       791.12    52_444.21       0.4662     5.060425        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           51_653.09       875.25    52_528.34       0.6031     2.790449        20.23
IVF-Binary-1024-nl547-itq (self)                      51_653.09     6_598.79    58_251.88       0.4750     4.912407        20.23
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
Exhaustive (query)                                        14.41    16_973.56    16_987.97       1.0000     0.000000        73.24
Exhaustive (self)                                         14.41   169_833.63   169_848.04       1.0000     0.000000        73.24
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)              3_931.06       995.38     4_926.44       0.0770          NaN         4.70
ExhaustiveBinary-256-random-rf5 (query)                3_931.06     1_053.36     4_984.42       0.1853    40.590911         4.70
ExhaustiveBinary-256-random-rf10 (query)               3_931.06     1_101.34     5_032.40       0.2680    26.716815         4.70
ExhaustiveBinary-256-random-rf20 (query)               3_931.06     1_216.56     5_147.62       0.3854    16.589779         4.70
ExhaustiveBinary-256-random (self)                     3_931.06    10_979.15    14_910.21       0.2721    26.279507         4.70
ExhaustiveBinary-256-itq_no_rr (query)                18_137.19     1_020.26    19_157.44       0.0334          NaN         4.70
ExhaustiveBinary-256-itq-rf5 (query)                  18_137.19     1_037.00    19_174.18       0.1047    59.774198         4.70
ExhaustiveBinary-256-itq-rf10 (query)                 18_137.19     1_086.13    19_223.32       0.1682    40.784107         4.70
ExhaustiveBinary-256-itq-rf20 (query)                 18_137.19     1_193.15    19_330.33       0.2648    26.557563         4.70
ExhaustiveBinary-256-itq (self)                       18_137.19    10_854.76    28_991.94       0.1711    40.460557         4.70
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)              7_757.83     1_546.66     9_304.50       0.1527          NaN         9.41
ExhaustiveBinary-512-random-rf5 (query)                7_757.83     1_587.48     9_345.32       0.3242    22.196757         9.41
ExhaustiveBinary-512-random-rf10 (query)               7_757.83     1_642.75     9_400.59       0.4419    13.545010         9.41
ExhaustiveBinary-512-random-rf20 (query)               7_757.83     1_769.29     9_527.12       0.5832     7.719977         9.41
ExhaustiveBinary-512-random (self)                     7_757.83    16_477.99    24_235.83       0.4431    13.453984         9.41
ExhaustiveBinary-512-itq_no_rr (query)                21_694.38     1_527.87    23_222.24       0.1302          NaN         9.41
ExhaustiveBinary-512-itq-rf5 (query)                  21_694.38     1_600.12    23_294.50       0.2779    26.538562         9.41
ExhaustiveBinary-512-itq-rf10 (query)                 21_694.38     1_685.01    23_379.39       0.3835    16.663114         9.41
ExhaustiveBinary-512-itq-rf20 (query)                 21_694.38     1_789.19    23_483.57       0.5194     9.834430         9.41
ExhaustiveBinary-512-itq (self)                       21_694.38    16_684.31    38_378.69       0.3863    16.548400         9.41
--------------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-1024-random_no_rr (query)            15_322.33     2_319.27    17_641.60       0.2353          NaN        18.81
ExhaustiveBinary-1024-random-rf5 (query)              15_322.33     2_379.22    17_701.55       0.4918    11.183271        18.81
ExhaustiveBinary-1024-random-rf10 (query)             15_322.33     2_443.96    17_766.29       0.6358     5.988378        18.81
ExhaustiveBinary-1024-random-rf20 (query)             15_322.33     2_578.18    17_900.52       0.7779     2.871209        18.81
ExhaustiveBinary-1024-random (self)                   15_322.33    24_487.01    39_809.34       0.6381     5.940314        18.81
ExhaustiveBinary-1024-itq_no_rr (query)               33_234.02     2_318.06    35_552.09       0.2191          NaN        18.81
ExhaustiveBinary-1024-itq-rf5 (query)                 33_234.02     2_393.38    35_627.40       0.4611    12.755652        18.81
ExhaustiveBinary-1024-itq-rf10 (query)                33_234.02     2_447.21    35_681.23       0.6023     7.013961        18.81
ExhaustiveBinary-1024-itq-rf20 (query)                33_234.02     2_588.79    35_822.81       0.7468     3.481342        18.81
ExhaustiveBinary-1024-itq (self)                      33_234.02    24_529.79    57_763.82       0.6045     6.950550        18.81
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)          13_482.86       132.29    13_615.15       0.0893          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-random (query)          13_482.86       162.88    13_645.74       0.0819          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-random (query)          13_482.86       167.24    13_650.10       0.0801          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-random (query)          13_482.86       172.75    13_655.61       0.2082    36.993599         5.98
IVF-Binary-256-nl273-np13-rf10-random (query)         13_482.86       218.67    13_701.53       0.2984    23.989933         5.98
IVF-Binary-256-nl273-np13-rf20-random (query)         13_482.86       297.88    13_780.74       0.4183    14.858401         5.98
IVF-Binary-256-nl273-np16-rf5-random (query)          13_482.86       188.68    13_671.54       0.1960    39.504638         5.98
IVF-Binary-256-nl273-np16-rf10-random (query)         13_482.86       229.04    13_711.90       0.2836    25.706140         5.98
IVF-Binary-256-nl273-np16-rf20-random (query)         13_482.86       311.47    13_794.33       0.4031    15.882565         5.98
IVF-Binary-256-nl273-np23-rf5-random (query)          13_482.86       208.07    13_690.93       0.1927    40.255954         5.98
IVF-Binary-256-nl273-np23-rf10-random (query)         13_482.86       251.10    13_733.96       0.2788    26.258549         5.98
IVF-Binary-256-nl273-np23-rf20-random (query)         13_482.86       343.37    13_826.23       0.3973    16.269686         5.98
IVF-Binary-256-nl273-random (self)                    13_482.86     2_284.98    15_767.84       0.2865    25.377132         5.98
IVF-Binary-256-nl387-np19-rf0-random (query)          17_410.50       157.75    17_568.25       0.0832          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-random (query)          17_410.50       167.01    17_577.51       0.0801          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-random (query)          17_410.50       185.26    17_595.76       0.1995    38.526819         6.04
IVF-Binary-256-nl387-np19-rf10-random (query)         17_410.50       234.87    17_645.37       0.2884    25.012886         6.04
IVF-Binary-256-nl387-np19-rf20-random (query)         17_410.50       310.19    17_720.70       0.4104    15.315024         6.04
IVF-Binary-256-nl387-np27-rf5-random (query)          17_410.50       205.79    17_616.29       0.1929    40.071006         6.04
IVF-Binary-256-nl387-np27-rf10-random (query)         17_410.50       248.96    17_659.46       0.2798    26.098762         6.04
IVF-Binary-256-nl387-np27-rf20-random (query)         17_410.50       340.38    17_750.88       0.4005    15.973391         6.04
IVF-Binary-256-nl387-random (self)                    17_410.50     2_327.19    19_737.69       0.2930    24.577105         6.04
IVF-Binary-256-nl547-np23-rf0-random (query)          23_190.71       157.95    23_348.66       0.0847          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-random (query)          23_190.71       163.57    23_354.28       0.0830          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-random (query)          23_190.71       174.85    23_365.56       0.0815          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-random (query)          23_190.71       198.91    23_389.62       0.2026    37.587761         6.12
IVF-Binary-256-nl547-np23-rf10-random (query)         23_190.71       256.89    23_447.60       0.2933    24.331195         6.12
IVF-Binary-256-nl547-np23-rf20-random (query)         23_190.71       322.04    23_512.75       0.4160    14.923996         6.12
IVF-Binary-256-nl547-np27-rf5-random (query)          23_190.71       221.22    23_411.93       0.1985    38.517389         6.12
IVF-Binary-256-nl547-np27-rf10-random (query)         23_190.71       251.79    23_442.50       0.2874    25.037409         6.12
IVF-Binary-256-nl547-np27-rf20-random (query)         23_190.71       331.13    23_521.84       0.4082    15.410763         6.12
IVF-Binary-256-nl547-np33-rf5-random (query)          23_190.71       215.80    23_406.51       0.1948    39.402369         6.12
IVF-Binary-256-nl547-np33-rf10-random (query)         23_190.71       262.75    23_453.46       0.2821    25.704890         6.12
IVF-Binary-256-nl547-np33-rf20-random (query)         23_190.71       346.86    23_537.57       0.4018    15.833447         6.12
IVF-Binary-256-nl547-random (self)                    23_190.71     2_477.73    25_668.44       0.2978    23.888812         6.12
IVF-Binary-256-nl273-np13-rf0-itq (query)             27_612.66       134.04    27_746.70       0.0461          NaN         5.98
IVF-Binary-256-nl273-np16-rf0-itq (query)             27_612.66       139.61    27_752.27       0.0370          NaN         5.98
IVF-Binary-256-nl273-np23-rf0-itq (query)             27_612.66       166.11    27_778.77       0.0355          NaN         5.98
IVF-Binary-256-nl273-np13-rf5-itq (query)             27_612.66       164.48    27_777.14       0.1367    53.774887         5.98
IVF-Binary-256-nl273-np13-rf10-itq (query)            27_612.66       209.35    27_822.01       0.2112    36.386925         5.98
IVF-Binary-256-nl273-np13-rf20-itq (query)            27_612.66       296.01    27_908.67       0.3174    23.363391         5.98
IVF-Binary-256-nl273-np16-rf5-itq (query)             27_612.66       180.46    27_793.12       0.1173    59.670319         5.98
IVF-Binary-256-nl273-np16-rf10-itq (query)            27_612.66       215.12    27_827.78       0.1872    40.769449         5.98
IVF-Binary-256-nl273-np16-rf20-itq (query)            27_612.66       317.26    27_929.92       0.2906    26.159206         5.98
IVF-Binary-256-nl273-np23-rf5-itq (query)             27_612.66       197.00    27_809.66       0.1132    61.887524         5.98
IVF-Binary-256-nl273-np23-rf10-itq (query)            27_612.66       235.94    27_848.60       0.1805    42.818975         5.98
IVF-Binary-256-nl273-np23-rf20-itq (query)            27_612.66       328.02    27_940.69       0.2805    27.530961         5.98
IVF-Binary-256-nl273-itq (self)                       27_612.66     2_138.48    29_751.14       0.1891    40.811980         5.98
IVF-Binary-256-nl387-np19-rf0-itq (query)             31_741.07       143.49    31_884.56       0.0385          NaN         6.04
IVF-Binary-256-nl387-np27-rf0-itq (query)             31_741.07       163.47    31_904.54       0.0357          NaN         6.04
IVF-Binary-256-nl387-np19-rf5-itq (query)             31_741.07       181.67    31_922.74       0.1233    57.094275         6.04
IVF-Binary-256-nl387-np19-rf10-itq (query)            31_741.07       232.07    31_973.14       0.1952    38.607769         6.04
IVF-Binary-256-nl387-np19-rf20-itq (query)            31_741.07       312.13    32_053.21       0.3019    24.520889         6.04
IVF-Binary-256-nl387-np27-rf5-itq (query)             31_741.07       198.07    31_939.14       0.1149    60.313477         6.04
IVF-Binary-256-nl387-np27-rf10-itq (query)            31_741.07       239.64    31_980.71       0.1836    41.059401         6.04
IVF-Binary-256-nl387-np27-rf20-itq (query)            31_741.07       322.00    32_063.08       0.2869    26.115381         6.04
IVF-Binary-256-nl387-itq (self)                       31_741.07     2_212.66    33_953.73       0.1984    38.246595         6.04
IVF-Binary-256-nl547-np23-rf0-itq (query)             37_317.20       155.03    37_472.24       0.0404          NaN         6.12
IVF-Binary-256-nl547-np27-rf0-itq (query)             37_317.20       161.90    37_479.10       0.0389          NaN         6.12
IVF-Binary-256-nl547-np33-rf0-itq (query)             37_317.20       171.98    37_489.18       0.0372          NaN         6.12
IVF-Binary-256-nl547-np23-rf5-itq (query)             37_317.20       202.49    37_519.69       0.1270    55.044851         6.12
IVF-Binary-256-nl547-np23-rf10-itq (query)            37_317.20       237.54    37_554.74       0.2012    37.294444         6.12
IVF-Binary-256-nl547-np23-rf20-itq (query)            37_317.20       310.73    37_627.93       0.3090    23.747014         6.12
IVF-Binary-256-nl547-np27-rf5-itq (query)             37_317.20       209.24    37_526.44       0.1227    56.666707         6.12
IVF-Binary-256-nl547-np27-rf10-itq (query)            37_317.20       243.86    37_561.06       0.1944    38.563023         6.12
IVF-Binary-256-nl547-np27-rf20-itq (query)            37_317.20       317.67    37_634.87       0.3002    24.576522         6.12
IVF-Binary-256-nl547-np33-rf5-itq (query)             37_317.20       210.40    37_527.60       0.1177    58.832410         6.12
IVF-Binary-256-nl547-np33-rf10-itq (query)            37_317.20       252.97    37_570.17       0.1868    40.201969         6.12
IVF-Binary-256-nl547-np33-rf20-itq (query)            37_317.20       355.51    37_672.72       0.2904    25.675710         6.12
IVF-Binary-256-nl547-itq (self)                       37_317.20     2_355.23    39_672.43       0.2043    37.027590         6.12
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)          17_281.22       228.92    17_510.14       0.1596          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-random (query)          17_281.22       247.83    17_529.05       0.1562          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-random (query)          17_281.22       308.85    17_590.07       0.1554          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-random (query)          17_281.22       295.72    17_576.94       0.3370    20.988087        10.69
IVF-Binary-512-nl273-np13-rf10-random (query)         17_281.22       349.17    17_630.39       0.4570    12.738355        10.69
IVF-Binary-512-nl273-np13-rf20-random (query)         17_281.22       431.26    17_712.48       0.5992     7.207613        10.69
IVF-Binary-512-nl273-np16-rf5-random (query)          17_281.22       318.40    17_599.62       0.3308    21.631589        10.69
IVF-Binary-512-nl273-np16-rf10-random (query)         17_281.22       377.80    17_659.02       0.4506    13.121710        10.69
IVF-Binary-512-nl273-np16-rf20-random (query)         17_281.22       471.94    17_753.16       0.5923     7.447403        10.69
IVF-Binary-512-nl273-np23-rf5-random (query)          17_281.22       366.33    17_647.55       0.3289    21.801775        10.69
IVF-Binary-512-nl273-np23-rf10-random (query)         17_281.22       427.56    17_708.78       0.4477    13.266981        10.69
IVF-Binary-512-nl273-np23-rf20-random (query)         17_281.22       527.76    17_808.98       0.5898     7.527176        10.69
IVF-Binary-512-nl273-random (self)                    17_281.22     3_722.79    21_004.01       0.4518    13.046910        10.69
IVF-Binary-512-nl387-np19-rf0-random (query)          21_295.34       249.13    21_544.47       0.1571          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-random (query)          21_295.34       286.37    21_581.71       0.1557          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-random (query)          21_295.34       314.61    21_609.95       0.3331    21.365134        10.74
IVF-Binary-512-nl387-np19-rf10-random (query)         21_295.34       367.62    21_662.96       0.4543    12.892643        10.74
IVF-Binary-512-nl387-np19-rf20-random (query)         21_295.34       450.64    21_745.98       0.5962     7.323626        10.74
IVF-Binary-512-nl387-np27-rf5-random (query)          21_295.34       353.88    21_649.23       0.3295    21.745451        10.74
IVF-Binary-512-nl387-np27-rf10-random (query)         21_295.34       414.36    21_709.70       0.4493    13.172576        10.74
IVF-Binary-512-nl387-np27-rf20-random (query)         21_295.34       508.98    21_804.32       0.5909     7.499165        10.74
IVF-Binary-512-nl387-random (self)                    21_295.34     3_637.10    24_932.44       0.4552    12.836028        10.74
IVF-Binary-512-nl547-np23-rf0-random (query)          26_910.39       252.97    27_163.36       0.1579          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-random (query)          26_910.39       266.19    27_176.58       0.1569          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-random (query)          26_910.39       286.84    27_197.23       0.1560          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-random (query)          26_910.39       345.04    27_255.43       0.3355    21.117117        10.82
IVF-Binary-512-nl547-np23-rf10-random (query)         26_910.39       372.13    27_282.52       0.4570    12.738527        10.82
IVF-Binary-512-nl547-np23-rf20-random (query)         26_910.39       452.28    27_362.67       0.5999     7.181679        10.82
IVF-Binary-512-nl547-np27-rf5-random (query)          26_910.39       335.77    27_246.16       0.3326    21.426167        10.82
IVF-Binary-512-nl547-np27-rf10-random (query)         26_910.39       389.51    27_299.90       0.4531    12.960791        10.82
IVF-Binary-512-nl547-np27-rf20-random (query)         26_910.39       509.51    27_419.90       0.5950     7.348787        10.82
IVF-Binary-512-nl547-np33-rf5-random (query)          26_910.39       357.92    27_268.31       0.3301    21.676595        10.82
IVF-Binary-512-nl547-np33-rf10-random (query)         26_910.39       415.36    27_325.75       0.4501    13.125195        10.82
IVF-Binary-512-nl547-np33-rf20-random (query)         26_910.39       503.94    27_414.33       0.5916     7.462517        10.82
IVF-Binary-512-nl547-random (self)                    26_910.39     3_694.43    30_604.82       0.4583    12.660460        10.82
IVF-Binary-512-nl273-np13-rf0-itq (query)             41_111.50       255.74    41_367.25       0.1373          NaN        10.69
IVF-Binary-512-nl273-np16-rf0-itq (query)             41_111.50       299.79    41_411.29       0.1336          NaN        10.69
IVF-Binary-512-nl273-np23-rf0-itq (query)             41_111.50       337.51    41_449.02       0.1327          NaN        10.69
IVF-Binary-512-nl273-np13-rf5-itq (query)             41_111.50       318.34    41_429.84       0.2921    25.040278        10.69
IVF-Binary-512-nl273-np13-rf10-itq (query)            41_111.50       360.49    41_471.99       0.4014    15.571976        10.69
IVF-Binary-512-nl273-np13-rf20-itq (query)            41_111.50       439.01    41_550.52       0.5382     9.170771        10.69
IVF-Binary-512-nl273-np16-rf5-itq (query)             41_111.50       337.83    41_449.33       0.2858    25.784143        10.69
IVF-Binary-512-nl273-np16-rf10-itq (query)            41_111.50       389.01    41_500.52       0.3942    16.060469        10.69
IVF-Binary-512-nl273-np16-rf20-itq (query)            41_111.50       475.77    41_587.27       0.5303     9.478394        10.69
IVF-Binary-512-nl273-np23-rf5-itq (query)             41_111.50       395.52    41_507.03       0.2834    26.075631        10.69
IVF-Binary-512-nl273-np23-rf10-itq (query)            41_111.50       452.79    41_564.30       0.3910    16.268067        10.69
IVF-Binary-512-nl273-np23-rf20-itq (query)            41_111.50       554.32    41_665.83       0.5267     9.612246        10.69
IVF-Binary-512-nl273-itq (self)                       41_111.50     3_860.09    44_971.59       0.3965    15.986556        10.69
IVF-Binary-512-nl387-np19-rf0-itq (query)             39_198.70       272.89    39_471.58       0.1354          NaN        10.74
IVF-Binary-512-nl387-np27-rf0-itq (query)             39_198.70       318.61    39_517.31       0.1337          NaN        10.74
IVF-Binary-512-nl387-np19-rf5-itq (query)             39_198.70       328.17    39_526.87       0.2879    25.437026        10.74
IVF-Binary-512-nl387-np19-rf10-itq (query)            39_198.70       381.38    39_580.08       0.3968    15.872358        10.74
IVF-Binary-512-nl387-np19-rf20-itq (query)            39_198.70       457.14    39_655.84       0.5338     9.321871        10.74
IVF-Binary-512-nl387-np27-rf5-itq (query)             39_198.70       403.09    39_601.79       0.2837    25.979498        10.74
IVF-Binary-512-nl387-np27-rf10-itq (query)            39_198.70       430.45    39_629.15       0.3911    16.259703        10.74
IVF-Binary-512-nl387-np27-rf20-itq (query)            39_198.70       526.09    39_724.78       0.5272     9.577543        10.74
IVF-Binary-512-nl387-itq (self)                       39_198.70     3_738.23    42_936.93       0.4000    15.738500        10.74
IVF-Binary-512-nl547-np23-rf0-itq (query)             41_345.50       270.66    41_616.15       0.1359          NaN        10.82
IVF-Binary-512-nl547-np27-rf0-itq (query)             41_345.50       292.33    41_637.83       0.1346          NaN        10.82
IVF-Binary-512-nl547-np33-rf0-itq (query)             41_345.50       322.35    41_667.84       0.1335          NaN        10.82
IVF-Binary-512-nl547-np23-rf5-itq (query)             41_345.50       336.25    41_681.75       0.2909    25.061820        10.82
IVF-Binary-512-nl547-np23-rf10-itq (query)            41_345.50       380.26    41_725.75       0.4003    15.624309        10.82
IVF-Binary-512-nl547-np23-rf20-itq (query)            41_345.50       454.92    41_800.42       0.5386     9.142162        10.82
IVF-Binary-512-nl547-np27-rf5-itq (query)             41_345.50       347.32    41_692.82       0.2880    25.409619        10.82
IVF-Binary-512-nl547-np27-rf10-itq (query)            41_345.50       394.40    41_739.90       0.3963    15.878587        10.82
IVF-Binary-512-nl547-np27-rf20-itq (query)            41_345.50       475.68    41_821.18       0.5335     9.325666        10.82
IVF-Binary-512-nl547-np33-rf5-itq (query)             41_345.50       378.60    41_724.09       0.2856    25.723268        10.82
IVF-Binary-512-nl547-np33-rf10-itq (query)            41_345.50       425.58    41_771.07       0.3927    16.122235        10.82
IVF-Binary-512-nl547-np33-rf20-itq (query)            41_345.50       515.59    41_861.08       0.5294     9.482494        10.82
IVF-Binary-512-nl547-itq (self)                       41_345.50     3_740.64    45_086.14       0.4033    15.508897        10.82
--------------------------------------------------------------------------------------------------------------------------------
IVF-Binary-1024-nl273-np13-rf0-random (query)         24_793.49       570.41    25_363.91       0.2398          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-random (query)         24_793.49       645.46    25_438.95       0.2382          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-random (query)         24_793.49       792.77    25_586.26       0.2377          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-random (query)         24_793.49       624.37    25_417.86       0.4991    10.836071        20.09
IVF-Binary-1024-nl273-np13-rf10-random (query)        24_793.49       668.14    25_461.63       0.6434     5.779229        20.09
IVF-Binary-1024-nl273-np13-rf20-random (query)        24_793.49       737.05    25_530.54       0.7844     2.748931        20.09
IVF-Binary-1024-nl273-np16-rf5-random (query)         24_793.49       693.47    25_486.96       0.4962    10.983115        20.09
IVF-Binary-1024-nl273-np16-rf10-random (query)        24_793.49       733.52    25_527.01       0.6406     5.858872        20.09
IVF-Binary-1024-nl273-np16-rf20-random (query)        24_793.49       810.98    25_604.47       0.7820     2.798482        20.09
IVF-Binary-1024-nl273-np23-rf5-random (query)         24_793.49       855.53    25_649.02       0.4951    11.038008        20.09
IVF-Binary-1024-nl273-np23-rf10-random (query)        24_793.49       888.79    25_682.29       0.6394     5.893246        20.09
IVF-Binary-1024-nl273-np23-rf20-random (query)        24_793.49       967.04    25_760.53       0.7808     2.821227        20.09
IVF-Binary-1024-nl273-random (self)                   24_793.49     7_349.15    32_142.64       0.6428     5.813892        20.09
IVF-Binary-1024-nl387-np19-rf0-random (query)         28_822.35       590.88    29_413.23       0.2387          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-random (query)         28_822.35       717.03    29_539.39       0.2377          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-random (query)         28_822.35       650.46    29_472.81       0.4978    10.900359        20.15
IVF-Binary-1024-nl387-np19-rf10-random (query)        28_822.35       689.97    29_512.33       0.6417     5.823510        20.15
IVF-Binary-1024-nl387-np19-rf20-random (query)        28_822.35       756.37    29_578.73       0.7830     2.778159        20.15
IVF-Binary-1024-nl387-np27-rf5-random (query)         28_822.35       771.17    29_593.52       0.4959    11.002308        20.15
IVF-Binary-1024-nl387-np27-rf10-random (query)        28_822.35       809.37    29_631.73       0.6394     5.889963        20.15
IVF-Binary-1024-nl387-np27-rf20-random (query)        28_822.35       939.41    29_761.77       0.7809     2.816233        20.15
IVF-Binary-1024-nl387-random (self)                   28_822.35     6_835.68    35_658.03       0.6441     5.772937        20.15
IVF-Binary-1024-nl547-np23-rf0-random (query)         34_251.20       581.70    34_832.90       0.2394          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-random (query)         34_251.20       622.92    34_874.12       0.2386          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-random (query)         34_251.20       689.53    34_940.73       0.2379          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-random (query)         34_251.20       619.26    34_870.46       0.4993    10.821107        20.23
IVF-Binary-1024-nl547-np23-rf10-random (query)        34_251.20       663.47    34_914.67       0.6438     5.766694        20.23
IVF-Binary-1024-nl547-np23-rf20-random (query)        34_251.20       727.30    34_978.50       0.7852     2.739505        20.23
IVF-Binary-1024-nl547-np27-rf5-random (query)         34_251.20       670.22    34_921.42       0.4976    10.907165        20.23
IVF-Binary-1024-nl547-np27-rf10-random (query)        34_251.20       711.45    34_962.65       0.6415     5.829337        20.23
IVF-Binary-1024-nl547-np27-rf20-random (query)        34_251.20       779.30    35_030.50       0.7828     2.783141        20.23
IVF-Binary-1024-nl547-np33-rf5-random (query)         34_251.20       731.64    34_982.84       0.4962    10.976797        20.23
IVF-Binary-1024-nl547-np33-rf10-random (query)        34_251.20       779.63    35_030.83       0.6400     5.874718        20.23
IVF-Binary-1024-nl547-np33-rf20-random (query)        34_251.20       855.70    35_106.90       0.7814     2.811286        20.23
IVF-Binary-1024-nl547-random (self)                   34_251.20     6_573.38    40_824.58       0.6462     5.710244        20.23
IVF-Binary-1024-nl273-np13-rf0-itq (query)            42_086.34       570.58    42_656.92       0.2232          NaN        20.09
IVF-Binary-1024-nl273-np16-rf0-itq (query)            42_086.34       671.39    42_757.73       0.2214          NaN        20.09
IVF-Binary-1024-nl273-np23-rf0-itq (query)            42_086.34       783.80    42_870.14       0.2211          NaN        20.09
IVF-Binary-1024-nl273-np13-rf5-itq (query)            42_086.34       625.86    42_712.20       0.4696    12.293205        20.09
IVF-Binary-1024-nl273-np13-rf10-itq (query)           42_086.34       665.18    42_751.52       0.6111     6.716065        20.09
IVF-Binary-1024-nl273-np13-rf20-itq (query)           42_086.34       744.72    42_831.06       0.7546     3.315146        20.09
IVF-Binary-1024-nl273-np16-rf5-itq (query)            42_086.34       704.39    42_790.73       0.4663    12.481033        20.09
IVF-Binary-1024-nl273-np16-rf10-itq (query)           42_086.34       733.47    42_819.81       0.6075     6.838278        20.09
IVF-Binary-1024-nl273-np16-rf20-itq (query)           42_086.34       809.42    42_895.76       0.7513     3.382954        20.09
IVF-Binary-1024-nl273-np23-rf5-itq (query)            42_086.34       890.14    42_976.48       0.4651    12.563092        20.09
IVF-Binary-1024-nl273-np23-rf10-itq (query)           42_086.34       894.48    42_980.82       0.6060     6.889035        20.09
IVF-Binary-1024-nl273-np23-rf20-itq (query)           42_086.34       969.38    43_055.73       0.7499     3.414539        20.09
IVF-Binary-1024-nl273-itq (self)                      42_086.34     7_332.19    49_418.53       0.6101     6.773217        20.09
IVF-Binary-1024-nl387-np19-rf0-itq (query)            48_106.56       603.97    48_710.53       0.2222          NaN        20.15
IVF-Binary-1024-nl387-np27-rf0-itq (query)            48_106.56       725.86    48_832.42       0.2210          NaN        20.15
IVF-Binary-1024-nl387-np19-rf5-itq (query)            48_106.56       640.10    48_746.66       0.4677    12.397384        20.15
IVF-Binary-1024-nl387-np19-rf10-itq (query)           48_106.56       691.34    48_797.90       0.6087     6.790787        20.15
IVF-Binary-1024-nl387-np19-rf20-itq (query)           48_106.56       753.09    48_859.65       0.7530     3.347757        20.15
IVF-Binary-1024-nl387-np27-rf5-itq (query)            48_106.56       795.86    48_902.43       0.4649    12.568854        20.15
IVF-Binary-1024-nl387-np27-rf10-itq (query)           48_106.56       822.74    48_929.30       0.6058     6.889699        20.15
IVF-Binary-1024-nl387-np27-rf20-itq (query)           48_106.56       887.71    48_994.27       0.7503     3.409587        20.15
IVF-Binary-1024-nl387-itq (self)                      48_106.56     6_816.45    54_923.01       0.6116     6.726623        20.15
IVF-Binary-1024-nl547-np23-rf0-itq (query)            48_708.47       569.82    49_278.28       0.2230          NaN        20.23
IVF-Binary-1024-nl547-np27-rf0-itq (query)            48_708.47       628.15    49_336.62       0.2223          NaN        20.23
IVF-Binary-1024-nl547-np33-rf0-itq (query)            48_708.47       681.23    49_389.70       0.2217          NaN        20.23
IVF-Binary-1024-nl547-np23-rf5-itq (query)            48_708.47       617.30    49_325.76       0.4690    12.315152        20.23
IVF-Binary-1024-nl547-np23-rf10-itq (query)           48_708.47       657.45    49_365.92       0.6108     6.716782        20.23
IVF-Binary-1024-nl547-np23-rf20-itq (query)           48_708.47       735.31    49_443.78       0.7554     3.299614        20.23
IVF-Binary-1024-nl547-np27-rf5-itq (query)            48_708.47       667.24    49_375.71       0.4670    12.424749        20.23
IVF-Binary-1024-nl547-np27-rf10-itq (query)           48_708.47       703.94    49_412.41       0.6084     6.798214        20.23
IVF-Binary-1024-nl547-np27-rf20-itq (query)           48_708.47       774.20    49_482.67       0.7527     3.358308        20.23
IVF-Binary-1024-nl547-np33-rf5-itq (query)            48_708.47       751.85    49_460.32       0.4659    12.497905        20.23
IVF-Binary-1024-nl547-np33-rf10-itq (query)           48_708.47       775.18    49_483.65       0.6068     6.851983        20.23
IVF-Binary-1024-nl547-np33-rf20-itq (query)           48_708.47       850.70    49_559.16       0.7510     3.393056        20.23
IVF-Binary-1024-nl547-itq (self)                      48_708.47     6_581.12    55_289.59       0.6138     6.650511        20.23
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
===========================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.05     2_327.35     2_330.39       1.0000     0.000000        18.31
Exhaustive (self)                                     3.05    23_422.90    23_425.95       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                      2_448.73     1_135.22     3_583.94       0.3141    41.590414         2.89
ExhaustiveRaBitQ-rf5 (query)                      2_448.73     1_222.08     3_670.81       0.6781     1.424540         2.89
ExhaustiveRaBitQ-rf10 (query)                     2_448.73     1_296.77     3_745.49       0.8317     0.540476         2.89
ExhaustiveRaBitQ-rf20 (query)                     2_448.73     1_526.84     3_975.57       0.9366     0.159663         2.89
ExhaustiveRaBitQ (self)                           2_448.73    13_021.80    15_470.52       0.8323     0.527522         2.89
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                 1_442.74       247.65     1_690.38       0.3204    41.573455         2.90
IVF-RaBitQ-nl273-np16-rf0 (query)                 1_442.74       299.65     1_742.38       0.3194    41.580514         2.90
IVF-RaBitQ-nl273-np23-rf0 (query)                 1_442.74       421.41     1_864.15       0.3181    41.584091         2.90
IVF-RaBitQ-nl273-np13-rf5 (query)                 1_442.74       305.36     1_748.10       0.6868     1.446637         2.90
IVF-RaBitQ-nl273-np13-rf10 (query)                1_442.74       354.42     1_797.16       0.8368     0.558447         2.90
IVF-RaBitQ-nl273-np13-rf20 (query)                1_442.74       446.75     1_889.49       0.9353     0.179393         2.90
IVF-RaBitQ-nl273-np16-rf5 (query)                 1_442.74       359.29     1_802.02       0.6864     1.449594         2.90
IVF-RaBitQ-nl273-np16-rf10 (query)                1_442.74       419.38     1_862.12       0.8389     0.547916         2.90
IVF-RaBitQ-nl273-np16-rf20 (query)                1_442.74       509.09     1_951.83       0.9396     0.160511         2.90
IVF-RaBitQ-nl273-np23-rf5 (query)                 1_442.74       483.47     1_926.21       0.6851     1.459478         2.90
IVF-RaBitQ-nl273-np23-rf10 (query)                1_442.74       539.90     1_982.64       0.8385     0.550184         2.90
IVF-RaBitQ-nl273-np23-rf20 (query)                1_442.74       644.40     2_087.14       0.9406     0.159050         2.90
IVF-RaBitQ-nl273 (self)                           1_442.74     6_680.01     8_122.75       0.9410     0.156346         2.90
IVF-RaBitQ-nl387-np19-rf0 (query)                 2_318.54       269.57     2_588.11       0.3216    41.563234         2.92
IVF-RaBitQ-nl387-np27-rf0 (query)                 2_318.54       366.72     2_685.26       0.3192    41.572820         2.92
IVF-RaBitQ-nl387-np19-rf5 (query)                 2_318.54       333.17     2_651.71       0.6917     1.394283         2.92
IVF-RaBitQ-nl387-np19-rf10 (query)                2_318.54       397.57     2_716.10       0.8431     0.518721         2.92
IVF-RaBitQ-nl387-np19-rf20 (query)                2_318.54       528.36     2_846.90       0.9398     0.165846         2.92
IVF-RaBitQ-nl387-np27-rf5 (query)                 2_318.54       526.91     2_845.45       0.6889     1.417503         2.92
IVF-RaBitQ-nl387-np27-rf10 (query)                2_318.54       654.41     2_972.95       0.8427     0.516801         2.92
IVF-RaBitQ-nl387-np27-rf20 (query)                2_318.54       657.74     2_976.28       0.9428     0.146455         2.92
IVF-RaBitQ-nl387 (self)                           2_318.54     5_937.58     8_256.12       0.9429     0.146907         2.92
IVF-RaBitQ-nl547-np23-rf0 (query)                 2_890.74       244.85     3_135.60       0.3273    41.545480         2.94
IVF-RaBitQ-nl547-np27-rf0 (query)                 2_890.74       277.79     3_168.54       0.3254    41.553623         2.94
IVF-RaBitQ-nl547-np33-rf0 (query)                 2_890.74       333.71     3_224.46       0.3239    41.559071         2.94
IVF-RaBitQ-nl547-np23-rf5 (query)                 2_890.74       294.34     3_185.08       0.6984     1.344605         2.94
IVF-RaBitQ-nl547-np23-rf10 (query)                2_890.74       340.63     3_231.38       0.8469     0.506009         2.94
IVF-RaBitQ-nl547-np23-rf20 (query)                2_890.74       462.42     3_353.16       0.9406     0.166928         2.94
IVF-RaBitQ-nl547-np27-rf5 (query)                 2_890.74       381.72     3_272.46       0.6963     1.355331         2.94
IVF-RaBitQ-nl547-np27-rf10 (query)                2_890.74       448.28     3_339.02       0.8473     0.498653         2.94
IVF-RaBitQ-nl547-np27-rf20 (query)                2_890.74       526.64     3_417.38       0.9441     0.147870         2.94
IVF-RaBitQ-nl547-np33-rf5 (query)                 2_890.74       399.07     3_289.81       0.6944     1.371568         2.94
IVF-RaBitQ-nl547-np33-rf10 (query)                2_890.74       543.33     3_434.07       0.8464     0.502978         2.94
IVF-RaBitQ-nl547-np33-rf20 (query)                2_890.74       622.61     3_513.35       0.9452     0.141918         2.94
IVF-RaBitQ-nl547 (self)                           2_890.74     5_744.18     8_634.93       0.9453     0.139731         2.94
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Cosine (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.17     2_660.57     2_664.74       1.0000     0.000000        18.88
Exhaustive (self)                                     4.17    26_838.40    26_842.58       1.0000     0.000000        18.88
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                      2_776.19     1_152.17     3_928.36       0.3168     0.168379         2.89
ExhaustiveRaBitQ-rf5 (query)                      2_776.19     1_259.49     4_035.68       0.6843     0.001049         2.89
ExhaustiveRaBitQ-rf10 (query)                     2_776.19     1_325.94     4_102.13       0.8374     0.000388         2.89
ExhaustiveRaBitQ-rf20 (query)                     2_776.19     1_491.75     4_267.94       0.9399     0.000111         2.89
ExhaustiveRaBitQ (self)                           2_776.19    12_967.53    15_743.72       0.8381     0.000385         2.89
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                 1_696.82       347.01     2_043.83       0.3162     0.167947         2.90
IVF-RaBitQ-nl273-np16-rf0 (query)                 1_696.82       353.41     2_050.22       0.3149     0.167653         2.90
IVF-RaBitQ-nl273-np23-rf0 (query)                 1_696.82       504.52     2_201.33       0.3136     0.167498         2.90
IVF-RaBitQ-nl273-np13-rf5 (query)                 1_696.82       358.78     2_055.60       0.6842     0.001129         2.90
IVF-RaBitQ-nl273-np13-rf10 (query)                1_696.82       430.63     2_127.45       0.8362     0.000438         2.90
IVF-RaBitQ-nl273-np13-rf20 (query)                1_696.82       588.44     2_285.25       0.9352     0.000143         2.90
IVF-RaBitQ-nl273-np16-rf5 (query)                 1_696.82       385.56     2_082.38       0.6838     0.001136         2.90
IVF-RaBitQ-nl273-np16-rf10 (query)                1_696.82       419.15     2_115.97       0.8375     0.000434         2.90
IVF-RaBitQ-nl273-np16-rf20 (query)                1_696.82       540.23     2_237.05       0.9396     0.000129         2.90
IVF-RaBitQ-nl273-np23-rf5 (query)                 1_696.82       506.06     2_202.88       0.6820     0.001146         2.90
IVF-RaBitQ-nl273-np23-rf10 (query)                1_696.82       561.91     2_258.72       0.8372     0.000435         2.90
IVF-RaBitQ-nl273-np23-rf20 (query)                1_696.82       688.60     2_385.41       0.9407     0.000126         2.90
IVF-RaBitQ-nl273 (self)                           1_696.82     7_284.00     8_980.82       0.9410     0.000126         2.90
IVF-RaBitQ-nl387-np19-rf0 (query)                 2_367.94       299.78     2_667.72       0.3194     0.168484         2.92
IVF-RaBitQ-nl387-np27-rf0 (query)                 2_367.94       422.98     2_790.93       0.3170     0.168113         2.92
IVF-RaBitQ-nl387-np19-rf5 (query)                 2_367.94       367.16     2_735.11       0.6898     0.001088         2.92
IVF-RaBitQ-nl387-np19-rf10 (query)                2_367.94       430.03     2_797.98       0.8414     0.000415         2.92
IVF-RaBitQ-nl387-np19-rf20 (query)                2_367.94       625.61     2_993.55       0.9397     0.000129         2.92
IVF-RaBitQ-nl387-np27-rf5 (query)                 2_367.94       447.70     2_815.65       0.6871     0.001107         2.92
IVF-RaBitQ-nl387-np27-rf10 (query)                2_367.94       495.53     2_863.48       0.8408     0.000418         2.92
IVF-RaBitQ-nl387-np27-rf20 (query)                2_367.94       621.30     2_989.25       0.9422     0.000120         2.92
IVF-RaBitQ-nl387 (self)                           2_367.94     6_361.57     8_729.51       0.9431     0.000118         2.92
IVF-RaBitQ-nl547-np23-rf0 (query)                 3_161.71       252.06     3_413.76       0.3220     0.169073         2.94
IVF-RaBitQ-nl547-np27-rf0 (query)                 3_161.71       291.08     3_452.78       0.3199     0.168766         2.94
IVF-RaBitQ-nl547-np33-rf0 (query)                 3_161.71       333.53     3_495.23       0.3183     0.168569         2.94
IVF-RaBitQ-nl547-np23-rf5 (query)                 3_161.71       328.86     3_490.57       0.6956     0.001055         2.94
IVF-RaBitQ-nl547-np23-rf10 (query)                3_161.71       356.65     3_518.35       0.8451     0.000401         2.94
IVF-RaBitQ-nl547-np23-rf20 (query)                3_161.71       424.50     3_586.20       0.9404     0.000129         2.94
IVF-RaBitQ-nl547-np27-rf5 (query)                 3_161.71       327.68     3_489.38       0.6930     0.001069         2.94
IVF-RaBitQ-nl547-np27-rf10 (query)                3_161.71       393.41     3_555.12       0.8449     0.000401         2.94
IVF-RaBitQ-nl547-np27-rf20 (query)                3_161.71       501.34     3_663.05       0.9437     0.000119         2.94
IVF-RaBitQ-nl547-np33-rf5 (query)                 3_161.71       381.33     3_543.03       0.6903     0.001086         2.94
IVF-RaBitQ-nl547-np33-rf10 (query)                3_161.71       445.03     3_606.74       0.8438     0.000406         2.94
IVF-RaBitQ-nl547-np33-rf20 (query)                3_161.71       555.34     3_717.04       0.9441     0.000117         2.94
IVF-RaBitQ-nl547 (self)                           3_161.71     5_338.25     8_499.95       0.9452     0.000112         2.94
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (Correlated)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.40     2_406.65     2_410.05       1.0000     0.000000        18.31
Exhaustive (self)                                     3.40    25_368.16    25_371.55       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                      2_653.97       938.58     3_592.55       0.4425     1.402782         2.89
ExhaustiveRaBitQ-rf5 (query)                      2_653.97       983.42     3_637.39       0.8784     0.030818         2.89
ExhaustiveRaBitQ-rf10 (query)                     2_653.97     1_146.71     3_800.68       0.9681     0.005683         2.89
ExhaustiveRaBitQ-rf20 (query)                     2_653.97     1_479.90     4_133.88       0.9954     0.000661         2.89
ExhaustiveRaBitQ (self)                           2_653.97    11_130.77    13_784.74       0.9697     0.005495         2.89
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                 1_623.56       284.30     1_907.85       0.4624     1.393408         2.90
IVF-RaBitQ-nl273-np16-rf0 (query)                 1_623.56       336.19     1_959.75       0.4620     1.393563         2.90
IVF-RaBitQ-nl273-np23-rf0 (query)                 1_623.56       483.67     2_107.22       0.4619     1.393620         2.90
IVF-RaBitQ-nl273-np13-rf5 (query)                 1_623.56       338.80     1_962.36       0.8931     0.028096         2.90
IVF-RaBitQ-nl273-np13-rf10 (query)                1_623.56       399.83     2_023.38       0.9745     0.004923         2.90
IVF-RaBitQ-nl273-np13-rf20 (query)                1_623.56       493.87     2_117.42       0.9968     0.000521         2.90
IVF-RaBitQ-nl273-np16-rf5 (query)                 1_623.56       397.77     2_021.32       0.8928     0.028222         2.90
IVF-RaBitQ-nl273-np16-rf10 (query)                1_623.56       469.19     2_092.75       0.9744     0.004947         2.90
IVF-RaBitQ-nl273-np16-rf20 (query)                1_623.56       578.80     2_202.35       0.9969     0.000502         2.90
IVF-RaBitQ-nl273-np23-rf5 (query)                 1_623.56       570.24     2_193.80       0.8926     0.028286         2.90
IVF-RaBitQ-nl273-np23-rf10 (query)                1_623.56       642.07     2_265.63       0.9743     0.004967         2.90
IVF-RaBitQ-nl273-np23-rf20 (query)                1_623.56       739.10     2_362.66       0.9969     0.000504         2.90
IVF-RaBitQ-nl273 (self)                           1_623.56     7_179.42     8_802.98       0.9973     0.000448         2.90
IVF-RaBitQ-nl387-np19-rf0 (query)                 2_198.21       294.15     2_492.36       0.4729     1.382072         2.92
IVF-RaBitQ-nl387-np27-rf0 (query)                 2_198.21       418.68     2_616.89       0.4728     1.382135         2.92
IVF-RaBitQ-nl387-np19-rf5 (query)                 2_198.21       353.86     2_552.08       0.9030     0.024211         2.92
IVF-RaBitQ-nl387-np19-rf10 (query)                2_198.21       373.14     2_571.35       0.9778     0.004106         2.92
IVF-RaBitQ-nl387-np19-rf20 (query)                2_198.21       438.39     2_636.60       0.9976     0.000386         2.92
IVF-RaBitQ-nl387-np27-rf5 (query)                 2_198.21       446.28     2_644.49       0.9028     0.024253         2.92
IVF-RaBitQ-nl387-np27-rf10 (query)                2_198.21       549.36     2_747.57       0.9778     0.004117         2.92
IVF-RaBitQ-nl387-np27-rf20 (query)                2_198.21       657.43     2_855.64       0.9976     0.000376         2.92
IVF-RaBitQ-nl387 (self)                           2_198.21     5_866.69     8_064.90       0.9978     0.000345         2.92
IVF-RaBitQ-nl547-np23-rf0 (query)                 3_012.00       266.86     3_278.86       0.4837     1.373040         2.94
IVF-RaBitQ-nl547-np27-rf0 (query)                 3_012.00       307.15     3_319.15       0.4835     1.373146         2.94
IVF-RaBitQ-nl547-np33-rf0 (query)                 3_012.00       342.99     3_354.99       0.4834     1.373184         2.94
IVF-RaBitQ-nl547-np23-rf5 (query)                 3_012.00       309.09     3_321.08       0.9115     0.021036         2.94
IVF-RaBitQ-nl547-np23-rf10 (query)                3_012.00       383.65     3_395.64       0.9815     0.003284         2.94
IVF-RaBitQ-nl547-np23-rf20 (query)                3_012.00       438.09     3_450.09       0.9980     0.000305         2.94
IVF-RaBitQ-nl547-np27-rf5 (query)                 3_012.00       323.26     3_335.26       0.9112     0.021118         2.94
IVF-RaBitQ-nl547-np27-rf10 (query)                3_012.00       371.61     3_383.61       0.9813     0.003309         2.94
IVF-RaBitQ-nl547-np27-rf20 (query)                3_012.00       451.74     3_463.74       0.9980     0.000300         2.94
IVF-RaBitQ-nl547-np33-rf5 (query)                 3_012.00       378.69     3_390.69       0.9111     0.021140         2.94
IVF-RaBitQ-nl547-np33-rf10 (query)                3_012.00       426.59     3_438.59       0.9813     0.003314         2.94
IVF-RaBitQ-nl547-np33-rf20 (query)                3_012.00       516.63     3_528.62       0.9980     0.000294         2.94
IVF-RaBitQ-nl547 (self)                           3_012.00     5_280.95     8_292.95       0.9982     0.000269         2.94
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.75     2_628.72     2_632.47       1.0000     0.000000        18.31
Exhaustive (self)                                     3.75    24_697.84    24_701.60       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                      2_582.61       963.59     3_546.20       0.4782     7.158455         2.89
ExhaustiveRaBitQ-rf5 (query)                      2_582.61     1_164.52     3_747.13       0.9141     0.089864         2.89
ExhaustiveRaBitQ-rf10 (query)                     2_582.61     1_180.02     3_762.63       0.9835     0.012770         2.89
ExhaustiveRaBitQ-rf20 (query)                     2_582.61     1_216.30     3_798.91       0.9985     0.001018         2.89
ExhaustiveRaBitQ (self)                           2_582.61    11_625.44    14_208.06       0.9834     0.013135         2.89
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                 1_487.46       241.25     1_728.71       0.4887     7.133549         2.90
IVF-RaBitQ-nl273-np16-rf0 (query)                 1_487.46       300.88     1_788.34       0.4887     7.133579         2.90
IVF-RaBitQ-nl273-np23-rf0 (query)                 1_487.46       413.27     1_900.73       0.4887     7.133584         2.90
IVF-RaBitQ-nl273-np13-rf5 (query)                 1_487.46       297.12     1_784.59       0.9216     0.078073         2.90
IVF-RaBitQ-nl273-np13-rf10 (query)                1_487.46       349.72     1_837.18       0.9858     0.010544         2.90
IVF-RaBitQ-nl273-np13-rf20 (query)                1_487.46       429.89     1_917.36       0.9988     0.000755         2.90
IVF-RaBitQ-nl273-np16-rf5 (query)                 1_487.46       348.03     1_835.50       0.9216     0.078134         2.90
IVF-RaBitQ-nl273-np16-rf10 (query)                1_487.46       400.49     1_887.95       0.9857     0.010553         2.90
IVF-RaBitQ-nl273-np16-rf20 (query)                1_487.46       484.69     1_972.16       0.9988     0.000741         2.90
IVF-RaBitQ-nl273-np23-rf5 (query)                 1_487.46       473.07     1_960.54       0.9216     0.078134         2.90
IVF-RaBitQ-nl273-np23-rf10 (query)                1_487.46       532.14     2_019.61       0.9857     0.010558         2.90
IVF-RaBitQ-nl273-np23-rf20 (query)                1_487.46       638.57     2_126.03       0.9988     0.000741         2.90
IVF-RaBitQ-nl273 (self)                           1_487.46     6_226.99     7_714.46       0.9988     0.000729         2.90
IVF-RaBitQ-nl387-np19-rf0 (query)                 2_000.53       260.24     2_260.77       0.5004     7.112936         2.92
IVF-RaBitQ-nl387-np27-rf0 (query)                 2_000.53       360.70     2_361.23       0.5004     7.112938         2.92
IVF-RaBitQ-nl387-np19-rf5 (query)                 2_000.53       310.32     2_310.85       0.9290     0.068037         2.92
IVF-RaBitQ-nl387-np19-rf10 (query)                2_000.53       385.18     2_385.71       0.9875     0.009140         2.92
IVF-RaBitQ-nl387-np19-rf20 (query)                2_000.53       469.93     2_470.46       0.9991     0.000516         2.92
IVF-RaBitQ-nl387-np27-rf5 (query)                 2_000.53       448.40     2_448.93       0.9290     0.068059         2.92
IVF-RaBitQ-nl387-np27-rf10 (query)                2_000.53       513.13     2_513.66       0.9875     0.009147         2.92
IVF-RaBitQ-nl387-np27-rf20 (query)                2_000.53       574.74     2_575.27       0.9991     0.000516         2.92
IVF-RaBitQ-nl387 (self)                           2_000.53     6_210.89     8_211.42       0.9991     0.000578         2.92
IVF-RaBitQ-nl547-np23-rf0 (query)                 2_949.91       241.43     3_191.34       0.5107     7.089669         2.94
IVF-RaBitQ-nl547-np27-rf0 (query)                 2_949.91       304.61     3_254.52       0.5107     7.089682         2.94
IVF-RaBitQ-nl547-np33-rf0 (query)                 2_949.91       374.90     3_324.81       0.5107     7.089686         2.94
IVF-RaBitQ-nl547-np23-rf5 (query)                 2_949.91       346.66     3_296.57       0.9363     0.058704         2.94
IVF-RaBitQ-nl547-np23-rf10 (query)                2_949.91       390.42     3_340.33       0.9896     0.007452         2.94
IVF-RaBitQ-nl547-np23-rf20 (query)                2_949.91       474.19     3_424.10       0.9994     0.000388         2.94
IVF-RaBitQ-nl547-np27-rf5 (query)                 2_949.91       383.79     3_333.70       0.9362     0.058746         2.94
IVF-RaBitQ-nl547-np27-rf10 (query)                2_949.91       479.90     3_429.81       0.9895     0.007453         2.94
IVF-RaBitQ-nl547-np27-rf20 (query)                2_949.91       539.00     3_488.91       0.9994     0.000393         2.94
IVF-RaBitQ-nl547-np33-rf5 (query)                 2_949.91       416.64     3_366.55       0.9362     0.058746         2.94
IVF-RaBitQ-nl547-np33-rf10 (query)                2_949.91       426.62     3_376.53       0.9895     0.007453         2.94
IVF-RaBitQ-nl547-np33-rf20 (query)                2_949.91       542.44     3_492.35       0.9994     0.000393         2.94
IVF-RaBitQ-nl547 (self)                           2_949.91     5_331.75     8_281.66       0.9993     0.000388         2.94
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With higher dimensionality

RaBitQ particularly shines with higher dimensionality in the data.

<details>
<summary><b>RaBitQ - Euclidean (Gaussian - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   14.66    17_148.68    17_163.34       1.0000     0.000000        73.24
Exhaustive (self)                                    14.66   172_852.88   172_867.55       1.0000     0.000000        73.24
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                     17_657.19     1_846.10    19_503.29       0.3640   349.112050         4.74
ExhaustiveRaBitQ-rf5 (query)                     17_657.19     1_917.72    19_574.91       0.7185     3.892803         4.74
ExhaustiveRaBitQ-rf10 (query)                    17_657.19     2_020.51    19_677.71       0.8547     1.509975         4.74
ExhaustiveRaBitQ-rf20 (query)                    17_657.19     2_179.55    19_836.74       0.9452     0.460724         4.74
ExhaustiveRaBitQ (self)                          17_657.19    20_179.30    37_836.49       0.8556     1.514829         4.74
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                10_962.81       518.83    11_481.64       0.3588   349.079217         4.78
IVF-RaBitQ-nl273-np16-rf0 (query)                10_962.81       626.18    11_589.00       0.3617   349.096834         4.78
IVF-RaBitQ-nl273-np23-rf0 (query)                10_962.81       881.10    11_843.91       0.3630   349.108998         4.78
IVF-RaBitQ-nl273-np13-rf5 (query)                10_962.81       583.49    11_546.30       0.6970     4.366886         4.78
IVF-RaBitQ-nl273-np13-rf10 (query)               10_962.81       647.43    11_610.24       0.8214     2.045318         4.78
IVF-RaBitQ-nl273-np13-rf20 (query)               10_962.81       763.02    11_725.83       0.8985     1.067900         4.78
IVF-RaBitQ-nl273-np16-rf5 (query)                10_962.81       707.68    11_670.49       0.7109     4.059120         4.78
IVF-RaBitQ-nl273-np16-rf10 (query)               10_962.81       761.39    11_724.20       0.8427     1.697651         4.78
IVF-RaBitQ-nl273-np16-rf20 (query)               10_962.81       884.16    11_846.97       0.9272     0.680650         4.78
IVF-RaBitQ-nl273-np23-rf5 (query)                10_962.81       954.18    11_916.99       0.7190     3.885257         4.78
IVF-RaBitQ-nl273-np23-rf10 (query)               10_962.81     1_023.46    11_986.27       0.8557     1.500937         4.78
IVF-RaBitQ-nl273-np23-rf20 (query)               10_962.81     1_149.09    12_111.91       0.9453     0.461556         4.78
IVF-RaBitQ-nl273 (self)                          10_962.81    11_771.88    22_734.70       0.9459     0.461210         4.78
IVF-RaBitQ-nl387-np19-rf0 (query)                15_309.36       744.55    16_053.91       0.3614   349.088030         4.83
IVF-RaBitQ-nl387-np27-rf0 (query)                15_309.36       918.83    16_228.18       0.3638   349.108435         4.83
IVF-RaBitQ-nl387-np19-rf5 (query)                15_309.36       738.89    16_048.25       0.7076     4.173263         4.83
IVF-RaBitQ-nl387-np19-rf10 (query)               15_309.36       869.34    16_178.70       0.8336     1.885756         4.83
IVF-RaBitQ-nl387-np19-rf20 (query)               15_309.36       994.65    16_304.01       0.9144     0.883033         4.83
IVF-RaBitQ-nl387-np27-rf5 (query)                15_309.36     1_011.20    16_320.56       0.7214     3.859034         4.83
IVF-RaBitQ-nl387-np27-rf10 (query)               15_309.36     1_076.33    16_385.69       0.8551     1.518658         4.83
IVF-RaBitQ-nl387-np27-rf20 (query)               15_309.36     1_145.93    16_455.29       0.9447     0.471678         4.83
IVF-RaBitQ-nl387 (self)                          15_309.36    11_545.28    26_854.64       0.9440     0.496418         4.83
IVF-RaBitQ-nl547-np23-rf0 (query)                21_325.26       664.63    21_989.90       0.3605   349.077705         4.91
IVF-RaBitQ-nl547-np27-rf0 (query)                21_325.26       859.67    22_184.93       0.3623   349.092475         4.91
IVF-RaBitQ-nl547-np33-rf0 (query)                21_325.26       918.86    22_244.13       0.3640   349.103920         4.91
IVF-RaBitQ-nl547-np23-rf5 (query)                21_325.26       734.33    22_059.60       0.7002     4.297148         4.91
IVF-RaBitQ-nl547-np23-rf10 (query)               21_325.26       789.84    22_115.10       0.8236     2.020674         4.91
IVF-RaBitQ-nl547-np23-rf20 (query)               21_325.26       901.38    22_226.64       0.9017     1.026832         4.91
IVF-RaBitQ-nl547-np27-rf5 (query)                21_325.26       840.48    22_165.74       0.7108     4.041964         4.91
IVF-RaBitQ-nl547-np27-rf10 (query)               21_325.26       911.52    22_236.78       0.8400     1.726002         4.91
IVF-RaBitQ-nl547-np27-rf20 (query)               21_325.26     1_006.42    22_331.69       0.9242     0.707000         4.91
IVF-RaBitQ-nl547-np33-rf5 (query)                21_325.26       991.48    22_316.75       0.7185     3.897222         4.91
IVF-RaBitQ-nl547-np33-rf10 (query)               21_325.26     1_060.91    22_386.17       0.8524     1.552195         4.91
IVF-RaBitQ-nl547-np33-rf20 (query)               21_325.26     1_166.81    22_492.07       0.9419     0.506911         4.91
IVF-RaBitQ-nl547 (self)                          21_325.26    11_837.28    33_162.55       0.9415     0.507018         4.91
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (Correlated - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.39    17_816.64    17_832.04       1.0000     0.000000        73.24
Exhaustive (self)                                    15.39   176_519.54   176_534.94       1.0000     0.000000        73.24
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                     18_285.24     1_770.34    20_055.57       0.5388    88.769526         4.74
ExhaustiveRaBitQ-rf5 (query)                     18_285.24     1_834.77    20_120.01       0.9252     0.242534         4.74
ExhaustiveRaBitQ-rf10 (query)                    18_285.24     1_912.22    20_197.46       0.9837     0.038293         4.74
ExhaustiveRaBitQ-rf20 (query)                    18_285.24     2_067.85    20_353.09       0.9982     0.003165         4.74
ExhaustiveRaBitQ (self)                          18_285.24    19_165.43    37_450.67       0.9837     0.038454         4.74
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                10_898.61       575.68    11_474.29       0.5445    88.760207         4.78
IVF-RaBitQ-nl273-np16-rf0 (query)                10_898.61       637.45    11_536.06       0.5443    88.762194         4.78
IVF-RaBitQ-nl273-np23-rf0 (query)                10_898.61       896.99    11_795.60       0.5441    88.762984         4.78
IVF-RaBitQ-nl273-np13-rf5 (query)                10_898.61       601.86    11_500.46       0.9267     0.258481         4.78
IVF-RaBitQ-nl273-np13-rf10 (query)               10_898.61       660.10    11_558.71       0.9797     0.061728         4.78
IVF-RaBitQ-nl273-np13-rf20 (query)               10_898.61       789.11    11_687.72       0.9920     0.028015         4.78
IVF-RaBitQ-nl273-np16-rf5 (query)                10_898.61       704.07    11_602.67       0.9294     0.243781         4.78
IVF-RaBitQ-nl273-np16-rf10 (query)               10_898.61       776.16    11_674.77       0.9839     0.044223         4.78
IVF-RaBitQ-nl273-np16-rf20 (query)               10_898.61       882.66    11_781.27       0.9969     0.009773         4.78
IVF-RaBitQ-nl273-np23-rf5 (query)                10_898.61       963.03    11_861.63       0.9301     0.238563         4.78
IVF-RaBitQ-nl273-np23-rf10 (query)               10_898.61     1_040.30    11_938.90       0.9852     0.037819         4.78
IVF-RaBitQ-nl273-np23-rf20 (query)               10_898.61     1_147.17    12_045.78       0.9984     0.003020         4.78
IVF-RaBitQ-nl273 (self)                          10_898.61    12_012.64    22_911.25       0.9985     0.002883         4.78
IVF-RaBitQ-nl387-np19-rf0 (query)                15_754.13       649.45    16_403.59       0.5474    88.755526         4.83
IVF-RaBitQ-nl387-np27-rf0 (query)                15_754.13       919.93    16_674.06       0.5471    88.757288         4.83
IVF-RaBitQ-nl387-np19-rf5 (query)                15_754.13       692.05    16_446.18       0.9299     0.242751         4.83
IVF-RaBitQ-nl387-np19-rf10 (query)               15_754.13       750.39    16_504.53       0.9824     0.050003         4.83
IVF-RaBitQ-nl387-np19-rf20 (query)               15_754.13       923.08    16_677.21       0.9945     0.018225         4.83
IVF-RaBitQ-nl387-np27-rf5 (query)                15_754.13       940.84    16_694.97       0.9319     0.231169         4.83
IVF-RaBitQ-nl387-np27-rf10 (query)               15_754.13       999.86    16_753.99       0.9858     0.036282         4.83
IVF-RaBitQ-nl387-np27-rf20 (query)               15_754.13     1_172.20    16_926.33       0.9985     0.003058         4.83
IVF-RaBitQ-nl387 (self)                          15_754.13    11_674.11    27_428.24       0.9985     0.003024         4.83
IVF-RaBitQ-nl547-np23-rf0 (query)                20_773.68       683.63    21_457.31       0.5509    88.748270         4.91
IVF-RaBitQ-nl547-np27-rf0 (query)                20_773.68       770.32    21_544.00       0.5509    88.750175         4.91
IVF-RaBitQ-nl547-np33-rf0 (query)                20_773.68       927.18    21_700.86       0.5508    88.751214         4.91
IVF-RaBitQ-nl547-np23-rf5 (query)                20_773.68       726.71    21_500.39       0.9303     0.234283         4.91
IVF-RaBitQ-nl547-np23-rf10 (query)               20_773.68       782.17    21_555.85       0.9806     0.054133         4.91
IVF-RaBitQ-nl547-np23-rf20 (query)               20_773.68       910.72    21_684.41       0.9917     0.025287         4.91
IVF-RaBitQ-nl547-np27-rf5 (query)                20_773.68       848.72    21_622.40       0.9330     0.222092         4.91
IVF-RaBitQ-nl547-np27-rf10 (query)               20_773.68       913.76    21_687.44       0.9845     0.039766         4.91
IVF-RaBitQ-nl547-np27-rf20 (query)               20_773.68     1_028.84    21_802.52       0.9962     0.009999         4.91
IVF-RaBitQ-nl547-np33-rf5 (query)                20_773.68     1_027.28    21_800.96       0.9342     0.216642         4.91
IVF-RaBitQ-nl547-np33-rf10 (query)               20_773.68     1_073.85    21_847.53       0.9865     0.033022         4.91
IVF-RaBitQ-nl547-np33-rf20 (query)               20_773.68     1_191.24    21_964.92       0.9985     0.002712         4.91
IVF-RaBitQ-nl547 (self)                          20_773.68    11_872.24    32_645.92       0.9983     0.003509         4.91
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>RaBitQ - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D - IVF-RaBitQ
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.03    18_541.61    18_556.64       1.0000     0.000000        73.24
Exhaustive (self)                                    15.03   180_609.47   180_624.50       1.0000     0.000000        73.24
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveRaBitQ-rf0 (query)                     18_645.27     1_768.08    20_413.36       0.7026    73.760190         4.74
ExhaustiveRaBitQ-rf5 (query)                     18_645.27     1_788.65    20_433.93       0.9952     0.017496         4.74
ExhaustiveRaBitQ-rf10 (query)                    18_645.27     1_861.63    20_506.91       0.9999     0.000471         4.74
ExhaustiveRaBitQ-rf20 (query)                    18_645.27     2_022.76    20_668.04       1.0000     0.000000         4.74
ExhaustiveRaBitQ (self)                          18_645.27    21_176.72    39_822.00       0.9999     0.000423         4.74
---------------------------------------------------------------------------------------------------------------------------
IVF-RaBitQ-nl273-np13-rf0 (query)                11_644.15       546.91    12_191.06       0.7066    73.750566         4.78
IVF-RaBitQ-nl273-np16-rf0 (query)                11_644.15       660.74    12_304.89       0.7066    73.750743         4.78
IVF-RaBitQ-nl273-np23-rf0 (query)                11_644.15       929.94    12_574.09       0.7066    73.750749         4.78
IVF-RaBitQ-nl273-np13-rf5 (query)                11_644.15       632.96    12_277.11       0.9954     0.017101         4.78
IVF-RaBitQ-nl273-np13-rf10 (query)               11_644.15       669.11    12_313.26       0.9996     0.002368         4.78
IVF-RaBitQ-nl273-np13-rf20 (query)               11_644.15       738.51    12_382.66       0.9997     0.002034         4.78
IVF-RaBitQ-nl273-np16-rf5 (query)                11_644.15       720.58    12_364.73       0.9956     0.015255         4.78
IVF-RaBitQ-nl273-np16-rf10 (query)               11_644.15       874.24    12_518.39       0.9999     0.000515         4.78
IVF-RaBitQ-nl273-np16-rf20 (query)               11_644.15       945.09    12_589.24       1.0000     0.000182         4.78
IVF-RaBitQ-nl273-np23-rf5 (query)                11_644.15       971.50    12_615.65       0.9957     0.015072         4.78
IVF-RaBitQ-nl273-np23-rf10 (query)               11_644.15     1_038.98    12_683.13       0.9999     0.000333         4.78
IVF-RaBitQ-nl273-np23-rf20 (query)               11_644.15     1_165.09    12_809.24       1.0000     0.000000         4.78
IVF-RaBitQ-nl273 (self)                          11_644.15    11_963.56    23_607.72       1.0000     0.000001         4.78
IVF-RaBitQ-nl387-np19-rf0 (query)                16_107.22       656.27    16_763.50       0.7122    73.742382         4.83
IVF-RaBitQ-nl387-np27-rf0 (query)                16_107.22     1_034.16    17_141.39       0.7121    73.742402         4.83
IVF-RaBitQ-nl387-np19-rf5 (query)                16_107.22     1_107.35    17_214.57       0.9959     0.014181         4.83
IVF-RaBitQ-nl387-np19-rf10 (query)               16_107.22       859.45    16_966.67       0.9999     0.000347         4.83
IVF-RaBitQ-nl387-np19-rf20 (query)               16_107.22     1_072.14    17_179.37       1.0000     0.000111         4.83
IVF-RaBitQ-nl387-np27-rf5 (query)                16_107.22       980.81    17_088.03       0.9959     0.014082         4.83
IVF-RaBitQ-nl387-np27-rf10 (query)               16_107.22     1_029.90    17_137.12       0.9999     0.000236         4.83
IVF-RaBitQ-nl387-np27-rf20 (query)               16_107.22     1_135.23    17_242.45       1.0000     0.000000         4.83
IVF-RaBitQ-nl387 (self)                          16_107.22    10_922.43    27_029.66       1.0000     0.000004         4.83
IVF-RaBitQ-nl547-np23-rf0 (query)                21_613.27       662.24    22_275.51       0.7164    73.731258         4.91
IVF-RaBitQ-nl547-np27-rf0 (query)                21_613.27       757.95    22_371.22       0.7164    73.731347         4.91
IVF-RaBitQ-nl547-np33-rf0 (query)                21_613.27       905.62    22_518.90       0.7164    73.731358         4.91
IVF-RaBitQ-nl547-np23-rf5 (query)                21_613.27       721.39    22_334.67       0.9962     0.013858         4.91
IVF-RaBitQ-nl547-np23-rf10 (query)               21_613.27       778.86    22_392.13       0.9998     0.000942         4.91
IVF-RaBitQ-nl547-np23-rf20 (query)               21_613.27       892.07    22_505.35       0.9999     0.000615         4.91
IVF-RaBitQ-nl547-np27-rf5 (query)                21_613.27       820.90    22_434.18       0.9963     0.013368         4.91
IVF-RaBitQ-nl547-np27-rf10 (query)               21_613.27       880.73    22_494.00       0.9999     0.000388         4.91
IVF-RaBitQ-nl547-np27-rf20 (query)               21_613.27       987.81    22_601.08       1.0000     0.000062         4.91
IVF-RaBitQ-nl547-np33-rf5 (query)                21_613.27       972.51    22_585.79       0.9963     0.013307         4.91
IVF-RaBitQ-nl547-np33-rf10 (query)               21_613.27     1_031.65    22_644.93       0.9999     0.000326         4.91
IVF-RaBitQ-nl547-np33-rf20 (query)               21_613.27     1_140.44    22_753.72       1.0000     0.000000         4.91
IVF-RaBitQ-nl547 (self)                          21_613.27    11_429.69    33_042.97       1.0000     0.000001         4.91
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls.

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*