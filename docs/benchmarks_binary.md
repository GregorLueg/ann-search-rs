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

<details>
<summary><b>Binary - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.14     2_532.91     2_536.05       1.0000     0.000000        18.31
Exhaustive (self)                                     3.14    25_457.81    25_460.95       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)         1_265.42     1_015.71     2_281.13       0.2031          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)           1_265.42     1_068.67     2_334.09       0.4176     3.722044         4.61
ExhaustiveBinary-256-random-rf10 (query)          1_265.42     1_144.75     2_410.17       0.5464     2.069161         4.61
ExhaustiveBinary-256-random-rf20 (query)          1_265.42     1_257.08     2_522.49       0.6860     1.049073         4.61
ExhaustiveBinary-256-random (self)                1_265.42    12_212.96    13_478.38       0.5461     2.056589         4.61
ExhaustiveBinary-256-itq_no_rr (query)           13_009.23       979.51    13_988.75       0.1848          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)             13_009.23     1_058.83    14_068.06       0.3784     4.301141         4.61
ExhaustiveBinary-256-itq-rf10 (query)            13_009.23     1_096.23    14_105.47       0.4999     2.462130         4.61
ExhaustiveBinary-256-itq-rf20 (query)            13_009.23     1_152.91    14_162.14       0.6378     1.293333         4.61
ExhaustiveBinary-256-itq (self)                  13_009.23    10_932.95    23_942.19       0.4993     2.443967         4.61
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)         2_480.30     1_492.97     3_973.27       0.2906          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)           2_480.30     1_531.97     4_012.27       0.5836     1.792334         9.22
ExhaustiveBinary-512-random-rf10 (query)          2_480.30     1_641.45     4_121.75       0.7240     0.832332         9.22
ExhaustiveBinary-512-random-rf20 (query)          2_480.30     1_729.47     4_209.77       0.8440     0.344416         9.22
ExhaustiveBinary-512-random (self)                2_480.30    16_203.41    18_683.71       0.7223     0.828915         9.22
ExhaustiveBinary-512-itq_no_rr (query)           10_907.71     1_704.01    12_611.72       0.2820          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)             10_907.71     1_773.13    12_680.84       0.5671     1.854040         9.22
ExhaustiveBinary-512-itq-rf10 (query)            10_907.71     1_814.79    12_722.50       0.7050     0.880641         9.22
ExhaustiveBinary-512-itq-rf20 (query)            10_907.71     1_926.04    12_833.75       0.8277     0.369415         9.22
ExhaustiveBinary-512-itq (self)                  10_907.71    18_230.87    29_138.59       0.7046     0.877496         9.22
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)      2_761.89        92.00     2_853.88       0.2064          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)      2_761.89        96.46     2_858.34       0.2053          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)      2_761.89       122.61     2_884.50       0.2044          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)      2_761.89       120.63     2_882.52       0.4238     3.637807         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)     2_761.89       141.20     2_903.08       0.5532     2.020523         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)     2_761.89       201.54     2_963.42       0.6927     1.020427         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)      2_761.89       151.55     2_913.43       0.4222     3.662764         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)     2_761.89       181.42     2_943.31       0.5515     2.033556         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)     2_761.89       355.96     3_117.84       0.6913     1.025018         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)      2_761.89       209.87     2_971.75       0.4198     3.692510         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)     2_761.89       228.65     2_990.53       0.5484     2.057232         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)     2_761.89       335.58     3_097.46       0.6879     1.040725         5.79
IVF-Binary-256-nl273-random (self)                2_761.89     2_155.80     4_917.69       0.5511     2.018621         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)      3_432.81        94.44     3_527.25       0.2060          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)      3_432.81       117.47     3_550.28       0.2047          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)      3_432.81       130.14     3_562.95       0.4237     3.636098         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)     3_432.81       160.02     3_592.83       0.5534     2.013063         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)     3_432.81       217.75     3_650.56       0.6934     1.014705         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)      3_432.81       144.59     3_577.40       0.4203     3.684091         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)     3_432.81       186.76     3_619.57       0.5496     2.044464         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)     3_432.81       253.22     3_686.03       0.6889     1.035879         5.80
IVF-Binary-256-nl387-random (self)                3_432.81     1_581.83     5_014.64       0.5532     1.998215         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)      4_247.80        87.09     4_334.89       0.2077          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)      4_247.80        95.67     4_343.47       0.2063          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)      4_247.80       110.32     4_358.12       0.2051          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)      4_247.80       116.77     4_364.57       0.4267     3.592322         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)     4_247.80       147.96     4_395.76       0.5577     1.983086         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)     4_247.80       212.15     4_459.95       0.6978     0.993692         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)      4_247.80       131.27     4_379.07       0.4239     3.627604         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)     4_247.80       156.66     4_404.46       0.5539     2.008636         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)     4_247.80       219.76     4_467.56       0.6943     1.006723         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)      4_247.80       140.87     4_388.67       0.4211     3.671793         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)     4_247.80       176.59     4_424.40       0.5506     2.035867         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)     4_247.80       231.21     4_479.01       0.6901     1.028793         5.82
IVF-Binary-256-nl547-random (self)                4_247.80     1_493.71     5_741.52       0.5570     1.967723         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)        14_925.82        82.42    15_008.24       0.1881          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)        14_925.82       100.49    15_026.32       0.1868          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)        14_925.82       134.10    15_059.92       0.1857          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)        14_925.82       121.68    15_047.51       0.3855     4.193849         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)       14_925.82       156.09    15_081.91       0.5090     2.387001         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)       14_925.82       208.51    15_134.33       0.6465     1.254028         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)        14_925.82       136.73    15_062.55       0.3832     4.227451         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)       14_925.82       163.13    15_088.95       0.5065     2.407819         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)       14_925.82       216.96    15_142.78       0.6440     1.262045         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)        14_925.82       147.57    15_073.40       0.3804     4.270226         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)       14_925.82       177.17    15_102.99       0.5029     2.437981         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)       14_925.82       234.92    15_160.74       0.6401     1.281961         5.79
IVF-Binary-256-nl273-itq (self)                  14_925.82     1_470.55    16_396.38       0.5054     2.393919         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)        11_026.30        96.06    11_122.37       0.1878          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)        11_026.30       119.44    11_145.74       0.1863          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)        11_026.30       126.21    11_152.51       0.3853     4.183095         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)       11_026.30       158.17    11_184.47       0.5089     2.378886         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)       11_026.30       207.65    11_233.96       0.6465     1.248463         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)        11_026.30       148.12    11_174.43       0.3813     4.251386         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)       11_026.30       182.35    11_208.66       0.5040     2.422572         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)       11_026.30       245.22    11_271.53       0.6414     1.270337         5.80
IVF-Binary-256-nl387-itq (self)                  11_026.30     1_568.38    12_594.68       0.5077     2.366617         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)        13_231.67        88.53    13_320.21       0.1886          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)        13_231.67       100.67    13_332.34       0.1874          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)        13_231.67       118.65    13_350.32       0.1861          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)        13_231.67       126.71    13_358.38       0.3890     4.131423         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)       13_231.67       158.32    13_389.99       0.5133     2.339310         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)       13_231.67       193.31    13_424.98       0.6523     1.218765         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)        13_231.67       117.29    13_348.96       0.3857     4.182464         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)       13_231.67       145.10    13_376.77       0.5094     2.372792         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)       13_231.67       197.85    13_429.53       0.6473     1.238890         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)        13_231.67       131.02    13_362.69       0.3826     4.238962         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)       13_231.67       158.16    13_389.84       0.5053     2.412698         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)       13_231.67       210.08    13_441.76       0.6429     1.264457         5.82
IVF-Binary-256-nl547-itq (self)                  13_231.67     1_416.44    14_648.11       0.5122     2.326728         5.82
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)      3_999.38       170.36     4_169.74       0.2931          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)      3_999.38       202.63     4_202.01       0.2924          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)      3_999.38       259.69     4_259.07       0.2915          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)      3_999.38       223.48     4_222.85       0.5865     1.777906        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)     3_999.38       261.32     4_260.69       0.7256     0.834132        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)     3_999.38       343.92     4_343.29       0.8433     0.358346        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)      3_999.38       259.84     4_259.22       0.5863     1.774488        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)     3_999.38       312.54     4_311.91       0.7262     0.823357        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)     3_999.38       370.59     4_369.96       0.8455     0.341292        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)      3_999.38       292.19     4_291.57       0.5848     1.784245        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)     3_999.38       366.58     4_365.95       0.7249     0.828433        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)     3_999.38       441.72     4_441.09       0.8447     0.342598        10.40
IVF-Binary-512-nl273-random (self)                3_999.38     2_950.59     6_949.97       0.7247     0.819999        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)      4_537.05       175.66     4_712.71       0.2926          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)      4_537.05       237.21     4_774.26       0.2917          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)      4_537.05       225.95     4_763.00       0.5872     1.769163        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)     4_537.05       270.38     4_807.43       0.7264     0.828521        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)     4_537.05       334.49     4_871.54       0.8448     0.354339        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)      4_537.05       270.70     4_807.75       0.5853     1.779139        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)     4_537.05       284.79     4_821.84       0.7253     0.826501        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)     4_537.05       347.58     4_884.64       0.8452     0.340766        10.41
IVF-Binary-512-nl387-random (self)                4_537.05     2_495.85     7_032.90       0.7248     0.827306        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)      5_393.36       162.69     5_556.05       0.2937          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)      5_393.36       191.29     5_584.65       0.2930          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)      5_393.36       207.42     5_600.78       0.2922          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)      5_393.36       203.67     5_597.03       0.5886     1.763029        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)     5_393.36       253.61     5_646.98       0.7282     0.824085        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)     5_393.36       308.01     5_701.37       0.8461     0.353746        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)      5_393.36       230.71     5_624.07       0.5871     1.766634        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)     5_393.36       269.36     5_662.72       0.7272     0.820294        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)     5_393.36       344.42     5_737.78       0.8464     0.341244        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)      5_393.36       264.40     5_657.76       0.5855     1.778317        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)     5_393.36       298.97     5_692.33       0.7259     0.823829        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)     5_393.36       377.91     5_771.27       0.8456     0.339064        10.43
IVF-Binary-512-nl547-random (self)                5_393.36     2_452.31     7_845.67       0.7268     0.818373        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)        11_654.25       166.82    11_821.06       0.2840          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)        11_654.25       193.80    11_848.05       0.2836          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)        11_654.25       254.14    11_908.39       0.2829          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)        11_654.25       208.87    11_863.11       0.5701     1.835829        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)       11_654.25       251.67    11_905.92       0.7075     0.879350        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)       11_654.25       318.91    11_973.16       0.8282     0.380931        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)        11_654.25       226.49    11_880.73       0.5697     1.835319        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)       11_654.25       241.26    11_895.51       0.7078     0.870826        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)       11_654.25       319.07    11_973.31       0.8297     0.366581        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)        11_654.25       289.36    11_943.61       0.5685     1.844737        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)       11_654.25       364.17    12_018.42       0.7061     0.877200        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)       11_654.25       452.62    12_106.87       0.8286     0.367581        10.40
IVF-Binary-512-nl273-itq (self)                  11_654.25     2_785.43    14_439.68       0.7072     0.867842        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)        12_538.12       154.25    12_692.37       0.2843          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)        12_538.12       199.36    12_737.49       0.2831          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)        12_538.12       193.85    12_731.97       0.5708     1.829482        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)       12_538.12       241.73    12_779.85       0.7083     0.876465        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)       12_538.12       335.02    12_873.14       0.8293     0.377665        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)        12_538.12       280.44    12_818.57       0.5686     1.841879        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)       12_538.12       314.61    12_852.73       0.7069     0.873721        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)       12_538.12       408.30    12_946.42       0.8288     0.366298        10.41
IVF-Binary-512-nl387-itq (self)                  12_538.12     2_430.76    14_968.88       0.7073     0.874000        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)        13_221.26       162.72    13_383.98       0.2853          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)        13_221.26       176.52    13_397.78       0.2843          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)        13_221.26       199.33    13_420.59       0.2833          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)        13_221.26       208.36    13_429.62       0.5728     1.814255        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)       13_221.26       242.81    13_464.06       0.7103     0.865519        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)       13_221.26       306.50    13_527.75       0.8313     0.375421        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)        13_221.26       226.01    13_447.27       0.5710     1.823361        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)       13_221.26       260.82    13_482.08       0.7092     0.864315        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)       13_221.26       293.44    13_514.70       0.8310     0.365453        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)        13_221.26       214.00    13_435.26       0.5692     1.837419        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)       13_221.26       297.53    13_518.79       0.7073     0.870297        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)       13_221.26       386.97    13_608.23       0.8296     0.365107        10.43
IVF-Binary-512-nl547-itq (self)                  13_221.26     2_401.16    15_622.42       0.7095     0.864695        10.43
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Cosine (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    5.23     2_746.39     2_751.62       1.0000     0.000000        18.88
Exhaustive (self)                                     5.23    25_755.24    25_760.47       1.0000     0.000000        18.88
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)         1_256.33     1_006.08     2_262.41       0.2159          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)           1_256.33     1_052.81     2_309.14       0.4401     0.002523         4.61
ExhaustiveBinary-256-random-rf10 (query)          1_256.33     1_318.90     2_575.23       0.5704     0.001386         4.61
ExhaustiveBinary-256-random-rf20 (query)          1_256.33     1_292.91     2_549.24       0.7082     0.000695         4.61
ExhaustiveBinary-256-random (self)                1_256.33    11_833.68    13_090.01       0.5696     0.001382         4.61
ExhaustiveBinary-256-itq_no_rr (query)            9_357.14     1_042.51    10_399.65       0.1944          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)              9_357.14     1_101.72    10_458.86       0.3979     0.002980         4.61
ExhaustiveBinary-256-itq-rf10 (query)             9_357.14     1_307.50    10_664.64       0.5222     0.001701         4.61
ExhaustiveBinary-256-itq-rf20 (query)             9_357.14     1_349.98    10_707.12       0.6574     0.000897         4.61
ExhaustiveBinary-256-itq (self)                   9_357.14    11_906.07    21_263.21       0.5203     0.001696         4.61
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)         2_538.81     1_693.11     4_231.92       0.3136          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)           2_538.81     1_790.67     4_329.48       0.6181     0.001103         9.22
ExhaustiveBinary-512-random-rf10 (query)          2_538.81     1_735.17     4_273.98       0.7540     0.000502         9.22
ExhaustiveBinary-512-random-rf20 (query)          2_538.81     1_719.26     4_258.07       0.8651     0.000202         9.22
ExhaustiveBinary-512-random (self)                2_538.81    16_834.21    19_373.02       0.7519     0.000503         9.22
ExhaustiveBinary-512-itq_no_rr (query)           10_657.79     1_840.46    12_498.25       0.3028          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)             10_657.79     1_810.66    12_468.45       0.5982     0.001185         9.22
ExhaustiveBinary-512-itq-rf10 (query)            10_657.79     1_658.78    12_316.57       0.7328     0.000558         9.22
ExhaustiveBinary-512-itq-rf20 (query)            10_657.79     1_908.46    12_566.25       0.8481     0.000232         9.22
ExhaustiveBinary-512-itq (self)                  10_657.79    17_273.54    27_931.33       0.7318     0.000555         9.22
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)      2_778.49        89.40     2_867.89       0.2191          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)      2_778.49       111.42     2_889.91       0.2178          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)      2_778.49       147.79     2_926.28       0.2169          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)      2_778.49       133.54     2_912.03       0.4458     0.002468         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)     2_778.49       164.24     2_942.73       0.5771     0.001353         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)     2_778.49       229.96     3_008.45       0.7143     0.000677         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)      2_778.49       158.22     2_936.71       0.4441     0.002487         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)     2_778.49       156.90     2_935.39       0.5757     0.001360         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)     2_778.49       271.74     3_050.23       0.7129     0.000679         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)      2_778.49       184.22     2_962.70       0.4420     0.002507         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)     2_778.49       188.20     2_966.69       0.5729     0.001375         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)     2_778.49       279.43     3_057.92       0.7100     0.000689         5.79
IVF-Binary-256-nl273-random (self)                2_778.49     1_521.41     4_299.90       0.5746     0.001354         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)      3_263.09        85.51     3_348.59       0.2187          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-random (query)      3_263.09       113.67     3_376.76       0.2171          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-random (query)      3_263.09       123.37     3_386.46       0.4458     0.002463         5.81
IVF-Binary-256-nl387-np19-rf10-random (query)     3_263.09       170.01     3_433.10       0.5776     0.001347         5.81
IVF-Binary-256-nl387-np19-rf20-random (query)     3_263.09       227.89     3_490.98       0.7150     0.000670         5.81
IVF-Binary-256-nl387-np27-rf5-random (query)      3_263.09       167.06     3_430.15       0.4423     0.002500         5.81
IVF-Binary-256-nl387-np27-rf10-random (query)     3_263.09       197.56     3_460.64       0.5736     0.001368         5.81
IVF-Binary-256-nl387-np27-rf20-random (query)     3_263.09       255.04     3_518.13       0.7108     0.000685         5.81
IVF-Binary-256-nl387-random (self)                3_263.09     1_498.04     4_761.12       0.5768     0.001341         5.81
IVF-Binary-256-nl547-np23-rf0-random (query)      4_379.55        88.01     4_467.56       0.2198          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-random (query)      4_379.55        95.16     4_474.71       0.2184          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-random (query)      4_379.55       111.68     4_491.23       0.2175          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-random (query)      4_379.55       114.41     4_493.96       0.4485     0.002437         5.83
IVF-Binary-256-nl547-np23-rf10-random (query)     4_379.55       135.32     4_514.87       0.5811     0.001326         5.83
IVF-Binary-256-nl547-np23-rf20-random (query)     4_379.55       181.83     4_561.38       0.7192     0.000656         5.83
IVF-Binary-256-nl547-np27-rf5-random (query)      4_379.55       126.63     4_506.18       0.4459     0.002463         5.83
IVF-Binary-256-nl547-np27-rf10-random (query)     4_379.55       156.63     4_536.18       0.5778     0.001344         5.83
IVF-Binary-256-nl547-np27-rf20-random (query)     4_379.55       218.00     4_597.55       0.7160     0.000665         5.83
IVF-Binary-256-nl547-np33-rf5-random (query)      4_379.55       153.10     4_532.65       0.4432     0.002491         5.83
IVF-Binary-256-nl547-np33-rf10-random (query)     4_379.55       182.82     4_562.37       0.5744     0.001364         5.83
IVF-Binary-256-nl547-np33-rf20-random (query)     4_379.55       231.36     4_610.91       0.7120     0.000680         5.83
IVF-Binary-256-nl547-random (self)                4_379.55     1_435.44     5_814.99       0.5805     0.001321         5.83
IVF-Binary-256-nl273-np13-rf0-itq (query)        14_674.65        82.31    14_756.96       0.1978          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)        14_674.65       112.72    14_787.36       0.1967          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)        14_674.65       137.64    14_812.29       0.1955          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)        14_674.65       126.15    14_800.80       0.4054     0.002902         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)       14_674.65       156.59    14_831.24       0.5309     0.001647         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)       14_674.65       207.05    14_881.70       0.6657     0.000869         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)        14_674.65       139.26    14_813.90       0.4031     0.002927         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)       14_674.65       255.94    14_930.59       0.5284     0.001663         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)       14_674.65       201.78    14_876.43       0.6635     0.000875         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)        14_674.65       150.91    14_825.56       0.4002     0.002960         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)       14_674.65       182.22    14_856.87       0.5247     0.001686         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)       14_674.65       276.55    14_951.19       0.6594     0.000890         5.79
IVF-Binary-256-nl273-itq (self)                  14_674.65     1_575.30    16_249.94       0.5261     0.001659         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)        11_770.13        81.70    11_851.82       0.1976          NaN         5.81
IVF-Binary-256-nl387-np27-rf0-itq (query)        11_770.13       102.40    11_872.53       0.1958          NaN         5.81
IVF-Binary-256-nl387-np19-rf5-itq (query)        11_770.13       110.48    11_880.61       0.4051     0.002894         5.81
IVF-Binary-256-nl387-np19-rf10-itq (query)       11_770.13       135.47    11_905.59       0.5310     0.001639         5.81
IVF-Binary-256-nl387-np19-rf20-itq (query)       11_770.13       184.75    11_954.88       0.6663     0.000861         5.81
IVF-Binary-256-nl387-np27-rf5-itq (query)        11_770.13       131.45    11_901.58       0.4007     0.002945         5.81
IVF-Binary-256-nl387-np27-rf10-itq (query)       11_770.13       160.14    11_930.26       0.5257     0.001676         5.81
IVF-Binary-256-nl387-np27-rf20-itq (query)       11_770.13       212.73    11_982.85       0.6603     0.000884         5.81
IVF-Binary-256-nl387-itq (self)                  11_770.13     1_365.72    13_135.85       0.5289     0.001637         5.81
IVF-Binary-256-nl547-np23-rf0-itq (query)        12_001.64        80.72    12_082.36       0.1987          NaN         5.83
IVF-Binary-256-nl547-np27-rf0-itq (query)        12_001.64        82.88    12_084.52       0.1974          NaN         5.83
IVF-Binary-256-nl547-np33-rf0-itq (query)        12_001.64       104.04    12_105.68       0.1961          NaN         5.83
IVF-Binary-256-nl547-np23-rf5-itq (query)        12_001.64       131.78    12_133.42       0.4088     0.002857         5.83
IVF-Binary-256-nl547-np23-rf10-itq (query)       12_001.64       152.94    12_154.59       0.5350     0.001613         5.83
IVF-Binary-256-nl547-np23-rf20-itq (query)       12_001.64       177.12    12_178.76       0.6712     0.000843         5.83
IVF-Binary-256-nl547-np27-rf5-itq (query)        12_001.64       113.52    12_115.17       0.4055     0.002894         5.83
IVF-Binary-256-nl547-np27-rf10-itq (query)       12_001.64       151.25    12_152.89       0.5312     0.001638         5.83
IVF-Binary-256-nl547-np27-rf20-itq (query)       12_001.64       218.65    12_220.29       0.6668     0.000858         5.83
IVF-Binary-256-nl547-np33-rf5-itq (query)        12_001.64       131.63    12_133.27       0.4021     0.002933         5.83
IVF-Binary-256-nl547-np33-rf10-itq (query)       12_001.64       154.08    12_155.72       0.5271     0.001666         5.83
IVF-Binary-256-nl547-np33-rf20-itq (query)       12_001.64       245.04    12_246.68       0.6620     0.000878         5.83
IVF-Binary-256-nl547-itq (self)                  12_001.64     1_364.35    13_365.99       0.5328     0.001611         5.83
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)      3_851.24       173.87     4_025.12       0.3155          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)      3_851.24       183.71     4_034.96       0.3151          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)      3_851.24       235.29     4_086.53       0.3144          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)      3_851.24       189.61     4_040.85       0.6203     0.001094        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)     3_851.24       219.00     4_070.25       0.7547     0.000506        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)     3_851.24       292.33     4_143.57       0.8637     0.000214        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)      3_851.24       258.21     4_109.45       0.6204     0.001091        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)     3_851.24       309.85     4_161.09       0.7558     0.000498        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)     3_851.24       378.66     4_229.91       0.8663     0.000202        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)      3_851.24       319.37     4_170.62       0.6191     0.001098        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)     3_851.24       466.25     4_317.49       0.7548     0.000500        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)     3_851.24       441.29     4_292.54       0.8659     0.000201        10.40
IVF-Binary-512-nl273-random (self)                3_851.24     2_893.15     6_744.40       0.7541     0.000497        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)      4_555.04       232.52     4_787.56       0.3156          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)      4_555.04       301.57     4_856.61       0.3145          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)      4_555.04       255.85     4_810.90       0.6209     0.001091        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)     4_555.04       310.15     4_865.19       0.7558     0.000501        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)     4_555.04       366.78     4_921.83       0.8654     0.000209        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)      4_555.04       335.96     4_891.00       0.6194     0.001096        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)     4_555.04       358.90     4_913.94       0.7551     0.000499        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)     4_555.04       385.72     4_940.77       0.8662     0.000200        10.41
IVF-Binary-512-nl387-random (self)                4_555.04     2_369.15     6_924.19       0.7539     0.000502        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)      5_289.70       142.26     5_431.96       0.3165          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-random (query)      5_289.70       196.08     5_485.78       0.3156          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-random (query)      5_289.70       226.71     5_516.41       0.3148          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-random (query)      5_289.70       231.76     5_521.46       0.6226     0.001081        10.44
IVF-Binary-512-nl547-np23-rf10-random (query)     5_289.70       249.45     5_539.15       0.7570     0.000498        10.44
IVF-Binary-512-nl547-np23-rf20-random (query)     5_289.70       267.13     5_556.83       0.8661     0.000209        10.44
IVF-Binary-512-nl547-np27-rf5-random (query)      5_289.70       223.43     5_513.13       0.6215     0.001086        10.44
IVF-Binary-512-nl547-np27-rf10-random (query)     5_289.70       262.77     5_552.47       0.7567     0.000494        10.44
IVF-Binary-512-nl547-np27-rf20-random (query)     5_289.70       284.01     5_573.71       0.8670     0.000201        10.44
IVF-Binary-512-nl547-np33-rf5-random (query)      5_289.70       235.60     5_525.30       0.6197     0.001094        10.44
IVF-Binary-512-nl547-np33-rf10-random (query)     5_289.70       266.17     5_555.87       0.7556     0.000497        10.44
IVF-Binary-512-nl547-np33-rf20-random (query)     5_289.70       327.10     5_616.80       0.8664     0.000200        10.44
IVF-Binary-512-nl547-random (self)                5_289.70     2_169.55     7_459.25       0.7555     0.000498        10.44
IVF-Binary-512-nl273-np13-rf0-itq (query)        12_098.36       203.10    12_301.46       0.3049          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)        12_098.36       222.97    12_321.33       0.3045          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)        12_098.36       321.96    12_420.32       0.3037          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)        12_098.36       242.95    12_341.31       0.6008     0.001175        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)       12_098.36       278.50    12_376.86       0.7348     0.000558        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)       12_098.36       316.98    12_415.34       0.8477     0.000241        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)        12_098.36       303.24    12_401.60       0.6008     0.001171        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)       12_098.36       331.75    12_430.11       0.7354     0.000550        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)       12_098.36       397.89    12_496.25       0.8498     0.000230        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)        12_098.36       298.80    12_397.17       0.5993     0.001178        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)       12_098.36       315.61    12_413.97       0.7339     0.000554        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)       12_098.36       529.22    12_627.58       0.8488     0.000231        10.40
IVF-Binary-512-nl273-itq (self)                  12_098.36     2_949.08    15_047.45       0.7342     0.000548        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)        12_786.71       156.61    12_943.32       0.3053          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)        12_786.71       193.91    12_980.62       0.3041          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)        12_786.71       197.21    12_983.92       0.6014     0.001169        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)       12_786.71       261.24    13_047.94       0.7354     0.000554        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)       12_786.71       317.11    13_103.82       0.8492     0.000236        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)        12_786.71       270.73    13_057.44       0.5996     0.001177        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)       12_786.71       323.22    13_109.93       0.7339     0.000553        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)       12_786.71       467.15    13_253.86       0.8489     0.000230        10.41
IVF-Binary-512-nl387-itq (self)                  12_786.71     2_806.65    15_593.36       0.7341     0.000552        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)        13_543.75       131.71    13_675.46       0.3060          NaN        10.44
IVF-Binary-512-nl547-np27-rf0-itq (query)        13_543.75       149.22    13_692.97       0.3051          NaN        10.44
IVF-Binary-512-nl547-np33-rf0-itq (query)        13_543.75       167.97    13_711.72       0.3042          NaN        10.44
IVF-Binary-512-nl547-np23-rf5-itq (query)        13_543.75       171.92    13_715.67       0.6030     0.001163        10.44
IVF-Binary-512-nl547-np23-rf10-itq (query)       13_543.75       206.59    13_750.35       0.7370     0.000550        10.44
IVF-Binary-512-nl547-np23-rf20-itq (query)       13_543.75       262.88    13_806.63       0.8503     0.000236        10.44
IVF-Binary-512-nl547-np27-rf5-itq (query)        13_543.75       189.44    13_733.19       0.6018     0.001166        10.44
IVF-Binary-512-nl547-np27-rf10-itq (query)       13_543.75       240.29    13_784.04       0.7364     0.000548        10.44
IVF-Binary-512-nl547-np27-rf20-itq (query)       13_543.75       301.03    13_844.79       0.8507     0.000229        10.44
IVF-Binary-512-nl547-np33-rf5-itq (query)        13_543.75       213.92    13_757.67       0.6002     0.001174        10.44
IVF-Binary-512-nl547-np33-rf10-itq (query)       13_543.75       267.85    13_811.60       0.7346     0.000552        10.44
IVF-Binary-512-nl547-np33-rf20-itq (query)       13_543.75       323.30    13_867.05       0.8496     0.000229        10.44
IVF-Binary-512-nl547-itq (self)                  13_543.75     2_087.61    15_631.36       0.7359     0.000548        10.44
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (Correlated)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.03     2_348.55     2_351.58       1.0000     0.000000        18.31
Exhaustive (self)                                     3.03    23_376.38    23_379.40       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)         1_239.90       979.54     2_219.44       0.1352          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)           1_239.90     1_027.27     2_267.16       0.3069     0.751173         4.61
ExhaustiveBinary-256-random-rf10 (query)          1_239.90     1_062.16     2_302.06       0.4237     0.452899         4.61
ExhaustiveBinary-256-random-rf20 (query)          1_239.90     1_167.69     2_407.59       0.5614     0.256662         4.61
ExhaustiveBinary-256-random (self)                1_239.90    10_621.94    11_861.84       0.4294     0.437191         4.61
ExhaustiveBinary-256-itq_no_rr (query)           10_526.26       972.72    11_498.98       0.1177          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)             10_526.26     1_023.84    11_550.10       0.2732     0.869268         4.61
ExhaustiveBinary-256-itq-rf10 (query)            10_526.26     1_057.59    11_583.85       0.3812     0.534178         4.61
ExhaustiveBinary-256-itq-rf20 (query)            10_526.26     1_162.34    11_688.59       0.5162     0.308493         4.61
ExhaustiveBinary-256-itq (self)                  10_526.26    10_515.53    21_041.79       0.3867     0.517541         4.61
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)         2_460.26     1_478.22     3_938.48       0.2248          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)           2_460.26     1_514.64     3_974.90       0.4749     0.355326         9.22
ExhaustiveBinary-512-random-rf10 (query)          2_460.26     1_562.01     4_022.27       0.6150     0.188541         9.22
ExhaustiveBinary-512-random-rf20 (query)          2_460.26     1_663.04     4_123.30       0.7526     0.090979         9.22
ExhaustiveBinary-512-random (self)                2_460.26    15_641.91    18_102.17       0.6185     0.184365         9.22
ExhaustiveBinary-512-itq_no_rr (query)           10_466.90     1_471.12    11_938.02       0.2142          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)             10_466.90     1_526.13    11_993.03       0.4559     0.383350         9.22
ExhaustiveBinary-512-itq-rf10 (query)            10_466.90     1_564.22    12_031.13       0.5923     0.206975         9.22
ExhaustiveBinary-512-itq-rf20 (query)            10_466.90     1_685.46    12_152.37       0.7320     0.100842         9.22
ExhaustiveBinary-512-itq (self)                  10_466.90    15_624.26    26_091.16       0.5949     0.203446         9.22
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)      2_527.01        73.69     2_600.70       0.1443          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)      2_527.01        87.07     2_614.09       0.1389          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)      2_527.01       117.13     2_644.15       0.1371          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)      2_527.01       100.30     2_627.32       0.3253     0.693095         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)     2_527.01       120.81     2_647.82       0.4471     0.412064         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)     2_527.01       162.94     2_689.96       0.5876     0.229141         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)      2_527.01       112.44     2_639.46       0.3165     0.721918         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)     2_527.01       141.23     2_668.25       0.4368     0.431054         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)     2_527.01       182.26     2_709.28       0.5764     0.241196         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)      2_527.01       155.33     2_682.34       0.3123     0.739525         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)     2_527.01       174.89     2_701.90       0.4312     0.443202         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)     2_527.01       223.37     2_750.38       0.5697     0.249327         5.79
IVF-Binary-256-nl273-random (self)                2_527.01     1_376.08     3_903.10       0.4422     0.416770         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)      3_066.47        78.69     3_145.16       0.1403          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)      3_066.47        98.04     3_164.51       0.1380          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)      3_066.47       101.95     3_168.42       0.3203     0.707597         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)     3_066.47       125.68     3_192.15       0.4414     0.420873         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)     3_066.47       168.99     3_235.46       0.5825     0.233592         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)      3_066.47       124.91     3_191.38       0.3145     0.730872         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)     3_066.47       150.78     3_217.25       0.4334     0.437743         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)     3_066.47       217.16     3_283.63       0.5726     0.245415         5.80
IVF-Binary-256-nl387-random (self)                3_066.47     1_272.76     4_339.23       0.4467     0.406728         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)      3_820.80        73.18     3_893.98       0.1414          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)      3_820.80        82.86     3_903.66       0.1400          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)      3_820.80        95.99     3_916.79       0.1386          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)      3_820.80        96.59     3_917.39       0.3225     0.696766         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)     3_820.80       125.44     3_946.23       0.4444     0.412716         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)     3_820.80       163.21     3_984.01       0.5867     0.227725         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)      3_820.80       104.33     3_925.12       0.3186     0.711741         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)     3_820.80       126.98     3_947.78       0.4392     0.423763         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)     3_820.80       183.14     4_003.94       0.5799     0.235990         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)      3_820.80       117.58     3_938.38       0.3149     0.727204         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)     3_820.80       143.97     3_964.77       0.4341     0.435337         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)     3_820.80       192.84     4_013.64       0.5738     0.244002         5.82
IVF-Binary-256-nl547-random (self)                3_820.80     1_186.72     5_007.52       0.4498     0.399532         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)        11_429.23        78.47    11_507.70       0.1280          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)        11_429.23        86.71    11_515.95       0.1214          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)        11_429.23       121.34    11_550.57       0.1197          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)        11_429.23       105.04    11_534.28       0.2948     0.795873         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)       11_429.23       127.40    11_556.63       0.4089     0.482130         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)       11_429.23       170.98    11_600.22       0.5478     0.274338         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)        11_429.23       119.97    11_549.21       0.2826     0.839629         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)       11_429.23       147.16    11_576.39       0.3947     0.510538         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)       11_429.23       192.27    11_621.50       0.5319     0.292247         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)        11_429.23       152.72    11_581.95       0.2786     0.859116         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)       11_429.23       185.33    11_614.56       0.3893     0.524035         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)       11_429.23       230.41    11_659.65       0.5247     0.302361         5.79
IVF-Binary-256-nl273-itq (self)                  11_429.23     1_436.74    12_865.98       0.4003     0.494326         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)        13_256.26        85.55    13_341.81       0.1245          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)        13_256.26       107.78    13_364.04       0.1211          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)        13_256.26       113.61    13_369.87       0.2874     0.820655         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)       13_256.26       130.67    13_386.93       0.4003     0.498124         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)       13_256.26       178.29    13_434.55       0.5386     0.283016         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)        13_256.26       132.69    13_388.95       0.2810     0.851761         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)       13_256.26       171.30    13_427.56       0.3920     0.519024         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)       13_256.26       206.00    13_462.26       0.5281     0.297414         5.80
IVF-Binary-256-nl387-itq (self)                  13_256.26     1_325.74    14_582.00       0.4057     0.482074         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)        12_485.29        74.35    12_559.64       0.1250          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)        12_485.29        81.18    12_566.47       0.1231          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)        12_485.29        91.89    12_577.18       0.1212          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)        12_485.29       102.88    12_588.18       0.2896     0.805797         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)       12_485.29       122.25    12_607.54       0.4040     0.487268         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)       12_485.29       172.36    12_657.65       0.5433     0.275808         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)        12_485.29       122.26    12_607.55       0.2851     0.827063         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)       12_485.29       141.79    12_627.08       0.3980     0.502163         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)       12_485.29       169.50    12_654.80       0.5353     0.286311         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)        12_485.29       118.26    12_603.55       0.2809     0.845994         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)       12_485.29       139.48    12_624.78       0.3925     0.515502         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)       12_485.29       207.35    12_692.64       0.5287     0.295512         5.82
IVF-Binary-256-nl547-itq (self)                  12_485.29     1_219.52    13_704.81       0.4092     0.473007         5.82
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)      3_937.67       137.83     4_075.51       0.2304          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)      3_937.67       161.56     4_099.24       0.2278          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)      3_937.67       219.03     4_156.70       0.2270          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)      3_937.67       179.58     4_117.26       0.4861     0.337636        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)     3_937.67       214.37     4_152.04       0.6274     0.176910        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)     3_937.67       297.98     4_235.65       0.7653     0.083632        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)      3_937.67       194.04     4_131.72       0.4812     0.344838        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)     3_937.67       257.89     4_195.56       0.6224     0.181320        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)     3_937.67       441.25     4_378.93       0.7603     0.086510        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)      3_937.67       334.06     4_271.74       0.4791     0.349067        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)     3_937.67       387.40     4_325.07       0.6195     0.184434        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)     3_937.67       443.03     4_380.70       0.7568     0.088660        10.40
IVF-Binary-512-nl273-random (self)                3_937.67     3_106.58     7_044.25       0.6260     0.177569        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)      4_504.02       251.04     4_755.06       0.2289          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)      4_504.02       283.01     4_787.04       0.2274          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)      4_504.02       268.46     4_772.48       0.4839     0.341409        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)     4_504.02       335.40     4_839.42       0.6254     0.178808        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)     4_504.02       326.00     4_830.02       0.7626     0.084991        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)      4_504.02       243.03     4_747.05       0.4809     0.347106        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)     4_504.02       402.81     4_906.83       0.6210     0.183001        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)     4_504.02       433.95     4_937.98       0.7578     0.087878        10.41
IVF-Binary-512-nl387-random (self)                4_504.02     2_169.98     6_674.01       0.6282     0.175444        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)      5_154.05       191.05     5_345.11       0.2295          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)      5_154.05       168.88     5_322.93       0.2286          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)      5_154.05       170.60     5_324.66       0.2277          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)      5_154.05       214.54     5_368.60       0.4844     0.340121        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)     5_154.05       202.48     5_356.54       0.6262     0.177882        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)     5_154.05       257.32     5_411.37       0.7650     0.083906        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)      5_154.05       264.30     5_418.35       0.4822     0.344427        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)     5_154.05       268.94     5_423.00       0.6234     0.180818        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)     5_154.05       323.63     5_477.69       0.7614     0.086088        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)      5_154.05       279.71     5_433.76       0.4802     0.348092        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)     5_154.05       274.00     5_428.05       0.6210     0.183289        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)     5_154.05       321.37     5_475.43       0.7585     0.087937        10.43
IVF-Binary-512-nl547-random (self)                5_154.05     2_024.81     7_178.86       0.6297     0.174184        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)        12_149.60       173.02    12_322.61       0.2205          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)        12_149.60       226.46    12_376.06       0.2175          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)        12_149.60       262.20    12_411.80       0.2165          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)        12_149.60       182.30    12_331.90       0.4678     0.364532        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)       12_149.60       202.82    12_352.42       0.6055     0.194694        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)       12_149.60       297.96    12_447.56       0.7455     0.093040        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)        12_149.60       247.52    12_397.12       0.4627     0.372944        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)       12_149.60       294.88    12_444.47       0.5996     0.199967        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)       12_149.60       361.44    12_511.03       0.7396     0.096303        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)        12_149.60       260.24    12_409.84       0.4599     0.377237        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)       12_149.60       378.67    12_528.27       0.5969     0.202840        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)       12_149.60       452.45    12_602.05       0.7361     0.098394        10.40
IVF-Binary-512-nl273-itq (self)                  12_149.60     2_522.97    14_672.57       0.6028     0.196260        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)        12_786.77       152.77    12_939.54       0.2189          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)        12_786.77       188.00    12_974.77       0.2174          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)        12_786.77       190.34    12_977.11       0.4643     0.368946        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)       12_786.77       231.17    13_017.94       0.6026     0.197111        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)       12_786.77       302.49    13_089.26       0.7422     0.094717        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)        12_786.77       250.33    13_037.10       0.4606     0.375930        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)       12_786.77       270.46    13_057.23       0.5980     0.202033        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)       12_786.77       343.27    13_130.04       0.7370     0.097882        10.41
IVF-Binary-512-nl387-itq (self)                  12_786.77     2_214.69    15_001.46       0.6053     0.193911        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)        12_903.51       132.39    13_035.90       0.2195          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)        12_903.51       146.90    13_050.42       0.2184          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)        12_903.51       167.69    13_071.21       0.2176          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)        12_903.51       165.36    13_068.87       0.4660     0.366461        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)       12_903.51       196.28    13_099.79       0.6043     0.195491        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)       12_903.51       246.72    13_150.24       0.7450     0.093174        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)        12_903.51       180.52    13_084.04       0.4633     0.371329        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)       12_903.51       213.51    13_117.02       0.6008     0.199134        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)       12_903.51       265.11    13_168.63       0.7408     0.095707        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)        12_903.51       205.97    13_109.48       0.4611     0.375531        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)       12_903.51       239.15    13_142.66       0.5983     0.201713        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)       12_903.51       294.30    13_197.82       0.7379     0.097580        10.43
IVF-Binary-512-nl547-itq (self)                  12_903.51     1_955.87    14_859.38       0.6073     0.191957        10.43
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Binary - Euclidean (LowRank)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D - Binary Quantisation
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.23     2_365.54     2_368.77       1.0000     0.000000        18.31
Exhaustive (self)                                     3.23    24_849.44    24_852.67       1.0000     0.000000        18.31
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-256-random_no_rr (query)         1_269.49     1_127.51     2_397.00       0.0889          NaN         4.61
ExhaustiveBinary-256-random-rf5 (query)           1_269.49     1_232.63     2_502.11       0.2205     5.185433         4.61
ExhaustiveBinary-256-random-rf10 (query)          1_269.49     1_129.61     2_399.10       0.3209     3.263431         4.61
ExhaustiveBinary-256-random-rf20 (query)          1_269.49     1_345.65     2_615.14       0.4521     1.957008         4.61
ExhaustiveBinary-256-random (self)                1_269.49    11_325.62    12_595.11       0.3238     3.208083         4.61
ExhaustiveBinary-256-itq_no_rr (query)            9_524.87     1_141.01    10_665.89       0.0789          NaN         4.61
ExhaustiveBinary-256-itq-rf5 (query)              9_524.87     1_075.98    10_600.85       0.2034     5.707507         4.61
ExhaustiveBinary-256-itq-rf10 (query)             9_524.87     1_083.95    10_608.82       0.2988     3.630297         4.61
ExhaustiveBinary-256-itq-rf20 (query)             9_524.87     1_386.20    10_911.07       0.4255     2.198529         4.61
ExhaustiveBinary-256-itq (self)                   9_524.87    11_783.10    21_307.98       0.3019     3.549284         4.61
---------------------------------------------------------------------------------------------------------------------------
ExhaustiveBinary-512-random_no_rr (query)         2_538.62     1_618.74     4_157.35       0.1603          NaN         9.22
ExhaustiveBinary-512-random-rf5 (query)           2_538.62     1_721.09     4_259.71       0.3496     2.953827         9.22
ExhaustiveBinary-512-random-rf10 (query)          2_538.62     1_793.07     4_331.68       0.4779     1.726989         9.22
ExhaustiveBinary-512-random-rf20 (query)          2_538.62     1_870.11     4_408.73       0.6243     0.943773         9.22
ExhaustiveBinary-512-random (self)                2_538.62    17_940.67    20_479.29       0.4788     1.710526         9.22
ExhaustiveBinary-512-itq_no_rr (query)           11_155.79     1_638.91    12_794.70       0.1573          NaN         9.22
ExhaustiveBinary-512-itq-rf5 (query)             11_155.79     1_623.15    12_778.94       0.3466     3.048135         9.22
ExhaustiveBinary-512-itq-rf10 (query)            11_155.79     1_818.79    12_974.58       0.4716     1.796962         9.22
ExhaustiveBinary-512-itq-rf20 (query)            11_155.79     1_711.20    12_866.99       0.6178     0.986492         9.22
ExhaustiveBinary-512-itq (self)                  11_155.79    17_610.27    28_766.06       0.4723     1.780317         9.22
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-256-nl273-np13-rf0-random (query)      2_786.15        86.08     2_872.23       0.1015          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-random (query)      2_786.15        95.50     2_881.65       0.0926          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-random (query)      2_786.15       122.99     2_909.14       0.0904          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-random (query)      2_786.15       106.34     2_892.49       0.2466     4.682597         5.79
IVF-Binary-256-nl273-np13-rf10-random (query)     2_786.15       156.74     2_942.89       0.3527     2.916537         5.79
IVF-Binary-256-nl273-np13-rf20-random (query)     2_786.15       212.28     2_998.43       0.4886     1.698168         5.79
IVF-Binary-256-nl273-np16-rf5-random (query)      2_786.15       134.36     2_920.51       0.2318     5.166092         5.79
IVF-Binary-256-nl273-np16-rf10-random (query)     2_786.15       158.57     2_944.72       0.3354     3.190525         5.79
IVF-Binary-256-nl273-np16-rf20-random (query)     2_786.15       182.86     2_969.01       0.4700     1.862358         5.79
IVF-Binary-256-nl273-np23-rf5-random (query)      2_786.15       139.51     2_925.66       0.2280     5.296145         5.79
IVF-Binary-256-nl273-np23-rf10-random (query)     2_786.15       179.34     2_965.49       0.3304     3.274036         5.79
IVF-Binary-256-nl273-np23-rf20-random (query)     2_786.15       223.88     3_010.03       0.4638     1.915926         5.79
IVF-Binary-256-nl273-random (self)                2_786.15     1_369.78     4_155.93       0.3387     3.130112         5.79
IVF-Binary-256-nl387-np19-rf0-random (query)      3_115.57        75.97     3_191.54       0.0945          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-random (query)      3_115.57        99.30     3_214.87       0.0919          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-random (query)      3_115.57       100.93     3_216.50       0.2358     4.931507         5.80
IVF-Binary-256-nl387-np19-rf10-random (query)     3_115.57       124.31     3_239.89       0.3421     3.052683         5.80
IVF-Binary-256-nl387-np19-rf20-random (query)     3_115.57       161.56     3_277.14       0.4777     1.802324         5.80
IVF-Binary-256-nl387-np27-rf5-random (query)      3_115.57       121.45     3_237.02       0.2299     5.156744         5.80
IVF-Binary-256-nl387-np27-rf10-random (query)     3_115.57       144.79     3_260.37       0.3340     3.196200         5.80
IVF-Binary-256-nl387-np27-rf20-random (query)     3_115.57       212.67     3_328.24       0.4679     1.888188         5.80
IVF-Binary-256-nl387-random (self)                3_115.57     1_342.13     4_457.70       0.3448     3.003189         5.80
IVF-Binary-256-nl547-np23-rf0-random (query)      4_218.79        91.02     4_309.81       0.0954          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-random (query)      4_218.79        83.07     4_301.86       0.0934          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-random (query)      4_218.79        94.57     4_313.36       0.0917          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-random (query)      4_218.79       111.13     4_329.92       0.2397     4.790559         5.82
IVF-Binary-256-nl547-np23-rf10-random (query)     4_218.79       144.04     4_362.83       0.3461     2.990596         5.82
IVF-Binary-256-nl547-np23-rf20-random (query)     4_218.79       158.94     4_377.73       0.4824     1.757345         5.82
IVF-Binary-256-nl547-np27-rf5-random (query)      4_218.79       118.61     4_337.40       0.2354     4.926699         5.82
IVF-Binary-256-nl547-np27-rf10-random (query)     4_218.79       164.02     4_382.81       0.3400     3.085684         5.82
IVF-Binary-256-nl547-np27-rf20-random (query)     4_218.79       193.95     4_412.74       0.4749     1.823733         5.82
IVF-Binary-256-nl547-np33-rf5-random (query)      4_218.79       125.91     4_344.70       0.2310     5.077683         5.82
IVF-Binary-256-nl547-np33-rf10-random (query)     4_218.79       145.97     4_364.76       0.3339     3.187133         5.82
IVF-Binary-256-nl547-np33-rf20-random (query)     4_218.79       189.48     4_408.27       0.4678     1.890091         5.82
IVF-Binary-256-nl547-random (self)                4_218.79     1_225.69     5_444.48       0.3482     2.939443         5.82
IVF-Binary-256-nl273-np13-rf0-itq (query)        10_974.27        98.78    11_073.05       0.0922          NaN         5.79
IVF-Binary-256-nl273-np16-rf0-itq (query)        10_974.27       103.08    11_077.35       0.0826          NaN         5.79
IVF-Binary-256-nl273-np23-rf0-itq (query)        10_974.27       134.92    11_109.19       0.0807          NaN         5.79
IVF-Binary-256-nl273-np13-rf5-itq (query)        10_974.27       117.18    11_091.45       0.2327     5.129309         5.79
IVF-Binary-256-nl273-np13-rf10-itq (query)       10_974.27       202.71    11_176.98       0.3355     3.249173         5.79
IVF-Binary-256-nl273-np13-rf20-itq (query)       10_974.27       352.18    11_326.45       0.4664     1.924112         5.79
IVF-Binary-256-nl273-np16-rf5-itq (query)        10_974.27       184.60    11_158.88       0.2150     5.713879         5.79
IVF-Binary-256-nl273-np16-rf10-itq (query)       10_974.27       200.20    11_174.47       0.3146     3.640894         5.79
IVF-Binary-256-nl273-np16-rf20-itq (query)       10_974.27       277.58    11_251.85       0.4438     2.161676         5.79
IVF-Binary-256-nl273-np23-rf5-itq (query)        10_974.27       174.31    11_148.59       0.2106     5.906866         5.79
IVF-Binary-256-nl273-np23-rf10-itq (query)       10_974.27       227.46    11_201.73       0.3094     3.749775         5.79
IVF-Binary-256-nl273-np23-rf20-itq (query)       10_974.27       311.30    11_285.57       0.4378     2.226585         5.79
IVF-Binary-256-nl273-itq (self)                  10_974.27     1_697.38    12_671.66       0.3170     3.549023         5.79
IVF-Binary-256-nl387-np19-rf0-itq (query)        11_861.50        79.06    11_940.56       0.0853          NaN         5.80
IVF-Binary-256-nl387-np27-rf0-itq (query)        11_861.50        99.12    11_960.62       0.0826          NaN         5.80
IVF-Binary-256-nl387-np19-rf5-itq (query)        11_861.50        99.59    11_961.09       0.2202     5.593077         5.80
IVF-Binary-256-nl387-np19-rf10-itq (query)       11_861.50       120.72    11_982.22       0.3209     3.500958         5.80
IVF-Binary-256-nl387-np19-rf20-itq (query)       11_861.50       166.08    12_027.58       0.4531     2.071410         5.80
IVF-Binary-256-nl387-np27-rf5-itq (query)        11_861.50       148.22    12_009.72       0.2134     5.809710         5.80
IVF-Binary-256-nl387-np27-rf10-itq (query)       11_861.50       175.45    12_036.95       0.3126     3.649095         5.80
IVF-Binary-256-nl387-np27-rf20-itq (query)       11_861.50       205.37    12_066.87       0.4429     2.167236         5.80
IVF-Binary-256-nl387-itq (self)                  11_861.50     1_542.79    13_404.29       0.3241     3.414668         5.80
IVF-Binary-256-nl547-np23-rf0-itq (query)        12_531.03        73.45    12_604.48       0.0864          NaN         5.82
IVF-Binary-256-nl547-np27-rf0-itq (query)        12_531.03        80.31    12_611.34       0.0844          NaN         5.82
IVF-Binary-256-nl547-np33-rf0-itq (query)        12_531.03        97.37    12_628.40       0.0828          NaN         5.82
IVF-Binary-256-nl547-np23-rf5-itq (query)        12_531.03        95.62    12_626.66       0.2230     5.350284         5.82
IVF-Binary-256-nl547-np23-rf10-itq (query)       12_531.03       139.42    12_670.45       0.3253     3.356318         5.82
IVF-Binary-256-nl547-np23-rf20-itq (query)       12_531.03       194.72    12_725.75       0.4607     1.969077         5.82
IVF-Binary-256-nl547-np27-rf5-itq (query)        12_531.03       124.96    12_656.00       0.2172     5.556188         5.82
IVF-Binary-256-nl547-np27-rf10-itq (query)       12_531.03       205.30    12_736.33       0.3178     3.500425         5.82
IVF-Binary-256-nl547-np27-rf20-itq (query)       12_531.03       269.11    12_800.15       0.4514     2.064914         5.82
IVF-Binary-256-nl547-np33-rf5-itq (query)        12_531.03       155.43    12_686.46       0.2132     5.727838         5.82
IVF-Binary-256-nl547-np33-rf10-itq (query)       12_531.03       164.89    12_695.93       0.3118     3.617221         5.82
IVF-Binary-256-nl547-np33-rf20-itq (query)       12_531.03       248.84    12_779.88       0.4438     2.140695         5.82
IVF-Binary-256-nl547-itq (self)                  12_531.03     1_421.85    13_952.89       0.3277     3.304345         5.82
---------------------------------------------------------------------------------------------------------------------------
IVF-Binary-512-nl273-np13-rf0-random (query)      3_935.43       136.15     4_071.58       0.1685          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-random (query)      3_935.43       200.38     4_135.82       0.1642          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-random (query)      3_935.43       255.66     4_191.09       0.1629          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-random (query)      3_935.43       232.45     4_167.88       0.3648     2.762852        10.40
IVF-Binary-512-nl273-np13-rf10-random (query)     3_935.43       249.05     4_184.48       0.4946     1.613586        10.40
IVF-Binary-512-nl273-np13-rf20-random (query)     3_935.43       322.88     4_258.31       0.6428     0.865216        10.40
IVF-Binary-512-nl273-np16-rf5-random (query)      3_935.43       246.77     4_182.20       0.3574     2.870953        10.40
IVF-Binary-512-nl273-np16-rf10-random (query)     3_935.43       285.06     4_220.49       0.4866     1.678608        10.40
IVF-Binary-512-nl273-np16-rf20-random (query)     3_935.43       317.89     4_253.32       0.6348     0.903671        10.40
IVF-Binary-512-nl273-np23-rf5-random (query)      3_935.43       367.80     4_303.23       0.3551     2.903769        10.40
IVF-Binary-512-nl273-np23-rf10-random (query)     3_935.43       309.68     4_245.11       0.4839     1.699072        10.40
IVF-Binary-512-nl273-np23-rf20-random (query)     3_935.43       348.33     4_283.76       0.6317     0.917169        10.40
IVF-Binary-512-nl273-random (self)                3_935.43     2_270.54     6_205.97       0.4885     1.655177        10.40
IVF-Binary-512-nl387-np19-rf0-random (query)      4_364.01       139.32     4_503.33       0.1646          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-random (query)      4_364.01       177.07     4_541.08       0.1631          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-random (query)      4_364.01       170.18     4_534.19       0.3604     2.826193        10.41
IVF-Binary-512-nl387-np19-rf10-random (query)     4_364.01       201.33     4_565.34       0.4905     1.642344        10.41
IVF-Binary-512-nl387-np19-rf20-random (query)     4_364.01       254.86     4_618.88       0.6385     0.885648        10.41
IVF-Binary-512-nl387-np27-rf5-random (query)      4_364.01       241.23     4_605.24       0.3567     2.886896        10.41
IVF-Binary-512-nl387-np27-rf10-random (query)     4_364.01       261.68     4_625.70       0.4857     1.679722        10.41
IVF-Binary-512-nl387-np27-rf20-random (query)     4_364.01       301.75     4_665.76       0.6332     0.909692        10.41
IVF-Binary-512-nl387-random (self)                4_364.01     2_045.47     6_409.48       0.4917     1.624012        10.41
IVF-Binary-512-nl547-np23-rf0-random (query)      5_276.54       135.69     5_412.23       0.1660          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-random (query)      5_276.54       180.67     5_457.21       0.1648          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-random (query)      5_276.54       179.12     5_455.66       0.1637          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-random (query)      5_276.54       175.02     5_451.56       0.3623     2.788789        10.43
IVF-Binary-512-nl547-np23-rf10-random (query)     5_276.54       196.11     5_472.65       0.4927     1.618030        10.43
IVF-Binary-512-nl547-np23-rf20-random (query)     5_276.54       250.55     5_527.09       0.6419     0.867389        10.43
IVF-Binary-512-nl547-np27-rf5-random (query)      5_276.54       174.89     5_451.43       0.3595     2.836677        10.43
IVF-Binary-512-nl547-np27-rf10-random (query)     5_276.54       207.29     5_483.83       0.4890     1.650040        10.43
IVF-Binary-512-nl547-np27-rf20-random (query)     5_276.54       269.57     5_546.11       0.6371     0.891042        10.43
IVF-Binary-512-nl547-np33-rf5-random (query)      5_276.54       213.25     5_489.79       0.3568     2.876216        10.43
IVF-Binary-512-nl547-np33-rf10-random (query)     5_276.54       250.00     5_526.53       0.4854     1.678082        10.43
IVF-Binary-512-nl547-np33-rf20-random (query)     5_276.54       303.87     5_580.41       0.6333     0.908544        10.43
IVF-Binary-512-nl547-random (self)                5_276.54     2_017.98     7_294.52       0.4940     1.602597        10.43
IVF-Binary-512-nl273-np13-rf0-itq (query)        13_607.95       146.52    13_754.47       0.1650          NaN        10.40
IVF-Binary-512-nl273-np16-rf0-itq (query)        13_607.95       160.43    13_768.38       0.1605          NaN        10.40
IVF-Binary-512-nl273-np23-rf0-itq (query)        13_607.95       207.44    13_815.39       0.1590          NaN        10.40
IVF-Binary-512-nl273-np13-rf5-itq (query)        13_607.95       189.50    13_797.46       0.3626     2.845309        10.40
IVF-Binary-512-nl273-np13-rf10-itq (query)       13_607.95       237.30    13_845.25       0.4907     1.665137        10.40
IVF-Binary-512-nl273-np13-rf20-itq (query)       13_607.95       299.28    13_907.23       0.6363     0.907561        10.40
IVF-Binary-512-nl273-np16-rf5-itq (query)        13_607.95       222.28    13_830.23       0.3546     2.969500        10.40
IVF-Binary-512-nl273-np16-rf10-itq (query)       13_607.95       270.85    13_878.81       0.4815     1.737951        10.40
IVF-Binary-512-nl273-np16-rf20-itq (query)       13_607.95       348.37    13_956.33       0.6281     0.946697        10.40
IVF-Binary-512-nl273-np23-rf5-itq (query)        13_607.95       304.18    13_912.13       0.3524     3.006598        10.40
IVF-Binary-512-nl273-np23-rf10-itq (query)       13_607.95       310.45    13_918.40       0.4787     1.761636        10.40
IVF-Binary-512-nl273-np23-rf20-itq (query)       13_607.95       353.41    13_961.36       0.6249     0.961770        10.40
IVF-Binary-512-nl273-itq (self)                  13_607.95     2_492.54    16_100.49       0.4818     1.724536        10.40
IVF-Binary-512-nl387-np19-rf0-itq (query)        13_048.82       171.29    13_220.10       0.1621          NaN        10.41
IVF-Binary-512-nl387-np27-rf0-itq (query)        13_048.82       186.71    13_235.53       0.1605          NaN        10.41
IVF-Binary-512-nl387-np19-rf5-itq (query)        13_048.82       192.06    13_240.88       0.3572     2.927775        10.41
IVF-Binary-512-nl387-np19-rf10-itq (query)       13_048.82       278.69    13_327.51       0.4850     1.706392        10.41
IVF-Binary-512-nl387-np19-rf20-itq (query)       13_048.82       349.08    13_397.90       0.6316     0.927251        10.41
IVF-Binary-512-nl387-np27-rf5-itq (query)        13_048.82       293.95    13_342.77       0.3536     2.982566        10.41
IVF-Binary-512-nl387-np27-rf10-itq (query)       13_048.82       303.91    13_352.73       0.4806     1.745379        10.41
IVF-Binary-512-nl387-np27-rf20-itq (query)       13_048.82       487.42    13_536.24       0.6266     0.952456        10.41
IVF-Binary-512-nl387-itq (self)                  13_048.82     2_405.78    15_454.60       0.4858     1.689719        10.41
IVF-Binary-512-nl547-np23-rf0-itq (query)        13_818.74       159.75    13_978.49       0.1631          NaN        10.43
IVF-Binary-512-nl547-np27-rf0-itq (query)        13_818.74       180.39    13_999.13       0.1617          NaN        10.43
IVF-Binary-512-nl547-np33-rf0-itq (query)        13_818.74       202.95    14_021.69       0.1605          NaN        10.43
IVF-Binary-512-nl547-np23-rf5-itq (query)        13_818.74       194.29    14_013.03       0.3598     2.869488        10.43
IVF-Binary-512-nl547-np23-rf10-itq (query)       13_818.74       236.34    14_055.08       0.4886     1.676771        10.43
IVF-Binary-512-nl547-np23-rf20-itq (query)       13_818.74       328.22    14_146.96       0.6346     0.914158        10.43
IVF-Binary-512-nl547-np27-rf5-itq (query)        13_818.74       238.18    14_056.92       0.3566     2.923912        10.43
IVF-Binary-512-nl547-np27-rf10-itq (query)       13_818.74       252.08    14_070.81       0.4843     1.713529        10.43
IVF-Binary-512-nl547-np27-rf20-itq (query)       13_818.74       295.66    14_114.40       0.6304     0.935097        10.43
IVF-Binary-512-nl547-np33-rf5-itq (query)        13_818.74       222.99    14_041.73       0.3540     2.964443        10.43
IVF-Binary-512-nl547-np33-rf10-itq (query)       13_818.74       244.30    14_063.04       0.4809     1.741643        10.43
IVF-Binary-512-nl547-np33-rf20-itq (query)       13_818.74       310.79    14_129.53       0.6268     0.952606        10.43
IVF-Binary-512-nl547-itq (self)                  13_818.74     1_974.88    15_793.62       0.4879     1.669911        10.43
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### With higher dimensionality

<details>
<summary><b>Binary - Euclidean (Gaussian)</b>:</summary>
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
centroid calculations in the quantiser.

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
<summary><b>Binary - Euclidean (Gaussian)</b>:</summary>
<pre><code>

</code></pre>
</details>