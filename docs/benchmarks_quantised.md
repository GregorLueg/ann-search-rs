## Quantised indices benchmarks and parameter gridsearch

Quantised indices compress the data stored in the index structure itself via
quantisation. This can also in some cases accelerated substantially the query
speed. The core idea is to trade in Recall for reduction in memory finger print.
</br>
In the benchmarks below, some of these indices were run on synthetic data with 
better structure in higher dimensions to avoid the curse of dimensionality, via 
for example this command:

```bash
cargo run --example gridsearch_ivf_sq8 --release --features quantised -- --distance euclidean --dim 128 --data correlated
```

If you wish to run all of the benchmarks, below, you can just run:

```bash
bash examples/run_benchmarks.sh --quantised
```

## Table of Contents

- [BF16 quantisation](#bf16-ivf-and-exhaustive)
- [SQ8](#sq8-ivf)

### BF16 (IVF and exhaustive)

The BF16 quantisation reduces the floats to `bf16` which keeps the range of 
`f32`, but loses precision in the digits from ~3 onwards. The actual distance
calculations in the index happen in `f32`; however, due to lossy compression
to `bf16` there is some Recall loss. This is compensated with drastically
reduced memory fingerprint (nearly halved for f32). The performance for Cosine
distances however is way, way worse–compared to Euclidean.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

<details>
<summary><b>BF16 quantisations - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.40     2_316.79     2_320.19       1.0000     0.000000        18.31
Exhaustive (self)                                     3.40    23_490.77    23_494.17       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                               5.57     3_136.70     3_142.27       0.9867     0.065388         9.16
Exhaustive-BF16 (self)                                5.57    24_331.76    24_337.32       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                       1_356.57       231.91     1_588.47       0.9771     0.096213        10.34
IVF-BF16-nl273-np16 (query)                       1_356.57       284.74     1_641.30       0.9840     0.070989        10.34
IVF-BF16-nl273-np23 (query)                       1_356.57       393.15     1_749.71       0.9867     0.065388        10.34
IVF-BF16-nl273 (self)                             1_356.57     4_359.58     5_716.14       0.9830     0.094689        10.34
IVF-BF16-nl387-np19 (query)                       1_851.52       234.22     2_085.73       0.9804     0.093392        10.35
IVF-BF16-nl387-np27 (query)                       1_851.52       317.44     2_168.96       0.9865     0.065987        10.35
IVF-BF16-nl387 (self)                             1_851.52     3_590.21     5_441.72       0.9828     0.095265        10.35
IVF-BF16-nl547-np23 (query)                       2_687.57       212.04     2_899.61       0.9767     0.102772        10.37
IVF-BF16-nl547-np27 (query)                       2_687.57       237.59     2_925.17       0.9833     0.078352        10.37
IVF-BF16-nl547-np33 (query)                       2_687.57       284.86     2_972.43       0.9865     0.066298        10.37
IVF-BF16-nl547 (self)                             2_687.57     3_173.05     5_860.62       0.9827     0.095506        10.37
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Cosine (Gaussian)</b>:</summary>
</br>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.20     2_418.67     2_422.88       1.0000     0.000000        18.88
Exhaustive (self)                                     4.20    23_956.95    23_961.15       1.0000     0.000000        18.88
Exhaustive-BF16 (query)                               6.77     2_952.29     2_959.06       0.7437     0.000827         9.44
Exhaustive-BF16 (self)                                6.77    24_091.47    24_098.24       1.0000     0.000000         9.44
IVF-BF16-nl273-np13 (query)                       1_364.81       246.90     1_611.71       0.7400     0.000825        10.62
IVF-BF16-nl273-np16 (query)                       1_364.81       325.59     1_690.40       0.7428     0.000825        10.62
IVF-BF16-nl273-np23 (query)                       1_364.81       438.19     1_803.00       0.7437     0.000827        10.62
IVF-BF16-nl273 (self)                             1_364.81     4_638.80     6_003.61       0.7414     0.001477        10.62
IVF-BF16-nl387-np19 (query)                       1_925.20       246.72     2_171.92       0.7412     0.000828        10.64
IVF-BF16-nl387-np27 (query)                       1_925.20       339.03     2_264.23       0.7437     0.000827        10.64
IVF-BF16-nl387 (self)                             1_925.20     3_721.05     5_646.25       0.7413     0.001477        10.64
IVF-BF16-nl547-np23 (query)                       2_680.46       216.85     2_897.32       0.7399     0.000826        10.66
IVF-BF16-nl547-np27 (query)                       2_680.46       251.45     2_931.91       0.7425     0.000826        10.66
IVF-BF16-nl547-np33 (query)                       2_680.46       299.52     2_979.98       0.7436     0.000827        10.66
IVF-BF16-nl547 (self)                             2_680.46     3_267.48     5_947.94       0.7413     0.001478        10.66
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (Correlated)</b>:</summary>
</br>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.15     2_332.24     2_335.39       1.0000     0.000000        18.31
Exhaustive (self)                                     3.15    23_270.86    23_274.01       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                               5.46     2_878.84     2_884.30       0.9649     0.019029         9.16
Exhaustive-BF16 (self)                                5.46    23_371.53    23_376.99       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                       1_311.25       229.45     1_540.70       0.9648     0.019044        10.34
IVF-BF16-nl273-np16 (query)                       1_311.25       273.98     1_585.24       0.9649     0.019032        10.34
IVF-BF16-nl273-np23 (query)                       1_311.25       396.07     1_707.32       0.9649     0.019029        10.34
IVF-BF16-nl273 (self)                             1_311.25     4_443.41     5_754.66       0.9561     0.026167        10.34
IVF-BF16-nl387-np19 (query)                       1_841.38       230.73     2_072.11       0.9649     0.019036        10.35
IVF-BF16-nl387-np27 (query)                       1_841.38       315.07     2_156.45       0.9649     0.019029        10.35
IVF-BF16-nl387 (self)                             1_841.38     3_564.23     5_405.61       0.9561     0.026167        10.35
IVF-BF16-nl547-np23 (query)                       2_604.51       208.61     2_813.12       0.9649     0.019042        10.37
IVF-BF16-nl547-np27 (query)                       2_604.51       243.74     2_848.25       0.9649     0.019035        10.37
IVF-BF16-nl547-np33 (query)                       2_604.51       302.72     2_907.24       0.9649     0.019029        10.37
IVF-BF16-nl547 (self)                             2_604.51     3_137.88     5_742.39       0.9561     0.026169        10.37
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>BF16 quantisations - Euclidean (LowRank)</b>:</summary>
</br>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.16     2_335.83     2_338.99       1.0000     0.000000        18.31
Exhaustive (self)                                     3.16    23_754.94    23_758.10       1.0000     0.000000        18.31
Exhaustive-BF16 (query)                               4.53     2_923.60     2_928.13       0.9348     0.158338         9.16
Exhaustive-BF16 (self)                                4.53    23_703.28    23_707.81       1.0000     0.000000         9.16
IVF-BF16-nl273-np13 (query)                       1_393.53       222.70     1_616.23       0.9348     0.158347        10.34
IVF-BF16-nl273-np16 (query)                       1_393.53       269.48     1_663.01       0.9348     0.158338        10.34
IVF-BF16-nl273-np23 (query)                       1_393.53       367.65     1_761.18       0.9348     0.158338        10.34
IVF-BF16-nl273 (self)                             1_393.53     4_191.90     5_585.43       0.9174     0.221294        10.34
IVF-BF16-nl387-np19 (query)                       1_906.89       233.21     2_140.10       0.9348     0.158338        10.35
IVF-BF16-nl387-np27 (query)                       1_906.89       323.39     2_230.28       0.9348     0.158338        10.35
IVF-BF16-nl387 (self)                             1_906.89     3_623.11     5_530.00       0.9174     0.221294        10.35
IVF-BF16-nl547-np23 (query)                       2_657.27       208.00     2_865.27       0.9348     0.158338        10.37
IVF-BF16-nl547-np27 (query)                       2_657.27       237.98     2_895.25       0.9348     0.158338        10.37
IVF-BF16-nl547-np33 (query)                       2_657.27       281.63     2_938.90       0.9348     0.158338        10.37
IVF-BF16-nl547 (self)                             2_657.27     3_156.56     5_813.83       0.9174     0.221294        10.37
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### SQ8 (IVF)

This index uses scalar quantisation to 8-bits. It projects every dimensions
onto an `i8`. This also causes a reduction of the memory finger print. In the
case of 96 dimensions in f32 per vector, we go from *96 x 32 bits = 384 bytes*
to *96 x 8 bits = 96 bytes per vector*, a **4x reduction in memory per vector** 
(with overhead of the codebook).

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

<details>
<summary><b>SQ8 quantisations - Euclidean (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.24     2_352.99     2_356.23       1.0000     0.000000        18.31
Exhaustive (self)                                     3.24    23_512.61    23_515.85       1.0000     0.000000        18.31
IVF-SQ8-nl273-np13 (query)                        1_389.54        57.45     1_446.99       0.7971          NaN         5.76
IVF-SQ8-nl273-np16 (query)                        1_389.54        65.99     1_455.53       0.8001          NaN         5.76
IVF-SQ8-nl273-np23 (query)                        1_389.54        94.10     1_483.64       0.8011          NaN         5.76
IVF-SQ8-nl273 (self)                              1_389.54       911.57     2_301.11       0.8007          NaN         5.76
IVF-SQ8-nl387-np19 (query)                        1_930.62        60.17     1_990.80       0.7986          NaN         5.77
IVF-SQ8-nl387-np27 (query)                        1_930.62        77.68     2_008.31       0.8011          NaN         5.77
IVF-SQ8-nl387 (self)                              1_930.62       783.64     2_714.27       0.8007          NaN         5.77
IVF-SQ8-nl547-np23 (query)                        2_692.16        56.25     2_748.41       0.7968          NaN         5.79
IVF-SQ8-nl547-np27 (query)                        2_692.16        62.44     2_754.60       0.7995          NaN         5.79
IVF-SQ8-nl547-np33 (query)                        2_692.16        76.76     2_768.92       0.8010          NaN         5.79
IVF-SQ8-nl547 (self)                              2_692.16       723.33     3_415.49       0.8006          NaN         5.79
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Cosine (Gaussian)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.24     2_352.99     2_356.23       1.0000     0.000000        18.31
Exhaustive (self)                                     3.24    23_512.61    23_515.85       1.0000     0.000000        18.31
IVF-SQ8-nl273-np13 (query)                        1_389.54        57.45     1_446.99       0.7971          NaN         5.76
IVF-SQ8-nl273-np16 (query)                        1_389.54        65.99     1_455.53       0.8001          NaN         5.76
IVF-SQ8-nl273-np23 (query)                        1_389.54        94.10     1_483.64       0.8011          NaN         5.76
IVF-SQ8-nl273 (self)                              1_389.54       911.57     2_301.11       0.8007          NaN         5.76
IVF-SQ8-nl387-np19 (query)                        1_930.62        60.17     1_990.80       0.7986          NaN         5.77
IVF-SQ8-nl387-np27 (query)                        1_930.62        77.68     2_008.31       0.8011          NaN         5.77
IVF-SQ8-nl387 (self)                              1_930.62       783.64     2_714.27       0.8007          NaN         5.77
IVF-SQ8-nl547-np23 (query)                        2_692.16        56.25     2_748.41       0.7968          NaN         5.79
IVF-SQ8-nl547-np27 (query)                        2_692.16        62.44     2_754.60       0.7995          NaN         5.79
IVF-SQ8-nl547-np33 (query)                        2_692.16        76.76     2_768.92       0.8010          NaN         5.79
IVF-SQ8-nl547 (self)                              2_692.16       723.33     3_415.49       0.8006          NaN         5.79
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Gaussian - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 96D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   10.40    10_728.92    10_739.33       1.0000     0.000000        54.93
Exhaustive (self)                                    10.40   114_652.01   114_662.42       1.0000     0.000000        54.93
IVF-SQ8-nl273-np13 (query)                        7_129.02       231.45     7_360.47       0.8232          NaN        14.98
IVF-SQ8-nl273-np16 (query)                        7_129.02       286.78     7_415.80       0.8339          NaN        14.98
IVF-SQ8-nl273-np23 (query)                        7_129.02       392.53     7_521.55       0.8382          NaN        14.98
IVF-SQ8-nl273 (self)                              7_129.02     3_955.20    11_084.21       0.8385          NaN        14.98
IVF-SQ8-nl387-np19 (query)                       10_585.91       250.80    10_836.71       0.8284          NaN        15.02
IVF-SQ8-nl387-np27 (query)                       10_585.91       364.08    10_950.00       0.8382          NaN        15.02
IVF-SQ8-nl387 (self)                             10_585.91     3_518.60    14_104.51       0.8385          NaN        15.02
IVF-SQ8-nl547-np23 (query)                       15_221.95       249.65    15_471.59       0.8230          NaN        15.08
IVF-SQ8-nl547-np27 (query)                       15_221.95       290.91    15_512.86       0.8318          NaN        15.08
IVF-SQ8-nl547-np33 (query)                       15_221.95       348.66    15_570.61       0.8370          NaN        15.08
IVF-SQ8-nl547 (self)                             15_221.95     3_563.30    18_785.25       0.8372          NaN        15.08
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 96D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   10.50    14_444.50    14_455.00       1.0000     0.000000        54.93
Exhaustive (self)                                    10.50   116_074.46   116_084.95       1.0000     0.000000        54.93
IVF-SQ8-nl273-np13 (query)                        6_116.81       196.56     6_313.37       0.6925          NaN        14.98
IVF-SQ8-nl273-np16 (query)                        6_116.81       240.56     6_357.37       0.6931          NaN        14.98
IVF-SQ8-nl273-np23 (query)                        6_116.81       316.30     6_433.11       0.6933          NaN        14.98
IVF-SQ8-nl273 (self)                              6_116.81     3_099.54     9_216.35       0.6932          NaN        14.98
IVF-SQ8-nl387-np19 (query)                        8_666.05       219.92     8_885.96       0.6930          NaN        15.02
IVF-SQ8-nl387-np27 (query)                        8_666.05       280.35     8_946.40       0.6933          NaN        15.02
IVF-SQ8-nl387 (self)                              8_666.05     2_782.61    11_448.65       0.6932          NaN        15.02
IVF-SQ8-nl547-np23 (query)                       12_028.65       211.97    12_240.62       0.6928          NaN        15.08
IVF-SQ8-nl547-np27 (query)                       12_028.65       263.05    12_291.70       0.6932          NaN        15.08
IVF-SQ8-nl547-np33 (query)                       12_028.65       281.85    12_310.50       0.6933          NaN        15.08
IVF-SQ8-nl547 (self)                             12_028.65     2_771.27    14_799.92       0.6932          NaN        15.08
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 96D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   10.37    10_781.23    10_791.60       1.0000     0.000000        54.93
Exhaustive (self)                                    10.37   108_542.96   108_553.32       1.0000     0.000000        54.93
IVF-SQ8-nl273-np13 (query)                        6_549.70       204.91     6_754.61       0.7794          NaN        14.98
IVF-SQ8-nl273-np16 (query)                        6_549.70       273.32     6_823.01       0.7794          NaN        14.98
IVF-SQ8-nl273-np23 (query)                        6_549.70       352.13     6_901.83       0.7794          NaN        14.98
IVF-SQ8-nl273 (self)                              6_549.70     3_379.37     9_929.07       0.7835          NaN        14.98
IVF-SQ8-nl387-np19 (query)                       10_373.86       238.43    10_612.29       0.7794          NaN        15.02
IVF-SQ8-nl387-np27 (query)                       10_373.86       299.43    10_673.29       0.7793          NaN        15.02
IVF-SQ8-nl387 (self)                             10_373.86     2_875.30    13_249.16       0.7835          NaN        15.02
IVF-SQ8-nl547-np23 (query)                       13_244.99       262.59    13_507.58       0.7794          NaN        15.08
IVF-SQ8-nl547-np27 (query)                       13_244.99       340.44    13_585.43       0.7793          NaN        15.08
IVF-SQ8-nl547-np33 (query)                       13_244.99       395.41    13_640.40       0.7793          NaN        15.08
IVF-SQ8-nl547 (self)                             13_244.99     3_461.07    16_706.05       0.7835          NaN        15.08
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Gaussian - even more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.95    18_934.87    18_950.82       1.0000     0.000000        73.24
Exhaustive (self)                                    15.95   194_461.85   194_477.80       1.0000     0.000000        73.24
IVF-SQ8-nl273-np13 (query)                        9_838.84       304.67    10_143.50       0.8083          NaN        19.59
IVF-SQ8-nl273-np16 (query)                        9_838.84       344.57    10_183.40       0.8266          NaN        19.59
IVF-SQ8-nl273-np23 (query)                        9_838.84       469.28    10_308.12       0.8377          NaN        19.59
IVF-SQ8-nl273 (self)                              9_838.84     5_117.64    14_956.48       0.8374          NaN        19.59
IVF-SQ8-nl387-np19 (query)                       14_665.55       320.47    14_986.02       0.8177          NaN        19.65
IVF-SQ8-nl387-np27 (query)                       14_665.55       452.42    15_117.97       0.8361          NaN        19.65
IVF-SQ8-nl387 (self)                             14_665.55     4_303.32    18_968.87       0.8359          NaN        19.65
IVF-SQ8-nl547-np23 (query)                       21_667.36       316.05    21_983.40       0.8098          NaN        19.73
IVF-SQ8-nl547-np27 (query)                       21_667.36       369.80    22_037.16       0.8237          NaN        19.73
IVF-SQ8-nl547-np33 (query)                       21_667.36       415.28    22_082.64       0.8344          NaN        19.73
IVF-SQ8-nl547 (self)                             21_667.36     4_266.60    25_933.96       0.8344          NaN        19.73
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (Correlated - even more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.70    19_653.94    19_669.64       1.0000     0.000000        73.24
Exhaustive (self)                                    15.70   196_163.90   196_179.61       1.0000     0.000000        73.24
IVF-SQ8-nl273-np13 (query)                       11_341.14       476.41    11_817.55       0.6125          NaN        19.59
IVF-SQ8-nl273-np16 (query)                       11_341.14       717.13    12_058.27       0.6133          NaN        19.59
IVF-SQ8-nl273-np23 (query)                       11_341.14       692.84    12_033.98       0.6135          NaN        19.59
IVF-SQ8-nl273 (self)                             11_341.14     6_651.71    17_992.85       0.6146          NaN        19.59
IVF-SQ8-nl387-np19 (query)                       15_787.39       310.15    16_097.54       0.6129          NaN        19.65
IVF-SQ8-nl387-np27 (query)                       15_787.39       415.58    16_202.98       0.6135          NaN        19.65
IVF-SQ8-nl387 (self)                             15_787.39     4_048.90    19_836.29       0.6146          NaN        19.65
IVF-SQ8-nl547-np23 (query)                       20_287.77       308.77    20_596.54       0.6125          NaN        19.73
IVF-SQ8-nl547-np27 (query)                       20_287.77       343.13    20_630.91       0.6132          NaN        19.73
IVF-SQ8-nl547-np33 (query)                       20_287.77       392.86    20_680.63       0.6135          NaN        19.73
IVF-SQ8-nl547 (self)                             20_287.77     4_997.44    25_285.21       0.6146          NaN        19.73
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>SQ8 quantisations - Euclidean (LowRank - even more dimensions)</b>:</summary>
<pre><code>
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.14    20_062.82    20_077.96       1.0000     0.000000        73.24
Exhaustive (self)                                    15.14   186_690.24   186_705.38       1.0000     0.000000        73.24
IVF-SQ8-nl273-np13 (query)                        9_880.52       275.87    10_156.39       0.8081          NaN        19.59
IVF-SQ8-nl273-np16 (query)                        9_880.52       325.66    10_206.17       0.8081          NaN        19.59
IVF-SQ8-nl273-np23 (query)                        9_880.52       452.00    10_332.52       0.8081          NaN        19.59
IVF-SQ8-nl273 (self)                              9_880.52     4_458.39    14_338.91       0.8096          NaN        19.59
IVF-SQ8-nl387-np19 (query)                       14_377.07       303.29    14_680.36       0.8081          NaN        19.65
IVF-SQ8-nl387-np27 (query)                       14_377.07       381.00    14_758.07       0.8081          NaN        19.65
IVF-SQ8-nl387 (self)                             14_377.07     4_053.30    18_430.36       0.8096          NaN        19.65
IVF-SQ8-nl547-np23 (query)                       20_324.80       360.17    20_684.97       0.8081          NaN        19.73
IVF-SQ8-nl547-np27 (query)                       20_324.80       446.46    20_771.26       0.8081          NaN        19.73
IVF-SQ8-nl547-np33 (query)                       20_324.80       457.41    20_782.21       0.8081          NaN        19.73
IVF-SQ8-nl547 (self)                             20_324.80     4_001.27    24_326.07       0.8095          NaN        19.73
---------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### IVF-PQ

This index uses product quantisation. To note, the quantisation is quite harsh 
and hence, reduces the Recall quite substantially. Each vector gets reduced to
from *192 x 32 bits (192 x f32) = 768 bytes* to for 
*m = 32 (32 sub vectors) to 32 x u8 = 32 bytes*, a 
**24x reduction in memory usage** (of course with overhead from the cook book). 
However, it can still be useful in situation where good enough works and you 
have VERY large scale data.</br>
This version is run on a special version of the synthetic data that is less
affected by the curse of dimensionality! Also, there are some correlated 
features in there that can be theoretically better exploited by optimised
product quantisation, see below.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.
- *Number of subvectors (m)*: In how many subvectors to divide the given main
  vector. The initial dimensionality needs to be divisable by m.

The self queries here run on the compressed indices stored in the structure
itself. We can appreciate that the lossy compression affects the recall here. If
you wish to get great kNN graphs from these indices, you need to re-supply the
non-compressed data (at cost of memory!). Again, similar to `IVF-SQ8` the 
distances are difficult to interpret/compare against original vectors due to 
the heavy quantisation, thus, are not reported.

**Euclidean:**

With 128 dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   15.02     18288.27     18303.29       1.0000     0.000000        73.24
Exhaustive (self)                                    15.02    174754.84    174769.86       1.0000     0.000000        73.24
IVF-PQ-nl273-m16-np13 (query)                     17232.81       453.24     17686.06       0.5578          NaN         3.69
IVF-PQ-nl273-m16-np16 (query)                     17232.81       543.61     17776.42       0.5584          NaN         3.69
IVF-PQ-nl273-m16-np23 (query)                     17232.81       748.60     17981.42       0.5585          NaN         3.69
IVF-PQ-nl273-m16 (self)                           17232.81      7809.20     25042.01       0.4607          NaN         3.69
IVF-PQ-nl273-m32-np13 (query)                     19319.67       700.87     20020.53       0.7273          NaN         5.98
IVF-PQ-nl273-m32-np16 (query)                     19319.67       838.40     20158.07       0.7287          NaN         5.98
IVF-PQ-nl273-m32-np23 (query)                     19319.67      1202.35     20522.02       0.7291          NaN         5.98
IVF-PQ-nl273-m32 (self)                           19319.67     12045.04     31364.71       0.6608          NaN         5.98
IVF-PQ-nl387-m16-np19 (query)                     20417.56       541.78     20959.34       0.5580          NaN         3.75
IVF-PQ-nl387-m16-np27 (query)                     20417.56       750.22     21167.79       0.5585          NaN         3.75
IVF-PQ-nl387-m16 (self)                           20417.56      7585.76     28003.32       0.4624          NaN         3.75
IVF-PQ-nl387-m32-np19 (query)                     23508.80       891.40     24400.20       0.7302          NaN         6.04
IVF-PQ-nl387-m32-np27 (query)                     23508.80      1234.14     24742.94       0.7312          NaN         6.04
IVF-PQ-nl387-m32 (self)                           23508.80     12621.52     36130.32       0.6633          NaN         6.04
IVF-PQ-nl547-m16-np23 (query)                     26843.28       645.61     27488.89       0.5610          NaN         3.83
IVF-PQ-nl547-m16-np27 (query)                     26843.28       789.26     27632.54       0.5614          NaN         3.83
IVF-PQ-nl547-m16-np33 (query)                     26843.28       949.42     27792.70       0.5617          NaN         3.83
IVF-PQ-nl547-m16 (self)                           26843.28      9089.45     35932.72       0.4652          NaN         3.83
IVF-PQ-nl547-m32-np23 (query)                     29472.25       988.81     30461.06       0.7310          NaN         6.12
IVF-PQ-nl547-m32-np27 (query)                     29472.25      1289.50     30761.75       0.7322          NaN         6.12
IVF-PQ-nl547-m32-np33 (query)                     29472.25      1525.52     30997.77       0.7328          NaN         6.12
IVF-PQ-nl547-m32 (self)                           29472.25     13438.71     42910.97       0.6656          NaN         6.12
---------------------------------------------------------------------------------------------------------------------------
```

One can appreciate that PQ in this case can yield higher Recalls than just
SQ8 going from ca. 0.62 to 0.72. However, index building and query time are both
higher but at massive reduction of the memory finger print.

**With more dimensions**

With 192 dimensions. In this case, also m = 48, i.e., dividing the original 
vectors into 48 subvectors was tested.

```
===========================================================================================================================
Benchmark: 150k cells, 192D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   22.77     28768.85     28791.63       1.0000     0.000000       109.86
Exhaustive (self)                                    22.77    304666.13    304688.90       1.0000     0.000000       109.86
IVF-PQ-nl273-m16-np13 (query)                     24989.29       510.14     25499.42       0.4551          NaN         3.82
IVF-PQ-nl273-m16-np16 (query)                     24989.29       605.66     25594.95       0.4554          NaN         3.82
IVF-PQ-nl273-m16-np23 (query)                     24989.29       855.85     25845.13       0.4555          NaN         3.82
IVF-PQ-nl273-m16 (self)                           24989.29      8529.15     33518.44       0.3547          NaN         3.82
IVF-PQ-nl273-m32-np13 (query)                     30712.07      1053.19     31765.26       0.6326          NaN         6.11
IVF-PQ-nl273-m32-np16 (query)                     30712.07      1135.41     31847.48       0.6338          NaN         6.11
IVF-PQ-nl273-m32-np23 (query)                     30712.07      1735.02     32447.09       0.6340          NaN         6.11
IVF-PQ-nl273-m32 (self)                           30712.07     16236.94     46949.01       0.5492          NaN         6.11
IVF-PQ-nl273-m48-np13 (query)                     32343.71      1095.30     33439.00       0.7318          NaN         8.40
IVF-PQ-nl273-m48-np16 (query)                     32343.71      1290.47     33634.18       0.7335          NaN         8.40
IVF-PQ-nl273-m48-np23 (query)                     32343.71      1778.10     34121.81       0.7338          NaN         8.40
IVF-PQ-nl273-m48 (self)                           32343.71     18077.79     50421.50       0.6707          NaN         8.40
IVF-PQ-nl387-m16-np19 (query)                     34313.26       742.89     35056.15       0.4585          NaN         3.91
IVF-PQ-nl387-m16-np27 (query)                     34313.26       970.34     35283.60       0.4587          NaN         3.91
IVF-PQ-nl387-m16 (self)                           34313.26      9888.05     44201.31       0.3581          NaN         3.91
IVF-PQ-nl387-m32-np19 (query)                     36978.29      1152.81     38131.10       0.6336          NaN         6.20
IVF-PQ-nl387-m32-np27 (query)                     36978.29      1681.90     38660.18       0.6341          NaN         6.20
IVF-PQ-nl387-m32 (self)                           36978.29     16139.78     53118.07       0.5521          NaN         6.20
IVF-PQ-nl387-m48-np19 (query)                     38754.55      1530.66     40285.20       0.7333          NaN         8.49
IVF-PQ-nl387-m48-np27 (query)                     38754.55      2230.67     40985.21       0.7343          NaN         8.49
IVF-PQ-nl387-m48 (self)                           38754.55     19156.14     57910.69       0.6719          NaN         8.49
IVF-PQ-nl547-m16-np23 (query)                     43206.86       971.77     44178.62       0.4606          NaN         4.03
IVF-PQ-nl547-m16-np27 (query)                     43206.86      1133.87     44340.73       0.4610          NaN         4.03
IVF-PQ-nl547-m16-np33 (query)                     43206.86      1365.84     44572.70       0.4611          NaN         4.03
IVF-PQ-nl547-m16 (self)                           43206.86     13609.93     56816.79       0.3601          NaN         4.03
IVF-PQ-nl547-m32-np23 (query)                     51061.84      1310.41     52372.26       0.6351          NaN         6.32
IVF-PQ-nl547-m32-np27 (query)                     51061.84      1605.42     52667.26       0.6359          NaN         6.32
IVF-PQ-nl547-m32-np33 (query)                     51061.84      1786.68     52848.53       0.6362          NaN         6.32
IVF-PQ-nl547-m32 (self)                           51061.84     18043.41     69105.26       0.5530          NaN         6.32
IVF-PQ-nl547-m48-np23 (query)                     50243.09      1651.08     51894.17       0.7338          NaN         8.60
IVF-PQ-nl547-m48-np27 (query)                     50243.09      1931.16     52174.26       0.7351          NaN         8.60
IVF-PQ-nl547-m48-np33 (query)                     50243.09      2271.64     52514.73       0.7355          NaN         8.60
IVF-PQ-nl547-m48 (self)                           50243.09     21822.83     72065.92       0.6734          NaN         8.60
---------------------------------------------------------------------------------------------------------------------------
```

### IVF-OPQ

This index uses optimised product quantisation - this substantially increases
the build time. Similar to IVF-PQ, the quantisation is quite harsh and hence, 
reduces the recall quite substantially compared to exhaustive search. Each 
vector gets reduced to from *192 x 32 bits (192 x f32) = 768 bytes* to for 
*m = 32 (32 sub vectors) to 32 x u8 = 32 bytes*, a 
**24x reduction in memory usage** (of course with overhead from the cook book). 
However, it can still be useful in situation where good enough works and you 
have VERY large scale data. The theoretical benefits at least in this
synthetic data do not translate very well. IVF-PQ is usually more than enough, 
outside of cases in which a specific correlation structure can be exploited
by the optimised PQ. If in doubt, use the IVF-PQ index. 

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.
- *Number of subvectors (m)*: In how many subvectors to divide the given main
  vector. The initial dimensionality needs to be divisable by m.

Similar to IVF-OP, the self kNN generation is run on the compressed indices,
with the same loss of Recall due to the severe compression. Again, similar to 
`IVF-SQ8` the distances are difficult to interpret/compare against original 
vectors due to the heavy quantisation (plus rotation), thus, are not reported.

**Euclidean:**

With 128 dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 128D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   14.44     18310.79     18325.23       1.0000     0.000000        73.24
Exhaustive (self)                                    14.44    174999.86    175014.30       1.0000     0.000000        73.24
IVF-OPQ-nl273-m16-np13 (query)                    31730.82       513.62     32244.44       0.5560          NaN         3.76
IVF-OPQ-nl273-m16-np16 (query)                    31730.82       599.52     32330.34       0.5566          NaN         3.76
IVF-OPQ-nl273-m16-np23 (query)                    31730.82       847.82     32578.64       0.5568          NaN         3.76
IVF-OPQ-nl273-m16 (self)                          31730.82      7741.97     39472.79       0.4607          NaN         3.76
IVF-OPQ-nl273-m32-np13 (query)                    37127.97       705.35     37833.32       0.7288          NaN         6.05
IVF-OPQ-nl273-m32-np16 (query)                    37127.97       851.71     37979.68       0.7303          NaN         6.05
IVF-OPQ-nl273-m32-np23 (query)                    37127.97      1325.35     38453.33       0.7307          NaN         6.05
IVF-OPQ-nl273-m32 (self)                          37127.97     13479.71     50607.68       0.6652          NaN         6.05
IVF-OPQ-nl387-m16-np19 (query)                    34500.44       579.01     35079.45       0.5587          NaN         3.81
IVF-OPQ-nl387-m16-np27 (query)                    34500.44       753.76     35254.20       0.5592          NaN         3.81
IVF-OPQ-nl387-m16 (self)                          34500.44      7825.79     42326.24       0.4628          NaN         3.81
IVF-OPQ-nl387-m32-np19 (query)                    41614.44       899.07     42513.50       0.7327          NaN         6.10
IVF-OPQ-nl387-m32-np27 (query)                    41614.44      1266.10     42880.54       0.7337          NaN         6.10
IVF-OPQ-nl387-m32 (self)                          41614.44     12916.84     54531.27       0.6671          NaN         6.10
IVF-OPQ-nl547-m16-np23 (query)                    41269.15       669.71     41938.86       0.5618          NaN         3.89
IVF-OPQ-nl547-m16-np27 (query)                    41269.15       754.22     42023.38       0.5623          NaN         3.89
IVF-OPQ-nl547-m16-np33 (query)                    41269.15       928.76     42197.92       0.5626          NaN         3.89
IVF-OPQ-nl547-m16 (self)                          41269.15      9464.63     50733.78       0.4654          NaN         3.89
IVF-OPQ-nl547-m32-np23 (query)                    50331.22      1259.92     51591.14       0.7327          NaN         6.18
IVF-OPQ-nl547-m32-np27 (query)                    50331.22      1450.12     51781.34       0.7338          NaN         6.18
IVF-OPQ-nl547-m32-np33 (query)                    50331.22      1760.89     52092.10       0.7345          NaN         6.18
IVF-OPQ-nl547-m32 (self)                          50331.22     15280.95     65612.17       0.6689          NaN         6.18
---------------------------------------------------------------------------------------------------------------------------
```

**With more dimensions**

With 192 dimensions.

```
===========================================================================================================================
Benchmark: 150k cells, 192D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   21.79     27923.13     27944.92       1.0000     0.000000       109.86
Exhaustive (self)                                    21.79    288515.89    288537.68       1.0000     0.000000       109.86
IVF-OPQ-nl273-m16-np13 (query)                    47598.83       516.93     48115.76       0.4618          NaN         3.96
IVF-OPQ-nl273-m16-np16 (query)                    47598.83       631.76     48230.59       0.4623          NaN         3.96
IVF-OPQ-nl273-m16-np23 (query)                    47598.83       874.81     48473.64       0.4624          NaN         3.96
IVF-OPQ-nl273-m16 (self)                          47598.83      9324.44     56923.27       0.3650          NaN         3.96
IVF-OPQ-nl273-m32-np13 (query)                    57200.66       856.83     58057.49       0.6343          NaN         6.25
IVF-OPQ-nl273-m32-np16 (query)                    57200.66      1036.25     58236.91       0.6355          NaN         6.25
IVF-OPQ-nl273-m32-np23 (query)                    57200.66      1461.49     58662.15       0.6357          NaN         6.25
IVF-OPQ-nl273-m32 (self)                          57200.66     15067.78     72268.44       0.5543          NaN         6.25
IVF-OPQ-nl273-m48-np13 (query)                    65944.02      1049.44     66993.46       0.7348          NaN         8.54
IVF-OPQ-nl273-m48-np16 (query)                    65944.02      1275.15     67219.17       0.7365          NaN         8.54
IVF-OPQ-nl273-m48-np23 (query)                    65944.02      1780.09     67724.11       0.7368          NaN         8.54
IVF-OPQ-nl273-m48 (self)                          65944.02     18432.73     84376.75       0.6761          NaN         8.54
IVF-OPQ-nl387-m16-np19 (query)                    56256.93       719.50     56976.43       0.4643          NaN         4.05
IVF-OPQ-nl387-m16-np27 (query)                    56256.93       973.22     57230.15       0.4645          NaN         4.05
IVF-OPQ-nl387-m16 (self)                          56256.93     10343.36     66600.29       0.3679          NaN         4.05
IVF-OPQ-nl387-m32-np19 (query)                    65727.08      1162.76     66889.84       0.6371          NaN         6.34
IVF-OPQ-nl387-m32-np27 (query)                    65727.08      1599.09     67326.17       0.6380          NaN         6.34
IVF-OPQ-nl387-m32 (self)                          65727.08     16507.21     82234.29       0.5573          NaN         6.34
IVF-OPQ-nl387-m48-np19 (query)                    69889.65      1300.99     71190.64       0.7372          NaN         8.63
IVF-OPQ-nl387-m48-np27 (query)                    69889.65      1805.67     71695.32       0.7382          NaN         8.63
IVF-OPQ-nl387-m48 (self)                          69889.65     18702.80     88592.45       0.6769          NaN         8.63
IVF-OPQ-nl547-m16-np23 (query)                    63899.68       824.86     64724.54       0.4655          NaN         4.17
IVF-OPQ-nl547-m16-np27 (query)                    63899.68       934.51     64834.19       0.4659          NaN         4.17
IVF-OPQ-nl547-m16-np33 (query)                    63899.68      1122.14     65021.82       0.4660          NaN         4.17
IVF-OPQ-nl547-m16 (self)                          63899.68     11732.55     75632.23       0.3694          NaN         4.17
IVF-OPQ-nl547-m32-np23 (query)                    73802.83      1278.52     75081.35       0.6376          NaN         6.46
IVF-OPQ-nl547-m32-np27 (query)                    73802.83      1491.19     75294.03       0.6385          NaN         6.46
IVF-OPQ-nl547-m32-np33 (query)                    73802.83      1930.58     75733.41       0.6388          NaN         6.46
IVF-OPQ-nl547-m32 (self)                          73802.83     18357.73     92160.56       0.5582          NaN         6.46
IVF-OPQ-nl547-m48-np23 (query)                    81040.84      1474.67     82515.50       0.7382          NaN         8.75
IVF-OPQ-nl547-m48-np27 (query)                    81040.84      1709.56     82750.40       0.7394          NaN         8.75
IVF-OPQ-nl547-m48-np33 (query)                    81040.84      2057.34     83098.17       0.7398          NaN         8.75
IVF-OPQ-nl547-m48 (self)                          81040.84     21391.77    102432.61       0.6788          NaN         8.75
---------------------------------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.