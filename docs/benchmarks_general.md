## Standard indices benchmarks and parameter gridsearch

Below are all of the standard indices shown and their performance. These
are run via

```bash
# Run with default parameters
cargo run --example gridsearch_<INDEX> --release -- --distance euclidean

# Override specific parameters
cargo run --example gridsearch_<INDEX> --release -- --distance cosine

# Available parameters with their defaults:
# --n-cells 150_000
# --dim 32
# --n-clusters 25
# --k 15
# --seed 10101
# --distance cosine
# --data gaussian
```

### Annoy

Approximate nearest neighbours Oh Yeah. A tree-based method for vector searches.
Fast index building and good query speed.

**Key parameters:**

- *Number of trees (nt)*: The number of trees to generate in the forest
- *Search budget (s)*: The search budget per tree. If set to auto it uses
  `k * n_trees * 20`; versions with a `10x` or `5x` (i.e., less) are also shown.

**Euclidean:**

Below are the results for the Euclidean distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.02     23204.51     23208.53       1.0000     0.000000
Annoy-nt5-s:auto                           73.62       794.92       868.54       0.6408    81.683927
Annoy-nt5-s:10x                            73.62       570.38       644.00       0.5175    81.587548
Annoy-nt5-s:5x                             73.62       359.21       432.82       0.3713    81.392453
Annoy-nt10-s:auto                         100.36      1519.07      1619.43       0.8479    81.814313
Annoy-nt10-s:10x                          100.36      1061.24      1161.60       0.7378    81.763178
Annoy-nt10-s:5x                           100.36       669.30       769.66       0.5603    81.632177
Annoy-nt15-s:auto                         158.77      2145.18      2303.94       0.9338    81.853076
Annoy-nt15-s:10x                          158.77      1528.20      1686.96       0.8542    81.824079
Annoy-nt15-s:5x                           158.77       952.47      1111.24       0.6898    81.734461
Annoy-nt25-s:auto                         234.09      3317.58      3551.68       0.9854    81.872143
Annoy-nt25-s:10x                          234.09      2407.78      2641.87       0.9512    81.862444
Annoy-nt25-s:5x                           234.09      1448.60      1682.69       0.8394    81.818440
Annoy-nt50-s:auto                         450.88      5865.30      6316.18       0.9994    81.876491
Annoy-nt50-s:10x                          450.88      4471.22      4922.10       0.9959    81.875733
Annoy-nt50-s:5x                           450.88      2975.32      3426.20       0.9647    81.867209
Annoy-nt75-s:auto                         687.39      8105.55      8792.94       1.0000    81.876624
Annoy-nt75-s:10x                          687.39      6273.23      6960.62       0.9995    81.876551
Annoy-nt75-s:5x                           687.39      4329.62      5017.01       0.9909    81.874592
Annoy-nt100-s:auto                        900.40     10331.99     11232.39       1.0000    81.876631
Annoy-nt100-s:10x                         900.40      8045.77      8946.17       0.9999    81.876621
Annoy-nt100-s:5x                          900.40      5620.07      6520.48       0.9974    81.876118
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  5.02     23963.35     23968.37       1.0000     0.000000
Annoy-nt5-s:auto                           75.00       306.31       381.31       0.3459     0.007802
Annoy-nt5-s:10x                            75.00       318.52       393.52       0.3459     0.007802
Annoy-nt5-s:5x                             75.00       307.78       382.78       0.3378     0.008007
Annoy-nt10-s:auto                          98.01       574.53       672.54       0.5300     0.003925
Annoy-nt10-s:10x                           98.01       588.22       686.22       0.5300     0.003925
Annoy-nt10-s:5x                            98.01       581.35       679.36       0.5241     0.004002
Annoy-nt15-s:auto                         156.68       843.23       999.91       0.6573     0.002336
Annoy-nt15-s:10x                          156.68       845.93      1002.61       0.6573     0.002336
Annoy-nt15-s:5x                           156.68       844.36      1001.04       0.6535     0.002372
Annoy-nt25-s:auto                         243.94      1298.36      1542.29       0.8131     0.000991
Annoy-nt25-s:10x                          243.94      1294.02      1537.95       0.8131     0.000991
Annoy-nt25-s:5x                           243.94      1279.33      1523.26       0.8115     0.001001
Annoy-nt50-s:auto                         474.03      2666.33      3140.36       0.9540     0.000177
Annoy-nt50-s:10x                          474.03      2663.56      3137.59       0.9540     0.000177
Annoy-nt50-s:5x                           474.03      2679.21      3153.24       0.9538     0.000178
Annoy-nt75-s:auto                         710.69      4250.26      4960.95       0.9871     0.000041
Annoy-nt75-s:10x                          710.69      4506.63      5217.32       0.9871     0.000041
Annoy-nt75-s:5x                           710.69      4385.25      5095.94       0.9871     0.000041
Annoy-nt100-s:auto                        948.39      5345.15      6293.55       0.9960     0.000011
Annoy-nt100-s:10x                         948.39      5489.02      6437.41       0.9960     0.000011
Annoy-nt100-s:5x                          948.39      6014.70      6963.09       0.9960     0.000011
----------------------------------------------------------------------------------------------------
```

### HNSW

Hierarchical navigatable small worlds. A graph-based index that needs more time
to build the index. However, fast query speed.

**Key parameters:**

- *M (m)*: The number of connections between layers
- *EF construction (ef)*: The budget to generate good connections during 
  construction of the index.
- *EF search (s)*: The budget for the search queries. 

**Euclidean:**

Below are the results for the Euclidean distance measure for HSNW.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.43     25397.50     25400.93       1.0000     0.000000
HNSW-M16-ef50-s50                        2432.58      1516.70      3949.28       0.8879     4.623140
HNSW-M16-ef50-s75                        2432.58      2045.14      4477.73       0.9120     3.478027
HNSW-M16-ef50-s100                       2432.58      2521.75      4954.33       0.9244     2.700429
HNSW-M16-ef100-s50                       3746.53      1563.14      5309.67       0.9765     1.656335
HNSW-M16-ef100-s75                       3746.53      2041.77      5788.30       0.9863     0.914656
HNSW-M16-ef100-s100                      3746.53      2562.24      6308.77       0.9908     0.656868
HNSW-M16-ef200-s50                       4775.80      1568.20      6344.00       0.9656     4.918914
HNSW-M16-ef200-s75                       4775.80      2283.27      7059.07       0.9726     2.126443
HNSW-M16-ef200-s100                      4775.80      2672.91      7448.72       0.9748     1.470300
HNSW-M24-ef100-s50                       6282.72      1820.96      8103.68       0.9400     7.885647
HNSW-M24-ef100-s75                       6282.72      2428.13      8710.85       0.9479     5.724750
HNSW-M24-ef100-s100                      6282.72      2987.34      9270.06       0.9523     4.500737
HNSW-M24-ef200-s50                       7860.27      2036.46      9896.72       0.9795     6.384675
HNSW-M24-ef200-s75                       7860.27      2770.16     10630.42       0.9872     2.047901
HNSW-M24-ef200-s100                      7860.27      3347.92     11208.19       0.9884     1.540868
HNSW-M24-ef300-s50                       9479.17      2203.89     11683.06       0.9909     4.437885
HNSW-M24-ef300-s75                       9479.17      3059.72     12538.90       0.9939     2.657133
HNSW-M24-ef300-s100                      9479.17      3683.63     13162.80       0.9949     2.165261
HNSW-M32-ef200-s50                      12526.94      2176.31     14703.25       0.9906     5.062128
HNSW-M32-ef200-s75                      12526.94      3336.55     15863.49       0.9931     3.659440
HNSW-M32-ef200-s100                     12526.94      3672.57     16199.52       0.9940     3.158058
HNSW-M32-ef300-s50                      13970.91      2402.67     16373.58       0.9690     6.915551
HNSW-M32-ef300-s75                      13970.91      3492.69     17463.60       0.9749     3.424761
HNSW-M32-ef300-s100                     13970.91      3721.96     17692.87       0.9773     2.026151
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for HSNW.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.57     25403.83     25408.40       1.0000     0.000000
HNSW-M16-ef50-s50                        2571.50      1447.74      4019.24       0.8785     0.019600
HNSW-M16-ef50-s75                        2571.50      2161.70      4733.19       0.9228     0.013238
HNSW-M16-ef50-s100                       2571.50      2772.29      5343.78       0.9402     0.010435
HNSW-M16-ef100-s50                       3786.68      1627.38      5414.06       0.9421     0.015589
HNSW-M16-ef100-s75                       3786.68      2408.17      6194.85       0.9610     0.009599
HNSW-M16-ef100-s100                      3786.68      2905.29      6691.97       0.9701     0.006626
HNSW-M16-ef200-s50                       4931.96      2053.27      6985.23       0.9565     0.012105
HNSW-M16-ef200-s75                       4931.96      2338.34      7270.29       0.9692     0.005377
HNSW-M16-ef200-s100                      4931.96      3337.83      8269.78       0.9736     0.003202
HNSW-M24-ef100-s50                       6523.06      1932.47      8455.53       0.9317     0.025393
HNSW-M24-ef100-s75                       6523.06      2702.79      9225.84       0.9389     0.022835
HNSW-M24-ef100-s100                      6523.06      3247.83      9770.89       0.9427     0.021139
HNSW-M24-ef200-s50                       7704.75      2149.33      9854.08       0.9211     0.013118
HNSW-M24-ef200-s75                       7704.75      2539.71     10244.46       0.9356     0.005597
HNSW-M24-ef200-s100                      7704.75      3058.88     10763.64       0.9389     0.003964
HNSW-M24-ef300-s50                       9605.09      2108.23     11713.32       0.9540     0.023076
HNSW-M24-ef300-s75                       9605.09      2866.73     12471.81       0.9753     0.010799
HNSW-M24-ef300-s100                      9605.09      3362.17     12967.26       0.9813     0.007274
HNSW-M32-ef200-s50                      12424.73      2541.91     14966.64       0.9084     0.037275
HNSW-M32-ef200-s75                      12424.73      2956.55     15381.28       0.9198     0.030290
HNSW-M32-ef200-s100                     12424.73      3574.42     15999.15       0.9249     0.027301
HNSW-M32-ef300-s50                      13840.60      2282.51     16123.10       0.9660     0.016257
HNSW-M32-ef300-s75                      13840.60      2997.39     16837.99       0.9720     0.013021
HNSW-M32-ef300-s100                     13840.60      3584.61     17425.21       0.9745     0.011981
----------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index. Uses Voronoi cells to sub-partition the original data.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.


**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.41     24485.65     24489.06       1.0000     0.000000
IVF-nl273-np13                           1332.21      3223.10      4555.31       0.9954     0.036065
IVF-nl273-np16                           1332.21      4070.12      5402.33       0.9998     0.001561
IVF-nl273-np23                           1332.21      5655.86      6988.07       1.0000     0.000000
IVF-nl387-np19                           1885.38      3394.14      5279.52       0.9962     0.020320
IVF-nl387-np27                           1885.38      5047.88      6933.26       1.0000     0.000000
IVF-nl547-np23                           2715.44      3037.62      5753.06       0.9905     0.043293
IVF-nl547-np27                           2715.44      3440.96      6156.40       0.9971     0.009402
IVF-nl547-np33                           2715.44      4231.32      6946.76       0.9997     0.000616
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.67     25495.86     25500.53       1.0000     0.000000
IVF-nl273-np13                           4028.34      3813.64      7841.98       0.9957     0.000021
IVF-nl273-np16                           4028.34      4055.35      8083.69       0.9998     0.000001
IVF-nl273-np23                           4028.34      5689.40      9717.73       1.0000     0.000000
IVF-nl387-np19                           5812.22      3484.11      9296.33       0.9965     0.000012
IVF-nl387-np27                           5812.22      4857.49     10669.71       1.0000     0.000000
IVF-nl547-np23                           7955.42      3074.57     11029.98       0.9914     0.000027
IVF-nl547-np27                           7955.42      3648.66     11604.08       0.9974     0.000006
IVF-nl547-np33                           7955.42      4332.03     12287.44       0.9997     0.000000
----------------------------------------------------------------------------------------------------
```

### LSH

Locality sensitive hashing.

**Key parameters:**

- *Number of tables (nt)*: The number of independent hash tables to generate. 
  More tables improve recall at the cost of query time and memory.
- *Number of bits (nb)*: The bit resolution of the hash functions. Higher values 
  create finer partitions but may reduce collision rates.
- *Max candidates (s)*: The search budget limiting the number of candidates 
  examined. Set to 'auto' for full search or a fixed value (e.g., 5k) for faster 
  queries with lower recall.

**Euclidean:**

Below are the results for the Euclidean distance measure for LSH.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.51     25277.12     25280.62       1.0000     0.000000
LSH-nt10-nb8-s:auto                        68.72      6286.11      6354.83       0.9638     0.480160
LSH-nt10-nb8-s:5k                          68.72      3461.24      3529.96       0.8537     1.571043
LSH-nt20-nb8-s:auto                       125.52     10695.94     10821.46       0.9967     0.038897
LSH-nt20-nb8-s:5k                         125.52      3618.50      3744.02       0.8595     1.469591
LSH-nt25-nb8-s:auto                       148.28     13496.94     13645.22       0.9987     0.014704
LSH-nt25-nb8-s:5k                         148.28      3641.49      3789.77       0.8596     1.469395
LSH-nt10-nb10-s:auto                       82.04      3977.43      4059.46       0.9114     1.283957
LSH-nt10-nb10-s:5k                         82.04      2786.54      2868.58       0.8493     1.675435
LSH-nt20-nb10-s:auto                      149.43      6301.92      6451.35       0.9818     0.229392
LSH-nt20-nb10-s:5k                        149.43      3149.04      3298.47       0.8902     0.944406
LSH-nt25-nb10-s:auto                      171.04      7827.87      7998.90       0.9914     0.104942
LSH-nt25-nb10-s:5k                        171.04      3170.87      3341.91       0.8932     0.897108
LSH-nt10-nb12-s:auto                       93.51      3165.84      3259.35       0.8431     2.506719
LSH-nt10-nb12-s:5k                         93.51      2397.94      2491.45       0.8021     2.687833
LSH-nt20-nb12-s:auto                      174.68      4968.07      5142.75       0.9547     0.632133
LSH-nt20-nb12-s:5k                        174.68      2873.25      3047.93       0.8929     0.977663
LSH-nt25-nb12-s:auto                      218.89      5482.12      5701.01       0.9720     0.373990
LSH-nt25-nb12-s:5k                        218.89      3127.12      3346.01       0.9044     0.780383
LSH-nt10-nb16-s:auto                      142.56      2076.96      2219.52       0.6891     6.654130
LSH-nt10-nb16-s:5k                        142.56      1735.79      1878.35       0.6680     6.721914
LSH-nt20-nb16-s:auto                      273.09      3131.83      3404.92       0.8391     2.829489
LSH-nt20-nb16-s:5k                        273.09      2411.04      2684.13       0.8022     2.961812
LSH-nt25-nb16-s:auto                      298.95      3610.89      3909.84       0.8751     2.077903
LSH-nt25-nb16-s:5k                        298.95      2199.53      2498.48       0.8346     2.227713
LSH-nt50-nb16-s:auto                      610.82      5648.65      6259.48       0.9597     0.585007
LSH-nt50-nb16-s:5k                        610.82      2942.43      3553.25       0.9107     0.793054
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for LSH.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.71     25246.81     25251.51       1.0000     0.000000
LSH-nt10-nb8-s:auto                        33.95      9194.68      9228.63       0.9807     0.000164
LSH-nt10-nb8-s:5k                          33.95      3786.69      3820.64       0.8236     0.001536
LSH-nt20-nb8-s:auto                       240.83     16340.34     16581.17       0.9992     0.000006
LSH-nt20-nb8-s:5k                         240.83      4036.03      4276.86       0.8239     0.001533
LSH-nt25-nb8-s:auto                        71.34     19574.58     19645.92       0.9998     0.000002
LSH-nt25-nb8-s:5k                          71.34      4018.83      4090.17       0.8239     0.001533
LSH-nt10-nb10-s:auto                       43.02      5901.33      5944.35       0.9543     0.000426
LSH-nt10-nb10-s:5k                         43.02      3280.28      3323.30       0.8669     0.000950
LSH-nt20-nb10-s:auto                       75.90      8585.23      8661.13       0.9948     0.000042
LSH-nt20-nb10-s:5k                         75.90      3665.60      3741.50       0.8797     0.000791
LSH-nt25-nb10-s:auto                       88.88      9786.34      9875.21       0.9981     0.000016
LSH-nt25-nb10-s:5k                         88.88      3682.90      3771.77       0.8799     0.000789
LSH-nt10-nb12-s:auto                       54.52      3750.15      3804.68       0.9078     0.000924
LSH-nt10-nb12-s:5k                         54.52      2570.94      2625.46       0.8592     0.001114
LSH-nt20-nb12-s:auto                       95.20      6119.94      6215.14       0.9813     0.000162
LSH-nt20-nb12-s:5k                         95.20      2846.66      2941.86       0.9078     0.000521
LSH-nt25-nb12-s:auto                      111.64      6480.08      6591.72       0.9904     0.000084
LSH-nt25-nb12-s:5k                        111.64      2909.10      3020.74       0.9116     0.000480
LSH-nt10-nb16-s:auto                       74.11      2180.71      2254.82       0.7512     0.003265
LSH-nt10-nb16-s:5k                         74.11      1827.45      1901.56       0.7274     0.003324
LSH-nt20-nb16-s:auto                      121.68      3432.00      3553.68       0.9067     0.001003
LSH-nt20-nb16-s:5k                        121.68      2243.54      2365.22       0.8673     0.001112
LSH-nt25-nb16-s:auto                      149.79      4062.47      4212.26       0.9344     0.000677
LSH-nt25-nb16-s:5k                        149.79      2494.87      2644.66       0.8916     0.000801
LSH-nt50-nb16-s:auto                      287.78      5938.67      6226.45       0.9852     0.000139
LSH-nt50-nb16-s:5k                        287.78      3010.85      3298.63       0.9332     0.000314
----------------------------------------------------------------------------------------------------
```

### NNDescent

The NNDescent implementation in this crate, heavily inspired by the amazing
[PyNNDescent](https://github.com/lmcinnes/pynndescent), shows a very good
compromise between index building and fast querying. It's a great arounder
that reaches easily performance of ≥0.98 Recalls@k neighbours. You can even
heavily short cut the initialisation of the index with only 12 trees (instead
of 32) and get in 4 seconds to a recall ≥0.9 (compared to 48 seconds for 
exhaustive search)!

**Key parameters:**

- *Number of trees (nt)*: Number of trees to use for the initialisation. If set
  to auto, it defaults to 32.
- *Search budget (s)*: The search budget for the exploration of the graph during
  querying. Here it defaults to `k * 2` (with min 60, maximum 200).
- *Diversify probability (dp)*: This is based on the original papers leveraging
  NNDescent and it is supposed to remove redundant edges from the graph to
  increase query speed.

**Euclidean:**

Below are the results for the Euclidean distance measure for NNDescent
implementation in this `crate`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  3.87     23929.25     23933.11       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                2021.42      1226.24      3247.65       0.9687     0.189218
NNDescent-nt24-s:auto-dp0                3396.03      1228.06      4624.09       0.9885     0.061129
NNDescent-nt:auto-s75-dp0                3050.66      1737.59      4788.25       0.9939     0.031297
NNDescent-nt:auto-s100-dp0               3050.66      1816.99      4867.65       0.9960     0.020544
NNDescent-nt:auto-s:auto-dp0             3050.66       969.59      4020.25       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.25          2820.79      1163.20      3983.99       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp0.5           3532.68       982.95      4515.63       0.9893     0.056484
NNDescent-nt:auto-s:auto-dp1             2849.11      1125.53      3974.64       0.9893     0.056484
----------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for NNDescent
implementation in this `crate`.

```
====================================================================================================
Benchmark: 150k cells, 32D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                  4.63     28409.27     28413.90       1.0000     0.000000
NNDescent-nt12-s:auto-dp0                7270.95       999.62      8270.57       0.9997     0.000001
NNDescent-nt24-s:auto-dp0                6613.81       897.66      7511.47       0.9998     0.000001
NNDescent-nt:auto-s75-dp0                6025.44      1579.33      7604.77       0.9999     0.000000
NNDescent-nt:auto-s100-dp0               6025.44      1840.56      7866.00       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0             6025.44      1073.47      7098.92       0.9997     0.000001
NNDescent-nt:auto-s:auto-dp0.25          6849.90      1072.01      7921.91       0.9899     0.000030
NNDescent-nt:auto-s:auto-dp0.5           6818.67       861.79      7680.46       0.9806     0.000062
NNDescent-nt:auto-s:auto-dp1             6338.33       763.17      7101.50       0.9613     0.000135
----------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.