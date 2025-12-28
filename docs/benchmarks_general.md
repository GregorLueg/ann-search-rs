## Standard indices benchmarks and parameter gridsearch

Below are all of the standard indices shown and their performance. These
are run via:

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
# --seed 42
# --distance cosine
# --data gaussian
```

If you wish to run all of them, you can just run:

```bash
bash ./examples/run_benchmarks.sh --standard
```

### Annoy

Approximate nearest neighbours Oh Yeah. A tree-based method for vector searches.
Fast index building and good query speed.

**Key parameters:**

- *Number of trees (nt)*: The number of trees to generate in the forest
- *Search budget (s)*: The search budget per tree. If set to auto it uses
  `k * n_trees * 20`; versions with a `10x` or `5x` (i.e., less) are also shown.

**Euclidean:**

Below are the results for the Euclidean distance measure for Annoy. Self is
queried with the default search budget.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.20      2321.72      2324.91       1.0000     0.000000
Exhaustive (self)                                     3.20     23238.83     23242.03       1.0000     0.000000
Annoy-nt5-s:auto (query)                             77.26        91.55       168.81       0.6834    40.648110
Annoy-nt5-s:10x (query)                              77.26        58.92       136.18       0.5240    40.565845
Annoy-nt5-s:5x (query)                               77.26        37.93       115.19       0.3732    40.431273
Annoy-nt5 (self)                                     77.26       908.45       985.71       0.6838    40.084992
Annoy-nt10-s:auto (query)                           101.96       170.10       272.06       0.8810    40.727710
Annoy-nt10-s:10x (query)                            101.96       108.14       210.10       0.7412    40.683223
Annoy-nt10-s:5x (query)                             101.96        68.69       170.65       0.5626    40.594216
Annoy-nt10 (self)                                   101.96      1693.62      1795.58       0.8804    40.163717
Annoy-nt15-s:auto (query)                           162.02       239.97       401.99       0.9524    40.748981
Annoy-nt15-s:10x (query)                            162.02       155.30       317.32       0.8546    40.723903
Annoy-nt15-s:5x (query)                             162.02        97.09       259.10       0.6907    40.662767
Annoy-nt15 (self)                                   162.02      2382.65      2544.67       0.9516    40.184645
Annoy-nt25-s:auto (query)                           250.71       364.46       615.17       0.9908    40.758739
Annoy-nt25-s:10x (query)                            250.71       255.92       506.63       0.9508    40.750722
Annoy-nt25-s:5x (query)                             250.71       158.25       408.96       0.8410    40.720771
Annoy-nt25 (self)                                   250.71      3666.93      3917.65       0.9906    40.194411
Annoy-nt50-s:auto (query)                           461.24       640.45      1101.69       0.9997    40.760798
Annoy-nt50-s:10x (query)                            461.24       418.16       879.39       0.9957    40.760169
Annoy-nt50-s:5x (query)                             461.24       279.13       740.37       0.9644    40.754108
Annoy-nt50 (self)                                   461.24      5855.93      6317.17       0.9997    40.196443
Annoy-nt75-s:auto (query)                           674.85       868.49      1543.34       1.0000    40.760868
Annoy-nt75-s:10x (query)                            674.85       625.04      1299.89       0.9995    40.760801
Annoy-nt75-s:5x (query)                             674.85       423.29      1098.14       0.9912    40.759442
Annoy-nt75 (self)                                   674.85      8589.54      9264.39       1.0000    40.196503
Annoy-nt100-s:auto (query)                          894.90      1060.41      1955.31       1.0000    40.760873
Annoy-nt100-s:10x (query)                           894.90       776.66      1671.56       0.9999    40.760864
Annoy-nt100-s:5x (query)                            894.90       570.91      1465.81       0.9975    40.760539
Annoy-nt100 (self)                                  894.90     10514.74     11409.64       1.0000    40.196506
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.40      2421.35      2425.75       1.0000     0.000000
Exhaustive (self)                                     4.40     23965.20     23969.61       1.0000     0.000000
Annoy-nt5-s:auto (query)                             76.91        34.17       111.08       0.3536     0.004314
Annoy-nt5-s:10x (query)                              76.91        34.30       111.21       0.3536     0.004314
Annoy-nt5-s:5x (query)                               76.91        32.23       109.15       0.3432     0.004456
Annoy-nt5 (self)                                     76.91       319.02       395.93       0.3540     0.004254
Annoy-nt10-s:auto (query)                           102.04        60.71       162.75       0.5411     0.002165
Annoy-nt10-s:10x (query)                            102.04        60.88       162.92       0.5411     0.002165
Annoy-nt10-s:5x (query)                             102.04        59.28       161.32       0.5329     0.002226
Annoy-nt10 (self)                                   102.04       635.68       737.73       0.5394     0.002140
Annoy-nt15-s:auto (query)                           160.27        84.89       245.16       0.6685     0.001280
Annoy-nt15-s:10x (query)                            160.27        83.18       243.45       0.6685     0.001280
Annoy-nt15-s:5x (query)                             160.27        83.33       243.60       0.6626     0.001312
Annoy-nt15 (self)                                   160.27       850.49      1010.75       0.6668     0.001272
Annoy-nt25-s:auto (query)                           258.53       147.18       405.71       0.8222     0.000544
Annoy-nt25-s:10x (query)                            258.53       136.28       394.81       0.8222     0.000544
Annoy-nt25-s:5x (query)                             258.53       144.69       403.22       0.8195     0.000554
Annoy-nt25 (self)                                   258.53      1414.26      1672.79       0.8214     0.000537
Annoy-nt50-s:auto (query)                           490.84       299.59       790.43       0.9575     0.000095
Annoy-nt50-s:10x (query)                            490.84       271.56       762.40       0.9575     0.000095
Annoy-nt50-s:5x (query)                             490.84       278.94       769.78       0.9569     0.000096
Annoy-nt50 (self)                                   490.84      2837.44      3328.28       0.9573     0.000095
Annoy-nt75-s:auto (query)                           752.84       379.97      1132.81       0.9887     0.000021
Annoy-nt75-s:10x (query)                            752.84       393.61      1146.44       0.9887     0.000021
Annoy-nt75-s:5x (query)                             752.84       409.87      1162.70       0.9885     0.000022
Annoy-nt75 (self)                                   752.84      4164.32      4917.16       0.9885     0.000022
Annoy-nt100-s:auto (query)                         1092.01       594.21      1686.22       0.9965     0.000006
Annoy-nt100-s:10x (query)                          1092.01       593.75      1685.76       0.9965     0.000006
Annoy-nt100-s:5x (query)                           1092.01       613.05      1705.06       0.9965     0.000006
Annoy-nt100 (self)                                 1092.01      5705.68      6797.69       0.9964     0.000006
--------------------------------------------------------------------------------------------------------------
```

### HNSW

Hierarchical navigatable small worlds. A graph-based index that needs more time
to build the index. However, fast query speed.

**Key parameters:**

- *M (m)*: The number of connections between layers
- *EF construction (ef)*: The budget to generate good connections during 
  construction of the index.
- *EF search (s)*: The budget for the search queries. 

Self is queried with `s=100`.

**Euclidean:**

Below are the results for the Euclidean distance measure for HSNW.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.15      2560.58      2563.73       1.0000     0.000000
Exhaustive (self)                                     3.15     24184.98     24188.13       1.0000     0.000000
HNSW-M16-ef50-s50 (query)                          2191.75       135.70      2327.44       0.7281     6.433825
HNSW-M16-ef50-s75 (query)                          2191.75       181.06      2372.81       0.7767     4.239678
HNSW-M16-ef50-s100 (query)                         2191.75       223.85      2415.59       0.8019     2.903725
HNSW-M16-ef50 (self)                               2191.75      2226.22      4417.97       0.7978     3.363466
HNSW-M16-ef100-s50 (query)                         3421.45       178.55      3600.00       0.9529     1.868497
HNSW-M16-ef100-s75 (query)                         3421.45       220.33      3641.77       0.9620     1.014502
HNSW-M16-ef100-s100 (query)                        3421.45       288.56      3710.01       0.9660     0.665652
HNSW-M16-ef100 (self)                              3421.45      3582.11      7003.56       0.9657     0.762760
HNSW-M16-ef200-s50 (query)                         4709.33       164.50      4873.83       0.8976    26.215633
HNSW-M16-ef200-s75 (query)                         4709.33       221.74      4931.08       0.9202    10.703025
HNSW-M16-ef200-s100 (query)                        4709.33       314.27      5023.60       0.9277     6.512692
HNSW-M16-ef200 (self)                              4709.33      2640.54      7349.88       0.9327     6.018998
HNSW-M24-ef100-s50 (query)                         6300.21       163.66      6463.88       0.7192    10.205247
HNSW-M24-ef100-s75 (query)                         6300.21       207.00      6507.21       0.7250     7.544174
HNSW-M24-ef100-s100 (query)                        6300.21       254.74      6554.96       0.7395     5.336395
HNSW-M24-ef100 (self)                              6300.21      2518.50      8818.72       0.7364     5.537069
HNSW-M24-ef200-s50 (query)                         7579.19       200.46      7779.65       0.9209     7.221936
HNSW-M24-ef200-s75 (query)                         7579.19       267.16      7846.35       0.9332     1.983098
HNSW-M24-ef200-s100 (query)                        7579.19       316.69      7895.87       0.9369     1.278188
HNSW-M24-ef200 (self)                              7579.19      3121.24     10700.43       0.9373     1.468927
HNSW-M24-ef300-s50 (query)                         8684.03       221.45      8905.48       0.9680    13.511406
HNSW-M24-ef300-s75 (query)                         8684.03       260.16      8944.19       0.9772     7.771752
HNSW-M24-ef300-s100 (query)                        8684.03       314.63      8998.66       0.9813     5.381370
HNSW-M24-ef300 (self)                              8684.03      3149.38     11833.40       0.9815     5.510665
HNSW-M32-ef200-s50 (query)                        11644.06       224.78     11868.84       0.9695     8.868971
HNSW-M32-ef200-s75 (query)                        11644.06       285.10     11929.16       0.9775     1.907418
HNSW-M32-ef200-s100 (query)                       11644.06       347.93     11991.99       0.9794     0.571944
HNSW-M32-ef200 (self)                             11644.06      3408.83     15052.89       0.9785     0.723485
HNSW-M32-ef300-s50 (query)                        12608.65       226.77     12835.42       0.9327    12.056741
HNSW-M32-ef300-s75 (query)                        12608.65       294.43     12903.08       0.9439     3.886539
HNSW-M32-ef300-s100 (query)                       12608.65       354.19     12962.83       0.9463     2.146095
HNSW-M32-ef300 (self)                             12608.65      3532.50     16141.14       0.9468     2.072899
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for HSNW.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.70      2431.64      2436.33       1.0000     0.000000
Exhaustive (self)                                     4.70     24458.25     24462.94       1.0000     0.000000
HNSW-M16-ef50-s50 (query)                          2216.30       128.97      2345.26       0.7656     0.006658
HNSW-M16-ef50-s75 (query)                          2216.30       166.70      2383.00       0.7947     0.003925
HNSW-M16-ef50-s100 (query)                         2216.30       206.98      2423.27       0.8141     0.002877
HNSW-M16-ef50 (self)                               2216.30      2202.12      4418.42       0.8211     0.002752
HNSW-M16-ef100-s50 (query)                         3497.48       160.28      3657.76       0.9445     0.004571
HNSW-M16-ef100-s75 (query)                         3497.48       224.72      3722.20       0.9602     0.002167
HNSW-M16-ef100-s100 (query)                        3497.48       248.73      3746.21       0.9742     0.001186
HNSW-M16-ef100 (self)                              3497.48      2596.27      6093.75       0.9745     0.001205
HNSW-M16-ef200-s50 (query)                         4903.33       185.57      5088.89       0.9562     0.011812
HNSW-M16-ef200-s75 (query)                         4903.33       291.00      5194.33       0.9710     0.001564
HNSW-M16-ef200-s100 (query)                        4903.33       291.45      5194.77       0.9734     0.000626
HNSW-M16-ef200 (self)                              4903.33      3019.62      7922.95       0.9714     0.000882
HNSW-M24-ef100-s50 (query)                         6624.42       191.90      6816.32       0.9472     0.000871
HNSW-M24-ef100-s75 (query)                         6624.42       244.40      6868.82       0.9517     0.000750
HNSW-M24-ef100-s100 (query)                        6624.42       300.09      6924.51       0.9534     0.000741
HNSW-M24-ef100 (self)                              6624.42      3236.14      9860.56       0.9531     0.000553
HNSW-M24-ef200-s50 (query)                         8032.75       199.63      8232.38       0.9679     0.005253
HNSW-M24-ef200-s75 (query)                         8032.75       266.61      8299.36       0.9756     0.002814
HNSW-M24-ef200-s100 (query)                        8032.75       324.92      8357.67       0.9803     0.001307
HNSW-M24-ef200 (self)                              8032.75      3211.02     11243.77       0.9794     0.001628
HNSW-M24-ef300-s50 (query)                         9503.78       211.29      9715.07       0.9852     0.007809
HNSW-M24-ef300-s75 (query)                         9503.78       264.02      9767.80       0.9924     0.002536
HNSW-M24-ef300-s100 (query)                        9503.78       351.06      9854.84       0.9939     0.001697
HNSW-M24-ef300 (self)                              9503.78      3627.52     13131.30       0.9934     0.001756
HNSW-M32-ef200-s50 (query)                        13448.79       229.28     13678.07       0.9668     0.024336
HNSW-M32-ef200-s75 (query)                        13448.79       300.85     13749.64       0.9900     0.006695
HNSW-M32-ef200-s100 (query)                       13448.79       385.51     13834.30       0.9976     0.000884
HNSW-M32-ef200 (self)                             13448.79      3500.21     16949.00       0.9977     0.000819
HNSW-M32-ef300-s50 (query)                        12722.74       211.12     12933.86       0.9786     0.005085
HNSW-M32-ef300-s75 (query)                        12722.74       277.48     13000.22       0.9839     0.001891
HNSW-M32-ef300-s100 (query)                       12722.74       331.72     13054.46       0.9855     0.001155
HNSW-M32-ef300 (self)                             12722.74      3312.16     16034.90       0.9851     0.001187
--------------------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index. Uses Voronoi cells to sub-partition the original data.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

Self query was done with `2 * sqrt(nlist)`.

**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.16      2386.06      2389.22       1.0000     0.000000
Exhaustive (self)                                     3.16     26743.51     26746.66       1.0000     0.000000
IVF-nl273-np13 (query)                             1512.08       345.03      1857.11       0.9889     0.034523
IVF-nl273-np16 (query)                             1512.08       430.61      1942.69       0.9967     0.006608
IVF-nl273-np23 (query)                             1512.08       621.87      2133.95       1.0000     0.000000
IVF-nl273 (self)                                   1512.08      6199.14      7711.23       1.0000     0.000000
IVF-nl387-np19 (query)                             1989.11       331.12      2320.23       0.9926     0.031089
IVF-nl387-np27 (query)                             1989.11       514.04      2503.15       0.9998     0.000653
IVF-nl387 (self)                                   1989.11      5942.09      7931.20       0.9998     0.000690
IVF-nl547-np23 (query)                             2855.22       341.96      3197.18       0.9882     0.041944
IVF-nl547-np27 (query)                             2855.22       374.29      3229.51       0.9958     0.014725
IVF-nl547-np33 (query)                             2855.22       465.89      3321.11       0.9997     0.001035
IVF-nl547 (self)                                   2855.22      4457.31      7312.53       0.9997     0.001003
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.54      2402.57      2407.11       1.0000     0.000000
Exhaustive (self)                                     4.54     24042.20     24046.74       1.0000     0.000000
IVF-nl273-np13 (query)                             1359.70       321.73      1681.42       0.9894     0.000025
IVF-nl273-np16 (query)                             1359.70       395.71      1755.40       0.9968     0.000006
IVF-nl273-np23 (query)                             1359.70       558.38      1918.07       1.0000     0.000000
IVF-nl273 (self)                                   1359.70      5551.90      6911.59       1.0000     0.000000
IVF-nl387-np19 (query)                             1918.06       324.55      2242.61       0.9930     0.000019
IVF-nl387-np27 (query)                             1918.06       449.00      2367.06       0.9998     0.000000
IVF-nl387 (self)                                   1918.06      4457.70      6375.76       0.9998     0.000000
IVF-nl547-np23 (query)                             2682.51       287.39      2969.90       0.9892     0.000025
IVF-nl547-np27 (query)                             2682.51       330.24      3012.75       0.9963     0.000009
IVF-nl547-np33 (query)                             2682.51       397.90      3080.41       0.9997     0.000001
IVF-nl547 (self)                                   2682.51      3955.37      6637.88       0.9997     0.000001
--------------------------------------------------------------------------------------------------------------
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
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.24      2511.00      2514.24       1.0000     0.000000
Exhaustive (self)                                     3.24     24575.02     24578.26       1.0000     0.000000
LSH-nt10-nb8-s:auto (query)                          57.81       691.09       748.89       0.9880     0.150786
LSH-nt10-nb8-s:5k (query)                            57.81       367.54       425.35       0.8853     0.682000
LSH-nt10-nb8 (self)                                  57.81      7072.97      7130.77       0.9879     0.148408
LSH-nt20-nb8-s:auto (query)                         116.83      1103.08      1219.91       0.9990     0.011816
LSH-nt20-nb8-s:5k (query)                           116.83       501.24       618.07       0.8866     0.657057
LSH-nt20-nb8 (self)                                 116.83     11107.17     11224.00       0.9990     0.010957
LSH-nt25-nb8-s:auto (query)                         143.98      1320.14      1464.12       0.9997     0.003853
LSH-nt25-nb8-s:5k (query)                           143.98       389.27       533.25       0.8866     0.657057
LSH-nt25-nb8 (self)                                 143.98     11988.84     12132.82       0.9996     0.003632
LSH-nt10-nb10-s:auto (query)                         74.52       532.17       606.68       0.9699     0.414554
LSH-nt10-nb10-s:5k (query)                           74.52       299.53       374.04       0.8848     0.679656
LSH-nt10-nb10 (self)                                 74.52      5223.73      5298.25       0.9700     0.407188
LSH-nt20-nb10-s:auto (query)                        135.04       746.89       881.93       0.9951     0.060176
LSH-nt20-nb10-s:5k (query)                          135.04       304.61       439.66       0.8992     0.425955
LSH-nt20-nb10 (self)                                135.04      7764.76      7899.80       0.9949     0.061070
LSH-nt25-nb10-s:auto (query)                        179.34       824.96      1004.30       0.9974     0.030311
LSH-nt25-nb10-s:5k (query)                          179.34       324.17       503.51       0.8997     0.417316
LSH-nt25-nb10 (self)                                179.34      8412.66      8592.00       0.9973     0.031546
LSH-nt10-nb12-s:auto (query)                         90.32       430.12       520.44       0.9411     0.804150
LSH-nt10-nb12-s:5k (query)                           90.32       278.95       369.26       0.8786     0.967038
LSH-nt10-nb12 (self)                                 90.32      4325.35      4415.67       0.9420     0.793338
LSH-nt20-nb12-s:auto (query)                        178.42       592.65       771.07       0.9844     0.197321
LSH-nt20-nb12-s:5k (query)                          178.42       302.58       481.00       0.9095     0.422210
LSH-nt20-nb12 (self)                                178.42      5929.80      6108.22       0.9842     0.200060
LSH-nt25-nb12-s:auto (query)                        219.77       662.37       882.14       0.9906     0.118465
LSH-nt25-nb12-s:5k (query)                          219.77       297.99       517.76       0.9133     0.361454
LSH-nt25-nb12 (self)                                219.77      6907.92      7127.69       0.9903     0.122014
LSH-nt10-nb16-s:auto (query)                        122.40       312.05       434.46       0.8498     2.429547
LSH-nt10-nb16-s:5k (query)                          122.40       225.44       347.85       0.8060     2.509321
LSH-nt10-nb16 (self)                                122.40      3191.31      3313.72       0.8527     2.402690
LSH-nt20-nb16-s:auto (query)                        220.21       468.02       688.22       0.9416     0.933309
LSH-nt20-nb16-s:5k (query)                          220.21       284.96       505.17       0.8775     1.076314
LSH-nt20-nb16 (self)                                220.21      5400.81      5621.02       0.9422     0.931024
LSH-nt25-nb16-s:auto (query)                        290.44       625.38       915.82       0.9589     0.613957
LSH-nt25-nb16-s:5k (query)                          290.44       321.26       611.70       0.8914     0.770405
LSH-nt25-nb16 (self)                                290.44      6437.16      6727.61       0.9588     0.626807
LSH-nt50-nb16-s:auto (query)                        547.73       782.36      1330.10       0.9856     0.188848
LSH-nt50-nb16-s:5k (query)                          547.73       289.53       837.26       0.9138     0.365119
LSH-nt50-nb16 (self)                                547.73      8455.91      9003.64       0.9856     0.191776
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for LSH.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.85      2669.80      2674.65       1.0000     0.000000
Exhaustive (self)                                     4.85     25885.21     25890.07       1.0000     0.000000
LSH-nt10-nb8-s:auto (query)                          34.44       880.12       914.56       0.9942     0.000049
LSH-nt10-nb8-s:5k (query)                            34.44       388.92       423.36       0.8717     0.000877
LSH-nt10-nb8 (self)                                  34.44      8732.25      8766.69       0.9945     0.000045
LSH-nt20-nb8-s:auto (query)                          58.44      1484.08      1542.52       0.9997     0.000002
LSH-nt20-nb8-s:5k (query)                            58.44       362.32       420.77       0.8719     0.000875
LSH-nt20-nb8 (self)                                  58.44     14728.32     14786.76       0.9998     0.000002
LSH-nt25-nb8-s:auto (query)                          73.03      1766.83      1839.85       0.9999     0.000001
LSH-nt25-nb8-s:5k (query)                            73.03       368.78       441.80       0.8719     0.000875
LSH-nt25-nb8 (self)                                  73.03     17711.12     17784.15       0.9999     0.000000
LSH-nt10-nb10-s:auto (query)                         38.33       674.34       712.67       0.9806     0.000171
LSH-nt10-nb10-s:5k (query)                           38.33       334.08       372.41       0.8911     0.000557
LSH-nt10-nb10 (self)                                 38.33      6829.14      6867.47       0.9808     0.000167
LSH-nt20-nb10-s:auto (query)                         79.04      1115.32      1194.37       0.9983     0.000015
LSH-nt20-nb10-s:5k (query)                           79.04       408.66       487.70       0.8971     0.000485
LSH-nt20-nb10 (self)                                 79.04     11264.67     11343.72       0.9982     0.000015
LSH-nt25-nb10-s:auto (query)                         98.65      1163.30      1261.95       0.9993     0.000006
LSH-nt25-nb10-s:5k (query)                           98.65       363.65       462.30       0.8972     0.000484
LSH-nt25-nb10 (self)                                 98.65     11142.92     11241.57       0.9992     0.000006
LSH-nt10-nb12-s:auto (query)                         46.52       462.01       508.54       0.9611     0.000315
LSH-nt10-nb12-s:5k (query)                           46.52       295.45       341.97       0.9018     0.000467
LSH-nt10-nb12 (self)                                 46.52      4780.04      4826.56       0.9610     0.000310
LSH-nt20-nb12-s:auto (query)                         90.85       704.25       795.10       0.9922     0.000067
LSH-nt20-nb12-s:5k (query)                           90.85       330.95       421.81       0.9170     0.000288
LSH-nt20-nb12 (self)                                 90.85      6952.19      7043.04       0.9921     0.000067
LSH-nt25-nb12-s:auto (query)                        111.49       810.26       921.76       0.9961     0.000032
LSH-nt25-nb12-s:5k (query)                          111.49       325.75       437.24       0.9188     0.000267
LSH-nt25-nb12 (self)                                111.49      7744.07      7855.56       0.9961     0.000032
LSH-nt10-nb16-s:auto (query)                         61.91       396.96       458.87       0.9026     0.001020
LSH-nt10-nb16-s:5k (query)                           61.91       260.59       322.49       0.8499     0.001106
LSH-nt10-nb16 (self)                                 61.91      3659.09      3721.00       0.9040     0.000995
LSH-nt20-nb16-s:auto (query)                        109.68       518.71       628.40       0.9637     0.000366
LSH-nt20-nb16-s:5k (query)                          109.68       269.01       378.69       0.8909     0.000507
LSH-nt20-nb16 (self)                                109.68      5274.71      5384.40       0.9641     0.000362
LSH-nt25-nb16-s:auto (query)                        131.20       586.11       717.31       0.9750     0.000244
LSH-nt25-nb16-s:5k (query)                          131.20       277.46       408.66       0.9002     0.000391
LSH-nt25-nb16 (self)                                131.20      6100.16      6231.36       0.9748     0.000247
LSH-nt50-nb16-s:auto (query)                        261.54       804.25      1065.79       0.9948     0.000043
LSH-nt50-nb16-s:5k (query)                          261.54       283.88       545.41       0.9167     0.000208
LSH-nt50-nb16 (self)                                261.54      9553.86      9815.40       0.9947     0.000043
--------------------------------------------------------------------------------------------------------------
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
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.84      2757.87      2761.71       1.0000     0.000000
Exhaustive (self)                                     3.84     27356.63     27360.46       1.0000     0.000000
NNDescent-nt12-s:auto-dp0 (query)                  1896.27       121.81      2018.07       0.9689     0.102828
NNDescent-nt12-dp0 (self)                          1896.27       997.47      2893.74       0.9684     0.099935
NNDescent-nt24-s:auto-dp0 (query)                  2518.99        95.88      2614.87       0.9887     0.032393
NNDescent-nt24-dp0 (self)                          2518.99       961.57      3480.55       0.9886     0.031472
NNDescent-nt:auto-s75-dp0 (query)                  2546.01       133.53      2679.54       0.9942     0.016358
NNDescent-nt:auto-s100-dp0 (query)                 2546.01       169.24      2715.25       0.9962     0.010879
NNDescent-nt:auto-s:auto-dp0 (query)               2546.01        92.08      2638.09       0.9895     0.029672
NNDescent-nt:auto-dp0 (self)                       2546.01       881.16      3427.17       0.9894     0.029051
NNDescent-nt:auto-s:auto-dp0.25 (query)            2628.07        94.46      2722.53       0.9895     0.029672
NNDescent-nt:auto-dp0.25 (self)                    2628.07       902.32      3530.39       0.9894     0.029051
NNDescent-nt:auto-s:auto-dp0.5 (query)             2610.76        95.21      2705.96       0.9895     0.029672
NNDescent-nt:auto-dp0.5 (self)                     2610.76       927.91      3538.67       0.9894     0.029051
NNDescent-nt:auto-s:auto-dp1 (query)               2774.13        97.10      2871.22       0.9895     0.029672
NNDescent-nt:auto-dp1 (self)                       2774.13       899.96      3674.08       0.9894     0.029051
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for NNDescent
implementation in this `crate`.

```
==============================================================================================================
Benchmark: 150k cells, 32D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.27      2588.66      2592.93       1.0000     0.000000
Exhaustive (self)                                     4.27     26213.78     26218.06       1.0000     0.000000
NNDescent-nt12-s:auto-dp0 (query)                  6743.19        91.35      6834.54       0.9996     0.000002
NNDescent-nt12-dp0 (self)                          6743.19       872.05      7615.24       0.9997     0.000001
NNDescent-nt24-s:auto-dp0 (query)                  6787.18       107.82      6895.00       0.9997     0.000001
NNDescent-nt24-dp0 (self)                          6787.18      1073.01      7860.20       0.9998     0.000000
NNDescent-nt:auto-s75-dp0 (query)                  6600.55       130.74      6731.29       0.9998     0.000000
NNDescent-nt:auto-s100-dp0 (query)                 6600.55       179.87      6780.42       0.9999     0.000000
NNDescent-nt:auto-s:auto-dp0 (query)               6600.55       101.13      6701.68       0.9997     0.000001
NNDescent-nt:auto-dp0 (self)                       6600.55       877.23      7477.79       0.9998     0.000000
NNDescent-nt:auto-s:auto-dp0.25 (query)            6405.98        90.78      6496.76       0.9894     0.000018
NNDescent-nt:auto-dp0.25 (self)                    6405.98       818.17      7224.15       0.9897     0.000017
NNDescent-nt:auto-s:auto-dp0.5 (query)             7250.34        98.41      7348.75       0.9806     0.000036
NNDescent-nt:auto-dp0.5 (self)                     7250.34       816.23      8066.57       0.9805     0.000035
NNDescent-nt:auto-s:auto-dp1 (query)               7047.27        75.15      7122.42       0.9602     0.000080
NNDescent-nt:auto-dp1 (self)                       7047.27       714.81      7762.08       0.9609     0.000078
--------------------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.