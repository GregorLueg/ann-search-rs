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
Fast index building and good query speed. The size of the index however 
increases quite drastically the increased number of trees. 

**Key parameters:**

- *Number of trees (nt)*: The number of trees to generate in the forest
- *Search budget (s)*: The search budget per tree. If set to auto it uses
  `k * n_trees * 20`; versions with a `10x` or `5x` (i.e., less) are also shown.

**Euclidean:**

Below are the results for the Euclidean distance measure for Annoy. Self is
queried with the default search budget.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.08      2624.54      2627.62       1.0000     0.000000        18.31
Exhaustive (self)                                     3.08     26272.39     26275.47       1.0000     0.000000        18.31
Annoy-nt5-s:auto (query)                             75.44       102.63       178.07       0.6834    40.648110        33.67
Annoy-nt5-s:10x (query)                              75.44        64.08       139.52       0.5240    40.565845        33.67
Annoy-nt5-s:5x (query)                               75.44        38.69       114.13       0.3732    40.431273        33.67
Annoy-nt5 (self)                                     75.44      1115.57      1191.02       0.6838    40.084992        33.67
Annoy-nt10-s:auto (query)                           123.12       207.97       331.09       0.8810    40.727710        49.03
Annoy-nt10-s:10x (query)                            123.12       154.74       277.86       0.7412    40.683223        49.03
Annoy-nt10-s:5x (query)                             123.12       121.14       244.26       0.5626    40.594216        49.03
Annoy-nt10 (self)                                   123.12      2042.55      2165.68       0.8804    40.163717        49.03
Annoy-nt15-s:auto (query)                           182.46       310.76       493.22       0.9524    40.748981        49.65
Annoy-nt15-s:10x (query)                            182.46       214.62       397.08       0.8546    40.723903        49.65
Annoy-nt15-s:5x (query)                             182.46       120.67       303.13       0.6907    40.662767        49.65
Annoy-nt15 (self)                                   182.46      3049.54      3232.00       0.9516    40.184645        49.65
Annoy-nt25-s:auto (query)                           288.81       449.79       738.61       0.9908    40.758739        80.37
Annoy-nt25-s:10x (query)                            288.81       296.68       585.50       0.9508    40.750722        80.37
Annoy-nt25-s:5x (query)                             288.81       193.07       481.88       0.8410    40.720771        80.37
Annoy-nt25 (self)                                   288.81      3811.18      4099.99       0.9906    40.194411        80.37
Annoy-nt50-s:auto (query)                           478.23       637.89      1116.13       0.9997    40.760798       142.43
Annoy-nt50-s:10x (query)                            478.23       474.78       953.02       0.9957    40.760169       142.43
Annoy-nt50-s:5x (query)                             478.23       328.73       806.97       0.9644    40.754108       142.43
Annoy-nt50 (self)                                   478.23      6363.14      6841.38       0.9997    40.196443       142.43
Annoy-nt75-s:auto (query)                           725.26      1060.17      1785.43       1.0000    40.760868       177.49
Annoy-nt75-s:10x (query)                            725.26       940.54      1665.80       0.9995    40.760801       177.49
Annoy-nt75-s:5x (query)                             725.26       432.21      1157.47       0.9912    40.759442       177.49
Annoy-nt75 (self)                                   725.26      9252.40      9977.66       1.0000    40.196503       177.49
Annoy-nt100-s:auto (query)                         1002.04      1191.55      2193.59       1.0000    40.760873       266.55
Annoy-nt100-s:10x (query)                          1002.04       816.07      1818.11       0.9999    40.760864       266.55
Annoy-nt100-s:5x (query)                           1002.04       562.94      1564.97       0.9975    40.760539       266.55
Annoy-nt100 (self)                                 1002.04     12486.90     13488.93       1.0000    40.196506       266.55
---------------------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for Annoy.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.33      3404.05      3408.37       1.0000     0.000000        18.88
Exhaustive (self)                                     4.33     26897.61     26901.94       1.0000     0.000000        18.88
Annoy-nt5-s:auto (query)                             81.68        39.95       121.63       0.3536     0.004314        32.62
Annoy-nt5-s:10x (query)                              81.68        36.21       117.88       0.3536     0.004314        32.62
Annoy-nt5-s:5x (query)                               81.68        37.61       119.29       0.3432     0.004456        32.62
Annoy-nt5 (self)                                     81.68       364.88       446.56       0.3540     0.004254        32.62
Annoy-nt10-s:auto (query)                           102.44        65.20       167.64       0.5411     0.002165        46.35
Annoy-nt10-s:10x (query)                            102.44        63.81       166.25       0.5411     0.002165        46.35
Annoy-nt10-s:5x (query)                             102.44        73.42       175.86       0.5329     0.002226        46.35
Annoy-nt10 (self)                                   102.44       672.91       775.36       0.5394     0.002140        46.35
Annoy-nt15-s:auto (query)                           163.79        88.88       252.67       0.6685     0.001280        46.96
Annoy-nt15-s:10x (query)                            163.79        85.29       249.08       0.6685     0.001280        46.96
Annoy-nt15-s:5x (query)                             163.79        94.70       258.49       0.6626     0.001312        46.96
Annoy-nt15 (self)                                   163.79       918.13      1081.92       0.6668     0.001272        46.96
Annoy-nt25-s:auto (query)                           281.18       155.71       436.89       0.8222     0.000544        74.43
Annoy-nt25-s:10x (query)                            281.18       163.03       444.21       0.8222     0.000544        74.43
Annoy-nt25-s:5x (query)                             281.18       169.21       450.39       0.8195     0.000554        74.43
Annoy-nt25 (self)                                   281.18      1558.94      1840.12       0.8214     0.000537        74.43
Annoy-nt50-s:auto (query)                           435.65       271.90       707.55       0.9575     0.000095       129.99
Annoy-nt50-s:10x (query)                            435.65       275.18       710.83       0.9575     0.000095       129.99
Annoy-nt50-s:5x (query)                             435.65       269.70       705.35       0.9569     0.000096       129.99
Annoy-nt50 (self)                                   435.65      2929.17      3364.82       0.9573     0.000095       129.99
Annoy-nt75-s:auto (query)                           716.86       437.95      1154.81       0.9887     0.000021       238.04
Annoy-nt75-s:10x (query)                            716.86       425.33      1142.18       0.9887     0.000021       238.04
Annoy-nt75-s:5x (query)                             716.86       449.34      1166.20       0.9885     0.000022       238.04
Annoy-nt75 (self)                                   716.86      5300.68      6017.53       0.9885     0.000022       238.04
Annoy-nt100-s:auto (query)                          968.44       589.79      1558.23       0.9965     0.000006       241.09
Annoy-nt100-s:10x (query)                           968.44       567.00      1535.45       0.9965     0.000006       241.09
Annoy-nt100-s:5x (query)                            968.44       575.21      1543.65       0.9965     0.000006       241.09
Annoy-nt100 (self)                                  968.44      5700.78      6669.22       0.9964     0.000006       241.09
---------------------------------------------------------------------------------------------------------------------------
```

### HNSW

Hierarchical navigatable small worlds. A graph-based index that needs more time
to build the index. However, fast query speed and compared to some other
approximate indices limited memory finger print.

**Key parameters:**

- *M (m)*: The number of connections between layers
- *EF construction (ef)*: The budget to generate good connections during 
  construction of the index.
- *EF search (s)*: The budget for the search queries. 

Self is queried with `s=100`.

**Euclidean:**

Below are the results for the Euclidean distance measure for HSNW.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.62      2546.44      2550.06       1.0000     0.000000        18.31
Exhaustive (self)                                     3.62     24785.69     24789.31       1.0000     0.000000        18.31
HNSW-M16-ef50-s50 (query)                          2098.95       136.37      2235.32       0.6742    10.487238        52.45
HNSW-M16-ef50-s75 (query)                          2098.95       177.40      2276.35       0.7157     6.496600        52.45
HNSW-M16-ef50-s100 (query)                         2098.95       216.95      2315.91       0.7342     5.473296        52.45
HNSW-M16-ef50 (self)                               2098.95      2463.97      4562.92       0.7327     5.808804        52.45
HNSW-M16-ef100-s50 (query)                         4131.66       208.50      4340.16       0.9482     5.775413        52.45
HNSW-M16-ef100-s75 (query)                         4131.66       276.85      4408.51       0.9616     2.658184        52.45
HNSW-M16-ef100-s100 (query)                        4131.66       334.39      4466.05       0.9689     1.291409        52.45
HNSW-M16-ef100 (self)                              4131.66      3359.87      7491.54       0.9678     1.374629        52.45
HNSW-M16-ef200-s50 (query)                         4998.35       180.78      5179.13       0.9769     9.312603        52.45
HNSW-M16-ef200-s75 (query)                         4998.35       230.70      5229.05       0.9914     1.586795        52.45
HNSW-M16-ef200-s100 (query)                        4998.35       274.38      5272.73       0.9942     0.885721        52.45
HNSW-M16-ef200 (self)                              4998.35      2924.92      7923.27       0.9944     0.745413        52.45
HNSW-M24-ef100-s50 (query)                         6668.62       209.65      6878.26       0.9894     1.427168        68.45
HNSW-M24-ef100-s75 (query)                         6668.62       298.39      6967.00       0.9951     0.143253        68.45
HNSW-M24-ef100-s100 (query)                        6668.62       343.99      7012.61       0.9968     0.133934        68.45
HNSW-M24-ef100 (self)                              6668.62      3421.49     10090.11       0.9965     0.274497        68.45
HNSW-M24-ef200-s50 (query)                         8266.39       204.27      8470.65       0.9506     3.289924        68.45
HNSW-M24-ef200-s75 (query)                         8266.39       325.51      8591.90       0.9554     1.072559        68.45
HNSW-M24-ef200-s100 (query)                        8266.39       378.21      8644.59       0.9565     0.726441        68.45
HNSW-M24-ef200 (self)                              8266.39      3422.06     11688.44       0.9555     0.785733        68.45
HNSW-M24-ef300-s50 (query)                         9253.87       210.36      9464.23       0.9887     4.173036        68.45
HNSW-M24-ef300-s75 (query)                         9253.87       284.86      9538.73       0.9933     1.542559        68.45
HNSW-M24-ef300-s100 (query)                        9253.87       329.15      9583.02       0.9946     0.893997        68.45
HNSW-M24-ef300 (self)                              9253.87      3630.21     12884.08       0.9952     0.847291        68.45
HNSW-M32-ef200-s50 (query)                        13610.40       277.23     13887.63       0.9906     2.868064        84.45
HNSW-M32-ef200-s75 (query)                        13610.40       415.01     14025.41       0.9948     0.795003        84.45
HNSW-M32-ef200-s100 (query)                       13610.40       580.84     14191.24       0.9958     0.359084        84.45
HNSW-M32-ef200 (self)                             13610.40      3912.51     17522.90       0.9958     0.373062        84.45
HNSW-M32-ef300-s50 (query)                        14862.95       229.77     15092.73       0.9131    11.338606        84.45
HNSW-M32-ef300-s75 (query)                        14862.95       294.60     15157.55       0.9220     8.214121        84.45
HNSW-M32-ef300-s100 (query)                       14862.95       393.29     15256.25       0.9298     5.095915        84.45
HNSW-M32-ef300 (self)                             14862.95      4269.98     19132.93       0.9269     5.533713        84.45
---------------------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for HSNW.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.11      2613.66      2617.77       1.0000     0.000000        18.88
Exhaustive (self)                                     4.11     27205.04     27209.15       1.0000     0.000000        18.88
HNSW-M16-ef50-s50 (query)                          2214.80       135.81      2350.61       0.8698     0.013545        53.03
HNSW-M16-ef50-s75 (query)                          2214.80       180.48      2395.28       0.9005     0.012038        53.03
HNSW-M16-ef50-s100 (query)                         2214.80       224.01      2438.81       0.9142     0.010825        53.03
HNSW-M16-ef50 (self)                               2214.80      2294.45      4509.25       0.9131     0.011292        53.03
HNSW-M16-ef100-s50 (query)                         3627.60       187.64      3815.24       0.9686     0.001280        53.03
HNSW-M16-ef100-s75 (query)                         3627.60       233.27      3860.86       0.9780     0.000411        53.03
HNSW-M16-ef100-s100 (query)                        3627.60       251.49      3879.09       0.9824     0.000196        53.03
HNSW-M16-ef100 (self)                              3627.60      2749.97      6377.57       0.9825     0.000198        53.03
HNSW-M16-ef200-s50 (query)                         5074.95       133.67      5208.62       0.7831     0.011824        53.03
HNSW-M16-ef200-s75 (query)                         5074.95       221.73      5296.68       0.7895     0.009655        53.03
HNSW-M16-ef200-s100 (query)                        5074.95       227.92      5302.87       0.7936     0.008872        53.03
HNSW-M16-ef200 (self)                              5074.95      2588.36      7663.31       0.7939     0.008576        53.03
HNSW-M24-ef100-s50 (query)                         6559.33       190.64      6749.98       0.9585     0.005076        69.03
HNSW-M24-ef100-s75 (query)                         6559.33       265.62      6824.95       0.9688     0.003116        69.03
HNSW-M24-ef100-s100 (query)                        6559.33       312.65      6871.98       0.9739     0.002398        69.03
HNSW-M24-ef100 (self)                              6559.33      3088.24      9647.57       0.9742     0.002446        69.03
HNSW-M24-ef200-s50 (query)                         7627.27       186.59      7813.86       0.9312     0.011883        69.03
HNSW-M24-ef200-s75 (query)                         7627.27       266.22      7893.49       0.9431     0.003990        69.03
HNSW-M24-ef200-s100 (query)                        7627.27       300.86      7928.12       0.9462     0.002073        69.03
HNSW-M24-ef200 (self)                              7627.27      3022.56     10649.83       0.9507     0.002024        69.03
HNSW-M24-ef300-s50 (query)                         9309.83       236.04      9545.88       0.9784     0.016569        69.03
HNSW-M24-ef300-s75 (query)                         9309.83       292.62      9602.45       0.9900     0.006719        69.03
HNSW-M24-ef300-s100 (query)                        9309.83       399.92      9709.76       0.9937     0.003191        69.03
HNSW-M24-ef300 (self)                              9309.83      3635.50     12945.34       0.9935     0.003449        69.03
HNSW-M32-ef200-s50 (query)                        12695.22       254.73     12949.96       0.9933     0.002217        85.03
HNSW-M32-ef200-s75 (query)                        12695.22       372.77     13068.00       0.9965     0.000837        85.03
HNSW-M32-ef200-s100 (query)                       12695.22       387.61     13082.83       0.9976     0.000466        85.03
HNSW-M32-ef200 (self)                             12695.22      3997.27     16692.50       0.9973     0.000568        85.03
HNSW-M32-ef300-s50 (query)                        14083.14       232.08     14315.22       0.9858     0.009415        85.03
HNSW-M32-ef300-s75 (query)                        14083.14       325.33     14408.47       0.9933     0.003228        85.03
HNSW-M32-ef300-s100 (query)                       14083.14       457.09     14540.23       0.9959     0.001126        85.03
HNSW-M32-ef300 (self)                             14083.14      4027.59     18110.73       0.9952     0.001374        85.03
---------------------------------------------------------------------------------------------------------------------------
```

### IVF

Inverted file index. Uses Voronoi cells to sub-partition the original data.
Very small index size in memory.

**Key parameters:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search. 
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

Self query was done with `2 * sqrt(nlist)`.

**Euclidean:**

Below are the results for the Euclidean distance measure for IVF.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.49      2695.45      2698.94       1.0000     0.000000        18.31
Exhaustive (self)                                     3.49     25554.12     25557.61       1.0000     0.000000        18.31
IVF-nl273-np13 (query)                             1456.09       349.83      1805.92       0.9889     0.034523        19.49
IVF-nl273-np16 (query)                             1456.09       406.19      1862.28       0.9967     0.006608        19.49
IVF-nl273-np23 (query)                             1456.09       666.53      2122.62       1.0000     0.000000        19.49
IVF-nl273 (self)                                   1456.09      6145.20      7601.29       1.0000     0.000000        19.49
IVF-nl387-np19 (query)                             2154.99       363.10      2518.09       0.9926     0.031089        19.51
IVF-nl387-np27 (query)                             2154.99       518.47      2673.45       0.9998     0.000653        19.51
IVF-nl387 (self)                                   2154.99      4996.66      7151.65       0.9998     0.000690        19.51
IVF-nl547-np23 (query)                             2879.86       312.56      3192.42       0.9882     0.041944        19.53
IVF-nl547-np27 (query)                             2879.86       341.11      3220.97       0.9958     0.014725        19.53
IVF-nl547-np33 (query)                             2879.86       412.80      3292.66       0.9997     0.001035        19.53
IVF-nl547 (self)                                   2879.86      4315.43      7195.30       0.9997     0.001003        19.53
---------------------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for IVF.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.93      2664.28      2669.21       1.0000     0.000000        18.88
Exhaustive (self)                                     4.93     25972.65     25977.58       1.0000     0.000000        18.88
IVF-nl273-np13 (query)                             1396.36       321.79      1718.15       0.9894     0.000025        20.06
IVF-nl273-np16 (query)                             1396.36       390.65      1787.02       0.9969     0.000005        20.06
IVF-nl273-np23 (query)                             1396.36       648.32      2044.69       1.0000     0.000000        20.06
IVF-nl273 (self)                                   1396.36      5975.93      7372.30       1.0000     0.000000        20.06
IVF-nl387-np19 (query)                             2067.95       324.72      2392.66       0.9930     0.000019        20.08
IVF-nl387-np27 (query)                             2067.95       515.44      2583.38       0.9998     0.000000        20.08
IVF-nl387 (self)                                   2067.95      4942.53      7010.48       0.9998     0.000000        20.08
IVF-nl547-np23 (query)                             2871.69       326.45      3198.14       0.9892     0.000025        20.10
IVF-nl547-np27 (query)                             2871.69       364.80      3236.48       0.9963     0.000009        20.10
IVF-nl547-np33 (query)                             2871.69       486.57      3358.26       0.9997     0.000001        20.10
IVF-nl547 (self)                                   2871.69      4513.55      7385.23       0.9997     0.000001        20.10
---------------------------------------------------------------------------------------------------------------------------
```

### LSH

Locality sensitive hashing. Can be a very fast index, but with increasing
tables, the memory fingerprint does increase.

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
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.33      2349.04      2352.37       1.0000     0.000000        18.31
Exhaustive (self)                                     3.33     25452.89     25456.21       1.0000     0.000000        18.31
LSH-nt10-nb8-s:auto (query)                          60.68       666.08       726.77       0.9880     0.150786        34.90
LSH-nt10-nb8-s:5k (query)                            60.68       326.40       387.09       0.8853     0.682000        34.90
LSH-nt10-nb8 (self)                                  60.68      7443.45      7504.14       0.9879     0.148408        34.90
LSH-nt20-nb8-s:auto (query)                         114.65      1078.84      1193.49       0.9990     0.011816        51.65
LSH-nt20-nb8-s:5k (query)                           114.65       376.61       491.26       0.8866     0.657057        51.65
LSH-nt20-nb8 (self)                                 114.65     11288.66     11403.31       0.9990     0.010957        51.65
LSH-nt25-nb8-s:auto (query)                         150.05      1279.87      1429.92       0.9997     0.003853        59.86
LSH-nt25-nb8-s:5k (query)                           150.05       360.62       510.66       0.8866     0.657057        59.86
LSH-nt25-nb8 (self)                                 150.05     12633.92     12783.97       0.9996     0.003632        59.86
LSH-nt10-nb10-s:auto (query)                         72.55       632.94       705.49       0.9699     0.414554        35.55
LSH-nt10-nb10-s:5k (query)                           72.55       317.93       390.48       0.8848     0.679656        35.55
LSH-nt10-nb10 (self)                                 72.55      5836.52      5909.07       0.9700     0.407188        35.55
LSH-nt20-nb10-s:auto (query)                        155.97       953.77      1109.73       0.9951     0.060176        52.62
LSH-nt20-nb10-s:5k (query)                          155.97       325.92       481.89       0.8992     0.425955        52.62
LSH-nt20-nb10 (self)                                155.97      8321.73      8477.70       0.9949     0.061070        52.62
LSH-nt25-nb10-s:auto (query)                        175.42       801.17       976.59       0.9974     0.030311        61.37
LSH-nt25-nb10-s:5k (query)                          175.42       303.72       479.14       0.8997     0.417316        61.37
LSH-nt25-nb10 (self)                                175.42      8217.68      8393.11       0.9973     0.031546        61.37
LSH-nt10-nb12-s:auto (query)                         87.42       435.85       523.27       0.9411     0.804150        36.34
LSH-nt10-nb12-s:5k (query)                           87.42       281.21       368.63       0.8786     0.967038        36.34
LSH-nt10-nb12 (self)                                 87.42      4334.77      4422.20       0.9420     0.793338        36.34
LSH-nt20-nb12-s:auto (query)                        163.73       613.02       776.75       0.9844     0.197321        54.65
LSH-nt20-nb12-s:5k (query)                          163.73       306.48       470.21       0.9095     0.422210        54.65
LSH-nt20-nb12 (self)                                163.73      6110.27      6274.00       0.9842     0.200060        54.65
LSH-nt25-nb12-s:auto (query)                        205.19       664.82       870.01       0.9906     0.118465        63.59
LSH-nt25-nb12-s:5k (query)                          205.19       298.36       503.55       0.9133     0.361454        63.59
LSH-nt25-nb12 (self)                                205.19      7368.19      7573.38       0.9903     0.122014        63.59
LSH-nt10-nb16-s:auto (query)                        121.12       316.69       437.82       0.8498     2.429547        39.68
LSH-nt10-nb16-s:5k (query)                          121.12       222.00       343.13       0.8060     2.509321        39.68
LSH-nt10-nb16 (self)                                121.12      3407.48      3528.61       0.8527     2.402690        39.68
LSH-nt20-nb16-s:auto (query)                        214.22       464.50       678.72       0.9416     0.933309        60.83
LSH-nt20-nb16-s:5k (query)                          214.22       259.02       473.24       0.8775     1.076314        60.83
LSH-nt20-nb16 (self)                                214.22      5261.02      5475.24       0.9422     0.931024        60.83
LSH-nt25-nb16-s:auto (query)                        279.93       537.85       817.78       0.9589     0.613957        71.67
LSH-nt25-nb16-s:5k (query)                          279.93       270.01       549.94       0.8914     0.770405        71.67
LSH-nt25-nb16 (self)                                279.93      5689.52      5969.45       0.9588     0.626807        71.67
LSH-nt50-nb16-s:auto (query)                        554.41       812.04      1366.44       0.9856     0.188848       125.67
LSH-nt50-nb16-s:5k (query)                          554.41       324.11       878.51       0.9138     0.365119       125.67
LSH-nt50-nb16 (self)                                554.41      8033.17      8587.58       0.9856     0.191776       125.67
---------------------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for LSH.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.26      2473.85      2478.11       1.0000     0.000000        18.88
Exhaustive (self)                                     4.26     25924.11     25928.37       1.0000     0.000000        18.88
LSH-nt10-nb8-s:auto (query)                          34.17       990.65      1024.83       0.9942     0.000049        35.39
LSH-nt10-nb8-s:5k (query)                            34.17       394.91       429.09       0.8717     0.000877        35.39
LSH-nt10-nb8 (self)                                  34.17      9491.31      9525.49       0.9945     0.000045        35.39
LSH-nt20-nb8-s:auto (query)                          60.07      1593.80      1653.87       0.9997     0.000002        52.15
LSH-nt20-nb8-s:5k (query)                            60.07       383.87       443.94       0.8719     0.000875        52.15
LSH-nt20-nb8 (self)                                  60.07     15743.26     15803.33       0.9998     0.000002        52.15
LSH-nt25-nb8-s:auto (query)                          77.45      1982.75      2060.19       0.9999     0.000001        60.44
LSH-nt25-nb8-s:5k (query)                            77.45       370.98       448.43       0.8719     0.000875        60.44
LSH-nt25-nb8 (self)                                  77.45     18652.16     18729.61       0.9999     0.000000        60.44
LSH-nt10-nb10-s:auto (query)                         41.34       598.09       639.43       0.9806     0.000171        35.60
LSH-nt10-nb10-s:5k (query)                           41.34       335.12       376.46       0.8911     0.000557        35.60
LSH-nt10-nb10 (self)                                 41.34      6128.24      6169.59       0.9808     0.000167        35.60
LSH-nt20-nb10-s:auto (query)                         70.39       909.18       979.58       0.9983     0.000015        52.16
LSH-nt20-nb10-s:5k (query)                           70.39       331.51       401.91       0.8971     0.000485        52.16
LSH-nt20-nb10 (self)                                 70.39     10078.60     10149.00       0.9982     0.000015        52.16
LSH-nt25-nb10-s:auto (query)                         85.43      1006.23      1091.66       0.9993     0.000006        60.48
LSH-nt25-nb10-s:5k (query)                           85.43       333.05       418.48       0.8972     0.000484        60.48
LSH-nt25-nb10 (self)                                 85.43     11459.06     11544.49       0.9992     0.000006        60.48
LSH-nt10-nb12-s:auto (query)                         47.32       470.16       517.48       0.9611     0.000315        36.15
LSH-nt10-nb12-s:5k (query)                           47.32       308.87       356.19       0.9018     0.000467        36.15
LSH-nt10-nb12 (self)                                 47.32      4969.55      5016.87       0.9610     0.000310        36.15
LSH-nt20-nb12-s:auto (query)                         92.29       761.51       853.80       0.9922     0.000067        53.47
LSH-nt20-nb12-s:5k (query)                           92.29       347.75       440.04       0.9170     0.000288        53.47
LSH-nt20-nb12 (self)                                 92.29      7500.80      7593.09       0.9921     0.000067        53.47
LSH-nt25-nb12-s:auto (query)                        118.90       857.75       976.66       0.9961     0.000032        62.17
LSH-nt25-nb12-s:5k (query)                          118.90       393.69       512.60       0.9188     0.000267        62.17
LSH-nt25-nb12 (self)                                118.90      8048.74      8167.64       0.9961     0.000032        62.17
LSH-nt10-nb16-s:auto (query)                         57.87       382.39       440.27       0.9026     0.001020        37.78
LSH-nt10-nb16-s:5k (query)                           57.87       258.38       316.25       0.8499     0.001106        37.78
LSH-nt10-nb16 (self)                                 57.87      3706.70      3764.57       0.9040     0.000995        37.78
LSH-nt20-nb16-s:auto (query)                        106.52       533.22       639.74       0.9637     0.000366        57.16
LSH-nt20-nb16-s:5k (query)                          106.52       281.07       387.58       0.8909     0.000507        57.16
LSH-nt20-nb16 (self)                                106.52      5422.60      5529.12       0.9641     0.000362        57.16
LSH-nt25-nb16-s:auto (query)                        156.02       603.01       759.03       0.9750     0.000244        66.61
LSH-nt25-nb16-s:5k (query)                          156.02       263.93       419.95       0.9002     0.000391        66.61
LSH-nt25-nb16 (self)                                156.02      6171.50      6327.52       0.9748     0.000247        66.61
LSH-nt50-nb16-s:auto (query)                        258.76       845.32      1104.08       0.9948     0.000043       114.52
LSH-nt50-nb16-s:5k (query)                          258.76       299.56       558.31       0.9167     0.000208       114.52
LSH-nt50-nb16 (self)                                258.76      8552.16      8810.91       0.9947     0.000043       114.52
---------------------------------------------------------------------------------------------------------------------------
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
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.30      2320.04      2323.34       1.0000     0.000000        18.31
Exhaustive (self)                                     3.30     23337.67     23340.97       1.0000     0.000000        18.31
NNDescent-nt12-s:auto-dp0 (query)                  1724.72        96.49      1821.21       0.9689     0.102828       139.69
NNDescent-nt12-dp0 (self)                          1724.72       938.54      2663.27       0.9684     0.099935       139.69
NNDescent-nt24-s:auto-dp0 (query)                  2356.78        91.44      2448.23       0.9887     0.032393       170.66
NNDescent-nt24-dp0 (self)                          2356.78       878.61      3235.39       0.9886     0.031472       170.66
NNDescent-nt:auto-s75-dp0 (query)                  2493.73       127.91      2621.64       0.9942     0.016358       170.78
NNDescent-nt:auto-s100-dp0 (query)                 2493.73       173.33      2667.05       0.9962     0.010879       170.78
NNDescent-nt:auto-s:auto-dp0 (query)               2493.73        89.99      2583.72       0.9895     0.029672       170.78
NNDescent-nt:auto-dp0 (self)                       2493.73       887.35      3381.07       0.9894     0.029051       170.78
NNDescent-nt:auto-s:auto-dp0.25 (query)            2548.84        98.49      2647.33       0.9895     0.029672       175.36
NNDescent-nt:auto-dp0.25 (self)                    2548.84       871.33      3420.17       0.9894     0.029051       175.36
NNDescent-nt:auto-s:auto-dp0.5 (query)             2653.78        90.24      2744.02       0.9895     0.029672       175.36
NNDescent-nt:auto-dp0.5 (self)                     2653.78       901.17      3554.95       0.9894     0.029051       175.36
NNDescent-nt:auto-s:auto-dp1 (query)               2622.84        90.13      2712.97       0.9895     0.029672       175.36
NNDescent-nt:auto-dp1 (self)                       2622.84       873.78      3496.62       0.9894     0.029051       175.36
---------------------------------------------------------------------------------------------------------------------------
```

**Cosine:**

Below are the results for the Cosine distance measure for NNDescent
implementation in this `crate`.

```
===========================================================================================================================
Benchmark: 150k cells, 32D
===========================================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
---------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.51      2715.41      2719.92       1.0000     0.000000        18.88
Exhaustive (self)                                     4.51     24548.10     24552.61       1.0000     0.000000        18.88
NNDescent-nt12-s:auto-dp0 (query)                  5652.94        97.70      5750.64       0.9996     0.000002       137.58
NNDescent-nt12-dp0 (self)                          5652.94       919.35      6572.30       0.9997     0.000001       137.58
NNDescent-nt24-s:auto-dp0 (query)                  5624.57        92.74      5717.31       0.9997     0.000001       165.29
NNDescent-nt24-dp0 (self)                          5624.57       828.90      6453.48       0.9998     0.000000       165.29
NNDescent-nt:auto-s75-dp0 (query)                  5445.07       122.70      5567.77       0.9998     0.000000       165.41
NNDescent-nt:auto-s100-dp0 (query)                 5445.07       160.88      5605.95       0.9999     0.000000       165.41
NNDescent-nt:auto-s:auto-dp0 (query)               5445.07        84.35      5529.42       0.9997     0.000001       165.41
NNDescent-nt:auto-dp0 (self)                       5445.07       925.35      6370.42       0.9998     0.000000       165.41
NNDescent-nt:auto-s:auto-dp0.25 (query)            5718.87       102.71      5821.58       0.9894     0.000018       157.96
NNDescent-nt:auto-dp0.25 (self)                    5718.87       871.59      6590.46       0.9897     0.000017       157.96
NNDescent-nt:auto-s:auto-dp0.5 (query)             6087.73        76.89      6164.62       0.9806     0.000036       143.58
NNDescent-nt:auto-dp0.5 (self)                     6087.73       766.47      6854.19       0.9805     0.000035       143.58
NNDescent-nt:auto-s:auto-dp1 (query)               5964.58        80.63      6045.21       0.9602     0.000080       129.42
NNDescent-nt:auto-dp1 (self)                       5964.58       834.52      6799.11       0.9609     0.000078       129.42
---------------------------------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.