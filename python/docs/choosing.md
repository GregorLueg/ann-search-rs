# Choosing an index

Twenty-seven indices is a lot of choice, and for any given problem most of them
are the wrong answer. This page is the short version.

## Start here

If you want one answer and no reading: **`HnswIndex`**. High recall at low query
latency, works on every metric, and the defaults are sensible. Everything below
is about when it isn't the right call.

Beyond that:

- **On a small dataset**, use `ExhaustiveIndex`. It's exact, and on a large
  batch it runs a blocked GEMM rather than a naive scan, so the point where
  building an approximate index starts paying for itself is further out than
  you'd guess. Measure it on your data rather than assuming.
- **You want the full self-kNN graph**, as for UMAP or a scanpy neighbourhood
  graph: `NNDescentIndex`, then `extract_knn()`. It converges on the graph
  directly instead of searching for something it already has.
- **You need exactness but brute force is too slow**: `KmknnIndex`. Recall is 1
  by construction. The triangle-inequality bounds go slack as the
  dimensionality climbs: on the benchmark runs it queries 8.5x faster than
  brute force at 32 dimensions and 2.4x at 128.
- **Memory is the binding constraint**: `HnswSq8uIndex`, which is the usual
  first choice at a quarter of the vector memory. See
  [Quantised](quantised.md). Staying uncompressed, `IvfIndex` is the smallest
  of the approximate indices and the two knobs (`nlist`, `nprobe`) are easy to
  reason about.
- **You have a GPU and n is large**: `CagraGpuIndex`. See [GPU](gpu.md).
- **You need Manhattan**: that rules out twenty of them. See the table.

## The table

| Index | Exact | Structure | Manhattan | Main recall knob |
| --- | --- | --- | --- | --- |
| `ExhaustiveIndex` | yes | none | yes | none, recall is 1 |
| `KmknnIndex` | yes | k-means cells, triangle-inequality pruning | no | none, recall is 1 |
| `HnswIndex` | no | layered proximity graph | yes | `ef_search` |
| `VamanaIndex` | no | flat navigable graph | yes | `ef_search` |
| `NsgIndex` | no | pruned monotonic graph | yes | `ef_search` |
| `RnnDescentIndex` | no | graph, pruned during descent | yes | `ef_search` |
| `NNDescentIndex` | no | kNN graph | yes | `ef_search` |
| `IvfIndex` | no | k-means Voronoi cells | no | `nprobe` |
| `SoarIndex` | no | Voronoi cells with spilling | no | `nprobe` |
| `AnnoyIndex` | no | random-projection forest | no | `search_budget` |
| `KdTreeIndex` | no | randomised kd spill-tree forest | yes | `search_budget` |
| `BallTreeIndex` | no | nested hyperspheres | no | `search_budget` |
| `LshIndex` | no | multi-probe LSH | no | `n_probe` |
| `ExhaustiveGpuIndex` | yes | none, on device | no | none, recall is 1 |
| `IvfGpuIndex` | no | Voronoi cells, on device | no | `nprobe` |
| `CagraGpuIndex` | no | CAGRA graph, on device | no | `beam_width` |
| `ExhaustiveBf16Index` | no | none, `bf16` storage | no | none |
| `IvfBf16Index` | no | Voronoi cells, `bf16` storage | no | `nprobe` |
| `ExhaustiveSq8Index` | no | none, 8-bit codes | no | none |
| `IvfSq8Index` | no | Voronoi cells, 8-bit codes | no | `nprobe` |
| `HnswSq8uIndex` | no | layered graph over 8-bit codes | no | `ef_search` |
| `ExhaustivePqIndex` | no | none, product codes | no | none |
| `IvfPqIndex` | no | Voronoi cells, product codes | no | `nprobe` |
| `ExhaustiveOpqIndex` | no | none, rotated product codes | no | none |
| `IvfOpqIndex` | no | Voronoi cells, rotated product codes | no | `nprobe` |
| `SoarPqIndex` | no | spilled cells, product codes | no | `nprobe` |
| `SoarOpqIndex` | no | spilled cells, rotated product codes | no | `nprobe` |

## What each one is good and bad at

### Graph indices

**`HnswIndex`** is a layered proximity graph: fast to build, fast to query, and
modest in memory for what it gives you. Raise `ef_search` for recall at query
time, `ef_construction` for a better graph at build time. The thing to watch is
`ef_construction`, which is what that build cost is actually sensitive to.

At its cheapest grid setting it is the cheapest graph index here to build, on
every distribution. Don't read that as a general result: match on recall and it
goes away. Vamana reaches 0.99 on the gaussian data for slightly less build
time in a 35% smaller index, and at 128 dimensions HNSW is the *most* expensive
of the four graph indices at recall 0.999.

**`VamanaIndex`** is the graph behind DiskANN, in its in-memory variant. A
single flat graph rather than a layered one, refined over two alpha-pruning
passes. At its cheapest setting it builds slower than HNSW, though the gap
closes and can reverse once you match on recall. The size win over HNSW is
small, 2 to 5%, and `NsgIndex` is smaller than both, so reach for it for the
flat structure rather than for either number.

**`NsgIndex`** thins a kNN graph with the MRNG rule, so each node ends up with
edges pointing in spread-out directions instead of a cluster of near-duplicates.
It has to materialise a full kNN graph first (`knn_k` sizes it), so the build
pays twice, and it is the most expensive graph index here to build.

It is the sparsest of the graph indices, but only of those: `IvfIndex`,
`SoarIndex` and `KmknnIndex` are all smaller again. "Fast once built" holds
inside the graph family and not outside it. On the gaussian runs NSG queries in
93.76 ms at recall 0.9957, which `KmknnIndex` (78.45 ms, recall 1.0) and
`IvfIndex` (87.57 ms, 0.9996) both beat on either axis.

**`RnnDescentIndex`** folds the pruning into the descent loop, so one pass hands
back a graph that's already search-ready and no intermediate kNN graph is ever
materialised. That saves it the double build NSG pays, though HNSW still gets
there first. `r` caps the out-degree and is the size knob.

Watch the recall ceiling. On the gaussian runs no setting in the grid gets past
0.9702, so if you need 0.99 this is not the index.

**`NNDescentIndex`** is the odd one out: `n_neighbors` is a build parameter, not
just a query one, and `extract_knn()` hands back the converged graph without
searching. That is its whole argument, and it is a real one when the graph *is*
the deliverable rather than a queryable index.

Whether it is the quickest route there depends on the shape, and the benchmark
runs split. At 32 dimensions `KmknnIndex`'s self path gets there in 1.07 s at
recall 1.0 against `extract_knn`'s 2.51 s at 0.9997. At 128 dimensions it
reverses hard: `extract_knn` takes 2.52 s against KmknnIndex's 4.60 s and
`IvfIndex`'s 7.63 s, all three at recall 1.0. Measure both on your data.

### Partition indices

**`IvfIndex`** cuts the space into `nlist` k-means cells and scans the `nprobe`
nearest ones. Cheap to build, easy to tune, and the smallest of the approximate
unquantised indices: it stores the vectors and a permutation, and little else.
Recall degrades when a true neighbour sits just over a cell boundary, which is
exactly what SOAR fixes.

**`SoarIndex`** writes every vector into two cells: its nearest centroid, and a
second chosen so its residual points somewhere the first one doesn't. Better
recall per `nprobe` than plain IVF, for twice the posting-list *entries*. That
is a few per cent of index size rather than double, since the vectors
themselves are stored once either way: 19.49 MB against IVF's 18.35 on the
gaussian runs.

Read the trade against **query time**, not against `nprobe`: at a fixed
`nprobe` a spilled index scans about twice the candidates, so comparing at
equal `nprobe` flatters it and answers nothing. At matched `nprobe` SOAR costs
2.6 to 3.4x the query time.

**`KmknnIndex`** uses the same Voronoi structure but prunes with the triangle
inequality instead of skipping cells outright, which is what makes it exact.
Cosine normalises at build time and runs in Euclidean space internally.

### Tree indices

**`AnnoyIndex`** is a forest of random-projection trees. More trees means better
recall and a larger index, and that's essentially the whole tuning story.

**`KdTreeIndex`** is the same trade with axis-aligned splits. A split is one
coordinate comparison rather than a full dot product, so per tree the build,
the query and the index are all cheaper than Annoy's. It's also the only tree
index here that supports Manhattan.

Per tree is not per recall. At 128 dimensions the random-projection splits
carry more information each, and Annoy reaches 0.9864 in 492 ms total where
KdTree needs 876 ms to get to 0.9920. At 32 dimensions the two are level.

**`BallTreeIndex`** prunes with the triangle inequality over nested
hyperspheres, so one tree does the job of a forest. It pays off when the data
has genuine cluster structure and the dimensionality is moderate. The default
`search_budget` of 5% is fine at 32 dimensions, where it reaches 0.9865 to
0.9996 across the four distributions, and thin at 128, where it drops to 0.9135
and 10% is the better starting point (0.9924).

Past 10% recall plateaus exactly, to four decimal places, and query time
doesn't fall, so there's no reason to go higher. Going from 5% to 10% at 32
dimensions is nearly free, so treat 5% as the floor rather than the default to
stay on.

**`LshIndex`** is the cheapest index here to build, by a wide margin: 15 to 30
ms on the gaussian runs against the next cheapest at 61 ms. It is the weakest
on recall *for the query time it costs*, which is not the same as having the
lowest floor. Tuned up it does reach 1.0, and it is the one index here whose
query can come out slower than brute force, so time it rather than assuming.

The projections are orthogonalised and the quantile boundaries are fitted to a
subsample rather than passing through the origin, which matters on data carrying
a large shared mean offset. Foundation-model cell embeddings are exactly that
case, and plain SimHash collapses on them.

### Exact search

**`ExhaustiveIndex`** scores every query against every point, so recall is 1 by
construction. It is not a naive scan: the core dispatches on batch size between
a fused per-query SIMD scan, which keeps each accumulation in registers but
re-reads the whole database per query, and a blocked GEMM path that blocks both
axes so a database tile is reused across a tile of queries. The self-kNN graph
is the largest batch there is, which is exactly where the GEMM path wins.

The dispatch is on batch size, and the threshold scales with your thread count
rather than being a fixed number. Manhattan never takes the GEMM path at all.

Its main job is still ground truth: build one, measure your approximate index
against it, and only then decide whether the recall you're getting is the recall
you need. But check the timings while you're there. Beating a blocked GEMM is a
higher bar than beating brute force.

### Quantised

Eleven more estimators, covered on their own page: [Quantised](quantised.md).
The short version is that they store compressed vectors instead of floats, so
they trade recall for memory, and that the distances they hand back are the
codec's estimate rather than the distance.

Reach for one when memory is what's binding, and measure the recall before you
commit. `HnswSq8uIndex` is the usual starting point, at a quarter of the
*vector* memory with the same `ef_search` knob, but the codec is not cheap: on
the benchmark runs it gives up 0.07 recall against plain HNSW on Euclidean and
0.26 to 0.33 on cosine. `IvfSq8Index` if you were already on `IvfIndex`. The PQ
family when a quarter isn't enough of a saving, which in practice means a
high-dimensional embedding space.

## Measuring rather than guessing

None of the above tells you what recall you'll get on your data. The crate's
benchmark tables sweep every index over four synthetic distributions with hard
numbers for build time, query time, recall and index size:

- [Standard indices](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_standard.md)
- [kNN graph construction](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_knn_graph.md)
- [GPU indices](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_gpu.md)

Those runs use the same generators and seeds as
[`ann_search.datasets`](api/datasets.md), so a Python measurement and a
`cargo run --example gridsearch_hnsw` run see identical points.

Mind what they cover. The standard tables are 150k samples at 32 and 128
dimensions, on one M1 Max, one run each. 128 dimensions only ever appears as
the cell-embedding generator, so dimensionality and distribution move together
and nothing above separates the two. Every "in high dimensions" statement on
this page inherits that.

Don't extrapolate ideal parameters for your own problem from them. Do the
measurement: [Quickstart](quickstart.md) has the recipe.
