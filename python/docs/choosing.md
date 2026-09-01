# Choosing an index

Sixteen indices is a lot of choice, and for any given problem most of them are
the wrong answer. This page is the short version.

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
  by construction. Its home ground is 20 to 100 dimensions; above that the
  triangle-inequality bounds go slack and it degenerates towards a full scan.
- **Memory is the binding constraint**: `IvfIndex`. Smallest footprint of the
  unquantised indices, and the two knobs (`nlist`, `nprobe`) are easy to reason
  about.
- **You have a GPU and n is large**: `CagraGpuIndex`. See [GPU](gpu.md).
- **You need Manhattan**: that rules out seven of them. See the table.

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
| `IvfIndex` | no | k-means Voronoi cells | yes | `nprobe` |
| `SoarIndex` | no | Voronoi cells with spilling | no | `nprobe` |
| `AnnoyIndex` | no | random-projection forest | no | `search_budget` |
| `KdTreeIndex` | no | randomised kd spill-tree forest | yes | `search_budget` |
| `BallTreeIndex` | no | nested hyperspheres | no | `search_budget` |
| `LshIndex` | no | multi-probe LSH | no | `n_probe` |
| `ExhaustiveGpuIndex` | yes | none, on device | no | none, recall is 1 |
| `IvfGpuIndex` | no | Voronoi cells, on device | no | `nprobe` |
| `CagraGpuIndex` | no | CAGRA graph, on device | no | `beam_width` |

## What each one is good and bad at

### Graph indices

**`HnswIndex`** is a layered proximity graph: expensive to build, fast to query,
and modest in memory for what it gives you. Raise `ef_search` for recall at
query time, `ef_construction` for a better graph at build time. The build cost
is the thing to watch: it's the slowest of the graph indices here.

**`VamanaIndex`** is the graph behind DiskANN, in its in-memory variant. A
single flat graph rather than a layered one, refined over two alpha-pruning
passes. Builds faster than HNSW at comparable recall, which is the reason to
reach for it.

**`NsgIndex`** thins a kNN graph with the MRNG rule, so each node ends up with
edges pointing in spread-out directions instead of a cluster of near-duplicates.
It has to materialise a full kNN graph first (`knn_k` sizes it), so the build
pays twice. Sparse and fast once built.

**`RnnDescentIndex`** folds the pruning into the descent loop, so one pass hands
back a graph that's already search-ready and no intermediate kNN graph is ever
materialised. The cheapest route to a graph index. `r` caps the out-degree and
is the size knob.

**`NNDescentIndex`** is the odd one out: `n_neighbors` is a build parameter, not
just a query one, and `extract_knn()` hands back the converged graph without
searching. If a self-kNN graph is the deliverable, this is the fastest route to
it on the CPU. As a queryable index it's fine but not special.

### Partition indices

**`IvfIndex`** cuts the space into `nlist` k-means cells and scans the `nprobe`
nearest ones. Cheap to build, easy to tune, smallest in-memory footprint of the
unquantised indices. Recall degrades when a true neighbour sits just over a cell
boundary, which is exactly what SOAR fixes.

**`SoarIndex`** writes every vector into two cells: its nearest centroid, and a
second chosen so its residual points somewhere the first one doesn't. Better
recall per `nprobe` than plain IVF, for roughly twice the posting-list size.
Read the trade against **query time**, not against `nprobe`: at a fixed `nprobe`
a spilled index scans about twice the candidates, so comparing at equal `nprobe`
flatters it and answers nothing.

**`KmknnIndex`** uses the same Voronoi structure but prunes with the triangle
inequality instead of skipping cells outright, which is what makes it exact.
Cosine normalises at build time and runs in Euclidean space internally.

### Tree indices

**`AnnoyIndex`** is a forest of random-projection trees. More trees means better
recall and a larger index, and that's essentially the whole tuning story.

**`KdTreeIndex`** is the same trade with axis-aligned splits. A split is one
coordinate comparison rather than a full dot product, so traversal and build are
both cheaper than Annoy's. It's also the only tree index here that supports
Manhattan.

**`BallTreeIndex`** prunes with the triangle inequality over nested
hyperspheres, so one tree does the job of a forest. It pays off when the data
has genuine cluster structure and the dimensionality is moderate. Watch the
default `search_budget`: 5% of the indexed points is thin on small datasets, and
10% is often the better starting point.

**`LshIndex`** is the cheapest index here to build and the weakest on recall.
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

Its main job is still ground truth: build one, measure your approximate index
against it, and only then decide whether the recall you're getting is the recall
you need. But check the timings while you're there. Beating a blocked GEMM is a
higher bar than beating brute force.

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

Don't extrapolate ideal parameters for your own problem from them. Do the
measurement: [Quickstart](quickstart.md) has the recipe.
