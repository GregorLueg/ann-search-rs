# Quickstart

Three worked examples over the synthetic generators. Everything here runs on
CPU in well under a minute on a laptop, and nothing needs a downloaded dataset.

## The data

`ann_search.datasets` ships four generators with structure real single-cell data
has. Uniform Gaussian noise is a bad ANN benchmark: past a few dozen dimensions
every point sits at roughly the same distance from every other, and recall stops
telling you anything.

```python
import numpy as np
import ann_search as ann
from ann_search import datasets

X, labels = datasets.make_clustered(n_samples=50_000, dim=32, n_clusters=25, seed=42)
X.shape, X.dtype
```

Every generator returns `(X, labels)`, so ground-truth cluster labels come free.
Output is always float32.

## Measuring recall

You need this before anything else, because none of the defaults are promises.
Recall is the fraction of true neighbours an index actually retrieved, so it's a
set intersection per row, not a positional comparison:

```python
def recall(found: np.ndarray, expected: np.ndarray) -> float:
    """Fraction of the true neighbours that were retrieved."""
    hits = sum(len(set(a) & set(b)) for a, b in zip(found, expected, strict=True))
    return hits / found.size
```

Comparing positionally (`(found == expected).mean()`) is the tempting one-liner
and it undercounts: two indices can retrieve exactly the same neighbours and
order tied distances differently.

## 1. Self-kNN and the recall knob

The self-kNN graph is the common case in single-cell work. Every index has a
fast path for it, taken by passing `None` (the default) to `kneighbors`, rather
than re-entering the index from outside with its own points.

Start with ground truth:

```python
truth = ann.ExhaustiveIndex(n_neighbors=15).fit(X).kneighbors(return_distance=False)
```

`ExhaustiveIndex` is exact, so that's the target. It is not a naive scan
either: past a batch-size threshold it switches from a fused per-query SIMD
scan to a blocked GEMM that reuses each database tile across a tile of queries.
The self-kNN graph is the largest batch there is, so it always takes that
path.

Now sweep the recall knob. `ef_search` is a search-time parameter, so one fitted
index answers the whole sweep:

```python
index = ann.HnswIndex(n_neighbors=15, m=16, ef_construction=200).fit(X)

for ef in (50, 100, 200):
    found = index.kneighbors(return_distance=False, ef_search=ef)
    print(f"ef_search={ef:>3}  recall={recall(found, truth):.4f}")
```

```text
ef_search= 50  recall=0.9883
ef_search=100  recall=0.9946
ef_search=200  recall=0.9967
```

Recall climbs with `ef_search`, and so does query time, roughly in step. Where
you stop is a judgement about your problem, not something the library can
decide, so time the sweep as well as scoring it. Build-time knobs (`m`,
`ef_construction`) need a refit to change, which is why the sweep above only
moves the one that doesn't.

If a self-kNN graph is the actual deliverable rather than a queryable index,
`NNDescentIndex` converges on one directly and hands it back without a search
pass:

```python
nnd = ann.NNDescentIndex(n_neighbors=15, max_candidates=30).fit(X)
found = nnd.extract_knn(return_distance=False)
print(f"extract_knn  recall={recall(found, truth):.4f}")
```

```text
extract_knn  recall=0.9975
```

Worth noticing: that beats HNSW at `ef_search=200` on the same data, off a build
of comparable cost. `kneighbors()` on the same index reaches 0.9987.

`extract_knn` reads the graph the descent already built. `kneighbors()` on the
same index runs a beam search over it instead, which costs more and recovers a
little recall. The gap between the two is exactly what the search buys you.

One thing that makes the comparison above honest: a kNN graph stores no
self-edge, but `ExhaustiveIndex` counts a point as its own nearest neighbour at
distance zero. `extract_knn` defaults to `include_self=True` so both sides use
a slot for it. Pass `include_self=False` and recall against this ground truth
caps at 14/15.

## 2. Cross-set queries and padding

Querying an index with rows it was built from flatters it: every query has an
exact hit at distance zero. `subsample_queries` draws a query set and perturbs
it, so the queries sit near the data rather than on it.

```python
Q = datasets.subsample_queries(X, n_samples=5_000, seed=0)

exact = ann.ExhaustiveIndex(n_neighbors=15).fit(X)
truth_q = exact.kneighbors(Q, return_distance=False)

index = ann.IvfIndex(n_neighbors=15).fit(X)

for nprobe in (4, 16, 64):
    found = index.kneighbors(Q, return_distance=False, nprobe=nprobe)
    print(f"nprobe={nprobe:>3}  recall={recall(found, truth_q):.4f}")
```

```text
nprobe=  4  recall=0.8219
nprobe= 16  recall=0.9995
nprobe= 64  recall=1.0000
```

`IvfIndex` leaves both `nlist` and `nprobe` at `None` here, so the crate picks
`sqrt(n)` cells and probes `sqrt(nlist)` of them. The sweep overrides only the
query-time half.

An approximate index can come back with fewer than `k` neighbours for a query.
Those slots are index `-1` and distance `inf`, so mask before you use them as
indices:

```python
distances, indices = index.kneighbors(Q)

found_per_query = (indices >= 0).sum(axis=1)
print(f"{(found_per_query < 15).sum()} queries came back short")

mask = indices >= 0
neighbour_labels = np.full(indices.shape, -1, dtype=labels.dtype)
neighbour_labels[mask] = labels[indices[mask]]
```

On this data nothing comes back short, which is the normal case. Do the check
anyway: `np.take` on a `-1` wraps around to the last row and hands you a
plausible wrong answer rather than an error, which is the failure mode worth
guarding against.

## 3. A sparse graph, and keeping it

`kneighbors_graph` gives you a scipy CSR, which is what scanpy, UMAP and most of
the graph tooling actually want. Padding slots are dropped, so rows can hold
fewer than `k` entries.

```python
index = ann.HnswIndex(n_neighbors=15, metric="cosine").fit(X)

graph = index.kneighbors_graph()  # distance-weighted
adjacency = index.kneighbors_graph(mode="connectivity")  # 1 per edge

graph.shape, graph.nnz
```

This needs scipy, which is the `sparse` extra:
`uv pip install "ann-search[sparse]"`.

The estimators implement `get_params`, `set_params`, `fit` and `transform`, so
the same object drops in wherever a `KNeighborsTransformer` is expected:

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

X, y = datasets.make_clustered(n_samples=5_000, dim=32, n_clusters=8, seed=1)

pipe = make_pipeline(
    ann.HnswIndex(n_neighbors=15, metric="euclidean"),
    KNeighborsClassifier(n_neighbors=10, metric="precomputed"),
)
print(cross_val_score(pipe, X, y, cv=3))
```

```text
[1.     0.9988 0.9196]
```

scikit-learn isn't an install requirement for any of that; it's duck-typing.

One thing to know when handing the graph to scikit-learn: it warns that the rows
aren't sorted by value. That's cosmetic, and
`sklearn.neighbors.sort_graph_by_row_values` silences it. Distances are already
guaranteed non-negative, so `DBSCAN(metric="precomputed")` and anything else
that validates a precomputed matrix will take a cosine graph directly.

Building an index is the expensive part, so keep it:

```python
index.save("hnsw_50k")  # a directory, not a file
index = ann.HnswIndex.load("hnsw_50k")

import pickle

blob = pickle.dumps(index)  # works with joblib and multiprocessing
```

The GPU indices are the exception: they hold device buffers and raise
`NotImplementedError` on `save`, `load` and pickle. Rebuild instead.

## What's next

[Choosing an index](choosing.md) if `HnswIndex` isn't the right shape for your
problem, [GPU](gpu.md) if you have a device, and the
[Guide](guide.md) for metrics, threads and the sharp edges.
