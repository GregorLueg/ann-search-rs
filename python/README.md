[![PyPI](https://img.shields.io/pypi/v/ann-search.svg)](https://pypi.org/project/ann-search/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://gregorlueg.github.io/ann-search-rs/api/indices/)
[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/python-test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/python-test.yml)

# ann-search

Python bindings for [`ann-search-rs`](https://github.com/GregorLueg/ann-search-rs):
approximate nearest-neighbour search built for single-cell and computational
biology workloads. The Rust crate does the work. This is a thin scikit-learn
shaped layer over it.

**Documentation: <https://gregorlueg.github.io/ann-search-rs/>**

## Install

```bash
uv pip install ann-search           # numpy only
uv pip install "ann-search[sparse]" # adds scipy, for kneighbors_graph
```

## Use

Every index is a scikit-learn style estimator. Parameters go in the constructor,
data goes into `fit`, results come out of `kneighbors`.

```python
import numpy as np
import ann_search as ann

X = np.random.default_rng(0).standard_normal((50_000, 50)).astype(np.float32)

index = ann.HnswIndex(n_neighbors=15, metric="cosine").fit(X)

distances, indices = index.kneighbors()  # self-kNN graph, fast path
distances, indices = index.kneighbors(X[:1000])  # cross-set query
graph = index.kneighbors_graph()  # scipy CSR
```

`kneighbors` returns distances first, matching scikit-learn and FAISS.

The estimators implement `get_params`, `set_params`, `fit` and `transform`, so
they drop into scikit-learn pipelines and anywhere a `KNeighborsTransformer` is
expected, scanpy included. scikit-learn isn't an install requirement for any of
that.

## Indices

| Class | Notes |
| --- | --- |
| `ExhaustiveIndex` | Exact. Blocked GEMM on large batches, not a naive scan. Ground truth. |
| `KmknnIndex` | Exact, k-means pruned. No Manhattan. |
| `AnnoyIndex` | Random projection forest. No Manhattan. |
| `KdTreeIndex` | Randomised kd spill-tree forest. Axis-aligned splits. |
| `BallTreeIndex` | Metric tree of nested hyperspheres. No Manhattan. |
| `HnswIndex` | Hierarchical small-world graph. The usual first choice. |
| `IvfIndex` | Inverted file over k-means cells. Smallest of the approximate indices. No Manhattan. |
| `SoarIndex` | IVF with spilling. Better recall per `nprobe`, twice the posting-list entries. No Manhattan. |
| `LshIndex` | Multi-probe LSH. Cheapest build, weakest recall. No Manhattan. |
| `NNDescentIndex` | Hands back the graph it built, without a search pass. |
| `RnnDescentIndex` | Builds and prunes in one pass, no intermediate kNN graph. |
| `VamanaIndex` | DiskANN-style flat graph. |
| `NsgIndex` | Navigating spreading-out graph. |

`BallTreeIndex` defaults its `search_budget` to 5% of the indexed points, which
is the crate's own heuristic. It reaches 0.9865 to 0.9996 recall at 32
dimensions and drops to 0.9135 at 128, where 10% is the better starting point
and takes it back to 0.9924. Past 10% recall plateaus and query time doesn't
fall, so there's no reason to go higher.

## Quantised

Eleven more estimators over compressed vectors, for when memory is the binding
constraint. Distances from these are the codec's estimate rather than the
distance, and none of them support Manhattan.

Measure recall against `ExhaustiveIndex` before committing to one. The codec
cost on the benchmark runs is larger than the memory saving makes it look, and
it is worst on cosine. See [Quantised](docs/quantised.md).

| Class | Notes |
| --- | --- |
| `ExhaustiveBf16Index` | Brute force at `bf16`. Half the memory, but the query runs 2 to 4x slower and recall drops to ~0.98 Euclidean, ~0.89 cosine. |
| `IvfBf16Index` | `IvfIndex` with `bf16` posting lists. |
| `ExhaustiveSq8Index` | Brute force on 8-bit codes. 3.5x smaller at 32D, 3.9x at 128D. Slower than the float scan at 32D. |
| `IvfSq8Index` | `IvfIndex` on 8-bit codes. |
| `HnswSq8uIndex` | HNSW built *and* searched on 8-bit codes. The usual starting point. Faster to build and query; costs 0.07 recall on Euclidean, 0.26+ on cosine. |
| `ExhaustivePqIndex` | Product quantisation, `m` bytes per vector. |
| `IvfPqIndex` | IVF plus PQ, codes learned on the cell residual. |
| `ExhaustiveOpqIndex` | PQ with a learned rotation in front. |
| `IvfOpqIndex` | IVF-PQ with the rotation. |
| `SoarPqIndex` | IVF-PQ with SOAR spilling. |
| `SoarOpqIndex` | Spilling plus the rotation, so it pays for both. The *largest* of the eleven, not the smallest. |

The binary indices in the Rust crate aren't bound yet. They follow the same
pattern when they land.

## GPU

Three more estimators. They ship in the ordinary wheel: there's no separate
package and no extra to ask for.

| Class | Notes |
| --- | --- |
| `ExhaustiveGpuIndex` | Brute force on the device. Exact, and cheap ground truth. |
| `IvfGpuIndex` | k-means and vectors both resident. Bounded by device memory. |
| `CagraGpuIndex` | NN-Descent on device, pruned to a CAGRA graph, beam-searched. |

```python
import ann_search as ann

if ann.gpu_available():
    index = ann.CagraGpuIndex(n_neighbors=15).fit(X)
```

`gpu_available()` answers the only question worth asking: is there an adapter on
this machine. The backend is wgpu, so that means Metal on macOS and Vulkan or
DX12 elsewhere. There's no CUDA runtime to install and nothing extra to
`pip install`. On a box with no GPU it returns False and the CPU estimators are
unaffected.

On the M1 Max the benchmarks ran on, the GPU exhaustive path comes out around
1.7x faster than the CPU one, and CAGRA's build cost means it only wins overall
past a few hundred thousand points. See [GPU](docs/gpu.md).

Three differences from the CPU estimators, all forced by the backend:

- **float32 only.** WGSL has no float64, so `fit` narrows rather than failing
  inside a kernel. Not the only narrowing in the package: a float64 query
  against a float32 index is narrowed too, on every index.
- **No persistence.** These hold device buffers and sit outside the crate's
  `serialise` feature, so `save`, `load` and pickle raise `NotImplementedError`.
  Rebuild instead.
- **No Manhattan**, on any of the three.

A fitted `CagraGpuIndex` is also the one index here that is not safe to query
from two threads at once: the beam search memoises its graph upload behind a
mutable borrow, and the GIL is released while the kernel runs, so concurrent
calls raise rather than queueing. Give each thread its own handle.

### A CPU-only build

GPU support is compiled in by default, which costs roughly 3 MB of wheel. That
seemed the better trade than a second distribution, since wgpu has no driver
runtime to ship and the alternative is a version pin between two packages that
can drift.

If the megabytes matter, build without it:

```bash
maturin develop --release --no-default-features
```

`gpu_available()` then returns False on any machine, and `import ann_search.gpu`
raises with an explanation. Everything on the CPU side is untouched.

## Synthetic data

Uniform Gaussian noise is a bad ANN benchmark. Past a few dozen dimensions every
point sits at roughly the same distance from every other, so recall stops
telling you anything. Four generators with structure real single-cell data has:

```python
from ann_search import datasets

X, labels = datasets.make_clustered(50_000, dim=32, n_clusters=25, seed=42)
Q = datasets.subsample_queries(X, 5_000, seed=42)
```

| Generator | What it stresses |
| --- | --- |
| `make_clustered` | Separated blobs with inter-cluster bridges. The baseline. |
| `make_correlated` | Local anisotropy plus a shared off-axis subspace. Where OPQ and PQ pull apart. |
| `make_low_rank` | A low-dimensional manifold in a high-dimensional space, with trajectories. |
| `make_cell_embeddings` | Geneformer/scGPT flavoured: heavy tails, rogue dimensions, anisotropy cone. Gets painful for quantised indices. |

Each returns `(X, labels)`, so ground-truth cluster labels come free. Output is
float32.

These are the same generators, same seeds, behind the benchmark tables in the
Rust crate's `docs/`. A Python benchmark and a `cargo run --example
gridsearch_hnsw` run see identical points, and the Python test suite pins
checksums taken from the Rust side to keep it that way.

`subsample_queries` matters more than it looks. Querying an index with rows it
was built from flatters it: every query has an exact hit at distance zero.

## Metrics

`"euclidean"` / `"l2"`, `"sqeuclidean"`, `"cosine"`, `"manhattan"` / `"l1"`.

The Rust core computes squared Euclidean distances. `"euclidean"` and `"l2"`
take the square root on the way out so the numbers match scikit-learn and scipy;
`"sqeuclidean"` hands back the raw squared values and skips that.

An unknown metric raises `ValueError`. The Rust core would quietly fall back to
squared Euclidean and warn to a stdout you can't see, which is a much worse
failure across FFI than a loud one.

## Padding

Approximate indices can return fewer than `k` neighbours for a query. Those
slots come back as index `-1` and distance `inf`. Mask on `indices >= 0` before
you slice with them. `kneighbors_graph` already drops them.

## Threads

```python
ann.set_num_threads(8)  # 0 restores the default pool
ann.num_threads()
```

The default pool honours `RAYON_NUM_THREADS`. Rayon worker threads don't survive
`fork`, so use the `spawn` start method for `multiprocessing`.

## Persistence

```python
index.save("my_index")  # a directory, not a file
index = ann.HnswIndex.load("my_index")

import pickle

blob = pickle.dumps(index)  # works with joblib and multiprocessing
```

## Caveats

- `verbose=True` writes to the process stdout, not `sys.stdout`. In Jupyter that
  lands in the terminal running the kernel, not in the cell.
- Ctrl-C can't interrupt an index build. Python signal handlers only run while
  the GIL is held, and the build releases it.
- `return_distance=False` saves the copy into numpy but not the distance
  computation, which happens either way.
- Indices are immutable. There's no incremental `add`, so rebuild instead.

## Development

```bash
uv venv
uv pip install "maturin>=1.15,<2" numpy scipy pytest scikit-learn beartype
maturin develop --release   # --release matters, the tests build real indices
pytest tests -q
```
