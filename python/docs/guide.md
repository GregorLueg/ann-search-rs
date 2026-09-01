# Guide

Everything that isn't about picking an index.

## Metrics

`"euclidean"` / `"l2"`, `"sqeuclidean"`, `"cosine"`, `"manhattan"` / `"l1"`.

The Rust core computes squared Euclidean distances. `"euclidean"` and `"l2"`
take the square root on the way out, so the numbers match scikit-learn and
scipy. `"sqeuclidean"` hands back the raw squared values and skips that, which
is free and preserves ordering.

An unknown metric raises `ValueError`. The core would quietly fall back to
squared Euclidean and warn to a stdout you can't see, which is a much worse
failure across FFI than a loud one. Nothing in this package sends the core a
string it hasn't already validated.

Seven indices don't support Manhattan. See the
[table](choosing.md#the-table); asking for it raises rather than falling back.

## Padding

Approximate indices can return fewer than `k` neighbours for a query. Those
slots come back as index `-1` and distance `inf`. Mask on `indices >= 0` before
you slice with them:

```python
mask = indices >= 0
labels_out = np.full(indices.shape, -1)
labels_out[mask] = labels[indices[mask]]
```

`kneighbors_graph` already drops them, so rows there can hold fewer than `k`
entries.

## A cosine wart

Cosine distance is computed as `1 - dot / (|x| |y|)`, and for a point against
itself that ratio can round to just above 1, so the distance lands one float32
ulp below zero (`-2.4e-07`). Harmless for ordering, but anything validating a
precomputed distance matrix as non-negative rejects it. `DBSCAN` and friends do.
Clip before handing a cosine graph over:

```python
graph = index.kneighbors_graph()
graph.data.clip(0, out=graph.data)
```

Euclidean is unaffected; it's already clamped at zero.

## Input types

`fit` takes anything `np.asarray` handles at shape `(n_samples, n_features)`.
float32 and float64 pass through untouched; any other numeric type is promoted
to float64 rather than narrowed, so precision is never silently lost. The result
is made C-contiguous because the core borrows the buffer rather than copying it.

Non-finite values raise. The core doesn't check, and would build a silently
useless index.

The GPU estimators are the exception: they force float32, since WGSL has no
float64.

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

Saving writes a directory rather than a file: the core's own bundle plus a small
JSON sidecar holding the estimator parameters, so `load` reconstructs the whole
object and not just a bare handle. `load` is a classmethod on the specific index
type and refuses a directory written by a different one.

The three GPU indices raise `NotImplementedError` on all of it.

## Versions

Two of them, and they are not the same number:

```python
ann.__version__  # this package
ann.__core_version__  # the ann-search-rs crate compiled into it
```

The bindings version on their own line. A docstring fix ships as a patch here
without dragging the crate through a release it did not need, and the crate can
move without forcing a wheel rebuild.

That leaves one honest question the wheel has to answer itself: which numerics
are actually inside it. The wheel vendors the crate source through a path
dependency rather than pulling a published version off crates.io, so
`__version__` cannot tell you. `__core_version__` can. Quote both in a bug
report.

## Errors

`AnnSearchError` is the base class for everything the Rust core raises.
`IndexIoError` subclasses it and means a bundle is missing, truncated, or of the
wrong kind or dtype.

The Python layer raises the ordinary Python ones: `ValueError` for a bad metric
or a shape mismatch, `TypeError` for a non-numeric array, and `NotFittedError`
(which subclasses both `ValueError` and `AttributeError`, matching
`sklearn.exceptions.NotFittedError`) for a query before `fit`.

## Sharp edges

- `verbose=True` writes to the process stdout, not `sys.stdout`. In Jupyter that
  lands in the terminal running the kernel, not in the cell.
- Ctrl-C can't interrupt an index build. Python signal handlers only run while
  the GIL is held, and the build releases it.
- `return_distance=False` saves the copy into numpy but not the distance
  computation, which happens either way.
- Indices are immutable. There's no incremental `add`, so rebuild instead. That
  is also why the scikit-learn shape is the honest one here: a FAISS-style
  `add()` would be a method callable exactly once.
- `NNDescentIndex` uses `n_neighbors` at build time as well as query time, so
  changing it means refitting. Every other index treats it as query-only.

## Development

```bash
uv venv
uv pip install "maturin>=1.15,<2" numpy scipy pytest scikit-learn beartype
maturin develop --release   # --release matters, the tests build real indices
pytest tests -q
```

Docs. mkdocstrings reads the installed package, so the extension has to be built
first:

```bash
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
maturin develop --release
mkdocs serve
```
