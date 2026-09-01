# ann-search

Approximate nearest-neighbour search built for single-cell and computational
biology workloads. The [Rust crate](https://github.com/GregorLueg/ann-search-rs)
does the work. This is a thin scikit-learn shaped layer over it.

Sixteen indices, thirteen on the CPU and three on the GPU, all behind the same
four-method surface. No CUDA runtime to install: the GPU backend is wgpu, so it
runs on Metal, Vulkan or DX12 and ships in the ordinary wheel.

## Install

```bash
uv pip install ann-search           # numpy only
uv pip install "ann-search[sparse]" # adds scipy, for kneighbors_graph
```

Wheels are built for Linux x86_64 and macOS on both architectures, against
Python 3.10 and up.

## Thirty seconds

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

## Where to go next

- [Choosing an index](choosing.md) if you don't already know which one you
  want. Sixteen is a lot of choice and most of them are wrong for your problem.
- [Quickstart](quickstart.md) for worked examples over the synthetic
  generators, including how to measure your own recall.
- [GPU](gpu.md) for the three device-resident indices and what they cost you.
- [Guide](guide.md) for metrics, padding, threads, persistence and the sharp
  edges.
- [API reference](api/indices.md) for every parameter of every index, with what
  each `None` default resolves to.
