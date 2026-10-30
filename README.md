[![CI](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/ann-search-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/ann-search-rs)
[![docs.rs](https://img.shields.io/docsrs/ann-search-rs)](https://docs.rs/ann-search-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/ann-search.svg)](https://pypi.org/project/ann-search/)

# ann-search-rs

Approximate nearest neighbour search in Rust, built for computational biology
and single cell in particular. Everything is in memory and tuned for speed:
SIMD distance kernels with runtime dispatch (SSE, AVX2, AVX-512, NEON), rayon
across queries, and GPU paths via wgpu or, on Apple Silicon, MLX.

It started as the kNN code in [bixverse](https://github.com/GregorLueg/bixverse),
where kNN graphs are everywhere, and grew into far more indices than originally
planned. The algorithms are other people's great ideas; this is a Rust
implementation of them. There is a Python wrapper on
[PyPI](https://pypi.org/project/ann-search/) (`ann-search`), and changes live in
the [CHANGELOG](https://github.com/GregorLueg/ann-search-rs/blob/main/CHANGELOG.md).

## What's in it

| Family | Indices | Feature |
|---|---|---|
| Exact | Exhaustive (GEMM), KmKnn | default |
| Trees and hashing | Annoy, BallTree, Kd forest, LSH | default |
| Graphs | HNSW, NN-Descent, Vamana, NSG, RNN-Descent | default |
| Clustered | IVF, SOAR | default |
| Quantised | BF16, SQ8 (exhaustive, IVF, HNSW), PQ, OPQ, SOAR-PQ, SOAR-OPQ | `quantised` |
| Binary | Bitwise (SimHash, PCA hashing, sign), RaBitQ (exhaustive, IVF, HNSW), TurboQuant, Quantised Graph | `binary` |
| GPU (wgpu) | Exhaustive, IVF, CAGRA-style, NN-Descent and clustered NN-Descent kNN graphs | `gpu` |
| GPU (MLX) | Exhaustive, IVF, CAGRA-style, NN-Descent | `mlx`, experimental |

Distances: squared Euclidean, cosine, and Manhattan on a subset of the indices.

Every index has a `query_*_self` path for the full self-kNN graph, and most take
a shortcut there: IVF exploits its Voronoi cells, HNSW walks the graph without
re-entering, NN-Descent hands back the graph it built anyway.

If all you want is the kNN graph (UMAP, Leiden, anything single cell),
`build_knn_graph_nndescent` and `build_knn_graph_nndescent_gpu` skip the index
entirely: matrix in, flat `u32` graph out, nothing kept.

```rust
use ann_search_rs::build_knn_graph_nndescent;
use ann_search_rs::prelude::{NNDescentKnnParams, Verbosity};

let params = NNDescentKnnParams { k: 15, include_self: true, ..Default::default() };
let graph = build_knn_graph_nndescent(data.as_ref(), "euclidean", Some(params), 42, Verbosity::Quiet)?;
let row = graph.row(0); // k + 1 ids, the point itself first
```

## Installation

```toml
[dependencies]
ann-search-rs = { version = "0.12", features = ["quantised", "binary"] }
```

Optional features:

- `quantised`, `binary`, `gpu`, `mlx`: the index families above.
- `serialise`: save indices to disk and load them back.
- `ndarray`: accept `ndarray` 2-D arrays. Off by default, since it makes
  `ndarray` a public dependency.
- `synthetic`: the dataset generators behind the benchmark tables.
- `mimalloc`: mimalloc as the global allocator.
- `accelerate` (default): routes dense GEMM through Apple Accelerate on macOS.
  No effect elsewhere. Turn it off with `default-features = false`.

## Usage

```rust
use ann_search_rs::{build_hnsw_index, query_hnsw_index, query_hnsw_self, Verbosity};
use faer::Mat;

let data = Mat::from_fn(10_000, 32, |_, _| rand::random::<f32>());
let index = build_hnsw_index(
    data.as_ref(),
    16,          // m
    100,         // ef_construction
    "euclidean", // or "cosine", "manhattan"
    42,          // seed
    Verbosity::Normal,
)?;

// Cross-set query: k = 15, ef_search = 200, return distances
let query = Mat::from_fn(100, 32, |_, _| rand::random::<f32>());
let (indices, dists) = query_hnsw_index(query.as_ref(), &index, 15, 200, true, Verbosity::Quiet)?;

// Self-kNN graph, indices only
let (graph, _) = query_hnsw_self(&index, 15, 200, false, Verbosity::Quiet)?;
```

Every build and query takes a `Verbosity`. `Quiet` prints nothing but
warnings. `Normal` prints one line at the start and end of a build or query,
plus a progress line every 100_000 queried samples. `Detailed` adds the timed
build stages and per-iteration lines.

Callers that carry verbosity as an integer (R or Python bindings) can use
`parse_annsearch_verbosity(0 | 1 | 2)`.

Every index follows the same `build_*` / `query_*` / `query_*_self` pattern.
Inputs can be a faer matrix, an `ndarray` view (with the feature) or a flat
row-major slice with its shape. Rows are samples, columns are features, always:

```rust
let index = build_hnsw_index((flat.as_slice(), 10_000, 32), 16, 100, "euclidean", 42, Verbosity::Quiet)?;
```

The `examples/gridsearch_*.rs` scripts run every index across a parameter grid.
Quickest way to see how one behaves on your kind of data.

## Benchmarks

Against hnswlib, faiss, annoy, pynndescent and usearch on Fashion-MNIST, SIFT
and GloVe, compared at equal recall:
[comparison](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_comparison.md).

Internal gridsearches, per feature:

- [Standard CPU indices](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_standard.md)
- [Quantised](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_quantised.md)
- [Binary](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_binary.md)
- [GPU (wgpu)](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_gpu.md)
- [MLX](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_mlx.md)
- [kNN graph generation](https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_knn_graph.md)

These run on four synthetic datasets from the `synthetic` feature: Gaussian
clusters joined by bridges, correlated anisotropic clusters (separates OPQ from
PQ), low-rank manifolds with lineage trajectories, and a mock of
foundation-model cell embeddings. To rerun one:

```bash
cargo run --example gridsearch_hnsw --release
cargo run --example gridsearch_ivf --release -- --n-samples 500000 --dim 128 --distance cosine
cargo run --example gridsearch_pq --release --features quantised
```

The tables are generated without Accelerate, so they show what Linux and
Windows get. On an M1 Max, Accelerate makes exhaustive search about 2x faster
and IVF builds about 1.5x.

## GPU

The `gpu` feature runs on [CubeCL](https://github.com/tracel-ai/cubecl) with
the wgpu backend, so the same kernels run on Metal, Vulkan and DX12. Any
dimensionality works; rows are padded internally. Device limits are queried at
runtime, so a smaller GPU gets smaller staging tiles.

Beyond the indices, `build_knn_graph_gpu` returns a raw kNN graph for anything
that fits the device's per-binding limit, and `build_clustered_knn_graph_gpu`
splits larger datasets into overlapping k-means clusters first. Either can feed
`build_nsg_from_gpu_knn`, which is the way to sweep NSG parameters over one
graph. `build_nsg_index_gpu` does both in one call, matrix in, NSG out, and
moves the vectors into the NSG instead of copying them.

## MLX (experimental)

Apple Silicon only. The `mlx` feature drives Metal through
[mlx-c](https://github.com/ml-explore/mlx-c) instead of wgpu: MLX's GEMM plus
hand-written Metal kernels for top-k, the IVF cluster scan and the CAGRA beam
search. On an M1 Max at 500k x 128, exhaustive self-kNN goes 47.7 s to 15.4 s
against wgpu and the CAGRA build 6.2 s to 4.4 s. f32 only.

```bash
brew install mlx-c
```

```toml
ann-search-rs = { version = "0.12", features = ["mlx"] }
```

If mlx-c lives somewhere other than `/opt/homebrew`, set `MLX_C_PREFIX`. If it
is not found at all, the build warns and carries on without the MLX code;
`ann_search_rs::MLX_AVAILABLE` tells you which one you got. Crates building on
this one can read `DEP_MLXC_AVAILABLE` in their build script to pick the MLX or
the wgpu path.

## Saving and loading

With `serialise`, every CPU, quantised and binary index can be saved and loaded:

```rust
use ann_search_rs::cpu::hnsw::HnswIndex;
use ann_search_rs::{load_index, save_index};

save_index(&index, "my_index")?;
let index: HnswIndex<f32> = load_index("my_index")?;
```

A saved index is a directory: `index.bin` plus, for binary indices with a
re-ranking store, the on-disk vectors. Every file is written under a temporary
name and `index.bin` is renamed last, so a save that dies half way never leaves
something that loads clean and answers wrong. The header records the format
version, the index kind and the float width, so loading an HNSW as an IVF, or
`f32` as `f64`, is a typed error. The store files are raw native-endian dumps
and do not survive a move between little- and big-endian machines.

GPU and MLX indices hold device buffers and cannot be saved. Rebuild them.

## Licence

MIT, see [LICENSE](LICENSE).
