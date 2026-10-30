# News

Changes to the `ann-search-rs` crate. The Python package `ann-search` has its
own changelog at [`python/CHANGELOG.md`](python/CHANGELOG.md).

## 0.12.0

**Features**

- `build_knn_graph_nndescent()`: matrix in, kNN graph out, on the CPU. Nothing
  else is kept. The seeding trees are built in batches of one per thread, joined
  and dropped, so the full forest never sits in memory. With the same seed and
  tree count it returns the same graph as `extract_knn` on an NN-Descent index.
  Takes `NNDescentKnnParams` (`k`, iterations, candidates, trees, `delta`,
  `include_self`, `return_dist`).
- `build_knn_graph_nndescent_gpu()`: the same on the GPU, through the
  clustered build, so datasets past the per-binding limit work too. Takes
  `NNDescentGpuKnnParams`.
- `build_nsg_index_gpu()`: matrix in, NSG out, with the kNN graph built on
  the GPU through the clustered build. The vectors move from the kNN handoff
  into the NSG instead of being copied. Takes `NNDescentGpuKnnParams` for the
  kNN graph. `build_nsg_from_gpu_knn` stays for reusing one graph across
  several NSG builds.
- The two kNN-graph builders return a flat `KnnGraph<T>`: `n * k` `u32` ids, optional distances,
  rows sorted ascending, `KNN_GRAPH_EMPTY` for unfilled slots. `include_self`
  puts each point first in its own row at distance zero, as UMAP expects.
- Manhattan NN-Descent seeds by Kd-tree leaf join as well, instead of one
  forest query per point. Recall@15 at 200k x 32 goes from 0.960 to 0.970.

- One verbosity grammar across every index, k-means and kNN-graph builder:
  `Verbosity { Quiet, Normal, Detailed }`. `Normal` prints `Building <name>
  ...` / `... built <name> in 1.23s`, `Querying <name> ...`, a progress line
  every 100_000 samples and `... queried <name> in`. `Detailed` adds timed
  build stages (forest, k-means, assignment, quantiser training, NSG steps,
  CAGRA prune) and per-iteration lines. A sub-component run by a parent (the
  k-means inside IVF, the NN-Descent inside NSG) stays quiet and the parent
  prints one timed line for it. Warnings still print at `Quiet`.
- `parse_annsearch_verbosity(usize)` maps `0`/`1`/`2` to the enum for callers
  that carry verbosity as an integer.
- The binary, RaBitQ and TurboQuant indices (exhaustive and IVF) can keep
  their re-ranking store at f16: `StoreEncoding::F16`, against the default
  `StoreEncoding::Full`. Halves the store file. Rows decode in chunks through
  `half` (F16C or the aarch64 fp16 conversion, picked at runtime) into the
  existing SIMD kernels, so re-ranking runs at f32 speed on an M1 Max. Over
  four synthetic datasets at dim 32 to 512 and rerank factor 20, the mean
  distance ratio moves by at most 0.0001 and Recall@15 drops by 0.001 to
  0.022, worst on correlated data. Data past the f16 range (65504) refuses the
  build. `parse_store_encoding` takes `"full"` or `"f16"`.

**Memory**

- Annoy and Kd forests build in place: one `u32` permutation per tree,
  partitioned recursively, with leaves as ranges into it. No per-node `Vec`s
  and no transient build tree. Leaf ids are `u32` instead of `usize`. The trees
  are bit-identical to before.
- The NN-Descent index no longer keeps a second, reordered copy of the vectors
  inside its forest. The router walks the trees and scores the candidates
  against the index's own vectors. At 200k x 32, `memory_usage_bytes` drops
  from 175 MB to 102 MB (Euclidean), 175 MB to 99 MB (cosine) and 139 MB to
  84 MB (Manhattan).
- Stored point ids are `u32` crate-wide: `original_ids` on every index, the
  IVF-family id and offset arrays, the k-means assignments, the HNSW update
  buffers, the NN-Descent and NSG build buffers and the GPU/MLX host storage.
  Index arithmetic stays `usize`.
- `build_nsg_index` runs its internal NN-Descent graph-only: no forest or
  query graph, and the vectors move into the NSG instead of being copied. The
  resulting index is identical.
- Ball tree builds in place on the same pattern and drops `leaf_indices`, which
  had become the identity once the vectors were stored in leaf order.

**Breaking changes**

- Every `verbose: bool` parameter is now `verbosity: Verbosity`, including
  `train_centroids`, `k_means_clusters_gpu`, `k_means_clusters_mlx` and
  `GraphBuildParams::verbosity` (was `verbose`). Builders and self-queries that
  took no flag before take one now: `build_annoy_index`, `build_balltree_index`,
  `build_exhaustive_index` and its BF16, SQ8, binary, RaBitQ, TurboQuant, GPU
  and MLX variants, `build_kd_tree_index`, `build_lsh_index`,
  `build_vamana_index`, `build_qg_index`, `build_qg_index_with_rotator`,
  `query_nndescent_index_gpu_self`, `query_nndescent_index_mlx_self`,
  `query_cagra_index_mlx` and `query_cagra_index_mlx_self`. The matching
  struct constructors follow.
- The LSH "more than 1% of samples not in the buckets" warning prints at every
  verbosity, as the cross-query path already did.
- Datasets with more than `u32::MAX - 1` samples (`2^31 - 2` for NN-Descent and
  RNN-Descent) are refused with the new `AnnSearchErrors::TooManySamples`.
- `build_hnsw_index`, `build_kd_tree_index` and `build_vamana_index` return a
  `Result` now, like the other builders.
- `KnnValidation::original_ids` returns `&[u32]`.
- `KnnGraphGpu::knn_graph` and `KnnGraphMlx::knn_graph` hold `(u32, T)` pairs.
  `NsgIndex::build_from_knn` takes `&[(u32, T)]`.
- `k_means_clusters_gpu`, `k_means_clusters_mlx`, `k_means_gpu_flat`,
  `assign_all_gpu`, `assign_all_mlx` and `assign_all_parallel` return `u32`
  assignments. `build_csr_layout` and `invert_assignments_csr` return `u32` ids
  and offsets. `assign_secondary_soar` and `adjust_centers` take `u32`
  assignments, `select_probed_clusters` takes `u32` offsets.
- `RaBitQStorage::vector_indices` and `offsets` are `Vec<u32>`, and
  `cluster_vector_indices` returns `&[u32]`.
- `compact_knn_rows` returns `(u32, T)` and lost its id type parameter. The
  `GraphPid` trait is gone.
- The vector store behind the binary indices is one file, `vectors.bin`,
  instead of `vectors_flat.bin` plus `norms.bin`. It carries its own header
  (magic, version, float type, encoding, shape) and a fingerprint of the
  norms, which the index records and checks on load. A store from a
  different dataset of the same shape used to load silently; it is refused
  now. Little-endian hosts only.
- `build_exhaustive_index_binary`, `build_ivf_index_binary`,
  `build_exhaustive_index_rabitq`, `build_ivf_index_rabitq`,
  `build_exhaustive_index_turboquant` and `build_ivf_index_turboquant` take
  `store_encoding: StoreEncoding` after `save_path`, as do the matching
  `new_with_vector_store` / `build_with_vector_store` constructors.
- `MmapVectorStore` is `VectorStore`, `load_vector` is `row` and returns a
  `Cow<[T]>`. The old `VectorStore` trait and `StoreMeta` are gone.
  `AnnSearchErrors::SizeMismatch` is replaced by `StoreInvalid { path, reason }`.
- The `binary` feature pulls in `half`.
- Save format bumped from 6 to 7: the Annoy, Kd, NN-Descent, ball tree,
  IVF-family and binary-index layouts changed. Older indices error on load.

## 0.11.2

**Chore**

- `faer` is pulled in without its default `npy` feature. Nothing here reads
  `.npy` files, and the feature dragged `npyz` -> `py_literal` -> `pest_derive`
  -> `stacker` into the build. On Windows ARM64 cross-builds (MSVC host,
  `aarch64-pc-windows-gnullvm` target) the `stacker` build script fails to find
  `windows.h`. Dependants have to drop the `npy` default as well, as feature
  unification brings it back otherwise.

## 0.11.1

**Feature**

- `k_means_clusters_mlx()` is a public function now to be used in other crates.

## 0.11.0

**Features**

- New experimental `mlx` feature: an Apple Silicon backend that drives Metal
  through [mlx-c](https://github.com/ml-explore/mlx-c) instead of wgpu. Covers
  exhaustive search, IVF (with its own k-means), the NN-Descent kNN graph, CAGRA
  graph optimisation and beam search, and a queryable NN-Descent index
  (`build_*_mlx` / `query_*_mlx` / `query_*_mlx_self`). f32 only. Needs
  `brew install mlx-c`, or `MLX_C_PREFIX` pointing at an install.
- If mlx-c is not found, the build warns and leaves the MLX code out rather
  than failing. `MLX_AVAILABLE` says whether it made it in, and the
  `links = "mlxc"` metadata hands dependants `DEP_MLXC_AVAILABLE`, so their
  build scripts can pick the MLX or the wgpu path.
- MLX against wgpu on an M1 Max at 500k x 128: exhaustive self-kNN 47.7 s to
  15.4 s, IVF build 1.1 s to 0.6 s, CAGRA build 6.2 s to 4.4 s, CAGRA self query
  (beam width 16) 1.27 s to 0.75 s. Full tables in
  [`docs/benchmarks_mlx.md`](docs/benchmarks_mlx.md).
- CPU NN-Descent seeds its graph by joining every forest leaf exhaustively, as
  PyNNDescent does, instead of one forest query per point. On 2.86M Tahoe PCA
  cells (dim 32, degree 15) at 64 trees: build 50.3 s to 28.0 s, recall@15
  0.976 to 0.990. Manhattan keeps the Kd forest and the per-point query.
- CPU NN-Descent defaults to scanpy's tree count, `min(5 + round(sqrt(n) / 20),
  64)`, up from `min(5 + round(n^0.25), 12)`.

**Performance**

- GPU NN-Descent drops mutual-neighbour duplicates before the local join.
- GPU forest leaves that filled the shared-memory budget exactly (dim 128 on a
  32 KiB device) ran far slower on Apple Silicon. Staging now keeps one point of
  headroom.

**Fixes**

- GPU NN-Descent random initialisation could draw the same neighbour id twice.

**Breaking changes**

- CubeCL 0.11 dropped the runtime generic. The GPU builders take a type-erased
  `cubecl::Device` instead of `R::Device`, so
  `build_*_gpu::<f32, WgpuRuntime>(..., device)` becomes
  `build_*_gpu::<f32>(..., device.into())`. Same for `k_means_clusters_gpu`.
- The `gpu-cpu` feature is gone. The cubecl-cpu runtime in 0.11 caps a cube at
  `available_parallelism` units, and every kernel here launches 32- or
  128-wide cubes, which fail silently there.

Only matters if you launch the GPU kernels yourself:

- The Euclidean and cosine kernel pairs merged behind a comptime flag:
  `euclidean_tiled` / `cosine_tiled` are now `dist_tiled`, and likewise
  `dist_tiled_reg`, `compute_ivf_mega_cached`, `flash_assign_vec` and
  `min_dist_vec`. Cosine is normalised on the host instead (`normalise_rows`),
  so the norm arguments are gone.
- `ForestRouter`, `default_forest_trees` and `compact_knn_rows` moved to
  `utils::rp_forest`, shared between wgpu and MLX.
- `knn_graph()` on the wgpu and MLX NN-Descent indices returns `&[(u32, T)]`
  instead of `usize` ids, halving the host graph. `extract_knn` is unchanged.

**Docs**

- New single-cell self-kNN benchmark on 2.86M Tahoe PCA cells against
  pynndescent (scanpy settings), hnswlib and FAISS:
  [`docs/benchmarks_single_cell.md`](docs/benchmarks_single_cell.md).
- MLX rows in the cross-library comparison, and an MLX against wgpu kNN-graph
  section in [`docs/benchmarks_mlx.md`](docs/benchmarks_mlx.md). Every
  benchmark doc rerun.

**Chore**

- `wide` floor raised to 1.6.0, which replaced the deprecated `blend` with
  `select`.
- `cubecl` 0.10 to 0.11, `cubecl-utils-rs` 0.1 to 0.2, `faer` 0.23 to 0.24.

## 0.10.1

**Performance**

- Slightly faster k-means on the GPU and CPU.

## 0.10.0

**Performance**

- Improved speed on the GPU indices across the board. Exhaustive is pending
  dimensions 1.5 to 2x faster, IVF is ~40% faster, NNDescent scales better to
  large k.
- Substantially faster query path on HNSW RaBitQ (related to the bug below).

**Fix**

- HNSW-RaBitQ ex_bits = 0 was buggy. The loop zipped over empty buffers. These
  are now properly initialised.

**Breaking changes**

- Save format has been bumped from version 4 to version 5. Older indices will
  error now upon load.

## 0.9.3

**Performance**

- Ball tree build is several times faster, and the tree no longer depends on
  the number of threads.
- Ball tree query and self-query are roughly twice as fast; vectors are now
  stored in leaf order.
- Faster Annoy build (SIMD hyperplane projection) and Annoy batch and self
  queries (per-thread scratch).
- Faster NN-Descent queries (reused forest scratch).

**Fixes**

- Ball tree query pruned with a bound that was too tight, which cost recall.
  It now uses the exact ball lower bound, so recall is higher at every search
  budget.

**Behaviour changes**

- Ball tree builds produce a different tree for the same seed than earlier
  versions.

## 0.9.2

**Features**

- Make the GEMM accelerated by Mac's Accelerate public.

## 0.9.1

**Features**

- Improved k-means clustering. That impacts every index that is touched by
  k-means clustering.
- `Accelerate` GEMM enabled on MacOS. This makes the GEMM heavy indices even
  faster on MacOS.

## 0.9.0

**Features**

- New quantised graph index (`QgIndex`, `build_qg_index` / `query_qg_index` /
  `query_qg_self`), after SymphonyQG. A Vamana graph where every vertex also
  carries its own neighbours' one-bit RaBitQ codes, quantised against that
  vertex and laid out so one hop estimates all of them with a single SIMD
  sweep. Exact distances are computed only for the vertices the walk pops,
  which is also the anchor the estimates need, so there is no re-ranking stage.
  It is the fastest index in the crate at high recall and it costs roughly two
  to three times the memory of the raw vectors, since each code is stored once
  per in-edge. Build time is Vamana's.
- New HNSW over RaBitQ+ codes (`HnswRaBitQIndex`, `build_hnsw_rabitq_index` /
  `query_hnsw_rabitq_index` / `query_hnsw_rabitq_self`, needs `binary` and
  `quantised`). The graph is linked on exact distances and the vectors are
  dropped, so the index is its topology plus one `ex_bits + 1`-bit code per
  coordinate. `ex_bits` trades size for recall. Saves and loads under
  `serialise` like the other indices.
- RaBitQ gained a fast Hadamard plus Kac-walk rotation, selected automatically
  from 192 dimensions up. It replaces a dense `dim x dim` matvec per vector with
  an `O(dim log dim)` transform and shrinks the stored rotation from `dim^2`
  floats to four bits per coordinate. `RotatorKind` picks it explicitly where
  the default would not. Recall is unchanged.
- The rotation pads to a whole number of code bytes, and at the Hadamard's
  64-coordinate multiple those extra bits carry real signal rather than padding,
  so recall improves where the dimensionality is not already a multiple of 64.
- Better default setting for the beam search in Vamana yielding faster index
  build times and additionally performance improvements from the other indices
  were integrated into Vamana.

**Behaviour changes**

- The RaBitQ indices (`IvfIndexRaBitQ`, `ExhaustiveIndexRaBitQ`) now estimate
  distances with a fast-scan nibble table rather than int4 query bit-planes. The
  int4 path is gone, and with it `RaBitQQuery`, `build_query_planes`,
  `unpack_query_planes`, `dot_query_binary_planes`, `RABITQ_QUERY_PLANES` and
  the `rabitq_dist`, `rabitq_dist_sq`, `rabitq_block_sq`, `dot_query_binary` and
  `popcount` methods on `VectorDistanceRaBitQ`. `encode_query` and
  `encode_query_prerotated` return a `SignScanQuery` and the latter is now
  fallible. Codes are stored only in the blocked scan layout, so
  `RaBitQStorage::binary_codes`, `cluster_binary_codes` and `vector_binary` are
  gone too; the saved format records which architecture packed the codes and
  re-blocks on load, so indices still move between x86 and aarch64.
- `RaBitQPackedVector` no longer carries `popcount`, and `encode_vector` returns
  three values rather than four.
- Saved RaBitQ indices from 0.8.x cannot be loaded.

## 0.8.5

**Features**

- More improvements on the GPU index speed. Faster exhaustive GPU at higher
  dimensionality, generally faster IVF GPU and the GPU forest build for the
  NNDescent is faster at lower dimensions.

## 0.8.4

**Fix**

- The GPU indices (`ExhaustiveIndexGpu`, `IvfIndexGpu`, `NNDescentGpu`) reported
  their padded dimensionality from `DimensionValidation::dim`, while the query
  paths checked the raw query dimensionality against it. Any query against an
  index whose `dim` was not a multiple of the kernels' line size was rejected
  with a bogus `DimensionMismatch`, even though the padding right after the
  check was correct. `dim()` now returns the raw dimensionality.

## 0.8.3

**Fix**

- avx2 and avx512 SIMD wired in properly for x86_64. Benchmarks show decent
  acceleration for avx2 over SSE.

## 0.8.2

**Features**

- Faster RNN implementation and different grid search.

## 0.8.1

**Fixes**

- The PQ family (`ExhaustivePqIndex`, `IvfPqIndex`, `ExhaustiveOpqIndex`,
  `IvfOpqIndex`, `SoarPqIndex`, `SoarOpqIndex`) returned **twice** the cosine
  distance under `Dist::Cosine`. They normalise the data and the query, then
  run ADC in that space, where the sum estimates `||q - v||^2 = 2 (1 - cos)`,
  and nothing rescaled it. Now halved at the lookup table, which is exact and
  costs `m * n_centroids` multiplies per query rather than one per candidate.
  Rank order is untouched, so recall and every published benchmark number are
  unaffected; only callers reading the distances see a change.
- `ExhaustiveIndexBinary::new` hardcoded `Dist::Cosine` and silently discarded
  the metric `build_exhaustive_index_binary` had just parsed, and neither
  constructor validated it, so a Manhattan build plus `rerank = true` hit an
  `unreachable!()` and panicked instead of erroring.

**Features**

- `HnswSq8uIndex` implements `IndexIo`, so the quantised HNSW can be saved and
  loaded like every other index.
- `pub fn n()` on the ten quantised index structs, pairing with the existing
  `DimensionValidation::dim()` so every index exposes its shape the same way.

**Breaking changes**

Shipped in a patch deliberately.

- `ExhaustiveIndexBinary::new` takes a `metric: Dist` argument, after `n_bits`,
  matching `new_with_vector_store` and `IvfIndexBinary::build`. The free
  function `build_exhaustive_index_binary` is unchanged.
- Both `ExhaustiveIndexBinary` constructors now reject `Dist::Manhattan` with
  `DistanceNotSupported`, as `IvfIndexBinary` already did.
- `VectorDistanceAdc` gains a required `metric()` method. Only matters if you
  implement the trait outside this crate.

## 0.8.0

- `gpu` no longer pulls CubeCL's CPU runtime and its prebuilt LLVM; it moved to
  a new test-only `gpu-cpu` feature. **Breaking** if you relied on
  `ann-search-rs/gpu` enabling `cubecl/cpu`.

## 0.7.0

**Features**

- `HnswQuantisedIndex`: an HNSW built and searched entirely on uniformly
  quantised 8-bit codes, inspired by pyglass. One scale shared across all
  dimensions is what makes the integer code distance preserve the ordering of
  the float one, so a single kernel serves construction and query.
  `build_hnsw_sq8u_index` / `query_hnsw_sq8u_index` / `query_hnsw_sq8u_self`.
- New `synthetic` feature: the four dataset generators that produce the
  benchmark tables now live in `src/synthetic/` instead of `examples/commons/`,
  so the published numbers are reproducible outside this repository. Gridsearch
  commands are unchanged.
- `clap` and `approx` moved to dev-dependencies.
- PCAHashing has now ITQ which improves it performance.
- Improved speed for a large number of indices:
  * Exhaustive uses GEMM now
  * Binary build times massively reduced
  * RaBitQ has faster querying times.
  * Improved HNSW and IVF in terms of speed.

**Docs**

- All published benchmark tables regenerated. Stale parameter descriptions and
  measurement claims removed from the templates. Some of the performances have
  changed substantially compared to prior releases.
- New `docs/benchmarks_knn_graph.md`, covering the self-kNN-graph paths: CPU
  NN-Descent extract vs self-beam, the GPU/CAGRA equivalents, and the clustered
  GPU builder.
- SOAR, SOAR-PQ, SOAR-OPQ and the quantised HNSW now have benchmark sections and
  entries in `fill_benchmarks.sh`. `--kind knn_graph` is the fifth kind.

**Breaking changes**

- `ExhaustiveSq8Index` and `IvfSq8Index` rebuilt on the shared-scale `u8`
  quantiser. `build_exhaustive_sq8_index` and `build_ivf_sq8_index` gain an
  `Option<UniformQuantParams>`. `ScalarQuantiser` and `VectorDistanceSq8` are
  gone.
- Sign-based IVF binary indices drop the per-cell residual frame added in 0.5.0
  and go back to global sign codes. The residual frame bought recall at low
  `rerank_factor` on clustered data (0.739 vs 0.579 on the fixture in
  `ivf_binary.rs`) but made Hamming distances incomparable across cells, so
  widening `nprobe` *cost* recall, every query paid a re-encode per probed cell,
  and a kNN graph could not be built without a vector store at all. The
  asymmetric query path covers the same ground for less.
  `Binariser::encode_residual`,
  `AnnSearchErrors::ResidualEncodingUnsupported` and
  `AnnSearchErrors::ResidualCodesRequireVectorStore` are all gone.

**Fixes**

- The indices cannot return distances < 0 due to rounding imprecisions anymore.

## 0.6.0

**Features**

- More inputs accepted: flat arrays, ndarray with 2-dimensional structures.
  First step for a full Python wrapping release of the package.

**Breaking changes**

- Code passing faer matrices to `build_*` / `query_*` is unaffected. Below that
  line, the binariser and quantiser constructors take flat row-major slices
  instead of a `MatRef`: `Binariser::new_simhash` and
  `Binariser::new_pca_hashing` now take `(data, n, dim)`, and
  `RaBitQQuantiser::new` and `TurboQuantQuantiser::new` take anything
  implementing `AnnMatrix`.

## 0.5.2

**Features**

- Clustered NNDescent implementation that uses k-means clusters to split large
  data sets into smaller data sets that fit into GPU-accelerated NNDescent
  approaches.
- GPU-accelerated k-means for the IVF index. Faster initial clustering.
- Improved LSH index that is substantially faster and uses better hashing
  functions.
- SOAR for full vector searches, PQ and OPQ.
- Improved kernels for the GPU-accelerated NNDescent at larger ndim.

**Fix**

- Race condition in the random initialisation of for the GPU-accelerated
  NNDescent kNN generation. This degraded the kNN graphs' quality.

## 0.5.1

**Features**

- Radix seach for GPU-accelerated exhaustive and IVF search. In the former case,
  faster at high k, in the latter case always faster.

## 0.5.0

**Features**

- New `serialise` feature: save indices to disk and load them back. Covers all
  CPU, quantised and binary indices via the `IndexIo` trait plus `save_index` /
  `load_index`. GPU indices are not covered yet.
- Sign-based IVF binary indices now encode the residual against the assigned
  cell's centroid instead of the raw vector. A trade, not a free win: better
  recall at low `rerank_factor`, worse above roughly 15. Documented on
  `IvfIndexBinary::query`. Reverted in 0.7.0.

**Fixes**

- Fixed the sign-based binariser, which was not working properly. Bad indexing
  plus a broken asymmetric query path. It now has proper benchmarks, which is
  why the bugs survived this long.
- Fixed the re-ranking funnel on the sign-based path, which could never improve
  recall because the exact stage only ever saw `k` candidates.
- SimHash now centres the data. Without it, off-origin data put nearly every
  point on the same side of nearly every plane. Note this breaks scale
  invariance, so L2-normalise rows if your magnitudes vary a lot.
- A pile of GPU device limits were assumed rather than queried, all matching
  Apple Silicon. A kernel that busts a limit silently returns zeros, so these
  were wrong-answer bugs on smaller devices. All staging plans and grid
  dispatches now derive from the queried `GpuLimits`.

**Breaking changes**

- Shared GPU primitives moved to `cubecl-utils-rs`. `GpuTensor`, `grid_2d`,
  `pad_vectors` and `LINE_SIZE` are gone from this crate; import them from
  `cubecl_utils_rs::prelude`. `AnnSearchGpuFloat` is now `CubeclFloat`, still
  re-exported from the prelude.
- `GpuTensor::empty` / `from_slice` are fallible. `pick_wg_y`, `grid_2d` and the
  staging planners take a `GpuLimits`.
- `AnnSearchErrors` is `#[non_exhaustive]`; add a `_` arm. Several new variants
  for the store and serialisation paths.
- `Binariser::new_simhash` takes the data matrix first, to fit the centring mean.
- `MmapVectorStore::copy_to_dir` is replaced by `stage_copy_into`, and
  `IndexIo::save_aux` by `IndexIo::stage_aux`.
- With `serialise` on, `AnnSearchFloat` gains `Serialize + DeserializeOwned`.
  `f32` and `f64` are unaffected; a custom float type is not.

**Chore**

- `tempfile` moved to `[dev-dependencies]`, so `binary` no longer drags it into
  normal builds.
- `"itq"` was still documented as a valid binarisation string in five places.
  The parser takes `"pca"`, `"random"` and `"sign"`.

## 0.4.5

**Features**

- Added Navigating Spreading-out Graph (NSG) as an index. The initial kNN
  generation can happen on CPU or GPU.
- Added Relative NN-Descent (RNN-Descent) index.
- Updated the diversification in NNDescent to be more useful. Instead of just
  pruning the graph, it now keeps the desired node degree but yields a better
  graph.
- Further speed improvements on NNDescent.
- **GPU improvements:**
  - Improved speed for exhaustive, IVF and CAGRA GPU indices. Up to 2x faster
    on Apple Silicon via better kernel design.
  - Dropped the CPU beam-search fallback from the CAGRA GPU index. Every query
    batch now goes through the GPU beam search. `query_nndescent_index_gpu`
    loses its `ef_search` parameter; use
    `CagraGpuSearchParams::new(Some(ef), None, None, None)` instead.
  - The CAGRA build no longer recomputes the navigational graph distances on
    the CPU. Those distances were only ever read by the removed fallback, so
    this drops an `O(n * k * dim)` host pass and shrinks the navigational graph
    from 16 to 4 bytes per edge.

## 0.4.4

**Features**

- `CLAUDE.md` for agentic engineering.

**Fixes**

- Added a safeguard for not returning enough neighbours to every IVF index
  in the crate. Enough clusters will be sampled to always return the k
  demanded neighbours.

## 0.4.3

**Fixes**

- Nasty branch bug in wgpu <> metal interaction that made the NNDescent
  iterations for the CAGRA-style ANN search not work.

## 0.4.2

**Features**

- Accessor in the Tensor implementation for the handle within the crate for
  easier sharing across crates.

## 0.4.1

**Features**

- Faster convergence criterium for k-means iterations.

**Fixes**

- Collapse of performance of the GPU indices when reaching higher
  dimensionalities with `SharedMemory::new()`.

## 0.4.0

**Features**

- Version bump on cubecl to `"0.10.0"`. This might introduce *breaking* changes
  if you are using the package with older versions of cubecl.
- More and improved error handling across the board.
- TurboQuantisation with exhaustive and IVF index implemented.
- Better benchmarking with an cell embedding-like benchmark.
- Better entry seeds for the CAGRA style graph.

**Fixes**

- Nasty SIMD overflow bug hitting RaBitQ at higher dimensionalities.

## 0.3.1

**Fix**

- The GPU indices would throw errors due to a bug with the wrong dimensionality
  being checked.

## 0.3.0

Large update with breaking changes!

**Features**

- Manhattan distance enabled on some indices
- More fine control over k-means parameters
- Canberra distance implemented for the SimdDistance trait
  - Renamed of `Euclidean` to `SquaredEuclidean` to be more explicit
- Proper error handling across the board and return of results over asserts

## 0.2.15

**Features**

- A SIMD path version of Hamerly's algorithm for k-means

## 0.2.14

**Features**

- Improved parallel back-ends for Vamana and HNSW which have better memory
  bandwidth
- Improved CAGRA search evaluating more neighbours in one go per iteration.

## 0.2.13

**Features**

- Implemented KmKnn as an accelerated exhaustive search algorithm.

## 0.2.12

**Features**

- Documentation fixes
- Better metrics in the benchmarks

## 0.2.11

**Features**

- Various documentation updates and benchmark updates.
- Improved NNDescent with faster sorts.
- Improved HNSW with less allocation pressure.
- Kd tree/forest implementation.
- Better benchmarks for the quantisation methods with a data set that is more
  challenging for the data sets - also templated version of running the
  benchmarks.
- Removed the ITQ binarisation approach and replaced for PcaHashing.

## 0.2.10

**Features**

- Making some other functions public in the k-means part supporting IVF for
  easier re-use in other crates.

## 0.2.9

**Features**

- Harmonised Annoy to also use the SIMD-accelerated distance metrics and
  returning squared Euclidean distance instead of Euclidean distance.

## 0.2.8

**Features**

- Improved GPU searches. Padding used for exhaustive and IVF, speed increases
  thanks to shared memory.

## 0.2.7

**Features**

- Fix: KnnValidation trait on Annoy and IVF
- Fix: GPU indices dealing with large data sets.
- Better documentation

## 0.2.6

Same as version 0.2.6; however, the MiMalloc activation was made optional via
a feature flag.

## 0.2.5

**Yanked**

Aggressive performance optimisations for various CPU-based indices, removed a
nasty memory corruption bug from the exhaustive GPU search. Reordering of the
module structure to clean up the library.

**Features:**

- Improved Annoy with better memory layout for faster querying.
- Better documentation (more Rust idiomatic), plus correction of copy and paste
  errors.
- Vamana index added and optimised.
- Massive improvement in the IVF indices due to better memory layout. This
  impacts the quantised and some of the binary indices, too.
- Improvements in some of the GPU kernels for exhaustive and IVF search for
  better performance.
- [CAGRA style kNN search](https://arxiv.org/abs/2308.15136) with wgpu
  backend.
- Faster index building for HNSW with a first sequential and then parallel
  phase.
- MiMalloc for better allocations patterns.

**Bugs:**

- *Nasty GPU memory pointer bug* in the exhaustive GPU implementation which
  could cause corruption errors.

## 0.2.4

**Features:**

- *New*: Binary signed quantiser with reranking - for very large vectors.
- SIMD add and assign add added - used for better k-means clustering.
- Improved k-means clustering (impacting IVF) for higher dimensions.
- Improved NNDescent with less
- Improved LSH index with multi-probe support.
- Updated benchmarks with 128 dimensions tested across various indices.

## 0.2.3

**Features:**

- Hotfix for missing avx512 annotations that broke compiling under certain
  conditions.

## 0.2.2

Large update with SIMD improvements across the board.

**Features:**

- SIMD acceleration added for distance calculations.
- BallTree implementation added.
- Binary indices now have reranking based on on-disk reranking.
- Improved GPU kernels for better performance.

## 0.2.1

Larger update with first GPU support and binary quantisations

**Features:**

- GPU acceleration added for IVF and exhaustive.
- BF16 quantisation added.
- Binarised quantisation added, amongst them RaBitQ.

## 0.2.0

Larger update

**Features:**

- Further HNSW index improvements.
- IVF index added.
- First quantisations added: SQ8, PQ, OPQ.

## 0.1.9

**Features:**

- Faster Annoy Descent query time.

## 0.1.8

**Features:**

- Improved NNDescent memory pressure for large detasets.

## 0.1.7

**Features:**

- Fixed HNSW index building bug.

## 0.1.6 (yanked!)

Larger refactor

**Features:**

- Distance trait implementation.
- Exhausive search implementation.
- LSH index added.
- Benchmarks added.
- Introduced a bug to the HNSW index building.

## 0.1.5

Larger refactor.

**Features:**

- Distance trait implementation.
- Exhausive search implementation.
- LSH index added.
- Benchmarks added.

## 0.1.4

**Features:**

- [FANNG](https://openaccess.thecvf.com/content_cvpr_2016/papers/Harwood_FANNG_Fast_Approximate_CVPR_2016_paper.pdf)
  index added

## 0.1.3

**Features:**

- Improved NNDescent implementation.
- Updates to documentations.

## 0.1.2

**Features:**

- Faster HNSW index building speed.
- Updates to documentations.

## 0.1.1

**Features:**

- Faster HNSW query speed and updates to documentations.

## 0.1.0 (release)

First release of the package

**Features:**

- Annoy, HNSW and NNDescent were ported over from the original
  [bixverse](https://github.com/GregorLueg/bixverse) codebase.
