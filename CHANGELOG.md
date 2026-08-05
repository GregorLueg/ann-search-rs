# News

## 0.5.0

**Breaking changes**

- The shared GPU primitives now live in `cubecl-utils-rs`, which the `gpu`
  feature pulls in. `ann_search_rs::gpu::tensor::GpuTensor`,
  `ann_search_rs::gpu::grid_2d`, `pad_vectors` and `LINE_SIZE` are gone from
  this crate; import them from `cubecl_utils_rs::prelude` instead. There are no
  compatibility re-exports.
- `AnnSearchGpuFloat` is renamed `CubeclFloat` and now comes from
  `cubecl-utils-rs`. It is still re-exported from `ann_search_rs::prelude`.
- `GpuTensor::empty` and `GpuTensor::from_slice` are fallible, and check the
  requested allocation against the device's per-binding limit before asking for
  it. `GpuTensor::read` now truncates to the tensor's shape, which only changes
  behaviour for a `reshaped_view` whose parent allocation is larger.
- `pick_wg_y` takes the element size and a `GpuLimits`. `grid_2d` takes a
  `GpuLimits` and returns a `Result`. `plan_local_join_staging` and
  `plan_topk_merge` take a `GpuLimits` in place of a raw byte budget.
- `AnnSearchErrors::DimTooHighForSharedMemory` carries `required` and
  `available` alongside `chosen_dim`. New variant `CubeclUtils` wraps
  `cubecl_utils_rs::CubeclUtilsErrors`.

**Fixes**

- Several device limits were assumed rather than queried, all of them matching
  what Apple Silicon reports. A kernel that busts a limit does no work and
  returns zeros without reporting anything, so these were silent-wrong-answer
  bugs on any smaller device rather than tuning misses.
- `pick_wg_y` was a fixed `dim -> wg_y` table sized for 32 KiB of shared memory
  and `f32`. It now starts from that table as an upper bound and shrinks until
  the staging fits the device's shared memory, unit-per-cube and cube-dimension
  limits. On a 32 KiB device with `f32` it returns exactly what it did before.
- `CubeDim::new_2d(32, 32)` at `dim_padded <= 128` is 1024 units per cube,
  which was never checked against `max_units_per_cube`.
- The CAGRA beam search allocated a fixed 8 KiB visited-node hash table plus
  `4 * dim_padded` with no budget check at all. `plan_beam_search_staging`
  shrinks the hash table instead, which costs revisits rather than correctness.
- `two_hop_refinement` and `cagra_rank_prune_shared` had unbounded
  shared-memory allocations.
- `compute_max_leaf_size` clamped its lower bound to two rather than failing, so
  above roughly `dim_padded = 4096` it returned a leaf size whose staging did
  not fit. It now errors, and accounts for the element size instead of assuming
  `f32`.
- `grid_2d` panicked on a cube count of zero, and only ever bounded its x
  dimension, so `y` could exceed the device limit past `max_dim^2`.
- Four hand-rolled copies of the 65535 clamp replaced with `grid_2d`.
  `build_knn_graph_gpu` dispatched a flat 65535 cubes in x regardless of `n`,
  which at `n = 10_000` launched 55_535 cubes that did nothing, on every
  iteration and every refinement sweep.
- Grid dimensions that scale with `k`, the cluster count or the largest
  cluster's size now go through `checked_cube_count`, so an oversized value is
  a typed error rather than a zero-filled result.
- The exhaustive path held an `n_q * db_chunk` transient of 512 MiB for `f32`,
  sized from two flat constants. The DB chunk now shrinks to the device's
  binding limit.
- `calculate_safe_batch_size` hardcoded eight bytes per candidate and never
  consulted the device; it now uses the element size and caps its target
  against the binding limit.

- `AnnSearchErrors` is now `#[non_exhaustive]`. Variants come and go with the
  optional features, so a downstream exhaustive `match` was never going to hold
  across a feature-flag change. Add a `_` arm.
- `AnnSearchErrors::IoError` is no longer gated behind `binary`, and the
  `From<std::io::Error>` impl is now unconditional. Under default features that
  is a variant which did not exist before; `serialise` adds six more.
- New variants: `StoreFileUnavailable`, `StoreShapeMismatch`,
  `ResidualEncodingUnsupported`, `ResidualCodesRequireVectorStore` (all
  `binary`), `TruncatedIndexFile` and `TrailingBytes` (both `serialise`).
- `Binariser::new_simhash` takes the data matrix as its first argument. It needs
  it to fit the centring mean.
- `IvfIndexBinary::generate_knn` returns `ResidualCodesRequireVectorStore` for a
  sign-based index built without a vector store. Its codes are relative to each
  cell's centroid and only comparable within a cell, but that path scans every
  cluster, so it would have returned a quietly degraded graph. Build with
  `build_with_vector_store()`.
- `MmapVectorStore::copy_to_dir` is gone, replaced by
  `MmapVectorStore::stage_copy_into`, which writes under temporary names and
  hands the rename to the caller. The `IndexIo::save_aux` hook becomes
  `IndexIo::stage_aux` for the same reason.
- With `serialise` on, `AnnSearchFloat` gains `Serialize + DeserializeOwned`.
  Cargo unifies features across the graph, so one crate anywhere enabling
  `serialise` adds that bound for every other consumer. `f32` and `f64` are
  unaffected; a downstream custom float type is not.

**Features**

- New `serialise` feature: indices can be saved to disk and loaded back in.
  Covers all CPU, quantised and binary indices via a new `IndexIo` trait, plus
  `save_index` / `load_index` wrappers in the crate root. Backed by `serde` and
  `bincode`, both optional and only pulled in with the feature.
  - An index is saved as a directory. `index.bin` holds the payload; the binary
    indices copy their on-disk re-ranking store alongside it, so a saved index
    is self-contained and can be moved. Saving into the directory the store
    already occupies skips the copy.
  - Every file is written under a temporary name and renamed into place at the
    end, `index.bin` last. A save that fails part-way leaves the previous bundle
    untouched, or leaves no `index.bin` and fails loudly on the next load. It
    never leaves a directory that loads clean and re-ranks against the wrong
    vectors.
  - `index.bin` carries a header with a magic number, format version, index kind
    and float width. Loading the wrong index type or the wrong float type is a
    typed error rather than silent garbage. Truncated files and trailing bytes
    are both rejected.
  - Integers are varint-encoded and floats are little-endian, so `index.bin`
    moves between 32- and 64-bit machines and between endiannesses. The binary
    indices' store files are raw native-endian dumps and do not, so a bundle
    carrying one is little-endian only in practice.
  - GPU indices are not covered yet. They hold live device handles, and
    `IvfIndexGpu` keeps its centroids GPU-side with no host mirror.
- Sign-based IVF binary indices encode the residual against the assigned cell's
  centroid rather than the raw vector. A cluster far from the origin puts every
  member on the same side of every coordinate plane, so a global sign code
  identifies the cluster and says nothing about position inside it, which is the
  resolution kNN works at. For Cosine the residual is taken between unit-length
  vectors, expressed without a division as a scale pair, since neither the data
  nor the centroids are normalised at build time.
  - **This is a trade, not a free win.** Measured on 8 blobs at dim 32 with
    32-bit codes, recall@10, residual against the previous global coding:

    | rerank_factor | nprobe | residual | global |
    |---------------|--------|----------|--------|
    | 5             | 8      | **0.553**| 0.337  |
    | 10            | 8      | **0.739**| 0.584  |
    | 25            | 8      | 0.950    |**1.000**|
    | 50            | 8      | 0.997    |**1.000**|
    | 10            | 16     | 0.542    |**0.584**|

    It wins below `rerank_factor` ≈ 15 and costs a few points above it. The
    global coding was insensitive to `nprobe`; this one degrades as `nprobe`
    grows, because Hamming distances are only strictly comparable *within* a
    cell: every vector in a distant cell shares a direction from that cell's
    centroid, so their residual signs agree with the query's spuriously. The
    cross-cell term is currently normalised away rather than corrected for, so
    there is a known repair outstanding. Prefer narrow pools, or reach for
    `IvfIndexRaBitQ`, which carries the per-vector correction terms this index
    deliberately does not. Documented on `IvfIndexBinary::query`.

**Fixes**

- The sign-based binariser was not behaving. Two bugs, one related to wrong
  indexing and another bug affecting the asymmetric queries. Both were fixed
  now.
- The sign-based index was never properly benchmarked, why the bugs persisted
  that long. Also fixed.
- Re-ranking on the sign-based path could never improve recall. `query_reranking`
  passed `k` as the *k* argument to `query_asymmetric`, which truncates to that
  argument internally, so the exact-distance stage received exactly `k`
  candidates and could only reorder what the binary stages had already picked.
  The funnel was `k*2*rf -> k -> k`; it is now `k*2*rf -> k*rf -> k`. Affected
  both the exhaustive and the IVF binary indices. The projection methods were
  never affected.
- SimHash never centred the data. Its hyperplanes pass through the origin, so on
  data whose mean sits away from it nearly every point landed on the same side
  of nearly every plane and the codes carried almost no information. It now fits
  a per-feature mean like PCA hashing does. **This changes results on
  previously-valid input and is also a trade**: centring breaks SimHash's scale
  invariance, so `encode(v)` and `encode(50v)` no longer agree even though their
  cosine distance is zero. On non-negative data with magnitudes spread over two
  decades, cosine recall@10 measured 0.206 centred against 0.318 uncentred,
  while on fixed-magnitude off-origin data centring wins. Single-cell count
  matrices with varying library size fall in the losing regime; L2-normalising
  rows before indexing sidesteps it.

**Documentation**

- `"itq"` was documented as a valid binarisation string on five call sites (old
  method that was removed). The parser accepts `"pca"`, `"random"` and `"sign"`
  and falls back to random projections for anything else.

**Chore**

- `tempfile` moved from the `binary` feature to `[dev-dependencies]`. It was
  only ever used by tests and the gridsearch examples, so `binary` no longer
  drags it into normal builds.
- A handful of impl blocks in `binary/` spelled out `AnnSearchFloat`'s bounds
  longhand; they now name the trait.

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
