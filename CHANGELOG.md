# News

## 0.6.0

**Features**

- More inputs accepted: flat arrays, ndarray with 2-dimensional structures.
  First step for a full Python wrapping release of the package.

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
  `IvfIndexBinary::query`.

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
