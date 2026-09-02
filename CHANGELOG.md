# News

## 0.8.1

**Python**

- The eleven quantised indices are now bound: `ExhaustiveBf16Index`,
  `IvfBf16Index`, `ExhaustiveSq8Index`, `IvfSq8Index`, `HnswSq8uIndex`,
  `ExhaustivePqIndex`, `IvfPqIndex`, `ExhaustiveOpqIndex`, `IvfOpqIndex`,
  `SoarPqIndex`, `SoarOpqIndex`. Same four-method surface as the rest, save /
  load / pickle included, float32 and float64 both supported. Shipped as
  `ann-search` 0.2.0. The binary indices are not bound yet.
- Python docs corrected against the regenerated benchmark tables. HNSW is now
  the cheapest graph index to build rather than the most expensive, which
  reversed three separate claims in `choosing.md`.

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

**Python**

- Python bindings under `python/`, built with PyO3 and maturin. scikit-learn
  shaped estimators over the CPU indices, plus the synthetic generators below.
  Not published yet, but installable from the repo.

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
