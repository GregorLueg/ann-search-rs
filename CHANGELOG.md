# News

## Unreleased

**Features**

- Register-tiled distance kernels for the GPU exhaustive path,
  `euclidean_tiled_reg` and `cosine_tiled_reg`. Each thread computes a 4x4
  block of the distance matrix instead of a single entry, cutting memory
  operations per FMA from 1.25 to 0.3125. Measured 2.2x at dim=32, 1.9x at 64,
  1.8x at 128 and 1.4x at 512 on the kernel, and ~1.7x end to end. Bit-exact
  against the untiled kernels. The untiled versions stay as the fallback for
  dim 1025-2048, where the query tile is too short to divide.
- The GPU benches now report device limits, cross-check every kernel against a
  reference before timing it, and assert the output is real. Every launch in
  the crate is `launch_unchecked`, which returns zeros and reports no error
  when it busts a device limit.

- The IVF centroid probe uses the tiled kernels too: 1.58 -> 1.00 ms/launch
  Euclidean, 1.50 -> 0.80 ms/launch cosine. The IVF *mega* kernels do not.
  Tiled variants were written and measured 1.8x slower, so they were dropped;
  see the note at the launch site. Same technique, opposite result, because
  those kernels bind one task per y-row with its own cluster extent.

**Fixes**

- `extract_topk` held its running top-k in global memory, so it re-read the
  k-th distance on every column of the chunk and the whole k-row on every
  accepted candidate. Now staged in registers and flushed once. 26% off the
  kernel, 6-13% off the exhaustive pipeline.
- The IVF task list is now grouped by cluster rather than by query. The mega
  kernel binds one task per `UNIT_POS_Y` row, so ordered by query a cube was
  one query against `wg_y` different clusters: every row read a disjoint DB
  region and nothing was reused. Grouped by cluster it is `wg_y` queries
  against one cluster, so the DB tile is read once and reused across rows.
  Mega kernel 4 422 -> 3 057 ms over the gridsearch (1.45x), self-kNN queries
  1.28-1.38x faster end to end, recall unchanged to four decimal places.
  The previous `sort_unstable_by_key(|t| t.0)` was a no-op: the build loop
  already emitted queries in ascending order. Also dropped the intermediate
  `Vec` of tuples and the four map-collect passes that split it.
- `local_join_shared`, `two_hop_refinement` and `cagra_rank_prune_shared`
  dispatched a hardcoded 65535 cubes in x regardless of `n`, wasting 6.5x at
  n=10k and ~31% at n=100k. They use `grid_2d` now, like every other launch.
- `TopkCoalescedBench` was defined but never constructed, in every revision
  since it was added, so the coalesced-vs-serial top-k comparison the bench
  advertised had never run. It runs now, and the coalesced kernel loses by 32%
  on wgpu/Metal. Recorded in the kernel docs so it does not get re-litigated.

- The IVF candidate buffers are reused across query batches instead of being
  allocated per batch. The mega kernel's first write to a fresh allocation
  faults in ~1.4 GB at 15k queries: an identical second launch over the same
  buffers runs in 22.3 ms against 61.1 ms for the first, and the isolated
  kernel time is 22.6 ms, so nearly two thirds of the apparent cost was paging,
  not compute. Adds `GpuTensor::reshaped_view` to alias one allocation under
  several shapes. Single-batch queries cannot benefit; multi-batch self-kNN
  gains a few percent more on top of the cluster grouping.

- Index build is ~1.95x faster at 500k x 64D (5.10s -> 2.61s), from five
  kernel fixes:
  - `leaf_pairwise_proposals` and `local_join_shared` walked the candidate
    upper triangle by flat pair index, decoding each one with a serial loop
    that ran up to `leaf_size` times *per pair per thread*. At leaf_size 124
    that is ~29k serial steps against ~15k FMAs, so the indexing cost more
    than the distance computation it was indexing. Now walked by row.
    2.2x and 1.4x on those kernels.
  - `two_hop_refinement` re-read the node's own neighbour row from global
    memory once per candidate, k^3 loads per node of a row every thread in
    the cube shares, and re-issued each vector load four times by indexing
    scalars instead of lines. Row staged in shared memory, loop vectorised,
    node norm hoisted. 3.7x.
  - `merge_proposals` scanned and shifted the graph row in global memory for
    every proposal, up to `MAX_PROPOSALS` times per node. Row staged in
    registers and flushed once. 1.5x.
  Recall shifts by under 0.001: the traversal order changes which proposals
  survive when a node overflows `MAX_PROPOSALS`, so the graph differs slightly
  by construction.
  - Forest tree construction projected the data onto one random vector per
    level, so it read the whole vector matrix `max_depth` times and blocked on
    a readback after each. The projections never depend on the partitioning,
    only on the tree seed and level index, so they are now applied in a single
    kernel writing `[max_depth, n]` level-major, with one readback per tree.
    Only the median-and-scatter step stays sequential. Tree construction
    1.11s -> 369ms (3.0x); the dot-product kernel drops from 347ms over 260
    launches to 36ms over 20.
- `GpuTensor::shape` and `GpuTensor::len` / `is_empty` accessors.

**Housekeeping**

- Removed `compute_ivf_mega_cosine` (no call sites, no tests), the unused
  `xorshift_search` helper, a duplicate `pad_vectors` shadowing the one in
  `gpu::mod`, and ~915 lines of commented-out cubecl debug kernels. The
  findings from that debugging are distilled into four codegen rules in the
  `nndescent_gpu` module header.

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
