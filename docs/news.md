# News

## 0.2.11

**Features**

- Various documentation updates and benchmark updates.
- Improved NNDescent with faster sorts.
- Kd tree/forest implementation.
- Better benchmarks for the quantisation methods with a data set that is more
  challenging for the data sets.
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
