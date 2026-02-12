# News

## 0.2.4

**Features:**

- SIMD add and assign add added.

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
