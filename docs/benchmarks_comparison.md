## Comparison against other libraries

How does ann-search stack up against what people reach for in Python? We run
the [ann-benchmarks](https://ann-benchmarks.com) protocol on three of its
datasets and compare every library at **equal recall**, not at equal settings.
Two libraries with the same `ef_search` land at different recall, so "faster at
the same parameters" says little. Each library builds once, its search knob is
swept, and the tables give the best queries per second (QPS) it reaches at
recall@10 of 0.90, 0.95 and 0.99. GloVe is hard enough that most indices
never reach 0.95 there, so its targets are 0.70, 0.80 and 0.90.

Run via the Python bindings (from `benchmarks/`):

```bash
uv run bench.py run --dataset fashion-mnist-784-euclidean --threads 10
uv run bench.py run --dataset fashion-mnist-784-euclidean --threads 1
uv run bench.py run --dataset sift-128-euclidean --threads 10
uv run bench.py run --dataset glove-100-angular --threads 10
uv run bench.py summarise
```

### Setup

- **Datasets:** Fashion-MNIST (60,000 x 784, Euclidean), SIFT (1,000,000 x
  128, Euclidean) and GloVe (1,183,514 x 100, cosine). 10,000 held-out queries
  each, k = 10, ground truth from the HDF5 files. Recall is the overlap of
  neighbour ids.
- **Matched builds:** HNSW at M = 16 and ef_construction = 200 (ann-search,
  hnswlib, faiss `IndexHNSWFlat`, usearch at `f32`); the same graph on uniform
  8-bit codes (ann-search `HnswSq8uIndex`, faiss `IndexHNSWSQ` with
  `QT_8bit_uniform`), both building on the codes; Annoy at 50 trees (ann-search, annoy);
  NN-Descent graphs of degree 30 (ann-search, pynndescent); IVF at
  `nlist = sqrt(n)` (ann-search, faiss `IndexIVFFlat`), each with its own
  default training: 30 k-means iterations here, 10 in faiss, which is most of
  the gap in IVF build time; exact search as the baseline (ann-search, faiss `IndexFlat`).
- **Swept at query time:** `ef_search` 10 to 640 (NN-Descent to 2,560);
  Annoy `search_k` 500 to 250,000; `nprobe` 2 to 128; pynndescent `epsilon`
  0 to 0.5; CAGRA `beam_width` 16 to 256.
- **Timing:** one build, then per sweep point an untimed 100-query warm-up and
  the median of three timed calls over all 10,000 queries. pynndescent's
  `prepare()` counts towards its build, and its numba JIT is warmed beforehand.
- **Memory:** every library and method runs in its own process, measured as
  the kernel's `phys_footprint` (what Activity Monitor shows; RSS on Linux).
  *Build peak* is the peak above the baseline before the training data was
  loaded, so it includes the input data. *Index* is what stays once the
  caller's copy of the data is freed and one 100-query batch has run, so
  buffers an index allocates on first use count too: an index that copies the
  vectors and one that keeps a reference both count them once. Freed memory the allocator
  holds on to still counts, as it does for any process that built the index.
- **Threads:** pinned through each library's own argument plus
  `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` and friends. annoy's Python API
  queries one vector at a time, so its multi-threaded rows split the queries
  over a thread pool.

### Caveats

- The bindings build with the `accelerate` feature, so on macOS exact search
  runs through Apple Accelerate. The other docs here run on faer.
- `ann_search_gpu` rows are the wgpu indices (exhaustive, IVF, CAGRA) on the
  M1 Max's integrated GPU. Not apples to apples: they sit next to CPU
  libraries because wgpu runs on Metal, Vulkan and DX12, so no NVIDIA card is
  needed. Thread counts don't apply to them, so the 1-thread run leaves them
  out. Memory is unified here, so their footprint includes the device
  buffers.
- `ann_search_mlx` rows are the same three indices on the experimental MLX
  backend, which drives Metal directly. Apple Silicon only, and the binding has
  to be built with the `mlx` feature and mlx-c installed (`brew install
  mlx-c`); without it those entries fail to build. Same thread and memory
  caveats as the wgpu rows.
- Builds are a single run each. Read small gaps as ties.
- "Best QPS at or above a target" depends on where the sweep grid steps fall.
  A library whose grid point lands at 0.909 counts at 0.90; one that lands at
  0.896 has to take the next, much slower, point. The figures show the whole
  curve; check them before quoting a ratio from the tables.
- Fashion-MNIST and SIFT hold integers from 0 to 255, so uniform 8-bit codes
  are close to lossless on them and the SQ8 rows look better than they will on
  real embeddings. GloVe is the honest test.

## Table of Contents

- [Fashion-MNIST, 10 threads](#fashion-mnist-10-threads)
- [Fashion-MNIST, 1 thread](#fashion-mnist-1-thread)
- [SIFT, 10 threads](#sift-10-threads)
- [GloVe, 10 threads](#glove-10-threads)

### Fashion-MNIST, 10 threads

![](figures/comparison_fashion-mnist-784-euclidean_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 1.6 | 1,118 | 1,119 | 22,870 | 22,870 | 11,171 |
| annoy | annoy | 4.4 | 430 | 430 | 10,418 | 5,607 | 3,469 |
| cagra | **ann_search_mlx** | 4.7 | 772 | 658 | 163,700 | 163,700 | 163,700 |
| cagra | **ann_search_gpu** | 6.5 | 1,611 | 1,417 | 55,276 | 55,276 | 37,286 |
| exhaustive | **ann_search_mlx** | 0.0 | 584 | 608 | 66,746 | 66,746 | 66,746 |
| exhaustive | **ann_search** | 0.0 | 404 | 404 | 17,561 | 17,561 | 17,561 |
| exhaustive | **ann_search_gpu** | 0.0 | 584 | 1,128 | 13,913 | 13,913 | 13,913 |
| exhaustive | faiss | 0.0 | 359 | 359 | 13,276 | 13,276 | 13,276 |
| hnsw | **ann_search_sq8** | 1.3 | 602 | 602 | 321,450 | 218,133 | 140,462 |
| hnsw | faiss_sq8 | 3.7 | 234 | 234 | 166,719 | 110,922 | n/a |
| hnsw | faiss | 4.0 | 369 | 369 | 113,040 | 75,622 | 47,238 |
| hnsw | **ann_search** | 3.5 | 414 | 414 | 108,733 | 72,811 | 46,311 |
| hnsw | usearch | 7.0 | 375 | 375 | 53,167 | 39,087 | 26,211 |
| hnsw | hnswlib | 8.4 | 376 | 376 | 46,889 | 33,380 | 22,432 |
| ivf | **ann_search_mlx** | 0.3 | 1,654 | 1,128 | 393,023 | 393,023 | 252,162 |
| ivf | **ann_search_gpu** | 0.5 | 2,036 | 1,846 | 279,870 | 279,870 | 168,406 |
| ivf | **ann_search** | 0.3 | 804 | 804 | 145,064 | 145,064 | 124,128 |
| ivf | faiss | 0.5 | 437 | 437 | 35,348 | 18,447 | 18,447 |
| nndescent | **ann_search** | 2.6 | 1,790 | 433 | 74,473 | 74,473 | 48,056 |
| nndescent | pynndescent | 3.1 | 423 | 378 | 17,519 | 15,940 | 10,869 |

### Fashion-MNIST, 1 thread

The ann-benchmarks convention.

![](figures/comparison_fashion-mnist-784-euclidean_t1.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 5.7 | 1,106 | 1,106 | 4,466 | 4,466 | 2,206 |
| annoy | annoy | 17.7 | 410 | 410 | 1,443 | 788 | 436 |
| cagra | **ann_search_mlx** | 4.9 | 757 | 643 | 187,752 | 187,752 | 162,168 |
| cagra | **ann_search_gpu** | 6.6 | 1,606 | 1,408 | 29,266 | 29,266 | 23,174 |
| exhaustive | **ann_search_mlx** | 0.0 | 584 | 607 | 56,259 | 56,259 | 56,259 |
| exhaustive | **ann_search_gpu** | 0.0 | 584 | 1,128 | 13,964 | 13,964 | 13,964 |
| exhaustive | faiss | 0.0 | 359 | 359 | 8,941 | 8,941 | 8,941 |
| exhaustive | **ann_search** | 0.0 | 404 | 405 | 5,213 | 5,213 | 5,213 |
| hnsw | **ann_search_sq8** | 8.9 | 600 | 600 | 48,472 | 31,979 | 19,886 |
| hnsw | faiss | 16.9 | 368 | 368 | 26,224 | 17,347 | 10,579 |
| hnsw | faiss_sq8 | 20.7 | 233 | 233 | 23,056 | 15,966 | n/a |
| hnsw | **ann_search** | 17.1 | 413 | 413 | 22,760 | 15,332 | 9,911 |
| hnsw | usearch | 54.1 | 374 | 374 | 7,166 | 5,098 | 3,349 |
| hnsw | hnswlib | 67.7 | 375 | 375 | 5,668 | 4,052 | 2,719 |
| ivf | **ann_search_mlx** | 0.2 | 1,654 | 1,126 | 375,646 | 375,646 | 243,915 |
| ivf | **ann_search_gpu** | 0.5 | 2,035 | 1,846 | 284,494 | 284,494 | 174,510 |
| ivf | **ann_search** | 0.4 | 779 | 779 | 36,816 | 36,816 | 31,616 |
| ivf | faiss | 0.6 | 446 | 446 | 12,674 | 6,562 | 6,562 |
| nndescent | pynndescent | 13.0 | 438 | 394 | 18,494 | 16,746 | 11,266 |
| nndescent | **ann_search** | 12.8 | 1,396 | 463 | 13,820 | 13,820 | 9,919 |

### SIFT, 10 threads

![](figures/comparison_sift-128-euclidean_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 7.0 | 3,286 | 3,287 | 11,714 | 6,844 | 3,236 |
| annoy | annoy | 18.8 | 2,009 | 2,013 | 8,166 | 4,850 | 1,374 |
| cagra | **ann_search_mlx** | 13.2 | 4,373 | 2,473 | 83,354 | 50,536 | 14,470 |
| cagra | **ann_search_gpu** | 18.2 | 6,139 | 4,987 | 61,986 | 29,200 | 3,506 |
| exhaustive | **ann_search_mlx** | 0.1 | 1,596 | 1,978 | 11,311 | 11,311 | 11,311 |
| exhaustive | **ann_search_gpu** | 0.1 | 1,587 | 3,068 | 5,040 | 5,040 | 5,040 |
| exhaustive | **ann_search** | 0.1 | 1,099 | 1,100 | 3,506 | 3,506 | 3,506 |
| exhaustive | faiss | 0.0 | 977 | 977 | 1,003 | 1,003 | 1,003 |
| hnsw | **ann_search_sq8** | 24.5 | 1,301 | 944 | 191,208 | 108,216 | n/a |
| hnsw | faiss_sq8 | 43.5 | 782 | 782 | 99,987 | 54,299 | n/a |
| hnsw | **ann_search** | 44.2 | 1,247 | 1,248 | 85,270 | 48,693 | 26,961 |
| hnsw | faiss | 50.7 | 1,152 | 1,152 | 79,351 | 42,434 | 22,626 |
| hnsw | hnswlib | 80.3 | 1,268 | 1,268 | 40,794 | 24,335 | 13,711 |
| hnsw | usearch | 104.5 | 1,236 | 1,236 | 31,874 | 18,115 | 9,888 |
| ivf | **ann_search_mlx** | 0.5 | 5,485 | 2,713 | 231,092 | 117,389 | 64,803 |
| ivf | **ann_search_gpu** | 1.0 | 5,529 | 5,030 | 126,428 | 65,912 | 31,219 |
| ivf | **ann_search** | 0.7 | 1,653 | 1,653 | 72,496 | 56,162 | 40,256 |
| ivf | faiss | 8.8 | 1,350 | 1,350 | 13,574 | 6,984 | 3,575 |
| nndescent | **ann_search** | 23.9 | 4,183 | 2,551 | 63,917 | 36,800 | 20,107 |
| nndescent | pynndescent | 27.3 | 2,279 | 1,702 | 14,147 | 9,334 | 3,957 |

### GloVe, 10 threads

![](figures/comparison_glove-100-angular_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.70 | QPS @ 0.80 | QPS @ 0.90 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 10.5 | 2,875 | 2,875 | 6,899 | 3,012 | 1,650 |
| annoy | annoy | 25.2 | 2,546 | 2,525 | 3,843 | 1,706 | 936 |
| cagra | **ann_search_mlx** | 17.8 | 4,880 | 2,144 | 62,946 | 35,124 | n/a |
| cagra | **ann_search_gpu** | 21.8 | 6,749 | 5,165 | 26,917 | 9,597 | n/a |
| exhaustive | **ann_search_mlx** | 0.1 | 1,468 | 1,920 | 6,096 | 6,096 | 6,096 |
| exhaustive | **ann_search_gpu** | 0.1 | 1,473 | 3,295 | 5,902 | 5,902 | 5,902 |
| exhaustive | faiss | 0.1 | 1,364 | 1,364 | 3,915 | 3,915 | 3,915 |
| exhaustive | **ann_search** | 0.1 | 1,021 | 1,021 | 2,946 | 2,946 | 2,946 |
| hnsw | **ann_search_sq8** | 52.3 | 1,685 | 903 | 112,202 | 33,803 | 9,788 |
| hnsw | **ann_search** | 73.7 | 1,196 | 1,197 | 69,671 | 40,565 | 12,161 |
| hnsw | faiss | 69.3 | 1,578 | 1,578 | 65,158 | 19,782 | 5,106 |
| hnsw | hnswlib | 112.0 | 1,243 | 1,243 | 39,306 | 13,408 | 3,968 |
| hnsw | usearch | 160.1 | 1,244 | 1,243 | 26,428 | 8,580 | 2,470 |
| hnsw | faiss_sq8 | 120.0 | 1,234 | 1,234 | 22,869 | 12,722 | 3,428 |
| ivf | **ann_search_mlx** | 0.5 | 5,043 | 2,502 | 443,949 | 248,355 | 70,478 |
| ivf | **ann_search_gpu** | 0.9 | 5,065 | 4,599 | 256,301 | 136,007 | 32,873 |
| ivf | **ann_search** | 1.0 | 1,531 | 1,531 | 97,061 | 69,062 | 37,299 |
| ivf | faiss | 6.2 | 1,707 | 1,707 | 29,015 | 15,414 | 4,118 |
| nndescent | **ann_search** | 36.7 | 4,180 | 2,581 | 64,356 | 36,924 | 10,570 |
| nndescent | pynndescent | 47.9 | 2,416 | 1,750 | 7,449 | 4,848 | 1,893 |

---

### Runtime info

*ann-search-rs 0.12.2 (commit v0.12.2-22-g50a9abd-dirty), run on 2026-11-13 on Apple M1 Max. Runs span commits v0.12.2-21-g8f55df8, v0.12.2-22-g50a9abd-dirty.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
