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
- **Swept at query time:** `ef_search` 10 to 640; annoy `search_k` 500 to
  50,000; `nprobe` 2 to 128; pynndescent `epsilon` 0 to 0.3.
- **Timing:** one build, then per sweep point an untimed 100-query warm-up and
  the median of three timed calls over all 10,000 queries. pynndescent's
  `prepare()` counts towards its build, and its numba JIT is warmed beforehand.
- **Memory:** every library and method runs in its own process, measured as
  the kernel's `phys_footprint` (what Activity Monitor shows; RSS on Linux).
  *Build peak* is the peak above the baseline before the training data was
  loaded, so it includes the input data. *Index* is what stays once the
  caller's copy of the data is freed: an index that copies the vectors and one
  that keeps a reference both count them once. Freed memory the allocator
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
| annoy | **ann_search** | 1.7 | 893 | 534 | 23,910 | 23,910 | 11,767 |
| annoy | annoy | 4.4 | 500 | 308 | 10,639 | 6,472 | 3,691 |
| cagra | **ann_search_gpu** | 14.1 | 2,221 | 1,673 | 17,596 | 17,596 | 15,002 |
| exhaustive | **ann_search** | 0.0 | 404 | 225 | 17,577 | 17,577 | 17,577 |
| exhaustive | faiss | 0.0 | 360 | 180 | 12,734 | 12,734 | 12,734 |
| exhaustive | **ann_search_gpu** | 0.0 | 584 | 225 | 12,605 | 12,605 | 12,605 |
| hnsw | **ann_search_sq8** | 1.3 | 587 | 58 | 324,716 | 221,220 | 132,871 |
| hnsw | faiss_sq8 | 3.3 | 235 | 56 | 186,649 | 128,734 | n/a |
| hnsw | faiss | 3.7 | 369 | 189 | 118,899 | 78,102 | 49,048 |
| hnsw | **ann_search** | 3.4 | 412 | 233 | 112,397 | 75,612 | 46,819 |
| hnsw | usearch | 6.7 | 376 | 196 | 57,174 | 41,494 | 27,288 |
| hnsw | hnswlib | 8.0 | 375 | 196 | 46,700 | 33,910 | 21,966 |
| ivf | **ann_search_gpu** | 1.5 | 2,166 | 1,449 | 194,859 | 194,859 | 150,342 |
| ivf | **ann_search** | 1.2 | 762 | 223 | 36,431 | 36,431 | 18,466 |
| ivf | faiss | 0.5 | 419 | 236 | 35,615 | 19,238 | 19,238 |
| nndescent | **ann_search** | 3.2 | 1,045 | 495 | 79,726 | 79,726 | 27,437 |
| nndescent | pynndescent | 3.0 | 393 | 213 | 19,019 | 17,275 | 11,632 |

### Fashion-MNIST, 1 thread

The ann-benchmarks convention.

![](figures/comparison_fashion-mnist-784-euclidean_t1.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 5.7 | 805 | 446 | 4,690 | 4,690 | 2,298 |
| annoy | annoy | 17.9 | 434 | 254 | 1,433 | 786 | 434 |
| exhaustive | faiss | 0.0 | 359 | 179 | 8,760 | 8,760 | 8,760 |
| exhaustive | **ann_search** | 0.0 | 404 | 225 | 5,228 | 5,228 | 5,228 |
| hnsw | **ann_search_sq8** | 8.0 | 584 | 55 | 55,765 | 36,508 | 22,682 |
| hnsw | faiss | 17.0 | 368 | 189 | 25,925 | 17,073 | 10,563 |
| hnsw | **ann_search** | 16.5 | 411 | 231 | 24,018 | 16,568 | 10,491 |
| hnsw | faiss_sq8 | 21.0 | 234 | 54 | 23,009 | 15,819 | n/a |
| hnsw | usearch | 55.0 | 376 | 196 | 7,017 | 5,053 | 3,257 |
| hnsw | hnswlib | 66.0 | 374 | 195 | 5,806 | 4,142 | 2,784 |
| ivf | faiss | 0.6 | 439 | 260 | 12,598 | 6,517 | 6,517 |
| ivf | **ann_search** | 1.9 | 764 | 226 | 8,820 | 8,820 | 4,782 |
| nndescent | pynndescent | 12.7 | 368 | 188 | 17,759 | 16,219 | 11,137 |
| nndescent | **ann_search** | 15.4 | 1,019 | 593 | 16,214 | 16,214 | 5,218 |

### SIFT, 10 threads

![](figures/comparison_sift-128-euclidean_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 7.8 | 2,426 | 1,135 | 12,387 | 7,050 | 3,371 |
| annoy | annoy | 18.1 | 1,739 | 1,214 | 8,115 | 4,449 | 1,288 |
| cagra | **ann_search_gpu** | 21.9 | 10,332 | 9,356 | 43,200 | 27,799 | n/a |
| exhaustive | **ann_search_gpu** | 0.1 | 1,465 | 488 | 4,300 | 4,300 | 4,300 |
| exhaustive | **ann_search** | 0.1 | 978 | 489 | 3,538 | 3,538 | 3,538 |
| exhaustive | faiss | 0.0 | 978 | 489 | 1,149 | 1,149 | 1,149 |
| hnsw | **ann_search_sq8** | 22.0 | 1,037 | 198 | 195,566 | 111,815 | n/a |
| hnsw | faiss_sq8 | 40.9 | 770 | 279 | 112,461 | 61,340 | n/a |
| hnsw | **ann_search** | 42.3 | 1,136 | 636 | 90,128 | 50,514 | 28,178 |
| hnsw | faiss | 45.5 | 1,142 | 648 | 85,603 | 47,081 | 24,936 |
| hnsw | hnswlib | 76.6 | 1,255 | 766 | 43,663 | 25,525 | 14,422 |
| hnsw | usearch | 96.7 | 1,229 | 725 | 34,510 | 19,328 | 10,491 |
| ivf | **ann_search_gpu** | 2.7 | 4,036 | 1,603 | 94,584 | 56,905 | 27,603 |
| ivf | **ann_search** | 1.8 | 1,486 | 494 | 13,965 | 7,130 | 3,640 |
| ivf | faiss | 8.9 | 1,171 | 683 | 13,741 | 7,028 | 3,549 |
| nndescent | **ann_search** | 26.0 | 2,863 | 1,284 | 42,290 | 23,209 | 7,141 |
| nndescent | pynndescent | 27.3 | 1,491 | 814 | 14,089 | 9,134 | 3,942 |

### GloVe, 10 threads

![](figures/comparison_glove-100-angular_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.70 | QPS @ 0.80 | QPS @ 0.90 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | annoy | 25.4 | 1,988 | 1,498 | 3,484 | 1,585 | 863 |
| annoy | **ann_search** | 7.9 | 2,435 | 1,225 | 2,687 | 1,428 | n/a |
| cagra | **ann_search_gpu** | 25.0 | 10,201 | 9,298 | 15,187 | n/a | n/a |
| exhaustive | **ann_search_gpu** | 0.1 | 1,360 | 457 | 4,623 | 4,623 | 4,623 |
| exhaustive | faiss | 0.2 | 913 | 451 | 3,825 | 3,825 | 3,825 |
| exhaustive | **ann_search** | 0.1 | 909 | 457 | 2,867 | 2,867 | 2,867 |
| hnsw | **ann_search_sq8** | 52.6 | 1,411 | 227 | 90,605 | 33,546 | 9,748 |
| hnsw | **ann_search** | 73.2 | 1,091 | 639 | 70,797 | 40,809 | 12,425 |
| hnsw | faiss | 68.3 | 1,101 | 638 | 66,734 | 20,435 | 5,200 |
| hnsw | hnswlib | 110.7 | 1,232 | 780 | 40,507 | 13,454 | 3,958 |
| hnsw | usearch | 153.8 | 1,243 | 792 | 26,534 | 8,612 | 2,513 |
| hnsw | faiss_sq8 | 122.2 | 913 | 303 | 23,133 | 12,935 | 3,500 |
| ivf | **ann_search_gpu** | 2.4 | 3,822 | 1,582 | 170,370 | 102,319 | 28,708 |
| ivf | faiss | 6.3 | 1,082 | 631 | 29,665 | 14,758 | 4,181 |
| ivf | **ann_search** | 1.9 | 1,384 | 477 | 27,742 | 12,732 | 3,793 |
| nndescent | **ann_search** | 37.3 | 3,464 | 1,792 | 24,617 | 12,805 | n/a |
| nndescent | pynndescent | 45.6 | 1,628 | 810 | 7,687 | 4,965 | 1,945 |

---

### Runtime info

*ann-search-rs 0.10.1 (commit v0.10.1-6-gac3c903), run on 2026-10-09 on Apple M1 Max. Runs span commits v0.10.1-6-gac3c903, v0.10.1-7-gbeea508.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
