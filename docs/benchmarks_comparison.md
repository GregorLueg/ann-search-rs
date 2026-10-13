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
| annoy | **ann_search** | 1.6 | 767 | 429 | 23,525 | 23,525 | 11,610 |
| annoy | annoy | 4.5 | 437 | 244 | 10,409 | 6,306 | 3,501 |
| cagra | **ann_search_mlx** | 4.6 | 762 | 454 | 179,972 | 179,972 | 156,966 |
| cagra | **ann_search_gpu** | 6.4 | 1,945 | 1,634 | 55,274 | 55,274 | 37,455 |
| exhaustive | **ann_search_mlx** | 0.0 | 585 | 428 | 66,472 | 66,472 | 66,472 |
| exhaustive | **ann_search** | 0.0 | 405 | 226 | 17,727 | 17,727 | 17,727 |
| exhaustive | **ann_search_gpu** | 0.0 | 578 | 1,324 | 13,467 | 13,467 | 13,467 |
| exhaustive | faiss | 0.0 | 359 | 180 | 12,197 | 12,197 | 12,197 |
| hnsw | **ann_search_sq8** | 1.2 | 587 | 60 | 329,891 | 228,853 | 147,520 |
| hnsw | faiss_sq8 | 3.7 | 235 | 56 | 157,171 | 111,803 | n/a |
| hnsw | faiss | 3.9 | 370 | 190 | 115,802 | 76,206 | 47,977 |
| hnsw | **ann_search** | 3.5 | 414 | 235 | 113,141 | 75,086 | 48,207 |
| hnsw | usearch | 7.4 | 371 | 191 | 50,137 | 37,434 | 24,833 |
| hnsw | hnswlib | 8.3 | 376 | 196 | 46,893 | 33,786 | 22,752 |
| ivf | **ann_search_mlx** | 0.4 | 1,424 | 229 | 384,284 | 384,284 | 239,831 |
| ivf | **ann_search_gpu** | 1.5 | 1,989 | 1,085 | 198,456 | 198,456 | 147,956 |
| ivf | **ann_search** | 1.2 | 761 | 223 | 36,332 | 36,332 | 18,557 |
| ivf | faiss | 0.5 | 433 | 253 | 34,807 | 18,042 | 18,042 |
| nndescent | **ann_search** | 2.6 | 1,116 | 480 | 64,701 | 64,701 | 50,263 |
| nndescent | pynndescent | 3.2 | 388 | 210 | 17,121 | 15,577 | 10,693 |

### Fashion-MNIST, 1 thread

The ann-benchmarks convention.

![](figures/comparison_fashion-mnist-784-euclidean_t1.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 5.8 | 749 | 322 | 4,516 | 4,516 | 2,213 |
| annoy | annoy | 18.1 | 424 | 244 | 1,422 | 790 | 435 |
| exhaustive | faiss | 0.0 | 359 | 180 | 8,776 | 8,776 | 8,776 |
| exhaustive | **ann_search** | 0.0 | 404 | 225 | 5,166 | 5,166 | 5,166 |
| hnsw | **ann_search_sq8** | 8.7 | 583 | 53 | 50,327 | 32,368 | 19,967 |
| hnsw | faiss | 16.7 | 369 | 190 | 27,003 | 17,762 | 10,863 |
| hnsw | **ann_search** | 17.3 | 411 | 232 | 24,124 | 16,297 | 10,355 |
| hnsw | faiss_sq8 | 20.7 | 234 | 54 | 23,663 | 16,050 | n/a |
| hnsw | usearch | 53.3 | 375 | 196 | 7,290 | 5,155 | 3,432 |
| hnsw | hnswlib | 66.9 | 380 | 201 | 5,731 | 4,102 | 2,754 |
| ivf | faiss | 0.6 | 441 | 262 | 12,667 | 6,555 | 6,555 |
| ivf | **ann_search** | 1.9 | 770 | 232 | 8,608 | 8,608 | 4,637 |
| nndescent | pynndescent | 13.1 | 359 | 180 | 18,723 | 16,985 | 11,370 |
| nndescent | **ann_search** | 13.2 | 949 | 466 | 13,593 | 13,593 | 9,648 |

### SIFT, 10 threads

![](figures/comparison_sift-128-euclidean_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 8.0 | 2,422 | 1,186 | 11,885 | 6,720 | 3,180 |
| annoy | annoy | 18.3 | 1,760 | 1,244 | 8,100 | 4,657 | 1,301 |
| cagra | **ann_search_mlx** | 12.6 | 4,131 | 1,403 | 71,048 | 43,839 | 13,612 |
| cagra | **ann_search_gpu** | 17.9 | 5,175 | 4,682 | 66,072 | 30,851 | 3,890 |
| exhaustive | **ann_search_mlx** | 0.1 | 1,470 | 1,364 | 12,899 | 12,899 | 12,899 |
| exhaustive | **ann_search_gpu** | 0.1 | 1,466 | 1,597 | 4,952 | 4,952 | 4,952 |
| exhaustive | **ann_search** | 0.1 | 978 | 490 | 3,616 | 3,616 | 3,616 |
| exhaustive | faiss | 0.0 | 975 | 487 | 910 | 910 | 910 |
| hnsw | **ann_search_sq8** | 22.2 | 1,037 | 203 | 193,420 | 111,593 | n/a |
| hnsw | faiss_sq8 | 40.7 | 770 | 278 | 111,622 | 60,902 | n/a |
| hnsw | faiss | 45.9 | 1,143 | 648 | 85,752 | 46,523 | 24,439 |
| hnsw | **ann_search** | 43.4 | 1,131 | 637 | 85,715 | 49,063 | 27,246 |
| hnsw | hnswlib | 78.5 | 1,256 | 768 | 40,809 | 23,737 | 13,278 |
| hnsw | usearch | 100.1 | 1,229 | 725 | 32,992 | 19,129 | 10,530 |
| ivf | **ann_search_mlx** | 0.8 | 5,283 | 1,027 | 240,747 | 130,685 | 69,135 |
| ivf | **ann_search_gpu** | 2.6 | 4,037 | 1,113 | 101,469 | 59,617 | 29,041 |
| ivf | faiss | 9.3 | 1,187 | 698 | 14,010 | 7,132 | 3,549 |
| ivf | **ann_search** | 1.8 | 1,486 | 494 | 13,815 | 7,079 | 3,610 |
| nndescent | **ann_search** | 26.1 | 3,919 | 1,824 | 67,591 | 37,438 | 20,390 |
| nndescent | pynndescent | 27.5 | 1,482 | 812 | 14,095 | 9,370 | 3,981 |

### GloVe, 10 threads

![](figures/comparison_glove-100-angular_t10.png)

| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | QPS @ 0.70 | QPS @ 0.80 | QPS @ 0.90 |
|---|---|---:|---:|---:|---:|---:|---:|
| annoy | **ann_search** | 11.6 | 2,388 | 1,182 | 6,844 | 3,065 | 1,648 |
| annoy | annoy | 26.8 | 2,174 | 1,652 | 3,415 | 1,501 | 851 |
| cagra | **ann_search_mlx** | 17.7 | 4,485 | 1,570 | 60,705 | 34,136 | n/a |
| cagra | **ann_search_gpu** | 22.3 | 8,470 | 7,703 | 28,725 | 9,880 | n/a |
| exhaustive | **ann_search_mlx** | 0.1 | 1,356 | 1,356 | 7,254 | 7,254 | 7,254 |
| exhaustive | **ann_search_gpu** | 0.1 | 1,360 | 1,565 | 5,278 | 5,278 | 5,278 |
| exhaustive | faiss | 0.2 | 913 | 453 | 3,831 | 3,831 | 3,831 |
| exhaustive | **ann_search** | 0.1 | 908 | 457 | 2,873 | 2,873 | 2,873 |
| hnsw | **ann_search_sq8** | 51.8 | 1,410 | 228 | 110,103 | 35,122 | 9,897 |
| hnsw | **ann_search** | 72.7 | 1,090 | 640 | 70,483 | 40,947 | 12,470 |
| hnsw | faiss | 66.3 | 1,104 | 644 | 68,336 | 20,873 | 5,339 |
| hnsw | hnswlib | 109.5 | 1,231 | 780 | 39,660 | 13,584 | 4,015 |
| hnsw | usearch | 152.2 | 1,244 | 793 | 26,702 | 8,998 | 2,550 |
| hnsw | faiss_sq8 | 117.2 | 913 | 305 | 23,341 | 13,059 | 3,548 |
| ivf | **ann_search_mlx** | 0.9 | 4,727 | 957 | 415,076 | 247,529 | 73,210 |
| ivf | **ann_search_gpu** | 2.4 | 3,829 | 1,122 | 179,104 | 103,671 | 29,010 |
| ivf | faiss | 6.3 | 1,084 | 632 | 29,743 | 15,645 | 4,173 |
| ivf | **ann_search** | 1.7 | 1,386 | 476 | 28,553 | 14,719 | 3,897 |
| nndescent | **ann_search** | 39.0 | 4,196 | 1,888 | 65,077 | 36,697 | 10,336 |
| nndescent | pynndescent | 47.3 | 1,620 | 810 | 7,542 | 4,874 | 1,911 |

---

### Runtime info

*ann-search-rs 0.11.0 (commit v0.10.1-76-g288b4d6), run on 2026-10-11 on Apple M1 Max. Runs span commits py-v0.5.0-dirty, v0.10.1-76-g288b4d6.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
