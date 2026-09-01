## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown for the GPU-accelerated code. If you wish to run
the version with GPU-accelerated exhaustive and IVF script, please use:

```bash
cargo run --example gridsearch_gpu --features gpu --release
```

For the CAGRA style search, use:

```bash
cargo run --example gridsearch_cagra --features gpu --release
```

If you wish to run the Navigating Spread-out Graph (NSG) version where the
initial kNN is generated on the GPU, you can test this via:

```bash
cargo run --example gridsearch_nsg_gpu --features gpu --release
```

As with the other benchmarks: index build, query against a 10% subsample with
noise added, and full self-kNN generation, plus the in-memory index size (GPU
memory is not reported). Everything here runs on the wgpu backend; other
backends such as CUDA may do better still.

Looking for the self-kNN-graph paths (NN-Descent extract, self-beam, clustered)?
Those live in [the kNN-graph benchmarks](benchmarks_knn_graph.md).

## Table of Contents

- [GPU exhaustive and IVF](#gpu-accelerated-exhaustive-and-ivf-vs-cpu-exhaustive)
- [Comparison on larger data sets against the CPU](#comparison-against-ivf-cpu)
- [CAGRA style index](#cagra-type-querying)
- [CAGRA index on larger data](#larger-data-sets)
- [NSG with GPU-accelerated kNN generation](#navigating-spread-out-graph-nsg-with-gpu-accelerated-knn-generation)

### GPU-accelerated exhaustive and IVF vs CPU exhaustive

<details>
<summary><b>GPU - Euclidean (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU - Cosine (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:cosine:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (Correlated)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:correlated:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:lowrank:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:cell:128 -->
</code></pre>
</details>

### Comparison against IVF CPU

The CPU IVF implementation against the GPU one. The GPU pays a fixed setup
cost, so the sample count is raised to 250k and the dimensionality to 64 or 128
for these runs.

#### With 250k samples and 64 dimensions

<details>
<summary><b>CPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:ivf:euclidean:cell:64:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:cell:64:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:ivf:euclidean:cell:128:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:cell:128:250000 -->
</code></pre>
</details>

#### Increasing the number of samples

<details>
<summary><b>CPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:ivf:euclidean:cell:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:cell:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>CPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:ivf:euclidean:cell:128:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU-IVF (500k samples, 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:gpu:euclidean:cell:128:500000 -->
</code></pre>
</details>

### CAGRA-type querying

A [CAGRA-style index](https://arxiv.org/abs/2308.15136): NN-Descent runs
entirely on the GPU (random init, then a random-partition forest, then local
joins until convergence), and the resulting graph is pruned to degree `k` by
rank-based detour counting into a directed navigational graph. Queries are a GPU
beam search, one workgroup per query, with the query vector in shared memory and
a linear-probing hash table for visited-node deduplication.

**Tunable parameters:**

* `build_k`: Internal NNDescent degree before CAGRA pruning. Defaults to
  `1.5 * k`. Higher values give CAGRA more edges to choose from when building
  the navigational graph, at the cost of build time.
* `refine_knn`: Number of 2-hop refinement sweeps after NNDescent convergence.
  Each sweep evaluates all neighbours-of-neighbours and merges improvements.
  Defaults to 0. Mostly a lever on the extracted graph rather than on beam
  search recall.
* `n_trees`: Number of random partition trees for forest initialisation.
  Defaults to 5 + n^0.25, capped at 20. More trees raise the raw graph quality
  ceiling but increase build time linearly.
* `beam_width`: Number of active candidates maintained during beam search.
  Defaults to 2 * max(k_out, 16). Wider beams improve recall at the cost of
  query latency. Auto-scaled when using CagraGpuSearchParams::from_k().
* `max_beam_iters`: Safety cap on beam search iterations. Defaults to
  3 * beam_width. Most queries terminate naturally well before this limit; it
  only fires for pathological cases where the search keeps discovering better
  candidates.
* `n_entry_points`: Number of seed nodes per query for beam search. Defaults
  to 8. Entry points are sourced from a small Annoy forest (external queries)
  or from the kNN graph's closest neighbours (self-query).

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Cosine (Gaussian)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:cosine:gaussian:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (Correlated)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:correlated:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:lowrank:32 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning - Euclidean (LowRank; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:cell:128 -->
</code></pre>
</details>

#### Larger data sets

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:cell:64:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (250k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:cell:128:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 64 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:cell:64:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>GPU NNDescent with CAGRA style pruning (500k samples; 128 dimensions)</b>:</summary>
<pre><code>
<!-- BENCH:cagra:euclidean:cell:128:500000 -->
</code></pre>
</details>

### Navigating Spread-out Graph (NSG) with GPU-accelerated kNN generation

NSG builds on top of an existing kNN graph. The CPU path uses NN-Descent; the
GPU path swaps in the same NN-Descent that feeds CAGRA. The two columns below
differ only in that initialisation step.

<details>
<summary><b>NSG with CPU initialisation</b>:</summary>
<pre><code>
<!-- BENCH:nsg_cpu:euclidean:cell:128:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>NSG with GPU initialisation</b>:</summary>
<pre><code>
<!-- BENCH:nsg_gpu:euclidean:cell:128:250000 -->
</code></pre>
</details>

---

<details>
<summary><b>NSG with CPU initialisation (more samples)</b>:</summary>
<pre><code>
<!-- BENCH:nsg_cpu:euclidean:cell:128:500000 -->
</code></pre>
</details>

---

<details>
<summary><b>NSG with GPU initialisation (more samples)</b>:</summary>
<pre><code>
<!-- BENCH:nsg_gpu:euclidean:cell:128:500000 -->
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*The GPU backend was the `wgpu` backend.*
