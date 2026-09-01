## Binarised indices benchmarks and parameter

Binarised indices push the compression to (roughly) bits. Three consequences:

1. The index footprint collapses.
2. Queries usually get faster, because bitwise operations are cheap on modern
   CPUs.
3. Without re-ranking the top candidates, recall drops hard. Less so for RaBitQ,
   and for TurboQuant it depends on the data.

The benchmarks below show both, with and without re-ranking. For the simple
binary versions use:

```bash
cargo run --example gridsearch_binary --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For RaBitQ:

```bash
cargo run --example gridsearch_rabitq --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For TurboQuantisation

```bash
cargo run --example gridsearch_tq --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

As with the other benchmarks: index build, query against a 10% subsample with
noise added, and full self-kNN generation, plus the in-memory index size. These
runs use `"correlated"`, `"lowrank"` and `"embedding"` at higher dimensionality
with fewer samples, since that is where binarisation belongs.

**On the distance-ratio column.** A binarised index reports an approximate
distance, not the distance. Every ratio here is recomputed in `f32` from the
original vectors against the neighbours the index returned, so it measures
retrieval quality alone and the re-ranked and non-re-ranked rows sit on the same
footing.

## Table of Contents

- [Binarisation](#binary-ivf-and-exhaustive)
- [RaBitQ](#rabitq-ivf-and-exhaustive)
- [TurboQuant](#turboquant-ivf-and-exhaustive)

### <u>Binary (IVF and exhaustive)</u>

Three binarisations are offered in this crate:

- **SimHash**: Projects vectors onto random hyperplanes and encodes the sign of
  each projection as a bit. The random planes are orthogonalised to improve
  coverage of the vector space. The training data is only used to fit a
  per-feature mean: the hyperplanes pass through the origin, so on data sitting
  far from it every bit would otherwise land on the same side of every plane.
- **PCA Hashing**: Fits PCA on the (centred) training data and takes the sign of
  each point's score on a principal component as a bit. Only the leading
  components that cumulatively explain 90% of the variance are kept, and that
  count is capped at a sixteenth of the bit budget. The retained block is then
  rotated by ITQ (Gong and Lazebnik, "Iterative Quantization: A Procrustean
  Approach to Learning Binary Codes", CVPR 2011), which spreads variance evenly
  across those bits: raw PCA loadings pile nearly all of it into the first few
  components, leaving the trailing sign bits decided by rounding noise while
  they still count for a full unit of Hamming distance.

  Every bit past the retained block is a random orthogonal hyperplane, and that
  padding is the normal case rather than an edge case. At 512 bits at most 32
  are PCA bits, whatever the dimensionality. The cap is deliberate: past the
  genuinely structured directions a random hyperplane beats a PCA one, because
  it preserves angular distance by construction and a low-variance loading does
  not.

  More expensive to build than SimHash. Whether the data-adapted bits actually
  buy recall depends on the spectrum, so read it off the tables below rather
  than assuming they do.
- **Sign-based**: Simply encodes the sign of each embedding dimension directly
  as a bit, meaning `n_bits` is fixed to the number of dimensions.
  Straightforward but only sensible for high-dimensional data; at low
  dimensionality the recall degrades dramatically. Codes live in one global
  frame, on the IVF index too, so Hamming distances compare across Voronoi
  cells and widening `nprobe` can only add candidates.

These indices can keep the original vectors in a `VecStore` on disk for
re-ranking. Recommended if you want the recall to stay usable. Their home ground
is very high-dimensional data where memory is the binding constraint.

**Tunable parameters *(general)*:**

- *n_bits*: How many bits to encode each vector into. More bits, better recall,
  bigger index. For `"pca"` it also sets how many principal components can be
  spent, since the retained count is capped at `n_bits / 16`.
- *binarisation_init*: Three options are provided in the crate. `"random"` for
  random planes that are subsequently orthogonalised, `"pca"` to identify axes
  of maximum variation, or `"sign"` to just use the sign of the respective
  embedding dimensions. In that last case
  `n_bits` is set automatically to `n_dim`. Sign-based only really makes sense
  if you have a lot of dimensions; otherwise the performance is not great (at
  all). Unrecognised strings print a warning and fall back to `"random"`, so
  watch the spelling: `"random_projections"`, `"pca_hashing"` and `"sign_based"`
  are the accepted long forms, and `"signed"` is not one of them.
- *reranking_factor*: Hamming distance picks the candidates, then the on-disk
  vectors are loaded and the candidates re-scored exactly. The factor is how
  many more than `k` get re-scored, so `10` means `10 * k` vectors. More
  candidates, better recall. Default `20`; the grid runs lower values to show
  what that costs.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

Self queries run with `reranking_factor = 10`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:correlated:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:correlated:768:50000 -->
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:768:50000 -->
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:embedding:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:embedding:768:50000 -->
</code></pre>
</details>

### <u>RaBitQ (IVF and exhaustive)</u>

[RaBitQ](https://arxiv.org/abs/2405.12497) binarises against a centroid and
keeps enough side information to reconstruct an unbiased distance estimate, so
it holds up without re-ranking where plain sign bits do not. Better the higher
the dimensionality. `ExhaustiveRaBitQ` trains its own `sqrt(n)` centroids;
`IVF-RaBitQ` reuses the IVF centroids directly. The price against a plain binary
index is query speed, since the approximate distance is more work than a popcount.

**Tunable parameters *(RaBitQ)*:**

- *reranking_factor*: As for the binary indices. The RaBitQ estimate picks the
  candidates, then the on-disk vectors are loaded and re-scored exactly. `10`
  means `10 * k` vectors get re-scored.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:correlated:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:correlated:768:50000 -->
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:768:50000 -->
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:embedding:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:embedding:768:50000 -->
</code></pre>
</details>

### <u>TurboQuant (IVF and exhaustive)</u>

[TurboQuant](https://arxiv.org/abs/2504.19874) is a scalar quantisation scheme.
It applies a fixed random orthogonal rotation to each unit-normalised vector,
which drives every coordinate towards the same Beta distribution, and then
quantises each rotated coordinate against a Lloyd-Max codebook that is optimal
for that distribution. Codes are stored in bit-plane format and scored with a
FAISS PQ4-style fast-scan lookup table, so distance estimation is SIMD-friendly
and fast.

Encoding is data-oblivious: a single shared rotation and a single shared
codebook are used for every vector, with no per-cluster residuals. For the
`ExhaustiveTurboQuant` index every query scans the whole set via the block-fused
SIMD kernel. For the `IVF-TurboQuant` index the clustering is routing only — the
same global encoding is reused and vectors are merely bucketed into cells, so
the IVF centroids do not feed the quantiser (unlike IVF-RaBitQ). As with the
other indices, the original vectors can be stored on disk for exact re-ranking.

**Tunable parameters *(TurboQuant)*:**

- *bits*: Bits per coordinate, 2, 3 or 4. More bits, better recall, more memory.
  3-bit has no SIMD kernel and falls back to the scalar scorer, which is
  markedly slower, so prefer 4-bit unless memory forces otherwise. The grid runs
  2-bit and 4-bit.
- *reranking_factor*: As for the other indices. Default `20`.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

Self queries run with `reranking_factor = 20`. The encoding is data-oblivious,
so this one was designed for high-dimensional neural-network output rather than
for strongly clustered data.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:correlated:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:correlated:768:50000 -->
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:lowrank:768:50000 -->
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:embedding:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:embedding:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:turboquant:euclidean:embedding:768:50000 -->
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
