# Quantised indices

Eleven more estimators, all of them the same indices you already have over
compressed vectors. They exist for one reason: memory. A float32 index stores
`dim` floats per vector, and on a few million cells at 512 dimensions that
stops being a rounding error.

Nothing about the surface changes. Same `fit` / `kneighbors` / `save`, same
padding rules, float32 and float64 both supported.

## The one thing to internalise

**A quantised index reports the codec's estimate of a distance, not the
distance.** It's close enough to rank on, which is what an index is for. It is
not something to feed anywhere the absolute value matters without checking it
against `ExhaustiveIndex` first.

This also means recall is the only honest way to compare two of them. A
distance-ratio comparison between a quantised index and an exact one conflates
the retrieval error with the codec's, and the crate's own benchmark tables
recompute every ratio from the original vectors for exactly that reason.

The other constraint: **none of them support Manhattan**. Every codec here is
built on inner products, which is what makes the integer arithmetic work.
Asking for it raises rather than falling back.

## What each codec does

### BF16

`ExhaustiveBf16Index`, `IvfBf16Index`.

Storage drops to `bf16`, which keeps float32's exponent range and throws away
mantissa bits from roughly the third significant digit on. Half the memory, no
overflow traps, and nothing structural in the index changes. The cheapest
quantisation to reason about and the one with the least to go wrong, which also
makes it the smallest saving.

Two things do change. Distances are computed in float32, so the values get
widened back on every comparison and the query comes out around 2x slower at 32
dimensions and 4x slower at 128. And the codec error is not negligible: recall
lands near 0.98 on Euclidean and 0.89 on cosine. Under IVF it does not hide
inside the cell pruning either, since plain IVF already reaches 1.0 on the
correlated and low-rank generators, so the codec error is the whole error.

### SQ8

`ExhaustiveSq8Index`, `IvfSq8Index`, `HnswSq8uIndex`.

One byte per dimension, with per-dimension offsets and a single scale shared
across all of them. The shared scale is the whole trick: it makes the integer
code distance preserve the ordering of the float one, so the scan runs entirely
on `u8` kernels.

The memory saving is about 3.5x at 32 dimensions and 3.9x at 128, so "a
quarter" is the high-dimensional case rather than the general one. The speed
depends on which index you put it under. Exhaustive SQ8 is *slower* than the
float scan at 32 dimensions, by up to 1.5x, and only edges ahead at 128. Under
IVF it wins consistently, 1.15 to 1.5x on every matched `nlist`/`nprobe`
pairing.

The recall cost is the thing to measure. At a fixed `nprobe` `IvfSq8Index`
gives up around 0.07 against plain IVF on gaussian Euclidean data, 0.21 on the
low-rank generator and 0.26 on cosine.

`HnswSq8uIndex` is the interesting one. The graph is built and searched
entirely on codes, so there's no float copy sitting around for re-ranking and
no mismatch between the edges and the distances that traverse them. Build and
query are both faster than a plain `HnswIndex`.

It is the usual starting point, not a free win. The graph edges aren't
compressed, so the whole index lands at 0.44 to 0.80 of a plain HNSW depending
on `m` and dimensionality rather than at a quarter, and at matched `m`,
`ef_construction` and `ef_search` it gives up 0.07 recall on Euclidean and 0.26
to 0.33 on cosine. Its recall tracks `ExhaustiveSq8Index` almost exactly, which
tells you the loss is the codec rather than the graph. Measure it against
`HnswIndex` before taking the memory.

Two knobs worth knowing. `quant_drop_ratio` trims a fraction from each tail of
every dimension before the range is fixed, so a handful of outliers can't
stretch the range and waste code levels. `quant_sample_rows` caps how many rows
the calibration looks at.

### PQ and OPQ

`ExhaustivePqIndex`, `IvfPqIndex`, `ExhaustiveOpqIndex`, `IvfOpqIndex`,
`SoarPqIndex`, `SoarOpqIndex`.

Each vector is cut into `m` subvectors, and each subvector is replaced by the
id of its nearest sub-codebook centroid. A vector costs `m` bytes. At `dim=512`
and `m=64` that's 64 bytes against 2 KB, and the recall reflects it.

This is a method for high-dimensional embedding spaces: the whole point is
exploiting correlation between coordinates, and a PCA output has already spent
most of that. The benchmark tables don't settle where the crossover with
`ExhaustiveSq8Index` sits, since the PQ family is only run at 256 dimensions
and up and SQ8 only at 32 and 128. Measure it if you are near the boundary.

`m` has to divide `dim`, and `dim` has to be at least 32. `n_pq_centroids`
defaults to 256, which is what makes a code fit in a byte; it cannot exceed 256
and cannot exceed your sample count, so a small dataset needs it lowered.

**OPQ** adds a learned orthogonal rotation before the split. Plain PQ splits on
whatever axis order the data arrived in, so a space with variance concentrated
in a few coordinates gets subspaces of wildly unequal difficulty. The rotation
spreads it, and it buys recall everywhere it was measured: +0.01 to +0.19 over
plain PQ depending on the generator.

It isn't free at either end. Build costs 3 to 5x (`opq_iters` is the knob), and
the rotation has to be applied to every query, which runs from 1.1x the query
time at 256 dimensions to 3.1x at 768. It also stores a `dim x dim` matrix, so
the index is larger than the plain PQ equivalent.

**The IVF variants** learn the codes on the residual from the cell centroid
rather than on the vector, so the sub-codebooks only have to cover within-cell
spread. That's why `IvfPqIndex` reaches better recall than `ExhaustivePqIndex`
at the same `m`, on top of scanning fewer candidates. It holds on every
generator in the tables, and costs about 2.2x the build time.

**The SOAR variants** spill every vector into a second cell chosen so its
residual points somewhere the first one doesn't, fixing IVF's main failure of a
true neighbour sitting just over a cell boundary. Twice the posting-list
entries, which is exactly what quantisation makes affordable. Read that trade
against query time, not against `nprobe`: at a fixed `nprobe` a spilled index
scans about twice the candidates, so comparing at equal `nprobe` flatters it
and answers nothing.

Unlike the unquantised `SoarIndex`, here the doubling really does show up in
the index size, because the codes *are* the storage: `SoarPqIndex` runs about
2x an `IvfPqIndex` at the same `m`. `SoarOpqIndex` is the largest index of the
eleven, not the smallest, since it pays for the spilling and the rotation
matrix both. An `IvfOpqIndex` at twice the `m` is smaller and quicker to build
at matching memory.

## Picking one

```python
import ann_search as ann

index = ann.HnswSq8uIndex(n_neighbors=15, metric="cosine").fit(X)
distances, indices = index.kneighbors()
```

Then measure. The recipe in [Quickstart](quickstart.md) works unchanged: build
an `ExhaustiveIndex` for ground truth, and compare recall rather than
distances.

The crate's [benchmark tables][1] sweep every codec over the synthetic
generators with build time, query time, recall and index size. The PQ-family
runs are at higher dimensionality with fewer samples, since that's the regime
those methods are for. Don't extrapolate parameters for your own problem from
them.

[1]: https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_quantised.md
