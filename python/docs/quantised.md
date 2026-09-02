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
overflow traps, and nothing else in the index changes. The cheapest
quantisation to reason about and the one with the least to go wrong, which also
makes it the smallest saving.

### SQ8

`ExhaustiveSq8Index`, `IvfSq8Index`, `HnswSq8uIndex`.

One byte per dimension, with per-dimension offsets and a single scale shared
across all of them. The shared scale is the whole trick: it makes the integer
code distance preserve the ordering of the float one, so the scan runs entirely
on `u8` kernels. A quarter of the memory, and the query usually comes out
*faster* than the float version rather than slower.

`HnswSq8uIndex` is the interesting one. The graph is built and searched
entirely on codes, so there's no float copy sitting around for re-ranking and
no mismatch between the edges and the distances that traverse them. If you want
one quantised index and no further reading, take this one.

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

This is a method for high-dimensional embedding spaces. On a 30-dimensional PCA
`ExhaustiveSq8Index` will beat it on both memory and recall, because there is
nothing left to exploit once the dimensions are already decorrelated and few.

`m` has to divide `dim`. `n_pq_centroids` defaults to 256, which is what makes
a code fit in a byte, and cannot exceed your sample count, so a small dataset
needs it lowered.

**OPQ** adds a learned orthogonal rotation before the split. Plain PQ splits on
whatever axis order the data arrived in, so a space with variance concentrated
in a few coordinates gets subspaces of wildly unequal difficulty. The rotation
spreads it. Costs more at build time (`opq_iters` is the knob) and nothing at
query time, since the rotation folds into the query once. Worth it on a raw
embedding space, rarely worth it after a PCA has already done the rotating.

**The IVF variants** learn the codes on the residual from the cell centroid
rather than on the vector, so the sub-codebooks only have to cover within-cell
spread. That's why `IvfPqIndex` reaches better recall than `ExhaustivePqIndex`
at the same `m`, on top of scanning fewer candidates.

**The SOAR variants** spill every vector into a second cell chosen so its
residual points somewhere the first one doesn't, fixing IVF's main failure of a
true neighbour sitting just over a cell boundary. Roughly twice the
posting-list size, which is exactly what quantisation makes affordable. Read
that trade against query time, not against `nprobe`: at a fixed `nprobe` a
spilled index scans about twice the candidates, so comparing at equal `nprobe`
flatters it and answers nothing.

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
