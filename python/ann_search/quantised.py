"""Quantised estimators: the same indices, over compressed vectors.

These trade recall for memory. Storage drops from ``dim`` floats per vector to
somewhere between two bytes per dimension (BF16) and a handful of bytes per
vector (PQ), and in several cases the query gets faster too, because integer
kernels beat float ones.

Two things differ from the uncompressed estimators, both consequences of the
codec rather than choices:

- **Distances are estimates.** A quantised index reports the codec's estimate
  of a distance, not the distance. It is close enough to rank on, which is what
  an index is for, but don't feed it anywhere the absolute value matters
  without checking it against `ExhaustiveIndex` first.
- **Manhattan is unavailable** on all eleven. Every codec here is built on
  inner products, which is what makes the integer arithmetic work.

Everything else is identical: same four-method surface, same padding, same
persistence, float32 and float64 both supported.

Which one to reach for, roughly. `HnswSq8uIndex` if you want the usual first
choice at a quarter of the memory. `IvfSq8Index` if you were already on
`IvfIndex`. The PQ family when a quarter is not enough of a saving, which
generally means a high-dimensional embedding space rather than a 30-dimensional
PCA. See [the benchmark tables][1] for numbers.

[1]: https://github.com/GregorLueg/ann-search-rs/blob/main/docs/benchmarks_quantised.md
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from . import _ann_search as _core
from ._base import BaseAnnIndex
from ._metrics import NO_MANHATTAN


class _BaseQuantisedIndex(BaseAnnIndex):
    """Shared settings for every quantised estimator.

    Exists to state the metric constraint once. It carries no methods of its
    own, so it is a settings holder rather than a layer.
    """

    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN


########
# BF16 #
########


class ExhaustiveBf16Index(_BaseQuantisedIndex):
    """Brute force over `bf16` storage.

    ``bf16`` keeps float32's exponent range and throws away mantissa bits from
    roughly the third significant digit on, so it halves the memory without the
    overflow traps of float16. The scan is still exhaustive, so the only thing
    between this and `ExhaustiveIndex` is the codec's rounding. The cheapest
    quantisation to reason about, and the one with the least to go wrong.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        verbose: Progress to the process stdout, not ``sys.stdout``. In Jupyter
            that lands in the terminal running the kernel.
    """

    _HANDLE: ClassVar[type] = _core.ExhaustiveBf16

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.ExhaustiveBf16.build(
            x, metric=self._core_metric, verbose=self.verbose
        )


class IvfBf16Index(_BaseQuantisedIndex):
    """`IvfIndex` with the posting lists held at `bf16`.

    Half the vector memory for a codec error that lands well inside IVF's own
    approximation, so at a fixed `nprobe` the recall barely moves. If you are
    already using `IvfIndex` and memory is the binding constraint, this is the
    swap to make first.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time: override it per call as
            ``index.kneighbors(nprobe=32)``.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration, evening out
            the posting lists.
        seed: Fixes k-means initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.IvfBf16
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfBf16.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            seed=self.seed,
            verbose=self.verbose,
        )


#######
# SQ8 #
#######


class ExhaustiveSq8Index(_BaseQuantisedIndex):
    """Brute force over 8-bit codes.

    One byte per dimension, with per-dimension offsets and a single scale shared
    across all of them. The shared scale is the point: it makes the integer code
    distance preserve the ordering of the float one, so the whole scan runs on
    ``u8`` kernels and usually comes out faster than the float version as well
    as four times smaller.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        quant_drop_ratio: Fraction trimmed from *each* tail of every dimension
            before the range is fixed; values outside it clamp to the end codes.
            ``None`` uses the crate default. Raise it when a few outliers are
            stretching the range and wasting code levels.
        quant_sample_rows: Rows sampled to calibrate the range. ``None``
            auto-picks, capped at the dataset size.
        seed: Fixes the calibration row sample.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.ExhaustiveSq8

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        quant_drop_ratio: float | None = None,
        quant_sample_rows: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.quant_drop_ratio = quant_drop_ratio
        self.quant_sample_rows = quant_sample_rows
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.ExhaustiveSq8.build(
            x,
            metric=self._core_metric,
            quant_drop_ratio=self.quant_drop_ratio,
            quant_sample_rows=self.quant_sample_rows,
            seed=self.seed,
            verbose=self.verbose,
        )


class IvfSq8Index(_BaseQuantisedIndex):
    """`IvfIndex` with the posting lists held as 8-bit codes.

    A quarter of IVF's vector memory, and the integer kernels usually make the
    cell scan faster rather than slower. The natural default if you were already
    on `IvfIndex` and want the saving without thinking about subspaces.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration.
        quant_drop_ratio: Fraction trimmed from each tail of every dimension
            before the range is fixed. ``None`` uses the crate default.
        quant_sample_rows: Rows sampled to calibrate the range. ``None``
            auto-picks.
        seed: Fixes k-means initialisation and the calibration sample.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.IvfSq8
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        quant_drop_ratio: float | None = None,
        quant_sample_rows: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.quant_drop_ratio = quant_drop_ratio
        self.quant_sample_rows = quant_sample_rows
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfSq8.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            quant_drop_ratio=self.quant_drop_ratio,
            quant_sample_rows=self.quant_sample_rows,
            seed=self.seed,
            verbose=self.verbose,
        )


class HnswSq8uIndex(_BaseQuantisedIndex):
    """`HnswIndex` built *and* searched entirely on 8-bit codes.

    Inspired by pyglass. The graph is constructed in the space it is searched
    in, so there is no float copy hanging around for re-ranking and no mismatch
    between the edges and the distances that traverse them. Roughly a quarter of
    HNSW's vector memory, and the same recall knob.

    If you want one quantised index and no further reading, this is it.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        m: Edges per node on the upper layers; the base layer gets ``2 * m``.
            Raising it gives a better graph, a larger index and a slower build.
        ef_construction: Candidate list size during insertion. Larger means a
            better graph and a slower build, and it costs nothing at query time.
        ef_search: Beam width at query time, the recall knob. Raised to ``k``
            internally when it is smaller. Search-time: override it per call as
            ``index.kneighbors(ef_search=200)``.
        quant_drop_ratio: Fraction trimmed from each tail of every dimension
            before the range is fixed. ``None`` uses the crate default.
        quant_sample_rows: Rows sampled to calibrate the range. ``None``
            auto-picks.
        seed: Fixes the level assignment and the calibration sample.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.HnswSq8u
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        quant_drop_ratio: float | None = None,
        quant_sample_rows: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.quant_drop_ratio = quant_drop_ratio
        self.quant_sample_rows = quant_sample_rows
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.HnswSq8u.build(
            x,
            m=self.m,
            ef_construction=self.ef_construction,
            metric=self._core_metric,
            quant_drop_ratio=self.quant_drop_ratio,
            quant_sample_rows=self.quant_sample_rows,
            seed=self.seed,
            verbose=self.verbose,
        )


######################
# Product quantisers #
######################


class ExhaustivePqIndex(_BaseQuantisedIndex):
    """Brute force over product-quantised codes.

    Each vector is cut into `m` subvectors and each subvector is replaced by the
    id of its nearest sub-codebook centroid, so a vector costs `m` bytes rather
    than ``dim`` floats. A query builds one lookup table per subspace and then
    scores each point by summing `m` table reads.

    The compression is the whole point and it is aggressive: at ``dim=512`` and
    ``m=64`` that is 64 bytes against 2 KB. Expect the recall to reflect that.
    This is a method for high-dimensional embedding spaces, not for a
    30-dimensional PCA where `ExhaustiveSq8Index` will do better on both counts.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it, and it sets the code length in
            bytes.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        max_iters: Lloyd iterations when training the sub-codebooks. ``None``
            uses the crate default.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256, which is what
            makes a code fit in a byte. Cannot exceed the sample count, so a
            small dataset needs this lowered.
        seed: Fixes the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.ExhaustivePq

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        max_iters: int | None = None,
        n_pq_centroids: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.max_iters = max_iters
        self.n_pq_centroids = n_pq_centroids
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.ExhaustivePq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            max_iters=self.max_iters,
            n_pq_centroids=self.n_pq_centroids,
            seed=self.seed,
            verbose=self.verbose,
        )


class ExhaustiveOpqIndex(_BaseQuantisedIndex):
    """`ExhaustivePqIndex` with a learned rotation in front of it.

    Plain PQ splits on the original axis order, so a space where the variance is
    concentrated in a few coordinates gets subspaces of wildly unequal
    difficulty. OPQ learns an orthogonal rotation that spreads it before
    splitting. It costs more at build time and nothing at query time, since the
    rotation folds into the query once.

    Worth it on a raw embedding space, rarely worth it after a PCA has already
    done the rotating.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        max_iters: Alternating rotation/codebook iterations. ``None`` uses the
            crate default. This is the build-cost knob.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256.
        seed: Fixes the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.ExhaustiveOpq

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        max_iters: int | None = None,
        n_pq_centroids: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.max_iters = max_iters
        self.n_pq_centroids = n_pq_centroids
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.ExhaustiveOpq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            max_iters=self.max_iters,
            n_pq_centroids=self.n_pq_centroids,
            seed=self.seed,
            verbose=self.verbose,
        )


class IvfPqIndex(_BaseQuantisedIndex):
    """Inverted file plus product quantisation.

    The two compress different things and stack cleanly: the inverted file cuts
    how many vectors get scored, PQ cuts what each one costs. Codes are learned
    on the residual from the cell centroid rather than the vector itself, so the
    sub-codebooks only have to cover within-cell spread, which is why this
    reaches better recall than `ExhaustivePqIndex` at the same `m`.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256.
        seed: Fixes both the cell and the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.IvfPq
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        n_pq_centroids: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.n_pq_centroids = n_pq_centroids
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfPq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            n_pq_centroids=self.n_pq_centroids,
            seed=self.seed,
            verbose=self.verbose,
        )


class IvfOpqIndex(_BaseQuantisedIndex):
    """`IvfPqIndex` with the learned rotation in front of the sub-codebooks.

    Same trade as `ExhaustiveOpqIndex` against `ExhaustivePqIndex`: better codes
    for a slower build, free at query time.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256.
        opq_iters: Alternating rotation/codebook iterations. ``None`` uses the
            crate default. This is the build-cost knob.
        seed: Fixes both the cell and the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.IvfOpq
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        n_pq_centroids: int | None = None,
        opq_iters: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.n_pq_centroids = n_pq_centroids
        self.opq_iters = opq_iters
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfOpq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            n_pq_centroids=self.n_pq_centroids,
            opq_iters=self.opq_iters,
            seed=self.seed,
            verbose=self.verbose,
        )


class SoarPqIndex(_BaseQuantisedIndex):
    """`IvfPqIndex` with SOAR spilling.

    Every vector also lands in a second cell, chosen so its residual points
    somewhere the first one doesn't. That fixes IVF's main failure, a true
    neighbour sitting just over a cell boundary, at the cost of roughly twice
    the posting-list size, which is exactly what quantisation makes affordable.

    Read the trade against **query time**, not against `nprobe`: at a fixed
    `nprobe` a spilled index scans about twice the candidates, so comparing at
    equal `nprobe` flatters it and answers nothing.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time.
        rule: Spilling rule: ``"nearest"``, ``"shifted"`` or ``"orthogonal"``.
            ``None`` lets the crate pick per metric, orthogonal for cosine and
            shifted otherwise.
        rule_param: ``mu`` for ``"shifted"``, ``lambda`` for ``"orthogonal"``.
            ``None`` uses the crate's value. Ignored by ``"nearest"``.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256.
        seed: Fixes both the cell and the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.SoarPq
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        rule: str | None = None,
        rule_param: float | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        n_pq_centroids: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.rule = rule
        self.rule_param = rule_param
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.n_pq_centroids = n_pq_centroids
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.SoarPq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            nlist=self.nlist,
            rule=self.rule,
            rule_param=self.rule_param,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            n_pq_centroids=self.n_pq_centroids,
            seed=self.seed,
            verbose=self.verbose,
        )


class SoarOpqIndex(_BaseQuantisedIndex):
    """`SoarPqIndex` with the learned rotation in front of the sub-codebooks.

    The most compressed index in the package, and the slowest to build. Reach
    for it when memory is the hard constraint and the build is a one-off.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        m: Subspaces. ``dim`` must divide by it.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Voronoi cells to cut the space into. ``None`` defaults to
            ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time.
        rule: Spilling rule: ``"nearest"``, ``"shifted"`` or ``"orthogonal"``.
            ``None`` lets the crate pick per metric.
        rule_param: ``mu`` for ``"shifted"``, ``lambda`` for ``"orthogonal"``.
            ``None`` uses the crate's value.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 30.
        kmeans_balanced: Reseed starved centroids each iteration.
        n_pq_centroids: Centroids per subspace. ``None`` uses 256.
        opq_iters: Alternating rotation/codebook iterations. ``None`` uses the
            crate default. This is the build-cost knob.
        seed: Fixes both the cell and the codebook initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``.
    """

    _HANDLE: ClassVar[type] = _core.SoarOpq
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        m: int = 8,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        rule: str | None = None,
        rule_param: float | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        n_pq_centroids: int | None = None,
        opq_iters: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.rule = rule
        self.rule_param = rule_param
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.n_pq_centroids = n_pq_centroids
        self.opq_iters = opq_iters
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.SoarOpq.build(
            x,
            m=self.m,
            metric=self._core_metric,
            nlist=self.nlist,
            rule=self.rule,
            rule_param=self.rule_param,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            n_pq_centroids=self.n_pq_centroids,
            opq_iters=self.opq_iters,
            seed=self.seed,
            verbose=self.verbose,
        )
