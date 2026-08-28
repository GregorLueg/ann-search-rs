"""The estimators.

Each class stores its constructor arguments verbatim, names the compiled handle
it drives, and says which of its parameters are search-time knobs. Everything
else comes from `BaseAnnIndex`.

Build defaults are lifted from the crate's own gridsearch examples and parameter
defaults rather than invented; see ``examples/gridsearch_*.rs`` and
``docs/benchmarks_standard.md`` in the repository.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from . import _ann_search as _core
from ._base import BaseAnnIndex
from ._metrics import NO_MANHATTAN

###########
# Globals #
###########

#: SOAR spilling rules. Validated here for the same reason metric names are: an
#: unrecognised value would otherwise reach the core, which only complains via
#: `println!` to the process stdout.
SOAR_RULES: frozenset[str] = frozenset({"nearest", "orthogonal", "shifted"})


class ExhaustiveIndex(BaseAnnIndex):
    """Brute-force exact search.

    Every query is scored against every point, so recall is 1 by construction.
    Useful as ground truth and, with SIMD behind it, competitive up to a few
    tens of thousands of points.
    """

    _HANDLE: ClassVar[type] = _core.Exhaustive

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
        return _core.Exhaustive.build(x, metric=self._core_metric)


class KmknnIndex(BaseAnnIndex):
    """k-means-based k-nearest-neighbours.

    Partitions the data with k-means and prunes clusters by the triangle
    inequality. Exact, and usually well ahead of brute force once the data
    clusters at all. Manhattan is not supported.
    """

    _HANDLE: ClassVar[type] = _core.Kmknn
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        nlist: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Kmknn.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            seed=self.seed,
            verbose=self.verbose,
        )


class AnnoyIndex(BaseAnnIndex):
    """Random projection forest, as in Spotify's Annoy.

    More trees means better recall and a larger index. Manhattan is not
    supported.
    """

    _HANDLE: ClassVar[type] = _core.Annoy
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("search_budget",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        n_trees: int = 25,
        search_budget: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.search_budget = search_budget
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Annoy.build(
            x, metric=self._core_metric, n_trees=self.n_trees, seed=self.seed
        )


class HnswIndex(BaseAnnIndex):
    """Hierarchical navigable small world graph.

    The usual first choice: high recall at low query latency, at the cost of a
    slower build and a graph roughly ``m`` edges per node wide. Raise
    `ef_search` for recall, `ef_construction` for a better graph.
    """

    _HANDLE: ClassVar[type] = _core.Hnsw
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Hnsw.build(
            x,
            m=self.m,
            ef_construction=self.ef_construction,
            metric=self._core_metric,
            seed=self.seed,
            verbose=self.verbose,
        )


class IvfIndex(BaseAnnIndex):
    """Inverted file index over k-means Voronoi cells.

    Cheap to build and easy to tune: `nlist` sets how finely the space is cut,
    `nprobe` how many cells a query visits. Both default to the crate's own
    heuristics when left as ``None``.
    """

    _HANDLE: ClassVar[type] = _core.Ivf
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
        return _core.Ivf.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            seed=self.seed,
            verbose=self.verbose,
        )


class NNDescentIndex(BaseAnnIndex):
    """NN-Descent kNN graph.

    Builds the neighbour graph directly by iterative local join, which makes it
    the fastest route to a full self-kNN graph. `n_neighbors` is used at build
    time as well as at query time, so changing it means rebuilding.

    `diversify_prob` prunes redundant edges after descent: ``0.0`` disables it,
    ``1.0`` prunes whenever the rule fires.
    """

    _HANDLE: ClassVar[type] = _core.NnDescent
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        delta: float = 0.001,
        diversify_prob: float = 0.0,
        max_iter: int | None = None,
        max_candidates: int | None = None,
        n_trees: int | None = None,
        ef_search: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.delta = delta
        self.diversify_prob = diversify_prob
        self.max_iter = max_iter
        self.max_candidates = max_candidates
        self.n_trees = n_trees
        self.ef_search = ef_search
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.NnDescent.build(
            x,
            metric=self._core_metric,
            delta=self.delta,
            diversify_prob=self.diversify_prob,
            k=self.n_neighbors,
            max_iter=self.max_iter,
            max_candidates=self.max_candidates,
            n_trees=self.n_trees,
            seed=self.seed,
            verbose=self.verbose,
        )


class VamanaIndex(BaseAnnIndex):
    """Vamana graph, as in DiskANN.

    A single flat graph of out-degree `r`, pruned in two passes with the
    relaxed-neighbour rule. Builds faster than HNSW at comparable recall.
    """

    _HANDLE: ClassVar[type] = _core.Vamana
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        r: int = 48,
        l_build: int = 100,
        alpha_pass1: float = 1.0,
        alpha_pass2: float = 1.2,
        ef_search: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.r = r
        self.l_build = l_build
        self.alpha_pass1 = alpha_pass1
        self.alpha_pass2 = alpha_pass2
        self.ef_search = ef_search
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Vamana.build(
            x,
            metric=self._core_metric,
            r=self.r,
            l_build=self.l_build,
            alpha_pass1=self.alpha_pass1,
            alpha_pass2=self.alpha_pass2,
            seed=self.seed,
        )


class NsgIndex(BaseAnnIndex):
    """Navigating spreading-out graph.

    Refines a kNN graph into a sparse monotonic one. `knn_k` sizes the
    NN-Descent graph it builds internally first, so it is a build cost rather
    than a query knob.
    """

    _HANDLE: ClassVar[type] = _core.Nsg
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        r: int = 32,
        l_build: int = 100,
        c: int = 500,
        knn_k: int = 64,
        ef_search: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.r = r
        self.l_build = l_build
        self.c = c
        self.knn_k = knn_k
        self.ef_search = ef_search
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Nsg.build(
            x,
            metric=self._core_metric,
            r=self.r,
            l_build=self.l_build,
            c=self.c,
            knn_k=self.knn_k,
            seed=self.seed,
            verbose=self.verbose,
        )


class BallTreeIndex(BaseAnnIndex):
    """Metric tree of nested hyperspheres.

    Prunes by the triangle inequality, which pays off when the data has genuine
    cluster structure and the dimensionality is moderate. `search_budget`
    defaults to 5% of the indexed points, so the defaults are approximate;
    raise it for recall. Manhattan is not supported.
    """

    _HANDLE: ClassVar[type] = _core.BallTree
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("search_budget",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        search_budget: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.search_budget = search_budget
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.BallTree.build(x, metric=self._core_metric, seed=self.seed)


class KdTreeIndex(BaseAnnIndex):
    """Forest of randomised kd spill-trees.

    Same trade as Annoy, with axis-aligned splits instead of random
    hyperplanes: more trees means better recall and a larger index. The one
    tree index here that supports Manhattan.
    """

    _HANDLE: ClassVar[type] = _core.KdTree
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("search_budget",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        n_trees: int = 25,
        search_budget: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.search_budget = search_budget
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.KdTree.build(
            x, metric=self._core_metric, n_trees=self.n_trees, seed=self.seed
        )


class LshIndex(BaseAnnIndex):
    """Multi-probe locality-sensitive hashing over random projections.

    The cheapest index to build here, and the weakest on recall. Lower
    `bits_per_hash` widens the buckets, trading query time for recall;
    `num_tables` trades index size for recall. `n_probe` defaults to one probe
    per projection. Manhattan is not supported.
    """

    _HANDLE: ClassVar[type] = _core.Lsh
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("n_probe", "max_candidates")

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        num_tables: int = 8,
        bits_per_hash: int = 12,
        slot_bits: int | None = None,
        n_probe: int | None = None,
        max_candidates: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.num_tables = num_tables
        self.bits_per_hash = bits_per_hash
        self.slot_bits = slot_bits
        self.n_probe = n_probe
        self.max_candidates = max_candidates
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.Lsh.build(
            x,
            metric=self._core_metric,
            num_tables=self.num_tables,
            bits_per_hash=self.bits_per_hash,
            slot_bits=self.slot_bits,
            seed=self.seed,
        )


class SoarIndex(BaseAnnIndex):
    """IVF with spilling, as in Google's SOAR.

    Every point also lands in a second cell, picked by a rule that accounts for
    the residual it already carries in its primary cell. That buys recall at a
    given `nprobe` over plain IVF, for roughly twice the posting-list size.

    `rule` is one of ``"nearest"``, ``"shifted"`` or ``"orthogonal"``, and
    ``None`` lets the core choose per metric: orthogonal for cosine, shifted
    otherwise. `rule_param` is ``mu`` for shifted and ``lambda`` for
    orthogonal, and ``None`` takes the core's own value. Manhattan is not
    supported.
    """

    _HANDLE: ClassVar[type] = _core.Soar
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe",)

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        rule: str | None = None,
        rule_param: float | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.rule = rule
        self.rule_param = rule_param
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        if self.rule is not None and self.rule not in SOAR_RULES:
            allowed = ", ".join(sorted(SOAR_RULES))
            raise ValueError(
                f"unknown SOAR rule {self.rule!r}; expected one of: {allowed}"
            )
        return _core.Soar.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            rule=self.rule,
            rule_param=self.rule_param,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            seed=self.seed,
            verbose=self.verbose,
        )


class RnnDescentIndex(BaseAnnIndex):
    """Relative NN-Descent graph.

    Builds and prunes in one pass, reaching a sparse navigable graph without
    the separate NSG-style refinement step, so it is the cheapest route to a
    graph index. `r` caps the out-degree and is the main size knob; `ef_search`
    is the main recall knob.
    """

    _HANDLE: ClassVar[type] = _core.RnnDescent
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("ef_search", "k_search")

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        s: int = 20,
        r: int = 96,
        t1: int = 4,
        t2: int = 15,
        n_trees: int | None = None,
        ef_search: int | None = None,
        k_search: int | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.s = s
        self.r = r
        self.t1 = t1
        self.t2 = t2
        self.n_trees = n_trees
        self.ef_search = ef_search
        self.k_search = k_search
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.RnnDescent.build(
            x,
            metric=self._core_metric,
            s=self.s,
            r=self.r,
            t1=self.t1,
            t2=self.t2,
            n_trees=self.n_trees,
            seed=self.seed,
            verbose=self.verbose,
        )
