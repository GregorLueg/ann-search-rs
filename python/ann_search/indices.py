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
