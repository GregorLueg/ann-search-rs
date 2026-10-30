"""MLX estimators, present only in a source build with the ``mlx`` feature.

Experimental and Apple Silicon only. MLX drives Metal directly through mlx-c
rather than going through wgpu, and comes out faster than the `ann_search.gpu`
counterparts at the same recall; the GPU page of the docs has the numbers. The
published wheel does not carry them, since it would have to bundle mlx-c.
Build from source instead::

    brew install mlx-c
    maturin develop --release --features mlx

`ann_search.mlx_available()` says whether the build picked it up.

Same constraints as the GPU estimators: float32 only (``fit`` casts), no
persistence, no Manhattan. One more on top: MLX streams are tied to the thread
that created them, so a fitted index raises if used from any other Python
thread. Host-side work inside the MLX paths uses rayon's global pool and does
not follow `ann_search.set_num_threads`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from beartype import beartype

from . import _ann_search as _core
from ._base import BaseAnnIndex, ExtractKnnMixin
from ._metrics import NO_MANHATTAN

if TYPE_CHECKING:
    from ._ann_search import _GpuHandle

if not hasattr(_core, "ExhaustiveMlx"):  # pragma: no cover - build-dependent
    raise ImportError(
        "this build of ann-search has no MLX support compiled in. Build from "
        "source with `maturin develop --release --features mlx` after "
        "`brew install mlx-c`; without mlx-c the build leaves MLX out."
    )


class _BaseMlxIndex(BaseAnnIndex):
    """Shared settings for every MLX estimator, plus the thread check."""

    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _FORCE_DTYPE: ClassVar[np.dtype | None] = np.dtype(np.float32)
    _SERIALISABLE: ClassVar[bool] = False
    # Not a `_Handle`: no save/load. `_SERIALISABLE` keeps the base off those.
    _HANDLE: ClassVar[type[_GpuHandle]]  # ty: ignore[invalid-attribute-override]

    def fit(self, X: Any, y: Any = None) -> BaseAnnIndex:
        self._thread = threading.get_ident()
        return super().fit(X, y)

    def _fitted_handle(self) -> Any:
        # Checked here because pyo3's own guard on an unsendable class raises
        # a PanicException, a BaseException that `except Exception` misses.
        if getattr(self, "_thread", None) not in (None, threading.get_ident()):
            raise RuntimeError(
                f"{type(self).__name__} was fitted on another thread; MLX "
                "streams are tied to the thread that built them. Fit and query "
                "on the same thread."
            )
        return super()._fitted_handle()


class ExhaustiveMlxIndex(_BaseMlxIndex):
    """Brute-force exact search on MLX: its GEMM plus a Metal top-k kernel.

    Recall is 1 by construction.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed. Prints to the
            process stdout, not ``sys.stdout``.
    """

    _HANDLE = _core.ExhaustiveMlx

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        verbose: int = 0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.ExhaustiveMlx.build(
            x, metric=self._core_metric, verbose=self.verbose
        )


class IvfMlxIndex(_BaseMlxIndex):
    """Inverted file on MLX: k-means, the vectors and the cell scan on device.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Number of Voronoi cells. ``None`` defaults to ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time: override it per call as
            ``index.kneighbors(nprobe=32)``.
        nquery: Queries per device tile. ``None`` lets the library size it.
            Search-time.
        kmeans_iters: Lloyd iterations. ``None`` defaults to 30. Every one
            runs; MLX k-means has no early stop.
        kmeans_balanced: Reseed starved centroids each iteration.
        seed: Fixes k-means initialisation.
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed. Prints to the
            process stdout, not ``sys.stdout``.
    """

    _HANDLE = _core.IvfMlx
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ("nprobe", "nquery")

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        nlist: int | None = None,
        nprobe: int | None = None,
        nquery: int | None = None,
        kmeans_iters: int | None = None,
        kmeans_balanced: bool = False,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.nquery = nquery
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfMlx.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            seed=self.seed,
            verbose=self.verbose,
        )


class CagraMlxIndex(ExtractKnnMixin, _BaseMlxIndex):
    """CAGRA graph on MLX: NN-Descent, the prune and the beam search on device.

    Same parameters and beam semantics as `CagraGpuIndex`, minus
    ``retain_gpu``: the MLX index is resident from the build on. External
    queries pick their entry points through a random projection forest on the
    device.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`. Not the graph degree, which is `graph_degree`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        graph_degree: Neighbours stored per node. ``None`` defaults to 30.
        build_k: Working degree before pruning. ``None`` defaults to
            ``1.5 * graph_degree``.
        max_iters: Descent iteration cap. ``None`` defaults to 15.
        n_trees: Random projection trees for the initial graph. ``None``
            defaults to ``min(5 + round(n ** 0.25), 20)``.
        delta: Convergence threshold. ``None`` defaults to 0.001.
        rho: Local-join sampling rate. ``None`` defaults to 1.0.
        refine_knn: Two-hop refinement sweeps. ``None`` defaults to 0.
        beam_width: Beam width at query time, the recall knob. Search-time.
        max_beam_iters: Iteration cap on the beam search. Search-time.
        n_entry_points: Entry points per query. Search-time.
        expand_per_iter: Beam entries expanded per iteration. Search-time.
        seed: Fixes the seed graph and the sampling.
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed. Prints to the
            process stdout, not ``sys.stdout``.

    Note:
        The four beam parameters are all-or-nothing, exactly as on
        `CagraGpuIndex`. Leave them all ``None`` and the beam is sized from
        ``k`` (``beam_width = 2 * max(k, 16)``, ``max_beam_iters =
        3 * beam_width``). Set any one and the rest fall back to flat
        constants: 16, 48, 8 entry points and 3 expansions. If you touch one,
        set `beam_width` too.
    """

    _HANDLE = _core.CagraMlx
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = (
        "beam_width",
        "max_beam_iters",
        "n_entry_points",
        "expand_per_iter",
    )

    @beartype
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        graph_degree: int | None = None,
        build_k: int | None = None,
        max_iters: int | None = None,
        n_trees: int | None = None,
        delta: float | None = None,
        rho: float | None = None,
        refine_knn: int | None = None,
        beam_width: int | None = None,
        max_beam_iters: int | None = None,
        n_entry_points: int | None = None,
        expand_per_iter: int | None = None,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.graph_degree = graph_degree
        self.build_k = build_k
        self.max_iters = max_iters
        self.n_trees = n_trees
        self.delta = delta
        self.rho = rho
        self.refine_knn = refine_knn
        self.beam_width = beam_width
        self.max_beam_iters = max_beam_iters
        self.n_entry_points = n_entry_points
        self.expand_per_iter = expand_per_iter
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.CagraMlx.build(
            x,
            metric=self._core_metric,
            k=self.graph_degree,
            build_k=self.build_k,
            max_iters=self.max_iters,
            n_trees=self.n_trees,
            delta=self.delta,
            rho=self.rho,
            refine_knn=self.refine_knn,
            seed=self.seed,
            verbose=self.verbose,
        )
