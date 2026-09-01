"""GPU estimators, present only when the extension was built with them.

`ann_search.gpu_available()` is the check worth making: it says whether this
machine has an adapter, which is the part that varies. The ordinary wheel has
GPU support compiled in, so importing this module only fails on a build made
with ``--no-default-features``.

Three things differ from the CPU estimators, all of them consequences of the
backend rather than choices:

- **float32 only.** WGSL has no float64, so `fit` casts rather than letting a
  float64 array fail somewhere inside a kernel. That is a narrowing conversion,
  and the only one this package performs silently.
- **No persistence.** These indices hold device buffers and sit outside the
  crate's `serialise` feature, so `save`, `load` and `pickle` raise. Rebuild.
- **Manhattan is unavailable** on all three.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from . import _ann_search as _core
from ._base import BaseAnnIndex, ExtractKnnMixin
from ._metrics import NO_MANHATTAN

if not hasattr(_core, "ExhaustiveGpu"):  # pragma: no cover - build-dependent
    raise ImportError(
        "this build of ann-search has no GPU support compiled in. The 'gpu' "
        "feature is on by default, so this is a --no-default-features build; "
        "reinstall without that flag to get the GPU indices."
    )

###########
# Globals #
###########

#: The only element type the GPU backend can carry.
_GPU_DTYPE: np.dtype = np.dtype(np.float32)


class _BaseGpuIndex(BaseAnnIndex):
    """Shared settings for every GPU estimator.

    Exists to state the three backend constraints once. It carries no methods
    of its own, so it is a settings holder rather than a layer.
    """

    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = NO_MANHATTAN
    _FORCE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE
    _SERIALISABLE: ClassVar[bool] = False


class ExhaustiveGpuIndex(_BaseGpuIndex):
    """Brute-force exact search on the device.

    Recall is 1 by construction, and on a dataset too large to score on the CPU
    this is the cheapest way to get ground truth. No build-time knobs: the data
    goes up, the norms get recorded, and that is the index.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        verbose: Progress to the process stdout, not ``sys.stdout``. In Jupyter
            that lands in the terminal running the kernel.
    """

    _HANDLE: ClassVar[type] = _core.ExhaustiveGpu

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
        return _core.ExhaustiveGpu.build(x, metric=self._core_metric)


class IvfGpuIndex(_BaseGpuIndex):
    """Inverted file with k-means and the vectors both on the device.

    Trains and queries without a readback, which is what makes it quick, and
    also what bounds it: the reordered vectors stay resident, so the dataset has
    to fit in device memory. `nquery` caps how many queries stage per batch if
    the upload is what does not fit.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        nlist: Number of Voronoi cells to cut the space into. ``None`` defaults
            to ``sqrt(n)``.
        nprobe: Cells visited per query, the recall knob. ``None`` defaults to
            ``sqrt(nlist)``. Search-time: override it per call as
            ``index.kneighbors(nprobe=32)``.
        nquery: Queries staged on the device per batch. ``None`` sizes the
            batch from `nprobe`, the average cell size and the device's
            maximum binding, clamped into ``100..=20_000``. Lower it if a
            large `nprobe` overruns the candidate buffer. Search-time.
        kmeans_iters: Lloyd iterations when training the cells. ``None``
            defaults to 50, which is the GPU default and higher than the CPU
            side's 30: the iterations are cheap once the data is resident.
        kmeans_balanced: Reseed starved centroids each iteration, RAFT-style.
            Evens out the posting lists, which matters more here than on the
            CPU because a straggler cell serialises its whole workgroup.
        quantise_to_f16: Hold the resident data buffer at fp16. Halves its
            memory and lifts effective bandwidth on the assignment kernels, at
            the cost of needing ``shader-f16`` on the adapter.
        seed: Fixes k-means initialisation.
        verbose: Progress to the process stdout, not ``sys.stdout``. In Jupyter
            that lands in the terminal running the kernel.
    """

    _HANDLE: ClassVar[type] = _core.IvfGpu
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
        quantise_to_f16: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.nquery = nquery
        self.kmeans_iters = kmeans_iters
        self.kmeans_balanced = kmeans_balanced
        self.quantise_to_f16 = quantise_to_f16
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        return _core.IvfGpu.build(
            x,
            metric=self._core_metric,
            nlist=self.nlist,
            kmeans_iters=self.kmeans_iters,
            kmeans_balanced=self.kmeans_balanced,
            quantise_to_f16=self.quantise_to_f16,
            seed=self.seed,
            verbose=self.verbose,
        )


class CagraGpuIndex(ExtractKnnMixin, _BaseGpuIndex):
    """CAGRA graph: NN-Descent on the device, pruned, then beam-searched.

    The fastest route to a kNN graph here when a GPU is present. `beam_width` is
    the recall knob; leaving it ``None`` sizes the beam as ``2 * max(k, 16)``,
    which is usually the right answer.

    `extract_knn` hands back the kNN graph the descent converged on, taken
    before the CAGRA prune. No kernel runs, and it is capped by `graph_degree`
    rather than by the beam.

    Unlike every other index in this package, a fitted handle is not safe to
    query from two threads at once: the beam search memoises its upload of the
    navigational graph behind a mutable borrow, so concurrent calls serialise.
    `extract_knn` is exempt.

    Args:
        n_neighbors: Neighbours per query, and the default ``k`` for
            `kneighbors`. Not the graph degree, which is `graph_degree`.
        metric: ``"euclidean"``/``"l2"``, ``"sqeuclidean"`` or ``"cosine"``.
        graph_degree: Neighbours stored per node in the final graph. ``None``
            defaults to 30. The memory knob, and the ceiling on what
            `extract_knn` can return.
        build_k: Working degree the descent keeps before pruning. ``None``
            defaults to ``1.5 * graph_degree``. Room to manoeuvre, so above
            `graph_degree` is the point.
        max_iters: Descent iteration cap. ``None`` defaults to 15.
        n_trees: Random projection trees used to seed the initial graph.
            ``None`` defaults to ``min(5 + round(n ** 0.25), 20)``.
        delta: Convergence threshold. Descent stops once the fraction of
            updated edges falls below it. ``None`` defaults to 0.001.
        rho: Local-join sampling rate. ``None`` defaults to 1.0, i.e. no
            sampling. Lower it to trade graph quality for build time.
        refine_knn: Two-hop refinement sweeps after the main loop. ``None``
            defaults to 0, so refinement is off.
        retain_gpu: Upload the navigational graph at build time rather than on
            the first query. On by default, because the first query would
            otherwise pay for it.
        beam_width: Beam width at query time, the recall knob. Search-time.
        max_beam_iters: Iteration cap on the beam search. Search-time.
        n_entry_points: Entry points into the graph per query. Search-time.
        expand_per_iter: Extra neighbours explored per beam iteration, usually
            1 to 4. Search-time.
        seed: Fixes the seed graph and the sampling.
        verbose: Progress to the process stdout, not ``sys.stdout``. In Jupyter
            that lands in the terminal running the kernel.

    Note:
        The four beam parameters are all-or-nothing. Leave every one of them
        ``None`` and the beam is sized from ``k``: ``beam_width =
        2 * max(k, 16)`` and ``max_beam_iters = 3 * beam_width``, with 8 entry
        points and 3 expansions per iteration. Set **any** one of them and that
        scaling is off for all four, and the ones still ``None`` fall back to
        flat constants: ``beam_width = 16``, ``max_beam_iters = 48``,
        ``n_entry_points = 8``, ``expand_per_iter = 3``. So asking for
        ``n_entry_points=16`` alone on a ``k=50`` query quietly narrows the beam
        from 100 to 16. If you touch one, set `beam_width` too.
    """

    _HANDLE: ClassVar[type] = _core.CagraGpu
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
        retain_gpu: bool = True,
        beam_width: int | None = None,
        max_beam_iters: int | None = None,
        n_entry_points: int | None = None,
        expand_per_iter: int | None = None,
        seed: int = 42,
        verbose: bool = False,
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
        self.retain_gpu = retain_gpu
        self.beam_width = beam_width
        self.max_beam_iters = max_beam_iters
        self.n_entry_points = n_entry_points
        self.expand_per_iter = expand_per_iter
        self.seed = seed
        self.verbose = verbose

    def _build(self, x: np.ndarray) -> Any:
        # `graph_degree` rather than `k`: this sizes the stored graph, which is
        # a build cost, and is independent of the `k` a query asks for.
        return _core.CagraGpu.build(
            x,
            metric=self._core_metric,
            k=self.graph_degree,
            build_k=self.build_k,
            max_iters=self.max_iters,
            n_trees=self.n_trees,
            delta=self.delta,
            rho=self.rho,
            refine_knn=self.refine_knn,
            retain_gpu=self.retain_gpu,
            seed=self.seed,
            verbose=self.verbose,
        )
