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
from ._base import BaseAnnIndex
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


class CagraGpuIndex(_BaseGpuIndex):
    """CAGRA graph: NN-Descent on the device, pruned, then beam-searched.

    The fastest route to a kNN graph here when a GPU is present. `beam_width` is
    the recall knob; leaving it ``None`` lets the library size the beam from `k`
    and the graph degree, which is usually the right answer.

    Unlike every other index in this package, a fitted handle is not safe to
    query from two threads at once: the beam search memoises its upload of the
    navigational graph behind a mutable borrow, so concurrent calls serialise.
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
