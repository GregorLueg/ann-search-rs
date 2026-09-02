"""Vector search for single-cell and computational biology, in Rust.

Every index is a scikit-learn style estimator: parameters go in the
constructor, data goes into ``fit``, results come out of ``kneighbors``.

    >>> import numpy as np, ann_search as ann
    >>> X = np.random.default_rng(0).standard_normal((5000, 50)).astype("float32")
    >>> index = ann.HnswIndex(n_neighbors=15, metric="cosine").fit(X)
    >>> distances, indices = index.kneighbors()        # self-kNN, fast path
    >>> distances, indices = index.kneighbors(X[:100]) # cross-set

`kneighbors` returns distances first, matching scikit-learn and FAISS. A query
that came back with fewer than ``k`` neighbours is padded with ``-1`` indices
and infinite distances, and `kneighbors_graph` drops those slots.

Indices are immutable. There's no incremental ``add``, so rebuild instead.
"""

from . import _ann_search, datasets, quantised
from ._ann_search import (
    AnnSearchError,
    IndexIoError,
    __core_version__,
    __version__,
    gpu_available,
    num_threads,
    set_num_threads,
)
from ._base import BaseAnnIndex, NotFittedError
from .indices import (
    AnnoyIndex,
    BallTreeIndex,
    ExhaustiveIndex,
    HnswIndex,
    IvfIndex,
    KdTreeIndex,
    KmknnIndex,
    LshIndex,
    NNDescentIndex,
    NsgIndex,
    RnnDescentIndex,
    SoarIndex,
    VamanaIndex,
)
from .quantised import (
    ExhaustiveBf16Index,
    ExhaustiveOpqIndex,
    ExhaustivePqIndex,
    ExhaustiveSq8Index,
    HnswSq8uIndex,
    IvfBf16Index,
    IvfOpqIndex,
    IvfPqIndex,
    IvfSq8Index,
    SoarOpqIndex,
    SoarPqIndex,
)

__all__ = [
    "AnnSearchError",
    "AnnoyIndex",
    "BallTreeIndex",
    "BaseAnnIndex",
    "ExhaustiveBf16Index",
    "ExhaustiveIndex",
    "ExhaustiveOpqIndex",
    "ExhaustivePqIndex",
    "ExhaustiveSq8Index",
    "HnswIndex",
    "HnswSq8uIndex",
    "IndexIoError",
    "IvfBf16Index",
    "IvfIndex",
    "IvfOpqIndex",
    "IvfPqIndex",
    "IvfSq8Index",
    "KdTreeIndex",
    "KmknnIndex",
    "LshIndex",
    "NNDescentIndex",
    "NotFittedError",
    "NsgIndex",
    "RnnDescentIndex",
    "SoarIndex",
    "SoarOpqIndex",
    "SoarPqIndex",
    "VamanaIndex",
    "__core_version__",
    "__version__",
    "datasets",
    "gpu_available",
    "num_threads",
    "quantised",
    "set_num_threads",
]

# The GPU estimators exist only when the extension was built with them, which is
# fixed at wheel-build time. They are re-exported at the top level when present
# so `ann.IvfGpuIndex` works, and `ann_search.gpu` stays importable either way
# for anyone who wants the ImportError to say why.
if hasattr(_ann_search, "ExhaustiveGpu"):  # pragma: no cover - build-dependent
    from .gpu import CagraGpuIndex, ExhaustiveGpuIndex, IvfGpuIndex

    __all__ += ["CagraGpuIndex", "ExhaustiveGpuIndex", "IvfGpuIndex"]
