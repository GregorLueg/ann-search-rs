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

from . import datasets
from ._ann_search import (
    AnnSearchError,
    IndexIoError,
    __version__,
    num_threads,
    set_num_threads,
)
from ._base import BaseAnnIndex, NotFittedError
from .indices import (
    AnnoyIndex,
    ExhaustiveIndex,
    HnswIndex,
    IvfIndex,
    KmknnIndex,
    NNDescentIndex,
    NsgIndex,
    VamanaIndex,
)

__all__ = [
    "AnnSearchError",
    "AnnoyIndex",
    "BaseAnnIndex",
    "ExhaustiveIndex",
    "HnswIndex",
    "IndexIoError",
    "IvfIndex",
    "KmknnIndex",
    "NNDescentIndex",
    "NotFittedError",
    "NsgIndex",
    "VamanaIndex",
    "__version__",
    "datasets",
    "num_threads",
    "set_num_threads",
]
