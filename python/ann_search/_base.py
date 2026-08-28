"""Shared estimator behaviour.

Every index here is immutable once built, so the scikit-learn shape (parameters
in ``__init__``, data in ``fit``, results from ``kneighbors``) is the honest one:
the FAISS-style ``add()`` would be a method callable exactly once.

``get_params`` and ``set_params`` introspect the subclass ``__init__``, which is
all ``sklearn.base.BaseEstimator`` does. Doing it here keeps scikit-learn out of
the install requirements while ``clone``, ``GridSearchCV`` and ``Pipeline`` still
work by duck-typing.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from beartype import beartype

from ._metrics import ALL_METRICS, resolve_metric
from ._validate import check_matrix, check_query

if TYPE_CHECKING:
    # Shape of the compiled handles, declared in `_ann_search.pyi`. It has no
    # runtime existence, which is why the import is guarded.
    from ._ann_search import _Handle

###########
# Globals #
###########

#: Sidecar written next to the Rust bundle so `load` can rebuild the estimator.
_PARAMS_FILE = "params.json"


class NotFittedError(ValueError, AttributeError):
    """Raised when a query is attempted before `fit`.

    Inherits from both `ValueError` and `AttributeError` to match
    `sklearn.exceptions.NotFittedError`, so code catching either still works.
    """


class BaseAnnIndex:
    """Common `fit` / `kneighbors` plumbing for every index.

    Subclasses supply an ``__init__`` that stores its parameters verbatim, a
    ``_build`` hook, a ``_search_kwargs`` hook naming the algorithm's
    search-time knobs, and the handle class from the compiled core.
    """

    #: The `_ann_search` handle class this estimator drives.
    _HANDLE: ClassVar[type[_Handle]]
    #: Metrics this algorithm supports. Narrowed by subclasses that need it.
    _SUPPORTED_METRICS: ClassVar[frozenset[str]] = ALL_METRICS
    #: Constructor parameters that are search-time knobs, so `kneighbors` can
    #: take a per-call override for each without a hook per subclass.
    _SEARCH_KNOBS: ClassVar[tuple[str, ...]] = ()

    # Every subclass takes these three in its `__init__`. Annotated but not
    # assigned, so they stay out of the class dict and `get_params` still reads
    # them off the instance.
    n_neighbors: int
    metric: str
    verbose: bool

    # Unfitted state, as class attributes so subclasses need no `super().__init__`.
    _handle: _Handle | None = None
    _core_metric: str = ""
    _sqrt: bool = False
    _dtype: np.dtype | None = None
    n_features_in_: int = 0
    n_samples_fit_: int = 0

    ############
    # Subclass #
    ############

    def _build(self, x: np.ndarray) -> Any:
        """Build the core handle from a validated, contiguous array."""
        raise NotImplementedError

    def _search_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Search-time knobs for this algorithm, with per-call overrides applied."""
        unknown = set(overrides) - set(self._SEARCH_KNOBS)
        if unknown:
            allowed = ", ".join(self._SEARCH_KNOBS) or "none"
            raise TypeError(
                f"unexpected search argument(s) {', '.join(sorted(unknown))} for "
                f"{type(self).__name__}; this index takes: {allowed}"
            )
        return {n: overrides.get(n, getattr(self, n)) for n in self._SEARCH_KNOBS}

    ##########
    # Params #
    ##########

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return sorted(p for p in sig.parameters if p != "self")

    @beartype
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Parameters this estimator was constructed with.

        Args:
            deep: Accepted for scikit-learn compatibility; these estimators hold
                no nested estimators, so it makes no difference.

        Returns:
            Constructor parameters, keyed by name.
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> BaseAnnIndex:
        """Set constructor parameters, invalidating any fitted index.

        Returns:
            self.

        Raises:
            ValueError: If a name is not a parameter of this estimator.
        """
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"invalid parameter {key!r} for {type(self).__name__}; "
                    f"expected one of: {', '.join(sorted(valid))}"
                )
            setattr(self, key, value)
        self._handle = None
        return self

    #######
    # Fit #
    #######

    def fit(self, X: Any, y: Any = None) -> BaseAnnIndex:
        """Build the index over `X`.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``. float32 and
                float64 are used as-is; other numeric types are promoted to
                float64.
            y: Ignored, present for scikit-learn pipeline compatibility.

        Returns:
            self.
        """
        arr = check_matrix(X)
        self._core_metric, self._sqrt = resolve_metric(
            self.metric, self._SUPPORTED_METRICS
        )
        self._handle = self._build(arr)
        self._dtype = arr.dtype
        self.n_samples_fit_, self.n_features_in_ = arr.shape
        return self

    def _fitted_handle(self) -> _Handle:
        """The built handle, or a clear error if `fit` has not run."""
        if self._handle is None:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted; call fit(X) first"
            )
        return self._handle

    ###########
    # Queries #
    ###########

    def _search(
        self,
        X: Any,
        n_neighbors: int | None,
        return_distance: bool,
        overrides: dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Run the query and return ``(distances or None, indices)``.

        Always a pair, unlike the public `kneighbors`, so internal callers do
        not have to unpick a union.
        """
        handle = self._fitted_handle()
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        kwargs = self._search_kwargs(overrides)

        if X is None:
            ind, dist = handle.query_self(
                k, return_distance=return_distance, verbose=self.verbose, **kwargs
            )
        else:
            q = check_query(X, self.n_features_in_, self._dtype)
            ind, dist = handle.query(
                q, k, return_distance=return_distance, verbose=self.verbose, **kwargs
            )

        if dist is not None and self._sqrt:
            # The core returns squared Euclidean; sqrt(inf) is inf, so the
            # padding survives untouched.
            np.sqrt(dist, out=dist)
        return dist, ind

    def kneighbors(
        self,
        X: Any = None,
        n_neighbors: int | None = None,
        *,
        return_distance: bool = True,
        **overrides: Any,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Find the nearest neighbours of `X` among the fitted points.

        Args:
            X: Query points of shape ``(n_queries, n_features)``. ``None`` means
                query the fitted data against itself, which takes each index's
                own fast path rather than re-entering from outside.
            n_neighbors: Neighbours per query. Defaults to ``self.n_neighbors``.
            return_distance: Whether to return distances alongside indices. This
                saves the copy into numpy but not the distance computation,
                which the core does either way.
            **overrides: Per-call values for this algorithm's search-time knobs,
                for example ``ef_search`` or ``nprobe``.

        Returns:
            ``(distances, indices)``, or just ``indices`` when
            `return_distance` is False. Both have shape ``(n_queries, k)``.
            A query that found fewer than `k` neighbours is padded with ``-1``
            indices and infinite distances.
        """
        dist, ind = self._search(X, n_neighbors, return_distance, overrides)
        if dist is None:
            return ind
        return dist, ind

    def kneighbors_graph(
        self,
        X: Any = None,
        n_neighbors: int | None = None,
        mode: str = "distance",
        **overrides: Any,
    ) -> Any:
        """Build the sparse neighbourhood graph.

        Args:
            X: Query points, or ``None`` for the self-kNN graph.
            n_neighbors: Neighbours per query. Defaults to ``self.n_neighbors``.
            mode: ``"distance"`` weights edges by distance, ``"connectivity"``
                by 1.
            **overrides: Per-call search-time knobs, as for `kneighbors`.

        Returns:
            A ``scipy.sparse.csr_matrix`` of shape
            ``(n_queries, n_samples_fit_)``. Padding slots are dropped, so rows
            can hold fewer than `k` entries.

        Raises:
            ImportError: If scipy is not installed.
            ValueError: If `mode` is not recognised.
        """
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "kneighbors_graph needs scipy; install ann-search[sparse]"
            ) from e

        match mode:
            case "distance":
                weighted = True
            case "connectivity":
                weighted = False
            case _:
                raise ValueError(
                    f"unknown mode {mode!r}; expected 'distance' or 'connectivity'"
                )

        dist, ind = self._search(X, n_neighbors, weighted, overrides)

        mask = ind >= 0
        indptr = np.zeros(ind.shape[0] + 1, dtype=np.int64)
        np.cumsum(mask.sum(axis=1), out=indptr[1:])
        indices = ind[mask]
        data = (
            dist[mask] if dist is not None else np.ones(indices.size, dtype=np.float64)
        )
        return csr_matrix(
            (data, indices, indptr), shape=(ind.shape[0], self.n_samples_fit_)
        )

    def transform(self, X: Any = None) -> Any:
        """Neighbourhood graph of `X`, for use as a `KNeighborsTransformer`."""
        return self.kneighbors_graph(X)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Fit on `X` and return its self-kNN graph."""
        return self.fit(X, y).kneighbors_graph(None)

    ###############
    # Persistence #
    ###############

    def save(self, path: str | Path) -> None:
        """Write the fitted index to a directory.

        The directory holds the core's own bundle plus a small JSON sidecar with
        the estimator parameters, so `load` reproduces the whole object rather
        than a bare handle.

        Args:
            path: Target directory. Created if it does not exist.
        """
        handle = self._fitted_handle()
        target = Path(path)
        handle.save(target)
        meta = {
            "class": type(self).__name__,
            "params": self.get_params(),
            "n_features_in_": self.n_features_in_,
            "n_samples_fit_": self.n_samples_fit_,
            "dtype": str(self._dtype),
        }
        (target / _PARAMS_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BaseAnnIndex:
        """Read an index written by `save`.

        Args:
            path: Directory holding the bundle.

        Returns:
            The reconstructed estimator.

        Raises:
            ValueError: If the directory was written by a different index type.
        """
        source = Path(path)
        meta = json.loads((source / _PARAMS_FILE).read_text())
        if meta["class"] != cls.__name__:
            raise ValueError(f"{source} holds a {meta['class']}, not a {cls.__name__}")
        obj = cls(**meta["params"])
        obj._handle = cls._HANDLE.load(source)
        obj._core_metric, obj._sqrt = resolve_metric(obj.metric, cls._SUPPORTED_METRICS)
        obj._dtype = np.dtype(meta["dtype"])
        obj.n_features_in_ = meta["n_features_in_"]
        obj.n_samples_fit_ = meta["n_samples_fit_"]
        return obj

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = {"params": self.get_params(), "fitted": None}
        if self._handle is not None:
            state["fitted"] = {
                "handle": self._handle.__getstate__(),
                "dtype": str(self._dtype),
                "n_features_in_": self.n_features_in_,
                "n_samples_fit_": self.n_samples_fit_,
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state["params"].items():
            setattr(self, name, value)
        fitted = state["fitted"]
        if fitted is None:
            return
        self._handle = type(self)._HANDLE.from_bytes(fitted["handle"])
        self._core_metric, self._sqrt = resolve_metric(
            self.metric, type(self)._SUPPORTED_METRICS
        )
        self._dtype = np.dtype(fitted["dtype"])
        self.n_features_in_ = fitted["n_features_in_"]
        self.n_samples_fit_ = fitted["n_samples_fit_"]

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params().items()))
        return f"{type(self).__name__}({args})"
