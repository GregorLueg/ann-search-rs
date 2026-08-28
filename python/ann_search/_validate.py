"""Array checks at the FFI boundary."""

from typing import Any

import numpy as np
from beartype import beartype

###########
# Globals #
###########

#: The two element types the Rust core is compiled for.
_NATIVE_DTYPES = (np.float32, np.float64)


@beartype
def check_matrix(x: Any, *, name: str = "X") -> np.ndarray:
    """Coerce an array-like into something the Rust core can borrow directly.

    float32 and float64 pass through untouched. Any other numeric type is
    promoted to float64 rather than narrowed to float32, so precision is never
    silently lost. The result is always C-contiguous, because the core borrows
    the buffer rather than copying it.

    Args:
        x: Array-like of shape ``(n_samples, n_features)``.
        name: Argument name, used in error messages.

    Returns:
        A C-contiguous 2-D float32 or float64 array.

    Raises:
        ValueError: If the input is not 2-D, is empty, or holds non-finite
            values. The core does not check finiteness and would build a
            silently useless index.
        TypeError: If the input does not hold numbers.
    """
    arr = np.asarray(x)

    if arr.dtype.kind not in "fiub":
        raise TypeError(f"{name} must hold numbers, got dtype {arr.dtype}")
    if arr.dtype.type not in _NATIVE_DTYPES:
        arr = arr.astype(np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x features), got {arr.ndim}-D")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return np.ascontiguousarray(arr)


@beartype
def check_query(x: Any, n_features: int, dtype: np.dtype | None) -> np.ndarray:
    """Coerce a query matrix to match an already-built index.

    Args:
        x: Array-like of shape ``(n_queries, n_features)``.
        n_features: Feature count the index was fitted on.
        dtype: Element type the index was built with. ``None`` only happens on
            an unfitted estimator, which the caller has already ruled out.

    Returns:
        A C-contiguous array of `dtype` with `n_features` columns.

    Raises:
        ValueError: If the width does not match the index.
    """
    arr = check_matrix(x, name="X")
    if arr.shape[1] != n_features:
        raise ValueError(
            f"query has {arr.shape[1]} features but the index was fitted on "
            f"{n_features}"
        )
    if dtype is not None and arr.dtype != dtype:
        arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr
