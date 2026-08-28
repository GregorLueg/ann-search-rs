"""Synthetic datasets with realistic structure.

Uniform Gaussian noise is a bad benchmark for nearest-neighbour search. Past a
few dozen dimensions every point sits at roughly the same distance from every
other, so recall stops telling you anything about the index. Each generator here
puts back structure that real single-cell data has, and each stresses a
different part of an index.

These are the same generators, with the same seeds, behind the benchmark tables
in the Rust crate's ``docs/``. A Python benchmark and a ``cargo run --example
gridsearch_hnsw`` run see identical points, so the numbers are comparable.

    >>> from ann_search import datasets
    >>> X, labels = datasets.make_clustered(n_samples=50_000, dim=32,
    ...                                     n_clusters=25, seed=42)
    >>> Q = datasets.subsample_queries(X, n_samples=5_000, seed=42)

Every generator returns ``(X, labels)``, so ground-truth cluster labels come
free: useful for recall, and for scoring whatever clustering you run downstream.

Output is always float32, which is what the benchmark tables use. Cast with
``X.astype(np.float64)`` if you want to exercise the f64 paths.

The tuning constants behind each generator (bridge fractions, spectral decay,
rogue-dimension counts) are fixed rather than exposed. They're what the
published tables were produced with, and changing one quietly invalidates the
comparison.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype

from . import _ann_search as _core

__all__ = [
    "make_cell_embeddings",
    "make_clustered",
    "make_correlated",
    "make_low_rank",
    "subsample_queries",
]


@beartype
def make_clustered(
    n_samples: int = 150_000,
    dim: int = 32,
    n_clusters: int = 25,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Separated Gaussian clusters joined by inter-cluster bridges.

    The baseline. Well-separated blobs with a fifth of the points placed on
    bridges between neighbouring clusters, so the boundaries aren't trivially
    clean.

    Args:
        n_samples: Rows to generate.
        dim: Features per row.
        n_clusters: Distinct clusters.
        seed: Fixes the whole draw.

    Returns:
        ``(X, labels)``, shapes ``(n_samples, dim)`` float32 and
        ``(n_samples,)`` int64.
    """
    return _core.make_clustered(n_samples, dim, n_clusters, seed)


@beartype
def make_correlated(
    n_samples: int = 150_000,
    dim: int = 32,
    n_clusters: int = 25,
    *,
    seed: int = 42,
    cor_strength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clusters with local anisotropy plus a shared off-axis subspace.

    The interesting case for quantisation. That shared subspace is exactly what
    OPQ's learned rotation exploits and what PQ's axis-aligned split can't see,
    so the two pull apart here in a way they don't on `make_clustered`.

    Args:
        n_samples: Rows to generate.
        dim: Features per row.
        n_clusters: Distinct clusters.
        seed: Fixes the whole draw.
        cor_strength: Share of structured variance routed to the global
            off-axis subspace, 0.0 to 1.0. ``None`` uses the value behind the
            published tables.

    Returns:
        ``(X, labels)``, shapes ``(n_samples, dim)`` float32 and
        ``(n_samples,)`` int64.

    Raises:
        ValueError: If `cor_strength` is outside 0.0 to 1.0.
    """
    if cor_strength is not None and not 0.0 <= cor_strength <= 1.0:
        raise ValueError(
            f"cor_strength must be between 0.0 and 1.0, got {cor_strength}"
        )
    return _core.make_correlated(n_samples, dim, n_clusters, seed, cor_strength)


@beartype
def make_low_rank(
    n_samples: int = 150_000,
    dim: int = 32,
    n_clusters: int = 25,
    *,
    intrinsic_dim: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Data on a low-dimensional manifold inside a high-dimensional space.

    The manifold hypothesis made concrete: cell types sit on an
    `intrinsic_dim`-dimensional surface embedded in `dim` ambient dimensions,
    with differentiation trajectories running between them.

    Args:
        n_samples: Rows to generate.
        dim: Ambient features per row.
        n_clusters: Distinct cell types.
        intrinsic_dim: True dimensionality of the manifold. Must not exceed
            `dim`.
        seed: Fixes the whole draw.

    Returns:
        ``(X, labels)``, shapes ``(n_samples, dim)`` float32 and
        ``(n_samples,)`` int64.

    Raises:
        ValueError: If `intrinsic_dim` exceeds `dim`.
    """
    if intrinsic_dim > dim:
        raise ValueError(f"intrinsic_dim ({intrinsic_dim}) cannot exceed dim ({dim})")
    return _core.make_low_rank(n_samples, dim, n_clusters, intrinsic_dim, seed)


@beartype
def make_cell_embeddings(
    n_samples: int = 150_000,
    dim: int = 32,
    n_clusters: int = 25,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Foundation-model cell embeddings, Geneformer or scGPT flavoured.

    The nastiest of the four. Heavy-tailed spectrum, a handful of high-variance
    rogue dimensions, and a shared mean offset that crams everything into an
    anisotropy cone. Quantised indices get painful here.

    Args:
        n_samples: Rows to generate.
        dim: Embedding width.
        n_clusters: Distinct cell types.
        seed: Fixes the whole draw.

    Returns:
        ``(X, labels)``, shapes ``(n_samples, dim)`` float32 and
        ``(n_samples,)`` int64.
    """
    return _core.make_cell_embeddings(n_samples, dim, n_clusters, seed)


@beartype
def subsample_queries(
    x: np.ndarray,
    n_samples: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Draw a query set from a dataset, with light Gaussian noise added.

    Querying an index with rows it was built from flatters it: every query has
    an exact hit at distance zero. This perturbs the draw so queries sit near
    the data instead of on it.

    Args:
        x: Dataset to draw from. Cast to float32 if it isn't already.
        n_samples: Rows to draw. Capped at the number available.
        seed: Fixes both the draw and the noise.

    Returns:
        ``(min(n_samples, len(x)), dim)`` float32.

    Raises:
        ValueError: If `x` is not 2-D.
    """
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"x must be 2-D (samples x features), got {arr.ndim}-D")
    return _core.subsample_queries(arr, n_samples, seed)
