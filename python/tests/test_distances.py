"""Distances that leave the FFI boundary are non-negative and finite.

Cosine is computed as ``1 - dot / (|x| |y|)``, and for a point against itself
that ratio rounds just above 1, so the raw core value lands one f32 ulp below
zero. One negative entry is enough for scikit-learn to reject an entire
precomputed matrix, so this is a contract rather than a nicety.
"""

import numpy as np
import pytest
from conftest import ALL_INDICES

import ann_search as ann

GPU_INDICES = (
    [ann.ExhaustiveGpuIndex, ann.IvfGpuIndex, ann.CagraGpuIndex]
    if hasattr(ann, "ExhaustiveGpuIndex")
    else []
)


@pytest.mark.parametrize(
    "cls", list(ALL_INDICES) + GPU_INDICES, ids=lambda v: getattr(v, "__name__", v)
)
def test_cosine_distances_are_non_negative(cls, clustered):
    """No index leaks a negative or NaN distance on any of its three paths."""
    if "cosine" not in cls._SUPPORTED_METRICS:
        pytest.skip(f"{cls.__name__} does not support cosine")

    index = cls(n_neighbors=10, metric="cosine").fit(clustered)
    queries = clustered[:200]

    for name, dist in [
        ("self", index.kneighbors()[0]),
        ("cross", index.kneighbors(queries)[0]),
        ("graph", index.kneighbors_graph().data),
    ]:
        assert not np.isnan(dist).any(), f"{name} produced NaN"
        # Padding comes back as inf, which is expected and not a failure.
        finite = dist[np.isfinite(dist)]
        assert (finite >= 0).all(), (
            f"{name} produced {(finite < 0).sum()} negative distances, "
            f"min {finite.min():.3e}"
        )


def test_euclidean_survives_the_clamp(clustered):
    """The clamp runs before the sqrt, so it must not disturb real distances."""
    index = ann.ExhaustiveIndex(n_neighbors=10, metric="euclidean").fit(clustered)
    dist, _ = index.kneighbors()
    sq, _ = (
        ann.ExhaustiveIndex(n_neighbors=10, metric="sqeuclidean")
        .fit(clustered)
        .kneighbors()
    )
    assert np.allclose(dist, np.sqrt(sq), rtol=1e-6)
