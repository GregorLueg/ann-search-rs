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

# Two separate skips, as in test_gpu.py. `hasattr` answers "was the extension
# built with the gpu feature", which is not the same question as "does this
# machine have an adapter". A hosted CI runner answers yes to the first and no
# to the second, and dispatching a kernel there panics inside cubecl rather
# than failing gracefully.
GPU_INDICES = [
    getattr(ann, name, None)
    for name in ("ExhaustiveGpuIndex", "IvfGpuIndex", "CagraGpuIndex")
]
GPU_INDICES = [c for c in GPU_INDICES if c is not None]


def check_non_negative(index, queries):
    """No negative or NaN distance on any of an index's three paths."""
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


@pytest.mark.parametrize(
    "cls", list(ALL_INDICES), ids=lambda v: getattr(v, "__name__", v)
)
def test_cosine_distances_are_non_negative(cls, clustered):
    """No CPU index leaks a negative or NaN distance."""
    if "cosine" not in cls._SUPPORTED_METRICS:
        pytest.skip(f"{cls.__name__} does not support cosine")
    index = cls(n_neighbors=10, metric="cosine").fit(clustered)
    check_non_negative(index, clustered[:200])


@pytest.mark.skipif(
    not ann.gpu_available(),
    reason="no GPU support compiled in, or no adapter on this machine",
)
@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda v: getattr(v, "__name__", v))
def test_gpu_cosine_distances_are_non_negative(cls, clustered):
    """Same contract for the device-resident indices."""
    index = cls(n_neighbors=10, metric="cosine").fit(clustered)
    check_non_negative(index, clustered[:200])


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
