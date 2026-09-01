"""GPU estimators.

Every test here skips twice over: once when the extension was built without the
`gpu` feature, once when the machine has no adapter. Both are ordinary states,
not failures, so CI on a headless runner stays green.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import recall

import ann_search as ann

pytestmark = pytest.mark.skipif(
    not ann.gpu_available(),
    reason="no GPU support compiled in, or no adapter on this machine",
)

# Collected at import time, so the skip above has to come first. Guarded with
# getattr because a CPU-only build has no such attributes at all.
GPU_INDICES = [
    getattr(ann, name, None)
    for name in ("ExhaustiveGpuIndex", "IvfGpuIndex", "CagraGpuIndex")
]
GPU_INDICES = [c for c in GPU_INDICES if c is not None]

#: Per-algorithm recall floors on `clustered` at k=10, as in `conftest`. The
#: exhaustive GPU index is exact, so it is pinned at 1.0.
FLOORS: dict[str, float] = {
    "ExhaustiveGpuIndex": 1.0,
    "IvfGpuIndex": 0.60,
    "CagraGpuIndex": 0.90,
}


@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda c: c.__name__)
def test_self_query_clears_its_recall_floor(cls, clustered, truth):
    found = cls(n_neighbors=10).fit(clustered).kneighbors()[1]
    assert recall(found, truth) >= FLOORS[cls.__name__]


@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda c: c.__name__)
def test_cross_set_query_shape(cls, clustered):
    idx = cls(n_neighbors=7).fit(clustered)
    dist, ind = idx.kneighbors(clustered[:64])
    assert dist.shape == ind.shape == (64, 7)
    assert ind.max() < len(clustered)


@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda c: c.__name__)
def test_float64_input_is_narrowed_to_float32(cls, clustered):
    idx = cls(n_neighbors=5).fit(clustered.astype(np.float64))
    assert idx._handle.dtype == "float32"
    assert idx.kneighbors(clustered[:8].astype(np.float64))[0].dtype == np.float32


@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda c: c.__name__)
def test_persistence_refused_with_a_reason(cls, clustered):
    idx = cls(n_neighbors=5).fit(clustered)
    with pytest.raises(NotImplementedError, match="serialise feature"):
        idx.save(Path(tempfile.mkdtemp()) / "i")
    with pytest.raises(NotImplementedError, match="serialise feature"):
        pickle.dumps(idx)
    with pytest.raises(NotImplementedError, match="serialise feature"):
        cls.load(Path(tempfile.mkdtemp()))


@pytest.mark.parametrize("cls", GPU_INDICES, ids=lambda c: c.__name__)
def test_manhattan_rejected(cls, clustered):
    with pytest.raises(ValueError, match="unsupported metric"):
        cls(metric="manhattan").fit(clustered)


def test_exhaustive_gpu_matches_the_cpu_ground_truth(clustered, truth):
    found = ann.ExhaustiveGpuIndex(n_neighbors=10).fit(clustered).kneighbors()[1]
    assert np.array_equal(np.sort(found, axis=1), np.sort(truth, axis=1))


def test_search_knobs_are_per_call(clustered):
    idx = ann.IvfGpuIndex(n_neighbors=10, nprobe=1).fit(clustered)
    tight = recall(idx.kneighbors()[1], idx.kneighbors(nprobe=1)[1])
    assert tight == 1.0
    with pytest.raises(TypeError, match="unexpected search argument"):
        idx.kneighbors(clustered[:4], ef_search=10)


def test_cagra_beam_width_raises_recall(clustered, truth):
    idx = ann.CagraGpuIndex(n_neighbors=10).fit(clustered)
    narrow = recall(idx.kneighbors(beam_width=16)[1], truth)
    wide = recall(idx.kneighbors(beam_width=128)[1], truth)
    assert wide >= narrow
