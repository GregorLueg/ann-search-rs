"""Neighbour search: recall, shapes, dtypes and padding."""

import numpy as np
import pytest
from conftest import ALL_INDICES, recall

import ann_search as ann


@pytest.mark.parametrize(
    "cls,floor", ALL_INDICES.items(), ids=lambda v: getattr(v, "__name__", v)
)
def test_self_query_recall(cls, floor, clustered, truth):
    found = cls(n_neighbors=10).fit(clustered).kneighbors()[1]
    assert recall(found, truth) >= floor


@pytest.mark.parametrize("cls", ALL_INDICES, ids=lambda c: c.__name__)
def test_cross_query_shapes(cls, clustered):
    index = cls(n_neighbors=7).fit(clustered)
    dist, ind = index.kneighbors(clustered[:40])
    assert dist.shape == ind.shape == (40, 7)
    assert ind.dtype == np.int64
    assert dist.dtype == clustered.dtype


@pytest.mark.parametrize("cls", ALL_INDICES, ids=lambda c: c.__name__)
def test_return_distance_false(cls, clustered):
    index = cls(n_neighbors=5).fit(clustered)
    ind = index.kneighbors(clustered[:10], return_distance=False)
    assert ind.shape == (10, 5)
    assert np.array_equal(ind, index.kneighbors(clustered[:10])[1])


def test_exhaustive_finds_itself(clustered):
    dist, ind = ann.ExhaustiveIndex(n_neighbors=5).fit(clustered).kneighbors()
    assert np.array_equal(ind[:, 0], np.arange(len(clustered)))
    assert np.allclose(dist[:, 0], 0.0)


def test_n_neighbors_override(clustered):
    index = ann.HnswIndex(n_neighbors=5).fit(clustered)
    assert index.kneighbors(clustered[:3], n_neighbors=12)[1].shape == (3, 12)


def test_search_knob_override(clustered, truth):
    index = ann.IvfIndex(n_neighbors=10, nprobe=1).fit(clustered)
    tight = recall(index.kneighbors()[1], truth)
    wide = recall(index.kneighbors(nprobe=64)[1], truth)
    assert wide > tight


def test_padding_when_k_exceeds_n():
    small = np.arange(12, dtype=np.float32).reshape(4, 3)
    dist, ind = ann.HnswIndex(n_neighbors=8).fit(small).kneighbors()
    assert (ind[:, 4:] == -1).all()
    assert np.isinf(dist[:, 4:]).all()
    assert (ind[:, :4] >= 0).all()


def test_float32_and_float64_agree(clustered, truth):
    wide = ann.HnswIndex(n_neighbors=10).fit(clustered.astype(np.float64))
    assert recall(wide.kneighbors()[1], truth) >= 0.95
    assert wide.kneighbors()[0].dtype == np.float64
