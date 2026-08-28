"""Metric names and the squared-to-true Euclidean transform."""

import numpy as np
import pytest

import ann_search as ann

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.default_rng(1).standard_normal((400, 16))


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
def test_distances_match_sklearn(metric, data):
    ours = ann.ExhaustiveIndex(n_neighbors=5, metric=metric).fit(data)
    ref = sklearn_neighbors.NearestNeighbors(
        n_neighbors=5, metric=metric, algorithm="brute"
    ).fit(data)
    # atol rather than the default: sklearn's brute Euclidean uses the expanded
    # ||a||^2 + ||b||^2 - 2ab form, so its self-distances come out around 1e-7
    # instead of exactly 0. Ours are exact there.
    assert np.allclose(
        ours.kneighbors(data[:50])[0], ref.kneighbors(data[:50])[0], atol=1e-6
    )


def test_sqeuclidean_is_the_unrooted_form(data):
    rooted = ann.ExhaustiveIndex(n_neighbors=5, metric="euclidean").fit(data)
    squared = ann.ExhaustiveIndex(n_neighbors=5, metric="sqeuclidean").fit(data)
    assert np.allclose(rooted.kneighbors()[0], np.sqrt(squared.kneighbors()[0]))


@pytest.mark.parametrize("alias,canonical", [("l2", "euclidean"), ("l1", "manhattan")])
def test_metric_aliases(alias, canonical, data):
    a = ann.ExhaustiveIndex(n_neighbors=5, metric=alias).fit(data).kneighbors()[0]
    b = ann.ExhaustiveIndex(n_neighbors=5, metric=canonical).fit(data).kneighbors()[0]
    assert np.array_equal(a, b)


def test_unknown_metric_raises_rather_than_falling_back(data):
    with pytest.raises(ValueError, match="unsupported metric"):
        ann.HnswIndex(metric="jaccard").fit(data)


@pytest.mark.parametrize(
    "cls",
    [
        ann.AnnoyIndex,
        ann.BallTreeIndex,
        ann.KmknnIndex,
        ann.LshIndex,
        ann.SoarIndex,
    ],
    ids=lambda c: c.__name__,
)
def test_manhattan_rejected_where_unsupported(cls, data):
    with pytest.raises(ValueError, match="unsupported metric"):
        cls(metric="manhattan").fit(data)


@pytest.mark.parametrize(
    "cls", [ann.KdTreeIndex, ann.RnnDescentIndex], ids=lambda c: c.__name__
)
def test_manhattan_accepted_where_supported(cls, data):
    ours = cls(n_neighbors=5, metric="manhattan").fit(data)
    ref = sklearn_neighbors.NearestNeighbors(
        n_neighbors=5, metric="manhattan", algorithm="brute"
    ).fit(data)
    # Both are approximate, so compare the distances they did find rather than
    # demanding the same neighbour sets.
    assert ours.kneighbors(data[:50])[0].mean() >= ref.kneighbors(data[:50])[0].mean()
