"""Metric names and the squared-to-true Euclidean transform."""

import numpy as np
import pytest
from conftest import QUANTISED

import ann_search as ann

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    """400 points in 32 dimensions.

    32 rather than something smaller because it is the minimum product
    quantisation accepts, and it divides by every default `m`.
    """
    return np.random.default_rng(1).standard_normal((400, 32))


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
        # Every quantised codec is built on inner products, so none of them can
        # do Manhattan.
        *QUANTISED,
    ],
    ids=lambda c: c.__name__,
)
def test_manhattan_rejected_where_unsupported(cls, data):
    with pytest.raises(ValueError, match="unsupported metric"):
        cls(metric="manhattan").fit(data)


@pytest.mark.parametrize("cls", list(QUANTISED), ids=lambda c: c.__name__)
def test_quantised_cosine_is_on_the_cosine_scale(cls, data):
    """Distances are estimates, but they have to be estimates of the right thing.

    The PQ family normalises and then runs ADC in that space, where the raw sum
    is ``||q - v||^2 = 2 (1 - cos)``. Halving it is the crate's job; this is the
    guard that it still happens. A `m` and codebook large enough to make the
    codec near-lossless, so what is left is the scale rather than the codec.
    """
    metrics = pytest.importorskip("sklearn.metrics.pairwise")

    # `n_pq_centroids` is the PQ family and nothing else. `m` alone would also
    # catch HnswSq8uIndex, where it means graph connectivity. Sixteen subspaces
    # over 128 centroids on 400 points is effectively lossless, which is what
    # leaves only the scale under test.
    is_pq = "n_pq_centroids" in cls(metric="cosine").get_params()
    kwargs = {"m": 16, "n_pq_centroids": 128} if is_pq else {}
    index = cls(n_neighbors=5, metric="cosine", **kwargs).fit(data)

    queries = data[:40]
    got, found = index.kneighbors(queries)
    expected = np.take_along_axis(metrics.cosine_distances(queries, data), found, 1)

    # Only the entries far enough from zero to tell a factor of two apart.
    mask = expected > 0.05
    assert mask.any(), "fixture is degenerate: every neighbour is at distance ~0"
    assert np.median(got[mask] / expected[mask]) == pytest.approx(1.0, abs=0.15)


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
