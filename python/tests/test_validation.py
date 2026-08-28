"""Input handling and error paths."""

import numpy as np
import pytest

import ann_search as ann


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.default_rng(2).standard_normal((300, 12)).astype(np.float32)


def test_fortran_order_is_copied_not_rejected(data):
    ind = ann.HnswIndex(n_neighbors=5).fit(np.asfortranarray(data)).kneighbors()[1]
    assert ind.shape == (300, 5)


def test_strided_view_is_copied_not_rejected(data):
    ind = ann.HnswIndex(n_neighbors=5).fit(data[:, ::2]).kneighbors()[1]
    assert ind.shape == (300, 5)


def test_integer_input_promotes_to_float64():
    x = np.arange(60).reshape(20, 3)
    assert ann.ExhaustiveIndex(n_neighbors=3).fit(x).kneighbors()[0].dtype == np.float64


def test_query_dtype_is_coerced_to_the_index(data):
    index = ann.HnswIndex(n_neighbors=5).fit(data)
    assert index.kneighbors(data[:10].astype(np.float64))[0].dtype == np.float32


def test_non_finite_input_rejected():
    with pytest.raises(ValueError, match="NaN or infinite"):
        ann.HnswIndex().fit(np.array([[1.0, np.nan], [0.0, 1.0]]))


def test_non_numeric_input_rejected():
    with pytest.raises(TypeError, match="must hold numbers"):
        ann.HnswIndex().fit(np.array([["a", "b"]]))


@pytest.mark.parametrize("bad", [np.zeros((5,)), np.zeros((2, 2, 2))])
def test_wrong_rank_rejected(bad):
    with pytest.raises(ValueError, match="must be 2-D"):
        ann.HnswIndex().fit(bad)


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ann.HnswIndex().fit(np.zeros((0, 3)))


def test_query_width_must_match(data):
    with pytest.raises(ValueError, match="features but the index was fitted"):
        ann.HnswIndex().fit(data).kneighbors(data[:, :4])


def test_query_before_fit(data):
    with pytest.raises(ann.NotFittedError, match="not fitted"):
        ann.HnswIndex().kneighbors(data)


def test_unknown_search_knob_rejected(data):
    with pytest.raises(TypeError, match="unexpected search argument"):
        ann.HnswIndex().fit(data).kneighbors(data[:5], nprobe=3)


def test_unknown_soar_rule_rejected(data):
    with pytest.raises(ValueError, match="unknown SOAR rule"):
        ann.SoarIndex(rule="nonsense").fit(data)


@pytest.mark.parametrize("rule", ["nearest", "orthogonal", "shifted"])
def test_soar_rules_all_build(rule, data):
    idx = ann.SoarIndex(n_neighbors=5, rule=rule).fit(data)
    assert idx.kneighbors()[1].shape == (len(data), 5)
