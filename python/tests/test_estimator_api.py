"""Persistence, the sparse graph, and the scikit-learn estimator contract."""

import pickle

import numpy as np
import pytest
from conftest import ALL_INDICES

import ann_search as ann


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return np.random.default_rng(3).standard_normal((400, 16)).astype(np.float32)


###############
# Persistence #
###############


@pytest.mark.parametrize("cls", ALL_INDICES, ids=lambda c: c.__name__)
def test_save_load_round_trip(cls, data, tmp_path):
    index = cls(n_neighbors=6).fit(data)
    want = index.kneighbors(data[:20])

    index.save(tmp_path / "idx")
    got = cls.load(tmp_path / "idx").kneighbors(data[:20])

    assert np.array_equal(want[0], got[0])
    assert np.array_equal(want[1], got[1])


@pytest.mark.parametrize("cls", ALL_INDICES, ids=lambda c: c.__name__)
def test_pickle_round_trip(cls, data):
    index = cls(n_neighbors=6).fit(data)
    want = index.kneighbors(data[:20])
    got = pickle.loads(pickle.dumps(index)).kneighbors(data[:20])

    assert np.array_equal(want[0], got[0])
    assert np.array_equal(want[1], got[1])


def test_pickle_preserves_params(data):
    index = ann.HnswIndex(n_neighbors=6, m=24, ef_search=77).fit(data)
    assert pickle.loads(pickle.dumps(index)).get_params() == index.get_params()


def test_unfitted_pickles(data):
    index = ann.HnswIndex(m=24)
    restored = pickle.loads(pickle.dumps(index))
    assert restored.get_params() == index.get_params()
    assert restored.fit(data).kneighbors()[1].shape[0] == len(data)


def test_load_rejects_the_wrong_index_type(data, tmp_path):
    ann.HnswIndex(n_neighbors=6).fit(data).save(tmp_path / "idx")
    with pytest.raises(ValueError, match="holds a HnswIndex"):
        ann.VamanaIndex.load(tmp_path / "idx")


def test_load_of_a_float64_bundle_keeps_its_dtype(data, tmp_path):
    ann.HnswIndex(n_neighbors=6).fit(data.astype(np.float64)).save(tmp_path / "idx")
    assert ann.HnswIndex.load(tmp_path / "idx").kneighbors()[0].dtype == np.float64


##########
# Graphs #
##########


def test_kneighbors_graph_shape_and_weights(data):
    index = ann.HnswIndex(n_neighbors=5).fit(data)
    graph = index.kneighbors_graph()
    assert graph.shape == (400, 400)
    assert graph.nnz == 400 * 5
    assert np.allclose(np.sort(graph[0].data), np.sort(index.kneighbors()[0][0]))


def test_connectivity_mode_is_all_ones(data):
    graph = ann.HnswIndex(n_neighbors=5).fit(data).kneighbors_graph(mode="connectivity")
    assert (graph.data == 1.0).all()


def test_graph_drops_padding():
    small = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert ann.HnswIndex(n_neighbors=9).fit(small).kneighbors_graph().nnz == 16


def test_unknown_graph_mode(data):
    with pytest.raises(ValueError, match="unknown mode"):
        ann.HnswIndex().fit(data).kneighbors_graph(mode="nope")


def test_fit_transform_is_the_self_graph(data):
    index = ann.HnswIndex(n_neighbors=5)
    assert (index.fit_transform(data) != index.kneighbors_graph()).nnz == 0


###############
# Estimator   #
###############


def test_get_set_params_round_trip():
    index = ann.HnswIndex()
    index.set_params(m=32, ef_search=200)
    assert index.get_params()["m"] == 32
    assert index.get_params()["ef_search"] == 200


def test_set_params_invalidates_the_fit(data):
    index = ann.HnswIndex(n_neighbors=5).fit(data)
    index.set_params(m=32)
    with pytest.raises(ann.NotFittedError):
        index.kneighbors()


def test_set_params_rejects_unknown_names():
    with pytest.raises(ValueError, match="invalid parameter"):
        ann.HnswIndex().set_params(nprobe=4)


def test_sklearn_clone():
    sklearn_base = pytest.importorskip("sklearn.base")
    index = ann.HnswIndex(n_neighbors=9, m=24)
    assert sklearn_base.clone(index).get_params() == index.get_params()


def test_repr_round_trips_through_eval():
    index = ann.IvfIndex(n_neighbors=9, nprobe=4)
    rebuilt = eval(repr(index), {"IvfIndex": ann.IvfIndex})
    assert rebuilt.get_params() == index.get_params()


###########
# Threads #
###########


def test_thread_control_is_reentrant():
    original = ann.num_threads()
    ann.set_num_threads(2)
    assert ann.num_threads() == 2
    ann.set_num_threads(3)
    assert ann.num_threads() == 3
    ann.set_num_threads(0)
    assert ann.num_threads() == original
