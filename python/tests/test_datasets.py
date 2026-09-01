"""Synthetic generators, and that they agree with the Rust side."""

import numpy as np
import pytest

import ann_search as ann
from ann_search import datasets

###########
# Globals #
###########

#: Checksums emitted by `synthetic::tests::test_print_cross_language_checksums`
#: in the Rust crate. Pinned here because matching them is the whole point: a
#: Python benchmark and a `cargo run --example gridsearch_*` run have to see
#: identical points for their numbers to be comparable. Regenerate with
#: `cargo test --release --features synthetic synthetic:: -- --nocapture`.
RUST_CHECKSUMS: dict[str, float] = {
    "clustered": -31.3424437950,
    # Re-pinned 2026-09-01 from `test_print_cross_language_checksums`. Both
    # drifted from deliberate generator changes, not from a regression:
    # `correlated` when bridging was added to it in 771c6ae, `low_rank` when
    # DEFAULT_TRAJECTORY_FRACTION went 0.15 -> 0.2.
    "correlated": -9.4556382021,
    "low_rank": 14.4048010631,
    "cell": 199.4125903296,
}

GENERATORS = {
    "clustered": lambda: datasets.make_clustered(500, 16, 4, seed=42),
    "correlated": lambda: datasets.make_correlated(500, 16, 4, seed=42),
    "low_rank": lambda: datasets.make_low_rank(500, 16, 4, intrinsic_dim=8, seed=42),
    "cell": lambda: datasets.make_cell_embeddings(500, 16, 4, seed=42),
}


def checksum(x: np.ndarray) -> float:
    """Mirror of the Rust-side digest. Order-sensitive on purpose."""
    n, dim = x.shape
    weights = (np.arange(n * dim) % 97 + 1).astype(np.float64)
    return float((x.astype(np.float64).ravel() * weights).sum() / (n * dim))


###################
# Cross-language  #
###################


@pytest.mark.parametrize("name", GENERATORS)
def test_matches_the_rust_checksum(name):
    x, _ = GENERATORS[name]()
    assert checksum(x) == pytest.approx(RUST_CHECKSUMS[name], abs=1e-6)


##########
# Shapes #
##########


@pytest.mark.parametrize("name", GENERATORS)
def test_shape_dtype_and_labels(name):
    x, labels = GENERATORS[name]()
    assert x.shape == (500, 16)
    assert labels.shape == (500,)
    assert x.dtype == np.float32
    assert labels.dtype == np.int64
    assert x.flags["C_CONTIGUOUS"]
    assert np.isfinite(x).all()
    assert set(np.unique(labels)) <= set(range(4))


@pytest.mark.parametrize("name", GENERATORS)
def test_reproducible_for_a_fixed_seed(name):
    a, la = GENERATORS[name]()
    b, lb = GENERATORS[name]()
    assert np.array_equal(a, b)
    assert np.array_equal(la, lb)


def test_seed_changes_the_draw():
    a, _ = datasets.make_clustered(500, 16, 4, seed=42)
    b, _ = datasets.make_clustered(500, 16, 4, seed=7)
    assert not np.array_equal(a, b)


#############
# Modality  #
#############


def participation_ratio(x: np.ndarray) -> float:
    """How many dimensions actually carry the variance."""
    v = x.var(axis=0)
    return float(v.sum() ** 2 / (v**2).sum())


def test_modalities_differ_in_anisotropy():
    # The four exist because they stress different things. Cell embeddings
    # carry rogue high-variance dimensions and low-rank data lives on a
    # manifold, so both concentrate their spectrum far more than plain
    # clusters. If this ordering ever flips, a generator is broken.
    plain = participation_ratio(datasets.make_clustered(2000, 32, 8, seed=42)[0])
    low = participation_ratio(
        datasets.make_low_rank(2000, 32, 8, intrinsic_dim=8, seed=42)[0]
    )
    cell = participation_ratio(datasets.make_cell_embeddings(2000, 32, 8, seed=42)[0])
    assert cell < low < plain


##############
# Validation #
##############


def test_intrinsic_dim_cannot_exceed_dim():
    with pytest.raises(ValueError, match="cannot exceed dim"):
        datasets.make_low_rank(100, 8, 2, intrinsic_dim=16)


@pytest.mark.parametrize("bad", [-0.5, 1.5])
def test_cor_strength_is_bounded(bad):
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        datasets.make_correlated(100, 8, 2, cor_strength=bad)


##############
# Subsample  #
##############


def test_subsample_shape_and_perturbation():
    x, _ = datasets.make_clustered(500, 16, 4, seed=42)
    q = datasets.subsample_queries(x, 100, seed=42)
    assert q.shape == (100, 16)
    assert q.dtype == np.float32
    # Perturbed, so no query is an exact copy of a source row.
    assert not any(np.equal(x, row).all(axis=1).any() for row in q)


def test_subsample_is_capped_at_the_source_size():
    x, _ = datasets.make_clustered(50, 8, 2, seed=1)
    assert datasets.subsample_queries(x, 500, seed=1).shape == (50, 8)


def test_subsample_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        datasets.subsample_queries(np.zeros(10, dtype=np.float32), 5)


###############
# Integration #
###############


def test_datasets_feed_the_indices():
    x, _ = datasets.make_clustered(2000, 32, 5, seed=42)
    q = datasets.subsample_queries(x, 200, seed=42)

    truth = ann.ExhaustiveIndex(n_neighbors=10).fit(x).kneighbors(q)[1]
    found = ann.HnswIndex(n_neighbors=10).fit(x).kneighbors(q)[1]

    hits = sum(len(set(a) & set(b)) for a, b in zip(truth, found, strict=True))
    assert hits / truth.size >= 0.95
