"""Shared fixtures."""

import numpy as np
import pytest

import ann_search as ann

###########
# Globals #
###########

#: Every estimator, with the per-algorithm recall floor its defaults should
#: clear on `clustered` at k=10. Exact indices are pinned at 1.0.
APPROXIMATE: dict[type, float] = {
    ann.AnnoyIndex: 0.90,
    ann.HnswIndex: 0.95,
    ann.IvfIndex: 0.70,
    ann.NNDescentIndex: 0.90,
    ann.NsgIndex: 0.90,
    ann.VamanaIndex: 0.90,
}

EXACT: dict[type, float] = {
    ann.ExhaustiveIndex: 1.0,
    ann.KmknnIndex: 1.0,
}

ALL_INDICES: dict[type, float] = {**EXACT, **APPROXIMATE}


@pytest.fixture(scope="session")
def clustered() -> np.ndarray:
    """Three well-separated gaussian blobs, float32."""
    rng = np.random.default_rng(0)
    blobs = [rng.standard_normal((700, 32)) + centre for centre in (0.0, 6.0, -6.0)]
    return np.concatenate(blobs).astype(np.float32)


@pytest.fixture(scope="session")
def truth(clustered: np.ndarray) -> np.ndarray:
    """Exact self-kNN indices at k=10, for recall assertions."""
    return ann.ExhaustiveIndex(n_neighbors=10).fit(clustered).kneighbors()[1]


def recall(found: np.ndarray, expected: np.ndarray) -> float:
    """Fraction of the true neighbours that were retrieved."""
    hits = sum(len(set(a) & set(b)) for a, b in zip(found, expected, strict=True))
    return hits / found.size
