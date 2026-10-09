"""Build and query functions per (library, method), at matched parameters.

Each builder takes the training data, thread count and metric, and returns a
query function plus the grid its search knob is swept over. Libraries are
imported inside their builder so a worker process only ever loads the one it
measures.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from beartype import beartype

###########
# Globals #
###########

K: int = 10
HNSW_M: int = 16
HNSW_EFC: int = 200
EF_GRID: list[int] = [10, 20, 40, 80, 160, 320, 640]
ANNOY_TREES: int = 50
SEARCH_K_GRID: list[int] = [K * ANNOY_TREES * f for f in (1, 2, 5, 10, 20, 50, 100)]
NPROBE_GRID: list[int] = [2, 4, 8, 16, 32, 64, 128]
NND_K_GRAPH: int = 30
EPSILON_GRID: list[float] = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

QueryFn = Callable[[np.ndarray, int | float | None], np.ndarray]
Built = tuple[QueryFn, list[int | float | None]]

#########
# Utils #
#########


@beartype
def nlist_for(n: int) -> int:
    """IVF cell count shared by ann-search and faiss, the crate's sqrt(n)."""
    return int(np.sqrt(n))


@beartype
def normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, for faiss's inner-product cosine."""
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


##############
# ann-search #
##############


@beartype
def ann_search_builder(
    cls_name: str, knob: str | None, grid: list, **params: object
) -> Callable:
    """Builder for one ann-search index class with an optional search knob."""

    @beartype
    def build(train: np.ndarray, threads: int, metric: str) -> Built:
        import ann_search

        ann_search.set_num_threads(threads)
        index = getattr(ann_search, cls_name)(metric=metric, **params).fit(train)

        @beartype
        def query(x: np.ndarray, p: int | float | None) -> np.ndarray:
            overrides = {} if knob is None else {knob: p}
            return index.kneighbors(
                x, n_neighbors=K, return_distance=False, **overrides
            )

        return query, grid

    return build


###########
# hnswlib #
###########


@beartype
def hnswlib_hnsw(train: np.ndarray, threads: int, metric: str) -> Built:
    """hnswlib at M = 16, ef_construction = 200."""
    import hnswlib

    space = "l2" if metric == "euclidean" else "cosine"
    index = hnswlib.Index(space=space, dim=train.shape[1])
    index.init_index(max_elements=train.shape[0], ef_construction=HNSW_EFC, M=HNSW_M)
    index.add_items(train, num_threads=threads)

    @beartype
    def query(x: np.ndarray, ef: int | float | None) -> np.ndarray:
        index.set_ef(max(int(ef), K))
        return index.knn_query(x, k=K, num_threads=threads)[0]

    return query, EF_GRID


#########
# faiss #
#########


@beartype
def faiss_metric(metric: str) -> int:
    """faiss metric constant; cosine runs as inner product on normalised rows."""
    import faiss

    return faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT


@beartype
def faiss_prep(x: np.ndarray, metric: str) -> np.ndarray:
    """Normalise for cosine, otherwise hand the data through untouched."""
    return x if metric == "euclidean" else np.ascontiguousarray(normalise(x))


@beartype
def faiss_exhaustive(train: np.ndarray, threads: int, metric: str) -> Built:
    """faiss IndexFlat, the exact baseline."""
    import faiss

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlat(train.shape[1], faiss_metric(metric))
    index.add(faiss_prep(train, metric))

    @beartype
    def query(x: np.ndarray, _: int | float | None) -> np.ndarray:
        return index.search(faiss_prep(x, metric), K)[1]

    return query, [None]


@beartype
def faiss_hnsw(train: np.ndarray, threads: int, metric: str) -> Built:
    """faiss IndexHNSWFlat at M = 16, efConstruction = 200."""
    import faiss

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(train.shape[1], HNSW_M, faiss_metric(metric))
    index.hnsw.efConstruction = HNSW_EFC
    index.add(faiss_prep(train, metric))

    @beartype
    def query(x: np.ndarray, ef: int | float | None) -> np.ndarray:
        index.hnsw.efSearch = max(int(ef), K)
        return index.search(faiss_prep(x, metric), K)[1]

    return query, EF_GRID


@beartype
def faiss_ivf(train: np.ndarray, threads: int, metric: str) -> Built:
    """faiss IndexIVFFlat with the same nlist as ann-search, default training."""
    import faiss

    faiss.omp_set_num_threads(threads)
    data = faiss_prep(train, metric)
    m = faiss_metric(metric)
    quantiser = faiss.IndexFlat(train.shape[1], m)
    index = faiss.IndexIVFFlat(quantiser, train.shape[1], nlist_for(train.shape[0]), m)
    index.train(data)
    index.add(data)

    @beartype
    def query(x: np.ndarray, nprobe: int | float | None) -> np.ndarray:
        index.nprobe = int(nprobe)
        return index.search(faiss_prep(x, metric), K)[1]

    return query, NPROBE_GRID


#########
# annoy #
#########


@beartype
def annoy_annoy(train: np.ndarray, threads: int, metric: str) -> Built:
    """Spotify Annoy at 50 trees.

    The Python API queries one vector at a time. Threads split the queries into
    chunks; whether that scales depends on annoy releasing the GIL.
    """
    from annoy import AnnoyIndex

    index = AnnoyIndex(
        train.shape[1], "euclidean" if metric == "euclidean" else "angular"
    )
    index.set_seed(42)
    for i, row in enumerate(train):
        index.add_item(i, row)
    index.build(ANNOY_TREES, n_jobs=threads)

    @beartype
    def chunk(x: np.ndarray, search_k: int) -> np.ndarray:
        return np.array([index.get_nns_by_vector(v, K, search_k=search_k) for v in x])

    @beartype
    def query(x: np.ndarray, search_k: int | float | None) -> np.ndarray:
        if threads == 1:
            return chunk(x, int(search_k))
        with ThreadPoolExecutor(threads) as pool:
            parts = pool.map(
                lambda c: chunk(c, int(search_k)), np.array_split(x, threads)
            )
            return np.vstack(list(parts))

    return query, SEARCH_K_GRID


###############
# pynndescent #
###############


@beartype
def pynndescent_nndescent(train: np.ndarray, threads: int, metric: str) -> Built:
    """pynndescent at n_neighbors = 30. `prepare()` counts towards the build."""
    from pynndescent import NNDescent

    index = NNDescent(train, metric=metric, n_neighbors=NND_K_GRAPH, n_jobs=threads)
    index.prepare()

    @beartype
    def query(x: np.ndarray, eps: int | float | None) -> np.ndarray:
        return index.query(x, k=K, epsilon=float(eps))[0]

    return query, EPSILON_GRID


############
# Registry #
############

METHODS: dict[str, Callable[[np.ndarray, int, str], Built]] = {
    "ann_search:exhaustive": ann_search_builder("ExhaustiveIndex", None, [None]),
    "ann_search:hnsw": ann_search_builder(
        "HnswIndex", "ef_search", EF_GRID, m=HNSW_M, ef_construction=HNSW_EFC
    ),
    "ann_search:annoy": ann_search_builder(
        "AnnoyIndex", "search_budget", SEARCH_K_GRID, n_trees=ANNOY_TREES
    ),
    "ann_search:ivf": ann_search_builder("IvfIndex", "nprobe", NPROBE_GRID),
    "ann_search:nndescent": ann_search_builder(
        "NNDescentIndex", "ef_search", EF_GRID, n_neighbors=NND_K_GRAPH
    ),
    "faiss:exhaustive": faiss_exhaustive,
    "hnswlib:hnsw": hnswlib_hnsw,
    "faiss:hnsw": faiss_hnsw,
    "annoy:annoy": annoy_annoy,
    "faiss:ivf": faiss_ivf,
    "pynndescent:nndescent": pynndescent_nndescent,
}
