"""Time to a full self-kNN graph on real single-cell PCA, the crate's use case.

Every cell is a query against all cells, so this is build plus self-query timed
end to end. Recall is scored on the 10,000 cells the data file ships exact
ground truth for. Every method runs in its own process.

    uv run selfknn.py run --threads 10
    uv run selfknn.py summarise
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import click
import h5py
import numpy as np
import polars as pl
from beartype import beartype

from bench import CACHE, LIBRARY_STYLE, MIB, REPO, footprint, run_info

###########
# Globals #
###########

HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE / "results" / "single_cell"
DATASET: str = "tahoe-pca-32-euclidean"
# TODO(release): set once the data release exists.
DATA_URL: str | None = None
K: int = 15
RECALL_TARGETS: tuple[float, ...] = (0.9, 0.95, 0.99)

# One point per (build, knob) pair: (param label, build seconds, self seconds,
# the (n, K) id matrix).
Point = tuple[str, float, float, np.ndarray]
Method = Callable[[np.ndarray, int], list[Point]]

########
# Data #
########


@beartype
def data_path() -> Path:
    """Local copy of the data file, downloaded to `CACHE` once."""
    path = CACHE.parent / "single-cell" / f"{DATASET}.hdf5"
    if not path.exists():
        if DATA_URL is None:
            raise FileNotFoundError(f"{path} missing and no DATA_URL to fetch it")
        import shutil
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(DATA_URL) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(path)
    return path


@beartype
def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells, the ground-truth cell ids and their exact top-K, self included."""
    with h5py.File(data_path(), "r") as f:
        x = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        return x, f["test_ids"][:], f["neighbors"][:, :K]


@beartype
def recall(found: np.ndarray, test_ids: np.ndarray, truth: np.ndarray) -> float:
    """Id-set recall@K over the ground-truth rows of a self-kNN graph."""
    rows = np.asarray(found)[test_ids, :K]
    return float((rows[:, :, None] == truth[:, None, :]).any(axis=2).mean())


###########
# Methods #
###########


@beartype
def timed(f: Callable[[], object]) -> tuple[float, object]:
    """Wall time of one call, plus its result."""
    t0 = time.perf_counter()
    out = f()
    return time.perf_counter() - t0, out


@beartype
def ann_search_sweep(cls_name: str, knob: str | None, grid: list, **params: object):
    """Build once, then one self-query per grid value."""

    @beartype
    def method(x: np.ndarray, threads: int) -> list[Point]:
        import ann_search

        ann_search.set_num_threads(threads)
        build_s, index = timed(lambda: getattr(ann_search, cls_name)(**params).fit(x))
        points = []
        for p in grid:
            kw = {} if knob is None else {knob: p}
            s, ids = timed(
                lambda kw=kw: index.kneighbors(
                    n_neighbors=K, return_distance=False, **kw
                )
            )
            points.append(("-" if p is None else str(p), build_s, s, ids))
        return points

    return method


@beartype
def ann_search_extract(cls_name: str, degrees: list[int], degree_param: str):
    """NN-Descent paths: the graph the build converged on, no search at all.

    The descent graph is built at each degree and its first K columns read
    back, which is what a single-cell pipeline does with it.
    """

    @beartype
    def method(x: np.ndarray, threads: int) -> list[Point]:
        import ann_search

        ann_search.set_num_threads(threads)
        points = []
        for deg in degrees:
            build_s, index = timed(
                lambda deg=deg: getattr(ann_search, cls_name)(
                    **{degree_param: deg}
                ).fit(x)
            )
            s, ids = timed(lambda i=index: i.extract_knn(K, return_distance=False))
            points.append((f"deg{deg}", build_s, s, ids))
        return points

    return method


@beartype
def pynndescent_graph(x: np.ndarray, threads: int) -> list[Point]:
    """pynndescent's `neighbor_graph` at n_neighbors = K: what scanpy runs."""
    from pynndescent import NNDescent

    build_s, index = timed(lambda: NNDescent(x, n_neighbors=K, n_jobs=threads))
    return [("-", build_s, 0.0, index.neighbor_graph[0])]


@beartype
def hnswlib_self(x: np.ndarray, threads: int) -> list[Point]:
    """hnswlib at M = 16, ef_construction = 100, every cell queried."""
    import hnswlib

    def build():
        index = hnswlib.Index(space="l2", dim=x.shape[1])
        index.init_index(max_elements=x.shape[0], ef_construction=100, M=16)
        index.add_items(x, num_threads=threads)
        return index

    build_s, index = timed(build)
    points = []
    for ef in (16, 32, 64, 128):
        index.set_ef(max(ef, K))
        s, ids = timed(lambda: index.knn_query(x, k=K, num_threads=threads)[0])
        points.append((str(ef), build_s, s, ids))
    return points


@beartype
def faiss_ivf_self(x: np.ndarray, threads: int) -> list[Point]:
    """faiss IndexIVFFlat at nlist = sqrt(n), every cell queried."""
    import faiss

    faiss.omp_set_num_threads(threads)

    def build():
        q = faiss.IndexFlatL2(x.shape[1])
        index = faiss.IndexIVFFlat(q, x.shape[1], int(np.sqrt(x.shape[0])))
        index.train(x)
        index.add(x)
        return index

    build_s, index = timed(build)
    points = []
    for nprobe in (4, 8, 16, 32):
        index.nprobe = nprobe
        s, ids = timed(lambda: index.search(x, K)[1])
        points.append((str(nprobe), build_s, s, ids))
    return points


METHODS: dict[str, Method] = {
    "ann_search:hnsw": ann_search_sweep(
        "HnswIndex", "ef_search", [16, 32, 64, 128], m=16, ef_construction=100
    ),
    "ann_search:ivf": ann_search_sweep("IvfIndex", "nprobe", [4, 8, 16, 32]),
    "ann_search:nndescent": ann_search_extract(
        "NNDescentIndex", [15, 30], "n_neighbors"
    ),
    "ann_search_gpu:ivf": ann_search_sweep("IvfGpuIndex", "nprobe", [4, 8, 16, 32]),
    "ann_search_gpu:nndescent": ann_search_extract(
        "CagraGpuIndex", [15, 30], "graph_degree"
    ),
    "ann_search_mlx:exhaustive": ann_search_sweep("ExhaustiveMlxIndex", None, [None]),
    "ann_search_mlx:ivf": ann_search_sweep("IvfMlxIndex", "nprobe", [4, 8, 16, 32]),
    "ann_search_mlx:nndescent": ann_search_extract(
        "CagraMlxIndex", [15, 30], "graph_degree"
    ),
    "pynndescent:nndescent": pynndescent_graph,
    "hnswlib:hnsw": hnswlib_self,
    "faiss:ivf": faiss_ivf_self,
}

##########
# Worker #
##########


@click.command(hidden=True)
@click.option("--threads", type=int, required=True)
@click.option("--entry", required=True)
@click.option("--n", type=int, default=None)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@beartype
def worker(threads: int, entry: str, n: int | None, out: Path) -> None:
    """Run one method on the cells and write its points to `out`."""
    library, method = entry.split(":")
    x, test_ids, truth = load()
    if n is not None:
        # Smoke runs only: the ground truth is for the full set, so recall is
        # meaningless here and comes out as zero.
        keep = np.union1d(test_ids, np.arange(n))
        x = x[keep]
        test_ids = np.searchsorted(keep, test_ids)
        truth = np.full_like(truth, -1)
    # Imports, numba JIT and kernel compilation happen here, untimed.
    toy = np.random.default_rng(0).standard_normal((5_000, x.shape[1]))
    METHODS[entry](toy.astype(np.float32), threads)
    base, _ = footprint()
    points = METHODS[entry](x, threads)
    peak = footprint()[1] - base
    rows = []
    for param, build_s, self_s, ids in points:
        r = recall(ids, test_ids, truth)
        rows.append(
            {
                "library": library,
                "method": method,
                "param": param,
                "build_s": build_s,
                "self_s": self_s,
                "total_s": build_s + self_s,
                "peak_mib": peak / MIB,
                "recall": r,
            }
        )
        click.echo(
            f"  {entry:26s} {param:>6s} build {build_s:7.1f}s self {self_s:7.1f}s "
            f"recall {r:.4f}",
            err=True,
        )
    pl.DataFrame(rows).write_csv(out)


#######
# Run #
#######


@click.command()
@click.option("--threads", type=int, default=10, show_default=True)
@click.option("--only", multiple=True, type=click.Choice(sorted(METHODS)))
@click.option("--n", type=int, default=None, help="Subsample, for smoke runs.")
@beartype
def run(threads: int, only: tuple[str, ...], n: int | None) -> None:
    """Run every method, one process each."""
    import os

    stem = RESULTS if n is None else RESULTS.parent / "single_cell_smoke"
    stem.mkdir(parents=True, exist_ok=True)
    data_path()
    info = "run_info_partial.json" if only else "run_info.json"
    (stem / info).write_text(json.dumps(run_info(), indent=2))
    env = os.environ | {
        k: str(threads)
        for k in (
            "OMP_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "RAYON_NUM_THREADS",
        )
    }
    for entry in only or METHODS:
        click.echo(f"self-kNN t{threads}: {entry}", err=True)
        cmd = [sys.executable, str(HERE / "selfknn.py"), "worker"]
        cmd += ["--threads", str(threads), "--entry", entry]
        cmd += ["--out", str(stem / f"{entry.replace(':', '_')}.csv")]
        if n is not None:
            cmd += ["--n", str(n)]
        subprocess.run(cmd, env=env, check=True)


#############
# Summarise #
#############


@beartype
def to_markdown(sweep: pl.DataFrame) -> str:
    """Fastest build-plus-self-kNN time per method at each recall target."""
    summary = (
        sweep.group_by("library", "method")
        .agg(
            pl.col("peak_mib").first(),
            *[
                pl.col("total_s").filter(pl.col("recall") >= t).min().alias(str(t))
                for t in RECALL_TARGETS
            ],
        )
        .sort(str(RECALL_TARGETS[0]), nulls_last=True)
    )
    header = (
        "| Method | Library | Peak (MiB) | "
        + " | ".join(f"Total (s) @ {t:.2f}" for t in RECALL_TARGETS)
        + " |"
    )
    lines = [header, "|---|---|" + "---:|" * (1 + len(RECALL_TARGETS))]
    for r in summary.iter_rows(named=True):
        lib = r["library"]
        if lib.startswith("ann_search"):
            lib = f"**{lib}**"
        cells = [
            "n/a" if r[str(t)] is None else f"{r[str(t)]:.1f}" for t in RECALL_TARGETS
        ]
        lines.append(
            f"| {r['method']} | {lib} | {r['peak_mib']:,.0f} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


@beartype
def plot(sweep: pl.DataFrame, out: Path) -> None:
    """Recall against total time, one line per (library, method)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (library, method), pts in sorted(
        sweep.group_by("library", "method"), key=lambda g: g[0]
    ):
        pts = pts.sort("total_s")
        colour, marker = LIBRARY_STYLE[library]
        ax.plot(
            pts["total_s"],
            pts["recall"],
            color=colour,
            marker=marker,
            linestyle={"ivf": "-", "hnsw": "--", "nndescent": ":"}.get(method, "-"),
            label=f"{library} {method}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("build + self-kNN, seconds (log)")
    ax.set_ylabel(f"recall@{K}")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


@click.command()
@click.option(
    "--template",
    type=click.Path(path_type=Path),
    default=REPO / "docs" / "templates" / "benchmarks_single_cell.md.tmpl",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=REPO / "docs" / "benchmarks_single_cell.md",
)
@beartype
def summarise(template: Path, output: Path) -> None:
    """Fill the single-cell doc from results/single_cell/."""
    sweep = pl.concat(
        [
            pl.read_csv(c, schema_overrides={"param": pl.String})
            for c in sorted(RESULTS.glob("*.csv"))
        ]
    )
    plot(sweep, output.parent / "figures" / "single_cell_self_knn.png")
    table = to_markdown(sweep)
    click.echo(table, err=True)
    infos = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("run_info*"))]
    i = infos[0]
    commits = sorted({x["commit"] for x in infos})
    note = "" if len(commits) == 1 else f" Runs span commits {', '.join(commits)}."
    doc = template.read_text().replace("<!-- BENCH:single_cell -->", table)
    doc = doc.replace(
        "<!-- RUNINFO -->",
        f"*ann-search-rs {i['version']} (commit {i['commit']}), run on {i['date']} "
        f"on {i['cpu']}.{note}*",
    )
    output.write_text(doc)
    click.echo(f"Generated: {output}", err=True)


@click.group()
def cli() -> None:
    """Self-kNN on single-cell PCA."""


cli.add_command(run)
cli.add_command(worker)
cli.add_command(summarise)

if __name__ == "__main__":
    cli()
