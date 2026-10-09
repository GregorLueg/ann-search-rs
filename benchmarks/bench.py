"""ann-search against hnswlib, annoy, pynndescent and faiss on ann-benchmarks.

Mirrors the annsearchR harness: one build per library at matched parameters,
then the search knob is swept, and libraries are compared at equal recall@10,
not at equal settings. Every (library, method) runs in its own process so the
memory numbers are its own.

    uv run bench.py run --dataset fashion-mnist-784-euclidean --threads 10
    uv run bench.py summarise
"""

import ctypes
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import click
import h5py
import numpy as np
import polars as pl
import psutil
from beartype import beartype

from methods import METHODS, K

###########
# Globals #
###########

HERE: Path = Path(__file__).resolve().parent
REPO: Path = HERE.parent
RESULTS: Path = HERE / "results"
CACHE: Path = Path.home() / ".cache" / "ann-search-rs" / "ann-benchmarks"
DATASETS: tuple[str, ...] = (
    "fashion-mnist-784-euclidean",
    "sift-128-euclidean",
    "glove-100-angular",
)
RECALL_TARGETS: tuple[float, ...] = (0.9, 0.95, 0.99)
N_WARMUP: int = 100
# Median of this many timed calls per sweep point. Single calls swung 15% on
# an otherwise idle machine and did not reproduce.
N_REPS: int = 3
MIB: float = 1024.0**2
RUSAGE_INFO_V4: int = 4
LIBPROC: ctypes.CDLL | None = (
    ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    if sys.platform == "darwin"
    else None
)
# Fixed per library so colours never shift between plots.
LIBRARY_STYLE: dict[str, tuple[str, str]] = {
    "ann_search": ("#2a78d6", "o"),
    "hnswlib": ("#eb6834", "s"),
    "faiss": ("#1baf7a", "^"),
    "annoy": ("#eda100", "D"),
    "pynndescent": ("#e87ba4", "v"),
    "usearch": ("#008300", "P"),
    "ann_search_sq8": ("#4a3aa7", "X"),
    "faiss_sq8": ("#e34948", "*"),
}

########
# Data #
########


@beartype
def dataset_path(name: str) -> Path:
    """Local copy of an ann-benchmarks file, downloaded to `CACHE` once."""
    path = CACHE / f"{name}.hdf5"
    if not path.exists():
        import urllib.request

        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"http://ann-benchmarks.com/{name}.hdf5", tmp)
        tmp.rename(path)
    return path


@beartype
def load_train(name: str) -> np.ndarray:
    """The training vectors, float32 and row-major."""
    with h5py.File(dataset_path(name), "r") as f:
        return np.ascontiguousarray(f["train"][:], dtype=np.float32)


@beartype
def load_queries(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The test vectors and the first k ground-truth ids."""
    with h5py.File(dataset_path(name), "r") as f:
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        return test, f["neighbors"][:, :K]


@beartype
def metric_of(dataset: str) -> str:
    """Metric from the ann-benchmarks name suffix."""
    match dataset.rsplit("-", 1)[-1]:
        case "euclidean":
            return "euclidean"
        case "angular":
            return "cosine"
        case other:
            raise ValueError(f"unknown metric suffix '{other}'")


@beartype
def recall(truth: np.ndarray, found: np.ndarray) -> float:
    """Mean id-set recall@k against the file's ground truth."""
    hits = (found[:, :, None] == truth[:, None, :]).any(axis=2).sum()
    return float(hits / truth.size)


##########
# Memory #
##########


@beartype
def footprint() -> tuple[int, int]:
    """Current and peak memory of this process in bytes.

    On macOS the kernel's `phys_footprint`, which Activity Monitor reports.
    RSS is useless there: malloc keeps freed pages mapped and counted until
    there is memory pressure, so a freed array never leaves it. Elsewhere RSS
    and `ru_maxrss` (KiB on Linux).
    """
    if sys.platform == "darwin":
        # struct rusage_info_v4 from <sys/resource.h>: a 16-byte uuid, then 36
        # uint64 fields. phys_footprint is field 7, lifetime max field 28.
        buf = (ctypes.c_uint64 * 38)()
        if LIBPROC.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, buf) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return buf[2 + 7], buf[2 + 28]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return psutil.Process().memory_info().rss, peak


##########
# Worker #
##########


@click.command(hidden=True)
@click.option("--dataset", required=True)
@click.option("--threads", type=int, required=True)
@click.option("--entry", required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@beartype
def worker(dataset: str, threads: int, entry: str, out: Path) -> None:
    """Build one (library, method), sweep its knob, write the rows to `out`."""
    library, method = entry.split(":")
    build = METHODS[entry]
    metric = metric_of(dataset)
    test, truth = load_queries(dataset)

    # Imports, numba JIT and allocator warm-up happen here, outside the timings
    # and below the memory baseline.
    rng = np.random.default_rng(0)
    toy = rng.standard_normal((2_000, test.shape[1]), dtype=np.float32)
    toy_query, toy_grid = build(toy, threads, metric)
    toy_query(toy[:10], toy_grid[0])
    del toy, toy_query
    gc.collect()

    # Baseline before the training data loads. The index has to account for
    # the data once, whether it copies it or keeps a reference: freeing our
    # copy after the build leaves exactly that.
    baseline, _ = footprint()
    train = load_train(dataset)
    n, dim = train.shape
    t0 = time.perf_counter()
    query, grid = build(train, threads, metric)
    build_s = time.perf_counter() - t0
    build_peak = footprint()[1] - baseline
    del train
    gc.collect()
    resident = footprint()[0] - baseline

    rows = []
    for p in grid:
        query(test[:N_WARMUP], p)
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            found = query(test, p)
            times.append(time.perf_counter() - t0)
        query_s = sorted(times)[N_REPS // 2]
        rows.append(
            {
                "dataset": dataset,
                "threads": threads,
                "library": library,
                "method": method,
                "param": "-" if p is None else str(p),
                "build_s": build_s,
                "build_peak_mib": build_peak / MIB,
                "resident_mib": resident / MIB,
                "query_s": query_s,
                "qps": len(test) / query_s,
                "recall": recall(truth, np.asarray(found)),
            }
        )
        click.echo(
            f"  {entry:24s} {rows[-1]['param']:>8s} "
            f"qps {rows[-1]['qps']:>10,.0f} recall {rows[-1]['recall']:.4f}",
            err=True,
        )
    click.echo(
        f"  {entry:24s} n={n} dim={dim} build {build_s:.2f}s "
        f"peak {build_peak / MIB:,.0f} MiB resident {resident / MIB:,.0f} MiB",
        err=True,
    )
    pl.DataFrame(rows).write_csv(out)


#######
# Run #
#######


@beartype
def run_info() -> dict[str, str]:
    """Crate version of the installed binding, commit, date and machine."""
    import ann_search

    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return {
        "version": ann_search.__core_version__,
        "commit": commit,
        "date": date.today().isoformat(),
        "cpu": cpu,
    }


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS), required=True)
@click.option("--threads", type=int, default=10, show_default=True)
@click.option(
    "--only",
    multiple=True,
    type=click.Choice(sorted(METHODS)),
    help="Restrict to these library:method entries. Repeatable.",
)
@beartype
def run(dataset: str, threads: int, only: tuple[str, ...]) -> None:
    """Run every (library, method) on one dataset, one process each."""
    stem = RESULTS / f"{dataset}_t{threads}"
    stem.mkdir(parents=True, exist_ok=True)
    dataset_path(dataset)
    (stem / "run_info.json").write_text(json.dumps(run_info(), indent=2))

    # Pinned before any BLAS, OpenMP or numba runtime starts in the worker.
    env = os.environ | {
        "OMP_NUM_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "RAYON_NUM_THREADS": str(threads),
    }
    for entry in only or METHODS:
        click.echo(f"{dataset} t{threads}: {entry}", err=True)
        out = stem / f"{entry.replace(':', '_')}.csv"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "bench.py"),
                "worker",
                "--dataset",
                dataset,
                "--threads",
                str(threads),
                "--entry",
                entry,
                "--out",
                str(out),
            ],
            env=env,
            check=True,
        )


#############
# Summarise #
#############


@beartype
def summary_table(sweep: pl.DataFrame) -> pl.DataFrame:
    """Best QPS per library and method at each recall target; null if missed."""
    return (
        sweep.group_by("method", "library")
        .agg(
            pl.col("build_s").first(),
            pl.col("build_peak_mib").first(),
            pl.col("resident_mib").first(),
            *[
                pl.col("qps").filter(pl.col("recall") >= t).max().alias(f"qps@{t}")
                for t in RECALL_TARGETS
            ],
        )
        .sort(
            "method",
            f"qps@{RECALL_TARGETS[0]}",
            descending=[False, True],
            nulls_last=True,
        )
    )


@beartype
def to_markdown(summary: pl.DataFrame) -> str:
    """Render the summary as a GitHub markdown table."""
    header = (
        "| Method | Library | Build (s) | Build peak (MiB) | Index (MiB) | "
        + " | ".join(f"QPS @ {t:.2f}" for t in RECALL_TARGETS)
        + " |"
    )
    lines = [header, "|" + "---|" * 2 + "---:|" * (3 + len(RECALL_TARGETS))]
    for r in summary.iter_rows(named=True):
        lib = r["library"]
        if lib.startswith("ann_search"):
            lib = f"**{lib}**"
        qps = [r[f"qps@{t}"] for t in RECALL_TARGETS]
        lines.append(
            f"| {r['method']} | {lib} | {r['build_s']:.1f} | "
            f"{r['build_peak_mib']:,.0f} | {r['resident_mib']:,.0f} | "
            + " | ".join("n/a" if q is None else f"{q:,.0f}" for q in qps)
            + " |"
        )
    return "\n".join(lines)


@beartype
def plot_sweep(sweep: pl.DataFrame, title: str, out: Path) -> None:
    """Recall against QPS, one panel per method family, log QPS axis.

    Exact search gets no panel of its own: it is a dashed line in every panel,
    and an approximate index below it is not worth building on that data.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = sweep.filter(pl.col("method") == "exhaustive")
    methods = sorted(set(sweep["method"].unique()) - {"exhaustive"})
    fig, axes = plt.subplots(
        1, len(methods), figsize=(3.2 * len(methods), 3.4), sharey=True, squeeze=False
    )
    for ax, method in zip(axes[0], methods, strict=True):
        sub = sweep.filter(pl.col("method") == method)
        for library in sorted(sub["library"].unique()):
            pts = sub.filter(pl.col("library") == library).sort("recall")
            colour, marker = LIBRARY_STYLE[library]
            ax.plot(
                pts["recall"],
                pts["qps"],
                color=colour,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=library,
            )
        for library, qps in exact.select("library", "qps").iter_rows():
            ax.axhline(
                qps,
                color=LIBRARY_STYLE[library][0],
                linestyle="--",
                linewidth=1,
                label=f"{library} exact",
            )
        ax.set_xlim(right=1.002)
        ax.set_title(method, fontsize=10)
        ax.set_yscale("log")
        ax.set_xlabel("recall@10")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("queries per second")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


@click.command()
@click.option(
    "--template",
    type=click.Path(path_type=Path),
    default=REPO / "docs" / "templates" / "benchmarks_comparison.md.tmpl",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=REPO / "docs" / "benchmarks_comparison.md",
    show_default=True,
)
@beartype
def summarise(template: Path, output: Path) -> None:
    """Fill the comparison doc from every results/<dataset>_t<threads>/ run.

    Each `<!-- BENCH:<dataset>_t<threads> -->` marker becomes a table, the
    figure is written next to the doc. `<!-- RUNINFO -->` takes the run info
    of the first run found; runs from different commits are flagged.
    """
    doc = template.read_text()
    infos = []
    for stem in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        csvs = sorted(stem.glob("*.csv"))
        if not csvs:
            continue
        sweep = pl.concat(
            [pl.read_csv(c, schema_overrides={"param": pl.String}) for c in csvs]
        )
        infos.append(json.loads((stem / "run_info.json").read_text()))
        dataset, threads = stem.name.rsplit("_t", 1)
        plot_sweep(
            sweep,
            f"{dataset}, {threads} thread{'s' if threads != '1' else ''}",
            output.parent / "figures" / f"comparison_{stem.name}.png",
        )
        table = to_markdown(summary_table(sweep))
        click.echo(f"{stem.name}\n{table}\n", err=True)
        doc = doc.replace(f"<!-- BENCH:{stem.name} -->", table)

    if infos:
        i = infos[0]
        commits = sorted({x["commit"] for x in infos})
        note = "" if len(commits) == 1 else f" Runs span commits {', '.join(commits)}."
        doc = doc.replace(
            "<!-- RUNINFO -->",
            f"*ann-search-rs {i['version']} (commit {i['commit']}), run on {i['date']} "
            f"on {i['cpu']}.{note}*",
        )
    output.write_text(doc)
    click.echo(f"Generated: {output}", err=True)


#######
# CLI #
#######


@click.group()
def cli() -> None:
    """Cross-library CPU benchmarks for ann-search."""


cli.add_command(run)
cli.add_command(worker)
cli.add_command(summarise)

if __name__ == "__main__":
    cli()
