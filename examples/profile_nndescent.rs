//! Profile the NN-Descent build phase on a real dataset.
//!
//! Reads a row-major `f32` blob (the same bytes handed to PyNNDescent for the
//! head-to-head), builds with `diversify_prob = 0` so what comes out is the raw
//! kNN graph rather than a pruned search graph, and reports the per-phase
//! breakdown plus recall of the descent graph against exhaustive ground truth.
//!
//! ```bash
//! cargo run --release --example profile_nndescent -- \
//!     --data /path/to/pca_100000_32.f32.bin --dim 32 --k 30
//! ```

use std::fs;
use std::time::Instant;

use ann_search_rs::{
    build_exhaustive_index, build_nndescent_index, extract_nndescent_knn, query_exhaustive_index,
};
use clap::Parser;
use rustc_hash::FxHashSet;

/// Number of held-out rows used to estimate recall against exhaustive truth.
///
/// Exhaustive search is `O(n)` per probe, so this stays small: 2000 probes is
/// enough to separate a 0.90 graph from a 0.95 one, and costs a fraction of the
/// build it is measuring.
const RECALL_PROBES: usize = 2_000;

/// CLI for the profiler.
#[derive(Parser, Debug)]
#[command(about = "Per-phase profile of the NN-Descent build on real data")]
struct Cli {
    /// Path to a row-major f32 binary blob of shape (n, dim)
    #[arg(long)]
    data: String,

    /// Dimensionality of each row
    #[arg(long, default_value_t = 32)]
    dim: usize,

    /// Neighbours per node
    #[arg(long, default_value_t = 30)]
    k: usize,

    /// Distance metric
    #[arg(long, default_value = "euclidean")]
    distance: String,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: usize,

    /// Trees in the RP forest used for initialisation (None = crate default)
    #[arg(long)]
    n_trees: Option<usize>,

    /// Candidates sampled per node per iteration (None = crate default)
    #[arg(long)]
    max_candidates: Option<usize>,

    /// Skip the recall check and only report timings
    #[arg(long, default_value_t = false)]
    no_recall: bool,
}

/// Read a row-major `f32` blob and return it with its row count.
///
/// ### Params
///
/// * `path` - File holding `n * dim` little-endian `f32` values
/// * `dim` - Row length
///
/// ### Returns
///
/// The flattened data and the number of rows.
fn read_blob(path: &str, dim: usize) -> (Vec<f32>, usize) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert!(
        bytes.len() % (4 * dim) == 0,
        "{path}: {} bytes is not a whole number of {dim}-wide f32 rows",
        bytes.len()
    );

    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = values.len() / dim;

    assert!(
        values.iter().all(|v| v.is_finite()),
        "{path} contains non-finite values"
    );

    (values, n)
}

/// Mean recall@k of the descent graph against exhaustive ground truth.
///
/// Probes a strided sample of rows rather than all of them, and drops the
/// self-match from the exhaustive result so both sides describe the same thing.
///
/// ### Params
///
/// * `data` - Flattened row-major dataset
/// * `n` - Number of rows
/// * `dim` - Row length
/// * `graph` - Descent graph, one row of neighbours per node
/// * `k` - Neighbours to compare
/// * `distance` - Metric name
///
/// ### Returns
///
/// Mean recall in `[0, 1]`.
fn graph_recall(
    data: &[f32],
    n: usize,
    dim: usize,
    graph: &[Vec<usize>],
    k: usize,
    distance: &str,
) -> f64 {
    let stride = (n / RECALL_PROBES).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(RECALL_PROBES).collect();

    let mut queries = Vec::with_capacity(probes.len() * dim);
    for &i in &probes {
        queries.extend_from_slice(&data[i * dim..(i + 1) * dim]);
    }

    let exhaustive = build_exhaustive_index((data, n, dim), distance);
    let (truth, _) = query_exhaustive_index(
        (&queries[..], probes.len(), dim),
        &exhaustive,
        k + 1,
        false,
        false,
    )
    .expect("exhaustive query failed");

    let mut total = 0.0;
    for (probe_pos, &i) in probes.iter().enumerate() {
        let true_set: FxHashSet<usize> = truth[probe_pos]
            .iter()
            .copied()
            .filter(|&j| j != i)
            .take(k)
            .collect();
        let approx: FxHashSet<usize> = graph[i].iter().copied().filter(|&j| j != i).collect();
        total += approx.intersection(&true_set).count() as f64 / true_set.len().max(1) as f64;
    }

    total / probes.len() as f64
}

fn main() {
    let cli = Cli::parse();
    let (data, n) = read_blob(&cli.data, cli.dim);

    println!("Data: {} x {} f32 from {}", n, cli.dim, cli.data);
    println!("Threads: {}", rayon::current_num_threads());

    let start = Instant::now();
    let index = build_nndescent_index(
        (&data[..], n, cli.dim),
        &cli.distance,
        0.001f32,
        // Pure kNN generation: diversify is query prep, and PyNNDescent does
        // its equivalent pruning lazily in `_init_search_graph`, so including
        // it here would compare different things.
        0.0f32,
        Some(cli.k),
        None,
        cli.max_candidates,
        cli.n_trees,
        cli.seed,
        true,
    )
    .expect("build failed");
    let wall = start.elapsed();

    println!(
        "\nWall clock (build_nndescent_index): {:.3} s",
        wall.as_secs_f64()
    );
    println!("Converged: {}", index.index_converged());

    if !cli.no_recall {
        let (graph, _) = extract_nndescent_knn(&index, Some(cli.k), false).expect("extract failed");
        let recall = graph_recall(&data, n, cli.dim, &graph, cli.k, &cli.distance);
        println!("Raw graph recall@{}: {:.4}", cli.k, recall);
    }
}
