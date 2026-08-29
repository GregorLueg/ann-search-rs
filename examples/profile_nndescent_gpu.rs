//! Profile the GPU NN-Descent kNN graph build on a real dataset.
//!
//! Mirrors `profile_nndescent` on the GPU side: reads the same row-major `f32`
//! blob, builds the raw kNN graph with `build_knn_graph_gpu` and reports
//! wall-clock plus recall of the descent graph against exhaustive ground
//! truth. Optionally sweeps the RP-forest tree count.
//!
//! ```bash
//! cargo run --release --features gpu --example profile_nndescent_gpu -- \
//!     --data /path/to/pca_100000_32.f32.bin --dim 32 --k 30
//! ```

use std::fs;
use std::time::Instant;

use ann_search_rs::gpu::nndescent_gpu::default_forest_trees;
use ann_search_rs::{build_exhaustive_index, build_knn_graph_gpu, query_exhaustive_index};
use clap::Parser;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rustc_hash::FxHashSet;

/// Number of held-out rows used to estimate recall against exhaustive truth.
const RECALL_PROBES: usize = 2_000;

/// CLI for the GPU profiler.
#[derive(Parser, Debug)]
#[command(about = "Per-phase profile of the GPU NN-Descent build on real data")]
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

    /// Trees in the RP forest (None = crate default)
    #[arg(long)]
    n_trees: Option<usize>,

    /// Comma-separated tree counts to sweep instead of a single build
    #[arg(long, value_delimiter = ',')]
    tree_sweep: Option<Vec<usize>>,

    /// Internal NNDescent working degree (None = crate default)
    #[arg(long)]
    build_k: Option<usize>,

    /// Maximum NNDescent iterations (None = crate default)
    #[arg(long)]
    max_iters: Option<usize>,

    /// Local-join sampling rate (None = crate default)
    #[arg(long)]
    rho: Option<f32>,

    /// 2-hop refinement sweeps after the main loop (None = crate default)
    #[arg(long)]
    refine_knn: Option<usize>,

    /// Repeats per configuration; the best wall-clock is reported
    #[arg(long, default_value_t = 1)]
    repeats: usize,

    /// Skip the recall check and only report timings
    #[arg(long, default_value_t = false)]
    no_recall: bool,

    /// Print the per-phase verbose trace from the builder
    #[arg(long, default_value_t = false)]
    verbose: bool,
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
        bytes.len().is_multiple_of(4 * dim),
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

/// Exhaustive ground truth for a strided probe sample.
///
/// Computed once and reused across every configuration in a sweep, since the
/// probes and the data never change.
///
/// ### Params
///
/// * `data` - Flattened row-major dataset
/// * `n` - Number of rows
/// * `dim` - Row length
/// * `k` - Neighbours to retain
/// * `distance` - Metric name
///
/// ### Returns
///
/// Probe row ids and, per probe, the true neighbour set with self removed.
fn ground_truth(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    distance: &str,
) -> (Vec<usize>, Vec<FxHashSet<usize>>) {
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

    let sets = probes
        .iter()
        .enumerate()
        .map(|(pos, &i)| {
            truth[pos]
                .iter()
                .copied()
                .filter(|&j| j != i)
                .take(k)
                .collect()
        })
        .collect();

    (probes, sets)
}

/// Mean recall of a built graph against pre-computed ground truth.
///
/// ### Params
///
/// * `graph` - Descent graph, one row of neighbours per node
/// * `probes` - Row ids the truth was computed for
/// * `truth` - Per-probe true neighbour sets
///
/// ### Returns
///
/// Mean recall in `[0, 1]`.
fn graph_recall(graph: &[Vec<usize>], probes: &[usize], truth: &[FxHashSet<usize>]) -> f64 {
    let mut total = 0.0;
    for (pos, &i) in probes.iter().enumerate() {
        let approx: FxHashSet<usize> = graph[i].iter().copied().filter(|&j| j != i).collect();
        total += approx.intersection(&truth[pos]).count() as f64 / truth[pos].len().max(1) as f64;
    }
    total / probes.len() as f64
}

/// Fraction of graph slots that hold a real neighbour rather than a sentinel.
///
/// A silently no-op kernel leaves sentinels behind, so this is the cheap guard
/// against believing a timing from a dispatch that did nothing.
///
/// ### Params
///
/// * `graph` - Descent graph rows
/// * `k` - Neighbours each row should hold
///
/// ### Returns
///
/// Fill ratio in `[0, 1]`.
fn fill_ratio(graph: &[Vec<usize>], k: usize) -> f64 {
    let filled: usize = graph.iter().map(|r| r.len().min(k)).sum();
    filled as f64 / (graph.len() * k) as f64
}

fn main() {
    let cli = Cli::parse();
    let (data, n) = read_blob(&cli.data, cli.dim);

    println!("Data: {} x {} f32 from {}", n, cli.dim, cli.data);
    println!("Threads: {}", rayon::current_num_threads());
    println!(
        "Crate default forest trees at n={n}: {}",
        default_forest_trees(n)
    );

    let truth = if cli.no_recall {
        None
    } else {
        let t0 = Instant::now();
        let gt = ground_truth(&data, n, cli.dim, cli.k, &cli.distance);
        println!("Ground truth ({} probes): {:.2?}", gt.0.len(), t0.elapsed());
        Some(gt)
    };

    let configs: Vec<Option<usize>> = match &cli.tree_sweep {
        Some(sweep) => sweep.iter().copied().map(Some).collect(),
        None => vec![cli.n_trees],
    };

    println!(
        "\n{:>7}  {:>10}  {:>8}  {:>7}",
        "trees", "build (s)", "recall", "fill"
    );

    for cfg in configs {
        let mut best = f64::INFINITY;
        let mut recall = f64::NAN;
        let mut fill = f64::NAN;

        for _ in 0..cli.repeats.max(1) {
            let start = Instant::now();
            let graph = build_knn_graph_gpu::<f32, WgpuRuntime>(
                (&data[..], n, cli.dim),
                &cli.distance,
                Some(cli.k),
                cli.build_k,
                cli.max_iters,
                cfg,
                None,
                cli.rho,
                cli.refine_knn,
                cli.seed,
                cli.verbose,
                WgpuDevice::DefaultDevice,
            )
            .expect("gpu build failed");
            let elapsed = start.elapsed().as_secs_f64();
            best = best.min(elapsed);

            let (rows, _) = graph.extract_knn(Some(cli.k), false);
            fill = fill_ratio(&rows, cli.k);
            if let Some((probes, sets)) = &truth {
                recall = graph_recall(&rows, probes, sets);
            }
        }

        let label = cfg.map_or("auto".to_string(), |t| t.to_string());
        println!("{label:>7}  {best:>10.3}  {recall:>8.4}  {fill:>7.4}");
    }
}
