//! Profile the HNSW build phase.
//!
//! Builds one index at a single `(M, ef_construction)` pair and reports the
//! build time plus recall against exhaustive ground truth. One config, one
//! build, no grid: the point is a clean profile under `samply`, where the
//! gridsearch example smears eight builds and a self-query into the same
//! flame graph.
//!
//! ```bash
//! cargo run --release --example profile_hnsw -- --n-samples 150000 --dim 32
//! samply record ./target/release/examples/profile_hnsw --no-recall
//! ```

mod commons;

use std::time::Instant;

use ann_search_rs::{build_exhaustive_index, build_hnsw_index, query_exhaustive_index};
use clap::Parser;
use commons::*;
use faer::Mat;
use rustc_hash::FxHashSet;
use thousands::*;

/// Number of held-out rows used to estimate recall against exhaustive truth.
///
/// Exhaustive search is `O(n)` per probe, so this stays small: 2000 probes is
/// enough to separate a 0.90 graph from a 0.95 one, and costs a fraction of the
/// build it is measuring.
const RECALL_PROBES: usize = 2_000;

/// CLI for the profiler.
#[derive(Parser, Debug)]
#[command(about = "Single-config profile of the HNSW build")]
struct Cli {
    /// Number of samples to generate
    #[arg(long, default_value_t = DEFAULT_N_SAMPLES)]
    n_samples: usize,

    /// Dimensionality of each row
    #[arg(long, default_value_t = DEFAULT_DIM)]
    dim: usize,

    /// Number of clusters in the synthetic data
    #[arg(long, default_value_t = DEFAULT_N_CLUSTERS)]
    n_clusters: usize,

    /// Synthetic data generator: gaussian, correlated, lowrank or cell
    #[arg(long, default_value = DEFAULT_DATA)]
    data: String,

    /// Base connectivity parameter
    #[arg(long, default_value_t = 16)]
    m: usize,

    /// Size of the dynamic candidate list during construction
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// Search budget used for the recall probe
    #[arg(long, default_value_t = 100)]
    ef_search: usize,

    /// Neighbours to compare when measuring recall
    #[arg(long, default_value_t = DEFAULT_K)]
    k: usize,

    /// Distance metric
    #[arg(long, default_value = DEFAULT_DISTANCE)]
    distance: String,

    /// Random seed
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Skip the recall check and only report the build time
    #[arg(long, default_value_t = false)]
    no_recall: bool,

    /// Repeat the build this many times and report each timing
    #[arg(long, default_value_t = 1)]
    repeats: usize,
}

/// Mean recall@k of the index against exhaustive ground truth.
///
/// Probes a strided sample of the indexed rows rather than a held-out query
/// set, so the measurement describes the graph itself rather than the graph
/// plus a generalisation gap.
///
/// ### Params
///
/// * `data` - The indexed matrix
/// * `index` - Built HNSW index to probe
/// * `k` - Neighbours to compare
/// * `ef_search` - Search budget for the approximate side
/// * `distance` - Metric name
///
/// ### Returns
///
/// Mean recall in `[0, 1]`.
fn index_recall<I>(data: &Mat<f32>, index: &I, k: usize, ef_search: usize, distance: &str) -> f64
where
    I: Fn(&[f32], usize, usize) -> Vec<usize>,
{
    let (n, dim) = (data.nrows(), data.ncols());
    let stride = (n / RECALL_PROBES).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(RECALL_PROBES).collect();

    let flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..dim).map(move |j| (i, j)))
        .map(|(i, j)| data[(i, j)])
        .collect();

    let mut queries = Vec::with_capacity(probes.len() * dim);
    for &i in &probes {
        queries.extend_from_slice(&flat[i * dim..(i + 1) * dim]);
    }

    let exhaustive = build_exhaustive_index((&flat[..], n, dim), distance);
    let (truth, _) = query_exhaustive_index(
        (&queries[..], probes.len(), dim),
        &exhaustive,
        k,
        false,
        false,
    )
    .expect("exhaustive query failed");

    let total: f64 = probes
        .iter()
        .enumerate()
        .map(|(row, &i)| {
            let got = index(&flat[i * dim..(i + 1) * dim], k, ef_search);
            let want: FxHashSet<usize> = truth[row].iter().copied().collect();
            got.iter().filter(|id| want.contains(id)).count() as f64 / k as f64
        })
        .sum();

    total / probes.len() as f64
}

fn main() {
    let cli = Cli::parse();

    println!("-----------------------------");
    println!(
        "HNSW build profile: {} samples, {} dim, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("M = {}, ef_construction = {}", cli.m, cli.ef_construction);
    println!("-----------------------------");

    let gen_cli = commons::Cli {
        n_samples: cli.n_samples,
        dim: cli.dim,
        n_clusters: cli.n_clusters,
        k: cli.k,
        seed: cli.seed,
        distance: cli.distance.clone(),
        data: cli.data.clone(),
        intrinsic_dim: DEFAULT_INTRINSIC_DIM,
    };
    let (data, _) = generate_data(&gen_cli);

    let mut index = None;
    for run in 0..cli.repeats.max(1) {
        let start = Instant::now();
        let built = build_hnsw_index(
            data.as_ref(),
            cli.m,
            cli.ef_construction,
            &cli.distance,
            cli.seed as usize,
            false,
        );
        println!("Build {}: {:.2?}", run + 1, start.elapsed());
        index = Some(built);
    }

    let index = index.expect("at least one build");

    if !cli.no_recall {
        let recall = index_recall(
            &data,
            &|q: &[f32], k: usize, ef: usize| index.query(q, k, ef).expect("query failed").0,
            cli.k,
            cli.ef_search,
            &cli.distance,
        );
        println!(
            "Recall@{} (ef_search = {}): {:.4}",
            cli.k, cli.ef_search, recall
        );
    }

    println!(
        "Index size: {:.1} MB",
        index.memory_usage_bytes() as f64 / (1024.0 * 1024.0)
    );
}
