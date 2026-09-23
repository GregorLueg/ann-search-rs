//! Profile the Vamana build and query paths, and trace the recall-vs-time frontier.
//!
//! Three modes. `build` times the build alone, one config, no recall, which is
//! what you want under `samply`. `query` builds once and then times a probe set
//! at a single `ef_search`. `frontier` builds once and sweeps `ef_search`,
//! emitting `(recall, microseconds per query)` pairs.
//!
//! The build is what `QgIndex::build` spends 93-97% of its time in, so this is
//! the harness for any work on that. Two shapes matter: low dimensionality,
//! where the per-neighbour loop machinery dominates, and wide vectors, where
//! the distance kernel does. They rank build optimisations differently.
//!
//! ```bash
//! cargo run --release --example profile_vamana -- --n-samples 150000 --dim 32
//! cargo run --release --example profile_vamana -- --n-samples 50000 --dim 512
//! cargo run --release --example profile_vamana -- --mode frontier
//! samply record ./target/release/examples/profile_vamana --mode build --no-recall
//! ```

mod commons;

use std::time::Instant;

use ann_search_rs::{
    build_exhaustive_index, build_vamana_index, query_exhaustive_index, query_vamana_index,
};
use clap::Parser;
use commons::*;
use thousands::*;

/// Number of held-out rows used to estimate recall against exhaustive truth.
///
/// Exhaustive search is `O(n)` per probe, so this stays modest. 10 000 probes is
/// enough to separate a 0.95 graph from a 0.96 one and, at ~10 threads, long
/// enough that the query timing is not dominated by rayon start-up.
const RECALL_PROBES: usize = 10_000;

/// `ef_search` values swept by `--mode frontier` when none are given.
const DEFAULT_EF_SWEEP: &str = "10,16,24,32,48,64,100,150,200";

/// CLI for the profiler.
#[derive(Parser, Debug)]
#[command(about = "Vamana build/query profile and recall-vs-time frontier")]
struct Cli {
    /// What to profile: build, query or frontier
    #[arg(long, default_value = "build")]
    mode: String,

    /// Read row-major little-endian f32 from this file instead of generating
    #[arg(long)]
    data_file: Option<String>,

    /// Number of samples to generate, or rows to read from `--data-file`
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

    /// Maximum out-degree of the graph
    #[arg(long, default_value_t = 32)]
    r: usize,

    /// Beam width during construction
    #[arg(long, default_value_t = 128)]
    l_build: usize,

    /// Beam width for the first pass, defaults to `--l-build`
    #[arg(long)]
    l_build_pass1: Option<usize>,

    /// Pruning alpha for pass 1
    #[arg(long, default_value_t = 1.0)]
    alpha_pass1: f32,

    /// Pruning alpha for pass 2
    #[arg(long, default_value_t = 1.2)]
    alpha_pass2: f32,

    /// Search budget for `--mode query` and for the build-mode recall check
    #[arg(long, default_value_t = 100)]
    ef_search: usize,

    /// Comma-separated `ef_search` values for `--mode frontier`
    #[arg(long, default_value = DEFAULT_EF_SWEEP)]
    ef_search_sweep: String,

    /// Neighbours to compare when measuring recall
    #[arg(long, default_value_t = DEFAULT_K)]
    k: usize,

    /// Rows probed for recall and for the query timing
    #[arg(long, default_value_t = RECALL_PROBES)]
    n_probes: usize,

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

    /// Repeat each timed query sweep point this many times, keeping the best
    #[arg(long, default_value_t = 3)]
    query_repeats: usize,
}

/// Load the dataset as a flat row-major `Vec<f32>`.
///
/// Either reads raw little-endian f32 from `--data-file` or runs one of the
/// synthetic generators.
///
/// ### Params
///
/// * `cli` - Parsed command line
///
/// ### Returns
///
/// `(data, n, dim)` with `data.len() == n * dim`.
fn load_data(cli: &Cli) -> (Vec<f32>, usize, usize) {
    let Some(path) = &cli.data_file else {
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
        let (n, dim) = (data.nrows(), data.ncols());
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push(data[(i, j)]);
            }
        }
        return (flat, n, dim);
    };

    let bytes = std::fs::read(path).expect("could not read --data-file");
    let mut flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let available = flat.len() / cli.dim;
    let n = cli.n_samples.min(available);
    flat.truncate(n * cli.dim);

    (flat, n, cli.dim)
}

/// Pick a strided probe set and its exhaustive ground truth.
///
/// Probes strided rows of the indexed data rather than a held-out set, so the
/// measurement describes the graph itself rather than the graph plus a
/// generalisation gap.
///
/// ### Params
///
/// * `flat` - Row-major data
/// * `n` - Number of rows
/// * `dim` - Row width
/// * `cli` - Parsed command line
///
/// ### Returns
///
/// `(probe_rows_flat, n_probes, true_neighbours)`.
fn probe_set(flat: &[f32], n: usize, dim: usize, cli: &Cli) -> (Vec<f32>, usize, Vec<Vec<usize>>) {
    let stride = (n / cli.n_probes).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(cli.n_probes).collect();

    let mut queries = Vec::with_capacity(probes.len() * dim);
    for &i in &probes {
        queries.extend_from_slice(&flat[i * dim..(i + 1) * dim]);
    }

    let start = Instant::now();
    let exhaustive = build_exhaustive_index((flat, n, dim), &cli.distance);
    let (truth, _) = query_exhaustive_index(
        (&queries[..], probes.len(), dim),
        &exhaustive,
        cli.k,
        false,
        false,
    )
    .expect("exhaustive query failed");
    println!(
        "Ground truth for {} probes: {:.2?}",
        probes.len(),
        start.elapsed()
    );

    (queries, probes.len(), truth)
}

fn main() {
    let cli = Cli::parse();

    let (flat, n, dim) = load_data(&cli);

    println!("-----------------------------");
    println!(
        "Vamana {} profile: {} samples, {} dim, {} dist, {} threads",
        cli.mode,
        n.separate_with_underscores(),
        dim,
        cli.distance,
        rayon::current_num_threads()
    );
    println!(
        "R = {}, l_build = {} / pass1 {:?}, alpha = {} / {}",
        cli.r, cli.l_build, cli.l_build_pass1, cli.alpha_pass1, cli.alpha_pass2
    );
    println!("-----------------------------");

    let mut index = None;
    for run in 0..cli.repeats.max(1) {
        let start = Instant::now();
        let built = build_vamana_index(
            (&flat[..], n, dim),
            cli.r,
            cli.l_build,
            cli.l_build_pass1,
            cli.alpha_pass1,
            cli.alpha_pass2,
            &cli.distance,
            cli.seed as usize,
        );
        println!("Build {}: {:.2?}", run + 1, start.elapsed());

        index = Some(built);
    }
    let index = index.expect("at least one build");

    if cli.no_recall || cli.mode == "build" {
        println!(
            "Index size: {:.1} MB",
            index.memory_usage_bytes() as f64 / (1024.0 * 1024.0)
        );
        return;
    }

    let (queries, n_probes, truth) = probe_set(&flat, n, dim, &cli);

    let sweep: Vec<usize> = if cli.mode == "frontier" {
        cli.ef_search_sweep
            .split(',')
            .map(|s| s.trim().parse().expect("bad --ef-search-sweep value"))
            .collect()
    } else {
        vec![cli.ef_search]
    };

    println!();
    println!(
        "{:>10}  {:>10}  {:>12}  {:>12}",
        "ef_search", "recall", "query (ms)", "us/query"
    );

    for ef in sweep {
        // One untimed pass so the first sweep point is not charged for cold
        // pages in the graph and vector arrays.
        let _ = query_vamana_index(
            (&queries[..], n_probes, dim),
            &index,
            cli.k,
            Some(ef),
            false,
            false,
        )
        .expect("vamana query failed");

        let mut best = f64::INFINITY;
        let mut got = Vec::new();
        for _ in 0..cli.query_repeats.max(1) {
            let start = Instant::now();
            let (indices, _) = query_vamana_index(
                (&queries[..], n_probes, dim),
                &index,
                cli.k,
                Some(ef),
                false,
                false,
            )
            .expect("vamana query failed");
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            best = best.min(elapsed);
            got = indices;
        }

        let recall = calculate_recall(&truth, &got, cli.k);
        println!(
            "{:>10}  {:>10.4}  {:>12.2}  {:>12.2}",
            ef,
            recall,
            best,
            best * 1000.0 / n_probes as f64
        );
    }

    println!();
    println!(
        "Index size: {:.1} MB",
        index.memory_usage_bytes() as f64 / (1024.0 * 1024.0)
    );
}
