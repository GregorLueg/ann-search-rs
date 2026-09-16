//! Profile the graph indices that screen neighbours with RaBitQ codes.
//!
//! `--index qg` is the quantised graph, which stores a code per edge and sweeps
//! a fixed batch of fast-scan lanes per hop. `--mode frontier` traces recall
//! against microseconds per query, which is the only fair way to compare two
//! indices; `occupancy` reports how many of the quantised graph's sweep lanes
//! actually hold a neighbour.
//!
//! ```bash
//! cargo run --release --features binary --example profile_graph_rabitq -- --mode occupancy
//! cargo run --release --features binary --example profile_graph_rabitq -- --mode frontier
//! ```

mod commons;

use std::time::Instant;

use ann_search_rs::binary::qg::QgIndex;
use ann_search_rs::{
    build_exhaustive_index, build_qg_index, build_vamana_index, query_exhaustive_index,
    query_qg_index,
};
use clap::Parser;
use commons::*;
use thousands::*;

/// Sentinel Vamana writes into an unused neighbour slot.
const SENTINEL: u32 = u32::MAX;

/// Lanes scored per fast-scan sweep, mirroring `ann_search_rs::QG_BATCH`.
const BATCH: usize = 32;

/// Held-out rows used for the recall and query timing.
const RECALL_PROBES: usize = 10_000;

/// Degrees swept by `--mode occupancy`, matching the QG grid in `gridsearch_rabitq`.
const DEFAULT_DEGREE_SWEEP: &str = "32,64";

/// `l_build` values swept by `--mode occupancy`, matching the same grid.
const DEFAULT_L_BUILD_SWEEP: &str = "32,128";

/// `ef_search` values swept by `--mode frontier` when none are given.
const DEFAULT_EF_SWEEP: &str = "16,32,64,100,150,200,300,400,600,800";

/// CLI for the profiler.
#[derive(Parser, Debug)]
#[command(about = "Quantised graph occupancy, build and query profile")]
struct Cli {
    /// Which index to profile: qg
    #[arg(long, default_value = "qg")]
    index: String,

    /// What to profile: occupancy, build, query or frontier
    #[arg(long, default_value = "occupancy")]
    mode: String,

    /// Comma-separated `ef_search` values for `--mode frontier`
    #[arg(long, default_value = DEFAULT_EF_SWEEP)]
    ef_search_sweep: String,

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

    /// Comma-separated degrees for `--mode occupancy`
    #[arg(long, default_value = DEFAULT_DEGREE_SWEEP)]
    degree_sweep: String,

    /// Comma-separated `l_build` values for `--mode occupancy`
    #[arg(long, default_value = DEFAULT_L_BUILD_SWEEP)]
    l_build_sweep: String,

    /// Degree for `--mode build` and `--mode query`
    #[arg(long, default_value_t = 32)]
    degree: usize,

    /// Beam width during construction
    #[arg(long, default_value_t = 128)]
    l_build: usize,


    /// Beam width for Vamana's first pass, `None` picks the default
    #[arg(long)]
    l_build_pass1: Option<usize>,

    /// Pruning alpha for pass 1
    #[arg(long, default_value_t = 1.0)]
    alpha_pass1: f32,

    /// Pruning alpha for pass 2
    #[arg(long, default_value_t = 1.2)]
    alpha_pass2: f32,

    /// Search budget for `--mode query`
    #[arg(long, default_value_t = 100)]
    ef_search: usize,

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

    /// Repeat each timed query point this many times, keeping the best
    #[arg(long, default_value_t = 3)]
    query_repeats: usize,
}

/// Generate the dataset as a flat row-major `Vec<f32>`.
///
/// ### Params
///
/// * `cli` - Parsed command line
///
/// ### Returns
///
/// `(data, n, dim)` with `data.len() == n * dim`.
fn load_data(cli: &Cli) -> (Vec<f32>, usize, usize) {
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
    (flat, n, dim)
}

/// Parse a comma-separated list of `usize`.
///
/// ### Params
///
/// * `raw` - The comma-separated string
///
/// ### Returns
///
/// The parsed values
fn parse_list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

/// Report how many of each vertex's neighbour slots hold a real neighbour.
///
/// The graph comes straight from [`build_vamana_index`], which is the topology
/// `QgIndex::build` encodes, so this measures what the sweep will see without
/// reaching inside the index.
///
/// ### Params
///
/// * `graph` - Flat `n * degree` adjacency, sentinel-padded
/// * `n` - Number of vertices
/// * `degree` - Row stride
fn report_occupancy(graph: &[u32], n: usize, degree: usize) {
    let mut fill = vec![0usize; degree + 1];
    let mut total = 0usize;

    for node in 0..n {
        let occupied = graph[node * degree..(node + 1) * degree]
            .iter()
            .filter(|&&e| e != SENTINEL)
            .count();
        fill[occupied] += 1;
        total += occupied;
    }

    let mean = total as f64 / n as f64;
    let full = fill[degree];

    // A sweep is paid for in full once any of its lanes is occupied, so the
    // work done is `ceil(occupied / BATCH)` sweeps against `occupied` useful
    // lanes. Under the current fixed `n_batches` every vertex sweeps the whole
    // stride regardless, which is the second number.
    let mut lanes_needed = 0usize;
    for (occupied, &count) in fill.iter().enumerate() {
        lanes_needed += count * occupied.div_ceil(BATCH) * BATCH;
    }
    let lanes_swept = n * degree;

    println!(
        "  mean degree {:.2} / {} ({:.1}% of slots), {} of {} vertices full ({:.1}%)",
        mean,
        degree,
        100.0 * mean / degree as f64,
        full.separate_with_underscores(),
        n.separate_with_underscores(),
        100.0 * full as f64 / n as f64,
    );
    println!(
        "  useful lanes {:.1}% of the {} swept; skipping empty sweeps would cut to {}",
        100.0 * total as f64 / lanes_swept as f64,
        lanes_swept.separate_with_underscores(),
        lanes_needed.separate_with_underscores(),
    );

    print!("  histogram:");
    let mut shown = 0;
    for (occupied, &count) in fill.iter().enumerate() {
        if count == 0 {
            continue;
        }
        print!(" {}:{:.1}%", occupied, 100.0 * count as f64 / n as f64);
        shown += 1;
        if shown % 8 == 0 {
            print!("\n            ");
        }
    }
    println!();
}

/// The index under profile.
///
/// Both answer the same `(k, ef_search)` question, so the profiler only needs
/// to know which one to build and how to ask it.
enum GraphIndex {
    /// Quantised graph, one code per edge
    Qg(Box<QgIndex<f32>>),
}

impl GraphIndex {
    /// Build whichever index `--index` named.
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
    /// The built index
    fn build(flat: &[f32], n: usize, dim: usize, cli: &Cli) -> Self {
        match cli.index.as_str() {
            "qg" => Self::Qg(Box::new(
                build_qg_index(
                    (flat, n, dim),
                    cli.degree,
                    cli.l_build,
                    cli.l_build_pass1,
                    cli.alpha_pass1,
                    cli.alpha_pass2,
                    &cli.distance,
                    cli.seed as usize,
                )
                .expect("qg build failed"),
            )),
            other => panic!("unknown --index '{other}', expected qg"),
        }
    }

    /// Bytes the index holds, in megabytes.
    ///
    /// ### Returns
    ///
    /// Memory usage in MB
    fn memory_mb(&self) -> f64 {
        let bytes = match self {
            Self::Qg(i) => i.memory_usage_bytes(),
        };
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// Query every probe row.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major probe rows
    /// * `n_probes` - Number of probe rows
    /// * `dim` - Row width
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width
    ///
    /// ### Returns
    ///
    /// The neighbours of each probe
    fn query(
        &self,
        queries: &[f32],
        n_probes: usize,
        dim: usize,
        k: usize,
        ef: usize,
    ) -> Vec<Vec<usize>> {
        let mat = (queries, n_probes, dim);
        let (res, _) = match self {
            Self::Qg(i) => query_qg_index(mat, i, k, ef, false, false),
        }
        .expect("query failed");
        res
    }
}

/// Pick a strided probe set and its exhaustive ground truth.
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
/// `(probe_rows_flat, n_probes, true_neighbours)`
fn probe_set(flat: &[f32], n: usize, dim: usize, cli: &Cli) -> (Vec<f32>, usize, Vec<Vec<usize>>) {
    let stride = (n / cli.n_probes).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(cli.n_probes).collect();

    let mut queries = Vec::with_capacity(probes.len() * dim);
    for &i in &probes {
        queries.extend_from_slice(&flat[i * dim..(i + 1) * dim]);
    }

    let exhaustive = build_exhaustive_index((flat, n, dim), &cli.distance);
    let (truth, _) = query_exhaustive_index(
        (&queries[..], probes.len(), dim),
        &exhaustive,
        cli.k,
        false,
        false,
    )
    .expect("exhaustive query failed");

    (queries, probes.len(), truth)
}

/// Mean recall of `found` against `truth`.
///
/// ### Params
///
/// * `found` - Neighbours the index returned
/// * `truth` - Exhaustive neighbours
///
/// ### Returns
///
/// Recall in `[0, 1]`
fn recall(found: &[Vec<usize>], truth: &[Vec<usize>]) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (f, t) in found.iter().zip(truth.iter()) {
        hits += f.iter().filter(|id| t.contains(id)).count();
        total += t.len();
    }
    hits as f64 / total.max(1) as f64
}

fn main() {
    let cli = Cli::parse();
    let (flat, n, dim) = load_data(&cli);

    println!("-----------------------------");
    println!(
        "QG {} profile: {} samples, {} dim, {} dist, {} threads",
        cli.mode,
        n.separate_with_underscores(),
        dim,
        cli.distance,
        rayon::current_num_threads()
    );
    println!("-----------------------------");

    if cli.mode == "occupancy" {
        for degree in parse_list(&cli.degree_sweep) {
            for l_build in parse_list(&cli.l_build_sweep) {
                let start = Instant::now();
                let vamana = build_vamana_index(
                    (&flat[..], n, dim),
                    degree,
                    l_build,
                    cli.l_build_pass1,
                    cli.alpha_pass1,
                    cli.alpha_pass2,
                    &cli.distance,
                    cli.seed as usize,
                );
                println!(
                    "degree {}, l_build {} (built in {:.2?})",
                    degree,
                    l_build,
                    start.elapsed()
                );
                report_occupancy(&vamana.graph, n, degree);
            }
        }
        return;
    }

    let start = Instant::now();
    let index = GraphIndex::build(&flat, n, dim, &cli);
    println!("Build: {:.2?}", start.elapsed());
    println!("Index size: {:.1} MB", index.memory_mb());

    if cli.mode == "build" {
        return;
    }

    let (queries, n_probes, truth) = probe_set(&flat, n, dim, &cli);

    if cli.mode == "frontier" {
        println!("{:>8}  {:>8}  {:>10}", "ef", "recall", "us/query");
        for ef in parse_list(&cli.ef_search_sweep) {
            let mut best = f64::INFINITY;
            let mut found = Vec::new();
            for _ in 0..cli.query_repeats.max(1) {
                let start = Instant::now();
                let res = index.query(&queries, n_probes, dim, cli.k, ef);
                let micros = start.elapsed().as_secs_f64() * 1e6 / n_probes as f64;
                best = best.min(micros);
                found = res;
            }
            println!("{:>8}  {:>8.4}  {:>10.2}", ef, recall(&found, &truth), best);
        }
        return;
    }

    let mut best = f64::INFINITY;
    let mut found = Vec::new();
    for _ in 0..cli.query_repeats.max(1) {
        let start = Instant::now();
        let res = index.query(&queries, n_probes, dim, cli.k, cli.ef_search);
        let micros = start.elapsed().as_secs_f64() * 1e6 / n_probes as f64;
        best = best.min(micros);
        found = res;
    }

    println!(
        "ef_search {}: recall {:.4}, {:.2} us/query",
        cli.ef_search,
        recall(&found, &truth),
        best
    );
}
