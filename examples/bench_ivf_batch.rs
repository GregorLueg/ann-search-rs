//! CPU IVF query throughput: self-kNN over the whole index and a batch of
//! held-out cross queries, with Recall@k against an exhaustive search.
//!
//! The last `--n-query` rows are held out as cross queries. Self-kNN recall is
//! scored on `--n-eval` evenly spaced database rows.
//!
//! Run with:
//! cargo run --example bench_ivf_batch --release -- --data cell --dim 512 \
//!   --n-samples 260000
//! cargo run --example bench_ivf_batch --release -- --file pca32.f32 --dim 32 \
//!   --n-rows 500000

mod commons;

use ann_search_rs::prelude::*;
use ann_search_rs::utils::verbosity::Verbosity;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use std::time::Instant;

/// Command line.
#[derive(Parser)]
struct Args {
    /// Synthetic data options
    #[command(flatten)]
    data: Cli,
    /// Raw little-endian f32 file (`n * dim` values) instead of synthetic data
    #[arg(long)]
    file: Option<String>,
    /// Keep only the first this many rows of `--file`
    #[arg(long)]
    n_rows: Option<usize>,
    /// Held-out cross queries
    #[arg(long, default_value_t = 10_000)]
    n_query: usize,
    /// Rows scored for self-kNN recall
    #[arg(long, default_value_t = 2_000)]
    n_eval: usize,
    /// Probed clusters; the index default when absent
    #[arg(long)]
    nprobe: Option<usize>,
    /// Also build the self-kNN graph with NN-Descent for comparison
    #[arg(long)]
    nndescent: bool,
    /// Timed repetitions, the table shows the median
    #[arg(long, default_value_t = 1)]
    reps: usize,
}

/// Median of a non-empty list of seconds.
///
/// ### Params
///
/// * `t` - Timings
///
/// ### Returns
///
/// The median
fn median(mut t: Vec<f64>) -> f64 {
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

/// Mean overlap of two neighbour lists per row, divided by `k`.
///
/// ### Params
///
/// * `found` - Approximate neighbours
/// * `truth` - Exact neighbours
/// * `k` - Neighbours per row
///
/// ### Returns
///
/// Recall@k
fn recall(found: &[Vec<usize>], truth: &[Vec<usize>], k: usize) -> f64 {
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(f, t)| f.iter().filter(|i| t.contains(i)).count())
        .sum();
    hits as f64 / (truth.len() * k) as f64
}

fn main() {
    let args = Args::parse();
    let cli = &args.data;
    let dim = cli.dim;
    let k = cli.k;

    let mut flat = match &args.file {
        Some(path) => {
            let bytes = std::fs::read(path).expect("could not read --file");
            println!(">>> Using {path}. <<<");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<f32>>()
        }
        None => matrix_to_flat(generate_data(cli).0.as_ref()).0,
    };
    if let Some(r) = args.n_rows {
        flat.truncate(r * dim);
    }
    let n = flat.len() / dim - args.n_query;
    let (db, queries) = flat.split_at(n * dim);
    println!("{n} x {dim} database, {} queries, k = {k}", args.n_query);

    let t = Instant::now();
    let index = build_ivf_index(
        (db, n, dim),
        None,
        None,
        &cli.distance,
        cli.seed as usize,
        Verbosity::Quiet,
    )
    .unwrap();
    println!("build        {:>8.2} s", t.elapsed().as_secs_f64());

    let exact = build_exhaustive_index((db, n, dim), &cli.distance, Verbosity::Quiet);
    let (truth_q, _) = query_exhaustive_index(
        (queries, args.n_query, dim),
        &exact,
        k,
        false,
        Verbosity::Quiet,
    )
    .unwrap();
    let step = (n / args.n_eval).max(1);
    let eval_rows: Vec<usize> = (0..n).step_by(step).take(args.n_eval).collect();
    let eval_flat: Vec<f32> = eval_rows
        .iter()
        .flat_map(|&i| db[i * dim..(i + 1) * dim].iter().copied())
        .collect();
    let (truth_s, _) = query_exhaustive_index(
        (eval_flat.as_slice(), eval_rows.len(), dim),
        &exact,
        k,
        false,
        Verbosity::Quiet,
    )
    .unwrap();
    drop(exact);

    let mut tq = Vec::new();
    let mut found_q = Vec::new();
    for _ in 0..args.reps.max(1) {
        let t = Instant::now();
        found_q = query_ivf_index(
            (queries, args.n_query, dim),
            &index,
            k,
            args.nprobe,
            true,
            Verbosity::Quiet,
        )
        .unwrap()
        .0;
        tq.push(t.elapsed().as_secs_f64());
    }
    let mut ts = Vec::new();
    let mut found_s = Vec::new();
    for _ in 0..args.reps.max(1) {
        let t = Instant::now();
        found_s = query_ivf_self(&index, k, args.nprobe, true, Verbosity::Quiet)
            .unwrap()
            .0;
        ts.push(t.elapsed().as_secs_f64());
    }
    let found_s: Vec<Vec<usize>> = eval_rows.iter().map(|&i| found_s[i].clone()).collect();

    if args.nndescent {
        // k - 1 plus the self-edge, so rows match IVF's k including self
        let params = NNDescentKnnParams::new(k - 1, None, None, None, 0.001, true, false);
        let mut tn = Vec::new();
        let mut graph = None;
        for _ in 0..args.reps.max(1) {
            let t = Instant::now();
            graph = Some(
                build_knn_graph_nndescent(
                    (db, n, dim),
                    &cli.distance,
                    Some(params.clone()),
                    cli.seed as usize,
                    Verbosity::Quiet,
                )
                .unwrap(),
            );
            tn.push(t.elapsed().as_secs_f64());
        }
        let graph = graph.unwrap();
        let found_n: Vec<Vec<usize>> = eval_rows
            .iter()
            .map(|&i| {
                graph.indices[i * graph.k..(i + 1) * graph.k]
                    .iter()
                    .map(|&j| j as usize)
                    .collect()
            })
            .collect();
        let tn = median(tn);
        println!(
            "nndescent    {:>8.3} s  {:>10.0} QPS  recall {:.4}",
            tn,
            n as f64 / tn,
            recall(&found_n, &truth_s, k)
        );
    }

    let (tq, ts) = (median(tq), median(ts));
    println!(
        "cross query  {:>8.3} s  {:>10.0} QPS  recall {:.4}",
        tq,
        args.n_query as f64 / tq,
        recall(&found_q, &truth_q, k)
    );
    println!(
        "self kNN     {:>8.3} s  {:>10.0} QPS  recall {:.4}",
        ts,
        n as f64 / ts,
        recall(&found_s, &truth_s, k)
    );
}
