//! PQ / OPQ query throughput: cross queries and self-kNN for the six product
//! quantised indices, with Recall@k against an exhaustive search.
//!
//! The last `--n-query` rows are held out as cross queries. The exhaustive
//! indices only run the first `--n-query-exh` of them and skip self-kNN, which
//! is quadratic. Self-kNN recall is scored on `--n-eval` evenly spaced rows.
//!
//! Run with:
//! cargo run --example profile_pq_query --release --features quantised -- \
//!   --data cell --dim 512 --n-samples 260000

#[path = "../examples/commons/mod.rs"]
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
    /// Held-out cross queries
    #[arg(long, default_value_t = 10_000)]
    n_query: usize,
    /// Cross queries for the exhaustive indices
    #[arg(long, default_value_t = 1_000)]
    n_query_exh: usize,
    /// Rows scored for self-kNN recall
    #[arg(long, default_value_t = 2_000)]
    n_eval: usize,
    /// PQ subspaces; `dim / 8` when absent
    #[arg(long)]
    m: Option<usize>,
    /// Probed clusters; the index default when absent
    #[arg(long)]
    nprobe: Option<usize>,
    /// Comma-separated subset of exh-pq, exh-opq, ivf-pq, ivf-opq, soar-pq,
    /// soar-opq
    #[arg(long, default_value = "exh-pq,exh-opq,ivf-pq,ivf-opq,soar-pq,soar-opq")]
    only: String,
    /// Timed repetitions, the table shows the median
    #[arg(long, default_value_t = 1)]
    reps: usize,
}

/// Ground truth and the shapes every index is scored against.
struct Eval<'a> {
    /// Database rows
    n: usize,
    /// Neighbours per row
    k: usize,
    /// Timed repetitions
    reps: usize,
    /// Exact neighbours of the cross queries
    truth_q: &'a [Vec<usize>],
    /// Exact neighbours of the self-kNN evaluation rows
    truth_s: &'a [Vec<usize>],
    /// Database rows scored for self-kNN
    eval_rows: &'a [usize],
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
/// * `truth` - Exact neighbours, at least as many rows as `found`
/// * `k` - Neighbours per row
///
/// ### Returns
///
/// Recall@k over the rows of `found`
fn recall(found: &[Vec<usize>], truth: &[Vec<usize>], k: usize) -> f64 {
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(f, t)| f.iter().filter(|i| t.contains(i)).count())
        .sum();
    hits as f64 / (found.len() * k) as f64
}

/// Time a closure `reps` times.
///
/// ### Params
///
/// * `reps` - Repetitions
/// * `f` - Work to time, returns its neighbours
///
/// ### Returns
///
/// Median seconds and the neighbours of the last run
fn timed(reps: usize, mut f: impl FnMut() -> Vec<Vec<usize>>) -> (f64, Vec<Vec<usize>>) {
    let mut ts = Vec::new();
    let mut out = Vec::new();
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        out = f();
        ts.push(t.elapsed().as_secs_f64());
    }
    (median(ts), out)
}

/// Time cross queries and, when given, self-kNN for one index.
///
/// ### Params
///
/// * `name` - Row label
/// * `ev` - Ground truth and shapes
/// * `build_s` - Build time in seconds
/// * `cross` - Runs the cross queries, returns their neighbours
/// * `self_knn` - Runs the self-kNN, returns all `n` rows in original order
fn report(
    name: &str,
    ev: &Eval,
    build_s: f64,
    cross: impl FnMut() -> Vec<Vec<usize>>,
    self_knn: Option<&mut dyn FnMut() -> Vec<Vec<usize>>>,
) {
    let (tq, found_q) = timed(ev.reps, cross);
    println!(
        "{:<9} build {:>7.2} s  cross {:>8.3} s {:>9.0} QPS  recall {:.4}",
        name,
        build_s,
        tq,
        found_q.len() as f64 / tq,
        recall(&found_q, ev.truth_q, ev.k)
    );
    if let Some(f) = self_knn {
        let (ts, found_s) = timed(ev.reps, f);
        let found_s: Vec<Vec<usize>> = ev.eval_rows.iter().map(|&i| found_s[i].clone()).collect();
        println!(
            "{:<9} self  {:>8.3} s {:>9.0} QPS  recall {:.4}",
            name,
            ts,
            ev.n as f64 / ts,
            recall(&found_s, ev.truth_s, ev.k)
        );
    }
}

fn main() {
    let args = Args::parse();
    let cli = &args.data;
    let dim = cli.dim;
    let k = cli.k;
    let m = args.m.unwrap_or(dim / 8);
    let seed = cli.seed as usize;
    let dist = cli.distance.as_str();
    let want = |name: &str| args.only.split(',').any(|s| s.trim() == name);

    let flat = matrix_to_flat(generate_data(cli).0.as_ref()).0;
    let n = flat.len() / dim - args.n_query;
    let (db, queries) = flat.split_at(n * dim);
    let nq_exh = args.n_query_exh.min(args.n_query);
    let queries_exh = &queries[..nq_exh * dim];
    println!(
        "{n} x {dim} database, {} queries ({nq_exh} exhaustive), k = {k}, m = {m}, {dist}",
        args.n_query
    );

    let exact = build_exhaustive_index((db, n, dim), dist, Verbosity::Quiet);
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

    let ev = Eval {
        n,
        k,
        reps: args.reps,
        truth_q: &truth_q,
        truth_s: &truth_s,
        eval_rows: &eval_rows,
    };
    let q = || (queries, args.n_query, dim);
    let q_exh = || (queries_exh, nq_exh, dim);
    let np = args.nprobe;
    let quiet = Verbosity::Quiet;

    if want("exh-pq") {
        let t = Instant::now();
        let idx =
            build_exhaustive_pq_index((db, n, dim), m, None, None, dist, seed, quiet).unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "exh-pq",
            &ev,
            b,
            || {
                query_exhaustive_pq_index(q_exh(), &idx, k, false, quiet)
                    .unwrap()
                    .0
            },
            None,
        );
    }
    if want("exh-opq") {
        let t = Instant::now();
        let idx =
            build_exhaustive_opq_index((db, n, dim), m, None, None, dist, seed, quiet).unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "exh-opq",
            &ev,
            b,
            || {
                query_exhaustive_opq_index(q_exh(), &idx, k, false, quiet)
                    .unwrap()
                    .0
            },
            None,
        );
    }
    if want("ivf-pq") {
        let t = Instant::now();
        let idx = build_ivf_pq_index((db, n, dim), None, m, None, None, dist, seed, quiet).unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "ivf-pq",
            &ev,
            b,
            || {
                query_ivf_pq_index(q(), &idx, k, np, false, quiet)
                    .unwrap()
                    .0
            },
            Some(&mut || {
                query_ivf_pq_index_self(&idx, k, np, false, quiet)
                    .unwrap()
                    .0
            }),
        );
    }
    if want("ivf-opq") {
        let t = Instant::now();
        let idx = build_ivf_opq_index((db, n, dim), None, m, None, None, None, dist, seed, quiet)
            .unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "ivf-opq",
            &ev,
            b,
            || {
                query_ivf_opq_index(q(), &idx, k, np, false, quiet)
                    .unwrap()
                    .0
            },
            Some(&mut || {
                query_ivf_opq_index_self(&idx, k, np, false, quiet)
                    .unwrap()
                    .0
            }),
        );
    }
    if want("soar-pq") {
        let t = Instant::now();
        let idx = build_soar_pq_index((db, n, dim), None, m, None, None, None, dist, seed, quiet)
            .unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "soar-pq",
            &ev,
            b,
            || {
                query_soar_pq_index(q(), &idx, k, np, false, quiet)
                    .unwrap()
                    .0
            },
            Some(&mut || {
                query_soar_pq_index_self(&idx, k, np, false, quiet)
                    .unwrap()
                    .0
            }),
        );
    }
    if want("soar-opq") {
        let t = Instant::now();
        let idx = build_soar_opq_index(
            (db, n, dim),
            None,
            m,
            None,
            None,
            None,
            None,
            dist,
            seed,
            quiet,
        )
        .unwrap();
        let b = t.elapsed().as_secs_f64();
        report(
            "soar-opq",
            &ev,
            b,
            || {
                query_soar_opq_index(q(), &idx, k, np, false, quiet)
                    .unwrap()
                    .0
            },
            Some(&mut || {
                query_soar_opq_index_self(&idx, k, np, false, quiet)
                    .unwrap()
                    .0
            }),
        );
    }
}
