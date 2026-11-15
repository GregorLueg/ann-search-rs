//! IVF query throughput: self-kNN over the whole index and a batch of
//! held-out cross queries, with Recall@k against an exhaustive search. CPU by
//! default; `--gpu` adds the wgpu index (`gpu` feature) and `--mlx` the MLX
//! index with each of its scans (`gpu,mlx` features). Every backend builds its
//! own index at the shared defaults (nlist = sqrt(n), nprobe = sqrt(nlist)).
//!
//! The last `--n-query` rows are held out as cross queries. Self-kNN recall is
//! scored on `--n-eval` evenly spaced database rows.
//!
//! Run with:
//! cargo run --example bench_ivf_batch --release -- --data cell --dim 512 \
//!   --n-samples 260000
//! cargo run --example bench_ivf_batch --release -- --file pca32.f32 --dim 32 \
//!   --n-rows 500000
//! cargo run --example bench_ivf_batch --release --features gpu,mlx -- \
//!   --file pca32.f32 --dim 32 --n-rows 500000 --gpu --mlx --reps 3

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
    /// Also time the wgpu index (needs the `gpu` feature)
    #[arg(long)]
    gpu: bool,
    /// Also time the MLX index, every scan (needs the `gpu,mlx` features)
    #[arg(long)]
    mlx: bool,
}

/// Ground truth and the shapes every backend is scored against.
struct Eval<'a> {
    n: usize,
    n_query: usize,
    k: usize,
    reps: usize,
    truth_q: &'a [Vec<usize>],
    truth_s: &'a [Vec<usize>],
    eval_rows: &'a [usize],
}

/// Time cross and self queries for one backend and print its two rows.
///
/// ### Params
///
/// * `name` - Row label
/// * `ev` - Ground truth and shapes
/// * `cross` - Runs the held-out queries, returns their neighbours
/// * `self_knn` - Runs the self-kNN, returns all `n` rows in original order
fn time_backend(
    name: &str,
    ev: &Eval,
    mut cross: impl FnMut() -> Vec<Vec<usize>>,
    mut self_knn: impl FnMut() -> Vec<Vec<usize>>,
) {
    let mut tq = Vec::new();
    let mut found_q = Vec::new();
    for _ in 0..ev.reps.max(1) {
        let t = Instant::now();
        found_q = cross();
        tq.push(t.elapsed().as_secs_f64());
    }
    let mut ts = Vec::new();
    let mut found_s = Vec::new();
    for _ in 0..ev.reps.max(1) {
        let t = Instant::now();
        found_s = self_knn();
        ts.push(t.elapsed().as_secs_f64());
    }
    let found_s: Vec<Vec<usize>> = ev.eval_rows.iter().map(|&i| found_s[i].clone()).collect();

    // The first run carries any device warm-up, so it is shown separately
    let (cq, cs) = (tq[0], ts[0]);
    let (tq, ts) = (median(tq), median(ts));
    println!(
        "{:<8} cross {:>8.3} s (cold {:>7.3})  {:>10.0} QPS  recall {:.4}",
        name,
        tq,
        cq,
        ev.n_query as f64 / tq,
        recall(&found_q, ev.truth_q, ev.k)
    );
    println!(
        "{:<8} self  {:>8.3} s (cold {:>7.3})  {:>10.0} QPS  recall {:.4}",
        name,
        ts,
        cs,
        ev.n as f64 / ts,
        recall(&found_s, ev.truth_s, ev.k)
    );
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
    println!("build cpu    {:>8.2} s", t.elapsed().as_secs_f64());

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

    let ev = Eval {
        n,
        n_query: args.n_query,
        k,
        reps: args.reps,
        truth_q: &truth_q,
        truth_s: &truth_s,
        eval_rows: &eval_rows,
    };

    time_backend(
        "cpu",
        &ev,
        || {
            query_ivf_index(
                (queries, args.n_query, dim),
                &index,
                k,
                args.nprobe,
                false,
                Verbosity::Quiet,
            )
            .unwrap()
            .0
        },
        || {
            query_ivf_self(&index, k, args.nprobe, false, Verbosity::Quiet)
                .unwrap()
                .0
        },
    );
    drop(index);

    #[cfg(feature = "gpu")]
    if args.gpu {
        let t = Instant::now();
        let gpu = build_ivf_index_gpu::<f32>(
            (db, n, dim),
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            Verbosity::Quiet,
            cubecl::wgpu::WgpuDevice::default().into(),
        )
        .unwrap();
        println!("build wgpu   {:>8.2} s", t.elapsed().as_secs_f64());
        time_backend(
            "wgpu",
            &ev,
            || {
                query_ivf_index_gpu(
                    (queries, args.n_query, dim),
                    &gpu,
                    k,
                    args.nprobe,
                    None,
                    false,
                    Verbosity::Quiet,
                )
                .unwrap()
                .0
            },
            || {
                query_ivf_index_gpu_self(&gpu, k, args.nprobe, None, false, Verbosity::Quiet)
                    .unwrap()
                    .0
            },
        );
    }
    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        println!("[WARNING] --gpu needs the gpu feature, skipped");
    }

    #[cfg(mlx_available)]
    if args.mlx {
        use ann_search_rs::mlx::ivf_mlx::IvfScanMlx;
        let t = Instant::now();
        let mut mlx = build_ivf_index_mlx(
            (db, n, dim),
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            Verbosity::Quiet,
        )
        .unwrap();
        println!("build mlx    {:>8.2} s", t.elapsed().as_secs_f64());
        for (name, scan) in [
            ("mlx-qm", IvfScanMlx::QueryMajor),
            ("mlx-cm", IvfScanMlx::ClusterMajor),
            ("mlx-mma", IvfScanMlx::ClusterMatrix),
        ] {
            mlx = mlx.with_scan(scan);
            time_backend(
                name,
                &ev,
                || {
                    query_ivf_index_mlx(
                        (queries, args.n_query, dim),
                        &mlx,
                        k,
                        args.nprobe,
                        None,
                        false,
                        Verbosity::Quiet,
                    )
                    .unwrap()
                    .0
                },
                || {
                    query_ivf_index_mlx_self(&mlx, k, args.nprobe, None, false, Verbosity::Quiet)
                        .unwrap()
                        .0
                },
            );
        }
    }
    #[cfg(not(mlx_available))]
    if args.mlx {
        println!("[WARNING] --mlx needs the gpu,mlx features and libmlxc, skipped");
    }

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
                    Some(params),
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
            "nndescent self {:>6.3} s  {:>10.0} QPS  recall {:.4}",
            tn,
            n as f64 / tn,
            recall(&found_n, &truth_s, k)
        );
    }
}
