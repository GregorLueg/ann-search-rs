//! PQ codebook training: the quantiser's own mini-batch trainer against
//! Lloyd's and e-means from the shared k-means.
//!
//! Per method and `m`: codebook training time (all subspaces, run one after
//! another like `ProductQuantiser::train`), the mean squared reconstruction
//! error of the database, and Recall@k of a plain ADC scan against an exact
//! search. The last `--n-query` rows are held out as queries.
//!
//! `--stage-split` first builds IVF-PQ and IVF-OPQ at `Detailed` so the stage
//! timings show what share of the build the codebooks are.
//!
//! Run with:
//! cargo run --example bench_pq_codebooks --release --features quantised -- \
//!   --data cell --dim 512 --n-samples 251000 --ms 64

#[path = "../examples/commons/mod.rs"]
mod commons;

use ann_search_rs::prelude::*;
use ann_search_rs::quantised::quantisers::ProductQuantiser;
use ann_search_rs::utils::k_means_utils::train_centroids;
use ann_search_rs::utils::verbosity::Verbosity;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use rayon::prelude::*;
use std::time::Instant;

/// Command line.
#[derive(Parser)]
struct Args {
    /// Synthetic data options
    #[command(flatten)]
    data: Cli,
    /// Comma-separated subspace counts
    #[arg(long, default_value = "64")]
    ms: String,
    /// Comma-separated e-means epoch budgets
    #[arg(long, default_value = "1,2,3,5")]
    epochs: String,
    /// Iterations for the mini-batch trainer and Lloyd's
    #[arg(long, default_value_t = 30)]
    iters: usize,
    /// Held-out query rows
    #[arg(long, default_value_t = 1_000)]
    n_query: usize,
    /// Build IVF-PQ and IVF-OPQ at `Detailed` first
    #[arg(long)]
    stage_split: bool,
    /// Only run the PQ mini-batch trainer
    #[arg(long)]
    mini_batch_only: bool,
    /// Timed repetitions of each training, the table shows the median
    #[arg(long, default_value_t = 1)]
    reps: usize,
}

/// Codebook size; the `u8` code limit and the library default.
const N_CENTROIDS: usize = 256;

/// One codebook trainer.
#[derive(Clone, Copy)]
enum Method {
    /// `ProductQuantiser::train`, the PQ mini-batch trainer
    MiniBatch,
    /// Shared Lloyd's at `--iters`
    Lloyd,
    /// Shared e-means with this many epochs
    EMeans(f64),
}

impl Method {
    /// Label for the table.
    fn label(&self) -> String {
        match self {
            Method::MiniBatch => "pq mini-batch".into(),
            Method::Lloyd => "lloyd".into(),
            Method::EMeans(t) => format!("e-means {t} ep"),
        }
    }
}

/// Train one codebook per subspace.
///
/// ### Params
///
/// * `method` - The trainer
/// * `data` - Row-major database
/// * `dim` - Row length
/// * `m` - Number of subspaces
/// * `iters` - Iterations for the mini-batch trainer and Lloyd's
/// * `seed` - Seed; subspace `s` uses `seed + s` like the library
///
/// ### Returns
///
/// `m` flat `N_CENTROIDS * dim / m` codebooks
fn train_codebooks(
    method: Method,
    data: &[f32],
    dim: usize,
    m: usize,
    iters: usize,
    seed: usize,
) -> Vec<Vec<f32>> {
    if let Method::MiniBatch = method {
        return ProductQuantiser::train(
            data,
            dim,
            m,
            Some(N_CENTROIDS),
            &Dist::SquaredEuclidean,
            iters,
            seed,
            Verbosity::Quiet,
        )
        .unwrap()
        .codebooks()
        .to_vec();
    }
    let n = data.len() / dim;
    let sd = dim / m;
    (0..m)
        .map(|s| {
            let mut sub = Vec::with_capacity(n * sd);
            for row in data.chunks_exact(dim) {
                sub.extend_from_slice(&row[s * sd..(s + 1) * sd]);
            }
            match method {
                Method::MiniBatch => unreachable!(),
                Method::Lloyd => train_centroids(
                    &sub,
                    sd,
                    n,
                    N_CENTROIDS,
                    &Dist::SquaredEuclidean,
                    Some(KMeansTrainingParams::new(iters, None, None)),
                    seed + s,
                    Verbosity::Quiet,
                )
                .unwrap(),
                Method::EMeans(t) => train_centroids(
                    &sub,
                    sd,
                    n,
                    N_CENTROIDS,
                    &Dist::SquaredEuclidean,
                    Some(
                        KMeansTrainingParams::default()
                            .with_algorithm(KMeansAlgorithm::EMeans(EMeansParams::new(t))),
                    ),
                    seed + s,
                    Verbosity::Quiet,
                )
                .unwrap(),
            }
        })
        .collect()
}

/// Encode every row and return the codes plus the mean squared error.
///
/// ### Params
///
/// * `data` - Row-major database
/// * `dim` - Row length
/// * `books` - Codebooks from [`train_codebooks`]
///
/// ### Returns
///
/// Row-major `n * m` codes and the per-row mean squared reconstruction error
fn encode(data: &[f32], dim: usize, books: &[Vec<f32>]) -> (Vec<u8>, f64) {
    let m = books.len();
    let sd = dim / m;
    let per_row: Vec<(Vec<u8>, f64)> = data
        .par_chunks_exact(dim)
        .map(|row| {
            let mut codes = Vec::with_capacity(m);
            let mut err = 0.0f64;
            for (s, book) in books.iter().enumerate() {
                let q = &row[s * sd..(s + 1) * sd];
                let (best, d) = book
                    .chunks_exact(sd)
                    .map(|c| f32::euclidean_simd(q, c))
                    .enumerate()
                    .fold(
                        (0, f32::INFINITY),
                        |a, (i, d)| if d < a.1 { (i, d) } else { a },
                    );
                codes.push(best as u8);
                err += d as f64;
            }
            (codes, err)
        })
        .collect();
    let n = per_row.len();
    let mse = per_row.iter().map(|r| r.1).sum::<f64>() / n as f64;
    (per_row.into_iter().flat_map(|r| r.0).collect(), mse)
}

/// Indices of the `k` smallest values.
///
/// ### Params
///
/// * `d` - Distances, consumed
/// * `k` - How many
///
/// ### Returns
///
/// The `k` indices, unordered
fn top_k(d: Vec<f32>, k: usize) -> Vec<usize> {
    let mut idx: Vec<(f32, usize)> = d.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
    idx.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
    idx[..k].iter().map(|p| p.1).collect()
}

/// Recall@k of an ADC scan against the exact neighbours.
///
/// ### Params
///
/// * `queries` - Row-major queries
/// * `codes` - Database codes from [`encode`]
/// * `books` - Codebooks
/// * `dim` - Row length
/// * `truth` - Exact top-k per query
/// * `k` - Neighbours
///
/// ### Returns
///
/// Mean Recall@k
fn adc_recall(
    queries: &[f32],
    codes: &[u8],
    books: &[Vec<f32>],
    dim: usize,
    truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    let m = books.len();
    let sd = dim / m;
    let hits: usize = queries
        .par_chunks_exact(dim)
        .zip(truth.par_iter())
        .map(|(q, t)| {
            let mut lut = vec![0.0f32; m * N_CENTROIDS];
            for (s, book) in books.iter().enumerate() {
                for (c, cent) in book.chunks_exact(sd).enumerate() {
                    lut[s * N_CENTROIDS + c] = f32::euclidean_simd(&q[s * sd..(s + 1) * sd], cent);
                }
            }
            let d: Vec<f32> = codes
                .chunks_exact(m)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(s, &c)| lut[s * N_CENTROIDS + c as usize])
                        .sum()
                })
                .collect();
            let found = top_k(d, k);
            found.iter().filter(|i| t.contains(i)).count()
        })
        .sum();
    hits as f64 / (truth.len() * k) as f64
}

fn main() {
    let args = Args::parse();
    let cli = &args.data;
    let dim = cli.dim;
    let k = cli.k;
    let seed = cli.seed as usize;

    let (mat, _) = generate_data(cli);
    let flat = matrix_to_flat(mat.as_ref()).0;
    let n = flat.len() / dim - args.n_query;
    let (db, queries) = flat.split_at(n * dim);

    if args.stage_split {
        for m in parse_list::<usize>(&args.ms) {
            println!("--- IVF-PQ m = {m} ---");
            build_ivf_pq_index(
                (db, n, dim),
                None,
                m,
                None,
                None,
                &cli.distance,
                seed,
                Verbosity::Detailed,
            )
            .unwrap();
            println!("--- IVF-OPQ m = {m} ---");
            build_ivf_opq_index(
                (db, n, dim),
                None,
                m,
                None,
                None,
                None,
                &cli.distance,
                seed,
                Verbosity::Detailed,
            )
            .unwrap();
        }
    }

    let truth: Vec<Vec<usize>> = queries
        .par_chunks_exact(dim)
        .map(|q| {
            let d: Vec<f32> = db
                .chunks_exact(dim)
                .map(|r| f32::euclidean_simd(q, r))
                .collect();
            top_k(d, k)
        })
        .collect();

    let mut methods = vec![Method::MiniBatch];
    if !args.mini_batch_only {
        methods.push(Method::Lloyd);
        methods.extend(
            parse_list::<f64>(&args.epochs)
                .into_iter()
                .map(Method::EMeans),
        );
    }

    println!(
        "\n{} x {} train rows, {} queries, k = {}, {} centroids per subspace",
        n, dim, args.n_query, k, N_CENTROIDS
    );
    println!(
        "{:>5} {:>4} {:<16} {:>10} {:>12} {:>9} {:>18}",
        "m", "sd", "method", "train s", "mse", "recall", "checksum"
    );
    for m in parse_list::<usize>(&args.ms) {
        for &method in &methods {
            let mut times = Vec::with_capacity(args.reps);
            let mut books = Vec::new();
            for _ in 0..args.reps.max(1) {
                let t = Instant::now();
                books = train_codebooks(method, db, dim, m, args.iters, seed);
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(f64::total_cmp);
            let train_s = times[times.len() / 2];
            let (codes, mse) = encode(db, dim, &books);
            let recall = adc_recall(queries, &codes, &books, dim, &truth, k);
            let checksum = books.iter().flatten().fold(0u64, |h, v| {
                h.wrapping_mul(31).wrapping_add(v.to_bits() as u64)
            });
            println!(
                "{:>5} {:>4} {:<16} {:>10.2} {:>12.5} {:>9.4} {:>18x}",
                m,
                dim / m,
                method.label(),
                train_s,
                mse,
                recall,
                checksum
            );
        }
    }
}

/// Parse a comma-separated list.
///
/// ### Params
///
/// * `s` - The list
///
/// ### Returns
///
/// The parsed values
fn parse_list<V: std::str::FromStr>(s: &str) -> Vec<V>
where
    V::Err: std::fmt::Debug,
{
    s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}
