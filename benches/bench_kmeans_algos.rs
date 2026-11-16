//! Lloyd's k-means against e-means, per backend: training time and the final
//! mean squared distance of every row to its nearest trained centroid.
//!
//! The error is always scored on the CPU with an exact assignment, so the
//! backends are compared on the same yardstick. Each configuration runs once
//! untimed (shader JIT, buffer pools), then `--reps` timed runs; the table
//! shows the median time and the mean error over the timed runs.
//!
//! Run with:
//! cargo run --example bench_kmeans_algos --release -- --n-samples 250000 --dim 128
//! cargo run --example bench_kmeans_algos --release --features gpu,mlx -- \
//!   --backends cpu,gpu,mlx --data cell --dim 512
//! A raw little-endian f32 file (`n * dim` values) replaces the synthetic data
//! with `--file path --dim d`.

#[path = "../examples/commons/mod.rs"]
mod commons;

use ann_search_rs::prelude::*;
use ann_search_rs::utils::k_means_utils::{assign_all_parallel, train_centroids, CentroidDistance};
use ann_search_rs::utils::verbosity::Verbosity;
use clap::Parser;
use commons::*;
use faer::Mat;
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

/// Command line.
#[derive(Parser)]
struct Args {
    /// Synthetic data options
    #[command(flatten)]
    data: Cli,
    /// Raw f32 file to cluster instead of synthetic data
    #[arg(long)]
    file: Option<String>,
    /// Comma-separated centroid counts
    #[arg(long, default_value = "256,1024")]
    centroids: String,
    /// Comma-separated e-means epoch budgets
    #[arg(long, default_value = "3,5")]
    epochs: String,
    /// Lloyd's iterations; each backend's default when absent
    #[arg(long)]
    lloyd_iters: Option<usize>,
    /// Comma-separated backends: cpu, gpu, mlx
    #[arg(long, default_value = "cpu")]
    backends: String,
    /// Timed repetitions per configuration, each with a different seed
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Build a CPU IVF with each method instead (nlist from `--centroids`)
    /// and report build time, Recall@k and QPS
    #[arg(long)]
    ivf: bool,
}

/// Queries for the IVF mode; ground truth is an exhaustive search.
const IVF_N_QUERY: usize = 10_000;

/// One clustering method on one backend.
#[derive(Clone, Copy)]
enum Algo {
    /// Lloyd's with an optional iteration override
    Lloyd(Option<usize>),
    /// e-means with this many epochs
    EMeans(f64),
}

impl Algo {
    /// Label for the table.
    fn label(&self) -> String {
        match self {
            Algo::Lloyd(Some(i)) => format!("lloyd {i} it"),
            Algo::Lloyd(None) => "lloyd default".into(),
            Algo::EMeans(t) => format!("e-means {t} ep"),
        }
    }

    /// CPU / MLX parameters.
    fn cpu_params(&self) -> KMeansTrainingParams {
        match *self {
            Algo::Lloyd(Some(i)) => KMeansTrainingParams::new(i, None, None),
            Algo::Lloyd(None) => {
                KMeansTrainingParams::default().with_algorithm(KMeansAlgorithm::Lloyd)
            }
            Algo::EMeans(t) => KMeansTrainingParams::default()
                .with_algorithm(KMeansAlgorithm::EMeans(EMeansParams::new(t))),
        }
    }
}

/// Train once on the named backend.
///
/// ### Params
///
/// * `backend` - `cpu`, `gpu` or `mlx`
/// * `algo` - The method
/// * `data` - Row-major data
/// * `dim` - Row length
/// * `k` - Number of centroids
/// * `metric` - Distance metric
/// * `seed` - Seed
///
/// ### Returns
///
/// The trained centroids
fn train(
    backend: &str,
    algo: Algo,
    data: &[f32],
    dim: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Vec<f32> {
    let n = data.len() / dim;
    match backend {
        "cpu" => train_centroids(
            data,
            dim,
            n,
            k,
            metric,
            Some(algo.cpu_params()),
            seed,
            Verbosity::Quiet,
        )
        .unwrap(),
        #[cfg(feature = "gpu")]
        "gpu" => {
            use ann_search_rs::gpu::k_means_gpu::train_centroids_gpu;
            use cubecl::wgpu::WgpuDevice;
            let params = match algo {
                Algo::Lloyd(Some(i)) => KMeansGpuParams::new(i, None, true, false),
                Algo::Lloyd(None) => {
                    KMeansGpuParams::default().with_algorithm(KMeansAlgorithm::Lloyd)
                }
                Algo::EMeans(t) => KMeansGpuParams::default()
                    .with_algorithm(KMeansAlgorithm::EMeans(EMeansParams::new(t))),
            };
            let device: cubecl::Device = WgpuDevice::default().into();
            let client = device.client();
            train_centroids_gpu(
                data,
                dim,
                n,
                k,
                metric,
                Some(params),
                seed,
                &client,
                Verbosity::Quiet,
            )
            .unwrap()
        }
        #[cfg(mlx_available)]
        // The public MLX entry point also assigns every row and clears the
        // buffer cache, so its timings carry one extra pass over the data.
        "mlx" => {
            use ann_search_rs::mlx::k_means_mlx::k_means_clusters_mlx;
            let dist = if *metric == Dist::Cosine {
                "cosine"
            } else {
                "euclidean"
            };
            let (mat, _) = k_means_clusters_mlx(
                (data, n, dim),
                dist,
                k,
                Some(algo.cpu_params()),
                seed,
                Verbosity::Quiet,
            )
            .unwrap();
            (0..k)
                .flat_map(|i| (0..dim).map(move |j| (i, j)))
                .map(|(i, j)| mat[(i, j)])
                .collect()
        }
        other => panic!("backend '{other}' is unknown or not compiled in"),
    }
}

/// Mean distance of every row to its nearest centroid: squared Euclidean, or
/// `1 - cos` for cosine.
///
/// ### Params
///
/// * `data` - Row-major data
/// * `dim` - Row length
/// * `cents` - Row-major centroids
/// * `metric` - Distance metric
///
/// ### Returns
///
/// The mean error
fn mean_error(data: &[f32], dim: usize, cents: &[f32], metric: &Dist) -> f64 {
    let n = data.len() / dim;
    let k = cents.len() / dim;
    let cnorms: Vec<f32> = cents.chunks(dim).map(f32::calculate_l2_norm).collect();
    let assign = assign_all_parallel(data, &[], dim, n, cents, &cnorms, k, metric);
    let total: f64 = data
        .par_chunks(dim)
        .zip(assign.par_iter())
        .map(|(x, &a)| {
            let c = &cents[a as usize * dim..(a as usize + 1) * dim];
            match metric {
                Dist::Cosine => {
                    let denom = f32::calculate_l2_norm(x) * cnorms[a as usize];
                    (1.0 - f32::dot_simd(x, c) / denom.max(f32::MIN_POSITIVE)) as f64
                }
                _ => f32::euclidean_simd(x, c) as f64,
            }
        })
        .sum();
    total / n as f64
}

/// Build a CPU IVF index per method and nlist, query it at three nprobes and
/// print build time, Recall@k and QPS against exhaustive ground truth.
///
/// ### Params
///
/// * `data` - Row-major data
/// * `dim` - Row length
/// * `nlists` - Cluster counts to build
/// * `algos` - Methods to compare
/// * `args` - Command line, for `k`, the metric and the seed
fn run_ivf(data: &[f32], dim: usize, nlists: &[usize], algos: &[Algo], args: &Args) {
    let n = data.len() / dim;
    let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
    let queries = subsample_with_noise(&mat, IVF_N_QUERY, args.data.seed + 1);
    let k = args.data.k;
    let dist = &args.data.distance;

    let exh = ann_search_rs::build_exhaustive_index(mat.as_ref(), dist, Verbosity::Quiet);
    let (truth, _) =
        ann_search_rs::query_exhaustive_index(queries.as_ref(), &exh, k, false, Verbosity::Quiet)
            .unwrap();
    drop(exh);
    let metric = parse_ann_dist(dist).unwrap_or_default();
    let qflat = matrix_to_flat(queries.as_ref()).0;

    println!(
        "\n| nlist | method | build (s) | imbalance | max/mean | empty | nprobe | rows scanned/query | recall@{k} | QPS |\n|---|---|---|---|---|---|---|---|---|---|"
    );
    for &nlist in nlists {
        let root = (nlist as f64).sqrt();
        let mut nprobes = vec![root as usize, (2.0 * root) as usize, nlist / 20];
        nprobes.retain(|&p| p > 0);
        nprobes.sort();
        nprobes.dedup();
        for &algo in algos {
            let t = Instant::now();
            let idx = ann_search_rs::build_ivf_index(
                mat.as_ref(),
                Some(nlist),
                Some(algo.cpu_params()),
                dist,
                args.data.seed as usize,
                Verbosity::Quiet,
            )
            .unwrap();
            let build = t.elapsed().as_secs_f64();

            // Cluster sizes from the index's own centroids; the imbalance
            // factor is the expected rows scanned per probe over n / nlist.
            let cents = idx.centroids();
            let cnorms: Vec<f32> = cents.chunks(dim).map(f32::calculate_l2_norm).collect();
            let assign = assign_all_parallel(data, &[], dim, n, cents, &cnorms, nlist, &metric);
            let mut sizes = vec![0usize; nlist];
            for &a in &assign {
                sizes[a as usize] += 1;
            }
            let mean = n as f64 / nlist as f64;
            let imbalance = nlist as f64 * sizes.iter().map(|&s| (s * s) as f64).sum::<f64>()
                / (n as f64 * n as f64);
            let max_ratio = *sizes.iter().max().unwrap() as f64 / mean;
            let empty = sizes.iter().filter(|&&s| s == 0).count();

            for &np in &nprobes {
                let scanned: f64 = qflat
                    .par_chunks(dim)
                    .map(|q| {
                        let mut d: Vec<(f32, usize)> = cents
                            .chunks(dim)
                            .zip(&cnorms)
                            .enumerate()
                            .map(|(j, (c, &cn))| match metric {
                                Dist::Cosine => {
                                    (-f32::dot_simd(q, c) / cn.max(f32::MIN_POSITIVE), j)
                                }
                                _ => (f32::euclidean_simd(q, c), j),
                            })
                            .collect();
                        d.select_nth_unstable_by(np - 1, |a, b| a.0.total_cmp(&b.0));
                        d[..np].iter().map(|&(_, j)| sizes[j]).sum::<usize>() as f64
                    })
                    .sum::<f64>()
                    / IVF_N_QUERY as f64;
                let t = Instant::now();
                let (got, _) = ann_search_rs::query_ivf_index(
                    queries.as_ref(),
                    &idx,
                    k,
                    Some(np),
                    false,
                    Verbosity::Quiet,
                )
                .unwrap();
                let qps = IVF_N_QUERY as f64 / t.elapsed().as_secs_f64();
                println!(
                    "| {nlist} | {} | {build:.3} | {imbalance:.3} | {max_ratio:.2} | {empty} | {np} | {:.0} | {:.4} | {:.0} |",
                    algo.label(),
                    scanned,
                    calculate_recall(&truth, &got, k),
                    qps
                );
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    let metric = parse_ann_dist(&args.data.distance).unwrap_or_default();

    let (data, dim) = match &args.file {
        Some(path) => {
            let bytes = std::fs::read(path).expect("could not read --file");
            let vals: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            println!(">>> Using {path}. <<<");
            (vals, args.data.dim)
        }
        None => {
            let (mat, _) = generate_data(&args.data);
            (matrix_to_flat(mat.as_ref()).0, args.data.dim)
        }
    };
    let n = data.len() / dim;
    println!(
        "n = {}, dim = {dim}, metric = {metric:?}, threads = {}",
        n.separate_with_underscores(),
        rayon::current_num_threads()
    );

    let ks: Vec<usize> = args
        .centroids
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let mut algos = vec![Algo::Lloyd(args.lloyd_iters)];
    algos.extend(
        args.epochs
            .split(',')
            .map(|s| Algo::EMeans(s.parse().unwrap())),
    );

    if args.ivf {
        run_ivf(&data, dim, &ks, &algos, &args);
        return;
    }

    println!(
        "\n| backend | k | method | median time (s) | mean error | sd error |\n|---|---|---|---|---|---|"
    );
    for backend in args.backends.split(',') {
        for &k in &ks {
            for &algo in &algos {
                let _ = train(backend, algo, &data, dim, k, &metric, 0);
                let mut times = Vec::with_capacity(args.reps);
                let mut errs = Vec::with_capacity(args.reps);
                for rep in 0..args.reps {
                    let t = Instant::now();
                    let cents = train(backend, algo, &data, dim, k, &metric, rep + 1);
                    times.push(t.elapsed().as_secs_f64());
                    errs.push(mean_error(&data, dim, &cents, &metric));
                }
                times.sort_by(f64::total_cmp);
                let mean = errs.iter().sum::<f64>() / errs.len() as f64;
                let sd = (errs.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errs.len() as f64)
                    .sqrt();
                println!(
                    "| {backend} | {k} | {} | {:.3} | {mean:.5} | {sd:.5} |",
                    algo.label(),
                    times[times.len() / 2]
                );
            }
        }
    }
}
