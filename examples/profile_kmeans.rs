//! Profile `train_centroids` and report wall-clock, iterations run and WCSS.
//!
//! `--dump-file` writes the dataset as raw little-endian f32 so external
//! k-means implementations (SuperKMeans, FAISS) can cluster the identical rows.
//!
//! ```bash
//! cargo run --release --example profile_kmeans -- --n-samples 500000 --dim 50 --k 700 --data cell
//! samply record ./target/release/examples/profile_kmeans --dim 768 --k 1000
//! cargo run --release --features gpu --example profile_kmeans -- --gpu --dim 128 --k 1000 --repeats 4
//! ```

mod commons;

use std::io::Write;
use std::time::Instant;

use ann_search_rs::prelude::*;
use ann_search_rs::utils::k_means_utils::{assign_all_parallel, train_centroids};
use clap::Parser;
use commons::*;

/// CLI for the k-means profiler.
#[derive(Parser, Debug)]
#[command(about = "k-means training profile")]
struct Cli {
    /// Read row-major little-endian f32 from this file instead of generating
    #[arg(long)]
    data_file: Option<String>,

    /// Write the generated data as row-major little-endian f32 and exit
    #[arg(long)]
    dump_file: Option<String>,

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

    /// Number of centroids to train
    #[arg(long, default_value_t = 1000)]
    k: usize,

    /// Lloyd iterations
    #[arg(long, default_value_t = 30)]
    iters: usize,

    /// Force a Lloyd path: hamerly_gemm, hamerly_simd, gemm, parallel
    #[arg(long)]
    path: Option<String>,

    /// Force the init: random or kmeans_parallel
    #[arg(long)]
    init: Option<String>,

    /// Distance metric
    #[arg(long, default_value = DEFAULT_DISTANCE)]
    distance: String,

    /// Random seed
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Repeat the training this many times and report each timing
    #[arg(long, default_value_t = 1)]
    repeats: usize,

    /// Skip the WCSS evaluation
    #[arg(long, default_value_t = false)]
    no_wcss: bool,

    /// Train on the GPU (wgpu) with a fixed number of iterations; needs the
    /// `gpu` feature
    #[arg(long, default_value_t = false)]
    gpu: bool,
}

/// Train on the GPU with `cli.iters` fixed iterations.
///
/// ### Params
///
/// * `data` - Row-major data
/// * `n` - Number of rows
/// * `dim` - Row width
/// * `metric` - Distance metric
/// * `init` - Optional initialisation
/// * `cli` - Parsed command line
///
/// ### Returns
///
/// Row-major centroids.
#[cfg(feature = "gpu")]
fn train_gpu(
    data: &[f32],
    n: usize,
    dim: usize,
    metric: &Dist,
    init: Option<KMeansInit>,
    cli: &Cli,
) -> Vec<f32> {
    use ann_search_rs::gpu::k_means_gpu::train_centroids_gpu;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl::Runtime;

    let client = WgpuRuntime::client(&WgpuDevice::default());
    let params = KMeansGpuParams::new(cli.iters, init, true, false);
    train_centroids_gpu::<f32, WgpuRuntime>(
        data,
        dim,
        n,
        cli.k,
        metric,
        Some(params),
        cli.seed as usize,
        &client,
        true,
    )
    .expect("train_centroids_gpu failed")
}

/// Stub for builds without the `gpu` feature.
///
/// ### Params
///
/// * `_data`, `_n`, `_dim`, `_metric`, `_init`, `_cli` - Unused
///
/// ### Returns
///
/// Never returns.
#[cfg(not(feature = "gpu"))]
fn train_gpu(
    _data: &[f32],
    _n: usize,
    _dim: usize,
    _metric: &Dist,
    _init: Option<KMeansInit>,
    _cli: &Cli,
) -> Vec<f32> {
    panic!("--gpu needs the `gpu` feature");
}

/// Load the dataset as a flat row-major `Vec<f32>`.
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
            k: DEFAULT_K,
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

    let n = cli.n_samples.min(flat.len() / cli.dim);
    flat.truncate(n * cli.dim);

    (flat, n, cli.dim)
}

/// Within-cluster sum of squared Euclidean distances, accumulated in f64.
///
/// ### Params
///
/// * `data` - Row-major data
/// * `n` - Number of rows
/// * `dim` - Row width
/// * `centroids` - Row-major centroids
/// * `k` - Number of centroids
///
/// ### Returns
///
/// WCSS against the nearest centroid of every row.
fn wcss(data: &[f32], n: usize, dim: usize, centroids: &[f32], k: usize) -> f64 {
    let data_norms: Vec<f32> = data
        .chunks_exact(dim)
        .map(|v| f32::dot_simd(v, v))
        .collect();
    let cent_norms: Vec<f32> = centroids
        .chunks_exact(dim)
        .map(|c| f32::dot_simd(c, c))
        .collect();
    let assign = assign_all_parallel(
        data,
        &data_norms,
        dim,
        n,
        centroids,
        &cent_norms,
        k,
        &Dist::SquaredEuclidean,
    );

    data.chunks_exact(dim)
        .zip(&assign)
        .map(|(v, &c)| {
            let c = &centroids[c * dim..(c + 1) * dim];
            v.iter()
                .zip(c)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
        })
        .sum()
}

fn main() {
    let cli = Cli::parse();
    let (data, n, dim) = load_data(&cli);

    if let Some(path) = &cli.dump_file {
        let mut f = std::fs::File::create(path).expect("could not create --dump-file");
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        f.write_all(&bytes).expect("could not write --dump-file");
        println!("Wrote {} x {} f32 to {}", n, dim, path);
        return;
    }

    let metric = parse_ann_dist(&cli.distance).unwrap_or_default();
    let path = cli.path.as_deref().map(|p| match p {
        "hamerly_gemm" => LloydPath::HamerlyGemm,
        "hamerly_simd" => LloydPath::HamerlySimd,
        "gemm" => LloydPath::GemmLloyd,
        "parallel" => LloydPath::ParallelLloyd,
        other => panic!("unknown --path {other}"),
    });
    let init = cli.init.as_deref().map(|i| match i {
        "random" => KMeansInit::Random,
        "kmeans_parallel" => KMeansInit::KMeansParallel,
        other => panic!("unknown --init {other}"),
    });

    println!(
        "n = {}, dim = {}, k = {}, iters = {}, metric = {:?}, path = {:?}, init = {:?}",
        n, dim, cli.k, cli.iters, metric, path, init
    );

    let mut centroids = Vec::new();
    for r in 0..cli.repeats {
        let params = KMeansTrainingParams::new(cli.iters, init, path);
        let start = Instant::now();
        centroids = if cli.gpu {
            train_gpu(&data, n, dim, &metric, init, &cli)
        } else {
            train_centroids(
                &data,
                dim,
                n,
                cli.k,
                &metric,
                Some(params),
                cli.seed as usize,
                true,
            )
            .expect("train_centroids failed")
        };
        println!(
            "repeat {}: {:.1} ms",
            r,
            start.elapsed().as_secs_f64() * 1e3
        );
    }

    if !cli.no_wcss {
        println!("WCSS: {:.6e}", wcss(&data, n, dim, &centroids, cli.k));
    }
}
