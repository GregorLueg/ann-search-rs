//! IVF self-kNN: MLX against cubecl/wgpu, CPU exhaustive as ground truth.
//! Both indices use nlist = sqrt(n), nprobe = sqrt(nlist) and the same
//! number of fixed k-means iterations.
//!
//! Run with:
//! cargo run --example bench_mlx_ivf --release --features gpu,mlx,synthetic -- \
//!   --n-samples 10000 --dim 32 --data cell

mod commons;

use ann_search_rs::prelude::*;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Number of warm self-queries timed after the cold one.
const WARM_RUNS: usize = 3;

/// Lloyd's iterations for both k-means.
const KMEANS_ITERS: usize = 30;

/// Time a cold call then `WARM_RUNS` warm ones, keeping the last result.
///
/// ### Params
///
/// * `f` - The self-query to time
///
/// ### Returns
///
/// `(cold ms, best warm ms, neighbours)`
fn time_cold_warm(mut f: impl FnMut() -> Vec<Vec<usize>>) -> (f64, f64, Vec<Vec<usize>>) {
    let start = Instant::now();
    let mut res = f();
    let cold = start.elapsed().as_secs_f64() * 1e3;
    let mut warm = f64::INFINITY;
    for _ in 0..WARM_RUNS {
        let start = Instant::now();
        res = f();
        warm = warm.min(start.elapsed().as_secs_f64() * 1e3);
    }
    (cold, warm, res)
}

fn main() {
    let cli = Cli::parse();
    println!(
        "{} samples, dim {}, k {}, {} distance",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.k,
        cli.distance
    );
    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let seed = cli.seed as usize;

    let start = Instant::now();
    let cpu = build_exhaustive_index(data.as_ref(), &cli.distance);
    let (truth, _) = query_exhaustive_self(&cpu, cli.k, false, false).unwrap();
    println!(
        "CPU exhaustive (ground truth): {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let start = Instant::now();
    let wgpu = build_ivf_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        None,
        Some(KMeansGpuParams::new(KMEANS_ITERS, None, true, false)),
        &cli.distance,
        seed,
        false,
        device,
    )
    .unwrap();
    let wgpu_build = start.elapsed().as_secs_f64() * 1e3;
    let (wgpu_cold, wgpu_warm, wgpu_nn) = time_cold_warm(|| {
        query_ivf_index_gpu_self(&wgpu, cli.k, None, None, false, false)
            .unwrap()
            .0
    });

    let start = Instant::now();
    let mlx = build_ivf_index_mlx(
        data.as_ref(),
        None,
        Some(KMeansTrainingParams::new(KMEANS_ITERS, None, None)),
        &cli.distance,
        seed,
        false,
    )
    .unwrap();
    let mlx_build = start.elapsed().as_secs_f64() * 1e3;
    let (mlx_cold, mlx_warm, mlx_nn) = time_cold_warm(|| {
        query_ivf_index_mlx_self(&mlx, cli.k, None, None, false, true)
            .unwrap()
            .0
    });

    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "backend", "build ms", "cold ms", "warm ms", "recall"
    );
    for (name, build, cold, warm, nn) in [
        ("wgpu", wgpu_build, wgpu_cold, wgpu_warm, &wgpu_nn),
        ("mlx", mlx_build, mlx_cold, mlx_warm, &mlx_nn),
    ] {
        println!(
            "{:<8} {:>10.1} {:>10.1} {:>10.1} {:>10.4}",
            name,
            build,
            cold,
            warm,
            calculate_recall(&truth, nn, cli.k)
        );
    }
}
