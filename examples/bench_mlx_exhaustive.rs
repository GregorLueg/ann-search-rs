//! Exhaustive self-kNN: MLX against cubecl/wgpu, CPU exhaustive as ground
//! truth.
//!
//! Run with:
//! cargo run --example bench_mlx_exhaustive --release --features gpu,mlx,synthetic -- \
//!   --n-samples 10000 --dim 32 --data cell

mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Number of warm self-queries timed after the cold one.
const WARM_RUNS: usize = 3;

/// Fraction of rows whose top-k index set equals the ground truth exactly.
///
/// ### Params
///
/// * `truth` - Ground-truth neighbours per row
/// * `approx` - Neighbours under test per row
///
/// ### Returns
///
/// Fraction in `[0, 1]`
fn exact_set_fraction(truth: &[Vec<usize>], approx: &[Vec<usize>]) -> f64 {
    let hits = truth
        .iter()
        .zip(approx)
        .filter(|(t, a)| {
            let mut t = (*t).clone();
            let mut a = (*a).clone();
            t.sort_unstable();
            a.sort_unstable();
            t == a
        })
        .count();
    hits as f64 / truth.len() as f64
}

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

    let start = Instant::now();
    let cpu = build_exhaustive_index(data.as_ref(), &cli.distance);
    let (truth, _) = query_exhaustive_self(&cpu, cli.k, false, false).unwrap();
    println!(
        "CPU exhaustive (ground truth): {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let start = Instant::now();
    let wgpu = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device,
    )
    .unwrap();
    let wgpu_build = start.elapsed().as_secs_f64() * 1e3;
    let (wgpu_cold, wgpu_warm, wgpu_nn) = time_cold_warm(|| {
        query_exhaustive_index_gpu_self(&wgpu, cli.k, false, false)
            .unwrap()
            .0
    });

    let start = Instant::now();
    let mlx = build_exhaustive_index_mlx(data.as_ref(), &cli.distance).unwrap();
    let mlx_build = start.elapsed().as_secs_f64() * 1e3;
    let (mlx_cold, mlx_warm, mlx_nn) = time_cold_warm(|| {
        query_exhaustive_index_mlx_self(&mlx, cli.k, false, true)
            .unwrap()
            .0
    });

    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "backend", "build ms", "cold ms", "warm ms", "recall", "exact set"
    );
    for (name, build, cold, warm, nn) in [
        ("wgpu", wgpu_build, wgpu_cold, wgpu_warm, &wgpu_nn),
        ("mlx", mlx_build, mlx_cold, mlx_warm, &mlx_nn),
    ] {
        println!(
            "{:<8} {:>10.1} {:>10.1} {:>10.1} {:>10.4} {:>10.4}",
            name,
            build,
            cold,
            warm,
            calculate_recall(&truth, nn, cli.k),
            exact_set_fraction(&truth, nn)
        );
    }
}
