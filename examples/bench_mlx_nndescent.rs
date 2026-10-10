//! NN-Descent kNN graph build: MLX against cubecl/wgpu, CPU exhaustive as
//! ground truth.
//!
//! Run with:
//! cargo run --example bench_mlx_nndescent --release --features gpu,mlx,synthetic -- \
//!   --n-samples 10000 --dim 32 --data cell

mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Number of warm builds timed after the cold one.
const WARM_RUNS: usize = 3;

/// Time a cold build then `WARM_RUNS` warm ones, keeping the last graph.
///
/// ### Params
///
/// * `f` - The build, returning the graph rows (self included)
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
    let k = cli.k;

    let start = Instant::now();
    let cpu = build_exhaustive_index(data.as_ref(), &cli.distance);
    let (truth, _) = query_exhaustive_self(&cpu, k, false, false).unwrap();
    println!(
        "CPU exhaustive (ground truth): {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    // The graph excludes self; ask for k - 1 neighbours and prepend self so
    // the rows line up with the exhaustive self-kNN.
    let (wgpu_cold, wgpu_warm, wgpu_nn) = time_cold_warm(|| {
        build_knn_graph_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
            data.as_ref(),
            &cli.distance,
            Some(k - 1),
            None,
            None,
            None,
            None,
            None,
            None,
            cli.seed as usize,
            false,
            Default::default(),
        )
        .unwrap()
        .extract_knn(None, true, false)
        .0
    });

    let (mlx_cold, mlx_warm, mlx_nn) = time_cold_warm(|| {
        build_knn_graph_mlx(
            data.as_ref(),
            &cli.distance,
            Some(k - 1),
            None,
            None,
            None,
            None,
            None,
            None,
            cli.seed as usize,
            false,
        )
        .unwrap()
        .extract_knn(None, true, false)
        .0
    });

    println!();
    println!(
        "{:<8} {:>14} {:>14} {:>10}",
        "backend", "cold build ms", "warm build ms", "recall"
    );
    for (name, cold, warm, nn) in [
        ("wgpu", wgpu_cold, wgpu_warm, &wgpu_nn),
        ("mlx", mlx_cold, mlx_warm, &mlx_nn),
    ] {
        println!(
            "{:<8} {:>14.1} {:>14.1} {:>10.4}",
            name,
            cold,
            warm,
            calculate_recall(&truth, nn, k),
        );
    }
}
