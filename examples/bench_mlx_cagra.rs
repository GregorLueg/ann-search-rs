//! CAGRA beam search: MLX against cubecl/wgpu on the same navigational graph
//! and the same entry points. The graph is built once with the wgpu
//! NN-Descent; CPU exhaustive is the ground truth.
//!
//! Run with:
//! cargo run --example bench_mlx_cagra --release --features gpu,mlx,synthetic -- \
//!   --n-samples 10000 --dim 32 --data cell

mod commons;

use ann_search_rs::mlx::cagra_mlx::*;
use ann_search_rs::prelude::*;
use ann_search_rs::synthetic::subsample_with_noise;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Number of warm runs timed after the cold one.
const WARM_RUNS: usize = 3;

/// Time a cold call then `WARM_RUNS` warm ones, keeping the last result.
///
/// ### Params
///
/// * `f` - The query to time
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
    let queries = subsample_with_noise(&data, cli.n_samples / 10, cli.seed + 1);
    let (q_flat, n_q, dim) = queries.as_ref().into_row_major();

    let start = Instant::now();
    let cpu = build_exhaustive_index(data.as_ref(), &cli.distance);
    let (truth_q, _) = query_exhaustive_index(queries.as_ref(), &cpu, cli.k, false, false).unwrap();
    let (truth_self, _) = query_exhaustive_self(&cpu, cli.k, false, false).unwrap();
    println!(
        "CPU exhaustive (ground truth): {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    let start = Instant::now();
    let mut wgpu = build_nndescent_index_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        Some(cli.k),
        None,
        Some(20),
        None,
        Some(0.0005),
        None,
        Some(1),
        cli.seed as usize,
        false,
        true,
        WgpuDevice::default(),
    )
    .unwrap();
    let wgpu_build = start.elapsed().as_secs_f64() * 1e3;

    let (vec_flat, n, _) = data.as_ref().into_row_major();
    let start = Instant::now();
    let mlx = CagraSearchMlx::new(
        &vec_flat,
        n,
        dim,
        wgpu.metric(),
        wgpu.nav_graph().to_vec(),
        wgpu.k,
        wgpu.medoid,
    )
    .unwrap();
    let mlx_upload = start.elapsed().as_secs_f64() * 1e3;
    let knn_rows: Vec<u32> = wgpu.knn_graph().iter().map(|&(p, _)| p as u32).collect();
    let n_entry = CagraMlxSearchParams::from_k(cli.k).get_n_entry();

    let (wq_cold, wq_warm, wq_nn) = time_cold_warm(|| {
        query_nndescent_index_gpu(queries.as_ref(), &mut wgpu, cli.k, None, false, false)
            .unwrap()
            .0
    });
    let (ws_cold, ws_warm, ws_nn) = time_cold_warm(|| {
        query_nndescent_index_gpu_self(&mut wgpu, cli.k, None, false)
            .unwrap()
            .0
    });
    // Entry selection is host work on both paths, so it sits inside the timing.
    let (mq_cold, mq_warm, mq_nn) = time_cold_warm(|| {
        let entries = wgpu.query_entry_points(&q_flat, n_q, n_entry);
        query_cagra_index_mlx(queries.as_ref(), &mlx, cli.k, None, Some(&entries), false)
            .unwrap()
            .0
    });
    let (ms_cold, ms_warm, ms_nn) = time_cold_warm(|| {
        let entries = self_entry_points(&knn_rows, wgpu.k, n, n_entry, 42);
        query_cagra_index_mlx_self(&mlx, cli.k, None, Some(&entries), false)
            .unwrap()
            .0
    });

    println!();
    println!(
        "wgpu NN-Descent + CAGRA build: {:.1} ms; MLX upload: {:.1} ms",
        wgpu_build, mlx_upload
    );
    println!(
        "{:<8} {:<6} {:>10} {:>10} {:>10}",
        "backend", "mode", "cold ms", "warm ms", "recall"
    );
    for (name, mode, cold, warm, truth, nn) in [
        ("wgpu", "query", wq_cold, wq_warm, &truth_q, &wq_nn),
        ("mlx", "query", mq_cold, mq_warm, &truth_q, &mq_nn),
        ("wgpu", "self", ws_cold, ws_warm, &truth_self, &ws_nn),
        ("mlx", "self", ms_cold, ms_warm, &truth_self, &ms_nn),
    ] {
        println!(
            "{:<8} {:<6} {:>10.1} {:>10.1} {:>10.4}",
            name,
            mode,
            cold,
            warm,
            calculate_recall(truth, nn, cli.k)
        );
    }
}
