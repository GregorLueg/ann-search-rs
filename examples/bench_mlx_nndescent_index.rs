//! Full NN-Descent + CAGRA pipeline: MLX against cubecl/wgpu, CPU exhaustive
//! as ground truth. Times the index build (NN-Descent plus CAGRA
//! optimisation), an external query and a self query.
//!
//! Run with:
//! cargo run --example bench_mlx_nndescent_index --release --features gpu,mlx,synthetic -- \
//!   --n-samples 10000 --dim 32 --data cell

mod commons;

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
/// * `f` - The call to time
///
/// ### Returns
///
/// `(cold ms, best warm ms, last result)`
fn time_cold_warm<R>(mut f: impl FnMut() -> R) -> (f64, f64, R) {
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
    let k = cli.k;
    let seed = cli.seed as usize;

    let start = Instant::now();
    let cpu = build_exhaustive_index(data.as_ref(), &cli.distance);
    let (truth_q, _) = query_exhaustive_index(queries.as_ref(), &cpu, k, false, false).unwrap();
    let (truth_self, _) = query_exhaustive_self(&cpu, k, false, false).unwrap();
    println!(
        "CPU exhaustive (ground truth): {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    // Index degree 30 (the default), queries at k.
    let (w_build_cold, w_build_warm, mut wgpu) = time_cold_warm(|| {
        build_nndescent_index_gpu::<f32, WgpuRuntime>(
            data.as_ref(),
            &cli.distance,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            seed,
            false,
            true,
            WgpuDevice::DefaultDevice,
        )
        .unwrap()
    });
    let (w_q_cold, w_q_warm, w_q) = time_cold_warm(|| {
        query_nndescent_index_gpu(queries.as_ref(), &mut wgpu, k, None, false, false)
            .unwrap()
            .0
    });
    let (w_s_cold, w_s_warm, w_s) = time_cold_warm(|| {
        query_nndescent_index_gpu_self(&mut wgpu, k, None, false)
            .unwrap()
            .0
    });

    let (m_build_cold, m_build_warm, mlx) = time_cold_warm(|| {
        build_nndescent_index_mlx(
            data.as_ref(),
            &cli.distance,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            seed,
            false,
        )
        .unwrap()
    });
    let (m_q_cold, m_q_warm, m_q) = time_cold_warm(|| {
        query_nndescent_index_mlx(queries.as_ref(), &mlx, k, None, false, false)
            .unwrap()
            .0
    });
    let (m_s_cold, m_s_warm, m_s) = time_cold_warm(|| {
        query_nndescent_index_mlx_self(&mlx, k, None, false)
            .unwrap()
            .0
    });

    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "backend",
        "build ms",
        "build wm",
        "query ms",
        "query wm",
        "q recall",
        "self ms",
        "self wm",
        "s recall"
    );
    for (name, b, q, s, qn, sn) in [
        (
            "wgpu",
            (w_build_cold, w_build_warm),
            (w_q_cold, w_q_warm),
            (w_s_cold, w_s_warm),
            &w_q,
            &w_s,
        ),
        (
            "mlx",
            (m_build_cold, m_build_warm),
            (m_q_cold, m_q_warm),
            (m_s_cold, m_s_warm),
            &m_q,
            &m_s,
        ),
    ] {
        println!(
            "{:<8} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.4} {:>10.1} {:>10.1} {:>10.4}",
            name,
            b.0,
            b.1,
            q.0,
            q.1,
            calculate_recall(&truth_q, qn, k),
            s.0,
            s.1,
            calculate_recall(&truth_self, sn, k),
        );
    }
    println!("(cold = first call, wm = best of {WARM_RUNS} warm)");
}
