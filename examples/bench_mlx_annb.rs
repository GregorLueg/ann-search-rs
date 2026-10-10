//! wgpu against MLX on an ann-benchmarks dataset: exhaustive, IVF and the
//! NN-Descent/CAGRA index, test queries against the dataset's own ground
//! truth. Expects `train.bin`, `test.bin` (f32) and `neighbors.bin` (u32) in
//! `--dir`, each a `[rows: u32, cols: u32]` header followed by row-major data.
//!
//! Run with:
//! cargo run --example bench_mlx_annb --release --features gpu,mlx -- \
//!   --dir /path/to/fashion --metric euclidean --k 10

use ann_search_rs::*;
use clap::Parser;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Number of warm query runs timed after the cold one.
const WARM_RUNS: usize = 3;

/// Command line
#[derive(Parser)]
struct Args {
    /// Directory holding the three `.bin` files
    #[arg(long)]
    dir: PathBuf,
    /// "euclidean" or "cosine"
    #[arg(long, default_value = "euclidean")]
    metric: String,
    /// Neighbours per query
    #[arg(long, default_value_t = 10)]
    k: usize,
}

/// Read a `[rows, cols]`-headed binary file.
///
/// ### Params
///
/// * `path` - File to read
///
/// ### Returns
///
/// `(raw little-endian words, rows, cols)`
fn read_bin(path: &Path) -> (Vec<u32>, usize, usize) {
    let bytes = std::fs::read(path).expect("readable dataset file");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (rows, cols) = (words[0] as usize, words[1] as usize);
    (words[2..].to_vec(), rows, cols)
}

/// Recall@k against the first k ground-truth ids.
///
/// ### Params
///
/// * `truth` - Ground-truth ids, `cols` per row
/// * `cols` - Width of the ground-truth rows
/// * `found` - Returned neighbours per query
/// * `k` - Neighbours compared
///
/// ### Returns
///
/// Mean recall
fn recall(truth: &[u32], cols: usize, found: &[Vec<usize>], k: usize) -> f64 {
    let hits: usize = found
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let t = &truth[i * cols..i * cols + k];
            row.iter()
                .take(k)
                .filter(|&&j| t.contains(&(j as u32)))
                .count()
        })
        .sum();
    hits as f64 / (found.len() * k) as f64
}

/// Time a cold query then `WARM_RUNS` warm ones.
///
/// ### Params
///
/// * `f` - The query to time
///
/// ### Returns
///
/// `(cold ms, best warm ms, neighbours)`
fn time_query(mut f: impl FnMut() -> Vec<Vec<usize>>) -> (f64, f64, Vec<Vec<usize>>) {
    let t = Instant::now();
    let mut res = f();
    let cold = t.elapsed().as_secs_f64() * 1e3;
    let mut warm = f64::INFINITY;
    for _ in 0..WARM_RUNS {
        let t = Instant::now();
        res = f();
        warm = warm.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (cold, warm, res)
}

/// Print one result row.
///
/// ### Params
///
/// * `name` - Row label
/// * `build` - Build ms
/// * `q` - `(cold ms, warm ms, neighbours)` from [`time_query`]
/// * `truth` - Ground-truth ids
/// * `cols` - Ground-truth row width
/// * `k` - Neighbours compared
fn row(
    name: &str,
    build: f64,
    q: &(f64, f64, Vec<Vec<usize>>),
    truth: &[u32],
    cols: usize,
    k: usize,
) {
    println!(
        "{:<16} {:>10.1} {:>10.1} {:>10.1} {:>8.4}",
        name,
        build,
        q.0,
        q.1,
        recall(truth, cols, &q.2, k)
    );
}

fn main() {
    let a = Args::parse();
    let (train, n, dim) = read_bin(&a.dir.join("train.bin"));
    let (test, nq, _) = read_bin(&a.dir.join("test.bin"));
    let (truth, _, cols) = read_bin(&a.dir.join("neighbors.bin"));
    let train: Vec<f32> = train.into_iter().map(f32::from_bits).collect();
    let test: Vec<f32> = test.into_iter().map(f32::from_bits).collect();
    let (k, metric) = (a.k, a.metric.as_str());
    println!("{n} x {dim} train, {nq} queries, k {k}, {metric}");
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>8}",
        "index", "build ms", "cold ms", "warm ms", "recall"
    );
    let dev = WgpuDevice::default();

    // Exhaustive
    let t = Instant::now();
    let ex_g =
        build_exhaustive_index_gpu::<f32, WgpuRuntime>((&train[..], n, dim), metric, dev.clone())
            .unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_exhaustive_index_gpu((&test[..], nq, dim), &ex_g, k, false, false)
            .unwrap()
            .0
    });
    row("exhaustive wgpu", b, &q, &truth, cols, k);
    drop(ex_g);
    let t = Instant::now();
    let ex_m = build_exhaustive_index_mlx((&train[..], n, dim), metric).unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_exhaustive_index_mlx((&test[..], nq, dim), &ex_m, k, false, false)
            .unwrap()
            .0
    });
    row("exhaustive mlx", b, &q, &truth, cols, k);
    drop(ex_m);

    // IVF, default nlist and nprobe on both
    let t = Instant::now();
    let ivf_g = build_ivf_index_gpu::<f32, WgpuRuntime>(
        (&train[..], n, dim),
        None,
        None,
        metric,
        42,
        false,
        dev.clone(),
    )
    .unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_ivf_index_gpu((&test[..], nq, dim), &ivf_g, k, None, None, false, false)
            .unwrap()
            .0
    });
    row("ivf wgpu", b, &q, &truth, cols, k);
    drop(ivf_g);
    let t = Instant::now();
    let ivf_m = build_ivf_index_mlx((&train[..], n, dim), None, None, metric, 42, false).unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_ivf_index_mlx((&test[..], nq, dim), &ivf_m, k, None, None, false, false)
            .unwrap()
            .0
    });
    row("ivf mlx", b, &q, &truth, cols, k);
    drop(ivf_m);

    // NN-Descent + CAGRA, all defaults on both
    let t = Instant::now();
    let mut nnd_g = build_nndescent_index_gpu::<f32, WgpuRuntime>(
        (&train[..], n, dim),
        metric,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        42,
        false,
        true,
        dev,
    )
    .unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_nndescent_index_gpu((&test[..], nq, dim), &mut nnd_g, k, None, false, false)
            .unwrap()
            .0
    });
    row("cagra wgpu", b, &q, &truth, cols, k);
    drop(nnd_g);
    let t = Instant::now();
    let nnd_m = build_nndescent_index_mlx(
        (&train[..], n, dim),
        metric,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        42,
        false,
    )
    .unwrap();
    let b = t.elapsed().as_secs_f64() * 1e3;
    let q = time_query(|| {
        query_nndescent_index_mlx((&test[..], nq, dim), &nnd_m, k, None, false, false)
            .unwrap()
            .0
    });
    row("cagra mlx", b, &q, &truth, cols, k);
}
