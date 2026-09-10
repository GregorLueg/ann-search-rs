//! Microbenchmark for the SIMD distance kernels in `utils::dist`.
//!
//! The gridsearch examples cannot price these. Exhaustive search runs through
//! faer's `matmul` once the query batch is large enough, k-means assigns
//! through it too, and the graph walks that do call the kernels are bound by
//! the pointer chase and the result heap rather than the arithmetic. So the
//! kernels need measuring on their own.
//!
//! Set `ANN_SEARCH_SIMD` to `scalar`, `sse`, `avx2` or `avx512` to force a
//! path, and run the binary once per level. The level is process-wide and
//! resolved on first use, so it cannot be switched inside one run. On x86-64
//! `sse` is what the crate emitted for every level before the kernels were
//! written against `std::arch`, which makes `sse` against `avx2` the honest
//! before-and-after.
//!
//! Vectors are laid out row-major in a working set a few times L2, and each
//! pass walks all of them, so the numbers include a realistic stream of loads
//! rather than one row pinned in L1.

use std::hint::black_box;
use std::time::Instant;

use ann_search_rs::utils::dist::{detect_simd_level, SimdDistance};

/// Dimensionalities to measure, spanning PCA-sized through embedding-sized.
const DIMS: [usize; 6] = [32, 64, 96, 128, 256, 768];

/// Working set per measurement. Comfortably past L2 on a runner, so the loads
/// are streamed rather than served from L1.
const WORKING_SET_BYTES: usize = 8 << 20;

/// Timed passes per measurement. The best is reported: on a shared runner the
/// distribution has a hard floor and a long tail, and the floor is the signal.
const PASSES: usize = 7;

/// Deterministic filler, so every level sees identical data.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / 8_388_608.0 - 1.0
        })
        .collect()
}

/// Best wall-clock time over `PASSES` runs of one full pass.
fn best_pass(mut pass: impl FnMut() -> f64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..PASSES {
        let start = Instant::now();
        let acc = pass();
        let elapsed = start.elapsed().as_secs_f64();
        black_box(acc);
        best = best.min(elapsed);
    }
    best
}

fn main() {
    println!("SIMD level: {:?}", detect_simd_level());
    println!(
        "{:<22} {:>6} {:>12} {:>14}",
        "kernel", "dim", "ns/call", "GB/s"
    );
    println!("{}", "-".repeat(58));

    for dim in DIMS {
        let rows = (WORKING_SET_BYTES / (dim * 4)).max(64);
        let data32 = fill(rows * dim, 0x5eed);
        let query32 = fill(dim, 0xc0ffee);
        let data64: Vec<f64> = data32.iter().map(|&x| x as f64).collect();
        let query64: Vec<f64> = query32.iter().map(|&x| x as f64).collect();

        // Two bytes-per-call figures, because f64 moves twice the bytes.
        let gb32 = (rows * dim * 4) as f64 / 1e9;
        let gb64 = (rows * dim * 8) as f64 / 1e9;

        let report = |name: &str, secs: f64, calls: usize, gb: f64| {
            println!(
                "{:<22} {:>6} {:>12.2} {:>14.2}",
                name,
                dim,
                secs / calls as f64 * 1e9,
                gb / secs
            );
        };

        let q = black_box(&query32[..]);
        let secs = best_pass(|| {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += f32::euclidean_simd(q, &data32[r * dim..(r + 1) * dim]);
            }
            acc as f64
        });
        report("euclidean f32", secs, rows, gb32);

        let secs = best_pass(|| {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += f32::dot_simd(q, &data32[r * dim..(r + 1) * dim]);
            }
            acc as f64
        });
        report("dot f32", secs, rows, gb32);

        let secs = best_pass(|| {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += f32::manhattan_simd(q, &data32[r * dim..(r + 1) * dim]);
            }
            acc as f64
        });
        report("manhattan f32", secs, rows, gb32);

        // Four rows per call, so the per-call figure is divided by `rows`, not
        // by the number of calls, to stay comparable with the single-row rows.
        let groups = rows / 4;
        let secs = best_pass(|| {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let b = g * 4 * dim;
                let y = [
                    &data32[b..b + dim],
                    &data32[b + dim..b + 2 * dim],
                    &data32[b + 2 * dim..b + 3 * dim],
                    &data32[b + 3 * dim..b + 4 * dim],
                ];
                let out = f32::euclidean_simd_batch_4(q, y);
                acc += out[0] + out[1] + out[2] + out[3];
            }
            acc as f64
        });
        report("euclidean f32 batch4", secs, groups * 4, gb32);

        let q = black_box(&query64[..]);
        let secs = best_pass(|| {
            let mut acc = 0.0f64;
            for r in 0..rows {
                acc += f64::euclidean_simd(q, &data64[r * dim..(r + 1) * dim]);
            }
            acc
        });
        report("euclidean f64", secs, rows, gb64);

        let secs = best_pass(|| {
            let mut acc = 0.0f64;
            for r in 0..rows {
                acc += f64::dot_simd(q, &data64[r * dim..(r + 1) * dim]);
            }
            acc
        });
        report("dot f64", secs, rows, gb64);
    }
}
