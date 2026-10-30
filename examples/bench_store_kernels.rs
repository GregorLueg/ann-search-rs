//! Microbenchmark for the vector store's re-ranking distances, per encoding.
//!
//! Re-ranking reads rows in random order, one query against a few hundred
//! candidates, so that is the access pattern here: a shuffled walk over a
//! working set a few times L2, single-threaded. The store is warm in the page
//! cache; a cold store is a disk measurement, not a kernel one.

use std::hint::black_box;
use std::time::Instant;

use ann_search_rs::binary::vec_store::VectorStore;
use ann_search_rs::utils::dist::detect_simd_level;
use ann_search_rs::*;
use tempfile::TempDir;

/// Dimensionalities to measure, spanning PCA-sized through embedding-sized.
const DIMS: [usize; 5] = [32, 64, 128, 256, 768];

/// Working set of the full-precision rows per measurement.
const WORKING_SET_BYTES: usize = 32 << 20;

/// Timed passes per measurement. The best is reported.
const PASSES: usize = 7;

/// Deterministic filler, so every encoding sees identical data.
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
fn best_pass(mut pass: impl FnMut() -> f32) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..PASSES {
        let start = Instant::now();
        black_box(pass());
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    println!("SIMD level: {:?}", detect_simd_level());
    println!(
        "{:<10} {:>6} {:>8} {:>14} {:>14}",
        "encoding", "dim", "rows", "eucl ns/row", "cos ns/row"
    );
    println!("{}", "-".repeat(56));

    for dim in DIMS {
        let n = WORKING_SET_BYTES / (dim * 4);
        let data = fill(n * dim, 0x5eed);
        let query = fill(dim, 0xc0ffee);
        let qnorm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norms: Vec<f32> = data
            .chunks_exact(dim)
            .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        // Fixed shuffle of the row order: the re-ranking access pattern
        let mut order: Vec<usize> = (0..n).collect();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for i in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            order.swap(i, (state % (i as u64 + 1)) as usize);
        }

        for enc in [StoreEncoding::Full, StoreEncoding::F16] {
            let dir = TempDir::new().unwrap();
            let store = VectorStore::create(dir.path(), &data, &norms, n, dim, enc).unwrap();

            let eucl = best_pass(|| {
                order
                    .iter()
                    .map(|&i| store.euclidean_distance_to_query(i, &query))
                    .sum()
            });
            let cos = best_pass(|| {
                order
                    .iter()
                    .map(|&i| store.cosine_distance_to_query(i, &query, qnorm))
                    .sum()
            });

            println!(
                "{:<10} {:>6} {:>8} {:>14.1} {:>14.1}",
                format!("{enc:?}"),
                dim,
                n,
                eucl * 1e9 / n as f64,
                cos * 1e9 / n as f64
            );
        }
    }
}
