#![allow(dead_code)]

use num_traits::{Float, ToPrimitive};
use rustc_hash::FxHashSet;

/// BenchmarkResult
///
/// ### Fields
///
/// * `method` - Name of the method
/// * `build_time_ms` - The build time of the index in ms
/// * `query_time_ms` - The query time of the index in ms
/// * `total_time_ms` - Total time the index build & query takes in ms
/// * `recall_at_k` - Recall@k neighbours against ground truth
/// * `mean_dist_err` - Mean distance error against ground truth
pub struct BenchmarkResult {
    pub method: String,
    pub build_time_ms: f64,
    pub query_time_ms: f64,
    pub total_time_ms: f64,
    pub recall_at_k: f64,
    pub mean_dist_err: f64,
}

/// Calculate Recall@k
///
/// ### Params
///
/// * `true_neighbors` - Slice of true neighbours
/// * `approx_neighbors` - Slice of the approximate neighbours
/// * `k` - Number of selected k
///
/// ### Returns
///
/// The Recall@k
pub fn calculate_recall(
    true_neighbors: &[Vec<usize>],
    approx_neighbors: &[Vec<usize>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;

    for (true_nn, approx_nn) in true_neighbors.iter().zip(approx_neighbors.iter()) {
        let true_set: FxHashSet<_> = true_nn.iter().take(k).collect();
        let matches = approx_nn
            .iter()
            .take(k)
            .filter(|&idx| true_set.contains(idx))
            .count();
        total_recall += matches as f64 / k as f64;
    }

    total_recall / true_neighbors.len() as f64
}

/// Calculate mean distance error
///
/// ### Params
///
/// * `true_dist` - Slice of true distances to the neighbours
/// * `approx_dist` - Slice of approximate distances to the neighbours
/// * `k` - Number of selected k
///
/// ### Returns
///
/// The mean distance error
pub fn calculate_distance_error<T>(true_dist: &[Vec<T>], approx_dist: &[Vec<T>], k: usize) -> f64
where
    T: Float + ToPrimitive,
{
    let mut total_error = 0.0;

    for (true_dist, approx_dist) in true_dist.iter().zip(approx_dist.iter()) {
        for i in 0..k.min(true_dist.len()).min(approx_dist.len()) {
            let error = (true_dist[i].to_f64().unwrap() - approx_dist[i].to_f64().unwrap()).abs();
            total_error += error;
        }
    }

    total_error / (true_dist.len() * k) as f64
}

/// Helper to print results to console
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>95}", "");
    println!("Benchmark: {}", config);
    println!("{:=>95}", "");
    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k", "Dist Error"
    );
    println!("{:->95}", "");
    for result in results {
        println!(
            "{:<30} {:>12.2} {:>12.2} {:>12.2} {:>12.4} {:>12.6}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k,
            result.mean_dist_err
        );
    }
    println!("{:->95}\n", "");
}

/// Helper to print results to console (Recall only)
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results_recall_only(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>83}", "");
    println!("Benchmark: {}", config);
    println!("{:=>83}", "");
    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k"
    );
    println!("{:->83}", "");
    for result in results {
        println!(
            "{:<30} {:>12.2} {:>12.2} {:>12.2} {:>12.4}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k
        );
    }
    println!("{:->83}\n", "");
}
