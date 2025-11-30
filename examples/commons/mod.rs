use num_traits::{Float, ToPrimitive};
use rustc_hash::FxHashSet;

pub struct BenchmarkResult {
    pub method: String,
    pub build_time_ms: f64,
    pub query_time_ms: f64,
    pub recall_at_k: f64,
    pub mean_distance_error: f64,
}

pub fn calculate_recall<T>(
    true_neighbors: &[Vec<usize>],
    approx_neighbors: &[Vec<usize>],
    k: usize,
) -> f64
where
    T: Float,
{
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

pub fn calculate_distance_error<T>(
    true_distances: &[Vec<T>],
    approx_distances: &[Vec<T>],
    k: usize,
) -> f64
where
    T: Float + ToPrimitive,
{
    let mut total_error = 0.0;

    for (true_dist, approx_dist) in true_distances.iter().zip(approx_distances.iter()) {
        for i in 0..k.min(true_dist.len()).min(approx_dist.len()) {
            let error = (true_dist[i].to_f64().unwrap() - approx_dist[i].to_f64().unwrap()).abs();
            total_error += error;
        }
    }

    total_error / (true_distances.len() * k) as f64
}

pub fn print_results(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>80}", "");
    println!("Benchmark: {}", config);
    println!("{:=>80}", "");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Recall@k", "Dist Error"
    );
    println!("{:->80}", "");

    for result in results {
        println!(
            "{:<15} {:>12.2} {:>12.2} {:>12.4} {:>12.6}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.recall_at_k,
            result.mean_distance_error
        );
    }
    println!("{:->80}\n", "");
}
