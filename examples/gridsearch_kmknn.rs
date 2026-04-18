mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

fn main() {
    let cli = Cli::parse();

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} samples, {} dimensions, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);
    let mut results = Vec::new();

    // Exhaustive query benchmark
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        index_size_mb,
    });

    // Exhaustive self-query benchmark
    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&exhaustive_idx, cli.k, true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        index_size_mb,
    });

    println!("-----------------------------");

    let nlist_values = [
        (cli.n_samples as f32 * 0.5).sqrt() as usize,
        (cli.n_samples as f32).sqrt() as usize,
        (cli.n_samples as f32 * 2.0).sqrt() as usize,
    ];

    for nlist in nlist_values {
        println!("Building kMkNN index (nlist={})...", nlist);
        let start = Instant::now();
        let kmknn_idx = build_kmknn_index(
            data.as_ref(),
            &cli.distance,
            Some(nlist),
            None,
            cli.seed as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = kmknn_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Query benchmark
        println!("Querying kMkNN index (nlist={})...", nlist);
        let start = Instant::now();
        let (exact_neighbors, exact_distances) =
            query_kmknn_index(query_data.as_ref(), &kmknn_idx, cli.k, true, false);
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        // Recall should be 1.0 since kMkNN is exact; compute anyway as a
        // sanity check and to catch any correctness regressions
        let recall = calculate_recall(&true_neighbors, &exact_neighbors, cli.k);
        let dist_error = calculate_mean_distance_ratio(
            true_distances.as_ref().unwrap(),
            exact_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("kMkNN-nl{} (query)", nlist),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_rat: dist_error,
            index_size_mb,
        });

        // Self-query benchmark
        println!("Self-querying kMkNN index (nlist={})...", nlist);
        let start = Instant::now();
        let (exact_neighbors_self, exact_distances_self) =
            query_kmknn_self(&kmknn_idx, cli.k, true, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &exact_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            exact_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("kMkNN-nl{} (self)", nlist),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
