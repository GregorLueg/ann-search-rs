mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use tempfile::TempDir;
use thousands::*;

fn main() {
    let cli = Cli::parse();

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} cells, {} dimensions, {} clusters, {} dist.",
        cli.n_cells.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);
    let mut results = Vec::new();

    // Ground truth - Exhaustive
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
        mean_dist_err: 0.0,
        index_size_mb,
    });

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
        mean_dist_err: 0.0,
        index_size_mb,
    });

    println!("-----------------------------");

    // RaBitQ exhaustive benchmark with reranking
    let temp_dir = TempDir::new().unwrap();
    let rerank_factors = [5, 10, 20];

    println!("Building RaBitQ exhaustive index with reranking...");
    let start = Instant::now();
    let rabitq_idx = build_exhaustive_index_rabitq(
        data.as_ref(),
        None,
        &cli.distance,
        cli.seed as usize,
        true,
        Some(temp_dir.path()),
    )
    .unwrap();
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = rabitq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    for &rerank_factor in &rerank_factors {
        println!(
            "Querying RaBitQ exhaustive index (rerank_factor={})...",
            rerank_factor
        );
        let start = Instant::now();
        let (rabitq_neighbors, rabitq_distances) = query_exhaustive_index_rabitq(
            query_data.as_ref(),
            &rabitq_idx,
            cli.k,
            None,
            true,
            Some(rerank_factor),
            true,
            false,
        );
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &rabitq_neighbors, cli.k);
        let dist_error = calculate_dist_error(
            true_distances.as_ref().unwrap(),
            rabitq_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveRaBitQ-rf{} (query)", rerank_factor),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_err: dist_error,
            index_size_mb,
        });
    }

    println!("Self-querying RaBitQ exhaustive index...");
    let start = Instant::now();
    let (rabitq_neighbors_self, rabitq_distances_self) =
        query_exhaustive_index_rabitq_self(&rabitq_idx, cli.k, None, Some(10), true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &rabitq_neighbors_self, cli.k);
    let dist_error_self = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        rabitq_distances_self.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "ExhaustiveRaBitQ (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: recall_self,
        mean_dist_err: dist_error_self,
        index_size_mb,
    });

    println!("-----------------------------");

    print_results_size(
        &format!(
            "{}k cells, {}D - RaBitQ with Reranking",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
