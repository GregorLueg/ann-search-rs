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
        mean_dist_err: 0.0,
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
        mean_dist_err: 0.0,
        index_size_mb,
    });

    println!("-----------------------------");

    let build_params = [
        (Some(12), 0.0, vec![None]),
        (Some(24), 0.0, vec![None]),
        (None, 0.0, vec![Some(75), Some(100), None]),
        (None, 0.25, vec![None]),
        (None, 0.5, vec![None]),
        (None, 1.0, vec![None]),
    ];

    for (n_trees, diversify_prob, ef_search_values) in build_params {
        let n_trees_str = n_trees
            .map(|i| i.to_string())
            .unwrap_or_else(|| ":auto".to_string());

        println!(
            "Building NNDescent index (n_trees={}, diversify={})...",
            n_trees_str, diversify_prob
        );
        let start = Instant::now();
        let nndescent_idx = build_nndescent_index(
            data.as_ref(),
            &cli.distance,
            0.001,
            diversify_prob,
            None,
            None,
            None,
            n_trees,
            cli.seed as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Query benchmarks
        for ef_search in &ef_search_values {
            let ef_search_str = ef_search
                .map(|i| i.to_string())
                .unwrap_or_else(|| ":auto".to_string());

            println!("Querying NNDescent index (ef_search={})...", ef_search_str);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_nndescent_index(
                query_data.as_ref(),
                &nndescent_idx,
                cli.k,
                *ef_search,
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!(
                    "NNDescent-nt{}-s{}-dp{} (query)",
                    n_trees_str, ef_search_str, diversify_prob
                ),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
                index_size_mb,
            });
        }

        // Self-query benchmark
        println!("Self-querying NNDescent index...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_nndescent_self(&nndescent_idx, cli.k, None, true, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_distance_error(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("NNDescent-nt{}-dp{} (self)", n_trees_str, diversify_prob),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: dist_error_self,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
