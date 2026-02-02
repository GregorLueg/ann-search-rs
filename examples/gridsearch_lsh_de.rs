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

    // LSH DE-Tree parameters
    // Format: (num_trees, proj_dims)
    let build_params = [
        (5, 8),
        (10, 8),
        (15, 8),
        (5, 10),
        (10, 10),
        (15, 10),
        (5, 12),
        (10, 12),
        (15, 12),
    ];

    for (num_trees, proj_dims) in build_params {
        println!(
            "Building LSH DE-Tree index (num_trees={}, proj_dims={})...",
            num_trees, proj_dims
        );
        let start = Instant::now();
        let detree_index = build_lsh_de_tree_index(
            data.as_ref(),
            &cli.distance,
            num_trees,
            proj_dims,
            None, // Use default max_leaf_size
            cli.seed as usize,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = detree_index.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Adaptive radius queries
        // Starting radius: small value, typically 0.5-1.0
        // Max radius: based on projected dimensionality, roughly 2*sqrt(proj_dims)
        let max_radius = 2.0 * (proj_dims as f32).sqrt();
        let adaptive_params = [
            (0.5, 1.5, "init0.5-exp1.5"),
            (0.5, 2.0, "init0.5-exp2.0"),
            (1.0, 1.5, "init1.0-exp1.5"),
            (1.0, 2.0, "init1.0-exp2.0"),
        ];

        for (initial_radius, expansion_factor, label) in adaptive_params {
            println!("Querying LSH DE-Tree (adaptive: {})...", label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_lsh_de_tree_index(
                query_data.as_ref(),
                &detree_index,
                cli.k,
                true, // adaptive
                None,
                Some(initial_radius),
                Some(expansion_factor),
                Some(max_radius),
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("LSHDET-t{}-p{}-{} (query)", num_trees, proj_dims, label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
                index_size_mb,
            });
        }

        // Fixed radius queries
        // Typical radii based on projected dimensionality
        let base_radius = (proj_dims as f32).sqrt();
        let fixed_radii = [
            (base_radius, "r:sqrt(K)"),
            (1.5 * base_radius, "r:1.5sqrt(K)"),
            (2.0 * base_radius, "r:2sqrt(K)"),
        ];

        for (radius, label) in fixed_radii {
            println!("Querying LSH DE-Tree (fixed: {})...", label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_lsh_de_tree_index(
                query_data.as_ref(),
                &detree_index,
                cli.k,
                false, // not adaptive
                Some(radius),
                None,
                None,
                None,
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("LSHDET-t{}-p{}-{} (query)", num_trees, proj_dims, label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
                index_size_mb,
            });
        }

        // Self-query with adaptive radius
        println!("Self-querying LSH DE-Tree (adaptive)...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) = query_lsh_de_tree_index_self(
            &detree_index,
            cli.k,
            true, // adaptive
            None,
            Some(1.0),
            Some(1.5),
            Some(max_radius),
            true,
            false,
        );
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_dist_error(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("LSHDET-t{}-p{}-adaptive (self)", num_trees, proj_dims),
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
