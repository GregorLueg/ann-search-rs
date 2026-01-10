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
    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    let build_time = start.elapsed().as_secs_f64() * 1000.0;

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

    let search_budgets = [
        ("1%", (0.01 * cli.n_cells as f32) as usize),
        ("2%", (0.02 * cli.n_cells as f32) as usize),
        ("5%", (0.05 * cli.n_cells as f32) as usize),
        ("10%", (0.1 * cli.n_cells as f32) as usize),
        ("15%", (0.15 * cli.n_cells as f32) as usize),
        ("20%", (0.2 * cli.n_cells as f32) as usize),
    ];

    println!("Building BallTree index...");
    let start = Instant::now();
    let balltree_idx = build_balltree_index(data.as_ref(), cli.distance, cli.seed as usize);
    let index_size_mb = balltree_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    for (search_budget_str, search_budget) in search_budgets {
        println!(
            "Querying BallTree index with search budget {}...",
            search_budget_str
        );
        let start = Instant::now();
        let (approx_neighbours, approx_distances) = query_balltree_index(
            query_data.as_ref(),
            &balltree_idx,
            cli.k,
            Some(search_budget),
            true,
            false,
        );
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &approx_neighbours, cli.k);
        let dist_error = calculate_dist_error(
            true_distances.as_ref().unwrap(),
            approx_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("BallTree-s:{} (query)", search_budget_str),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_err: dist_error,
            index_size_mb,
        });
    }

    let start = Instant::now();
    let (approx_neighbours_self, approx_dist_self) =
        query_balltree_self(&balltree_idx, cli.k, None, true, false);

    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbours_self, cli.k);
    let dist_error_self = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        approx_dist_self.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "BallTree (self)".into(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: recall_self,
        mean_dist_err: dist_error_self,
        index_size_mb,
    });

    print_results_size(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
