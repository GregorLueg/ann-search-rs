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

    let data_type = parse_data(&cli.data).unwrap_or_default();
    let data: Mat<f32> = match data_type {
        SyntheticData::GaussianNoise => {
            generate_clustered_data(cli.n_cells, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!("Using data for high dimensional ANN searches...\n");
            generate_clustered_data_high_dim(
                cli.n_cells,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
    };

    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);

    let mut results = Vec::new();

    // Exhaustive query benchmark
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
    });

    // Exhaustive self-query benchmark
    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&exhaustive_idx, cli.k, true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
    });

    println!("-----------------------------");

    let n_trees_values = [5, 10, 15, 25, 50, 75, 100];

    for n_trees in n_trees_values {
        println!("Building Annoy index ({} trees)...", n_trees);
        let start = Instant::now();
        let annoy_idx = build_annoy_index(
            data.as_ref(),
            cli.distance.as_str().into(),
            n_trees,
            cli.seed as usize,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let search_budgets = [
            (None, "auto"),
            (Some(cli.k * n_trees * 10), "10x"),
            (Some(cli.k * n_trees * 5), "5x"),
        ];

        // Query benchmarks
        for (search_budget, budget_label) in search_budgets {
            println!("Querying Annoy index (search_budget={})...", budget_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_annoy_index(
                query_data.as_ref(),
                &annoy_idx,
                cli.k,
                search_budget,
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

            results.push(BenchmarkResult {
                method: format!("Annoy-nt{}-s:{} (query)", n_trees, budget_label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
            });
        }

        // Self-query benchmark
        println!("Self-querying Annoy index...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_annoy_self(&annoy_idx, cli.k, None, true, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_distance_error(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResult {
            method: format!("Annoy-nt{} (self)", n_trees),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: dist_error_self,
        });
    }

    print_results(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
