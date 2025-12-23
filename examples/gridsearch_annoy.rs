mod commons;

use ann_search_rs::utils::KnnValidation;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = DEFAULT_N_CELLS)]
    n_cells: usize,

    #[arg(long, default_value_t = DEFAULT_DIM)]
    dim: usize,

    #[arg(long, default_value_t = DEFAULT_N_CLUSTERS)]
    n_clusters: usize,

    #[arg(long, default_value_t = DEFAULT_K)]
    k: usize,

    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    #[arg(long, default_value = DEFAULT_DISTANCE)]
    distance: String,
}

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

    let data: Mat<f32> = generate_clustered_data(cli.n_cells, cli.dim, cli.n_clusters, cli.seed);
    let query_data = data.as_ref();
    let mut results = Vec::new();

    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data, &exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
    });

    println!("-----------------------------");
    let n_trees_values = [5, 10, 15, 25, 50, 75];
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

        for (search_budget, budget_label) in search_budgets {
            println!("Querying Annoy index (search_budget={})...", budget_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_annoy_index(query_data, &annoy_idx, cli.k, search_budget, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            let internal_recall = annoy_idx.validate_index(cli.k, cli.seed as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("Annoy-nt{}-s:{}", n_trees, budget_label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
            });
        }
    }

    print_results(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
