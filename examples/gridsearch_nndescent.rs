mod commons;

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

    #[arg(long, default_value = DEFAULT_DATA)]
    data: String,
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

        for ef_search in ef_search_values {
            let ef_search_str = ef_search
                .map(|i| i.to_string())
                .unwrap_or_else(|| ":auto".to_string());

            println!("Querying NNDescent index (ef_search={})...", ef_search_str);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_nndescent_index(query_data, &nndescent_idx, cli.k, ef_search, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResult {
                method: format!(
                    "NNDescent-nt{}-s{}-dp{}",
                    n_trees_str, ef_search_str, diversify_prob
                ),
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
