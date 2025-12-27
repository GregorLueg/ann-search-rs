mod commons;

use ann_search_rs::utils::KnnValidation;
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
        (10, 8),
        (20, 8),
        (25, 8),
        (10, 10),
        (20, 10),
        (25, 10),
        (10, 12),
        (20, 12),
        (25, 12),
        (10, 16),
        (20, 16),
        (25, 16),
        (50, 16),
    ];

    for (num_tables, bits_per_hash) in build_params {
        println!(
            "Building LSH index (num_tab={}, bits={})...",
            num_tables, bits_per_hash
        );
        let start = Instant::now();
        let lsh_index = build_lsh_index(
            data.as_ref(),
            &cli.distance,
            num_tables,
            bits_per_hash,
            cli.seed as usize,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let search_budgets = [(None, "auto"), (Some(5000), "5k")];
        for (max_cand, cand_label) in search_budgets {
            println!("Querying LSH index (cand={})...", cand_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_lsh_index(query_data, &lsh_index, cli.k, max_cand, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            let internal_recall = lsh_index.validate_index(cli.k, cli.seed as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("LSH-nt{}-nb{}-s:{}", num_tables, bits_per_hash, cand_label),
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
