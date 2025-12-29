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

    // =========================================================================
    // Exhaustive index (ground truth)
    // =========================================================================
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, _) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, false, false);
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
    let (true_neighbors_self, _) = query_exhaustive_self(&exhaustive_idx, cli.k, false, false);
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

    println!("-----------------------------------------------------------------------------------------------");

    // =========================================================================
    // Binary index benchmarks
    // =========================================================================
    let n_bits_values = [16, 32, 64, 128, 256, 512, 768];

    for n_bits in n_bits_values {
        println!("Building exhaustive binary index (n_bits={})...", n_bits);
        let start = Instant::now();
        let binary_idx = build_exhaustive_index_binary(data.as_ref(), n_bits, cli.seed as usize);
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = binary_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Query benchmark
        println!("Querying exhaustive binary index (n_bits={})...", n_bits);
        let start = Instant::now();
        let (binary_neighbors, _) =
            query_exhaustive_index_binary(query_data.as_ref(), &binary_idx, cli.k, false, false);
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &binary_neighbors, cli.k);

        results.push(BenchmarkResultSize {
            method: format!("Binary-{} (query)", n_bits),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_err: f64::NAN,
            index_size_mb,
        });

        // Self-query benchmark
        println!(
            "Self-querying exhaustive binary index (n_bits={})...",
            n_bits
        );
        let start = Instant::now();
        let (binary_neighbors_self, _) =
            query_exhaustive_self_binary(data.as_ref(), &binary_idx, cli.k, false, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &binary_neighbors_self, cli.k);

        results.push(BenchmarkResultSize {
            method: format!("Binary-{} (self)", n_bits),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: f64::NAN,
            index_size_mb,
        });
    }

    print_results_size(
        &format!(
            "{}k cells, {}D (Exhaustive vs Binary)",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
