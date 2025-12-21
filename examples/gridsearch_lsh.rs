mod commons;

use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::utils::KnnValidation;
use ann_search_rs::*;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 250_000;
    const DIM: usize = 24;
    const N_CLUSTERS: usize = 20;
    const K: usize = 15;
    const SEED: u64 = 42;

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} cells, {} dimensions, {} clusters.",
        N_CELLS.separate_with_underscores(),
        DIM,
        N_CLUSTERS,
    );
    println!("-----------------------------");

    let data: Mat<f32> = generate_clustered_data(N_CELLS, DIM, N_CLUSTERS, SEED);
    let query_data = data.as_ref();
    let mut results = Vec::new();

    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), "euclidean");
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data, &exhaustive_idx, K, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_distance_error: 0.0,
    });

    println!("-----------------------------");

    let build_params = [
        (10, 8),
        (20, 8),
        (50, 8),
        (10, 10),
        (20, 10),
        (50, 10),
        (10, 12),
        (20, 12),
        (50, 12),
        (10, 16),
        (20, 16),
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
            "euclidean",
            num_tables,
            bits_per_hash,
            SEED as usize,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let search_budgets = [
            (None, "auto"),
            (Some(1000), "1k_cand"),
            (Some(5000), "5k_cand"),
        ];

        for (max_cand, cand_label) in search_budgets {
            println!("Querying LSH index (cand={})...", cand_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_lsh_index(query_data, &lsh_index, K, max_cand, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

            let internal_recall = lsh_index.validate_index(K, SEED as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("LSH-nt{}-bits{}:{}", num_tables, bits_per_hash, cand_label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_distance_error: dist_error,
            });
        }
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
