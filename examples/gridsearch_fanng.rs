mod commons;
use ann_search_rs::fanng::FanngParams;
use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::utils::KnnValidation;
use ann_search_rs::*;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 25_000;
    const DIM: usize = 24;
    const N_CLUSTERS: usize = 20;
    const K: usize = 15;
    const SEED: u64 = 42;
    const CLUSTER_SD: f64 = 0.8;

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} cells, {} dimensions, {} clusters.",
        N_CELLS.separate_with_underscores(),
        DIM,
        N_CLUSTERS,
    );
    println!("-----------------------------");

    let data: Mat<f32> = generate_clustered_data(N_CELLS, DIM, N_CLUSTERS, CLUSTER_SD, SEED);
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

    // FANNG parameter grid
    let param_configs = [
        ("fast", FanngParams::fast()),
        ("balanced", FanngParams::balanced()),
        ("default", FanngParams::default()),
    ];

    // Query configurations
    let query_configs = [
        (K * 10, 0, "10x-no_sc"),
        (K * 20, 0, "20x-no_sc"),
        (K * 50, 0, "50x-no_sc"),
        (K * 10, 5, "10x-5sc"),
        (K * 20, 5, "20x-5sc"),
        (K * 50, 5, "50x-5sc"),
        (K * 10, 10, "10x-10sc"),
        (K * 20, 10, "20x-10sc"),
        (K * 50, 10, "50x-10sc"),
    ];

    for (param_name, params) in param_configs {
        println!(
            "Building FANNG index ({}, max_degree={}, tam={}, rn={})...",
            param_name,
            params.max_degree,
            params.traverse_add_multiplier,
            params.refinement_neighbour_no
        );
        let start = Instant::now();
        let fanng_idx = build_fanng_index(
            data.as_ref(),
            "euclidean",
            Some(params),
            SEED as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        for (max_calcs, num_shortcuts, query_label) in query_configs {
            println!("Querying FANNG index (config={})...", query_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_fanng_index(
                query_data,
                &fanng_idx,
                K,
                max_calcs,
                num_shortcuts,
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

            let internal_recall = fanng_idx.validate_index(K, SEED as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("FANNG-{}:{}", param_name, query_label),
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
