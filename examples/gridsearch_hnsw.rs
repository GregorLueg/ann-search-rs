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
        mean_dist_err: 0.0,
    });

    println!("-----------------------------");

    let build_params = [
        (16, 50),
        (16, 100),
        (16, 200),
        (24, 100),
        (24, 200),
        (24, 300),
        (32, 200),
        (32, 300),
    ];

    for (m, ef_construction) in build_params {
        println!(
            "Building HNSW index (M={}, ef_construction={})...",
            m, ef_construction
        );
        let start = Instant::now();
        let hnsw_idx = build_hnsw_index(
            data.as_ref(),
            m,
            ef_construction,
            "euclidean",
            SEED as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let ef_search_values = vec![50, 75, 100];

        for ef_search in ef_search_values {
            println!("Querying HNSW index (ef_search={})...", ef_search);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_hnsw_index(query_data, &hnsw_idx, K, ef_search, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

            let internal_recall = hnsw_idx.validate_index(K, SEED as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("HNSW-M{}-ef{}-s{}", m, ef_construction, ef_search),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
            });
        }
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
