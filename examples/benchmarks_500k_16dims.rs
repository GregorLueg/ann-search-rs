mod commons;

use faer::Mat;
use std::time::Instant;
use thousands::*;

use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::*;

use commons::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 500_000;
    const DIM: usize = 16;
    const N_CLUSTERS: usize = 20;
    const K: usize = 15;
    const SEED: u64 = 42;

    println!(
        "Generating synthetic data: {} cells, {} dimensions",
        N_CELLS.separate_with_underscores(),
        DIM
    );

    println!("-----------------------------");

    let data: Mat<f32> = generate_clustered_data(N_CELLS, DIM, N_CLUSTERS, 2.0, SEED);
    let query_data = data.as_ref();

    let mut results = Vec::new();

    // Exhaustive search (ground truth)
    println!("Building exhaustive index...");
    let start_total = Instant::now();
    let start = std::time::Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), "euclidean");
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = std::time::Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data, &exhaustive_idx, K, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;
    let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: end_total,
        recall_at_k: 1.0,
        mean_distance_error: 0.0,
    });

    println!("-----------------------------");

    // Annoy index
    for n_trees in [5, 10, 25, 50, 100] {
        println!("Building Annoy index ({} trees)...", n_trees);
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let annoy_idx = build_annoy_index(data.as_ref(), n_trees, SEED as usize);
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Querying Annoy index ({} trees)...", n_trees);
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) = query_annoy_index(
            query_data,
            &annoy_idx,
            K,
            "euclidean",
            n_trees * 4,
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
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResult {
            method: format!("Annoy-{}", n_trees),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    println!("-----------------------------");

    // HNSW
    // HNSW with different parameter combinations
    for (m, ef_construction, ef_search) in [
        (16, 100, 50),
        (16, 100, 100),
        (16, 200, 100),
        (16, 200, 200),
        (32, 200, 100),
        (32, 200, 200),
    ] {
        println!(
            "Building HNSW index (M={}, ef_construction={})...",
            m, ef_construction
        );
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let hnsw_idx = build_hnsw_index(
            data.as_ref(),
            m,
            ef_construction,
            "euclidean",
            SEED as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Querying HNSW index (ef_search={})...", ef_search);
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) =
            query_hnsw_index(query_data, &hnsw_idx, K, ef_search, true, false);
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
        let dist_error = calculate_distance_error(
            true_distances.as_ref().unwrap(),
            approx_distances.as_ref().unwrap(),
            K,
        );
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResult {
            method: format!("HNSW-M{}-ef{}-s{}", m, ef_construction, ef_search),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    println!("-----------------------------");

    // NNDescent with different parameters
    for (max_iter, rho) in [(10, 0.5), (25, 0.5), (25, 1.0), (50, 1.0)] {
        println!("Running NNDescent (max_iter={}, rho={})...", max_iter, rho);
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) = generate_knn_nndescent_with_dist(
            query_data,
            "euclidean",
            K,
            max_iter,
            0.001,
            rho,
            SEED as usize,
            false,
            true,
        );
        let total_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
        let dist_error = calculate_distance_error(
            true_distances.as_ref().unwrap(),
            approx_distances.as_ref().unwrap(),
            K,
        );
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResult {
            method: format!("NNDescent-i{}-r{}", max_iter, rho),
            build_time_ms: total_time,
            query_time_ms: 0.0,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    println!("-----------------------------");

    // FANNG with different search parameters
    for (max_calcs, no_shortcuts) in [(100, 20), (200, 20), (500, 20), (1000, 20)] {
        println!("Building FANNG index...");
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let fanng_idx = build_fanng_index(data.as_ref(), "euclidean", None, SEED as usize, false);
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "Querying FANNG index (max_calcs={}, shortcuts={})...",
            max_calcs, no_shortcuts
        );
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) = query_fanng_index(
            query_data,
            &fanng_idx,
            K,
            max_calcs,
            no_shortcuts,
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
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResult {
            method: format!("FANNG-c{}-s{}", max_calcs, no_shortcuts),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
