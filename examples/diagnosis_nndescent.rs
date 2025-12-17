mod commons;

use faer::Mat;
use std::time::Instant;
use thousands::*;

use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::*;

use commons::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 150_000;
    const DIM: usize = 32;
    const N_CLUSTERS: usize = 20;
    const K: usize = 10;
    const SEED: u64 = 42;

    println!("-----------------------------");
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

    for (n_trees, ef_search, diversify_prob) in [
        (Some(12), None, 0.0),
        (Some(24), None, 0.0),
        (None, Some(50), 0.0),
        (None, Some(100), 0.0),
        (None, None, 0.0),
        (None, None, 0.5),
        (None, None, 1.0),
    ] {
        let n_trees_str = n_trees
            .map(|i| i.to_string())
            .unwrap_or_else(|| ":auto".to_string());

        let ef_search_str = ef_search
            .map(|i| i.to_string())
            .unwrap_or_else(|| ":auto".to_string());

        println!(
            "Building NNDescent index (n_trees={}, ef_search={:?}, diversify={})...",
            n_trees_str, ef_search_str, diversify_prob
        );
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let nndescent_idx = build_nndescent_index(
            data.as_ref(),
            "euclidean",
            0.001,
            diversify_prob,
            None,
            None,
            None,
            n_trees,
            SEED as usize,
            true,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Querying NNDescent index (ef_search={})...", ef_search_str);
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) =
            query_nndescent_index(query_data, &nndescent_idx, K, ef_search, true, true);
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
        let dist_error = calculate_distance_error(
            true_distances.as_ref().unwrap(),
            approx_distances.as_ref().unwrap(),
            K,
        );
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResult {
            method: format!(
                "NNDescent-nt{}-s{}-dp{}",
                n_trees_str,
                ef_search_str,
                if diversify_prob > 0.5 {
                    1
                } else if diversify_prob > 0.0 {
                    5
                } else {
                    0
                }
            ),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
