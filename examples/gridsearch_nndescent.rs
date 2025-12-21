mod commons;

use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::*;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 500_000;
    const DIM: usize = 24;
    const N_CLUSTERS: usize = 20;
    const K: usize = 15;
    const SEED: u64 = 42;
    const DISTANCE: &str = "euclidean";

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} cells, {} dimensions, {} clusters, {} dist.",
        N_CELLS.separate_with_underscores(),
        DIM,
        N_CLUSTERS,
        DISTANCE
    );
    println!("-----------------------------");

    let data: Mat<f32> = generate_clustered_data(N_CELLS, DIM, N_CLUSTERS, SEED);
    let query_data = data.as_ref();
    let mut results = Vec::new();

    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), DISTANCE);
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
        (Some(12), 0.0, vec![None]),
        (Some(24), 0.0, vec![None]),
        (None, 0.0, vec![Some(50), Some(100), None]),
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
            DISTANCE,
            0.001,
            diversify_prob,
            None,
            None,
            None,
            n_trees,
            SEED as usize,
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
                query_nndescent_index(query_data, &nndescent_idx, K, ef_search, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

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
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
            });
        }
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
