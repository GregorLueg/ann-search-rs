mod commons;

use ann_search_rs::utils::KnnValidation;
use faer::Mat;
use std::time::Instant;
use thousands::*;

use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::*;

use commons::*;

fn main() {
    // test parameters
    const N_CELLS: usize = 100_000;
    const DIM: usize = 16;
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

    /////////////////
    // Annoy index //
    /////////////////

    println!("-----------------------------");

    for n_trees in [5, 10, 15, 25, 50, 100] {
        println!("Building Annoy index ({} trees)...", n_trees);
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let annoy_idx =
            build_annoy_index(data.as_ref(), "euclidean".into(), n_trees, SEED as usize);
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Querying Annoy index ({} trees)...", n_trees,);
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) =
            query_annoy_index(query_data, &annoy_idx, K, None, true, false);
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
        let dist_error = calculate_distance_error(
            true_distances.as_ref().unwrap(),
            approx_distances.as_ref().unwrap(),
            K,
        );
        let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

        let internal_recall = annoy_idx.validate_index(K, SEED as usize, None);

        println!(
            " Internal validation returned {:.3} as a score.",
            internal_recall
        );

        results.push(BenchmarkResult {
            method: format!("Annoy-nt{}", n_trees),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: end_total,
            recall_at_k: recall,
            mean_distance_error: dist_error,
        });
    }

    print_results(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
