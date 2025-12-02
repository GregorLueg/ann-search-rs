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

    // /////////////////
    // // Annoy index //
    // /////////////////

    // println!("-----------------------------");

    // for n_trees in [5, 10, 15, 25, 50, 100] {
    //     println!("Building Annoy index ({} trees)...", n_trees);
    //     let start_total = Instant::now();
    //     let start = std::time::Instant::now();
    //     let annoy_idx =
    //         build_annoy_index(data.as_ref(), "euclidean".into(), n_trees, SEED as usize);
    //     let build_time = start.elapsed().as_secs_f64() * 1000.0;

    //     println!("Querying Annoy index ({} trees)...", n_trees,);
    //     let start = std::time::Instant::now();
    //     let (approx_neighbors, approx_distances) =
    //         query_annoy_index(query_data, &annoy_idx, K, None, true, false);
    //     let query_time = start.elapsed().as_secs_f64() * 1000.0;

    //     let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
    //     let dist_error = calculate_distance_error(
    //         true_distances.as_ref().unwrap(),
    //         approx_distances.as_ref().unwrap(),
    //         K,
    //     );
    //     let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

    //     results.push(BenchmarkResult {
    //         method: format!("Annoy-nt{}", n_trees),
    //         build_time_ms: build_time,
    //         query_time_ms: query_time,
    //         total_time_ms: end_total,
    //         recall_at_k: recall,
    //         mean_distance_error: dist_error,
    //     });
    // }

    // ////////////////
    // // HNSW index //
    // ////////////////

    // println!("-----------------------------");

    // for (m, ef_construction, ef_search) in [
    //     (16, 100, 50),
    //     (16, 100, 100),
    //     (16, 200, 100),
    //     (16, 200, 200),
    //     (32, 200, 100),
    //     (32, 200, 200),
    // ] {
    //     println!(
    //         "Building HNSW index (M={}, ef_construction={})...",
    //         m, ef_construction
    //     );
    //     let start_total = Instant::now();
    //     let start = std::time::Instant::now();
    //     let hnsw_idx = build_hnsw_index(
    //         data.as_ref(),
    //         m,
    //         ef_construction,
    //         "euclidean",
    //         SEED as usize,
    //         false,
    //     );
    //     let build_time = start.elapsed().as_secs_f64() * 1000.0;

    //     println!("Querying HNSW index (ef_search={})...", ef_search);
    //     let start = std::time::Instant::now();
    //     let (approx_neighbors, approx_distances) =
    //         query_hnsw_index(query_data, &hnsw_idx, K, ef_search, true, false);
    //     let query_time = start.elapsed().as_secs_f64() * 1000.0;

    //     let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);
    //     let dist_error = calculate_distance_error(
    //         true_distances.as_ref().unwrap(),
    //         approx_distances.as_ref().unwrap(),
    //         K,
    //     );
    //     let end_total = start_total.elapsed().as_secs_f64() * 1000.0;

    //     results.push(BenchmarkResult {
    //         method: format!("HNSW-M{}-ef{}-s{}", m, ef_construction, ef_search),
    //         build_time_ms: build_time,
    //         query_time_ms: query_time,
    //         total_time_ms: end_total,
    //         recall_at_k: recall,
    //         mean_distance_error: dist_error,
    //     });
    // }

    /////////////////////////
    // (Py)NNDescent index //
    /////////////////////////

    println!("-----------------------------");

    for (max_iter, max_cand, diversify_prob, ef_search) in [
        // default with different diversifications
        (None, None, 0.0, 100),
        // (None, None, 0.0, 150), // higher ef search
        // (None, None, 0.5, 100),
        // (None, None, 1.0, 100),
        // // manual max_iter
        // (Some(5), None, 0.0, 100),
        // (Some(10), None, 0.0, 100),
        // // Test with diversification (use lower prob)
        // (None, Some(40), 0.0, 100),
        // (None, Some(80), 0.0, 100),
    ] {
        let iter_str = max_iter
            .map(|i: usize| i.to_string())
            .unwrap_or_else(|| "auto".to_string());

        let max_cand_str = max_cand
            .map(|i: usize| i.to_string())
            .unwrap_or_else(|| "auto".to_string());

        println!(
            "Building NNDescent index (max_iter={}, max_cand={:?}, diversify={})...",
            iter_str, max_cand_str, diversify_prob
        );
        let start_total = Instant::now();
        let start = std::time::Instant::now();
        let nndescent_idx = build_nndescent_index(
            data.as_ref(),
            "euclidean",
            0.001,
            diversify_prob,
            None,
            max_iter,
            max_cand,
            None,
            SEED as usize,
            true,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        println!("Querying NNDescent index (ef_search={})...", ef_search);
        let start = std::time::Instant::now();
        let (approx_neighbors, approx_distances) =
            query_nndescent_index(query_data, &nndescent_idx, K, None, None, true, false);
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
                "NNDescent-i{}-c{}-d{}-ef{}",
                iter_str,
                max_cand_str,
                if diversify_prob > 0.5 {
                    1
                } else if diversify_prob > 0.0 {
                    5
                } else {
                    0
                },
                ef_search
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
