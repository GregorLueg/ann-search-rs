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
    const N_CELLS: usize = 50_000;
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

    let n_trees_values = [5, 10, 15, 25, 50, 75];

    for n_trees in n_trees_values {
        println!("Building Annoy index ({} trees)...", n_trees);
        let start = Instant::now();
        let annoy_idx =
            build_annoy_index(data.as_ref(), "euclidean".into(), n_trees, SEED as usize);
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let search_budgets = [
            (None, "auto"),
            (Some(K * n_trees * 10), "10x"),
            (Some(K * n_trees * 5), "5x"),
        ];

        for (search_budget, budget_label) in search_budgets {
            println!("Querying Annoy index (search_budget={})...", budget_label);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_annoy_index(query_data, &annoy_idx, K, search_budget, true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

            let internal_recall = annoy_idx.validate_index(K, SEED as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("Annoy-nt{}:{}", n_trees, budget_label),
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
