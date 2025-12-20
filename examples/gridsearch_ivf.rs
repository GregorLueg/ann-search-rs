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

    let sqrt_n = (N_CELLS as f64).sqrt();
    let nlist_values = [
        (sqrt_n * 0.25) as usize,
        (sqrt_n * 0.5) as usize,
        sqrt_n as usize,
    ];

    for nlist in nlist_values {
        println!("Building IVF index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_idx = build_ivf_index(
            data.as_ref(),
            nlist,
            None,
            "euclidean",
            SEED as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let nprobe_values = [
            (0.05 * nlist as f64) as usize,
            (0.1 * nlist as f64) as usize,
            (0.15 * nlist as f64) as usize,
            (0.2 * nlist as f64) as usize,
        ];

        for nprobe in nprobe_values {
            if nprobe > nlist || nprobe == 0 {
                continue;
            }

            println!("Querying IVF index (nlist={}, nprobe={})...", nlist, nprobe);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_ivf_index(query_data, &ivf_idx, K, Some(nprobe), true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, K);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                K,
            );

            let internal_recall = ivf_idx.validate_index(K, SEED as usize, None);
            println!("  Internal validation: {:.3}", internal_recall);

            results.push(BenchmarkResult {
                method: format!("IVF-nl{}-np{}", nlist, nprobe),
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
