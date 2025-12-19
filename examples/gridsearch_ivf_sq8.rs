mod commons;
use ann_search_rs::synthetic::generate_clustered_data;
use ann_search_rs::*;
use commons::*;
use faer::Mat;
use std::time::Instant;

fn main() {
    const N_CELLS: usize = 50_000;
    const DIM: usize = 24;
    const N_CLUSTERS: usize = 20;
    const K: usize = 15;
    const SEED: u64 = 42;

    println!("===============================================================================================");
    println!(
        "Benchmark: {}k cells, {}D - IVF-SQ8 Symmetric Inner Product",
        N_CELLS / 1000,
        DIM
    );
    println!("===============================================================================================");

    let data: Mat<f32> = generate_clustered_data(N_CELLS, DIM, N_CLUSTERS, SEED);
    let query_data = data.as_ref();
    let mut results = Vec::new();

    // Exhaustive with inner product
    println!("Building exhaustive index (inner product)...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), "inner_product");
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, _) = query_exhaustive_index(query_data, &exhaustive_idx, K, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive-IP".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_distance_error: 0.0,
    });

    println!("-----------------------------------------------------------------------------------------------");

    let sqrt_n = (N_CELLS as f64).sqrt();
    let nlist_values = [
        (sqrt_n * 0.25) as usize,
        (sqrt_n * 0.5) as usize,
        sqrt_n as usize,
        (sqrt_n * 1.5) as usize,
    ];

    for nlist in nlist_values {
        println!("Building IVF-SQ8 index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_sq8_idx = build_ivf_sq8_index(data.as_ref(), nlist, None, SEED as usize, false);
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

            println!("Querying IVF-SQ8 (nlist={}, nprobe={})...", nlist, nprobe);
            let start = Instant::now();
            let (approx_neighbors, _) =
                query_ivf_sq8_index(query_data, &ivf_sq8_idx, K, Some(nprobe), true, false);
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall::<f32>(&true_neighbors, &approx_neighbors, K);

            results.push(BenchmarkResult {
                method: format!("IVF-SQ8-nl{}-np{}", nlist, nprobe),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_distance_error: 0.0, // Inner product doesn't have distance error metric
            });
        }
    }

    print_results_recall_only(&format!("{}k cells, {}D", N_CELLS / 1000, DIM), &results);
}
