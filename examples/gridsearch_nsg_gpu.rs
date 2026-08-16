mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

fn main() {
    let cli = Cli::parse();

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} samples, {} dimensions, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);
    let mut results = Vec::new();

    // Exhaustive baseline
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false).unwrap();
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        index_size_mb,
    });

    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&exhaustive_idx, cli.k, true, false).unwrap();
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        index_size_mb,
    });

    println!("-----------------------------");

    let device: cubecl::wgpu::WgpuDevice = Default::default();

    let knn_k: usize = 32;

    println!("Building GPU kNN graph (k={})...", knn_k);
    let knn_start = Instant::now();
    let knn_gpu = build_knn_graph_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        Some(knn_k),
        Some(knn_k),
        None,
        None,
        None,
        None,
        None,
        cli.seed as usize,
        false,
        device,
    )
    .unwrap();
    let knn_build_ms = knn_start.elapsed().as_secs_f64() * 1000.0;
    // println!("GPU kNN graph built in {:.2} ms", knn_build_ms);
    // println!("-----------------------------");

    let build_params: &[(usize, usize)] = &[
        (24, 50),
        (24, 100),
        (24, 150),
        (32, 50),
        (32, 100),
        (32, 150),
        (48, 50),
        (48, 100),
        (48, 150),
    ];
    let c_cap: usize = 500;
    let ef_search_values: &[Option<usize>] = &[Some(50), None, Some(150)];

    for &(r, l_build) in build_params {
        println!(
            "Building NSG index from GPU kNN (R={}, L_build={})...",
            r, l_build
        );

        let start = Instant::now();
        let nsg_idx =
            build_nsg_from_gpu_knn(&knn_gpu, r, l_build, c_cap, cli.seed as usize, false).unwrap();
        let nsg_build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let build_time = knn_build_ms + nsg_build_ms;

        let index_size_mb = nsg_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        for &ef_search in ef_search_values {
            let ef_label = ef_search
                .map(|e| e.to_string())
                .unwrap_or("auto".to_string());
            println!("Querying NSG index (ef_search={})...", ef_label);

            let start = Instant::now();
            let (approx_neighbors, approx_distances) =
                query_nsg_index(query_data.as_ref(), &nsg_idx, cli.k, ef_search, true, false)
                    .unwrap();
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_mean_distance_ratio(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("NSG-GPU-R{}-L{}-ef{} (query)", r, l_build, ef_label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_rat: dist_error,
                index_size_mb,
            });
        }

        println!("Self-querying NSG index...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_nsg_self(&nsg_idx, cli.k, None, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("NSG-GPU-R{}-L{} (self)", r, l_build),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
