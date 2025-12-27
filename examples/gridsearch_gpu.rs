mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::collections::HashSet;
use std::time::Instant;
use thousands::*;

fn main() {
    let cli = Cli::parse();

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} cells, {} dimensions, {} clusters, {} dist.",
        cli.n_cells.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let data_type = parse_data(&cli.data).unwrap_or_default();

    let data: Mat<f32> = match data_type {
        SyntheticData::GaussianNoise => {
            generate_clustered_data(cli.n_cells, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!("Using data for high dimensional ANN searches...\n");
            generate_clustered_data_high_dim(
                cli.n_cells,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
    };
    let query_data = data.as_ref();
    let mut results = Vec::new();

    // CPU Exhaustive (ground truth)
    println!("Building CPU exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying CPU exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data, &exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "CPU-Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
    });

    println!("-----------------------------");

    // GPU Exhaustive
    println!("Building GPU exhaustive index (WGPU)...");
    let device: cubecl::wgpu::WgpuDevice = Default::default();

    let start = Instant::now();
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device.clone(),
    );
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying GPU exhaustive index (WGPU)...");
    let start = Instant::now();
    let (gpu_neighbors, gpu_distances) =
        query_exhaustive_index_gpu(query_data, &gpu_exhaustive_idx, cli.k, true);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall = calculate_recall(&true_neighbors, &gpu_neighbors, cli.k);
    let dist_error = calculate_distance_error(
        true_distances.as_ref().unwrap(),
        gpu_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResult {
        method: "GPU-Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: recall,
        mean_dist_err: dist_error,
    });

    println!("-----------------------------");

    // IVF GPU with various nlist/nprobe combinations
    let nlist_values = [
        (cli.n_cells as f32 * 0.5).sqrt() as usize,
        (cli.n_cells as f32).sqrt() as usize,
        (cli.n_cells as f32 * 2.0).sqrt() as usize,
    ];

    for nlist in nlist_values {
        println!("Building IVF-GPU index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_gpu_idx = build_ivf_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            false,
            device.clone(),
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let nprobe_values = [
            (nlist as f32).sqrt() as usize,
            (nlist as f32 * 2.0).sqrt() as usize,
            (0.05 * nlist as f32) as usize,
        ];
        let mut nprobe_values: Vec<_> = nprobe_values
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        nprobe_values.sort();

        for nprobe in nprobe_values {
            if nprobe > nlist || nprobe == 0 {
                continue;
            }

            // Test build_knn_graph
            println!("Querying IVF-GPU (nlist={}, nprobe={})...", nlist, nprobe);
            let start = Instant::now();
            let (knn_neighbors, knn_distances) = query_ivf_index_gpu(
                data.as_ref(),
                &ivf_gpu_idx,
                cli.k,
                Some(nprobe),
                true,
                false,
            );
            let knn_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &knn_neighbors, cli.k);
            let dist_error = calculate_distance_error(
                true_distances.as_ref().unwrap(),
                knn_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResult {
                method: format!("IVF-GPU-kNN-nl{}-np{}", nlist, nprobe),
                build_time_ms: build_time,
                query_time_ms: knn_time,
                total_time_ms: build_time + knn_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
            });
        }

        println!("-----------------------------");
    }

    print_results(
        &format!(
            "{}k cells, {}D (CPU vs GPU Exhaustive vs IVF-GPU)",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
