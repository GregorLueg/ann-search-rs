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
        "Generating synthetic data: {} cells, {} dimensions, {} clusters, {} dist.",
        cli.n_cells.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, cli.n_cells / 10, cli.seed + 1);
    let mut results = Vec::new();

    // Ground truth: GPU exhaustive self-query
    let device: cubecl::wgpu::WgpuDevice = Default::default();

    println!("Building GPU exhaustive index...");
    let start = Instant::now();
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device.clone(),
    );
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = gpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Self-querying GPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_index_gpu_self(&gpu_exhaustive_idx, cli.k, true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb,
    });

    // Ground truth: GPU exhaustive external query
    println!("Querying GPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index_gpu(query_data.as_ref(), &gpu_exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb,
    });

    println!("-----------------------------");

    // CPU NNDescent (default params)
    println!("Building CPU NNDescent...");
    let start = Instant::now();
    let cpu_nndescent_idx = build_nndescent_index(
        data.as_ref(),
        &cli.distance,
        0.001,
        0.0,
        None,
        None,
        None,
        None,
        cli.seed as usize,
        true,
    );
    let cpu_build_time = start.elapsed().as_secs_f64() * 1000.0;
    let cpu_index_size_mb = cpu_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    // CPU NNDescent self-query (extract kNN graph)
    println!("Extracting CPU NNDescent kNN graph...");
    let start = Instant::now();
    let (cpu_self_neighbors, cpu_self_distances) =
        query_nndescent_self(&cpu_nndescent_idx, cli.k, None, true, false);
    let cpu_self_time = start.elapsed().as_secs_f64() * 1000.0;

    let cpu_self_recall = calculate_recall(&true_neighbors_self, &cpu_self_neighbors, cli.k);
    let cpu_self_dist_err = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        cpu_self_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "CPU-NNDescent (self)".to_string(),
        build_time_ms: cpu_build_time,
        query_time_ms: cpu_self_time,
        total_time_ms: cpu_build_time + cpu_self_time,
        recall_at_k: cpu_self_recall,
        mean_dist_err: cpu_self_dist_err,
        index_size_mb: cpu_index_size_mb,
    });

    // CPU NNDescent external query
    println!("Querying CPU NNDescent...");
    let start = Instant::now();
    let (cpu_query_neighbors, cpu_query_distances) = query_nndescent_index(
        query_data.as_ref(),
        &cpu_nndescent_idx,
        cli.k,
        None,
        true,
        false,
    );
    let cpu_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let cpu_query_recall = calculate_recall(&true_neighbors, &cpu_query_neighbors, cli.k);
    let cpu_query_dist_err = calculate_dist_error(
        true_distances.as_ref().unwrap(),
        cpu_query_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "CPU-NNDescent (query)".to_string(),
        build_time_ms: cpu_build_time,
        query_time_ms: cpu_query_time,
        total_time_ms: cpu_build_time + cpu_query_time,
        recall_at_k: cpu_query_recall,
        mean_dist_err: cpu_query_dist_err,
        index_size_mb: cpu_index_size_mb,
    });

    println!("-----------------------------");

    // GPU NNDescent (default params)
    println!("Building GPU NNDescent...");
    let start = Instant::now();
    let mut gpu_nndescent_idx = build_nndescent_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        Some(cli.k),
        None,
        Some(20),
        None,
        Some(0.0005),
        None,
        Some(1),
        cli.seed as usize,
        true,
        true,
        device.clone(),
    );
    let gpu_build_time = start.elapsed().as_secs_f64() * 1000.0;
    let gpu_index_size_mb = gpu_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    // GPU NNDescent extract kNN (no search, just reshape)
    println!("Extracting GPU NNDescent kNN graph...");
    let start = Instant::now();
    let (gpu_self_neighbors, gpu_self_distances) =
        extract_nndescent_knn_gpu(&gpu_nndescent_idx, true);
    let gpu_extract_time = start.elapsed().as_secs_f64() * 1000.0;

    let gpu_self_recall = calculate_recall(&true_neighbors_self, &gpu_self_neighbors, cli.k);
    let gpu_self_dist_err = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        gpu_self_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "GPU-NNDescent (extract)".to_string(),
        build_time_ms: gpu_build_time,
        query_time_ms: gpu_extract_time,
        total_time_ms: gpu_build_time + gpu_extract_time,
        recall_at_k: gpu_self_recall,
        mean_dist_err: gpu_self_dist_err,
        index_size_mb: gpu_index_size_mb,
    });

    // GPU NNDescent self-query via GPU beam search
    println!("Self-querying GPU NNDescent (GPU beam search)...");
    let start = Instant::now();
    let (gpu_self_beam_neighbors, gpu_self_beam_distances) =
        query_nndescent_index_gpu_self(&mut gpu_nndescent_idx, cli.k, true);
    let gpu_self_beam_time = start.elapsed().as_secs_f64() * 1000.0;

    let gpu_self_beam_recall =
        calculate_recall(&true_neighbors_self, &gpu_self_beam_neighbors, cli.k);
    let gpu_self_beam_dist_err = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        gpu_self_beam_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "GPU-NNDescent (self-beam)".to_string(),
        build_time_ms: gpu_build_time,
        query_time_ms: gpu_self_beam_time,
        total_time_ms: gpu_build_time + gpu_self_beam_time,
        recall_at_k: gpu_self_beam_recall,
        mean_dist_err: gpu_self_beam_dist_err,
        index_size_mb: gpu_index_size_mb,
    });

    // GPU NNDescent external query (CPU beam search over CAGRA graph)
    println!("Querying GPU NNDescent...");
    let start = Instant::now();
    let (gpu_query_neighbors, gpu_query_distances) = query_nndescent_index_gpu(
        query_data.as_ref(),
        &mut gpu_nndescent_idx,
        cli.k,
        None,
        true,
        false,
    );
    let gpu_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let gpu_query_recall = calculate_recall(&true_neighbors, &gpu_query_neighbors, cli.k);
    let gpu_query_dist_err = calculate_dist_error(
        true_distances.as_ref().unwrap(),
        gpu_query_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "GPU-NNDescent (query)".to_string(),
        build_time_ms: gpu_build_time,
        query_time_ms: gpu_query_time,
        total_time_ms: gpu_build_time + gpu_query_time,
        recall_at_k: gpu_query_recall,
        mean_dist_err: gpu_query_dist_err,
        index_size_mb: gpu_index_size_mb,
    });

    println!("-----------------------------");

    print_results_size(
        &format!(
            "{}k cells, {}D (GPU-Exhaustive vs CPU-NNDescent vs GPU-NNDescent)",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
