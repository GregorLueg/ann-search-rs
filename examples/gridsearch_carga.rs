mod commons;

use ann_search_rs::prelude::*;
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

    let device: cubecl::wgpu::WgpuDevice = Default::default();

    // Ground truth: CPU exhaustive
    println!("Building CPU exhaustive index...");
    let start = Instant::now();
    let cpu_exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let cpu_ex_build = start.elapsed().as_secs_f64() * 1000.0;
    let cpu_ex_size = cpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying CPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &cpu_exhaustive_idx, cli.k, true, false);
    let cpu_ex_query = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "CPU-Exhaustive (query)".to_string(),
        build_time_ms: cpu_ex_build,
        query_time_ms: cpu_ex_query,
        total_time_ms: cpu_ex_build + cpu_ex_query,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb: cpu_ex_size,
    });

    println!("Self-querying CPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&cpu_exhaustive_idx, cli.k, true, false);
    let cpu_ex_self = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "CPU-Exhaustive (self)".to_string(),
        build_time_ms: cpu_ex_build,
        query_time_ms: cpu_ex_self,
        total_time_ms: cpu_ex_build + cpu_ex_self,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb: cpu_ex_size,
    });

    println!("-----------------------------");

    // GPU exhaustive
    println!("Building GPU exhaustive index...");
    let start = Instant::now();
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device.clone(),
    );
    let gpu_ex_build = start.elapsed().as_secs_f64() * 1000.0;
    let gpu_ex_size = gpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying GPU exhaustive...");
    let start = Instant::now();
    let (gpu_ex_neighbors, gpu_ex_distances) =
        query_exhaustive_index_gpu(query_data.as_ref(), &gpu_exhaustive_idx, cli.k, true, false);
    let gpu_ex_query = start.elapsed().as_secs_f64() * 1000.0;

    let recall = calculate_recall(&true_neighbors, &gpu_ex_neighbors, cli.k);
    let dist_err = calculate_dist_error(
        true_distances.as_ref().unwrap(),
        gpu_ex_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (query)".to_string(),
        build_time_ms: gpu_ex_build,
        query_time_ms: gpu_ex_query,
        total_time_ms: gpu_ex_build + gpu_ex_query,
        recall_at_k: recall,
        mean_dist_err: dist_err,
        index_size_mb: gpu_ex_size,
    });

    println!("Self-querying GPU exhaustive...");
    let start = Instant::now();
    let (gpu_ex_self_neighbors, gpu_ex_self_distances) =
        query_exhaustive_index_gpu_self(&gpu_exhaustive_idx, cli.k, true, false);
    let gpu_ex_self = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &gpu_ex_self_neighbors, cli.k);
    let dist_err_self = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        gpu_ex_self_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (self)".to_string(),
        build_time_ms: gpu_ex_build,
        query_time_ms: gpu_ex_self,
        total_time_ms: gpu_ex_build + gpu_ex_self,
        recall_at_k: recall_self,
        mean_dist_err: dist_err_self,
        index_size_mb: gpu_ex_size,
    });

    println!("-----------------------------");

    // CAGRA beam search at different beam widths
    println!("Building GPU NNDescent/CAGRA index...");
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
        false,
        true,
        device.clone(),
    );
    let cagra_build = start.elapsed().as_secs_f64() * 1000.0;
    let cagra_size = gpu_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    let beam_widths = [16, 30, 48, 64];

    for &bw in &beam_widths {
        let max_iters = bw * 2;
        let params = CagraGpuSearchParams::new(Some(bw), Some(max_iters), None);

        // External query
        println!(
            "Querying CAGRA (beam_width={}, max_iters={})...",
            bw, max_iters
        );
        let start = Instant::now();
        let (cagra_neighbors, cagra_distances) = query_nndescent_index_gpu(
            query_data.as_ref(),
            &mut gpu_nndescent_idx,
            cli.k,
            None,
            Some(params),
            true,
            false,
        );
        let cagra_query = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &cagra_neighbors, cli.k);
        let dist_err = calculate_dist_error(
            true_distances.as_ref().unwrap(),
            cagra_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("CAGRA-bw{} (query)", bw),
            build_time_ms: cagra_build,
            query_time_ms: cagra_query,
            total_time_ms: cagra_build + cagra_query,
            recall_at_k: recall,
            mean_dist_err: dist_err,
            index_size_mb: cagra_size,
        });

        // Self query
        let params = CagraGpuSearchParams::new(Some(bw), Some(max_iters), None);

        println!("Self-querying CAGRA (beam_width={})...", bw);
        let start = Instant::now();
        let (cagra_self_neighbors, cagra_self_distances) =
            query_nndescent_index_gpu_self(&mut gpu_nndescent_idx, cli.k, Some(params), true);
        let cagra_self = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &cagra_self_neighbors, cli.k);
        let dist_err_self = calculate_dist_error(
            true_distances_self.as_ref().unwrap(),
            cagra_self_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("CAGRA-bw{} (self)", bw),
            build_time_ms: cagra_build,
            query_time_ms: cagra_self,
            total_time_ms: cagra_build + cagra_self,
            recall_at_k: recall_self,
            mean_dist_err: dist_err_self,
            index_size_mb: cagra_size,
        });
    }

    println!("-----------------------------");

    print_results_size(
        &format!(
            "{}k cells, {}D (Exhaustive vs CAGRA beam search)",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
