// bench_cagra_knn.rs
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
    let mut results = Vec::new();

    let device: cubecl::wgpu::WgpuDevice = Default::default();

    // Ground truth: GPU exhaustive self-query
    println!("Building GPU exhaustive index...");
    let start = Instant::now();
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device.clone(),
    );
    let ex_build = start.elapsed().as_secs_f64() * 1000.0;

    println!("Self-querying GPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index_gpu_self(&gpu_exhaustive_idx, cli.k, true, false);
    let ex_query = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (ground truth)".to_string(),
        build_time_ms: ex_build,
        query_time_ms: ex_query,
        total_time_ms: ex_build + ex_query,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb: gpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0),
    });

    println!("-----------------------------");

    // CPU NNDescent baseline
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
    let cpu_build = start.elapsed().as_secs_f64() * 1000.0;
    let cpu_size = cpu_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Extracting CPU NNDescent kNN graph...");
    let start = Instant::now();
    let (cpu_neighbors, cpu_distances) =
        query_nndescent_self(&cpu_nndescent_idx, cli.k, None, true, false);
    let cpu_extract = start.elapsed().as_secs_f64() * 1000.0;

    let cpu_recall = calculate_recall(&true_neighbors, &cpu_neighbors, cli.k);
    let cpu_dist_err = calculate_dist_error(
        true_distances.as_ref().unwrap(),
        cpu_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "CPU-NNDescent (k=15)".to_string(),
        build_time_ms: cpu_build,
        query_time_ms: cpu_extract,
        total_time_ms: cpu_build + cpu_extract,
        recall_at_k: cpu_recall,
        mean_dist_err: cpu_dist_err,
        index_size_mb: cpu_size,
    });

    println!("-----------------------------");

    // GPU NNDescent: sweep build_k multiplier x refinement sweeps
    let build_k_multipliers = [1, 2, 3];
    let refine_sweeps = [0, 1, 2];

    for &bk_mult in &build_k_multipliers {
        for &refine in &refine_sweeps {
            let build_k = cli.k * bk_mult;
            let label = format!("GPU-NND bk={}x refine={}", bk_mult, refine);

            println!(
                "Building GPU NNDescent (build_k={}, refine={})...",
                build_k, refine
            );

            let start = Instant::now();
            let mut gpu_idx = build_nndescent_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
                data.as_ref(),
                &cli.distance,
                Some(cli.k),
                Some(build_k),
                Some(20),
                None,
                Some(0.0005),
                None,
                Some(refine),
                cli.seed as usize,
                false,
                true,
                device.clone(),
            );
            let gpu_build = start.elapsed().as_secs_f64() * 1000.0;
            let gpu_size = gpu_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

            // Extract kNN graph
            println!("  Extracting kNN graph...");
            let start = Instant::now();
            let (gpu_neighbors, gpu_distances) = extract_nndescent_knn_gpu(&gpu_idx, true);
            let gpu_extract = start.elapsed().as_secs_f64() * 1000.0;

            let gpu_recall = calculate_recall(&true_neighbors, &gpu_neighbors, cli.k);
            let gpu_dist_err = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                gpu_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("{} (extract)", label),
                build_time_ms: gpu_build,
                query_time_ms: gpu_extract,
                total_time_ms: gpu_build + gpu_extract,
                recall_at_k: gpu_recall,
                mean_dist_err: gpu_dist_err,
                index_size_mb: gpu_size,
            });

            // Self-beam search
            println!("  Self-querying via beam search...");
            let start = Instant::now();
            let (gpu_beam_neighbors, gpu_beam_distances) =
                query_nndescent_index_gpu_self(&mut gpu_idx, cli.k, None, true);
            let gpu_beam = start.elapsed().as_secs_f64() * 1000.0;

            let gpu_beam_recall = calculate_recall(&true_neighbors, &gpu_beam_neighbors, cli.k);
            let gpu_beam_dist_err = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                gpu_beam_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("{} (self-beam)", label),
                build_time_ms: gpu_build,
                query_time_ms: gpu_beam,
                total_time_ms: gpu_build + gpu_beam,
                recall_at_k: gpu_beam_recall,
                mean_dist_err: gpu_beam_dist_err,
                index_size_mb: gpu_size,
            });

            println!("-----------------------------");
        }
    }

    print_results_size(
        &format!(
            "{}k cells, {}D kNN graph generation (build_k x refinement)",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
