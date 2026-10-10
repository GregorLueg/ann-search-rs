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
        "Generating synthetic data: {} samples, {} dimensions, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, cli.n_samples / 10, cli.seed + 1);
    let mut results = Vec::new();

    // Ground truth: CPU exhaustive
    println!("Building CPU exhaustive index...");
    let start = Instant::now();
    let cpu_exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let cpu_ex_build = start.elapsed().as_secs_f64() * 1000.0;
    let cpu_ex_size = cpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying CPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &cpu_exhaustive_idx, cli.k, true, false)
            .unwrap();
    let cpu_ex_query = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "CPU-Exhaustive (query)".to_string(),
        build_time_ms: cpu_ex_build,
        query_time_ms: cpu_ex_query,
        total_time_ms: cpu_ex_build + cpu_ex_query,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        median_dist_rat: 1.0,
        index_size_mb: cpu_ex_size,
    });

    println!("Self-querying CPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&cpu_exhaustive_idx, cli.k, true, false).unwrap();
    let cpu_ex_self = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "CPU-Exhaustive (self)".to_string(),
        build_time_ms: cpu_ex_build,
        query_time_ms: cpu_ex_self,
        total_time_ms: cpu_ex_build + cpu_ex_self,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        median_dist_rat: 1.0,
        index_size_mb: cpu_ex_size,
    });

    println!("-----------------------------");

    // MLX exhaustive
    println!("Building MLX exhaustive index...");
    let start = Instant::now();
    let mlx_exhaustive_idx = build_exhaustive_index_mlx(
        data.as_ref(),
        &cli.distance,
    )
    .unwrap();
    let mlx_ex_build = start.elapsed().as_secs_f64() * 1000.0;
    let mlx_ex_size = mlx_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying MLX exhaustive...");
    let start = Instant::now();
    let (mlx_ex_neighbors, mlx_ex_distances) =
        query_exhaustive_index_mlx(query_data.as_ref(), &mlx_exhaustive_idx, cli.k, true, false)
            .unwrap();
    let mlx_ex_query = start.elapsed().as_secs_f64() * 1000.0;

    let recall = calculate_recall(&true_neighbors, &mlx_ex_neighbors, cli.k);
    let dist_err = calculate_mean_distance_ratio(
        true_distances.as_ref().unwrap(),
        mlx_ex_distances.as_ref().unwrap(),
        cli.k,
    );
    let dist_err_median = calculate_median_distance_ratio(
        true_distances.as_ref().unwrap(),
        mlx_ex_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "MLX-Exhaustive (query)".to_string(),
        build_time_ms: mlx_ex_build,
        query_time_ms: mlx_ex_query,
        total_time_ms: mlx_ex_build + mlx_ex_query,
        recall_at_k: recall,
        mean_dist_rat: dist_err,
        median_dist_rat: dist_err_median,
        index_size_mb: mlx_ex_size,
    });

    println!("Self-querying MLX exhaustive...");
    let start = Instant::now();
    let (mlx_ex_self_neighbors, mlx_ex_self_distances) =
        query_exhaustive_index_mlx_self(&mlx_exhaustive_idx, cli.k, true, false).unwrap();
    let mlx_ex_self = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &mlx_ex_self_neighbors, cli.k);
    let dist_err_self = calculate_mean_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        mlx_ex_self_distances.as_ref().unwrap(),
        cli.k,
    );
    let dist_err_self_median = calculate_median_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        mlx_ex_self_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "MLX-Exhaustive (self)".to_string(),
        build_time_ms: mlx_ex_build,
        query_time_ms: mlx_ex_self,
        total_time_ms: mlx_ex_build + mlx_ex_self,
        recall_at_k: recall_self,
        mean_dist_rat: dist_err_self,
        median_dist_rat: dist_err_self_median,
        index_size_mb: mlx_ex_size,
    });

    println!("-----------------------------");

    // CAGRA beam search at different beam widths
    println!("Building MLX NNDescent/CAGRA index...");
    let start = Instant::now();
    let mlx_nndescent_idx = build_nndescent_index_mlx(
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
    )
    .unwrap();
    let cagra_build = start.elapsed().as_secs_f64() * 1000.0;
    let cagra_size = mlx_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    // Auto params (library defaults)
    println!("Querying CAGRA (auto params)...");
    let start = Instant::now();
    let (cagra_neighbors, cagra_distances) = query_nndescent_index_mlx(
        query_data.as_ref(),
        &mlx_nndescent_idx,
        cli.k,
        None,
        true,
        false,
    )
    .unwrap();
    let cagra_query = start.elapsed().as_secs_f64() * 1000.0;

    let recall = calculate_recall(&true_neighbors, &cagra_neighbors, cli.k);
    let dist_err = calculate_mean_distance_ratio(
        true_distances.as_ref().unwrap(),
        cagra_distances.as_ref().unwrap(),
        cli.k,
    );
    let dist_err_median = calculate_median_distance_ratio(
        true_distances.as_ref().unwrap(),
        cagra_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "CAGRA-auto (query)".to_string(),
        build_time_ms: cagra_build,
        query_time_ms: cagra_query,
        total_time_ms: cagra_build + cagra_query,
        recall_at_k: recall,
        mean_dist_rat: dist_err,
        median_dist_rat: dist_err_median,
        index_size_mb: cagra_size,
    });

    println!("Self-querying CAGRA (auto params)...");
    let start = Instant::now();
    let (cagra_self_neighbors, cagra_self_distances) =
        query_nndescent_index_mlx_self(&mlx_nndescent_idx, cli.k, None, true).unwrap();
    let cagra_self = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &cagra_self_neighbors, cli.k);
    let dist_err_self = calculate_mean_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        cagra_self_distances.as_ref().unwrap(),
        cli.k,
    );
    let dist_err_self_median = calculate_median_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        cagra_self_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "CAGRA-auto (self)".to_string(),
        build_time_ms: cagra_build,
        query_time_ms: cagra_self,
        total_time_ms: cagra_build + cagra_self,
        recall_at_k: recall_self,
        mean_dist_rat: dist_err_self,
        median_dist_rat: dist_err_self_median,
        index_size_mb: cagra_size,
    });

    let beam_widths = [16, 30, 48, 64];

    for &bw in &beam_widths {
        let max_iters = bw * 3;
        let params = CagraMlxSearchParams::new(Some(bw), Some(max_iters), None, None);

        // External query
        println!(
            "Querying CAGRA (beam_width={}, max_iters={})...",
            bw, max_iters
        );
        let start = Instant::now();
        let (cagra_neighbors, cagra_distances) = query_nndescent_index_mlx(
            query_data.as_ref(),
            &mlx_nndescent_idx,
            cli.k,
            Some(params),
            true,
            false,
        )
        .unwrap();
        let cagra_query = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &cagra_neighbors, cli.k);
        let dist_err = calculate_mean_distance_ratio(
            true_distances.as_ref().unwrap(),
            cagra_distances.as_ref().unwrap(),
            cli.k,
        );
        let dist_err_median = calculate_median_distance_ratio(
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
            mean_dist_rat: dist_err,
            median_dist_rat: dist_err_median,
            index_size_mb: cagra_size,
        });

        // Self query
        let params = CagraMlxSearchParams::new(Some(bw), Some(max_iters), None, None);

        println!("Self-querying CAGRA (beam_width={})...", bw);
        let start = Instant::now();
        let (cagra_self_neighbors, cagra_self_distances) =
            query_nndescent_index_mlx_self(&mlx_nndescent_idx, cli.k, Some(params), true)
                .unwrap();
        let cagra_self = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &cagra_self_neighbors, cli.k);
        let dist_err_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            cagra_self_distances.as_ref().unwrap(),
            cli.k,
        );
        let dist_err_self_median = calculate_median_distance_ratio(
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
            mean_dist_rat: dist_err_self,
            median_dist_rat: dist_err_self_median,
            index_size_mb: cagra_size,
        });
    }

    println!("-----------------------------");

    print_results_size(
        &format!(
            "{}k samples, {}D (Exhaustive vs CAGRA beam search)",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results,
    );
}
