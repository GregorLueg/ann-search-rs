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

    // CPU Exhaustive (ground truth for external queries)
    println!("Building CPU exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying CPU exhaustive index...");
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
        median_dist_rat: 1.0,
        index_size_mb,
    });

    // CPU Exhaustive (ground truth for self-query)
    println!("Self-querying CPU exhaustive index...");
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
        median_dist_rat: 1.0,
        index_size_mb,
    });

    println!("-----------------------------");

    // MLX Exhaustive
    println!("Building MLX exhaustive index (MLX)...");

    let start = Instant::now();
    let mlx_exhaustive_idx = build_exhaustive_index_mlx(
        data.as_ref(),
        &cli.distance,
    )
    .unwrap();
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = mlx_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying MLX exhaustive index (MLX)...");
    let start = Instant::now();
    let (mlx_neighbors, mlx_distances) =
        query_exhaustive_index_mlx(query_data.as_ref(), &mlx_exhaustive_idx, cli.k, true, false)
            .unwrap();
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall = calculate_recall(&true_neighbors, &mlx_neighbors, cli.k);
    let dist_error = calculate_mean_distance_ratio(
        true_distances.as_ref().unwrap(),
        mlx_distances.as_ref().unwrap(),
        cli.k,
    );
    let dist_error_median = calculate_median_distance_ratio(
        true_distances.as_ref().unwrap(),
        mlx_distances.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "MLX-Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: recall,
        mean_dist_rat: dist_error,
        median_dist_rat: dist_error_median,
        index_size_mb,
    });

    println!("Self-querying MLX exhaustive index (MLX)...");
    let start = Instant::now();
    let (mlx_neighbors_self, mlx_distances_self) =
        query_exhaustive_index_mlx_self(&mlx_exhaustive_idx, cli.k, true, false).unwrap();
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &mlx_neighbors_self, cli.k);
    let dist_error_self = calculate_mean_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        mlx_distances_self.as_ref().unwrap(),
        cli.k,
    );
    let dist_error_self_median = calculate_median_distance_ratio(
        true_distances_self.as_ref().unwrap(),
        mlx_distances_self.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "MLX-Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: recall_self,
        mean_dist_rat: dist_error_self,
        median_dist_rat: dist_error_self_median,
        index_size_mb,
    });

    println!("-----------------------------");

    // IVF GPU with various nlist/nprobe combinations
    let nlist_values = [
        (cli.n_samples as f32 * 0.5).sqrt() as usize,
        (cli.n_samples as f32).sqrt() as usize,
        (cli.n_samples as f32 * 2.0).sqrt() as usize,
    ];

    for nlist in nlist_values {
        println!("Building IVF-MLX index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_mlx_idx = build_ivf_index_mlx(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let (index_size_ram, _) = ivf_mlx_idx.memory_usage_bytes();
        let index_size_mb = index_size_ram as f64 / (1024.0 * 1024.0);

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

            // External query
            println!("Querying IVF-MLX (nlist={}, nprobe={})...", nlist, nprobe);
            let start = Instant::now();
            let (knn_neighbors, knn_distances) = query_ivf_index_mlx(
                query_data.as_ref(),
                &ivf_mlx_idx,
                cli.k,
                Some(nprobe),
                None,
                true,
                false,
            )
            .unwrap();
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &knn_neighbors, cli.k);
            let dist_error = calculate_mean_distance_ratio(
                true_distances.as_ref().unwrap(),
                knn_distances.as_ref().unwrap(),
                cli.k,
            );
            let dist_error_median = calculate_median_distance_ratio(
                true_distances.as_ref().unwrap(),
                knn_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("IVF-MLX-nl{}-np{} (query)", nlist, nprobe),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_rat: dist_error,
                median_dist_rat: dist_error_median,
                index_size_mb,
            });
        }

        // Self-query
        println!("Self-querying IVF-MLX index (nlist={})...", nlist);
        let nprobe_self = (nlist as f32 * 2.0).sqrt() as usize;

        let start = Instant::now();
        let (knn_neighbors_self, knn_distances_self) =
            query_ivf_index_mlx_self(&ivf_mlx_idx, cli.k, Some(nprobe_self), None, true, false)
                .unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &knn_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            knn_distances_self.as_ref().unwrap(),
            cli.k,
        );
        let dist_error_self_median = calculate_median_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            knn_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("IVF-MLX-nl{} (self)", nlist),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            median_dist_rat: dist_error_self_median,
            index_size_mb,
        });
    }

    print_results_size(
        &format!(
            "{}k samples, {}D (CPU vs MLX Exhaustive vs IVF-MLX)",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results,
    );
}
