mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::collections::HashSet;
use std::time::Instant;
use tempfile::TempDir;
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
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);
    let mut results = Vec::new();

    // Ground truth - Exhaustive
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (query)".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb,
    });

    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, true_distances_self) =
        query_exhaustive_self(&exhaustive_idx, cli.k, true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
        index_size_mb,
    });

    println!("-----------------------------");

    // RaBitQ exhaustive benchmark with reranking
    let temp_dir = TempDir::new().unwrap();
    let rerank_factors = [5, 10, 20];

    println!("Building RaBitQ exhaustive index with reranking...");
    let start = Instant::now();
    let rabitq_idx = build_exhaustive_index_rabitq(
        data.as_ref(),
        None,
        &cli.distance,
        cli.seed as usize,
        true,
        Some(temp_dir.path()),
    )
    .unwrap();
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = rabitq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    for &rerank_factor in &rerank_factors {
        println!(
            "Querying RaBitQ exhaustive index (rerank_factor={})...",
            rerank_factor
        );
        let start = Instant::now();
        let (rabitq_neighbors, rabitq_distances) = query_exhaustive_index_rabitq(
            query_data.as_ref(),
            &rabitq_idx,
            cli.k,
            None,
            true,
            Some(rerank_factor),
            true,
            false,
        );
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &rabitq_neighbors, cli.k);
        let dist_error = calculate_dist_error(
            true_distances.as_ref().unwrap(),
            rabitq_distances.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveRaBitQ-rf{} (query)", rerank_factor),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_err: dist_error,
            index_size_mb,
        });
    }

    println!("Self-querying RaBitQ exhaustive index...");
    let start = Instant::now();
    let (rabitq_neighbors_self, rabitq_distances_self) =
        query_exhaustive_index_rabitq_self(&rabitq_idx, cli.k, None, Some(10), true, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &rabitq_neighbors_self, cli.k);
    let dist_error_self = calculate_dist_error(
        true_distances_self.as_ref().unwrap(),
        rabitq_distances_self.as_ref().unwrap(),
        cli.k,
    );

    results.push(BenchmarkResultSize {
        method: "ExhaustiveRaBitQ (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: recall_self,
        mean_dist_err: dist_error_self,
        index_size_mb,
    });

    println!("-----------------------------");

    let nlist_values = [
        (cli.n_cells as f32 * 0.5).sqrt() as usize,
        (cli.n_cells as f32).sqrt() as usize,
        (cli.n_cells as f32 * 2.0).sqrt() as usize,
    ];

    let rerank_factors = [5, 10, 20];

    for nlist in nlist_values {
        let temp_dir = TempDir::new().unwrap();

        println!("Building IVF-RaBitQ index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_rabitq_idx = build_ivf_index_rabitq(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            true,
            Some(temp_dir.path()),
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = ivf_rabitq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

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

        // Query benchmarks without reranking
        for nprobe in &nprobe_values {
            if *nprobe > nlist || *nprobe == 0 {
                continue;
            }

            println!(
                "Querying IVF-RaBitQ index (nlist={}, nprobe={}, no rerank)...",
                nlist, nprobe
            );
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_ivf_index_rabitq(
                query_data.as_ref(),
                &ivf_rabitq_idx,
                cli.k,
                Some(*nprobe),
                false,
                None,
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
            let dist_error = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("IVF-RaBitQ-nl{}-np{}_no_rr (query)", nlist, nprobe),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
                index_size_mb,
            });
        }

        // Query benchmarks with reranking
        for nprobe in &nprobe_values {
            if *nprobe > nlist || *nprobe == 0 {
                continue;
            }

            for &rerank_factor in &rerank_factors {
                println!(
                    "Querying IVF-RaBitQ index (nlist={}, nprobe={}, rerank_factor={})...",
                    nlist, nprobe, rerank_factor
                );
                let start = Instant::now();
                let (approx_neighbors, approx_distances) = query_ivf_index_rabitq(
                    query_data.as_ref(),
                    &ivf_rabitq_idx,
                    cli.k,
                    Some(*nprobe),
                    true,
                    Some(rerank_factor),
                    true,
                    false,
                );
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
                let dist_error = calculate_dist_error(
                    true_distances.as_ref().unwrap(),
                    approx_distances.as_ref().unwrap(),
                    cli.k,
                );

                results.push(BenchmarkResultSize {
                    method: format!(
                        "IVF-RaBitQ-nl{}-np{}-rf{} (query)",
                        nlist, nprobe, rerank_factor
                    ),
                    build_time_ms: build_time,
                    query_time_ms: query_time,
                    total_time_ms: build_time + query_time,
                    recall_at_k: recall,
                    mean_dist_err: dist_error,
                    index_size_mb,
                });
            }
        }

        // Self-query benchmark with reranking
        let nprobe_self = (nlist as f32 * 2.0).sqrt() as usize;
        println!(
            "Self-querying IVF-RaBitQ index (nprobe={}, rerank_factor=20)...",
            nprobe_self
        );
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) = query_ivf_index_rabitq_self(
            &ivf_rabitq_idx,
            cli.k,
            Some(nprobe_self),
            None,
            true,
            false,
        );
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_dist_error(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("IVF-RaBitQ-nl{} (self)", nlist),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: dist_error_self,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k cells, {}D - IVF-RaBitQ", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
