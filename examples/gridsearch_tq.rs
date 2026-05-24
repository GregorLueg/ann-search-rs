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

    // -------------------------------------------------------------------
    // Ground truth: exact exhaustive search
    // -------------------------------------------------------------------
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

    let bit_widths = [2usize, 4];
    let rerank_factors = [5usize, 10, 20];

    for bits in bit_widths {
        let temp_dir = TempDir::new().unwrap();

        // 3 bits is crazily slow...the difference without SIMD is insane

        // if bits == 3 {
        //     println!("[NOTE] 3-bit has no SIMD kernel and uses the scalar scorer (slower).");
        // }

        println!("Building TurboQuant exhaustive index (bits={})...", bits);
        let start = Instant::now();
        let tq_idx = build_exhaustive_index_turboquant(
            data.as_ref(),
            &cli.distance,
            bits,
            cli.seed as usize,
            true,
            Some(temp_dir.path()),
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = tq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Approximate query, no reranking.
        println!("Querying TurboQuant index (bits={}, no rerank)...", bits);
        let start = Instant::now();
        let (tq_neighbors, _) = query_exhaustive_index_turboquant(
            query_data.as_ref(),
            &tq_idx,
            cli.k,
            false,
            None,
            false,
            false,
        )
        .unwrap();
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &tq_neighbors, cli.k);

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveTQ-b{}-rf0 (query)", bits),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_rat: f64::NAN,
            index_size_mb,
        });

        // Approximate query, with exact reranking.
        for &rerank_factor in &rerank_factors {
            println!(
                "Querying TurboQuant index (bits={}, rerank_factor={})...",
                bits, rerank_factor
            );
            let start = Instant::now();
            let (tq_neighbors, tq_distances) = query_exhaustive_index_turboquant(
                query_data.as_ref(),
                &tq_idx,
                cli.k,
                true,
                Some(rerank_factor),
                true,
                false,
            )
            .unwrap();
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &tq_neighbors, cli.k);
            let dist_error = calculate_mean_distance_ratio(
                true_distances.as_ref().unwrap(),
                tq_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("ExhaustiveTQ-b{}-rf{} (query)", bits, rerank_factor),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_rat: dist_error,
                index_size_mb,
            });
        }

        // Self-query: full kNN graph with reranking.
        println!(
            "Self-querying TurboQuant index (bits={}, rerank_factor=20)...",
            bits
        );
        let start = Instant::now();
        let (tq_neighbors_self, tq_distances_self) =
            query_exhaustive_index_turboquant_self(&tq_idx, cli.k, Some(20), true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &tq_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            tq_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveTQ-b{} (self)", bits),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            index_size_mb,
        });

        println!("-----------------------------");
    }

    // -------------------------------------------------------------------
    // IVF-TurboQuant: sweep over bits, nlist, and nprobe
    // -------------------------------------------------------------------
    let ivf_bit_widths = [2usize, 4];
    let nlist_values = [
        (cli.n_samples as f32 * 0.5).sqrt() as usize,
        (cli.n_samples as f32).sqrt() as usize,
        (cli.n_samples as f32 * 2.0).sqrt() as usize,
    ];
    let ivf_rerank_factors = [10usize, 20];

    for bits in ivf_bit_widths {
        for nlist in nlist_values {
            let temp_dir = TempDir::new().unwrap();

            println!(
                "Building IVF-TurboQuant index (bits={}, nlist={})...",
                bits, nlist
            );
            let start = Instant::now();
            let ivf_tq_idx = build_ivf_index_turboquant(
                data.as_ref(),
                Some(nlist),
                None,
                &cli.distance,
                bits,
                cli.seed as usize,
                true,
                Some(temp_dir.path()),
                false,
            )
            .unwrap();
            let build_time = start.elapsed().as_secs_f64() * 1000.0;
            let index_size_mb = ivf_tq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

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
                    "Querying IVF-TurboQuant index (bits={}, nlist={}, nprobe={}, no rerank)...",
                    bits, nlist, nprobe
                );
                let start = Instant::now();
                let (approx_neighbors, _) = query_ivf_index_turboquant(
                    query_data.as_ref(),
                    &ivf_tq_idx,
                    cli.k,
                    Some(*nprobe),
                    false,
                    None,
                    true,
                    false,
                )
                .unwrap();
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);

                results.push(BenchmarkResultSize {
                    method: format!("IVF-TQ-b{}-nl{}-np{}-rf0 (query)", bits, nlist, nprobe),
                    build_time_ms: build_time,
                    query_time_ms: query_time,
                    total_time_ms: build_time + query_time,
                    recall_at_k: recall,
                    mean_dist_rat: f64::NAN,
                    index_size_mb,
                });
            }

            // Query benchmarks with reranking
            for nprobe in &nprobe_values {
                if *nprobe > nlist || *nprobe == 0 {
                    continue;
                }

                for &rerank_factor in &ivf_rerank_factors {
                    println!(
                        "Querying IVF-TurboQuant index (bits={}, nlist={}, nprobe={}, rerank_factor={})...",
                        bits, nlist, nprobe, rerank_factor
                    );
                    let start = Instant::now();
                    let (approx_neighbors, approx_distances) = query_ivf_index_turboquant(
                        query_data.as_ref(),
                        &ivf_tq_idx,
                        cli.k,
                        Some(*nprobe),
                        true,
                        Some(rerank_factor),
                        true,
                        false,
                    )
                    .unwrap();
                    let query_time = start.elapsed().as_secs_f64() * 1000.0;

                    let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);
                    let dist_error = calculate_mean_distance_ratio(
                        true_distances.as_ref().unwrap(),
                        approx_distances.as_ref().unwrap(),
                        cli.k,
                    );

                    results.push(BenchmarkResultSize {
                        method: format!(
                            "IVF-TQ-b{}-nl{}-np{}-rf{} (query)",
                            bits, nlist, nprobe, rerank_factor
                        ),
                        build_time_ms: build_time,
                        query_time_ms: query_time,
                        total_time_ms: build_time + query_time,
                        recall_at_k: recall,
                        mean_dist_rat: dist_error,
                        index_size_mb,
                    });
                }
            }

            // Self-query benchmark with reranking
            let nprobe_self = (nlist as f32 * 2.0).sqrt() as usize;
            println!(
                "Self-querying IVF-TurboQuant index (bits={}, nprobe={}, rerank_factor=20)...",
                bits, nprobe_self
            );
            let start = Instant::now();
            let (approx_neighbors_self, approx_distances_self) = query_ivf_index_turboquant_self(
                &ivf_tq_idx,
                cli.k,
                Some(nprobe_self),
                None,
                true,
                false,
            )
            .unwrap();
            let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
            let dist_error_self = calculate_mean_distance_ratio(
                true_distances_self.as_ref().unwrap(),
                approx_distances_self.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("IVF-TQ-b{}-nl{} (self)", bits, nlist),
                build_time_ms: build_time,
                query_time_ms: self_query_time,
                total_time_ms: build_time + self_query_time,
                recall_at_k: recall_self,
                mean_dist_rat: dist_error_self,
                index_size_mb,
            });

            println!("-----------------------------");
        }
    }

    print_results_size(
        &format!(
            "{}k samples, {}D - TurboQuant + IVF",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results,
    );
}
