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

    // Binary exhaustive benchmarks - increase bits if higher dimensionality
    // is used
    let n_bits_values = if cli.dim <= 64 {
        vec![
            (256, "random"),
            // (256, "itq"),
            (512, "random"),
            // (512, "itq")
        ]
    } else {
        vec![
            (256, "random"),
            (256, "itq"),
            (512, "random"),
            (512, "itq"),
            (1024, "random"),
            (1024, "itq"),
        ]
    };
    let rerank_factors = [5, 10, 20];

    for (n_bits, init) in &n_bits_values {
        let temp_dir = TempDir::new().unwrap();

        println!(
            "Building binary exhaustive index (n_bits={}, init={})...",
            n_bits, init
        );
        let start = Instant::now();
        let binary_idx = build_exhaustive_index_binary(
            data.as_ref(),
            *n_bits,
            cli.seed as usize,
            init,
            &cli.distance,
            true,
            Some(temp_dir.path()),
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = binary_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Query without reranking
        println!(
            "Querying binary exhaustive index (n_bits={}, init={}, no rerank)...",
            n_bits, init
        );
        let start = Instant::now();
        let (binary_neighbors, _) = query_exhaustive_index_binary(
            query_data.as_ref(),
            &binary_idx,
            cli.k,
            false,
            None,
            false,
            false,
        );
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall = calculate_recall(&true_neighbors, &binary_neighbors, cli.k);

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveBinary-{}-{}_no_rr (query)", n_bits, init),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: recall,
            mean_dist_err: f64::NAN,
            index_size_mb,
        });

        // Query with reranking
        for &rerank_factor in &rerank_factors {
            println!(
                "Querying binary exhaustive index (n_bits={}, init={}, rerank_factor={})...",
                n_bits, init, rerank_factor
            );
            let start = Instant::now();
            let (binary_neighbors, binary_distances) = query_exhaustive_index_binary(
                query_data.as_ref(),
                &binary_idx,
                cli.k,
                true,
                Some(rerank_factor),
                true,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &binary_neighbors, cli.k);
            let dist_error = calculate_dist_error(
                true_distances.as_ref().unwrap(),
                binary_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!(
                    "ExhaustiveBinary-{}-{}-rf{} (query)",
                    n_bits, init, rerank_factor
                ),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: dist_error,
                index_size_mb,
            });
        }

        println!(
            "Self-querying binary exhaustive index (n_bits={}, init={})...",
            n_bits, init
        );
        let start = Instant::now();
        let (binary_neighbors_self, binary_distances_self) =
            query_exhaustive_index_binary_self(&binary_idx, cli.k, Some(10), true, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &binary_neighbors_self, cli.k);
        let dist_error_self = calculate_dist_error(
            true_distances_self.as_ref().unwrap(),
            binary_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("ExhaustiveBinary-{}-{} (self)", n_bits, init),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: dist_error_self,
            index_size_mb,
        });
    }

    // println!("-----------------------------");

    // // IVF binary benchmarks
    // let nlist_values = [
    //     (cli.n_cells as f32 * 0.5).sqrt() as usize,
    //     (cli.n_cells as f32).sqrt() as usize,
    //     (cli.n_cells as f32 * 2.0).sqrt() as usize,
    // ];

    // for (n_bits, init) in n_bits_values {
    //     for nlist in nlist_values {
    //         let temp_dir = TempDir::new().unwrap();

    //         println!(
    //             "Building IVF binary index (n_bits={}, nlist={}, init={})...",
    //             n_bits, nlist, init
    //         );
    //         let start = Instant::now();
    //         let ivf_binary_idx = build_ivf_index_binary(
    //             data.as_ref(),
    //             init,
    //             n_bits,
    //             Some(nlist),
    //             None,
    //             &cli.distance,
    //             cli.seed as usize,
    //             true,
    //             Some(temp_dir.path()),
    //             false,
    //         )
    //         .unwrap();
    //         let build_time = start.elapsed().as_secs_f64() * 1000.0;
    //         let index_size_mb = ivf_binary_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    //         let nprobe_values = [
    //             (nlist as f32).sqrt() as usize,
    //             (nlist as f32 * 2.0).sqrt() as usize,
    //             (0.05 * nlist as f32) as usize,
    //         ];
    //         let mut nprobe_values: Vec<_> = nprobe_values
    //             .into_iter()
    //             .collect::<HashSet<_>>()
    //             .into_iter()
    //             .collect();
    //         nprobe_values.sort();

    //         // Query without reranking
    //         for nprobe in &nprobe_values {
    //             if *nprobe > nlist || *nprobe == 0 {
    //                 continue;
    //             }

    //             println!(
    //                 "Querying IVF binary index (n_bits={}, init={}, nlist={}, nprobe={}, no rerank)...",
    //                 n_bits, init, nlist, nprobe
    //             );
    //             let start = Instant::now();
    //             let (ivf_binary_neighbors, _) = query_ivf_index_binary(
    //                 query_data.as_ref(),
    //                 &ivf_binary_idx,
    //                 cli.k,
    //                 Some(*nprobe),
    //                 false,
    //                 None,
    //                 false,
    //                 false,
    //             );
    //             let query_time = start.elapsed().as_secs_f64() * 1000.0;

    //             let recall = calculate_recall(&true_neighbors, &ivf_binary_neighbors, cli.k);

    //             results.push(BenchmarkResultSize {
    //                 method: format!(
    //                     "IVF-Binary-{}-nl{}-np{}-rf0-{} (query)",
    //                     n_bits, nlist, nprobe, init
    //                 ),
    //                 build_time_ms: build_time,
    //                 query_time_ms: query_time,
    //                 total_time_ms: build_time + query_time,
    //                 recall_at_k: recall,
    //                 mean_dist_err: f64::NAN,
    //                 index_size_mb,
    //             });
    //         }

    //         // Query with reranking
    //         for nprobe in &nprobe_values {
    //             if *nprobe > nlist || *nprobe == 0 {
    //                 continue;
    //             }

    //             for &rerank_factor in &rerank_factors {
    //                 println!(
    //                     "Querying IVF binary index (n_bits={}, init={}, nlist={}, nprobe={}, rerank_factor={})...",
    //                     n_bits, init, nlist, nprobe, rerank_factor
    //                 );
    //                 let start = Instant::now();
    //                 let (ivf_binary_neighbors, ivf_binary_distances) = query_ivf_index_binary(
    //                     query_data.as_ref(),
    //                     &ivf_binary_idx,
    //                     cli.k,
    //                     Some(*nprobe),
    //                     true,
    //                     Some(rerank_factor),
    //                     true,
    //                     false,
    //                 );
    //                 let query_time = start.elapsed().as_secs_f64() * 1000.0;

    //                 let recall = calculate_recall(&true_neighbors, &ivf_binary_neighbors, cli.k);
    //                 let dist_error = calculate_dist_error(
    //                     true_distances.as_ref().unwrap(),
    //                     ivf_binary_distances.as_ref().unwrap(),
    //                     cli.k,
    //                 );

    //                 results.push(BenchmarkResultSize {
    //                     method: format!(
    //                         "IVF-Binary-{}-nl{}-np{}-rf{}-{} (query)",
    //                         n_bits, nlist, nprobe, rerank_factor, init
    //                     ),
    //                     build_time_ms: build_time,
    //                     query_time_ms: query_time,
    //                     total_time_ms: build_time + query_time,
    //                     recall_at_k: recall,
    //                     mean_dist_err: dist_error,
    //                     index_size_mb,
    //                 });
    //             }
    //         }

    //         println!(
    //             "Self-querying IVF binary index (n_bits={}, init={}, nlist={})...",
    //             n_bits, init, nlist
    //         );
    //         let start = Instant::now();
    //         let (ivf_binary_neighbors_self, ivf_binary_distances_self) =
    //             query_ivf_index_binary_self(
    //                 &ivf_binary_idx,
    //                 cli.k,
    //                 Some((nlist as f32).sqrt() as usize),
    //                 Some(10),
    //                 true,
    //                 false,
    //             );
    //         let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    //         let recall_self =
    //             calculate_recall(&true_neighbors_self, &ivf_binary_neighbors_self, cli.k);
    //         let dist_error_self = calculate_dist_error(
    //             true_distances_self.as_ref().unwrap(),
    //             ivf_binary_distances_self.as_ref().unwrap(),
    //             cli.k,
    //         );

    //         results.push(BenchmarkResultSize {
    //             method: format!("IVF-Binary-{}-nl{}-{} (self)", n_bits, nlist, init),
    //             build_time_ms: build_time,
    //             query_time_ms: self_query_time,
    //             total_time_ms: build_time + self_query_time,
    //             recall_at_k: recall_self,
    //             mean_dist_err: dist_error_self,
    //             index_size_mb,
    //         });
    //     }
    // }

    // println!("-----------------------------");

    print_results_size(
        &format!(
            "{}k cells, {}D - Binary Quantisation",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
