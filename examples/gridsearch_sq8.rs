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

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);
    let mut results = Vec::new();

    // Exhaustive query benchmark
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, _) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, false, false);
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

    // Exhaustive self-query benchmark
    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, _) = query_exhaustive_self(&exhaustive_idx, cli.k, false, false);
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

    println!("-----------------------------------------------------------------------------------------------");

    // Exhaustive SQ8 query benchmark
    println!("Building exhaustive SQ8 index...");
    let start = Instant::now();
    let exhaustive_sq8_idx = build_exhaustive_sq8_index(data.as_ref(), &cli.distance, false);
    let build_time_sq8 = start.elapsed().as_secs_f64() * 1000.0;

    let index_size_mb_sq8 = exhaustive_sq8_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive SQ8 index...");
    let start = Instant::now();
    let (sq8_neighbors, _) = query_exhaustive_sq8_index(
        query_data.as_ref(),
        &exhaustive_sq8_idx,
        cli.k,
        false,
        false,
    );
    let query_time_sq8 = start.elapsed().as_secs_f64() * 1000.0;

    let recall_sq8 = calculate_recall(&true_neighbors, &sq8_neighbors, cli.k);

    results.push(BenchmarkResultSize {
        method: "Exhaustive-SQ8 (query)".to_string(),
        build_time_ms: build_time_sq8,
        query_time_ms: query_time_sq8,
        total_time_ms: build_time_sq8 + query_time_sq8,
        recall_at_k: recall_sq8,
        mean_dist_err: f64::NAN,
        index_size_mb: index_size_mb_sq8,
    });

    // Exhaustive SQ8 self-query benchmark
    println!("Self-querying exhaustive SQ8 index...");
    let start = Instant::now();
    let (sq8_neighbors_self, _) =
        query_exhaustive_sq8_self(&exhaustive_sq8_idx, cli.k, false, false);
    let self_query_time_sq8 = start.elapsed().as_secs_f64() * 1000.0;

    let recall_sq8_self = calculate_recall(&true_neighbors_self, &sq8_neighbors_self, cli.k);

    results.push(BenchmarkResultSize {
        method: "Exhaustive-SQ8 (self)".to_string(),
        build_time_ms: build_time_sq8,
        query_time_ms: self_query_time_sq8,
        total_time_ms: build_time_sq8 + self_query_time_sq8,
        recall_at_k: recall_sq8_self,
        mean_dist_err: f64::NAN,
        index_size_mb: index_size_mb_sq8,
    });

    println!("-----------------------------------------------------------------------------------------------");

    let nlist_values = [
        (cli.n_cells as f32 * 0.5).sqrt() as usize,
        (cli.n_cells as f32).sqrt() as usize,
        (cli.n_cells as f32 * 2.0).sqrt() as usize,
    ];

    for nlist in nlist_values {
        println!("Building IVF-SQ8 index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_sq8_idx = build_ivf_sq8_index(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = ivf_sq8_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

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

        // Query benchmarks
        for nprobe in &nprobe_values {
            if *nprobe > nlist || *nprobe == 0 {
                continue;
            }

            println!("Querying IVF-SQ8 (nlist={}, nprobe={})...", nlist, nprobe);
            let start = Instant::now();
            let (approx_neighbors, _) = query_ivf_sq8_index(
                query_data.as_ref(),
                &ivf_sq8_idx,
                cli.k,
                Some(*nprobe),
                false,
                false,
            );
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);

            results.push(BenchmarkResultSize {
                method: format!("IVF-SQ8-nl{}-np{} (query)", nlist, nprobe),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_err: f64::NAN,
                index_size_mb,
            });
        }

        // Self-query benchmark
        let nprobe_self = (nlist as f32 * 2.0).sqrt() as usize;
        println!("Self-querying IVF-SQ8 index (nprobe={})...", nprobe_self);
        let start = Instant::now();
        let (approx_neighbors_self, _) =
            query_ivf_sq8_self(&ivf_sq8_idx, cli.k, Some(nprobe_self), false, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);

        results.push(BenchmarkResultSize {
            method: format!("IVF-SQ8-nl{} (self)", nlist),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_err: f64::NAN,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
