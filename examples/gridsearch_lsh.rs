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

    // Exhaustive query benchmark
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

    // Exhaustive self-query benchmark
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

    // (num_tables, bits_per_hash, slot_bits). `None` lets the index pick:
    // 1 bit per projection for cosine, 2 for squared Euclidean.
    let build_params = [
        (2, 8, None),
        (4, 8, None),
        (8, 8, None),
        (2, 12, None),
        (4, 12, None),
        (8, 12, None),
        (12, 12, None),
        (2, 16, None),
        (4, 16, None),
        (8, 16, None),
        (12, 16, None),
    ];

    for (num_tables, bits_per_hash, slot_bits) in build_params {
        println!(
            "Building LSH index (num_tab={}, bits={}, slot_bits={:?})...",
            num_tables, bits_per_hash, slot_bits
        );
        let start = Instant::now();
        let lsh_index = build_lsh_index(
            data.as_ref(),
            &cli.distance,
            num_tables,
            bits_per_hash,
            slot_bits,
            cli.seed as usize,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = lsh_index.memory_usage_bytes() as f64 / (1024.0 * 1024.0);
        let n_proj = lsh_index.num_projections();
        let q_label = lsh_index.slot_bits();

        let search_budgets = [
            (None, "auto", 0),
            (None, "auto", 1),
            (None, "auto", 2),
            (Some(5000), "5k", 1),
            (Some(2000), "2k", 1),
            (Some(5000), "5k", 2),
            (Some(2000), "2k", 2),
        ];
        for (max_cand, cand_label, probe_mult) in search_budgets {
            let n_probe = n_proj * probe_mult;

            println!(
                "Querying LSH index (cand={}, probes={})...",
                cand_label, n_probe
            );
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_lsh_index(
                query_data.as_ref(),
                &lsh_index,
                cli.k,
                n_probe,
                max_cand,
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
                    "LSH-nt{}-nb{}-q{}-s:{}-n{} (query)",
                    num_tables, bits_per_hash, q_label, cand_label, n_probe
                ),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_rat: dist_error,
                index_size_mb,
            });
        }

        // Self-query benchmark
        println!("Self-querying LSH index...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_lsh_self(&lsh_index, cli.k, None, None, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!(
                "LSH-nt{}-nb{}-q{} (self)",
                num_tables, bits_per_hash, q_label
            ),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
