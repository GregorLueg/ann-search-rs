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
    let (true_neighbours, _) =
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
    let (true_neighbours_self, _) = query_exhaustive_self(&exhaustive_idx, cli.k, false, false);
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

    let nlist_values = [
        (cli.n_cells as f32 * 0.5).sqrt() as usize,
        (cli.n_cells as f32).sqrt() as usize,
        (cli.n_cells as f32 * 2.0).sqrt() as usize,
    ];

    let m_values: Vec<usize> = if cli.dim >= 128 {
        vec![16, 32, 48]
    } else {
        vec![8, 16]
    };

    for nlist in nlist_values {
        for m in &m_values {
            if cli.dim % m != 0 {
                continue;
            }

            println!("Building IVF-PQ index (nlist={}, m={})...", nlist, m);
            let start = Instant::now();
            let ivf_pq_idx = build_ivf_pq_index(
                data.as_ref(),
                Some(nlist),
                *m,
                None,
                None,
                &cli.distance,
                cli.seed as usize,
                false,
            );
            let build_time = start.elapsed().as_secs_f64() * 1000.0;

            let index_size_mb = ivf_pq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

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

                println!(
                    "Querying IVF-PQ (nlist={}, m={}, np={})...",
                    nlist, m, nprobe
                );
                let start = Instant::now();
                let (approx_neighbours, _) = query_ivf_pq_index(
                    query_data.as_ref(),
                    &ivf_pq_idx,
                    cli.k,
                    Some(*nprobe),
                    false,
                    false,
                );
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let recall = calculate_recall(&true_neighbours, &approx_neighbours, cli.k);

                results.push(BenchmarkResultSize {
                    method: format!("IVF-PQ-nl{}-m{}-np{} (query)", nlist, m, nprobe),
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
            println!("Self-querying IVF-PQ index (nprobe={})...", nprobe_self);
            let start = Instant::now();
            let (approx_neighbours_self, _) =
                query_ivf_pq_index_self(&ivf_pq_idx, cli.k, Some(nprobe_self), false, false);
            let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

            let recall_self =
                calculate_recall(&true_neighbours_self, &approx_neighbours_self, cli.k);

            results.push(BenchmarkResultSize {
                method: format!("IVF-PQ-nl{}-m{} (self)", nlist, m),
                build_time_ms: build_time,
                query_time_ms: self_query_time,
                total_time_ms: build_time + self_query_time,
                recall_at_k: recall_self,
                mean_dist_err: f64::NAN,
                index_size_mb,
            });
        }
    }

    print_results_size(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
