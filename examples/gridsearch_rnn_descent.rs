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

    // Exhaustive baseline
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
        median_dist_rat: 1.0,
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
        median_dist_rat: 1.0,
        index_size_mb,
    });

    println!("-----------------------------");

    // S x R x T1 x T2 grid.
    //
    // T2 is the paper's 15 rather than the 10 this grid used to run, with T1
    // dropped one round to pay for it. Measured at S20/R96 on 150k x 32D
    // Gaussian: `(T1=3, T2=15)` builds in 933 ms at 0.9316 recall against
    // `(T1=4, T2=10)` at 967 ms and 0.9311, so the swap is both faster and
    // slightly more accurate. AddReverseEdges re-arms every surviving edge as
    // new, so the pass straight after it does the most distance work in a
    // round, whereas the tail passes a lower T2 removes are the cheapest ones
    // (over half of their nodes hold no new edge at all and are skipped).
    // The R64 and R128 rows follow the same swap by analogy; only the R96 row
    // was measured.
    let build_params: &[(usize, usize, usize, usize)] =
        &[(20, 64, 2, 15), (20, 96, 3, 15), (40, 128, 2, 15)];

    // Query-time K sweep (paper Section 4.4). None uses the default min(32, R).
    // Include Some(r) as a "walk all neighbours" baseline for comparison.
    let k_search_values: &[Option<usize>] = &[Some(16), None, Some(64), Some(usize::MAX)];
    let ef_search: Option<usize> = None; // 100 by default, matching the paper

    for &(s, r, t1, t2) in build_params {
        println!(
            "Building RNN-Descent (S={}, R={}, T1={}, T2={})...",
            s, r, t1, t2
        );

        let start = Instant::now();
        let rnn_idx = build_rnn_descent_index(
            data.as_ref(),
            s,
            r,
            t1,
            t2,
            &cli.distance,
            None,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = rnn_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        for &k_search in k_search_values {
            let k_label = match k_search {
                None => "auto".to_string(),
                Some(v) if v == usize::MAX => format!("R={}", r),
                Some(v) => v.to_string(),
            };
            println!("Querying RNN-Descent (K={})...", k_label);

            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_rnn_descent_index(
                query_data.as_ref(),
                &rnn_idx,
                cli.k,
                ef_search,
                k_search,
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
            let dist_error_median = calculate_median_distance_ratio(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            );

            results.push(BenchmarkResultSize {
                method: format!("RNN-S{}-R{}-T1{}-K{} (query)", s, r, t1, k_label),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: recall,
                mean_dist_rat: dist_error,
                median_dist_rat: dist_error_median,
                index_size_mb,
            });
        }

        println!("Self-querying RNN-Descent...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_rnn_descent_self(&rnn_idx, cli.k, None, None, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );
        let dist_error_self_median = calculate_median_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!("RNN-S{}-R{}-T1{} (self)", s, r, t1),
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
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
