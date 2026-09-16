//! Compares the two RaBitQ graph indices against each other.
//!
//! Both walk a graph, and both estimate with RaBitQ codes, but they spend
//! memory in opposite directions. QG keeps the float vectors and stores every
//! vertex's code once per in-edge, so it computes exact distances for the
//! vertices it pops. The quantised HNSW drops the vectors entirely and holds
//! one multi-bit code per vertex, so it answers from the codes alone and the
//! `ex_bits` width is what buys accuracy back.
//!
//! The column to read across the two is `index_size_mb` at matched recall.

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

    // Ground truth
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

    let true_distances = true_distances.unwrap();
    let true_distances_self = true_distances_self.unwrap();

    let ef_values = [cli.k, cli.k * 2, cli.k * 4, cli.k * 8];
    let ef_self = cli.k * 4;

    println!("-----------------------------");

    // QG: Vamana carrying its neighbours' one-bit codes, float vectors kept.
    // `l_build` sets the second Vamana pass only; the first runs at the crate
    // default, which is deliberately narrow.
    for degree in [32usize, 64] {
        for l_build in [64usize, 128] {
            println!(
                "Building QG index (degree={}, l_build={})...",
                degree, l_build
            );
            let start = Instant::now();
            let qg_idx = build_qg_index(
                data.as_ref(),
                degree,
                l_build,
                None,
                1.2,
                1.2,
                &cli.distance,
                cli.seed as usize,
            )
            .unwrap();
            let build_time = start.elapsed().as_secs_f64() * 1000.0;
            let index_size_mb = qg_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

            for &ef in &ef_values {
                println!(
                    "Querying QG index (degree={}, l_build={}, ef={})...",
                    degree, l_build, ef
                );
                let start = Instant::now();
                let (neighbors, _) =
                    query_qg_index(query_data.as_ref(), &qg_idx, cli.k, ef, false, false).unwrap();
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let approx = exact_distances(&data, &query_data, &neighbors, &cli.distance);

                results.push(BenchmarkResultSize {
                    method: format!("QG-d{}-l{}-ef{} (query)", degree, l_build, ef),
                    build_time_ms: build_time,
                    query_time_ms: query_time,
                    total_time_ms: build_time + query_time,
                    recall_at_k: calculate_recall(&true_neighbors, &neighbors, cli.k),
                    mean_dist_rat: calculate_mean_distance_ratio(&true_distances, &approx, cli.k),
                    median_dist_rat: calculate_median_distance_ratio(
                        &true_distances,
                        &approx,
                        cli.k,
                    ),
                    index_size_mb,
                });
            }

            println!(
                "Self-querying QG index (degree={}, l_build={})...",
                degree, l_build
            );
            let start = Instant::now();
            let (neighbors_self, _) = query_qg_self(&qg_idx, cli.k, ef_self, false, false).unwrap();
            let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

            let approx_self = exact_distances(&data, &data, &neighbors_self, &cli.distance);

            results.push(BenchmarkResultSize {
                method: format!("QG-d{}-l{} (self)", degree, l_build),
                build_time_ms: build_time,
                query_time_ms: self_query_time,
                total_time_ms: build_time + self_query_time,
                recall_at_k: calculate_recall(&true_neighbors_self, &neighbors_self, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    &true_distances_self,
                    &approx_self,
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    &true_distances_self,
                    &approx_self,
                    cli.k,
                ),
                index_size_mb,
            });
        }
    }

    println!("-----------------------------");

    // Quantised HNSW over RaBitQ+ codes. The graph is linked on exact
    // distances and the vectors are then dropped, so `ex_bits` moves accuracy
    // and size together without touching the topology.
    let ef_construction = 200;

    for m in [16usize, 32] {
        for ex_bits in [0usize, 3, 5] {
            println!(
                "Building HNSW-RaBitQ index (m={}, ef_c={}, ex_bits={})...",
                m, ef_construction, ex_bits
            );
            let start = Instant::now();
            let hnsw_idx = build_hnsw_rabitq_index(
                data.as_ref(),
                m,
                ef_construction,
                &cli.distance,
                ex_bits,
                None,
                None,
                cli.seed as usize,
                false,
            )
            .unwrap();
            let build_time = start.elapsed().as_secs_f64() * 1000.0;
            let index_size_mb = hnsw_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

            for &ef in &ef_values {
                println!(
                    "Querying HNSW-RaBitQ index (m={}, ex_bits={}, ef={})...",
                    m, ex_bits, ef
                );
                let start = Instant::now();
                let (neighbors, _) = query_hnsw_rabitq_index(
                    query_data.as_ref(),
                    &hnsw_idx,
                    cli.k,
                    ef,
                    false,
                    false,
                )
                .unwrap();
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let approx = exact_distances(&data, &query_data, &neighbors, &cli.distance);

                results.push(BenchmarkResultSize {
                    method: format!("HnswRaBitQ-m{}-ex{}-ef{} (query)", m, ex_bits, ef),
                    build_time_ms: build_time,
                    query_time_ms: query_time,
                    total_time_ms: build_time + query_time,
                    recall_at_k: calculate_recall(&true_neighbors, &neighbors, cli.k),
                    mean_dist_rat: calculate_mean_distance_ratio(&true_distances, &approx, cli.k),
                    median_dist_rat: calculate_median_distance_ratio(
                        &true_distances,
                        &approx,
                        cli.k,
                    ),
                    index_size_mb,
                });
            }

            println!(
                "Self-querying HNSW-RaBitQ index (m={}, ex_bits={})...",
                m, ex_bits
            );
            let start = Instant::now();
            let (neighbors_self, _) =
                query_hnsw_rabitq_self(&hnsw_idx, cli.k, ef_self, false, false).unwrap();
            let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

            let approx_self = exact_distances(&data, &data, &neighbors_self, &cli.distance);

            results.push(BenchmarkResultSize {
                method: format!("HnswRaBitQ-m{}-ex{} (self)", m, ex_bits),
                build_time_ms: build_time,
                query_time_ms: self_query_time,
                total_time_ms: build_time + self_query_time,
                recall_at_k: calculate_recall(&true_neighbors_self, &neighbors_self, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    &true_distances_self,
                    &approx_self,
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    &true_distances_self,
                    &approx_self,
                    cli.k,
                ),
                index_size_mb,
            });
        }
    }

    print_results_size(
        &format!(
            "{}k samples, {}D - RaBitQ graph indices",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results,
    );
}
