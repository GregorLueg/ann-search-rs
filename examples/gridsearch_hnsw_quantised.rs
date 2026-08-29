//! Compares the quantised HNSW against the full-precision one at matched
//! build parameters.
//!
//! Both indices are built and queried over the same data with the same
//! `(M, ef_construction, ef_search)` grid, so the columns differ only by what
//! the graph stores and how it measures distance. Recall is against an
//! exhaustive search, and `DistRat` is the ratio of reported to true distance,
//! which for the quantised index is its distance distortion rather than a
//! search error.

mod commons;

use ann_search_rs::prelude::UniformQuantParams;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Tail-trim fractions to compare. Zero is the pyglass default for 8-bit;
/// the non-zero settings are what the shared scale wants when a handful of
/// points sit far out in one dimension.
const DROP_RATIOS: [f64; 3] = [0.0, 1e-3, 1e-2];

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

    // The same codec the graph uses, scanned exhaustively. Recall here is the
    // ceiling the quantised HNSW rows below are working against: anything they
    // lose beyond this is the graph, anything they lose up to it is the codec.
    println!("Building exhaustive SQ8 index...");
    let start = Instant::now();
    let sq8_idx = build_exhaustive_sq8_index(data.as_ref(), &cli.distance, None, false).unwrap();
    let build_time_sq8 = start.elapsed().as_secs_f64() * 1000.0;
    let size_sq8 = sq8_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive SQ8 index...");
    let start = Instant::now();
    let (sq8_neighbors, _) =
        query_exhaustive_sq8_index(query_data.as_ref(), &sq8_idx, cli.k, false, false).unwrap();
    let query_time_sq8 = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: "Exhaustive-SQ8 (query)".to_string(),
        build_time_ms: build_time_sq8,
        query_time_ms: query_time_sq8,
        total_time_ms: build_time_sq8 + query_time_sq8,
        recall_at_k: calculate_recall(&true_neighbors, &sq8_neighbors, cli.k),
        mean_dist_rat: f64::NAN,
        index_size_mb: size_sq8,
    });

    println!("-----------------------------");

    let build_params = [(16, 100), (16, 200), (24, 200), (32, 200)];
    let ef_search_values = [50, 100, 200];

    // Full-precision baseline at the same grid
    for (m, ef_construction) in build_params {
        println!("Building HNSW index (M={m}, ef_construction={ef_construction})...");
        let start = Instant::now();
        let hnsw_idx = build_hnsw_index(
            data.as_ref(),
            m,
            ef_construction,
            &cli.distance,
            cli.seed as usize,
            false,
        );
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = hnsw_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        for ef_search in ef_search_values {
            println!("Querying HNSW index (ef_search={ef_search})...");
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_hnsw_index(
                query_data.as_ref(),
                &hnsw_idx,
                cli.k,
                ef_search,
                true,
                false,
            )
            .unwrap();
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            results.push(BenchmarkResultSize {
                method: format!("HNSW-M{m}-ef{ef_construction}-s{ef_search} (query)"),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: calculate_recall(&true_neighbors, &approx_neighbors, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    approx_distances.as_ref().unwrap(),
                    cli.k,
                ),
                index_size_mb,
            });
        }

        println!("Self-querying HNSW index...");
        let start = Instant::now();
        let (approx_self, approx_dists_self) =
            query_hnsw_self(&hnsw_idx, cli.k, 100, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResultSize {
            method: format!("HNSW-M{m}-ef{ef_construction} (self)"),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: calculate_recall(&true_neighbors_self, &approx_self, cli.k),
            mean_dist_rat: calculate_mean_distance_ratio(
                true_distances_self.as_ref().unwrap(),
                approx_dists_self.as_ref().unwrap(),
                cli.k,
            ),
            index_size_mb,
        });
    }

    println!("-----------------------------");

    // Quantised, same grid
    for (m, ef_construction) in build_params {
        println!("Building HNSW-SQ8U index (M={m}, ef_construction={ef_construction})...");
        let start = Instant::now();
        let idx = build_hnsw_sq8u_index(
            data.as_ref(),
            m,
            ef_construction,
            &cli.distance,
            cli.seed as usize,
            None,
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        for ef_search in ef_search_values {
            println!("Querying HNSW-SQ8U index (ef_search={ef_search})...");
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_hnsw_sq8u_index(
                query_data.as_ref(),
                &idx,
                cli.k,
                ef_search,
                true,
                false,
            )
            .unwrap();
            let query_time = start.elapsed().as_secs_f64() * 1000.0;

            results.push(BenchmarkResultSize {
                method: format!("HNSW-SQ8U-M{m}-ef{ef_construction}-s{ef_search} (query)"),
                build_time_ms: build_time,
                query_time_ms: query_time,
                total_time_ms: build_time + query_time,
                recall_at_k: calculate_recall(&true_neighbors, &approx_neighbors, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    approx_distances.as_ref().unwrap(),
                    cli.k,
                ),
                index_size_mb,
            });
        }

        println!("Self-querying HNSW-SQ8U index...");
        let start = Instant::now();
        let (approx_self, approx_dists_self) =
            query_hnsw_sq8u_self(&idx, cli.k, 100, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResultSize {
            method: format!("HNSW-SQ8U-M{m}-ef{ef_construction} (self)"),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: calculate_recall(&true_neighbors_self, &approx_self, cli.k),
            mean_dist_rat: calculate_mean_distance_ratio(
                true_distances_self.as_ref().unwrap(),
                approx_dists_self.as_ref().unwrap(),
                cli.k,
            ),
            index_size_mb,
        });
    }

    println!("-----------------------------");

    // What the tail trim buys. One fixed graph setting, calibration varied:
    // a shared scale means the widest dimension sets the resolution for all of
    // them, so this is the knob that decides how much of the code range the
    // bulk of the data actually gets.
    let (m, ef_construction) = (16, 200);
    for drop_ratio in DROP_RATIOS {
        println!("Building HNSW-SQ8U index (drop_ratio={drop_ratio})...");
        let params = UniformQuantParams::new(drop_ratio, None, cli.seed as usize);
        let start = Instant::now();
        let idx = build_hnsw_sq8u_index(
            data.as_ref(),
            m,
            ef_construction,
            &cli.distance,
            cli.seed as usize,
            Some(params),
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        let start = Instant::now();
        let (approx_neighbors, approx_distances) =
            query_hnsw_sq8u_index(query_data.as_ref(), &idx, cli.k, 100, true, false).unwrap();
        let query_time = start.elapsed().as_secs_f64() * 1000.0;

        results.push(BenchmarkResultSize {
            method: format!("HNSW-SQ8U-drop{drop_ratio} (query)"),
            build_time_ms: build_time,
            query_time_ms: query_time,
            total_time_ms: build_time + query_time,
            recall_at_k: calculate_recall(&true_neighbors, &approx_neighbors, cli.k),
            mean_dist_rat: calculate_mean_distance_ratio(
                true_distances.as_ref().unwrap(),
                approx_distances.as_ref().unwrap(),
                cli.k,
            ),
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
