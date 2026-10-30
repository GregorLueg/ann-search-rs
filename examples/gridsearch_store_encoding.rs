mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::path::Path;
use std::time::Instant;
use tempfile::TempDir;
use thousands::*;

/// Encodings compared against each other at identical index parameters.
const ENCODINGS: [StoreEncoding; 3] =
    [StoreEncoding::Full, StoreEncoding::F16, StoreEncoding::Bf16];

/// Over-fetch factors for the re-ranking stage.
const RERANK_FACTORS: [usize; 3] = [5, 10, 20];

/// Bits per coordinate for the TurboQuant indices.
const TQ_BITS: usize = 4;

/// Size of the store file in MB.
fn store_mb(dir: &Path) -> f64 {
    std::fs::metadata(dir.join("vectors.bin")).unwrap().len() as f64 / (1024.0 * 1024.0)
}

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

    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance, Verbosity::Quiet);
    let (true_neighbors, true_distances) = query_exhaustive_index(
        query_data.as_ref(),
        &exhaustive_idx,
        cli.k,
        true,
        Verbosity::Quiet,
    )
    .unwrap();
    let true_distances = true_distances.unwrap();

    let nlist = (cli.n_samples as f32).sqrt() as usize;
    let nprobe = ((nlist as f32).sqrt() * 2.0) as usize;

    let mut record = |method: String, build_ms: f64, query_ms: f64, nn: &[Vec<usize>], mb: f64| {
        let exact = exact_distances(&data, &query_data, nn, &cli.distance);
        results.push(BenchmarkResultSize {
            method,
            build_time_ms: build_ms,
            query_time_ms: query_ms,
            total_time_ms: build_ms + query_ms,
            recall_at_k: calculate_recall(&true_neighbors, nn, cli.k),
            mean_dist_rat: calculate_mean_distance_ratio(&true_distances, &exact, cli.k),
            median_dist_rat: calculate_median_distance_ratio(&true_distances, &exact, cli.k),
            index_size_mb: mb,
        });
    };

    for enc in ENCODINGS {
        let dir = TempDir::new().unwrap();
        println!("Exhaustive-RaBitQ, {enc:?} store...");
        let start = Instant::now();
        let idx = build_exhaustive_index_rabitq(
            data.as_ref(),
            None,
            &cli.distance,
            cli.seed as usize,
            true,
            Some(dir.path()),
            enc,
            Verbosity::Quiet,
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        for rf in RERANK_FACTORS {
            let start = Instant::now();
            let (nn, _) = query_exhaustive_index_rabitq(
                query_data.as_ref(),
                &idx,
                cli.k,
                None,
                true,
                Some(rf),
                false,
                Verbosity::Quiet,
            )
            .unwrap();
            let query_ms = start.elapsed().as_secs_f64() * 1000.0;
            record(
                format!("ExhRaBitQ-{enc:?}-rf{rf}"),
                build_ms,
                query_ms,
                &nn,
                store_mb(dir.path()),
            );
        }
    }

    for enc in ENCODINGS {
        let dir = TempDir::new().unwrap();
        println!("IVF-RaBitQ (nlist={nlist}, nprobe={nprobe}), {enc:?} store...");
        let start = Instant::now();
        let idx = build_ivf_index_rabitq(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            true,
            Some(dir.path()),
            enc,
            Verbosity::Quiet,
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        for rf in RERANK_FACTORS {
            let start = Instant::now();
            let (nn, _) = query_ivf_index_rabitq(
                query_data.as_ref(),
                &idx,
                cli.k,
                Some(nprobe),
                true,
                Some(rf),
                false,
                Verbosity::Quiet,
            )
            .unwrap();
            let query_ms = start.elapsed().as_secs_f64() * 1000.0;
            record(
                format!("IVF-RaBitQ-{enc:?}-np{nprobe}-rf{rf}"),
                build_ms,
                query_ms,
                &nn,
                store_mb(dir.path()),
            );
        }
    }

    for enc in ENCODINGS {
        let dir = TempDir::new().unwrap();
        println!("IVF-TQ (bits={TQ_BITS}, nlist={nlist}, nprobe={nprobe}), {enc:?} store...");
        let start = Instant::now();
        let idx = build_ivf_index_turboquant(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            TQ_BITS,
            cli.seed as usize,
            true,
            Some(dir.path()),
            enc,
            Verbosity::Quiet,
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        for rf in RERANK_FACTORS {
            let start = Instant::now();
            let (nn, _) = query_ivf_index_turboquant(
                query_data.as_ref(),
                &idx,
                cli.k,
                Some(nprobe),
                true,
                Some(rf),
                false,
                Verbosity::Quiet,
            )
            .unwrap();
            let query_ms = start.elapsed().as_secs_f64() * 1000.0;
            record(
                format!("IVF-TQ{TQ_BITS}-{enc:?}-np{nprobe}-rf{rf}"),
                build_ms,
                query_ms,
                &nn,
                store_mb(dir.path()),
            );
        }
    }

    print_results_size(
        &format!(
            "{}k samples, {}D, {} data, store encodings (size column = store file MB)",
            cli.n_samples / 1000,
            cli.dim,
            cli.data
        ),
        &results,
    );
}
