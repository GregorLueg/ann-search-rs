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

    let data_type = parse_data(&cli.data).unwrap_or_default();

    let (data, cluster_labels): (Mat<f32>, Vec<usize>) = match data_type {
        SyntheticData::GaussianNoise => {
            generate_clustered_data(cli.n_cells, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!("Using data for high dimensional ANN searches...\n");
            generate_clustered_data_high_dim(
                cli.n_cells,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
        SyntheticData::LowRank => generate_low_rank_rotated_data(
            cli.n_cells,
            cli.dim,
            cli.intrinsic_dim,
            cli.n_clusters,
            cli.seed,
        ),
    };

    let mut results = Vec::new();

    // Ground truth
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Self-querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors_self, _) = query_exhaustive_self(&exhaustive_idx, cli.k, false, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let exhaustive_purity = calculate_cluster_purity(&true_neighbors_self, &cluster_labels);

    results.push(BenchmarkResultPurity {
        method: "Exhaustive (self)".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: 1.0,
        cluster_purity: exhaustive_purity,
        index_size_mb,
    });

    println!("-----------------------------");

    // Binary exhaustive benchmarks
    let n_bits_values = [128, 256, 512];

    for n_bits in n_bits_values {
        println!("Building binary exhaustive index (n_bits={})...", n_bits);
        let start = Instant::now();
        let binary_idx =
            build_exhaustive_index_binary(data.as_ref(), n_bits, cli.seed as usize, "random");
        let build_time = start.elapsed().as_secs_f64() * 1000.0;
        let index_size_mb = binary_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        println!(
            "Self-querying binary exhaustive index (n_bits={})...",
            n_bits
        );
        let start = Instant::now();
        let (binary_neighbors_self, _) =
            query_exhaustive_index_binary_self(data.as_ref(), &binary_idx, cli.k, false, false);
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &binary_neighbors_self, cli.k);
        let binary_purity = calculate_cluster_purity(&binary_neighbors_self, &cluster_labels);

        results.push(BenchmarkResultPurity {
            method: format!("Exhaustive-Binary-{}", n_bits),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            cluster_purity: binary_purity,
            index_size_mb,
        });
    }

    println!("-----------------------------");

    // IVF binary benchmarks
    let nlist_values = [
        (cli.n_cells as f32 * 0.5).sqrt() as usize,
        (cli.n_cells as f32).sqrt() as usize,
        (cli.n_cells as f32 * 2.0).sqrt() as usize,
    ];

    for n_bits in n_bits_values {
        for nlist in nlist_values {
            println!(
                "Building IVF binary index (n_bits={}, nlist={})...",
                n_bits, nlist
            );
            let start = Instant::now();
            let ivf_binary_idx = build_ivf_index_binary(
                data.as_ref(),
                "random",
                n_bits,
                Some(nlist),
                None,
                &cli.distance,
                cli.seed as usize,
                false,
            );
            let build_time = start.elapsed().as_secs_f64() * 1000.0;
            let index_size_mb = ivf_binary_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

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

            for nprobe in &nprobe_values {
                if *nprobe > nlist || *nprobe == 0 {
                    continue;
                }

                println!(
                    "Self-querying IVF binary index (n_bits={}, nlist={}, nprobe={})...",
                    n_bits, nlist, nprobe
                );
                let start = Instant::now();
                let (ivf_binary_neighbors_self, _) = query_ivf_index_binary_self(
                    data.as_ref(),
                    &ivf_binary_idx,
                    cli.k,
                    Some(*nprobe),
                    false,
                    false,
                );
                let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

                let recall_self =
                    calculate_recall(&true_neighbors_self, &ivf_binary_neighbors_self, cli.k);
                let ivf_binary_purity =
                    calculate_cluster_purity(&ivf_binary_neighbors_self, &cluster_labels);

                results.push(BenchmarkResultPurity {
                    method: format!("IVF-Binary-{}-nl{}-np{}", n_bits, nlist, nprobe),
                    build_time_ms: build_time,
                    query_time_ms: self_query_time,
                    total_time_ms: build_time + self_query_time,
                    recall_at_k: recall_self,
                    cluster_purity: ivf_binary_purity,
                    index_size_mb,
                });
            }
        }
    }

    println!("-----------------------------");

    print_results_purity(
        &format!(
            "{}k cells, {}D - Cluster Structure Preservation",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
