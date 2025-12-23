mod commons;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::collections::HashSet;
use std::time::Instant;
use thousands::*;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = DEFAULT_N_CELLS)]
    n_cells: usize,

    #[arg(long, default_value_t = DEFAULT_DIM)]
    dim: usize,

    #[arg(long, default_value_t = DEFAULT_N_CLUSTERS)]
    n_clusters: usize,

    #[arg(long, default_value_t = DEFAULT_K)]
    k: usize,

    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,

    #[arg(long, default_value = DEFAULT_DISTANCE)]
    distance: String,

    #[arg(long, default_value = DEFAULT_DATA)]
    data: String,
}

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

    let data: Mat<f32> = match data_type {
        SyntheticData::GaussianNoise => {
            generate_clustered_data(cli.n_cells, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!("Using data for high dimensional ANN searches.\n");
            generate_clustered_data_high_dim(
                cli.n_cells,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
    };

    let query_data = data.as_ref();
    let mut results = Vec::new();

    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, _) =
        query_exhaustive_index(query_data, &exhaustive_idx, cli.k, false, false);
    let query_time = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResult {
        method: "Exhaustive".to_string(),
        build_time_ms: build_time,
        query_time_ms: query_time,
        total_time_ms: build_time + query_time,
        recall_at_k: 1.0,
        mean_dist_err: 0.0,
    });

    println!("-----------------------------------------------------------------------------------------------");

    let nlist_values = [10, 20, 25, 50, 100];

    // IVF-OPQ benchmarks
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

            println!("Building IVF-OPQ index (nlist={}, m={})...", nlist, m);
            let start = Instant::now();
            let ivf_opq_idx = build_ivf_opq_index(
                data.as_ref(),
                nlist,
                *m,
                None,
                None,
                None,
                &cli.distance,
                cli.seed as usize,
                false,
            );
            let build_time = start.elapsed().as_secs_f64() * 1000.0;

            let nprobe_values = [
                (0.05 * nlist as f64) as usize,
                (0.1 * nlist as f64) as usize,
                (0.15 * nlist as f64) as usize,
            ];

            let mut nprobe_values: Vec<_> = nprobe_values
                .into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            nprobe_values.sort();

            for nprobe in nprobe_values {
                if nprobe > nlist || nprobe == 0 {
                    continue;
                }

                println!(
                    "Querying IVF-OPQ (nlist={}, m={}, np={})...",
                    nlist, m, nprobe
                );
                let start = Instant::now();
                let (approx_neighbors, _) =
                    query_ivf_opq_index(query_data, &ivf_opq_idx, cli.k, Some(nprobe), true, false);
                let query_time = start.elapsed().as_secs_f64() * 1000.0;

                let recall = calculate_recall(&true_neighbors, &approx_neighbors, cli.k);

                results.push(BenchmarkResult {
                    method: format!("IVF-OPQ-nl{}-m{}-np{}", nlist, m, nprobe),
                    build_time_ms: build_time,
                    query_time_ms: query_time,
                    total_time_ms: build_time + query_time,
                    recall_at_k: recall,
                    mean_dist_err: 0.0,
                });
            }
        }
    }

    print_results_recall_only(
        &format!("{}k cells, {}D", cli.n_cells / 1000, cli.dim),
        &results,
    );
}
