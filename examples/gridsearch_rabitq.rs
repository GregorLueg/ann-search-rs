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
        "Generating synthetic data: {} cells, {} dimensions, {} clusters, {} dist.",
        cli.n_cells.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, cluster_labels): (Mat<f32>, Vec<usize>) = generate_data(&cli);
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

    // RaBitQ exhaustive benchmark
    println!("Building RaBitQ exhaustive index...");
    let start = Instant::now();
    let rabitq_idx =
        build_exhaustive_index_rabitq(data.as_ref(), None, &cli.distance, cli.seed as usize);
    let build_time = start.elapsed().as_secs_f64() * 1000.0;
    let index_size_mb = rabitq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Self-querying RaBitQ exhaustive index...");
    let start = Instant::now();
    let (rabitq_neighbors_self, _) =
        query_exhaustive_index_rabitq_self(data.as_ref(), &rabitq_idx, cli.k, None, false, false);
    let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

    let recall_self = calculate_recall(&true_neighbors_self, &rabitq_neighbors_self, cli.k);
    let rabitq_purity = calculate_cluster_purity(&rabitq_neighbors_self, &cluster_labels);

    results.push(BenchmarkResultPurity {
        method: "ExhaustiveRaBitQ".to_string(),
        build_time_ms: build_time,
        query_time_ms: self_query_time,
        total_time_ms: build_time + self_query_time,
        recall_at_k: recall_self,
        cluster_purity: rabitq_purity,
        index_size_mb,
    });

    println!("-----------------------------");

    print_results_purity(
        &format!(
            "{}k cells, {}D - Exhaustive vs RaBitQ",
            cli.n_cells / 1000,
            cli.dim
        ),
        &results,
    );
}
