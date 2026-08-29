mod commons;

use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

/// Wraps the shared `Cli` and adds an optional `--knn-k` override so the
/// standalone NN-Descent build can be run at the same `k` NSG uses
/// internally (default `64`) for like-for-like timing comparisons.
///
/// Each build reports three rows: `(query)` for held-out queries, `(self)` for
/// the full self-kNN via beam search, and `(extract)` for the descent graph
/// handed back as-is. The last one costs one pass and is the honest measure of
/// how good the built graph is before any search refines it, so the gap between
/// `(self)` and `(extract)` is exactly what the beam search buys.
#[derive(Parser, Clone)]
struct Args {
    #[command(flatten)]
    cli: Cli,
    /// Override `k` for the NN-Descent build. Defaults to the library's own
    /// default (currently `30`) when not passed.
    #[arg(long)]
    knn_k: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let cli = args.cli.clone();
    let knn_k = args.knn_k;

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

    let build_params = [
        (Some(12), 0.0, vec![None]),
        (Some(24), 0.0, vec![None]),
        (None, 0.0, vec![Some(75), Some(100), None]),
        (None, 0.25, vec![None]),
        (None, 0.5, vec![None]),
        (None, 1.0, vec![None]),
    ];

    for (n_trees, diversify_prob, ef_search_values) in build_params {
        let n_trees_str = n_trees
            .map(|i| i.to_string())
            .unwrap_or_else(|| ":auto".to_string());

        let knn_k_str = knn_k
            .map(|k| k.to_string())
            .unwrap_or_else(|| ":auto".to_string());
        println!(
            "Building NNDescent index (n_trees={}, diversify={}, knn_k={})...",
            n_trees_str, diversify_prob, knn_k_str
        );
        let start = Instant::now();
        let nndescent_idx = build_nndescent_index(
            data.as_ref(),
            &cli.distance,
            0.001,
            diversify_prob,
            knn_k,
            None,
            None,
            n_trees,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let build_time = start.elapsed().as_secs_f64() * 1000.0;

        let index_size_mb = nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        // Query benchmarks
        for ef_search in &ef_search_values {
            let ef_search_str = ef_search
                .map(|i| i.to_string())
                .unwrap_or_else(|| ":auto".to_string());

            println!("Querying NNDescent index (ef_search={})...", ef_search_str);
            let start = Instant::now();
            let (approx_neighbors, approx_distances) = query_nndescent_index(
                query_data.as_ref(),
                &nndescent_idx,
                cli.k,
                *ef_search,
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
                    "NNDescent-k{}-nt{}-s{}-dp{} (query)",
                    knn_k_str, n_trees_str, ef_search_str, diversify_prob
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
        println!("Self-querying NNDescent index...");
        let start = Instant::now();
        let (approx_neighbors_self, approx_distances_self) =
            query_nndescent_self(&nndescent_idx, cli.k, None, true, false).unwrap();
        let self_query_time = start.elapsed().as_secs_f64() * 1000.0;

        let recall_self = calculate_recall(&true_neighbors_self, &approx_neighbors_self, cli.k);
        let dist_error_self = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            approx_distances_self.as_ref().unwrap(),
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!(
                "NNDescent-k{}-nt{}-dp{} (self)",
                knn_k_str, n_trees_str, diversify_prob
            ),
            build_time_ms: build_time,
            query_time_ms: self_query_time,
            total_time_ms: build_time + self_query_time,
            recall_at_k: recall_self,
            mean_dist_rat: dist_error_self,
            index_size_mb,
        });

        // Graph extraction: the descent's own output, no beam search. Against
        // the same ground truth as the self-query row, the gap between the two
        // is what the beam search is buying on top of the graph, and the
        // extract row on its own is the graph quality.
        if nndescent_idx.k + 1 < cli.k {
            println!(
                "  note: index built with k={}, so extraction cannot fill k={}",
                nndescent_idx.k, cli.k
            );
        }

        println!("Extracting NNDescent kNN graph (no beam search)...");
        let start = Instant::now();
        let (mut extract_neighbors, extract_distances) =
            extract_nndescent_knn(&nndescent_idx, Some(cli.k.saturating_sub(1)), true).unwrap();
        let extract_time = start.elapsed().as_secs_f64() * 1000.0;

        // The self-query rows and the ground truth both count a point as its
        // own nearest neighbour at distance zero. A kNN graph does not store
        // that trivial edge, so it is added back here rather than letting the
        // extract row lose a fixed 1/k against every other row and every other
        // gridsearch example.
        let mut extract_distances = extract_distances.unwrap();
        for (i, (ids, dists)) in extract_neighbors
            .iter_mut()
            .zip(extract_distances.iter_mut())
            .enumerate()
        {
            ids.insert(0, i);
            dists.insert(0, 0.0);
        }

        let recall_extract = calculate_recall(&true_neighbors_self, &extract_neighbors, cli.k);
        let dist_error_extract = calculate_mean_distance_ratio(
            true_distances_self.as_ref().unwrap(),
            &extract_distances,
            cli.k,
        );

        results.push(BenchmarkResultSize {
            method: format!(
                "NNDescent-k{}-nt{}-dp{} (extract)",
                knn_k_str, n_trees_str, diversify_prob
            ),
            build_time_ms: build_time,
            query_time_ms: extract_time,
            total_time_ms: build_time + extract_time,
            recall_at_k: recall_extract,
            mean_dist_rat: dist_error_extract,
            index_size_mb,
        });
    }

    print_results_size(
        &format!("{}k samples, {}D", cli.n_samples / 1000, cli.dim),
        &results,
    );
}
