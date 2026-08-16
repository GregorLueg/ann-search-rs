mod commons;

use ann_search_rs::gpu::clustered_nndescent_gpu::ClusteredBuildParams;
use ann_search_rs::gpu::nndescent_gpu::KnnGraphGpu;
use ann_search_rs::utils::nndescent_utils::SENTINEL_PID;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

// Modified bench compared to the others as other things are being tested here

////////////
// Consts //
////////////

/// Cluster counts swept over the batched build.
const CLUSTER_SWEEP: &[usize] = &[1, 2, 4, 8, 16];

/// Fraction of the dataset used to train the batching centroids.
const SAMPLE_FRAC: f32 = 0.1;

/// Clusters each point joins. Overlap is what stitches the batch boundaries
/// back together, so 1 is the pessimistic case rather than the sane one.
const N_ASSIGN: usize = 2;

/// Fill ratio below which the graph is treated as broken rather than merely
/// sparse.
const FILL_WARN_THRESHOLD: f64 = 0.99;

////////////////////
// ClusterResults //
////////////////////

/// One row of the sweep table.
struct ClusterResult {
    /// Human-readable label for the build
    method: String,
    /// Cluster count the build ran at, `None` for the unbatched baseline
    n_clusters: Option<usize>,
    /// Build wall-clock in milliseconds
    build_time_ms: f64,
    /// Recall@k of the self-kNN graph against exhaustive ground truth
    recall_at_k: f64,
    /// Non-sentinel entries in the flat graph
    filled: usize,
    /// Entries the graph would hold if every slot were populated (`n * k`)
    expected: usize,
}

/////////////
// Helpers //
/////////////

/// Strip self-matches from an exhaustive self-query and trim to `k`.
///
/// The GPU builders drop the `i == j` edge, so the ground truth has to as
/// well. Filtering rather than dropping the first column keeps this correct
/// when ties push the self-match off position zero.
///
/// ### Params
///
/// * `neighbours` - Self-query result, one row per sample
/// * `k` - Neighbours to retain per row
///
/// ### Returns
///
/// The neighbour rows with self removed, truncated to `k`.
fn drop_self_matches(neighbours: &[Vec<usize>], k: usize) -> Vec<Vec<usize>> {
    neighbours
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().copied().filter(|&j| j != i).take(k).collect())
        .collect()
}

/// Unpack the flat kNN graph into per-node neighbour rows, dropping the
/// unfilled slots.
///
/// ### Params
///
/// * `graph` - The GPU-built kNN graph
///
/// ### Returns
///
/// One neighbour vector per node, sentinel slots removed.
fn graph_to_rows(graph: &KnnGraphGpu<f32>) -> Vec<Vec<usize>> {
    (0..graph.n)
        .map(|i| {
            graph.knn_graph[i * graph.k..(i + 1) * graph.k]
                .iter()
                .map(|&(pid, _)| pid)
                .filter(|&pid| pid != SENTINEL_PID)
                .collect()
        })
        .collect()
}

/// Count the populated entries of the flat graph.
///
/// ### Params
///
/// * `graph` - The GPU-built kNN graph
///
/// ### Returns
///
/// The number of non-sentinel entries, which should sit at `n * k`.
fn count_filled(graph: &KnnGraphGpu<f32>) -> usize {
    graph
        .knn_graph
        .iter()
        .filter(|&&(pid, _)| pid != SENTINEL_PID)
        .count()
}

/// Validate the graph before its timing is allowed to mean anything.
///
/// Every launch in this crate is `launch_unchecked`. A dispatch that busts a
/// device limit does no work, returns zeros and reports **no error**: the
/// panic lands on a cubecl background thread. A batched build that silently
/// did nothing therefore looks like a spectacular speed-up, so the fill count
/// is checked and printed first and the timing is only meaningful once it
/// lands near `n * k`.
///
/// ### Params
///
/// * `label` - Label of the build being checked
/// * `graph` - The GPU-built kNN graph
///
/// ### Returns
///
/// `(filled, expected)` entry counts.
fn check_graph(label: &str, graph: &KnnGraphGpu<f32>) -> (usize, usize) {
    let filled = count_filled(graph);
    let expected = graph.n * graph.k;
    let ratio = filled as f64 / expected as f64;

    println!(
        "  checksum: {} / {} entries filled ({:.2}%), converged={}",
        filled.separate_with_underscores(),
        expected.separate_with_underscores(),
        ratio * 100.0,
        graph.converged
    );

    if ratio < FILL_WARN_THRESHOLD {
        println!(
            "  [WARNING] {}: only {:.2}% of the graph is populated. A kernel that busts a \
             device limit does no work and reports no error, so treat the timing below as \
             meaningless until this is near 100%.",
            label,
            ratio * 100.0
        );
    }

    (filled, expected)
}

/// Print the sweep table.
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Rows to print
fn print_cluster_results(config: &str, results: &[ClusterResult]) {
    println!("\n{:=>115}", "");
    println!("Benchmark: {}", config);
    println!("{:=>115}", "");
    println!(
        "{:<32} {:>10} {:>14} {:>12} {:>28} {:>10}",
        "Method", "Clusters", "Build (ms)", "Recall@k", "Filled", "Fill (%)"
    );
    println!("{:->115}", "");
    for result in results {
        let clusters = result
            .n_clusters
            .map(|c| c.to_string())
            .unwrap_or("-".to_string());
        println!(
            "{:<32} {:>10} {:>14.2} {:>12.4} {:>28} {:>10.2}",
            result.method,
            clusters,
            result.build_time_ms,
            result.recall_at_k,
            format!(
                "{} / {}",
                result.filled.separate_with_underscores(),
                result.expected.separate_with_underscores()
            ),
            result.filled as f64 / result.expected as f64 * 100.0
        );
    }
    println!("{:->115}\n", "");
}

///////////
// Bench //
///////////

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
    let n = data.nrows();
    let mut results: Vec<ClusterResult> = Vec::new();

    // Ground truth: exhaustive self-kNN, self-matches stripped.
    println!("Building exhaustive index...");
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);

    println!("Self-querying exhaustive index (ground truth)...");
    let start = Instant::now();
    let (true_neighbors_self, _) =
        query_exhaustive_self(&exhaustive_idx, cli.k + 1, false, false).unwrap();
    println!(
        "Ground truth in {:.2} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let true_neighbors_self = drop_self_matches(&true_neighbors_self, cli.k);

    println!("-----------------------------");

    let device: cubecl::wgpu::WgpuDevice = Default::default();

    // Baseline: the unbatched builder, so the table stands on its own.
    println!("Building unbatched GPU kNN graph (k={})...", cli.k);
    let start = Instant::now();
    let graph = build_knn_graph_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        Some(cli.k),
        None,
        None,
        None,
        None,
        None,
        None,
        cli.seed as usize,
        true,
        device.clone(),
    )
    .unwrap();
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (filled, expected) = check_graph("unbatched", &graph);
    let recall = calculate_recall(&true_neighbors_self, &graph_to_rows(&graph), cli.k);
    println!(
        "Built in {:.2} ms, Recall@{} {:.4}",
        build_time_ms, cli.k, recall
    );

    results.push(ClusterResult {
        method: "NNDescent-GPU (unbatched)".to_string(),
        n_clusters: None,
        build_time_ms,
        recall_at_k: recall,
        filled,
        expected,
    });

    println!("-----------------------------");

    for &n_clusters in CLUSTER_SWEEP {
        if n_clusters > n {
            continue;
        }

        println!(
            "Building clustered GPU kNN graph (k={}, n_clusters={}, n_assign={})...",
            cli.k, n_clusters, N_ASSIGN
        );

        let cluster_params =
            ClusteredBuildParams::new(Some(n_clusters), SAMPLE_FRAC, N_ASSIGN, None);

        let start = Instant::now();
        let graph = build_clustered_knn_graph_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
            data.as_ref(),
            &cli.distance,
            Some(cli.k),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(cluster_params),
            cli.seed as usize,
            true,
            device.clone(),
        )
        .unwrap();
        let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let label = format!("clustered-c{}", n_clusters);
        let (filled, expected) = check_graph(&label, &graph);
        let recall = calculate_recall(&true_neighbors_self, &graph_to_rows(&graph), cli.k);
        println!(
            "Built in {:.2} ms, Recall@{} {:.4}",
            build_time_ms, cli.k, recall
        );

        results.push(ClusterResult {
            method: format!("NNDescent-GPU (clustered, c={})", n_clusters),
            n_clusters: Some(n_clusters),
            build_time_ms,
            recall_at_k: recall,
            filled,
            expected,
        });
    }

    print_cluster_results(
        &format!(
            "{}k samples, {}D, k={} (clustered vs unbatched GPU NN-Descent)",
            cli.n_samples / 1000,
            cli.dim,
            cli.k
        ),
        &results,
    );
}
