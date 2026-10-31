mod commons;

use ann_search_rs::gpu::nndescent_gpu::NNDescentGpuKnnParams;
use ann_search_rs::prelude::{KnnGraph, NNDescentKnnParams, KNN_GRAPH_EMPTY};
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::time::Instant;
use thousands::*;

////////////
// Consts //
////////////

/// Internal working degree, as a multiple of `k`, swept over the build.
const BUILD_K_MULTIPLIERS: &[usize] = &[1, 2, 3];

/// 2-hop refinement sweeps run after the descent converges.
const REFINE_SWEEPS: &[usize] = &[0, 1, 2];

/// NN-Descent iteration cap for every build in the sweep.
const MAX_ITERS: usize = 20;

/// Convergence threshold on the fraction of edges updated per iteration.
const DELTA: f32 = 0.0005;

/// Fill ratio below which the graph is treated as broken rather than merely
/// sparse.
const FILL_WARN_THRESHOLD: f64 = 0.99;

/////////////
// Helpers //
/////////////

/// Validate the graph before its timing is allowed to mean anything.
///
/// Every launch in this crate is `launch_unchecked`. A dispatch that busts a
/// device limit does no work, returns zeros and reports **no error**: the panic
/// lands on a cubecl background thread. A build that silently did nothing
/// therefore looks like a spectacular speed-up, so the fill count is checked
/// before the row is recorded.
///
/// This prints above the first `=====` ruler, which is where
/// `examples/fill_benchmarks.sh` starts capturing, so it stays out of the
/// generated docs and in front of whoever is running the example.
///
/// ### Params
///
/// * `label` - Label of the build being checked
/// * `graph` - The kNN graph
fn check_graph(label: &str, graph: &KnnGraph<f32>) {
    let filled = graph
        .indices
        .iter()
        .filter(|&&pid| pid != KNN_GRAPH_EMPTY)
        .count();
    let expected = graph.indices.len();
    let ratio = filled as f64 / expected as f64;

    println!(
        "  checksum: {} / {} entries filled ({:.2}%)",
        filled.separate_with_underscores(),
        expected.separate_with_underscores(),
        ratio * 100.0,
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
}

/// Unpack a flat kNN graph into the nested rows the scoring helpers take.
///
/// ### Params
///
/// * `graph` - The kNN graph, built with distances
///
/// ### Returns
///
/// Tuple of (neighbour rows, distance rows).
fn unpack(graph: &KnnGraph<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let dists = graph.distances.as_ref().expect("built with return_dist");
    let ids = (0..graph.n)
        .map(|i| graph.row(i).iter().map(|&p| p as usize).collect())
        .collect();
    let ds = dists.chunks_exact(graph.k).map(|r| r.to_vec()).collect();
    (ids, ds)
}

/// Host size of a flat kNN graph in MB.
///
/// ### Params
///
/// * `graph` - The kNN graph
///
/// ### Returns
///
/// Size of the ids plus distances in MB.
fn graph_mb(graph: &KnnGraph<f32>) -> f64 {
    let bytes = graph.indices.len() * std::mem::size_of::<u32>()
        + graph.distances.as_ref().map_or(0, |d| d.len()) * std::mem::size_of::<f32>();
    bytes as f64 / (1024.0 * 1024.0)
}

/// Score a kNN graph against the ground truth into a result row.
///
/// The extraction column is zero: the graph comes back flat and final, there
/// is no separate query step.
///
/// ### Params
///
/// * `method` - Row label
/// * `build_ms` - Build time in ms
/// * `graph` - The kNN graph, self-edge included
/// * `true_neighbors` - Exhaustive ground-truth ids
/// * `true_distances` - Exhaustive ground-truth distances
/// * `k` - Neighbours scored per row
///
/// ### Returns
///
/// The benchmark row.
fn score(
    method: String,
    build_ms: f64,
    graph: &KnnGraph<f32>,
    true_neighbors: &[Vec<usize>],
    true_distances: &[Vec<f32>],
    k: usize,
) -> BenchmarkResultSize {
    let (ids, dists) = unpack(graph);
    BenchmarkResultSize {
        method,
        build_time_ms: build_ms,
        query_time_ms: 0.0,
        total_time_ms: build_ms,
        recall_at_k: calculate_recall(true_neighbors, &ids, k),
        mean_dist_rat: calculate_mean_distance_ratio(true_distances, &dists, k),
        median_dist_rat: calculate_median_distance_ratio(true_distances, &dists, k),
        index_size_mb: graph_mb(graph),
    }
}

//////////
// Main //
//////////

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
    let mut results = Vec::new();

    let device: cubecl::wgpu::WgpuDevice = Default::default();

    // Ground truth: GPU exhaustive self-query
    println!("Building GPU exhaustive index...");
    let start = Instant::now();
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32>(
        data.as_ref(),
        &cli.distance,
        device.clone().into(),
        Verbosity::Quiet,
    )
    .unwrap();
    let ex_build = start.elapsed().as_secs_f64() * 1000.0;

    println!("Self-querying GPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index_gpu_self(&gpu_exhaustive_idx, cli.k, true, Verbosity::Quiet)
            .unwrap();
    let ex_query = start.elapsed().as_secs_f64() * 1000.0;
    let true_distances = true_distances.unwrap();

    results.push(BenchmarkResultSize {
        method: "GPU-Exhaustive (ground truth)".to_string(),
        build_time_ms: ex_build,
        query_time_ms: ex_query,
        total_time_ms: ex_build + ex_query,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        median_dist_rat: 1.0,
        index_size_mb: gpu_exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0),
    });

    println!("-----------------------------");

    // `include_self` puts the trivial self-edge first in every row, so the
    // graphs are scored like for like against a ground truth that counts a
    // point as its own nearest neighbour. Scoring takes the first `k` columns.

    // CPU NN-Descent baseline through the graph-only build.
    println!("Building CPU NNDescent kNN graph...");
    let label = format!("CPU-NNDescent (k={})", cli.k);
    let params = NNDescentKnnParams {
        k: cli.k,
        include_self: true,
        ..Default::default()
    };
    let start = Instant::now();
    let cpu_graph = build_knn_graph_nndescent(
        data.as_ref(),
        &cli.distance,
        Some(params),
        cli.seed as usize,
        Verbosity::Quiet,
    )
    .unwrap();
    let cpu_build = start.elapsed().as_secs_f64() * 1000.0;

    check_graph(&label, &cpu_graph);
    results.push(score(
        label,
        cpu_build,
        &cpu_graph,
        &true_neighbors,
        &true_distances,
        cli.k,
    ));
    drop(cpu_graph);

    println!("-----------------------------");

    // GPU kNN graph: build_k multiplier x refinement sweeps, through the
    // graph-only build. No CAGRA kernels run; a dataset past the per-binding
    // limit is split into clusters, one that fits takes the unbatched path.
    for &bk_mult in BUILD_K_MULTIPLIERS {
        for &refine in REFINE_SWEEPS {
            let build_k = cli.k * bk_mult;
            let label = format!("GPU-kNN bk={}x refine={}", bk_mult, refine);

            println!(
                "Building GPU kNN graph (build_k={}, refine={})...",
                build_k, refine
            );

            let params = NNDescentGpuKnnParams::new(
                cli.k,
                Some(build_k),
                Some(MAX_ITERS),
                None,
                Some(DELTA),
                None,
                Some(refine),
                true,
                true,
            );
            let start = Instant::now();
            let graph = build_knn_graph_nndescent_gpu::<f32>(
                data.as_ref(),
                &cli.distance,
                Some(params),
                None,
                cli.seed as usize,
                Verbosity::Quiet,
                device.clone().into(),
            )
            .unwrap();
            let gpu_build = start.elapsed().as_secs_f64() * 1000.0;

            check_graph(&label, &graph);
            results.push(score(
                label,
                gpu_build,
                &graph,
                &true_neighbors,
                &true_distances,
                cli.k,
            ));

            println!("-----------------------------");
        }
    }

    print_results_size(
        &format!(
            "{}k samples, {}D kNN graph generation (build_k x refinement)",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results,
    );
}
