mod commons;

use ann_search_rs::gpu::nndescent_gpu::KnnGraphGpu;
use ann_search_rs::utils::nndescent_utils::SENTINEL_PID;
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
/// * `graph` - The GPU-built kNN graph
fn check_graph(label: &str, graph: &KnnGraphGpu<f32>) {
    let filled = graph
        .knn_graph
        .iter()
        .filter(|&&(pid, _)| pid != SENTINEL_PID)
        .count();
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
    let gpu_exhaustive_idx = build_exhaustive_index_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
        data.as_ref(),
        &cli.distance,
        device.clone(),
    )
    .unwrap();
    let ex_build = start.elapsed().as_secs_f64() * 1000.0;

    println!("Self-querying GPU exhaustive (ground truth)...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index_gpu_self(&gpu_exhaustive_idx, cli.k, true, false).unwrap();
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

    // CPU NNDescent baseline, scored through its own beam search so the row is
    // directly comparable to the ground truth above.
    println!("Building CPU NNDescent...");
    let start = Instant::now();
    let cpu_nndescent_idx = build_nndescent_index(
        data.as_ref(),
        &cli.distance,
        0.001,
        0.0,
        None,
        None,
        None,
        None,
        cli.seed as usize,
        false,
    )
    .unwrap();
    let cpu_build = start.elapsed().as_secs_f64() * 1000.0;
    let cpu_size = cpu_nndescent_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Extracting CPU NNDescent kNN graph...");
    let start = Instant::now();
    let (cpu_neighbors, cpu_distances) =
        query_nndescent_self(&cpu_nndescent_idx, cli.k, None, true, false).unwrap();
    let cpu_extract = start.elapsed().as_secs_f64() * 1000.0;

    results.push(BenchmarkResultSize {
        method: format!("CPU-NNDescent (k={})", cli.k),
        build_time_ms: cpu_build,
        query_time_ms: cpu_extract,
        total_time_ms: cpu_build + cpu_extract,
        recall_at_k: calculate_recall(&true_neighbors, &cpu_neighbors, cli.k),
        mean_dist_rat: calculate_mean_distance_ratio(
            &true_distances,
            cpu_distances.as_ref().unwrap(),
            cli.k,
        ),
        median_dist_rat: calculate_median_distance_ratio(
            &true_distances,
            cpu_distances.as_ref().unwrap(),
            cli.k,
        ),
        index_size_mb: cpu_size,
    });

    println!("-----------------------------");

    // GPU kNN graph: build_k multiplier x refinement sweeps. This is the slim
    // `build_knn_graph_gpu` path, so no CAGRA kernels run and there is no
    // navigational graph to beam-search. The extracted graph is the product.
    for &bk_mult in BUILD_K_MULTIPLIERS {
        for &refine in REFINE_SWEEPS {
            let build_k = cli.k * bk_mult;
            let label = format!("GPU-kNN bk={}x refine={}", bk_mult, refine);

            println!(
                "Building GPU kNN graph (build_k={}, refine={})...",
                build_k, refine
            );

            let start = Instant::now();
            let graph = build_knn_graph_gpu::<f32, cubecl::wgpu::WgpuRuntime>(
                data.as_ref(),
                &cli.distance,
                Some(cli.k),
                Some(build_k),
                Some(MAX_ITERS),
                None,
                Some(DELTA),
                None,
                Some(refine),
                cli.seed as usize,
                false,
                device.clone(),
            )
            .unwrap();
            let gpu_build = start.elapsed().as_secs_f64() * 1000.0;

            check_graph(&label, &graph);

            // `include_self` puts the trivial self-edge back, so the row is
            // scored like for like against a ground truth that counts a point
            // as its own nearest neighbour.
            println!("  Extracting kNN graph...");
            let start = Instant::now();
            let (gpu_neighbors, gpu_distances) =
                extract_knn_graph_gpu(&graph, Some(cli.k), true, true).unwrap();
            let gpu_extract = start.elapsed().as_secs_f64() * 1000.0;
            let gpu_distances = gpu_distances.unwrap();

            results.push(BenchmarkResultSize {
                method: label,
                build_time_ms: gpu_build,
                query_time_ms: gpu_extract,
                total_time_ms: gpu_build + gpu_extract,
                recall_at_k: calculate_recall(&true_neighbors, &gpu_neighbors, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    &true_distances,
                    &gpu_distances,
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    &true_distances,
                    &gpu_distances,
                    cli.k,
                ),
                index_size_mb: graph.memory_usage_bytes() as f64 / (1024.0 * 1024.0),
            });

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
