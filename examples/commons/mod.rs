// Shared by every gridsearch, and each one uses a different subset of it.
#![allow(dead_code)]
#![allow(unused_imports)]

use ann_search_rs::prelude::{matrix_to_flat, parse_ann_dist, Dist, SimdDistance};
use clap::Parser;
use faer::Mat;
use num_traits::{Float, ToPrimitive};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use thousands::*;

// The generators moved into the library so the benchmark tables are
// reproducible outside this repository. Re-exported here so every gridsearch
// keeps its unqualified names.
pub use ann_search_rs::synthetic::{
    generate_cell_embeddings, generate_clustered_data, generate_clustered_data_high_dim,
    generate_low_rank_rotated_data, parse_data, subsample_with_noise, SyntheticData,
    DEFAULT_COR_STRENGTH, DEFAULT_INTRINSIC_DIM,
};

////////////
// Consts //
////////////

/// Default number of samples
pub const DEFAULT_N_SAMPLES: usize = 150_000;

/// Default number for the querying
pub const DEFAULT_N_QUERY: usize = DEFAULT_N_SAMPLES / 10;

/// Default dimensionality -> typical for single cell
pub const DEFAULT_DIM: usize = 32;

/// Number of default clusters
pub const DEFAULT_N_CLUSTERS: usize = 25;

/// Default number of neighbours
pub const DEFAULT_K: usize = 15;

/// Default random seed
pub const DEFAULT_SEED: u64 = 42;

/// Default distance metric
pub const DEFAULT_DISTANCE: &str = "euclidean";

/// Default data type
pub const DEFAULT_DATA: &str = "gaussian";

////////////
// Parser //
////////////

/// Parsing structure
///
/// ### Fields
///
/// * `n_samples` - Number of samples/samples
/// * `dim` - Number of dimensions to use
/// * `n_clusters` - Number of clusters in the data
/// * `k` - Number of neighbours to search
/// * `seed` - Random seed for reproducibility
/// * `distance` - The distance to use. One of `"euclidean"` or `"cosine"`.
/// * `data` - The data to use. One of `"gaussian"`, `"correlated"`, `"lowrank"`
///   or `"cell"`.
/// * `intrinsic_dim` - True dimensionality for the `"lowrank"` manifold data.
/// * `spectral_decay` - Currently unused (was the quantisation-stress decay exponent).
#[derive(Parser, Clone)]
pub struct Cli {
    #[arg(long, default_value_t = DEFAULT_N_SAMPLES)]
    pub n_samples: usize,

    #[arg(long, default_value_t = DEFAULT_DIM)]
    pub dim: usize,

    #[arg(long, default_value_t = DEFAULT_N_CLUSTERS)]
    pub n_clusters: usize,

    #[arg(long, default_value_t = DEFAULT_K)]
    pub k: usize,

    #[arg(long, default_value_t = DEFAULT_SEED)]
    pub seed: u64,

    #[arg(long, default_value = DEFAULT_DISTANCE)]
    pub distance: String,

    #[arg(long, default_value = DEFAULT_DATA)]
    pub data: String,

    #[arg(long, default_value_t = DEFAULT_INTRINSIC_DIM)]
    pub intrinsic_dim: usize,
}

//////////
// Data //
//////////

////////////////////////////
// Main dispatch function //
////////////////////////////

/// Wrapper function to generate the synthetic data to use
///
/// ### Params
///
/// * `cli` - The Cli structure with the data
///
/// ### Returns
///
/// `(syn data, cluster assignments)`
pub fn generate_data(cli: &Cli) -> (Mat<f32>, Vec<usize>) {
    let data_type = parse_data(&cli.data).unwrap_or_default();
    let res: (Mat<f32>, Vec<usize>) = match data_type {
        SyntheticData::GaussianNoise => {
            println!(">>> Using simple Gaussian cluster data. <<<");
            generate_clustered_data(cli.n_samples, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!(">>> Using data with subspace structure and correlated features. <<<");
            generate_clustered_data_high_dim(
                cli.n_samples,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
        SyntheticData::LowRank => {
            println!(">>> Using data that simulating manifold hypothesis. <<<");
            generate_low_rank_rotated_data(
                cli.n_samples,
                cli.dim,
                cli.intrinsic_dim,
                cli.n_clusters,
                cli.seed,
            )
        }
        SyntheticData::CellEmbedding => {
            println!(">>> Using foundation-model cell embedding data. <<<");
            generate_cell_embeddings(cli.n_samples, cli.dim, cli.n_clusters, cli.seed)
        }
    };

    res
}

////////////////
// Benchmarks //
////////////////

/// BenchmarkResult
pub struct BenchmarkResultSize {
    /// Name of the method
    pub method: String,
    /// The build time of the index in ms
    pub build_time_ms: f64,
    /// The query time of the index in ms
    pub query_time_ms: f64,
    ///  Total time the index build & query takes in ms
    pub total_time_ms: f64,
    /// Recall@k neighbours against ground truth. Overlap in top k neighbours
    /// for given k
    pub recall_at_k: f64,
    /// Mean distance ratio
    pub mean_dist_rat: f64,
    /// Size of the index
    pub index_size_mb: f64,
}

/////////////
// Helpers //
/////////////

/// Calculate Recall@k
///
/// ### Params
///
/// * `true_neighbors` - Slice of true neighbours
/// * `approx_neighbors` - Slice of the approximate neighbours
/// * `k` - Number of selected k
///
/// ### Returns
///
/// The Recall@k
pub fn calculate_recall(
    true_neighbors: &[Vec<usize>],
    approx_neighbors: &[Vec<usize>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;

    for (true_nn, approx_nn) in true_neighbors.iter().zip(approx_neighbors.iter()) {
        let true_set: FxHashSet<_> = true_nn.iter().take(k).collect();
        let approx_set: FxHashSet<_> = approx_nn.iter().take(k).collect();

        let matches = approx_set.intersection(&true_set).count();

        total_recall += matches as f64 / k as f64;
    }

    total_recall / true_neighbors.len() as f64
}

/// Recompute exact `f32` distances to the neighbours an index returned
///
/// Quantised indices report the *codec's estimate* of the distance, not the
/// distance itself. Feeding those straight into
/// [`calculate_mean_distance_ratio`] conflates two different errors: the
/// retrieved neighbours being worse than the true ones, and the reported
/// number being a biased estimate of how far away they are. The second can
/// push the ratio below 1.0, which reads as "better than optimal" and is
/// nothing of the sort. This recomputes the distances in full precision from
/// the original vectors, so the ratio measures retrieval quality alone and is
/// directly comparable to an unquantised index's.
///
/// The distance definitions match the exhaustive index exactly: squared
/// Euclidean, `1 - cosine similarity`, or L1.
///
/// ### Params
///
/// * `data` - The indexed vectors, samples x features
/// * `queries` - The query vectors, samples x features. Pass `data` again for
///   a self-query
/// * `neighbours` - Indices returned by the index, one vec per query, holding
///   row numbers into `data`
/// * `distance` - Metric name, as handed to the index builder
///
/// ### Returns
///
/// Exact distances in the same layout as `neighbours`. Any index outside
/// `data`'s row range yields `f32::INFINITY` rather than panicking, so a
/// broken index shows up as a bad ratio instead of a crash.
pub fn exact_distances(
    data: &Mat<f32>,
    queries: &Mat<f32>,
    neighbours: &[Vec<usize>],
    distance: &str,
) -> Vec<Vec<f32>> {
    let metric = parse_ann_dist(distance).unwrap_or_default();

    // faer matrices are column-major, so rows are strided and the SIMD kernels
    // cannot see them. Flatten once here rather than per distance.
    let (data_flat, n, dim) = matrix_to_flat(data.as_ref());
    let (query_flat, n_queries, query_dim) = matrix_to_flat(queries.as_ref());
    assert_eq!(dim, query_dim, "query and data dimensionality differ");
    assert_eq!(
        n_queries,
        neighbours.len(),
        "one neighbour list per query expected"
    );

    // Only the cosine path needs norms, and it needs them for every candidate.
    let data_norms: Vec<f32> = if metric == Dist::Cosine {
        (0..n)
            .into_par_iter()
            .map(|i| f32::calculate_l2_norm(&data_flat[i * dim..(i + 1) * dim]))
            .collect()
    } else {
        Vec::new()
    };

    neighbours
        .par_iter()
        .enumerate()
        .map(|(q, ids)| {
            let query = &query_flat[q * dim..(q + 1) * dim];
            let query_norm = if metric == Dist::Cosine {
                f32::calculate_l2_norm(query)
            } else {
                1.0
            };

            ids.iter()
                .map(|&id| {
                    if id >= n {
                        return f32::INFINITY;
                    }
                    let vec = &data_flat[id * dim..(id + 1) * dim];
                    match metric {
                        Dist::SquaredEuclidean => f32::euclidean_simd(vec, query),
                        Dist::Manhattan => f32::manhattan_simd(vec, query),
                        Dist::Cosine => {
                            let denom = query_norm * data_norms[id];
                            if denom > 0.0 {
                                1.0 - f32::dot_simd(vec, query) / denom
                            } else {
                                1.0
                            }
                        }
                    }
                })
                .collect()
        })
        .collect()
}

/// Calculate mean distance ratio across queries
///
/// For each query, computes the ratio of the sum of approximate distances
/// to the sum of true distances across the top-k neighbours. A ratio of
/// 1.0 indicates perfect results; values above 1.0 indicate how much
/// worse the approximate distances are on average (e.g. 1.05 means 5%
/// worse than optimal).
///
/// This metric is stable across distance metrics and dimensionalities
/// because summing over k neighbours avoids the division-by-near-zero
/// instability that plagues per-pair relative error, particularly with
/// cosine distance. Queries where the true distance sum is negligible
/// (< 1e-12) are excluded.
///
/// ### Params
///
/// * `true_dist` - Slice of true distances to the neighbours (one vec per
///   query)
/// * `approx_dist` - Slice of approximate distances to the neighbours (one vec
///   per query)
/// * `k` - Number of neighbours to consider per query
///
/// ### Returns
///
/// The mean distance ratio (1.0 = perfect, >1.0 = proportionally worse).
/// Returns `NaN` if no queries have a non-negligible true distance sum.
pub fn calculate_mean_distance_ratio<T>(
    true_dist: &[Vec<T>],
    approx_dist: &[Vec<T>],
    k: usize,
) -> f64
where
    T: Float + ToPrimitive,
{
    let mut total_ratio = 0.0;
    let mut count = 0usize;
    for (td, ad) in true_dist.iter().zip(approx_dist.iter()) {
        let n = k.min(td.len()).min(ad.len());
        let sum_true: f64 = td[..n].iter().map(|v| v.to_f64().unwrap()).sum();
        let sum_approx: f64 = ad[..n].iter().map(|v| v.to_f64().unwrap()).sum();
        if sum_true > 1e-12 {
            total_ratio += sum_approx / sum_true;
            count += 1;
        }
    }
    total_ratio / count as f64
}

////////////
// Prints //
////////////

fn format_with_underscores(value: f64) -> String {
    let formatted = format!("{:.2}", value);
    let parts: Vec<&str> = formatted.split('.').collect();
    let int_part = parts[0].parse::<i64>().unwrap().separate_with_underscores();
    format!("{}.{}", int_part, parts[1])
}

/// Helper to print results to console
///
/// This version also returns the size of the index
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results_size(config: &str, results: &[BenchmarkResultSize]) {
    println!("\n{:=>131}", "");
    println!("Benchmark: {}", config);
    println!("{:=>131}", "");
    println!(
        "{:<50} {:>12} {:>12} {:>12} {:>12} {:>15} {:>12}",
        "Method",
        "Build (ms)",
        "Query (ms)",
        "Total (ms)",
        "Recall@k",
        "Mean dist ratio",
        "Size (MB)"
    );
    println!("{:->131}", "");
    for result in results {
        println!(
            "{:<50} {:>12} {:>12} {:>12} {:>12.4} {:>15.4} {:>12.2}",
            result.method,
            format_with_underscores(result.build_time_ms),
            format_with_underscores(result.query_time_ms),
            format_with_underscores(result.total_time_ms),
            result.recall_at_k,
            result.mean_dist_rat,
            result.index_size_mb
        );
    }
    println!("{:->131}\n", "");
}
