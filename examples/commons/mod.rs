#![allow(dead_code)]

use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rustc_hash::FxHashSet;

////////////
// Consts //
////////////

pub const DEFAULT_N_CELLS: usize = 150_000;
pub const DEFAULT_DIM: usize = 32;
pub const DEFAULT_N_CLUSTERS: usize = 20;
pub const DEFAULT_K: usize = 15;
pub const DEFAULT_SEED: u64 = 10101;
pub const DEFAULT_DISTANCE: &str = "euclidean";

/////////////
// Helpers //
/////////////

/// Generate synthetic single-cell-like data with cluster structure
///
/// Creates data with multiple Gaussian clusters to simulate clusters, cell
/// types in the data
///
/// ### Params
///
/// * `n_samples` - Number of cells (samples)
/// * `dim` - Embedding dimensionality
/// * `n_clusters` - Number of distinct clusters
/// * `cluster_std` - Standard deviation within clusters
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_clustered_data<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    // variable cluster sizes and std deviations
    let mut centres = Vec::with_capacity(n_clusters);
    let mut cluster_stds = Vec::new();

    for _ in 0..n_clusters {
        let centre: Vec<f64> = (0..dim).map(|_| rng.random_range(-7.5..7.5)).collect();
        centres.push(centre);
        cluster_stds.push(rng.random_range(0.5..2.5));
    }

    // assign samples with variable cluster sizes
    // with some clusters bigger than others
    let mut cluster_assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n_in_cluster = ((n_samples as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }

    // fill remaining
    while cluster_assignments.len() < n_samples {
        cluster_assignments.push(rng.random_range(0..n_clusters));
    }
    cluster_assignments.shuffle(&mut rng);
    cluster_assignments.truncate(n_samples);

    // generate with variable noise
    for (i, &cluster_idx) in cluster_assignments.iter().enumerate() {
        let centre = &centres[cluster_idx];
        let std = cluster_stds[cluster_idx];

        for j in 0..dim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data[(i, j)] = T::from_f64(centre[j] + noise * std).unwrap();
        }
    }

    data
}

/// Generate synthetic single-cell-like data with cluster structure
///
/// Designed to generate synthetic data for higher dimensionality
///
/// ### Params
///
/// * `n_samples` - Number of cells (samples)
/// * `dim` - Embedding dimensionality
/// * `n_clusters` - Number of distinct clusters
/// * `cluster_std` - Standard deviation within clusters
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_clustered_data_high_dim<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    // scale centre range with dimension to maintain separation
    let scale = (dim as f64).sqrt() * 2.0;

    // generate well-separated centres
    let mut centres = Vec::with_capacity(n_clusters);
    let min_separation = scale * 0.8;

    for _ in 0..n_clusters {
        let centre = loop {
            let candidate: Vec<f64> = (0..dim).map(|_| rng.random_range(-scale..scale)).collect();

            // ensure minimum distance from existing centres
            let too_close = centres.iter().any(|existing: &Vec<f64>| {
                let dist_sq: f64 = candidate
                    .iter()
                    .zip(existing.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                dist_sq < min_separation.powi(2)
            });

            if !too_close {
                break candidate;
            }
        };
        centres.push(centre);
    }

    // subspace structure: each cluster varies mainly in a subset of dimensions
    let active_dims_per_cluster = (dim / 2).max(3);
    let cluster_active_dims: Vec<Vec<usize>> = centres
        .iter()
        .map(|_| {
            let mut dims: Vec<usize> = (0..dim).collect();
            dims.shuffle(&mut rng);
            dims.truncate(active_dims_per_cluster);
            dims
        })
        .collect();

    let mut cluster_stds = Vec::new();
    for _ in 0..n_clusters {
        cluster_stds.push(rng.random_range(0.3..1.0) * scale / 10.0);
    }

    // assign samples
    let mut cluster_assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n_in_cluster = ((n_samples as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }

    while cluster_assignments.len() < n_samples {
        cluster_assignments.push(rng.random_range(0..n_clusters));
    }
    cluster_assignments.shuffle(&mut rng);
    cluster_assignments.truncate(n_samples);

    // generate samples with subspace structure
    for (i, &cluster_idx) in cluster_assignments.iter().enumerate() {
        let centre = &centres[cluster_idx];
        let std = cluster_stds[cluster_idx];
        let active_dims = &cluster_active_dims[cluster_idx];

        for j in 0..dim {
            let noise_scale = if active_dims.contains(&j) {
                std
            } else {
                std * 0.1
            };

            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data[(i, j)] = T::from_f64(centre[j] + noise * noise_scale).unwrap();
        }
    }

    data
}

/// BenchmarkResult
///
/// ### Fields
///
/// * `method` - Name of the method
/// * `build_time_ms` - The build time of the index in ms
/// * `query_time_ms` - The query time of the index in ms
/// * `total_time_ms` - Total time the index build & query takes in ms
/// * `recall_at_k` - Recall@k neighbours against ground truth
/// * `mean_dist_err` - Mean distance error against ground truth
pub struct BenchmarkResult {
    pub method: String,
    pub build_time_ms: f64,
    pub query_time_ms: f64,
    pub total_time_ms: f64,
    pub recall_at_k: f64,
    pub mean_dist_err: f64,
}

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
        let matches = approx_nn
            .iter()
            .take(k)
            .filter(|&idx| true_set.contains(idx))
            .count();
        total_recall += matches as f64 / k as f64;
    }

    total_recall / true_neighbors.len() as f64
}

/// Calculate mean distance error
///
/// ### Params
///
/// * `true_dist` - Slice of true distances to the neighbours
/// * `approx_dist` - Slice of approximate distances to the neighbours
/// * `k` - Number of selected k
///
/// ### Returns
///
/// The mean distance error
pub fn calculate_distance_error<T>(true_dist: &[Vec<T>], approx_dist: &[Vec<T>], k: usize) -> f64
where
    T: Float + ToPrimitive,
{
    let mut total_error = 0.0;

    for (true_dist, approx_dist) in true_dist.iter().zip(approx_dist.iter()) {
        for i in 0..k.min(true_dist.len()).min(approx_dist.len()) {
            let error = (true_dist[i].to_f64().unwrap() - approx_dist[i].to_f64().unwrap()).abs();
            total_error += error;
        }
    }

    total_error / (true_dist.len() * k) as f64
}

/// Helper to print results to console
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>100}", "");
    println!("Benchmark: {}", config);
    println!("{:=>100}", "");
    println!(
        "{:<35} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k", "Dist Error"
    );
    println!("{:->100}", "");
    for result in results {
        println!(
            "{:<35} {:>12.2} {:>12.2} {:>12.2} {:>12.4} {:>12.6}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k,
            result.mean_dist_err
        );
    }
    println!("{:->100}\n", "");
}

/// Helper to print results to console (Recall only)
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results_recall_only(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>83}", "");
    println!("Benchmark: {}", config);
    println!("{:=>83}", "");
    println!(
        "{:<30} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k"
    );
    println!("{:->83}", "");
    for result in results {
        println!(
            "{:<30} {:>12.2} {:>12.2} {:>12.2} {:>12.4}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k
        );
    }
    println!("{:->83}\n", "");
}
