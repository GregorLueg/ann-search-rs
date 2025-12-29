#![allow(dead_code)]

use clap::Parser;
use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rustc_hash::FxHashSet;

////////////
// Consts //
////////////

pub const DEFAULT_N_CELLS: usize = 150_000;
pub const DEFAULT_N_QUERY: usize = DEFAULT_N_CELLS / 10;
pub const DEFAULT_DIM: usize = 32;
pub const DEFAULT_N_CLUSTERS: usize = 25;
pub const DEFAULT_K: usize = 15;
pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_DISTANCE: &str = "euclidean";
pub const DEFAULT_COR_STRENGTH: f64 = 0.5;
pub const DEFAULT_DATA: &str = "gaussian";
pub const DEFAULT_INTRINSIC_DIM: usize = 16;

////////////
// Parser //
////////////

/// Parsing structure
///
/// ### Fields
///
/// * `n_cells` - Number of cells/samples
/// * `dim` - Number of dimensions to use
/// * `n_clusters` - Number of clusters in the data
/// * `k` - Number of neighbours to search
/// * `seed` - Random seed for reproducibility
/// * `distance` - The distance to use. One of `"euclidean"` or `"cosine"`.
/// * `data` - The data to use. One of `"gaussian"` or `"correlated"`.
#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value_t = DEFAULT_N_CELLS)]
    pub n_cells: usize,

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

#[derive(Default)]
pub enum SyntheticData {
    #[default]
    GaussianNoise,
    Correlated,
    LowRank, // Add this
}

/// Helper function to parse the data type
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// `Option<SyntheticData>`
pub fn parse_data(s: &str) -> Option<SyntheticData> {
    match s.to_lowercase().as_str() {
        "gaussian" => Some(SyntheticData::GaussianNoise),
        "correlated" => Some(SyntheticData::Correlated),
        "lowrank" => Some(SyntheticData::LowRank),
        _ => None,
    }
}

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
) -> (Mat<T>, Vec<usize>)
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

    (data, cluster_assignments)
}

/// Generate synthetic single-cell-like data with cluster structure and correlated dimensions
///
/// Creates well-separated clusters with subspace structure plus inter-dimension
/// correlations that OPQ can exploit.
///
/// ### Params
///
/// * `n_samples` - Number of cells (samples)
/// * `dim` - Embedding dimensionality
/// * `n_clusters` - Number of distinct clusters
/// * `correlation_strength` - How strongly correlated dims depend on source dims (0.0-1.0)
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_clustered_data_high_dim<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    correlation_strength: f64,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    let scale = (dim as f64).sqrt() * 2.0;

    // generate well-separated centres
    let mut centres = Vec::with_capacity(n_clusters);
    let min_separation = scale * 0.8;

    for _ in 0..n_clusters {
        let centre = loop {
            let candidate: Vec<f64> = (0..dim).map(|_| rng.random_range(-scale..scale)).collect();
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

    // subspace structure
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

    // generate base samples
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

    // add correlation structure: groups of dimensions that are linear combinations
    // of "source" dimensions plus noise
    let n_correlation_groups = dim / 8;
    let dims_per_group = 4;
    let noise_weight = 1.0 - correlation_strength;

    for group in 0..n_correlation_groups {
        let source_dim = group * 8;
        if source_dim >= dim {
            break;
        }

        // generate random mixing coefficients for this group
        let coeffs: Vec<f64> = (0..dims_per_group)
            .map(|_| rng.random_range(-2.0..2.0))
            .collect();

        for target_offset in 1..=dims_per_group {
            let target_dim = source_dim + target_offset;
            if target_dim >= dim {
                break;
            }

            for i in 0..n_samples {
                let source_val = data[(i, source_dim)].to_f64().unwrap();
                let original_val = data[(i, target_dim)].to_f64().unwrap();

                // correlated value = weighted sum of source + original noise
                let correlated = source_val * coeffs[target_offset - 1] * correlation_strength
                    + original_val * noise_weight;

                data[(i, target_dim)] = T::from_f64(correlated).unwrap();
            }
        }
    }

    (data, cluster_assignments)
}

/// Generate data specifically designed to benefit from PCA+ITQ
///
/// Creates high-dimensional data that actually lives in a low-dimensional
/// subspace with rotated cluster structure. This is where ITQ should dominate
/// random projections.
///
/// ### Params
///
/// * `n_samples` - Number of samples
/// * `embedding_dim` - Full dimensionality (e.g., 128, 256)
/// * `intrinsic_dim` - True dimensionality of data (e.g., 16, 32)
/// * `n_clusters` - Number of clusters
/// * `seed` - Random seed
///
/// ### Returns
///
/// Matrix of shape (n_samples, embedding_dim)
pub fn generate_low_rank_rotated_data<T>(
    n_samples: usize,
    embedding_dim: usize,
    intrinsic_dim: usize,
    n_clusters: usize,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    assert!(
        intrinsic_dim <= embedding_dim,
        "Intrinsic dim must be <= embedding dim"
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate well-separated clusters in LOW-dimensional space
    let cluster_separation = (intrinsic_dim as f64).sqrt() * 3.0;
    let mut centres_low_dim = Vec::with_capacity(n_clusters);

    for _ in 0..n_clusters {
        let centre = loop {
            let candidate: Vec<f64> = (0..intrinsic_dim)
                .map(|_| rng.random_range(-cluster_separation..cluster_separation))
                .collect();

            let too_close = centres_low_dim.iter().any(|existing: &Vec<f64>| {
                let dist_sq: f64 = candidate
                    .iter()
                    .zip(existing.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                dist_sq < (cluster_separation * 0.5).powi(2)
            });

            if !too_close {
                break candidate;
            }
        };
        centres_low_dim.push(centre);
    }

    // Assign samples to clusters
    let mut cluster_assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        let n_in_cluster = n_samples / n_clusters;
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }
    while cluster_assignments.len() < n_samples {
        cluster_assignments.push(rng.random_range(0..n_clusters));
    }
    cluster_assignments.shuffle(&mut rng);

    // Generate samples in low-dimensional space
    let mut data_low_dim = Mat::<T>::zeros(n_samples, intrinsic_dim);
    let cluster_std = 0.3;

    for (i, &cluster_idx) in cluster_assignments.iter().enumerate() {
        let centre = &centres_low_dim[cluster_idx];
        for j in 0..intrinsic_dim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data_low_dim[(i, j)] = T::from_f64(centre[j] + noise * cluster_std).unwrap();
        }
    }

    // Create random rotation matrix to embed into high-dimensional space
    let mut rotation = Mat::<T>::zeros(intrinsic_dim, embedding_dim);

    // Fill with random Gaussian values
    for i in 0..intrinsic_dim {
        for j in 0..embedding_dim {
            let val: f64 = rng.sample(StandardNormal);
            rotation[(i, j)] = T::from_f64(val).unwrap();
        }
    }

    // Orthonormalise columns via Gram-Schmidt
    for col in 0..embedding_dim.min(intrinsic_dim) {
        // Orthogonalise against previous columns
        for prev_col in 0..col {
            let mut dot = T::zero();
            for row in 0..intrinsic_dim {
                dot = dot + rotation[(row, col)] * rotation[(row, prev_col)];
            }
            for row in 0..intrinsic_dim {
                rotation[(row, col)] = rotation[(row, col)] - dot * rotation[(row, prev_col)];
            }
        }

        // Normalise
        let mut norm_sq = T::zero();
        for row in 0..intrinsic_dim {
            norm_sq = norm_sq + rotation[(row, col)] * rotation[(row, col)];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for row in 0..intrinsic_dim {
                rotation[(row, col)] = rotation[(row, col)] / norm;
            }
        }
    }

    // Project to high-dimensional space: data_high = data_low * rotation
    let mut data_high_dim = Mat::<T>::zeros(n_samples, embedding_dim);
    for i in 0..n_samples {
        for j in 0..embedding_dim {
            let mut sum = T::zero();
            for k in 0..intrinsic_dim {
                sum = sum + data_low_dim[(i, k)] * rotation[(k, j)];
            }
            data_high_dim[(i, j)] = sum;
        }
    }

    // Add small amount of isotropic noise in the full space
    let noise_std = 0.01;
    for i in 0..n_samples {
        for j in 0..embedding_dim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data_high_dim[(i, j)] = data_high_dim[(i, j)] + T::from_f64(noise * noise_std).unwrap();
        }
    }

    (data_high_dim, cluster_assignments)
}

/// Randomly subsample a matrix and add Gaussian noise
///
/// ### Params
///
/// * `data` - The input matrix to subsample
/// * `n_samples` - Number of samples to draw
/// * `noise_scale` - Standard deviation of noise to add
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim) with noise added
pub fn subsample_with_noise<T>(data: &Mat<T>, n_samples: usize, seed: u64) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    let (n_rows, n_cols) = data.shape();

    let mut indices: Vec<usize> = (0..n_rows).collect();
    indices.shuffle(&mut rng);
    indices.truncate(n_samples.min(n_rows));

    let mut result = Mat::<T>::zeros(n_samples.min(n_rows), n_cols);

    for (i, &row_idx) in indices.iter().enumerate() {
        for j in 0..n_cols {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let noised_value = data[(row_idx, j)].to_f64().unwrap() + noise * 0.05;
            result[(i, j)] = T::from_f64(noised_value).unwrap();
        }
    }

    result
}

////////////////
// Benchmarks //
////////////////

////////////////
// Structures //
////////////////

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
pub struct BenchmarkResultSize {
    pub method: String,
    pub build_time_ms: f64,
    pub query_time_ms: f64,
    pub total_time_ms: f64,
    pub recall_at_k: f64,
    pub mean_dist_err: f64,
    pub index_size_mb: f64,
}

/// BenchmarkResultPurity - includes cluster purity metric
///
/// ### Fields
///
/// * `method` - Name of the method
/// * `build_time_ms` - The build time of the index in ms
/// * `query_time_ms` - The query time of the index in ms
/// * `total_time_ms` - Total time the index build & query takes in ms
/// * `recall_at_k` - Recall@k neighbours against ground truth
/// * `cluster_purity` - Fraction of neighbors from same cluster
/// * `index_size_mb` - Index size in MB
pub struct BenchmarkResultPurity {
    pub method: String,
    pub build_time_ms: f64,
    pub query_time_ms: f64,
    pub total_time_ms: f64,
    pub recall_at_k: f64,
    pub cluster_purity: f64,
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

/// Calculate cluster purity of kNN graph
///
/// Measures what fraction of each point's neighbors belong to the same cluster.
/// High purity (>0.8) means the method preserves cluster structure well.
///
/// ### Params
///
/// * `knn_graph` - Neighbor indices for each point
/// * `cluster_labels` - Ground truth cluster assignment for each point
///
/// ### Returns
///
/// Average fraction of same-cluster neighbors
pub fn calculate_cluster_purity(knn_graph: &[Vec<usize>], cluster_labels: &[usize]) -> f64 {
    let mut total_purity = 0.0;

    for (i, neighbors) in knn_graph.iter().enumerate() {
        let my_cluster = cluster_labels[i];
        let same_cluster = neighbors
            .iter()
            .filter(|&&idx| cluster_labels[idx] == my_cluster)
            .count();
        total_purity += same_cluster as f64 / neighbors.len() as f64;
    }

    total_purity / knn_graph.len() as f64
}

////////////
// Prints //
////////////

/// Helper to print results to console
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results(config: &str, results: &[BenchmarkResult]) {
    println!("\n{:=>110}", "");
    println!("Benchmark: {}", config);
    println!("{:=>110}", "");
    println!(
        "{:<45} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k", "Dist Error"
    );
    println!("{:->110}", "");
    for result in results {
        println!(
            "{:<45} {:>12.2} {:>12.2} {:>12.2} {:>12.4} {:>12.6}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k,
            result.mean_dist_err
        );
    }
    println!("{:->110}\n", "");
}

/// Helper to print results to console
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results_size(config: &str, results: &[BenchmarkResultSize]) {
    println!("\n{:=>123}", "");
    println!("Benchmark: {}", config);
    println!("{:=>123}", "");
    println!(
        "{:<45} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k", "Dist Error", "Size (MB)"
    );
    println!("{:->123}", "");
    for result in results {
        println!(
            "{:<45} {:>12.2} {:>12.2} {:>12.2} {:>12.4} {:>12.6} {:>12.2}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k,
            result.mean_dist_err,
            result.index_size_mb
        );
    }
    println!("{:->123}\n", "");
}

/// Print results with cluster purity
pub fn print_results_purity(config: &str, results: &[BenchmarkResultPurity]) {
    println!("\n{:=>123}", "");
    println!("Benchmark: {}", config);
    println!("{:=>123}", "");
    println!(
        "{:<45} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Method", "Build (ms)", "Query (ms)", "Total (ms)", "Recall@k", "Purity", "Size (MB)"
    );
    println!("{:->123}", "");
    for result in results {
        println!(
            "{:<45} {:>12.2} {:>12.2} {:>12.2} {:>12.4} {:>12.4} {:>12.2}",
            result.method,
            result.build_time_ms,
            result.query_time_ms,
            result.total_time_ms,
            result.recall_at_k,
            result.cluster_purity,
            result.index_size_mb
        );
    }
    println!("{:->123}\n", "");
}
