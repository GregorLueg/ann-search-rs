use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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

    // Variable cluster sizes and std deviations
    let mut centres = Vec::with_capacity(n_clusters);
    let mut cluster_stds = Vec::new();

    for _ in 0..n_clusters {
        let centre: Vec<f64> = (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect();
        centres.push(centre);
        // variable density per cluster (0.5 to 2.0)
        cluster_stds.push(rng.random_range(0.5..2.0));
    }

    // Assign samples with variable cluster sizes
    let mut cluster_assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        // Some clusters bigger than others
        let weight = rng.random_range(0.5..2.0);
        let n_in_cluster = ((n_samples as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }

    // Fill remaining
    while cluster_assignments.len() < n_samples {
        cluster_assignments.push(rng.random_range(0..n_clusters));
    }
    cluster_assignments.shuffle(&mut rng);

    // Generate with variable noise
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
