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
    cluster_std: f64,
    seed: u64,
) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    // Generate cluster centres with reasonable separation
    let mut centres = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let centre: Vec<f64> = (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect();
        centres.push(centre);
    }

    // Create shuffled cluster assignments
    let base_size = n_samples / n_clusters;
    let remainder = n_samples % n_clusters;

    let mut cluster_assignments = Vec::with_capacity(n_samples);
    for cluster_idx in 0..n_clusters {
        let n_in_cluster = base_size + if cluster_idx < remainder { 1 } else { 0 };
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }

    // Shuffle assignments to avoid sorted clusters
    cluster_assignments.shuffle(&mut rng);

    // Generate samples
    for (i, &cluster_idx) in cluster_assignments.iter().enumerate() {
        let centre = &centres[cluster_idx];

        for j in 0..dim {
            // Box-Muller for Gaussian noise
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            data[(i, j)] = T::from_f64(centre[j] + noise * cluster_std).unwrap();
        }
    }

    data
}
