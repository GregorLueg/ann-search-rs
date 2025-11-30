use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
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

    // Generate cluster centres
    let mut centres = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let centre: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        centres.push(centre);
    }

    // Assign cells to clusters and generate data
    for i in 0..n_samples {
        let cluster_idx = i % n_clusters;
        let centre = &centres[cluster_idx];

        for j in 0..dim {
            let noise: f64 = rng.random_range(-cluster_std..cluster_std);
            let value = centre[j] + noise;
            data[(i, j)] = T::from_f64(value).unwrap();
        }
    }

    data
}
