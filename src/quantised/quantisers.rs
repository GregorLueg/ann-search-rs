use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;

use crate::utils::dist::*;
use crate::utils::k_means::*;

////////////
// Consts //
////////////

pub const N_CLUSTERS_PQ: usize = 256;

/////////////////////////
// Scalar quantisation //
/////////////////////////

/// ScalarQuantiser
///
/// ### Fields
///
/// * `scales` - The maximum absolute values across each dimensions for
///   renormalisation.
pub struct ScalarQuantiser<T> {
    pub scales: Vec<T>,
}

impl<T> ScalarQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Train the scalar quantiser on a flat vector
    ///
    /// ### Params
    ///
    /// * `vec` - Flat slice of the values to quantise
    /// * `dim` - Number features in the vector
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn train(vec: &[T], dim: usize) -> Self {
        let mut scales = vec![T::zero(); dim];

        for chunk in vec.chunks_exact(dim) {
            for (d, &val) in chunk.iter().enumerate() {
                scales[d] = scales[d].max(val.abs());
            }
        }

        for scale in &mut scales {
            if *scale <= T::zero() {
                *scale = T::one();
            } else {
                *scale = *scale / T::from_i8(127).unwrap();
            }
        }

        Self { scales }
    }

    /// Encode a vector
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to encode
    ///
    /// ### Returns
    ///
    /// The quantised vector
    pub fn encode(&self, vec: &[T]) -> Vec<i8> {
        vec.iter()
            .enumerate()
            .map(|(d, &val)| {
                let scaled = val / self.scales[d];
                let clamped = scaled
                    .min(T::from_i8(127).unwrap())
                    .max(T::from_i8(-127).unwrap());
                clamped.to_i8().unwrap_or(0)
            })
            .collect()
    }

    /// Decode a vector
    ///
    /// ### Params
    ///
    /// * `quantised` - The quantised vector
    ///
    /// ### Returns
    ///
    /// Original decompressed vector
    pub fn decode(&self, quantised: &[i8]) -> Vec<T> {
        quantised
            .iter()
            .enumerate()
            .map(|(d, &val)| T::from_i8(val).unwrap() * self.scales[d])
            .collect()
    }
}

//////////////////////
// ProductQuantiser //
//////////////////////

/// ProductQuantiser
///
/// ### Fields
///
/// * `codebooks` - M codebooks, each containing 256 centroids of dimension
///   subvec_dim
/// * `m` - Number of subspaces
/// * `subvec_dim` - Dimension of each subvector (dim / m)
pub struct ProductQuantiser<T> {
    codebooks: Vec<Vec<T>>,
    m: usize,
    subvec_dim: usize,
}

impl<T> ProductQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Train the product quantiser
    ///
    /// Splits vectors into M subspaces and trains a 256-centroid codebook
    /// for each subspace using k-means.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flat slice of training vectors
    /// * `dim` - Dimension of each vector
    /// * `m` - Number of subspaces
    /// * `metric` - Distance metric (for k-means clustering)
    /// * `max_iters` - Maximum k-means iterations
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Trained ProductQuantiser
    pub fn train(
        vectors_flat: &[T],
        dim: usize,
        m: usize,
        metric: &Dist,
        max_iters: usize,
        seed: usize,
        verbose: bool,
    ) -> Self {
        // checks
        assert!(dim.is_multiple_of(m), "Dimension must be divisible by m");
        assert!(dim >= 32, "Dimension too small for product quantisation");

        // function body
        let subvec_dim = dim / m;
        let n = vectors_flat.len() / dim;

        let mut codebooks = Vec::with_capacity(m);

        for subspace in 0..m {
            if verbose {
                println!("  Training codebook {} / {}", subspace + 1, m);
            }

            // get the subvectors
            let mut subvectors = Vec::with_capacity(n * subvec_dim);
            for vec_idx in 0..n {
                let vec_start = vec_idx * dim + subspace * subvec_dim;
                subvectors.extend_from_slice(&vectors_flat[vec_start..vec_start + subvec_dim]);
            }

            // Train k-means with 256 clusters
            let centroids = train_centroids(
                &subvectors,
                subvec_dim,
                n,
                N_CLUSTERS_PQ,
                metric,
                max_iters,
                seed + subspace,
                false,
            );

            codebooks.push(centroids);
        }

        if verbose {
            println!("  Product quantiser training complete");
        }

        Self {
            codebooks,
            m,
            subvec_dim,
        }
    }

    /// Encode a vector to M indices
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to encode
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Vector of M indices (u8), one per subspace
    pub fn encode(&self, vec: &[T], metric: &Dist) -> Vec<u8> {
        let mut codes: Vec<u8> = Vec::with_capacity(self.m);

        for subspace in 0..self.m {
            let subvec_start = subspace * self.subvec_dim;
            let subvec = &vec[subvec_start..subvec_start + self.subvec_dim];

            // find nearest centroid in this subspace
            let mut best_idx = 0;
            let mut best_dist = T::infinity();

            for centroid_idx in 0..256 {
                let centroid_start = centroid_idx * self.subvec_dim;
                let centroid =
                    &self.codebooks[subspace][centroid_start..centroid_start + self.subvec_dim];

                let dist = match metric {
                    Dist::Euclidean => euclidean_distance_static(subvec, centroid),
                    Dist::Cosine => cosine_distance_static(subvec, centroid),
                };

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = centroid_idx;
                }
            }

            codes.push(best_idx as u8);
        }

        codes
    }

    /// Get number of subspaces
    pub fn m(&self) -> usize {
        self.m
    }

    /// Get subvector dimension
    pub fn subvec_dim(&self) -> usize {
        self.subvec_dim
    }

    /// Get codebooks (for building lookup tables)
    pub fn codebooks(&self) -> &[Vec<T>] {
        &self.codebooks
    }
}
