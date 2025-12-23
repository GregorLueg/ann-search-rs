use faer::Mat;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;
use std::ops::AddAssign;

use crate::utils::dist::*;
use crate::utils::k_means::*;

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

pub const N_CLUSTERS_PQ: usize = 256;
pub const OPQ_ITER: usize = 3;

/// ProductQuantiser
///
/// ### Fields
///
/// * `codebooks` - M codebooks, each containing n centroids of dimension
///   subvec_dim
/// * `m` - Number of subspaces
/// * `subvec_dim` - Dimension of each subvector (dim / m)
/// * `n_centroids` - Number of centroids that were used to train the quantiser.
pub struct ProductQuantiser<T> {
    codebooks: Vec<Vec<T>>,
    m: usize,
    subvec_dim: usize,
    n_centroids: usize,
}

impl<T> ProductQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Train the product quantiser
    ///
    /// Splits vectors into M subspaces and trains a n-centroid codebook
    /// for each subspace using k-means.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flat slice of training vectors
    /// * `dim` - Dimension of each vector
    /// * `m` - Number of subspaces
    /// * `n_centroids` - Optional number of centroids to use per given
    ///   subspace. If not provided, it will default to `256`.
    /// * `metric` - Distance metric (for k-means clustering)
    /// * `max_iters` - Maximum k-means iterations
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Trained ProductQuantiser
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        vectors_flat: &[T],
        dim: usize,
        m: usize,
        n_centroids: Option<usize>,
        metric: &Dist,
        max_iters: usize,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let n_centroids = n_centroids.unwrap_or(N_CLUSTERS_PQ);

        // checks
        assert!(dim.is_multiple_of(m), "Dimension must be divisible by m");
        assert!(dim >= 32, "Dimension too small for product quantisation");
        assert!(
            n_centroids <= 256,
            "The number of centroids for PQ is limited to 256."
        );

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

            // train k-means with n clusters
            let centroids = train_centroids(
                &subvectors,
                subvec_dim,
                n,
                n_centroids,
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
            n_centroids,
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
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        let mut codes: Vec<u8> = Vec::with_capacity(self.m);

        for subspace in 0..self.m {
            let subvec_start = subspace * self.subvec_dim;
            let subvec = &vec[subvec_start..subvec_start + self.subvec_dim];

            let mut best_idx = 0;
            let mut best_dist = T::infinity();

            for centroid_idx in 0..self.n_centroids {
                let centroid_start = centroid_idx * self.subvec_dim;
                let centroid =
                    &self.codebooks[subspace][centroid_start..centroid_start + self.subvec_dim];

                let dist = euclidean_distance_static(subvec, centroid);

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
    #[inline(always)]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Get subvector dimension
    #[inline(always)]
    pub fn subvec_dim(&self) -> usize {
        self.subvec_dim
    }

    /// Return the number of centroids used for training
    #[inline(always)]
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// Get codebooks (for building lookup tables)
    pub fn codebooks(&self) -> &[Vec<T>] {
        &self.codebooks
    }
}

///////////////////////////////
// OptimisedProductQuantiser //
///////////////////////////////

/// OptimisedProductQuantiser
///
/// Product quantiser with learned rotation matrix to decorrelate dimensions
/// before quantisation, reducing quantisation error compared to standard PQ.
///
/// ### Fields
///
/// * `rotation_matrix` - Orthogonal rotation matrix (dim × dim, row-major)
/// * `pq` - Standard product quantiser applied after rotation
/// * `dim` - Dimension of vectors
pub struct OptimisedProductQuantiser<T> {
    rotation_matrix: Vec<T>,
    pq: ProductQuantiser<T>,
    dim: usize,
}

impl<T> OptimisedProductQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign,
{
    /// Train the optimised product quantiser
    ///
    /// Learns a rotation matrix via iterative refinement: alternates between
    /// training PQ codebooks on rotated vectors and updating the rotation
    /// matrix via SVD to minimise reconstruction error.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flat slice of training vectors
    /// * `dim` - Dimension of each vector
    /// * `m` - Number of subspaces
    /// * `n_centroids` - Optional number of centroids per subspace (defaults to 256)
    /// * `max_iters` - Maximum k-means iterations
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Trained OptimisedProductQuantiser
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        vectors_flat: &[T],
        dim: usize,
        m: usize,
        n_centroids: Option<usize>,
        n_iter: Option<usize>,
        max_iters: usize,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let n = vectors_flat.len() / dim;

        // identity rotation
        let mut rotation_matrix = vec![T::zero(); dim * dim];
        for i in 0..dim {
            rotation_matrix[i * dim + i] = T::one();
        }

        // reduced version to train for speed
        // should capture the majority of structure
        let (training_vecs, n_train) = if n > 50_000 {
            let (data, _) = sample_vectors(vectors_flat, dim, n, 50_000, seed);
            (data, 50_000)
        } else {
            (vectors_flat.to_vec(), n)
        };

        #[allow(unused_assignments)] // clippy being stupid here
        let mut pq = ProductQuantiser::train(
            &training_vecs,
            dim,
            m,
            n_centroids,
            &Dist::Euclidean,
            max_iters,
            seed,
            false,
        );

        let rotation_iter = n_iter.unwrap_or(OPQ_ITER);

        for iter in 0..rotation_iter {
            if verbose {
                println!("  OPQ iteration {} / {}", iter + 1, rotation_iter);
            }

            // rotate vectors
            let rotated = Self::apply_rotation(&training_vecs, &rotation_matrix, dim, n_train);

            // train PQ
            pq = ProductQuantiser::train(
                &rotated,
                dim,
                m,
                n_centroids,
                &Dist::Euclidean,
                max_iters,
                seed + iter,
                false,
            );

            // encode and reconstruct
            let reconstructed: Vec<T> = (0..n_train)
                .into_par_iter()
                .flat_map(|vec_idx| {
                    let vec_start = vec_idx * dim;
                    let codes = pq.encode(&rotated[vec_start..vec_start + dim]);
                    Self::decode_pq(&codes, &pq)
                })
                .collect();

            // update rotation via SVD
            rotation_matrix = Self::compute_rotation(&training_vecs, &reconstructed, dim, n_train);
        }

        // final training
        let rotated = Self::apply_rotation(vectors_flat, &rotation_matrix, dim, n);
        pq = ProductQuantiser::train(
            &rotated,
            dim,
            m,
            n_centroids,
            &Dist::Euclidean,
            max_iters,
            seed,
            verbose,
        );

        Self {
            rotation_matrix,
            pq,
            dim,
        }
    }

    /// Encode a vector to M indices
    ///
    /// Applies rotation matrix then encodes with standard PQ.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to encode
    ///
    /// ### Returns
    ///
    /// Vector of M indices (u8), one per subspace
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        let rotated = Self::rotate_vector(vec, &self.rotation_matrix, self.dim);
        self.pq.encode(&rotated)
    }

    /// Rotate a given vector
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to rotate
    ///
    /// ### Returns
    ///
    /// The returned vector
    pub fn rotate(&self, vec: &[T]) -> Vec<T> {
        Self::rotate_vector(vec, &self.rotation_matrix, self.dim)
    }

    /// Decode PQ codes to reconstructed vector
    ///
    /// Reconstructs a vector from its PQ codes by concatenating the appropriate
    /// centroids from each subspace codebook.
    ///
    /// ### Params
    ///
    /// * `codes` - M PQ codes (u8)
    /// * `pq` - Product quantiser containing codebooks
    ///
    /// ### Returns
    ///
    /// Reconstructed vector
    fn decode_pq(codes: &[u8], pq: &ProductQuantiser<T>) -> Vec<T> {
        let m = pq.m();
        let subvec_dim = pq.subvec_dim();
        let mut reconstructed = vec![T::zero(); m * subvec_dim];

        for subspace in 0..m {
            let code = codes[subspace] as usize;
            let centroid_start = code * subvec_dim;
            let centroid = &pq.codebooks()[subspace][centroid_start..centroid_start + subvec_dim];
            reconstructed[subspace * subvec_dim..(subspace + 1) * subvec_dim]
                .copy_from_slice(centroid);
        }

        reconstructed
    }

    /// Apply rotation matrix to a single vector
    ///
    /// Computes matrix-vector product: out = R * vec
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector
    /// * `rotation` - Rotation matrix (row-major)
    /// * `dim` - Vector dimension
    ///
    /// ### Returns
    ///
    /// Rotated vector
    fn rotate_vector(vec: &[T], rotation: &[T], dim: usize) -> Vec<T> {
        let mut out = vec![T::zero(); dim];
        for i in 0..dim {
            let mut sum = T::zero();
            for j in 0..dim {
                sum += rotation[i * dim + j] * vec[j];
            }
            out[i] = sum;
        }

        out
    }

    /// Apply rotation matrix to multiple vectors
    ///
    /// Rotates all vectors in the dataset by applying the rotation matrix to
    /// each vector individually.
    ///
    /// ### Params
    ///
    /// * `vectors` - Flat vector data (length = n * dim)
    /// * `rotation` - Rotation matrix (row-major)
    /// * `dim` - Vector dimension
    /// * `n` - Number of vectors
    ///
    /// ### Returns
    ///
    /// Flat rotated vectors (length = n * dim)
    fn apply_rotation(vectors: &[T], rotation: &[T], dim: usize, n: usize) -> Vec<T> {
        let mut rotated = vec![T::zero(); n * dim];
        rotated
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(vec_idx, chunk)| {
                let vec_start = vec_idx * dim;
                let rotated_vec =
                    Self::rotate_vector(&vectors[vec_start..vec_start + dim], rotation, dim);
                chunk.copy_from_slice(&rotated_vec);
            });

        rotated
    }

    /// Compute optimal rotation matrix via SVD
    ///
    /// Finds the orthogonal matrix R that minimises reconstruction error by
    /// computing `R = V * U^T` where `C = U * Σ * V^T`is the SVD of the cross-
    /// covariance matrix `X_rotated^T * X_reconstructed`.
    ///
    /// ### Params
    ///
    /// * `x_rotated` - Rotated training vectors (flat)
    /// * `x_recon` - PQ-reconstructed vectors (flat)
    /// * `dim` - Vector dimension
    /// * `n` - Number of vectors
    ///
    /// ### Returns
    ///
    /// Updated rotation matrix (dim × dim, row-major)
    fn compute_rotation(x_rotated: &[T], x_recon: &[T], dim: usize, n: usize) -> Vec<T> {
        let c_elements: Vec<f32> = (0..dim * dim)
            .into_par_iter()
            .map(|idx| {
                let i = idx / dim;
                let j = idx % dim;
                (0..n)
                    .map(|vec_idx| {
                        let vec_start = vec_idx * dim;
                        x_rotated[vec_start + i].to_f32().unwrap()
                            * x_recon[vec_start + j].to_f32().unwrap()
                    })
                    .sum()
            })
            .collect();

        let mut c = Mat::<f32>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                c[(i, j)] = c_elements[i * dim + j];
            }
        }

        // rest stays the same
        let svd = c.thin_svd().unwrap();
        let u = svd.U();
        let v = svd.V();
        let r = v * u.transpose();

        let mut rotation = vec![T::zero(); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                rotation[i * dim + j] = T::from_f32(r[(i, j)]).unwrap();
            }
        }
        rotation
    }

    /// Get number of subspaces
    #[inline(always)]
    pub fn m(&self) -> usize {
        self.pq.m()
    }

    /// Get subvector dimension
    #[inline(always)]
    pub fn subvec_dim(&self) -> usize {
        self.pq.subvec_dim()
    }

    /// Return the number of centroids used for training
    #[inline(always)]
    pub fn n_centroids(&self) -> usize {
        self.pq.n_centroids()
    }

    /// Get codebooks (for building lookup tables)
    pub fn codebooks(&self) -> &[Vec<T>] {
        self.pq.codebooks()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_scalar_quantiser_train() {
        let mut data = Vec::new();
        for i in 0..4 {
            for j in 0..32 {
                data.push((i * 32 + j) as f32);
            }
        }

        let pq = ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::Euclidean, 5, 42, false);

        assert_eq!(pq.m(), 2);
        assert_eq!(pq.subvec_dim(), 16);
        assert_eq!(pq.n_centroids(), 2);
        assert_eq!(pq.codebooks().len(), 2);
        assert_eq!(pq.codebooks()[0].len(), 32); // 2 centroids * 16 dims
        assert_eq!(pq.codebooks()[1].len(), 32);
    }

    #[test]
    fn test_scalar_quantiser_encode_decode() {
        let data = vec![127.0, 0.0, -127.0, 63.5, 0.0, -63.5];
        let sq = ScalarQuantiser::train(&data, 3);

        let vec = vec![100.0, -25.0, 50.0];
        let encoded = sq.encode(&vec);
        let decoded = sq.decode(&encoded);

        assert_eq!(encoded.len(), 3);
        assert_eq!(decoded.len(), 3);

        // Check values are reasonably close
        for (orig, dec) in vec.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < orig.abs() * 0.02);
        }
    }

    #[test]
    fn test_scalar_quantiser_clamping() {
        let data = vec![1.0, 1.0];
        let sq = ScalarQuantiser::train(&data, 2);

        let vec = vec![200.0, -200.0];
        let encoded = sq.encode(&vec);

        assert_eq!(encoded[0], 127);
        assert_eq!(encoded[1], -127);
    }

    #[test]
    fn test_scalar_quantiser_zero_scale() {
        let data = vec![0.0, 10.0, 0.0, 20.0];
        let sq = ScalarQuantiser::train(&data, 2);

        // First dimension is all zeros, should default to 1.0
        assert_relative_eq!(sq.scales[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_product_quantiser_train() {
        let mut data = Vec::new();
        for i in 0..4 {
            for j in 0..32 {
                data.push((i * 32 + j) as f32);
            }
        }

        let pq = ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::Euclidean, 5, 42, false);

        assert_eq!(pq.m(), 2);
        assert_eq!(pq.subvec_dim(), 16);
        assert_eq!(pq.n_centroids(), 2);
        assert_eq!(pq.codebooks().len(), 2);
        assert_eq!(pq.codebooks()[0].len(), 32); // 2 centroids * 16 dims
        assert_eq!(pq.codebooks()[1].len(), 32);
    }

    #[test]
    fn test_product_quantiser_encode() {
        let mut data = Vec::new();
        for i in 0..4 {
            for j in 0..32 {
                data.push((i * 10 + j % 10) as f32);
            }
        }

        let pq = ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::Euclidean, 10, 42, false);

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes = pq.encode(&vec);

        assert_eq!(codes.len(), 2);
        assert!(codes[0] < 2);
        assert!(codes[1] < 2);
    }

    #[test]
    fn test_product_quantiser_deterministic() {
        let mut data = Vec::new();
        for i in 0..3 {
            for j in 0..32 {
                data.push((i * 32 + j) as f32);
            }
        }

        let pq1 = ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::Euclidean, 5, 42, false);

        let pq2 = ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::Euclidean, 5, 42, false);

        let vec: Vec<f32> = (16..48).map(|x| x as f32).collect();
        let codes1 = pq1.encode(&vec);
        let codes2 = pq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }

    #[test]
    #[should_panic(expected = "Dimension must be divisible by m")]
    fn test_product_quantiser_invalid_m() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        ProductQuantiser::train(&data, 5, 2, Some(2), &Dist::Euclidean, 5, 42, false);
    }

    #[test]
    #[should_panic(expected = "Dimension too small")]
    fn test_product_quantiser_dim_too_small() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        ProductQuantiser::train(&data, 16, 2, Some(2), &Dist::Euclidean, 5, 42, false);
    }

    #[test]
    fn test_opq_train() {
        let mut data = Vec::new();
        for i in 0..100 {
            for j in 0..32 {
                data.push((i + j) as f32);
            }
        }

        let opq = OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false);

        assert_eq!(opq.m(), 4);
        assert_eq!(opq.subvec_dim(), 8);
        assert_eq!(opq.n_centroids(), 4);
    }

    #[test]
    fn test_opq_encode() {
        let mut data = Vec::new();
        for i in 0..50 {
            for j in 0..32 {
                data.push((i + j) as f32);
            }
        }

        let opq = OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false);

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes = opq.encode(&vec);

        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_opq_rotate() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq = OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false);

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let rotated = opq.rotate(&vec);

        assert_eq!(rotated.len(), 32);
    }

    #[test]
    fn test_opq_deterministic() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq1 = OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false);

        let opq2 = OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false);

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes1 = opq1.encode(&vec);
        let codes2 = opq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }
}
