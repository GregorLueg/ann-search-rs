//! Product and optimised-product quantisers, plus the BF16 helpers.
//!
//! Scalar 8-bit quantisation lives in [`crate::quantised::uniform_quant`]: it
//! shares one scale across every dimension, which is what lets a code-to-code
//! distance stay faithful to the metric the caller asked for.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par, Scale};
use half::bf16;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use std::num::NonZero;
use std::ops::AddAssign;

use crate::prelude::*;
use crate::quantised::k_means::*;
use crate::utils::k_means_utils::*;

/// Rows per tile of the OPQ rotation GEMM.
///
/// Bounds the intermediate to `ROTATION_ROW_TILE * dim` elements. Matches the
/// tile the k-means assignment path uses for the same shape of work.
const ROTATION_ROW_TILE: usize = 4096;

///////////////////////
// Bf16 quantisation //
///////////////////////

/// Encode a vector of floats to BF16 quantisation
///
/// Converts each element to f32 then to bf16, providing 2x memory compression
/// with minimal precision loss. BF16 preserves the dynamic range of f32
/// (8 exponent bits) whilst reducing mantissa precision (7 bits vs 23).
///
/// ### Params
///
/// * `vec` - Input vector of any float type
///
/// ### Returns
///
/// Vector quantised to bf16 format
pub fn encode_bf16_quantisation<T>(vec: &[T]) -> Vec<bf16>
where
    T: Float + ToPrimitive,
{
    vec.iter()
        .map(|x| bf16::from_f32(x.to_f32().unwrap()))
        .collect()
}

/// Decode a BF16 quantised vector back to float precision
///
/// Converts bf16 elements to f32 then to the target float type. This is
/// a lossless operation (bf16 → f32), but the original quantisation
/// (f32 → bf16) introduced ~3 decimal digits of precision loss.
///
/// ### Params
///
/// * `vec` - BF16 quantised vector
///
/// ### Returns
///
/// Vector decoded to target float type
pub fn decode_bf16_quantisation<T>(vec: &[bf16]) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    let res: Vec<T> = vec
        .iter()
        .map(|x| {
            let x_f32 = x.to_f32_const();
            T::from_f32(x_f32).unwrap()
        })
        .collect();

    res
}

/// Compute L2 norm of a BF16 quantised vector
///
/// Converts elements to f32 for computation to avoid bf16 accumulation
/// errors. Returns the norm in the target float type.
///
/// ### Params
///
/// * `vec` - BF16 quantised vector
///
/// ### Returns
///
/// L2 norm as target float type
pub fn bf16_norm<T>(vec: &[bf16]) -> T
where
    T: Float + FromPrimitive,
{
    let res = vec
        .iter()
        .map(|&v| v.to_f32() * v.to_f32())
        .fold(0_f32, |a, b| a + b)
        .sqrt();

    T::from_f32(res).unwrap()
}

//////////////////////
// ProductQuantiser //
//////////////////////

/// Number of clusters to use for k-means clustering during PQ
pub const N_CLUSTERS_PQ: usize = 256;

/// Number of iterations to run for in optimised product quantisation
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
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct ProductQuantiser<T> {
    codebooks: Vec<Vec<T>>,
    m: usize,
    subvec_dim: usize,
    n_centroids: usize,
}

impl<T> ProductQuantiser<T>
where
    T: AnnSearchFloat,
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
    ///
    /// ### Errors
    ///
    /// * `DimensionNotDivisibleByM` if `m` does not divide `dim`
    /// * `DimensionTooSmallForPQ` if `dim < 32`
    /// * `TooManyCentroidsForPQ` if `n_centroids > 256`, the `u8` code limit
    /// * `TooFewSamplesForCentroids` if there are fewer training vectors than
    ///   centroids. Seeding draws distinct rows, so it cannot fill the codebook.
    ///   Small datasets hit this with the default of 256.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        vectors_flat: &[T],
        dim: usize,
        m: usize,
        n_centroids: Option<usize>,
        _metric: &Dist,
        max_iters: usize,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let n_centroids = n_centroids.unwrap_or(N_CLUSTERS_PQ);
        // checks
        if !dim.is_multiple_of(m) {
            return Err(AnnSearchErrors::DimensionNotDivisibleByM { dim, m });
        }
        if dim < 32 {
            return Err(AnnSearchErrors::DimensionTooSmallForPQ { dim });
        }
        if n_centroids > 256 {
            return Err(AnnSearchErrors::TooManyCentroidsForPQ { n_centroids });
        }

        // function body
        let subvec_dim = dim / m;
        let n = vectors_flat.len() / dim;

        // Seeding samples `n_centroids` *distinct* training rows, so fewer rows
        // than centroids indexes past the end of the shuffled index list. Small
        // datasets hit this with the default codebook size of 256.
        if n_centroids > n {
            return Err(AnnSearchErrors::TooFewSamplesForCentroids {
                n_centroids,
                n_samples: n,
            });
        }

        let codebooks: Vec<Vec<T>> = (0..m)
            .map(|subspace| {
                if verbose {
                    println!("  Training codebook {} / {}", subspace + 1, m);
                }
                let mut subvectors = Vec::with_capacity(n * subvec_dim);
                for vec_idx in 0..n {
                    let vec_start = vec_idx * dim + subspace * subvec_dim;
                    subvectors.extend_from_slice(&vectors_flat[vec_start..vec_start + subvec_dim]);
                }
                train_centroids_pq(
                    &subvectors,
                    subvec_dim,
                    n,
                    n_centroids,
                    max_iters,
                    seed + subspace,
                )
            })
            .collect();

        if verbose {
            println!("  Product quantiser training complete");
        }

        Ok(Self {
            codebooks,
            m,
            subvec_dim,
            n_centroids,
        })
    }

    /// Encode a vector to M indices
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to encode
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

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<T> {
        let mut result = vec![T::zero(); self.m * self.subvec_dim];

        for (subspace, &code) in codes.iter().enumerate() {
            let out_start = subspace * self.subvec_dim;
            let centroid_start = code as usize * self.subvec_dim;
            let codebook = &self.codebooks[subspace];
            result[out_start..(self.subvec_dim + out_start)]
                .copy_from_slice(&codebook[centroid_start..(self.subvec_dim + centroid_start)]);
        }

        result
    }

    /// Returns the size of the quantiser
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        total += self.codebooks.capacity() * std::mem::size_of::<Vec<T>>();
        for codebook in &self.codebooks {
            total += codebook.capacity() * std::mem::size_of::<T>();
        }

        total
    }

    /// Batch encode multiple vectors - computes distances via matrix ops
    ///
    /// ### Params
    ///
    /// * `vectors` - The vectors
    /// * `n` - Size of the data set
    ///
    /// ### Returns
    ///
    /// Returns the assignments
    pub fn encode_batch(&self, vectors: &[T], n: usize) -> Vec<u8> {
        let dim = self.m() * self.subvec_dim();
        let mut all_codes = vec![0u8; n * self.m()];

        for subspace in 0..self.m() {
            let subvec_start = subspace * self.subvec_dim();

            // extract all subvectors for this subspace (n x subvec_dim)
            let mut subvecs = Mat::<T>::zeros(n, self.subvec_dim());
            for i in 0..n {
                for j in 0..self.subvec_dim() {
                    subvecs[(i, j)] = vectors[i * dim + subvec_start + j];
                }
            }

            // centroids matrix (n_centroids x subvec_dim)
            let mut centroids = Mat::<T>::zeros(self.n_centroids(), self.subvec_dim());
            for i in 0..self.n_centroids() {
                for j in 0..self.subvec_dim() {
                    centroids[(i, j)] = self.codebooks()[subspace][i * self.subvec_dim() + j];
                }
            }

            // Compute squared distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x@c^T
            // Precompute ||c||^2 for all centroids
            let centroid_norms: Vec<T> = (0..self.n_centroids())
                .map(|i| {
                    (0..self.subvec_dim())
                        .map(|j| centroids[(i, j)].powi(2))
                        .sum()
                })
                .collect();

            let dot_products = Scale(-T::from_f32(2.0).unwrap()) * &subvecs * centroids.transpose();

            all_codes
                .par_chunks_mut(self.m())
                .enumerate()
                .for_each(|(vec_idx, codes)| {
                    let mut best_idx = 0;
                    let mut best_dist = T::infinity();

                    for c in 0..self.n_centroids() {
                        let dist = centroid_norms[c] + dot_products[(vec_idx, c)];
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = c;
                        }
                    }

                    codes[subspace] = best_idx as u8;
                });
        }

        all_codes
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
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimisedProductQuantiser<T> {
    rotation_matrix: Vec<T>,
    pq: ProductQuantiser<T>,
    dim: usize,
}

impl<T> OptimisedProductQuantiser<T>
where
    T: AnnSearchFloat + AddAssign,
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
    ) -> Result<Self, AnnSearchErrors> {
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
            &Dist::SquaredEuclidean,
            max_iters,
            seed,
            false,
        )?;

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
                &Dist::SquaredEuclidean,
                max_iters,
                seed + iter,
                false,
            )?;

            // encode and reconstruct
            let codes = pq.encode_batch(&rotated, n_train);
            let reconstructed: Vec<T> = (0..n_train)
                .into_par_iter()
                .flat_map(|vec_idx| {
                    let vec_codes = &codes[vec_idx * pq.m()..(vec_idx + 1) * pq.m()];
                    pq.decode(vec_codes)
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
            &Dist::SquaredEuclidean,
            max_iters,
            seed,
            verbose,
        )?;

        Ok(Self {
            rotation_matrix,
            pq,
            dim,
        })
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
        let mut out = vec![T::zero(); n * dim];
        let r = MatRef::from_row_major_slice(rotation, dim, dim);

        // Tiled and rayon-parallel, writing straight into `out`. The input and
        // output slices are already the row-major layout faer wants, so nothing
        // is copied on the way in or out.
        vectors
            .par_chunks(ROTATION_ROW_TILE * dim)
            .zip(out.par_chunks_mut(ROTATION_ROW_TILE * dim))
            .for_each(|(rows_flat, out_tile)| {
                let rows = rows_flat.len() / dim;
                let mut dst = MatMut::from_row_major_slice_mut(out_tile, rows, dim);

                matmul(
                    dst.as_mut(),
                    Accum::Replace,
                    MatRef::from_row_major_slice(rows_flat, rows, dim),
                    r.transpose(),
                    T::one(),
                    Par::Seq,
                );
            });

        out
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
    fn compute_rotation(x_original: &[T], x_recon: &[T], dim: usize, n: usize) -> Vec<T> {
        let x = MatRef::from_row_major_slice(x_original, n, dim);
        let y = MatRef::from_row_major_slice(x_recon, n, dim);

        // One `dim x dim` GEMM with no outer parallelism, so faer gets the
        // threads. Both operands are already row-major, so neither is copied.
        let mut c = Mat::<T>::zeros(dim, dim);
        matmul(
            c.as_mut(),
            Accum::Replace,
            x.transpose(),
            y,
            T::one(),
            Par::Rayon(NonZero::new(rayon::current_num_threads().max(1)).unwrap()),
        );

        let svd = c.as_ref().thin_svd().unwrap();
        let r = svd.V() * svd.U().transpose();

        let mut rotation = vec![T::zero(); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                rotation[i * dim + j] = r[(i, j)];
            }
        }
        rotation
    }

    /// Decode PQ codes back to approximate original vector
    ///
    /// Decodes via PQ codebooks then applies inverse rotation (transpose).
    ///
    /// ### Params
    ///
    /// * `codes` - M PQ codes (u8)
    ///
    /// ### Returns
    ///
    /// Reconstructed vector in original space
    pub fn decode(&self, codes: &[u8]) -> Vec<T> {
        let rotated = Self::decode_pq(codes, &self.pq);
        self.inverse_rotate(&rotated)
    }

    /// Apply inverse rotation (transpose) to a vector
    ///
    /// ### Params
    ///
    /// * `vec` - The vector on which to inverse the rotation
    ///
    /// ### Returns
    ///
    /// Vector with inverted rotation
    fn inverse_rotate(&self, vec: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.dim];
        for i in 0..self.dim {
            let mut sum = T::zero();
            for j in 0..self.dim {
                // Transpose: swap i,j indices
                sum += self.rotation_matrix[j * self.dim + i] * vec[j];
            }
            out[i] = sum;
        }
        out
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

    /// Returns the size of the quantiser
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.rotation_matrix.capacity() * std::mem::size_of::<T>()
            + self.pq.memory_usage_bytes()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {

    /// The tiled GEMM replaced a whole-matrix rotation that materialised three
    /// copies of the data. It has to agree with the per-vector path, including
    /// on a row count that straddles the tile boundary.
    #[test]
    fn test_apply_rotation_matches_per_vector() {
        let dim = 16;
        let n = ROTATION_ROW_TILE + 53;

        // Orthogonal rotation from a QR of a deterministic pseudo-random matrix
        let mut raw = vec![0.0f64; dim * dim];
        for (i, v) in raw.iter_mut().enumerate() {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            *v = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
        }
        let q = Mat::<f64>::from_fn(dim, dim, |i, j| raw[i * dim + j])
            .as_ref()
            .qr()
            .compute_Q();
        let rotation: Vec<f64> = (0..dim * dim).map(|i| q[(i / dim, i % dim)]).collect();

        let mut data = vec![0.0f64; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f64) * 0.31).sin() * 2.0 - 0.4;
        }

        let batched = OptimisedProductQuantiser::<f64>::apply_rotation(&data, &rotation, dim, n);

        for i in 0..n {
            let one = OptimisedProductQuantiser::<f64>::rotate_vector(
                &data[i * dim..(i + 1) * dim],
                &rotation,
                dim,
            );
            for j in 0..dim {
                approx::assert_relative_eq!(
                    batched[i * dim + j],
                    one[j],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }
    use super::*;

    #[test]
    fn test_encode_bf16_empty() {
        let data: Vec<f32> = vec![];
        let encoded = encode_bf16_quantisation(&data);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_encode_bf16_single_value() {
        let data = vec![1.5_f32];
        let encoded = encode_bf16_quantisation(&data);
        assert_eq!(encoded.len(), 1);
        assert!((encoded[0].to_f32() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_encode_bf16_multiple_values() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let encoded = encode_bf16_quantisation(&data);
        assert_eq!(encoded.len(), 4);

        for (i, &val) in data.iter().enumerate() {
            assert!((encoded[i].to_f32() - val).abs() < 0.01);
        }
    }

    #[test]
    fn test_decode_bf16_empty() {
        let data: Vec<bf16> = vec![];
        let decoded = decode_bf16_quantisation::<f32>(&data);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_bf16_single_value() {
        let data = vec![bf16::from_f32(1.5)];
        let decoded = decode_bf16_quantisation::<f32>(&data);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = vec![1.0_f32, -2.5, 3.7, 0.0, 100.5];
        let encoded = encode_bf16_quantisation(&original);
        let decoded = decode_bf16_quantisation::<f32>(&encoded);

        assert_eq!(original.len(), decoded.len());
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.1);
        }
    }

    #[test]
    fn test_bf16_norm_empty() {
        let data: Vec<bf16> = vec![];
        let norm = bf16_norm::<f32>(&data);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_bf16_norm_single_value() {
        let data = vec![bf16::from_f32(3.0)];
        let norm = bf16_norm::<f32>(&data);
        assert!((norm - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bf16_norm_vector() {
        let data = vec![bf16::from_f32(3.0), bf16::from_f32(4.0)];
        let norm = bf16_norm::<f32>(&data);
        assert!((norm - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_bf16_norm_larger_vector() {
        let values = [1.0, 2.0, 2.0];
        let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        let norm = bf16_norm::<f32>(&data);
        let expected = (1.0_f32 + 4.0 + 4.0).sqrt();
        assert!((norm - expected).abs() < 0.1);
    }

    #[test]
    fn test_bf16_quantisation_preserves_sign() {
        let data = vec![-1.0_f32, -2.0, -3.0];
        let encoded = encode_bf16_quantisation(&data);

        for enc in encoded.iter() {
            assert!(enc.to_f32_const() < 0.0);
        }
    }

    #[test]
    fn test_bf16_quantisation_zero() {
        let data = vec![0.0_f32];
        let encoded = encode_bf16_quantisation(&data);
        assert_eq!(encoded[0].to_f32(), 0.0);
    }

    #[test]
    fn test_product_quantiser_train() {
        let mut data = Vec::new();
        for i in 0..4 {
            for j in 0..32 {
                data.push((i * 32 + j) as f32);
            }
        }

        let pq =
            ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::SquaredEuclidean, 5, 42, false)
                .unwrap();

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

        let pq = ProductQuantiser::train(
            &data,
            32,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            10,
            42,
            false,
        )
        .unwrap();

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

        let pq1 =
            ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::SquaredEuclidean, 5, 42, false)
                .unwrap();

        let pq2 =
            ProductQuantiser::train(&data, 32, 2, Some(2), &Dist::SquaredEuclidean, 5, 42, false)
                .unwrap();

        let vec: Vec<f32> = (16..48).map(|x| x as f32).collect();
        let codes1 = pq1.encode(&vec);
        let codes2 = pq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }

    #[test]
    fn test_product_quantiser_invalid_m() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result =
            ProductQuantiser::train(&data, 5, 2, Some(2), &Dist::SquaredEuclidean, 5, 42, false);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionNotDivisibleByM { dim: 5, m: 2 })
        ));
    }

    #[test]
    fn test_product_quantiser_dim_too_small() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result =
            ProductQuantiser::train(&data, 16, 2, Some(2), &Dist::SquaredEuclidean, 5, 42, false);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionTooSmallForPQ { dim: 16 })
        ));
    }

    #[test]
    fn test_opq_train() {
        let mut data = Vec::new();
        for i in 0..100 {
            for j in 0..32 {
                data.push((i + j) as f32);
            }
        }

        let opq =
            OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false).unwrap();

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

        let opq =
            OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false).unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes = opq.encode(&vec);

        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_opq_rotate() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq =
            OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false).unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let rotated = opq.rotate(&vec);

        assert_eq!(rotated.len(), 32);
    }

    #[test]
    fn test_opq_deterministic() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq1 =
            OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false).unwrap();

        let opq2 =
            OptimisedProductQuantiser::train(&data, 32, 4, Some(4), Some(1), 5, 42, false).unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes1 = opq1.encode(&vec);
        let codes2 = opq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }
}
