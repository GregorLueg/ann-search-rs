//! Product and optimised-product quantisers, plus the BF16 helpers.
//!
//! Scalar 8-bit quantisation lives in [`crate::quantised::uniform_quant`]: it
//! shares one scale across every dimension, which is what lets a code-to-code
//! distance stay faithful to the metric the caller asked for.

use crate::utils::gemm::gemm;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use half::bf16;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use std::num::NonZero;
use std::ops::AddAssign;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::prelude::*;
use crate::quantised::k_means::*;
use crate::quantised::pq_search::transpose_codebooks;
use crate::utils::k_means_utils::*;

/// Rows per tile when encoding many vectors. Each tile transposes every
/// codebook once, so the transpose is amortised over this many rows while the
/// tile itself stays small enough to keep per-thread scratch cheap. Not tuned.
pub(crate) const ENCODE_ROW_TILE: usize = 256;

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

/// Gather one subspace's slice of every row into a contiguous buffer.
///
/// ### Params
///
/// * `vectors` - Row-major vectors
/// * `dim` - Full dimension
/// * `n` - Number of vectors
/// * `s` - Subspace index
/// * `sd` - Subvector dimension
///
/// ### Returns
///
/// Row-major `n * sd` subvectors
fn subspace_vectors<T: Copy>(vectors: &[T], dim: usize, n: usize, s: usize, sd: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(n * sd);
    for row in vectors[..n * dim].chunks_exact(dim) {
        out.extend_from_slice(&row[s * sd..(s + 1) * sd]);
    }
    out
}

/// ProductQuantiser
///
/// ### Fields
///
/// * `codebooks` - M codebooks, each containing n centroids of dimension
///   subvec_dim
/// * `m` - Number of subspaces
/// * `subvec_dim` - Dimension of each subvector (dim / m)
/// * `n_centroids` - Number of centroids that were used to train the quantiser.
/// * `codebooks_t` - Dimension-major copy of the codebooks for lookup-table
///   builds. Derived, so not serialised: rebuilt by
///   `rebuild_codebooks_t` after a load.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct ProductQuantiser<T> {
    codebooks: Vec<Vec<T>>,
    m: usize,
    subvec_dim: usize,
    n_centroids: usize,
    #[cfg_attr(feature = "serialise", serde(skip, default = "Vec::new"))]
    codebooks_t: Vec<T>,
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
    /// * `verbosity` - Prints one line per codebook at `Detailed`
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
        verbosity: Verbosity,
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

        // subspaces in parallel: one at a time leaves most threads idle across
        // every mini-batch fork/join and the sequential centroid update
        let done = AtomicUsize::new(0);
        let codebooks: Vec<Vec<T>> = (0..m)
            .into_par_iter()
            .map(|subspace| {
                let subvectors = subspace_vectors(vectors_flat, dim, n, subspace, subvec_dim);
                let book = train_centroids_pq(
                    &subvectors,
                    subvec_dim,
                    n,
                    n_centroids,
                    max_iters,
                    seed + subspace,
                );
                if verbosity.detailed_verbosity() {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    println!("  Codebook {} / {}", d, m);
                }
                book
            })
            .collect();

        let codebooks_t = transpose_codebooks(&codebooks, subvec_dim, n_centroids);
        Ok(Self {
            codebooks,
            m,
            subvec_dim,
            n_centroids,
            codebooks_t,
        })
    }

    /// Dimension-major codebooks, see [`transpose_codebooks`]
    #[inline(always)]
    pub(crate) fn codebooks_t(&self) -> &[T] {
        &self.codebooks_t
    }

    /// Rebuild the derived dimension-major codebooks. Needed after a load,
    /// since they are not serialised.
    #[cfg(feature = "serialise")]
    pub(crate) fn rebuild_codebooks_t(&mut self) {
        self.codebooks_t = transpose_codebooks(&self.codebooks, self.subvec_dim, self.n_centroids);
    }

    /// Decode PQ codes into a caller-owned buffer
    ///
    /// ### Params
    ///
    /// * `codes` - `m` codes
    /// * `out` - Output, `m * subvec_dim`
    pub(crate) fn decode_into(&self, codes: &[u8], out: &mut [T]) {
        let sd = self.subvec_dim;
        for (s, &code) in codes.iter().enumerate() {
            let c = code as usize * sd;
            out[s * sd..(s + 1) * sd].copy_from_slice(&self.codebooks[s][c..c + sd]);
        }
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
        self.decode_into(codes, &mut result);
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
        total += self.codebooks_t.capacity() * std::mem::size_of::<T>();

        total
    }

    /// Encode a tile of vectors
    ///
    /// Works subspace by subspace: each codebook is transposed to
    /// dimension-major once, then every row of the tile is assigned against
    /// it with the PQ k-means kernel, so the scan vectorises across
    /// centroids. Sequential; callers parallelise over tiles.
    ///
    /// ### Params
    ///
    /// * `rows` - Row-major vectors, `t * dim` elements
    /// * `codes` - Output, `t * m` codes, row-major
    pub fn encode_rows(&self, rows: &[T], codes: &mut [u8]) {
        let m = self.m;
        let sd = self.subvec_dim;
        let k = self.n_centroids;
        let dim = m * sd;
        let mut book_t = vec![T::zero(); k * sd];
        let mut dist = vec![T::zero(); k];
        for (s, book) in self.codebooks.iter().enumerate() {
            transpose_centroids(book, sd, k, &mut book_t);
            for (r, row) in rows.chunks_exact(dim).enumerate() {
                let sub = &row[s * sd..(s + 1) * sd];
                codes[r * m + s] = find_nearest_transposed(sub, &book_t, k, &mut dist) as u8;
            }
        }
    }

    /// Encode many vectors in parallel tiles of [`ENCODE_ROW_TILE`] rows
    ///
    /// ### Params
    ///
    /// * `vectors` - The vectors, row-major
    /// * `n` - Number of vectors
    ///
    /// ### Returns
    ///
    /// Row-major `n * m` codes
    pub fn encode_batch(&self, vectors: &[T], n: usize) -> Vec<u8> {
        let dim = self.m * self.subvec_dim;
        let mut all_codes = vec![0u8; n * self.m];
        all_codes
            .par_chunks_mut(ENCODE_ROW_TILE * self.m)
            .zip(vectors[..n * dim].par_chunks(ENCODE_ROW_TILE * dim))
            .for_each(|(codes, rows)| self.encode_rows(rows, codes));
        all_codes
    }
}

///////////////////////////////
// OptimisedProductQuantiser //
///////////////////////////////

/// Rotate one vector: `out = R * vec`, one SIMD dot product per row of `R`.
///
/// ### Params
///
/// * `vec` - Input vector, length `dim`
/// * `rotation` - Rotation matrix, row-major `dim * dim`
/// * `dim` - Dimension
/// * `out` - Output, length `dim`
#[inline]
pub(crate) fn rotate_into<T: AnnSearchFloat>(vec: &[T], rotation: &[T], dim: usize, out: &mut [T]) {
    for (o, row) in out.iter_mut().zip(rotation.chunks_exact(dim)) {
        *o = T::dot_simd(row, vec);
    }
}

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
    /// * `verbosity` - Prints one line per rotation iteration at `Detailed`
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
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let n = vectors_flat.len() / dim;

        // identity rotation
        let mut rotation_matrix = vec![T::zero(); dim * dim];
        for i in 0..dim {
            rotation_matrix[i * dim + i] = T::one();
        }

        // reduced version to train for speed
        // should capture the majority of structure; scoped so the sample is
        // gone before the final full-size rotation
        {
            let sampled;
            let (training_vecs, n_train): (&[T], usize) = if n > 50_000 {
                sampled = sample_vectors(vectors_flat, dim, n, 50_000, seed).0;
                (&sampled, 50_000)
            } else {
                (vectors_flat, n)
            };

            let rotation_iter = n_iter.unwrap_or(OPQ_ITER);

            for iter in 0..rotation_iter {
                if verbosity.detailed_verbosity() {
                    println!("  OPQ iteration {} / {}", iter + 1, rotation_iter);
                }

                // rotate vectors
                let rotated = Self::apply_rotation(training_vecs, &rotation_matrix, dim, n_train);

                let pq = ProductQuantiser::train(
                    &rotated,
                    dim,
                    m,
                    n_centroids,
                    &Dist::SquaredEuclidean,
                    max_iters,
                    seed + iter,
                    Verbosity::Quiet,
                )?;

                // encode and reconstruct
                let codes = pq.encode_batch(&rotated, n_train);
                drop(rotated);
                let reconstructed: Vec<T> = (0..n_train)
                    .into_par_iter()
                    .flat_map(|vec_idx| {
                        let vec_codes = &codes[vec_idx * pq.m()..(vec_idx + 1) * pq.m()];
                        pq.decode(vec_codes)
                    })
                    .collect();

                // update rotation via SVD
                rotation_matrix =
                    Self::compute_rotation(training_vecs, &reconstructed, dim, n_train);
            }
        }

        // final training
        let rotated = Self::apply_rotation(vectors_flat, &rotation_matrix, dim, n);
        let pq = ProductQuantiser::train(
            &rotated,
            dim,
            m,
            n_centroids,
            &Dist::SquaredEuclidean,
            max_iters,
            seed,
            Verbosity::Quiet,
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

    /// Encode a tile of vectors
    ///
    /// Rotates the whole tile with one GEMM, then encodes it with
    /// [`ProductQuantiser::encode_rows`]. Callers parallelise over tiles.
    ///
    /// ### Params
    ///
    /// * `rows` - Row-major vectors, `t * dim` elements
    /// * `codes` - Output, `t * m` codes, row-major
    pub fn encode_rows(&self, rows: &[T], codes: &mut [u8]) {
        let t = rows.len() / self.dim;
        let rotated = Self::apply_rotation(rows, &self.rotation_matrix, self.dim, t);
        self.pq.encode_rows(&rotated, codes);
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
        rotate_into(vec, rotation, dim, &mut out);
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

                gemm(
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
        gemm(
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
        let rotated = self.pq.decode(codes);
        self.inverse_rotate(&rotated)
    }

    /// Apply inverse rotation (transpose) to a vector
    ///
    /// `R^T v` is `sum_j v[j] * R[j, :]`, so it walks contiguous rows of `R`
    /// and accumulates element-wise, which vectorises without a transpose.
    ///
    /// ### Params
    ///
    /// * `vec` - The vector on which to inverse the rotation
    ///
    /// ### Returns
    ///
    /// Vector with inverted rotation
    fn inverse_rotate(&self, vec: &[T]) -> Vec<T> {
        let dim = self.dim;
        let mut out = vec![T::zero(); dim];
        for (row, &v) in self.rotation_matrix.chunks_exact(dim).zip(vec) {
            for (o, &r) in out.iter_mut().zip(row) {
                *o += v * r;
            }
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

impl<T: AnnSearchFloat> OptimisedProductQuantiser<T> {
    /// The product quantiser applied after the rotation
    #[inline(always)]
    pub(crate) fn pq(&self) -> &ProductQuantiser<T> {
        &self.pq
    }

    /// Mutable access to the inner product quantiser, for rebuilding its
    /// derived tables after a load
    #[cfg(feature = "serialise")]
    pub(crate) fn pq_mut(&mut self) -> &mut ProductQuantiser<T> {
        &mut self.pq
    }

    /// Rotate a vector into a caller-owned buffer, see `rotate_into`
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector
    /// * `out` - Output, length `dim`
    #[inline]
    pub(crate) fn rotate_to(&self, vec: &[T], out: &mut [T]) {
        rotate_into(vec, &self.rotation_matrix, self.dim, out);
    }

    /// Rotation matrix, `dim * dim` row-major; a vector rotates as `R * v`
    pub(crate) fn rotation_matrix(&self) -> &[T] {
        &self.rotation_matrix
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

        let pq = ProductQuantiser::train(
            &data,
            32,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            5,
            42,
            Verbosity::Quiet,
        )
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
            Verbosity::Quiet,
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

        let pq1 = ProductQuantiser::train(
            &data,
            32,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let pq2 = ProductQuantiser::train(
            &data,
            32,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let vec: Vec<f32> = (16..48).map(|x| x as f32).collect();
        let codes1 = pq1.encode(&vec);
        let codes2 = pq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }

    #[test]
    fn test_product_quantiser_invalid_m() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ProductQuantiser::train(
            &data,
            5,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            5,
            42,
            Verbosity::Quiet,
        );
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionNotDivisibleByM { dim: 5, m: 2 })
        ));
    }

    #[test]
    fn test_product_quantiser_dim_too_small() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ProductQuantiser::train(
            &data,
            16,
            2,
            Some(2),
            &Dist::SquaredEuclidean,
            5,
            42,
            Verbosity::Quiet,
        );
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

        let opq = OptimisedProductQuantiser::train(
            &data,
            32,
            4,
            Some(4),
            Some(1),
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(opq.m(), 4);
        assert_eq!(opq.subvec_dim(), 8);
        assert_eq!(opq.n_centroids(), 4);
    }

    #[test]
    fn test_pq_encode_batch_matches_encode() {
        let dim = 32;
        let n = 2 * ENCODE_ROW_TILE + 37;
        let data: Vec<f32> = (0..n * dim)
            .map(|x| ((x * 2_654_435_761usize) % 1009) as f32 * 0.01)
            .collect();
        let pq = ProductQuantiser::train(
            &data,
            dim,
            4,
            Some(16),
            &Dist::SquaredEuclidean,
            10,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let batch = pq.encode_batch(&data, n);
        for (i, row) in data.chunks_exact(dim).enumerate() {
            assert_eq!(&batch[i * 4..(i + 1) * 4], pq.encode(row).as_slice());
        }
    }

    #[test]
    fn test_opq_encode() {
        let mut data = Vec::new();
        for i in 0..50 {
            for j in 0..32 {
                data.push((i + j) as f32);
            }
        }

        let opq = OptimisedProductQuantiser::train(
            &data,
            32,
            4,
            Some(4),
            Some(1),
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes = opq.encode(&vec);

        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_opq_rotate() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq = OptimisedProductQuantiser::train(
            &data,
            32,
            4,
            Some(4),
            Some(1),
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let rotated = opq.rotate(&vec);

        assert_eq!(rotated.len(), 32);
    }

    #[test]
    fn test_opq_deterministic() {
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();

        let opq1 = OptimisedProductQuantiser::train(
            &data,
            32,
            4,
            Some(4),
            Some(1),
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let opq2 = OptimisedProductQuantiser::train(
            &data,
            32,
            4,
            Some(4),
            Some(1),
            5,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let vec: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let codes1 = opq1.encode(&vec);
        let codes2 = opq2.encode(&vec);

        assert_eq!(codes1, codes2);
    }
}
