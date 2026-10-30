//! Implements the quantisation approach from RaBitQ, see:
//!
//! "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//! for Approximate Nearest Neighbor Search" (Gao and Long, 2024).

use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;

use crate::binary::dist_binary::*;
use crate::binary::rabitq::ex_bits::{
    const_scaling_factor, encode_ex_bits, excode_bytes, ERROR_EPSILON, MAX_EX_BITS,
};
use crate::binary::rabitq::fastscan::{
    build_sign_lut, build_sign_sums, pack_rabitq_blocked, unpack_rabitq_blocked, SignScanQuery,
    BLOCKED_ARCH,
};
use crate::binary::rabitq::rotator::{RaBitQRotator, RotatorKind};
use crate::binary::turboquant::pack::BlockedCodes;
use crate::prelude::*;
use crate::utils::k_means_utils::*;

////////////
// Consts //
////////////

/// Number of iterations for RaBitQ k-means clustering.
const RABITQ_K_MEANS_ITER: usize = 30;

/// Seed for the averaged magnitude scale. The scale is a mean over random
/// directions, so the seed only fixes it for reproducibility.
const T_CONST_SEED: u64 = 42;

///////////////////
// RaBitQEncoder //
///////////////////

/// Encoded vector
pub type VecEncoding<T> = (Vec<u8>, T, T);

/// Pure encoding logic for RaBitQ
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQEncoder<T> {
    /// The rotation applied before the sign bits are taken
    pub rotator: RaBitQRotator<T>,
    /// Dimensions of the encode
    pub dim: usize,
    /// Working dimensionality after rotation. Equal to `dim` for the dense
    /// rotation, rounded up to [`ROTATOR_PAD`](crate::binary::rabitq::rotator::ROTATOR_PAD)
    /// for the Hadamard one, so every
    /// post-rotation loop runs over this and not over `dim`.
    pub padded_dim: usize,
    /// Number of bytes
    pub n_bytes: usize,
    /// Distance metric to use
    pub metric: Dist,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for RaBitQEncoder<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> RaBitQEncoder<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + SimdDistance,
{
    /// Create encoder with random orthogonal rotation
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensions of the data set
    /// * `metric` - Distance metric to use
    /// * `seed` - Random seed to use
    pub fn new(dim: usize, metric: Dist, seed: u64) -> Self {
        Self::from_rotator(dim, metric, RaBitQRotator::new_auto(dim, seed))
    }

    /// Create an encoder with an explicitly chosen rotation
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensions of the data set
    /// * `metric` - Distance metric to use
    /// * `kind` - Which rotation to build
    /// * `seed` - Random seed to use
    ///
    /// ### Returns
    ///
    /// The encoder, or an error when the rotation cannot serve `dim`
    pub fn with_rotator_kind(
        dim: usize,
        metric: Dist,
        kind: RotatorKind,
        seed: u64,
    ) -> Result<Self, AnnSearchErrors> {
        Ok(Self::from_rotator(
            dim,
            metric,
            RaBitQRotator::new(dim, Some(kind), seed)?,
        ))
    }

    /// Assemble an encoder around a prebuilt rotation
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensions of the data set
    /// * `metric` - Distance metric to use
    /// * `rotator` - The rotation to encode with
    ///
    /// ### Returns
    ///
    /// The encoder
    fn from_rotator(dim: usize, metric: Dist, rotator: RaBitQRotator<T>) -> Self {
        let padded_dim = rotator.padded_dim();
        Self {
            rotator,
            dim,
            padded_dim,
            n_bytes: padded_dim.div_ceil(8),
            metric,
        }
    }

    /// Encode a vector relative to a centroid
    ///
    /// ### Params
    ///
    /// * `vec` - Slice of vector to encode
    /// * `centroid` - The centroid of the cluster.
    /// * `rotation` - The rotations to apply.
    ///
    /// ### Returns
    ///
    /// The `(binarised code, dist to centroid, inverse dot correction)`
    #[inline]
    pub fn encode_vector(
        &self,
        vec: &[T],
        centroid: &[T],
    ) -> Result<VecEncoding<T>, AnnSearchErrors> {
        self.check_dim(vec.len())?;

        // Compute residual
        let res = T::subtract_simd(vec, centroid);

        let dist_to_centroid = compute_l2_norm(&res);

        // Normalise residual to unit vector
        let v_c: Vec<T> = if dist_to_centroid > T::epsilon() {
            res.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); self.dim]
        };

        // Apply rotation
        let v_c_rotated = self.apply_rotation(&v_c);

        // Binary encode (sign bits)
        let mut binary = vec![0u8; self.n_bytes];
        for d in 0..self.padded_dim {
            if v_c_rotated[d] >= T::zero() {
                binary[d / 8] |= 1u8 << (d % 8);
            }
        }

        let l1: T = compute_l1_norm(&v_c_rotated);
        let dot_correction_inv = if l1 > T::from_f32(1e-6).unwrap() {
            T::one() / l1
        } else {
            T::zero()
        };

        Ok((binary, dist_to_centroid, dot_correction_inv))
    }

    /// Normalise a query the way this encoder's metric requires
    ///
    /// Cosine normalises to unit length, squared Euclidean passes through.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    ///
    /// ### Returns
    ///
    /// The metric-normalised query
    #[inline]
    pub fn normalise_query(&self, query: &[T]) -> Vec<T> {
        match self.metric {
            Dist::Cosine => {
                let norm = compute_l2_norm(query);
                if norm > T::epsilon() {
                    query.iter().map(|&x| x / norm).collect()
                } else {
                    query.to_vec()
                }
            }
            Dist::SquaredEuclidean => query.to_vec(),
            Dist::Manhattan => unreachable!(),
        }
    }

    /// Prepare a query for scanning any number of clusters
    ///
    /// Rotates once and builds one sign table over the rotated query. The
    /// table is shared by every probed cluster: the centroid enters the
    /// estimate only through each vector's `f_add` and the per-cluster
    /// `||q - c||`, so nothing here depends on which cluster is scanned.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector, already passed through
    ///   [`normalise_query`](Self::normalise_query)
    /// * `ex_bits` - Magnitude bits of the index the query is for. Above zero
    ///   the exact sign sums of the full-width estimate are built too.
    ///
    /// ### Returns
    ///
    /// The prepared query, or an error on a dimension mismatch
    #[inline]
    pub fn encode_query(
        &self,
        query: &[T],
        ex_bits: usize,
    ) -> Result<SignScanQuery<T>, AnnSearchErrors>
    where
        T: AnnSearchFloat,
    {
        self.check_dim(query.len())?;

        let rotated = self.apply_rotation(query);
        Ok(SignScanQuery {
            lut: build_sign_lut(&rotated)?,
            sum: rotated.iter().fold(T::zero(), |a, &b| a + b),
            sign_sums: if ex_bits > 0 {
                build_sign_sums(&rotated)
            } else {
                Vec::new()
            },
            rotated,
        })
    }

    /// Apply rotation to a vector
    ///
    /// ### Params
    ///
    /// * `vec` - The vector to which to apply the rotation.
    ///
    /// ### Returns
    ///
    /// The vector with rotation applied
    #[inline]
    pub fn apply_rotation(&self, vec: &[T]) -> Vec<T> {
        self.rotator.rotate(vec)
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.rotator.memory_usage_bytes()
    }
}

///////////////////
// RaBitQStorage //
///////////////////

/// Per-vector estimator factors
///
/// The squared distance estimate against a query `q` (rotated) whose cluster
/// centroid is `c` is
///
/// ```text
/// est = f_add + ||q - c||^2 + f_rescale * <sign, q>
/// low = est - f_error * ||q - c||
/// ```
///
/// with `sign` the `+-1` code. The centroid's own `<sign, c>` is folded into
/// `f_add`, which is what lets one sign table over `q` serve every cluster.
/// `low` is the reference's probabilistic lower bound, used to decide which
/// candidates are worth an exact distance.
#[repr(C)]
#[derive(Clone)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQPackedVector<T> {
    /// Query-independent additive term
    pub f_add: T,
    /// Multiplier on the signed query inner product
    pub f_rescale: T,
    /// Error bound coefficient, multiplied by `||q - c||` at query time
    pub f_error: T,
}

impl<T: Float + FromPrimitive> RaBitQPackedVector<T> {
    /// Factors from a one-bit encoding
    ///
    /// With `r` the residual, `dci = 1 / ||rot(r / ||r||)||_1` and `s` the
    /// `+-1` code, the original estimator
    /// `||r||^2 + ||q - c||^2 - 2 ||r|| dci <s, q - c>` splits into the
    /// query-independent `f_add` and the multiplier `f_rescale`. The error
    /// coefficient is the reference's 1-bit bound written in the same terms:
    /// its `||r||^2 ||x||^2 / <r, x>^2` collapses to `D * dci^2` for a sign
    /// code. Accumulated in f64, the `<s, c>` term is a long signed sum.
    ///
    /// ### Params
    ///
    /// * `code` - Sign bits, bit `d % 8` of byte `d / 8` set for `+1`
    /// * `dist_to_centroid` - `||r||`
    /// * `dot_correction_inv` - `dci`, zero when the L1 norm underflowed
    /// * `centroid_rotated` - The rotated centroid, `padded_dim` long
    ///
    /// ### Returns
    ///
    /// The factors
    pub fn from_code(
        code: &[u8],
        dist_to_centroid: T,
        dot_correction_inv: T,
        centroid_rotated: &[T],
    ) -> Self {
        let padded_dim = centroid_rotated.len() as f64;
        let ip_centroid: f64 = centroid_rotated
            .iter()
            .enumerate()
            .map(|(d, &c)| {
                let c = c.to_f64().unwrap_or(0.0);
                if code[d / 8] >> (d % 8) & 1 == 1 {
                    c
                } else {
                    -c
                }
            })
            .sum();

        let dist = dist_to_centroid.to_f64().unwrap_or(0.0);
        let dci = dot_correction_inv.to_f64().unwrap_or(0.0);
        let f_rescale = -2.0 * dist * dci;
        let f_add = dist * dist - f_rescale * ip_centroid;
        let f_error = 2.0
            * ERROR_EPSILON
            * dist
            * ((padded_dim * dci * dci - 1.0).max(0.0) / (padded_dim - 1.0).max(1.0)).sqrt();

        Self {
            f_add: T::from_f64(f_add).unwrap_or_else(T::zero),
            f_rescale: T::from_f64(f_rescale).unwrap_or_else(T::zero),
            f_error: T::from_f64(f_error).unwrap_or_else(T::zero),
        }
    }
}

impl<T> RaBitQPackedVector<T> {
    /// Memory usage in bytes for a single packed vector
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    #[inline]
    pub fn memory_usage_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// CSR-layout storage for RaBitQ encoded vectors
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQStorage<T> {
    /// The centroids of the data, nlist * dim, flattened
    pub centroids: Vec<T>,
    /// The same centroids in the encoder's rotated frame, nlist * padded_dim,
    /// flattened. Precomputed so the query path never rotates a centroid.
    pub centroids_rotated: Vec<T>,
    /// Norms of the centroids
    pub centroids_norm: Vec<T>,
    /// Per-vector estimator factors
    pub packed_vectors: Vec<RaBitQPackedVector<T>>,
    /// Original indices, ordered by cluster
    pub vector_indices: Vec<u32>,
    /// Cluster boundaries, len = nlist + 1
    pub offsets: Vec<u32>,
    /// The one-bit codes, one blocked fast-scan layout per cluster.
    ///
    /// This is the only copy: the scan kernels read this order directly, so
    /// there is no row-major array beside it. The byte order is
    /// architecture-specific, which is what `blocked_arch` records.
    pub blocked: Vec<BlockedCodes>,
    /// The [`BLOCKED_ARCH`] tag `blocked` was packed with.
    ///
    /// A load on a machine whose tag differs re-blocks before anything reads
    /// the codes, so a saved index still crosses architectures.
    pub blocked_arch: u8,
    /// Number of lists
    pub nlist: usize,
    /// Number of dimensions
    pub dim: usize,
    /// Dimensionality of the rotated frame, the stride of `centroids_rotated`
    pub padded_dim: usize,
    /// Number of bytes
    pub n_bytes: usize,
    /// Magnitude bits per coordinate on top of the sign, 0 for plain one-bit
    pub ex_bits: usize,
    /// Packed magnitude levels in cluster order, `n * ex_bytes`, empty at
    /// `ex_bits` 0. Read one vector at a time, only for candidates the
    /// one-bit bound cannot rule out, so row-major rather than blocked.
    pub ex_codes: Vec<u8>,
    /// `(f_add, f_rescale)` of the full-width estimate per vector in cluster
    /// order, interleaved; empty at `ex_bits` 0
    pub ex_factors: Vec<T>,
}

impl<T: Float + FromPrimitive + Clone> RaBitQStorage<T> {
    /// Create empty storage with given capacity
    ///
    /// ### Params
    ///
    /// * `nlist` - Number of lists
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality of the data
    /// * `padded_dim` - Dimensionality of the encoder's rotated frame
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn with_capacity(nlist: usize, n: usize, dim: usize, padded_dim: usize) -> Self {
        let n_bytes = padded_dim.div_ceil(8);
        Self {
            centroids: Vec::with_capacity(nlist * dim),
            centroids_rotated: Vec::with_capacity(nlist * padded_dim),
            centroids_norm: Vec::with_capacity(nlist),
            packed_vectors: Vec::with_capacity(n),
            vector_indices: Vec::with_capacity(n),
            offsets: vec![0; nlist + 1],
            blocked: Vec::new(),
            blocked_arch: BLOCKED_ARCH,
            nlist,
            dim,
            padded_dim,
            n_bytes,
            ex_bits: 0,
            ex_codes: Vec::new(),
            ex_factors: Vec::new(),
        }
    }

    /// Re-block the codes for this machine if they were packed on another
    ///
    /// A no-op in the common case. Called after a load, before anything reads
    /// the codes; the unpack-repack pair is a pure permutation, so no
    /// information is lost and the result is bit-identical to a fresh encode.
    pub fn reblock_for_this_arch(&mut self) {
        if self.blocked_arch == BLOCKED_ARCH {
            return;
        }

        let (n_bytes, src) = (self.n_bytes, self.blocked_arch);
        for c in 0..self.nlist {
            let n_vectors = (self.offsets[c + 1] - self.offsets[c]) as usize;
            let raw = unpack_rabitq_blocked(&self.blocked[c].data, n_vectors, n_bytes, src);
            self.blocked[c] = pack_rabitq_blocked(&raw, n_vectors, n_bytes);
        }
        self.blocked_arch = BLOCKED_ARCH;
    }

    /// Blocked codes for one cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The cluster's fast-scan layout
    #[inline]
    pub fn cluster_blocked(&self, cluster_idx: usize) -> &BlockedCodes {
        &self.blocked[cluster_idx]
    }

    /// Get centroid for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// Slice of the centroid
    #[inline]
    pub fn centroid(&self, cluster_idx: usize) -> &[T] {
        let start = cluster_idx * self.dim;
        &self.centroids[start..start + self.dim]
    }

    /// Get the rotated centroid for a cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// Slice of the centroid in the encoder's rotated frame
    #[inline]
    pub fn centroid_rotated(&self, cluster_idx: usize) -> &[T] {
        let start = cluster_idx * self.padded_dim;
        &self.centroids_rotated[start..start + self.padded_dim]
    }

    /// Returns the vector data for a given cluster index
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    /// * `local_idx` - Index position of within the cluster
    ///
    /// ### Returns
    ///
    /// The vector index in that cluster with the specific local index
    #[inline]
    pub fn get_vector_data(&self, cluster_idx: usize, local_idx: usize) -> &RaBitQPackedVector<T> {
        let global_idx = self.offsets[cluster_idx] as usize + local_idx;
        &self.packed_vectors[global_idx]
    }

    /// Slice access for cluster - only if you actually need to iterate
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// Slice of the packed vector in this cluster index
    #[inline]
    pub fn cluster_packed_data(&self, cluster_idx: usize) -> &[RaBitQPackedVector<T>] {
        let start = self.offsets[cluster_idx] as usize;
        let end = self.offsets[cluster_idx + 1] as usize;
        &self.packed_vectors[start..end]
    }

    /// Get vector indices for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The vector indices (original) for every vector in this cluster
    #[inline]
    pub fn cluster_vector_indices(&self, cluster_idx: usize) -> &[u32] {
        let start = self.offsets[cluster_idx] as usize;
        let end = self.offsets[cluster_idx + 1] as usize;
        &self.vector_indices[start..end]
    }

    /// Number of vectors in cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// Number of vectors in that cluster
    #[inline]
    pub fn cluster_size(&self, cluster_idx: usize) -> usize {
        (self.offsets[cluster_idx + 1] - self.offsets[cluster_idx]) as usize
    }

    /// Total vectors stored
    ///
    /// ### Returns
    ///
    /// Total number of internal vectors
    #[inline]
    pub fn n_vectors(&self) -> usize {
        self.vector_indices.len()
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_rotated.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self
                .blocked
                .iter()
                .map(|b| b.data.capacity())
                .sum::<usize>()
            + self.packed_vectors.capacity() * std::mem::size_of::<RaBitQPackedVector<T>>()
            + self.vector_indices.capacity() * std::mem::size_of::<u32>()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.ex_codes.capacity()
            + self.ex_factors.capacity() * std::mem::size_of::<T>()
    }
}

/// Build RaBitQStorage from data and cluster assignments
///
/// ### Params
///
/// * `data` - Flattened vectors
/// * `dim` - Dimensionality of the data
/// * `n` - Number of vectors in the data
/// * `centroids` - The generated centroids
/// * `nlist` - Number of centroids generated
/// * `assignments` - Assignment of vector to cluster
/// * `encoder` - The RaBitQEncoder
/// * `ex_bits` - Magnitude bits per coordinate, `0..=MAX_EX_BITS`. Zero keeps
///   the plain one-bit code
///
/// ### Returns
///
/// The RaBitQStorage, or an error when `ex_bits` is out of range
#[allow(clippy::too_many_arguments)]
pub fn build_rabitq_storage<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    nlist: usize,
    assignments: &[u32],
    encoder: &RaBitQEncoder<T>,
    ex_bits: usize,
) -> Result<RaBitQStorage<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    if ex_bits > MAX_EX_BITS {
        return Err(AnnSearchErrors::RaBitQInvalidExBits {
            ex_bits,
            max: MAX_EX_BITS,
        });
    }

    let padded_dim = encoder.padded_dim;
    let n_bytes = encoder.n_bytes;

    // Compute centroid norms
    let centroids_norm: Vec<T> = (0..nlist)
        .map(|i| compute_l2_norm(&centroids[i * dim..(i + 1) * dim]))
        .collect();

    // Count vectors per cluster
    let mut counts = vec![0u32; nlist];
    for &a in assignments {
        counts[a as usize] += 1;
    }

    // Build offsets
    let mut offsets = vec![0u32; nlist + 1];
    for i in 0..nlist {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    // Rotate every centroid once so the query path never has to. nlist is
    // small and this is a build-time one-off, so it stays sequential.
    let mut centroids_rotated = Vec::with_capacity(nlist * padded_dim);
    for c in 0..nlist {
        centroids_rotated
            .extend_from_slice(&encoder.apply_rotation(&centroids[c * dim..(c + 1) * dim]));
    }

    // Allocate storage
    let mut storage = RaBitQStorage {
        centroids: centroids.to_vec(),
        centroids_rotated,
        centroids_norm,
        packed_vectors: vec![
            RaBitQPackedVector {
                f_add: T::zero(),
                f_rescale: T::zero(),
                f_error: T::zero(),
            };
            n
        ],
        vector_indices: vec![0u32; n],
        offsets: offsets.clone(),
        blocked: Vec::new(),
        blocked_arch: BLOCKED_ARCH,
        nlist,
        dim,
        padded_dim,
        n_bytes,
        ex_bits,
        ex_codes: Vec::new(),
        ex_factors: Vec::new(),
    };

    // Row-major codes live only until they are blocked, which is why they are
    // a local rather than a field: the scan kernels read the blocked order and
    // nothing else ever wants this one.
    let mut codes_flat = vec![0u8; n * n_bytes];
    let mut insert_pos = offsets[..nlist].to_vec();
    for vec_idx in 0..n {
        let cluster_idx = assignments[vec_idx] as usize;
        storage.vector_indices[insert_pos[cluster_idx] as usize] = vec_idx as u32;
        insert_pos[cluster_idx] += 1;
    }

    // At `ex_bits` 0 an empty buffer yields no chunks and the zip below would
    // encode nothing, so keep a one-byte placeholder stride and drop it after.
    let ex_bytes = excode_bytes(padded_dim, ex_bits);
    let ex_stride = ex_bytes.max(1);
    let mut ex_codes = vec![0u8; n * ex_stride];
    let mut ex_factors = vec![T::zero(); n * 2];

    // One averaged scale instead of a sweep per vector, as the reference's
    // IVF does. The exact sweep's cost grows with the width.
    let t_const = (ex_bits > 0).then(|| const_scaling_factor(padded_dim, ex_bits, T_CONST_SEED));

    let centroids_rotated = &storage.centroids_rotated;
    let vector_indices = &storage.vector_indices;
    codes_flat
        .par_chunks_mut(n_bytes)
        .zip(storage.packed_vectors.par_iter_mut())
        .zip(ex_codes.par_chunks_mut(ex_stride))
        .zip(ex_factors.par_chunks_mut(2))
        .enumerate()
        .try_for_each(
            |(pos, (((code, packed), ex_code), ex_factor))| -> Result<(), AnnSearchErrors> {
                let vec_idx = vector_indices[pos] as usize;
                let c = assignments[vec_idx] as usize;
                let c_rot = &centroids_rotated[c * padded_dim..(c + 1) * padded_dim];

                // Rotate the vector, not the residual: the rotation is linear,
                // and both codes then read their signs off one residual array,
                // so they cannot disagree on a coordinate.
                let rotated = encoder.apply_rotation(&data[vec_idx * dim..(vec_idx + 1) * dim]);
                let residual = T::subtract_simd(&rotated, c_rot);

                let dist = compute_l2_norm(&residual);
                let l1 = compute_l1_norm(&residual);
                let dot_correction_inv = if l1 > T::from_f32(1e-6).unwrap() * dist {
                    dist / l1
                } else {
                    T::zero()
                };
                for (d, &r) in residual.iter().enumerate() {
                    if r > T::zero() {
                        code[d / 8] |= 1u8 << (d % 8);
                    }
                }
                *packed = RaBitQPackedVector::from_code(code, dist, dot_correction_inv, c_rot);

                if ex_bits > 0 {
                    let encoded = encode_ex_bits(&rotated, c_rot, ex_bits, t_const)?;
                    debug_assert_eq!(encoded.sign_code.as_slice(), &*code);
                    ex_code.copy_from_slice(&encoded.ex_code);
                    ex_factor[0] = encoded.f_add;
                    ex_factor[1] = encoded.f_rescale;
                }
                Ok(())
            },
        )?;

    if ex_bits > 0 {
        storage.ex_codes = ex_codes;
        storage.ex_factors = ex_factors;
    }

    storage.blocked = (0..nlist)
        .map(|c| {
            let (start, end) = (offsets[c] as usize, offsets[c + 1] as usize);
            pack_rabitq_blocked(
                &codes_flat[start * n_bytes..end * n_bytes],
                end - start,
                n_bytes,
            )
        })
        .collect();

    Ok(storage)
}

/////////////////////
// RaBitQQuantiser //
/////////////////////

/// RaBitQ quantiser using CSR storage
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQQuantiser<T> {
    /// The RaBitQ encoder structure
    pub encoder: RaBitQEncoder<T>,
    /// The RaBitQ storage structure
    pub storage: RaBitQStorage<T>,
}

impl<T> RaBitQQuantiser<T>
where
    T: AnnSearchFloat,
{
    /// Create a new RaBitQ quantiser
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data on which to train the Quantiser
    /// * `metric` - Which distance metric to use
    /// * `n_clusters` - Optional number of centroids. If not provided, defaults
    ///   to `0.5 * sqrt(n)`.
    /// * `ex_bits` - Magnitude bits per coordinate on top of the sign,
    ///   `0..=8`. Above zero, candidates the one-bit bound cannot rule out
    ///   are re-scored on the wider code
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        ex_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        Self::with_rotator_kind(data, metric, n_clusters, ex_bits, None, seed)
    }

    /// Create a new RaBitQ quantiser with an explicitly chosen rotation
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data on which to train the Quantiser
    /// * `metric` - Which distance metric to use
    /// * `n_clusters` - Optional number of centroids. If not provided, defaults
    ///   to `0.5 * sqrt(n)`.
    /// * `ex_bits` - Magnitude bits per coordinate on top of the sign,
    ///   `0..=8`. Above zero, candidates the one-bit bound cannot rule out
    ///   are re-scored on the wider code
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let
    ///   [`resolve_rotator_kind`](crate::binary::rabitq::rotator::resolve_rotator_kind)
    ///   decide
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn with_rotator_kind(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        ex_bits: usize,
        rotator_kind: Option<RotatorKind>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        let (mut data_flat, n, dim) = data.into_row_major();

        let k = n_clusters
            .unwrap_or_else(|| ((n as f64).sqrt() * 0.5).ceil() as usize)
            .max(1)
            .min(n);

        // Norms are captured before any rescaling. The cosine path normalises
        // the rows in place, but the Euclidean path needs the originals for
        // `cluster_norms` below.
        let mut data_norms = vec![T::zero(); n];
        let normalise = *metric == Dist::Cosine;

        data_flat
            .par_chunks_mut(dim)
            .zip(data_norms.par_iter_mut())
            .for_each(|(row, norm_out)| {
                let norm = compute_l2_norm(row);
                *norm_out = norm;

                if normalise && norm > T::epsilon() {
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            });

        let cluster_norms = if normalise {
            vec![T::one(); n]
        } else {
            data_norms
        };

        let k_means_params = KMeansTrainingParams::new(RABITQ_K_MEANS_ITER, None, None);

        // Train centroids
        let centroids_flat = train_centroids(
            &data_flat,
            dim,
            n,
            k,
            metric,
            Some(k_means_params),
            seed,
            Verbosity::Quiet,
        )?;

        let centroid_norms: Vec<T> = (0..k)
            .map(|c| {
                let cent = &centroids_flat[c * dim..(c + 1) * dim];
                compute_l2_norm(cent)
            })
            .collect();

        // Assign vectors to clusters
        let assignments = assign_all_parallel(
            &data_flat,
            &cluster_norms,
            dim,
            n,
            &centroids_flat,
            &centroid_norms,
            k,
            metric,
        );

        // Create encoder
        let encoder = match rotator_kind {
            Some(kind) => RaBitQEncoder::with_rotator_kind(dim, *metric, kind, seed as u64)?,
            None => RaBitQEncoder::new(dim, *metric, seed as u64),
        };

        // Build CSR storage
        let storage = build_rabitq_storage(
            &data_flat,
            dim,
            n,
            &centroids_flat,
            k,
            &assignments,
            &encoder,
            ex_bits,
        )?;

        Ok(Self { encoder, storage })
    }

    /// Returns the number of clusters
    ///
    /// ### Returns
    ///
    /// Number of cluster stored in the structure
    pub fn n_clusters(&self) -> usize {
        self.storage.nlist
    }

    /// Returns the number of vectors
    ///
    /// ### Returns
    ///
    /// Number of vectors in the structure
    pub fn n_vectors(&self) -> usize {
        self.storage.n_vectors()
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.encoder.memory_usage_bytes() + self.storage.memory_usage_bytes()
    }
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

/// Implementation of the trait for RaBitQQuantiser
impl<T> VectorDistanceRaBitQ<T> for RaBitQQuantiser<T>
where
    T: Float + FromPrimitive,
{
    fn storage(&self) -> &RaBitQStorage<T> {
        &self.storage
    }

    fn encoder(&self) -> &RaBitQEncoder<T> {
        &self.encoder
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for RaBitQQuantiser<T>
where
    T: Float + FromPrimitive + Sum + SimdDistance,
{
    fn centroids(&self) -> &[T] {
        &self.storage.centroids
    }

    fn dim(&self) -> usize {
        self.storage.dim
    }

    fn nlist(&self) -> usize {
        self.storage.nlist
    }

    fn metric(&self) -> Dist {
        self.encoder.metric
    }

    fn centroids_norm(&self) -> &[T] {
        &self.storage.centroids_norm
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use faer::Mat;

    fn sample_data_2d() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.5,
        ]
    }

    /// Decorrelated pseudo-random matrix, so cluster assignment is not
    /// degenerate and the rotated-frame comparison sees real residuals.
    fn rotation_test_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
    }

    #[test]
    fn test_stored_rotated_centroids_match_the_encoder() {
        let data = rotation_test_data(200, 32);
        let q =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(6), 0, 42).unwrap();

        for c in 0..q.storage.nlist {
            let expected = q.encoder.apply_rotation(q.storage.centroid(c));
            for (got, want) in q.storage.centroid_rotated(c).iter().zip(expected.iter()) {
                assert_abs_diff_eq!(got, want, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        assert_eq!(encoder.dim, 4);
        // Rounded up to a whole code byte even though the rotation is square.
        assert_eq!(encoder.padded_dim, 8);
        assert_eq!(encoder.n_bytes, 1);
        assert_eq!(encoder.rotator.kind(), RotatorKind::Dense);
    }

    #[test]
    fn test_encoder_pads_once_the_hadamard_rotation_kicks_in() {
        let encoder = RaBitQEncoder::<f32>::new(200, Dist::SquaredEuclidean, 42);
        assert_eq!(encoder.rotator.kind(), RotatorKind::FhtKac);
        assert_eq!(encoder.padded_dim, 256);
        assert_eq!(encoder.n_bytes, 32);

        let (binary, _, _) = encoder
            .encode_vector(&vec![1.0; 200], &vec![0.0; 200])
            .unwrap();
        assert_eq!(binary.len(), 32);
    }

    #[test]
    fn test_rotation_orthogonality() {
        // Both rotations must preserve norms, which is what the sign bits and
        // the L1 dot correction downstream assume.
        for (dim, kind) in [(8usize, RotatorKind::Dense), (128, RotatorKind::FhtKac)] {
            let encoder =
                RaBitQEncoder::<f32>::with_rotator_kind(dim, Dist::SquaredEuclidean, kind, 42)
                    .unwrap();

            for seed in 0..4u64 {
                let v: Vec<f32> = (0..dim)
                    .map(|i| ((i as u64 * 2654435761 + seed * 97) % 211) as f32 / 211.0 - 0.5)
                    .collect();
                let rotated = encoder.apply_rotation(&v);

                let norm_in = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_out = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert_abs_diff_eq!(norm_out, norm_in, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_encode_vector_basic() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        let vec = vec![1.0, 0.0, 0.0, 0.0];
        let centroid = vec![0.0, 0.0, 0.0, 0.0];

        let (binary, dist, correction) = encoder.encode_vector(&vec, &centroid).unwrap();

        assert_eq!(binary.len(), 1); // 4 dims = 1 byte
        assert_abs_diff_eq!(dist, 1.0, epsilon = 1e-5);
        assert!(correction > 0.0);
    }

    #[test]
    fn test_encode_vector_with_centroid() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        let vec = vec![2.0, 2.0, 0.0, 0.0];
        let centroid = vec![1.0, 1.0, 0.0, 0.0];

        let (_, dist, _) = encoder.encode_vector(&vec, &centroid).unwrap();

        let expected_dist = (1.0f32 + 1.0f32).sqrt();
        assert_abs_diff_eq!(dist, expected_dist, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_query_builds_one_sub_table_per_byte_group() {
        let encoder = RaBitQEncoder::<f32>::new(8, Dist::SquaredEuclidean, 42);
        let query = vec![1.0; 8];

        let encoded = encoder.encode_query(&query, 0).unwrap();

        assert_eq!(encoded.lut.n_byte_groups, encoder.n_bytes);
        assert_eq!(encoded.lut.luts_u8.len(), encoder.n_bytes * 32);
    }

    #[test]
    fn test_encode_query_rotates_the_normalised_query() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Cosine, 42);
        let normalised = encoder.normalise_query(&[2.0, 0.0, 0.0, 0.0]);

        let encoded = encoder.encode_query(&normalised, 0).unwrap();

        // Unit length in, unit length out: the rotation is orthogonal.
        let norm = encoded.rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-5);
        assert_eq!(encoded.rotated, encoder.apply_rotation(&normalised));
    }

    #[test]
    fn test_storage_creation() {
        let storage = RaBitQStorage::<f32>::with_capacity(10, 100, 8, 8);
        assert_eq!(storage.nlist, 10);
        assert_eq!(storage.dim, 8);
        assert_eq!(storage.n_bytes, 1);
        assert_eq!(storage.offsets.len(), 11);
    }

    #[test]
    fn test_build_rabitq_storage() {
        let data = sample_data_2d();
        let dim = 2;
        let n = 6;
        let nlist = 2;

        let centroids = vec![0.5, 0.0, -0.5, 0.0]; // 2 centroids
        let assignments = vec![0, 0, 1, 1, 0, 1]; // 3 vectors per cluster
        let encoder = RaBitQEncoder::new(dim, Dist::SquaredEuclidean, 42);

        let storage =
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder, 0)
                .unwrap();

        assert_eq!(storage.nlist, 2);
        assert_eq!(storage.n_vectors(), 6);
        assert_eq!(storage.cluster_size(0), 3);
        assert_eq!(storage.cluster_size(1), 3);
        assert_eq!(storage.centroids.len(), 4); // 2 * dim
        assert_eq!(storage.centroids_norm.len(), 2);
    }

    #[test]
    fn test_storage_accessors() {
        let data = sample_data_2d();
        let dim = 2;
        let n = 6;
        let nlist = 2;

        let centroids = vec![0.5, 0.0, -0.5, 0.0];
        let assignments = vec![0, 0, 1, 1, 0, 1];
        let encoder = RaBitQEncoder::new(dim, Dist::SquaredEuclidean, 42);

        let storage =
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder, 0)
                .unwrap();

        let centroid_0 = storage.centroid(0);
        assert_eq!(centroid_0.len(), dim);
        assert_abs_diff_eq!(centroid_0[0], 0.5, epsilon = 1e-5);

        let indices_0 = storage.cluster_vector_indices(0);
        assert_eq!(indices_0.len(), 3);

        // 3 vectors is one partial block, padded out to BLOCK lanes.
        assert_eq!(storage.cluster_blocked(0).n_blocks, 1);
        assert_eq!(storage.cluster_blocked(0).data.len(), storage.n_bytes * 32);
        assert_eq!(
            unpack_rabitq_blocked(
                &storage.cluster_blocked(0).data,
                3,
                storage.n_bytes,
                storage.blocked_arch
            )
            .len(),
            3
        );
    }

    #[test]
    fn test_quantiser_creation_euclidean() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, Some(2), 0, 42).unwrap();

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.n_vectors(), 6);
        assert_eq!(quantiser.encoder.dim, 2);
    }

    #[test]
    fn test_quantiser_creation_cosine() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Cosine, Some(2), 0, 42).unwrap();

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.encoder.metric, Dist::Cosine);
    }

    #[test]
    fn test_quantiser_encode_query() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);
        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, Some(2), 0, 42).unwrap();

        let query = vec![0.8, 0.2];
        let encoded = quantiser.encoder.encode_query(&query, 0).unwrap();

        assert_eq!(encoded.lut.n_byte_groups, quantiser.encoder.n_bytes);
        assert_eq!(encoded.rotated.len(), quantiser.encoder.padded_dim);
    }

    #[test]
    fn test_quantiser_default_nlist() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, None, 0, 42).unwrap();

        // Should default to 0.5 * sqrt(6) ≈ 1.22, ceiled and clamped
        assert!(quantiser.n_clusters() >= 1);
    }

    #[test]
    fn test_encode_zero_residual() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let centroid = vec.clone();

        let (_, dist, _) = encoder.encode_vector(&vec, &centroid).unwrap();

        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-5);
    }
}
