//! Implements the quantisation approach from RaBitQ, see:
//!
//! "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//! for Approximate Nearest Neighbor Search" (Gao and Long, 2024).

use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;

use crate::binary::dist_binary::*;
use crate::binary::rotator::{RaBitQRotator, RotatorKind};
use crate::prelude::*;
use crate::utils::k_means_utils::*;

/////////////
// Helpers //
/////////////

const RABITQ_K_MEANS_ITER: usize = 30;

/////////////////
// RaBitQQuery //
/////////////////

/// Encoded query for RaBitQ distance estimation
#[repr(C)]
pub struct RaBitQQuery<T> {
    /// Bit-planes of the int4 quantised values, `RABITQ_QUERY_PLANES` planes
    /// of `n_bytes` each. See [`build_query_planes`] for the layout.
    pub planes: Vec<u8>,
    /// Bytes per plane, `padded_dim.div_ceil(8)`
    pub n_bytes: usize,
    /// Distance from query to centroid
    pub dist_to_centroid: T,
    /// Lower bound used in quantisation
    pub lower: T,
    /// Bucket width used in quantisation
    pub width: T,
    /// Sum of all quantised values
    pub sum_quantised: u32,
}

///////////////////
// RaBitQEncoder //
///////////////////

/// Encoded vector
pub type VecEncoding<T> = (Vec<u8>, T, T, u32);

/// Pure encoding logic for RaBitQ
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQEncoder<T> {
    /// The rotation applied before the sign bits are taken
    pub rotator: RaBitQRotator<T>,
    /// Dimensions of the encode
    pub dim: usize,
    /// Working dimensionality after rotation. Equal to `dim` for the dense
    /// rotation, rounded up to [`ROTATOR_PAD`](crate::binary::rotator::ROTATOR_PAD)
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
    /// The `(binarised code, dist to centroid, inverse dot correction, popcount)`
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
        let mut popcount: u32 = 0;
        for d in 0..self.padded_dim {
            if v_c_rotated[d] >= T::zero() {
                binary[d / 8] |= 1u8 << (d % 8);
                popcount += 1;
            }
        }

        // Dot correction: L1 norm of the rotated unit residual, stored
        // inverted so the query path multiplies instead of dividing. Zero
        // stands in for "underflowed", which the query path reads as a zero
        // estimated cosine, matching the old guarded divide.
        let l1: T = compute_l1_norm(&v_c_rotated);
        let dot_correction_inv = if l1 > T::from_f32(1e-6).unwrap() {
            T::one() / l1
        } else {
            T::zero()
        };

        Ok((binary, dist_to_centroid, dot_correction_inv, popcount))
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

    /// Encode a query vector relative to a specific cluster
    ///
    /// Rotates on every call. A scan that probes many clusters should rotate
    /// the query once and use
    /// [`encode_query_prerotated`](Self::encode_query_prerotated) instead.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `centroid` - The centroid against which to encode the query vector
    ///
    /// ### Returns
    ///
    /// Encoded query for distance estimation
    #[inline]
    pub fn encode_query(
        &self,
        query: &[T],
        centroid: &[T],
    ) -> Result<RaBitQQuery<T>, AnnSearchErrors> {
        self.check_dim(query.len())?;

        let query_norm = self.normalise_query(query);

        // Residual relative to centroid
        let res = T::subtract_simd(&query_norm, centroid);

        let dist_to_centroid = compute_l2_norm(&res);

        // Normalise residual
        let q_c: Vec<T> = if dist_to_centroid > T::epsilon() {
            res.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); self.dim]
        };

        // Apply rotation
        let q_c_rotated = self.apply_rotation(&q_c);

        Ok(self.finish_query(&q_c_rotated, dist_to_centroid))
    }

    /// Encode an already-rotated query against an already-rotated centroid
    ///
    /// The rotation is linear and `R` is orthogonal, so
    /// `R(q - c) / ||q - c||` equals `(Rq - Rc) / ||Rq - Rc||`. Rotating the
    /// query once per query and the centroids once at build time drops the
    /// per-cluster cost from a `dim * dim` matvec to three `O(dim)` passes,
    /// which is what dominates an IVF scan once `nprobe` grows.
    ///
    /// ### Params
    ///
    /// * `q_rot` - The rotated, metric-normalised query
    /// * `c_rot` - The rotated centroid of the target cluster
    ///
    /// ### Returns
    ///
    /// Encoded query for distance estimation
    #[inline]
    pub fn encode_query_prerotated(&self, q_rot: &[T], c_rot: &[T]) -> RaBitQQuery<T> {
        debug_assert_eq!(q_rot.len(), self.padded_dim);
        debug_assert_eq!(c_rot.len(), self.padded_dim);

        let res_rot = T::subtract_simd(q_rot, c_rot);
        let dist_to_centroid = compute_l2_norm(&res_rot);

        let q_c_rotated: Vec<T> = if dist_to_centroid > T::epsilon() {
            res_rot.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); self.padded_dim]
        };

        self.finish_query(&q_c_rotated, dist_to_centroid)
    }

    /// Quantise a rotated unit residual into the int4 query representation
    ///
    /// Shared tail of [`encode_query`](Self::encode_query) and
    /// [`encode_query_prerotated`](Self::encode_query_prerotated).
    ///
    /// ### Params
    ///
    /// * `q_c_rotated` - The rotated, unit-length query residual
    /// * `dist_to_centroid` - `||q - c||`, carried into the distance estimate
    ///
    /// ### Returns
    ///
    /// The encoded query
    #[inline]
    fn finish_query(&self, q_c_rotated: &[T], dist_to_centroid: T) -> RaBitQQuery<T> {
        // Scalar quantise to int4 (0-15)
        let (mut lower, mut upper) = (q_c_rotated[0], q_c_rotated[0]);
        for d in 1..self.padded_dim {
            if q_c_rotated[d] < lower {
                lower = q_c_rotated[d];
            }
            if q_c_rotated[d] > upper {
                upper = q_c_rotated[d];
            }
        }

        let range = upper - lower;
        let width = if range > T::epsilon() {
            range / T::from_f32(15.0).unwrap()
        } else {
            T::one()
        };

        let mut quantised = vec![0u8; self.padded_dim];
        let mut sum_quantised: u32 = 0;

        for d in 0..self.padded_dim {
            let val = ((q_c_rotated[d] - lower) / width)
                .round()
                .to_u8()
                .unwrap_or(0)
                .min(15);
            quantised[d] = val;
            sum_quantised += val as u32;
        }

        RaBitQQuery {
            planes: build_query_planes(&quantised, self.padded_dim, self.n_bytes),
            n_bytes: self.n_bytes,
            dist_to_centroid,
            lower,
            width,
            sum_quantised,
        }
    }

    /// Apply rotation to a vector
    ///
    /// Public because the scan paths rotate a query once and then encode it
    /// against many pre-rotated centroids, see
    /// [`encode_query_prerotated`](Self::encode_query_prerotated).
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

/// RaBitQPackedVector
///
/// Packed vector representation for RaBitQ encoded vectors for better cache
/// locality and reduced misses
#[repr(C)]
#[derive(Clone)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RaBitQPackedVector<T> {
    /// Distance to centroid
    pub dist_to_centroid: T,
    /// Inverse of the dot correction (`1 / L1 norm` of the rotated unit
    /// residual), or zero when that norm underflowed
    pub dot_correction_inv: T,
    /// Popcount
    pub popcount: u32,
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
    /// All vectors, ordered by cluster
    pub binary_codes: Vec<u8>,
    /// Packed vectors of distances to centroids, dot corrections, and
    /// popcounts.
    pub packed_vectors: Vec<RaBitQPackedVector<T>>,
    /// Original indices, ordered by cluster
    pub vector_indices: Vec<usize>,
    /// Cluster boundaries, len = nlist + 1
    pub offsets: Vec<usize>,
    /// Number of lists
    pub nlist: usize,
    /// Number of dimensions
    pub dim: usize,
    /// Dimensionality of the rotated frame, the stride of `centroids_rotated`
    pub padded_dim: usize,
    /// Number of bytes
    pub n_bytes: usize,
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
            binary_codes: Vec::with_capacity(n * n_bytes),
            packed_vectors: Vec::with_capacity(n),
            vector_indices: Vec::with_capacity(n),
            offsets: vec![0; nlist + 1],
            nlist,
            dim,
            padded_dim,
            n_bytes,
        }
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

    /// Get binary codes for a cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The binary codes per cluster
    #[inline]
    pub fn cluster_binary_codes(&self, cluster_idx: usize) -> &[u8] {
        let start_vec = self.offsets[cluster_idx];
        let end_vec = self.offsets[cluster_idx + 1];
        let start_byte = start_vec * self.n_bytes;
        let end_byte = end_vec * self.n_bytes;
        &self.binary_codes[start_byte..end_byte]
    }

    /// Get binary code for specific vector within cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    /// * `local_idx` - Index position of within the cluster
    ///
    /// ### Returns
    ///
    /// Slice of binarised code for that specific vector
    #[inline]
    pub fn vector_binary(&self, cluster_idx: usize, local_idx: usize) -> &[u8] {
        let cluster_start = self.offsets[cluster_idx];
        let global_pos = cluster_start + local_idx;
        let byte_start = global_pos * self.n_bytes;
        &self.binary_codes[byte_start..byte_start + self.n_bytes]
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
        let global_idx = self.offsets[cluster_idx] + local_idx;
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
        let start = self.offsets[cluster_idx];
        let end = self.offsets[cluster_idx + 1];
        &self.packed_vectors[start..end]
    }

    /// Get popcounts slice for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The popcounts for every vector in this cluster
    #[inline]
    pub fn cluster_popcounts(&self, cluster_idx: usize) -> impl Iterator<Item = u32> + '_ {
        self.cluster_packed_data(cluster_idx)
            .iter()
            .map(|v| v.popcount)
    }

    /// Get dist_to_centroid slice for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The distance to centroid slice for every vector in this cluster
    #[inline]
    pub fn cluster_dist_to_centroid(&self, cluster_idx: usize) -> impl Iterator<Item = T> + '_ {
        self.cluster_packed_data(cluster_idx)
            .iter()
            .map(|v| v.dist_to_centroid)
    }

    /// Get inverse dot_corrections slice for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The inverse dot corrections for every vector in this cluster
    #[inline]
    pub fn cluster_dot_corrections(&self, cluster_idx: usize) -> impl Iterator<Item = T> + '_ {
        self.cluster_packed_data(cluster_idx)
            .iter()
            .map(|v| v.dot_correction_inv)
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
    pub fn cluster_vector_indices(&self, cluster_idx: usize) -> &[usize] {
        let start = self.offsets[cluster_idx];
        let end = self.offsets[cluster_idx + 1];
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
        self.offsets[cluster_idx + 1] - self.offsets[cluster_idx]
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
            + self.binary_codes.capacity()
            + self.packed_vectors.capacity() * std::mem::size_of::<RaBitQPackedVector<T>>()
            + self.vector_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
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
///
/// ### Returns
///
/// The RaBitQStorage
pub fn build_rabitq_storage<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    nlist: usize,
    assignments: &[usize],
    encoder: &RaBitQEncoder<T>,
) -> Result<RaBitQStorage<T>, AnnSearchErrors>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + Sum + SimdDistance + Clone,
{
    let padded_dim = encoder.padded_dim;
    let n_bytes = encoder.n_bytes;

    // Compute centroid norms
    let centroids_norm: Vec<T> = (0..nlist)
        .map(|i| compute_l2_norm(&centroids[i * dim..(i + 1) * dim]))
        .collect();

    // Count vectors per cluster
    let mut counts = vec![0usize; nlist];
    for &a in assignments {
        counts[a] += 1;
    }

    // Build offsets
    let mut offsets = vec![0usize; nlist + 1];
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
        binary_codes: vec![0u8; n * n_bytes],
        packed_vectors: vec![
            RaBitQPackedVector {
                dist_to_centroid: T::zero(),
                dot_correction_inv: T::zero(),
                popcount: 0,
            };
            n
        ],
        vector_indices: vec![0usize; n],
        offsets: offsets.clone(),
        nlist,
        dim,
        padded_dim,
        n_bytes,
    };

    let mut insert_pos = offsets[..nlist].to_vec();

    for vec_idx in 0..n {
        let cluster_idx = assignments[vec_idx];
        let pos = insert_pos[cluster_idx];
        insert_pos[cluster_idx] += 1;

        let vec = &data[vec_idx * dim..(vec_idx + 1) * dim];
        let centroid = &centroids[cluster_idx * dim..(cluster_idx + 1) * dim];

        let (binary, dist, dot_corr, popcount) = encoder.encode_vector(vec, centroid)?;

        let byte_start = pos * n_bytes;
        storage.binary_codes[byte_start..byte_start + n_bytes].copy_from_slice(&binary);

        storage.packed_vectors[pos] = RaBitQPackedVector {
            dist_to_centroid: dist,
            dot_correction_inv: dot_corr,
            popcount,
        };

        storage.vector_indices[pos] = vec_idx;
    }

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
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        Self::with_rotator_kind(data, metric, n_clusters, None, seed)
    }

    /// Create a new RaBitQ quantiser with an explicitly chosen rotation
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data on which to train the Quantiser
    /// * `metric` - Which distance metric to use
    /// * `n_clusters` - Optional number of centroids. If not provided, defaults
    ///   to `0.5 * sqrt(n)`.
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let
    ///   [`resolve_rotator_kind`](crate::binary::rotator::resolve_rotator_kind)
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
                    row.iter_mut().for_each(|x| *x = *x / norm);
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
            false,
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
        )?;

        Ok(Self { encoder, storage })
    }

    /// Encode query relative to a cluster
    ///
    /// ### Params
    ///
    /// * `query` - The query vector
    /// * `cluster_idx` - The cluster idx against which to encode the query
    ///
    /// ### Returns
    ///
    /// The RaBitQQuery structure
    #[inline]
    pub fn encode_query(
        &self,
        query: &[T],
        cluster_idx: usize,
    ) -> Result<RaBitQQuery<T>, AnnSearchErrors> {
        let centroid = self.storage.centroid(cluster_idx);
        self.encoder.encode_query(query, centroid)
    }

    /// Encode an already-rotated query relative to a cluster
    ///
    /// Rotate the query once with
    /// [`RaBitQEncoder::apply_rotation`] and call this per probed cluster.
    ///
    /// ### Params
    ///
    /// * `q_rot` - The rotated, metric-normalised query
    /// * `cluster_idx` - The cluster idx against which to encode the query
    ///
    /// ### Returns
    ///
    /// The RaBitQQuery structure
    #[inline]
    pub fn encode_query_prerotated(&self, q_rot: &[T], cluster_idx: usize) -> RaBitQQuery<T> {
        self.encoder
            .encode_query_prerotated(q_rot, self.storage.centroid_rotated(cluster_idx))
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
        let q = RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(6), 42).unwrap();

        for c in 0..q.storage.nlist {
            let expected = q.encoder.apply_rotation(q.storage.centroid(c));
            for (got, want) in q.storage.centroid_rotated(c).iter().zip(expected.iter()) {
                assert_abs_diff_eq!(got, want, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_prerotated_encoding_agrees_with_rotate_per_cluster() {
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let data = rotation_test_data(200, 32);
            let q = RaBitQQuantiser::new(data.as_ref(), &metric, Some(6), 42).unwrap();

            let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.41).cos()).collect();
            let normalised = q.encoder.normalise_query(&query);
            let q_rot = q.encoder.apply_rotation(&normalised);

            for c in 0..q.storage.nlist {
                let slow = q.encode_query(&query, c).unwrap();
                let fast = q.encode_query_prerotated(&q_rot, c);

                // R is orthogonal, so the residual norm survives the change of
                // frame; everything downstream is derived from it.
                assert_abs_diff_eq!(slow.dist_to_centroid, fast.dist_to_centroid, epsilon = 1e-4);
                assert_abs_diff_eq!(slow.lower, fast.lower, epsilon = 1e-4);
                assert_abs_diff_eq!(slow.width, fast.width, epsilon = 1e-5);

                // What actually matters: the estimated distances agree.
                for local in 0..q.storage.cluster_size(c) {
                    assert_abs_diff_eq!(
                        q.rabitq_dist(&slow, c, local),
                        q.rabitq_dist(&fast, c, local),
                        epsilon = 1e-3
                    );
                }
            }
        }
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        assert_eq!(encoder.dim, 4);
        assert_eq!(encoder.padded_dim, 4);
        assert_eq!(encoder.n_bytes, 1);
        assert_eq!(encoder.rotator.kind(), RotatorKind::Dense);
    }

    #[test]
    fn test_encoder_pads_once_the_hadamard_rotation_kicks_in() {
        let encoder = RaBitQEncoder::<f32>::new(200, Dist::SquaredEuclidean, 42);
        assert_eq!(encoder.rotator.kind(), RotatorKind::FhtKac);
        assert_eq!(encoder.padded_dim, 256);
        assert_eq!(encoder.n_bytes, 32);

        let (binary, _, _, popcount) = encoder
            .encode_vector(&vec![1.0; 200], &vec![0.0; 200])
            .unwrap();
        assert_eq!(binary.len(), 32);
        assert!(popcount <= 256);
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

        let (binary, dist, correction, _) = encoder.encode_vector(&vec, &centroid).unwrap();

        assert_eq!(binary.len(), 1); // 4 dims = 1 byte
        assert_abs_diff_eq!(dist, 1.0, epsilon = 1e-5);
        assert!(correction > 0.0);
    }

    #[test]
    fn test_encode_vector_with_centroid() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        let vec = vec![2.0, 2.0, 0.0, 0.0];
        let centroid = vec![1.0, 1.0, 0.0, 0.0];

        let (_, dist, _, _) = encoder.encode_vector(&vec, &centroid).unwrap();

        let expected_dist = (1.0f32 + 1.0f32).sqrt();
        assert_abs_diff_eq!(dist, expected_dist, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_query_int4_range() {
        let encoder = RaBitQEncoder::<f32>::new(8, Dist::SquaredEuclidean, 42);
        let query = vec![1.0; 8];
        let centroid = vec![0.0; 8];

        let encoded = encoder.encode_query(&query, &centroid).unwrap();
        let quantised = unpack_query_planes(&encoded.planes, 8, encoded.n_bytes);

        assert_eq!(quantised.len(), 8);
        for &val in &quantised {
            assert!(val <= 15); // int4 max value
        }
        assert_eq!(
            encoded.sum_quantised,
            quantised.iter().map(|&x| x as u32).sum::<u32>()
        );
    }

    #[test]
    fn test_encode_query_cosine_normalises() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Cosine, 42);
        let query = vec![2.0, 0.0, 0.0, 0.0]; // Will be normalised
        let centroid = vec![0.0; 4];

        let encoded = encoder.encode_query(&query, &centroid).unwrap();

        // Distance should be 1.0 since normalised query - centroid has norm 1
        assert_abs_diff_eq!(encoded.dist_to_centroid, 1.0, epsilon = 1e-5);
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
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder).unwrap();

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
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder).unwrap();

        let centroid_0 = storage.centroid(0);
        assert_eq!(centroid_0.len(), dim);
        assert_abs_diff_eq!(centroid_0[0], 0.5, epsilon = 1e-5);

        let indices_0 = storage.cluster_vector_indices(0);
        assert_eq!(indices_0.len(), 3);

        let binary_0 = storage.cluster_binary_codes(0);
        assert_eq!(binary_0.len(), 3); // 3 vectors * 1 byte
    }

    #[test]
    fn test_quantiser_creation_euclidean() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, Some(2), 42).unwrap();

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.n_vectors(), 6);
        assert_eq!(quantiser.encoder.dim, 2);
    }

    #[test]
    fn test_quantiser_creation_cosine() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Cosine, Some(2), 42).unwrap();

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.encoder.metric, Dist::Cosine);
    }

    #[test]
    fn test_quantiser_encode_query() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);
        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, Some(2), 42).unwrap();

        let query = vec![0.8, 0.2];
        let encoded = quantiser.encode_query(&query, 0).unwrap();

        assert_eq!(
            unpack_query_planes(&encoded.planes, 2, encoded.n_bytes).len(),
            2
        );
        assert!(encoded.dist_to_centroid >= 0.0);
        assert!(encoded.sum_quantised <= 30); // 2 dims * 15 max
    }

    #[test]
    fn test_quantiser_default_nlist() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser =
            RaBitQQuantiser::new(mat.as_ref(), &Dist::SquaredEuclidean, None, 42).unwrap();

        // Should default to 0.5 * sqrt(6) ≈ 1.22, ceiled and clamped
        assert!(quantiser.n_clusters() >= 1);
    }

    #[test]
    fn test_encode_zero_residual() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::SquaredEuclidean, 42);
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let centroid = vec.clone();

        let (_, dist, _, _) = encoder.encode_vector(&vec, &centroid).unwrap();

        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-5);
    }
}
