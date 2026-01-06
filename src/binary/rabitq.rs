use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::iter::Sum;

use crate::binary::dist_binary::*;
use crate::utils::dist::*;
use crate::utils::ivf_utils::*;

/////////////
// Helpers //
/////////////

const RABITQ_K_MEANS_ITER: usize = 30;

/////////////////
// RaBitQQuery //
/////////////////

/// Encoded query for RaBitQ distance estimation
///
/// ### Fields
///
/// * `quantised` - Int4 quantised values (one per dimension, stored as u8)
/// * `dist_to_centroid` - Distance from query to centroid
/// * `lower` - Lower bound used in quantisation
/// * `width` - Bucket width used in quantisation
/// * `sum_quantised` - Sum of all quantised values
#[repr(C)]
pub struct RaBitQQuery<T> {
    pub quantised: Vec<u8>,
    pub dist_to_centroid: T,
    pub lower: T,
    pub width: T,
    pub sum_quantised: u32,
}

///////////////////
// RaBitQEncoder //
///////////////////

/// Pure encoding logic for RaBitQ
///
/// ### Fields
///
/// * `rotation` - The rotation matrix
/// * `dim` - Dimensions of the encode
/// * `n_bytes` - Number of bytes
/// * `metric` - Distance metric to use
pub struct RaBitQEncoder<T> {
    pub rotation: Vec<T>,
    pub dim: usize,
    pub n_bytes: usize,
    pub metric: Dist,
}

impl<T> RaBitQEncoder<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    /// Create encoder with random orthogonal rotation
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensions of the data set
    /// * `metric` - Distance metric to use
    /// * `seed` - Random seed to use
    pub fn new(dim: usize, metric: Dist, seed: u64) -> Self {
        let rotation = Self::generate_random_orthogonal(dim, seed);
        let n_bytes = dim.div_ceil(8);
        Self {
            rotation,
            dim,
            n_bytes,
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
    /// The `(binarised code, dist to centroid, correction for the dot product)`
    #[inline]
    pub fn encode_vector(&self, vec: &[T], centroid: &[T]) -> (Vec<u8>, T, T) {
        // Compute residual
        let res: Vec<T> = vec
            .iter()
            .zip(centroid.iter())
            .map(|(&v, &c)| v - c)
            .collect();

        let dist_to_centroid = compute_norm(&res);

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
        for d in 0..self.dim {
            if v_c_rotated[d] >= T::zero() {
                binary[d / 8] |= 1u8 << (d % 8);
            }
        }

        // Dot correction: L1 norm of rotated unit residual
        let dot_correction: T = v_c_rotated
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |a, b| a + b);

        (binary, dist_to_centroid, dot_correction)
    }

    /// Encode a query vector relative to a specific cluster
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
    pub fn encode_query(&self, query: &[T], centroid: &[T]) -> RaBitQQuery<T> {
        // Normalise for cosine if needed
        let query_norm: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_norm(query);
                if norm > T::epsilon() {
                    query.iter().map(|&x| x / norm).collect()
                } else {
                    query.to_vec()
                }
            }
            Dist::Euclidean => query.to_vec(),
        };

        // Residual relative to centroid
        let res: Vec<T> = query_norm
            .iter()
            .zip(centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        let dist_to_centroid = compute_norm(&res);

        // Normalise residual
        let q_c: Vec<T> = if dist_to_centroid > T::epsilon() {
            res.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); self.dim]
        };

        // Apply rotation
        let q_c_rotated = self.apply_rotation(&q_c);

        // Scalar quantise to int4 (0-15)
        let (mut lower, mut upper) = (q_c_rotated[0], q_c_rotated[0]);
        for d in 1..self.dim {
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

        let mut quantised = vec![0u8; self.dim];
        let mut sum_quantised: u32 = 0;

        for d in 0..self.dim {
            let val = ((q_c_rotated[d] - lower) / width)
                .round()
                .to_u8()
                .unwrap_or(0)
                .min(15);
            quantised[d] = val;
            sum_quantised += val as u32;
        }

        RaBitQQuery {
            quantised,
            dist_to_centroid,
            lower,
            width,
            sum_quantised,
        }
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
    fn apply_rotation(&self, vec: &[T]) -> Vec<T> {
        let mut rotated = vec![T::zero(); self.dim];
        for i in 0..self.dim {
            let base = i * self.dim;
            let mut sum = T::zero();
            for j in 0..self.dim {
                sum = sum + self.rotation[base + j] * vec[j];
            }
            rotated[i] = sum;
        }
        rotated
    }

    /// Generate a random orthogonal matrix
    ///
    /// ### Params
    ///
    /// * `dim` - The dimensions of the rotation matrix
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// A flattened orthogonal rotation matrix
    fn generate_random_orthogonal(dim: usize, seed: u64) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut mat = Mat::<T>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let val: f64 = rng.sample(StandardNormal);
                mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }

        let qr = mat.as_ref().qr();
        let q = qr.compute_Q();

        let mut rotation = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                rotation.push(q[(i, j)]);
            }
        }
        rotation
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.rotation.capacity() * std::mem::size_of::<T>()
    }
}

///////////////////
// RaBitQStorage //
///////////////////

/// CSR-layout storage for RaBitQ encoded vectors
///
/// ### Fields
///
/// * `centroids` - the centroids of the data, nlist * dim, flattened
/// * `centroids_norm` - norms of the centroids
/// * `binary_codes` - all vectors, ordered by cluster
/// * `dist_to_centroid` - Dist to centroids per vector
/// * `dot_corrections` - Corrections per vector
/// * `vector_indices` - Original indices, ordered by cluster
/// * `offsets` - cluster boundaries, len = nlist + 1
/// * `nlist` - Number of lists
/// * `dim` - Number of dimensions
/// * `n_bytes` - Number of bytes
pub struct RaBitQStorage<T> {
    pub centroids: Vec<T>,
    pub centroids_norm: Vec<T>,
    pub binary_codes: Vec<u8>,
    pub dist_to_centroid: Vec<T>,
    pub dot_corrections: Vec<T>,
    pub vector_indices: Vec<usize>,
    pub offsets: Vec<usize>,
    pub nlist: usize,
    pub dim: usize,
    pub n_bytes: usize,
}

impl<T: Float + FromPrimitive> RaBitQStorage<T> {
    /// Create empty storage with given capacity
    ///
    /// ### Params
    ///
    /// * `nlist` - Number of lists
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality of the data
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn with_capacity(nlist: usize, n: usize, dim: usize) -> Self {
        let n_bytes = dim.div_ceil(8);
        Self {
            centroids: Vec::with_capacity(nlist * dim),
            centroids_norm: Vec::with_capacity(nlist),
            binary_codes: Vec::with_capacity(n * n_bytes),
            dist_to_centroid: Vec::with_capacity(n),
            dot_corrections: Vec::with_capacity(n),
            vector_indices: Vec::with_capacity(n),
            offsets: vec![0; nlist + 1],
            nlist,
            dim,
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
    pub fn cluster_dist_to_centroid(&self, cluster_idx: usize) -> &[T] {
        let start = self.offsets[cluster_idx];
        let end = self.offsets[cluster_idx + 1];
        &self.dist_to_centroid[start..end]
    }

    /// Get dot_corrections slice for cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` Index position of the cluster
    ///
    /// ### Returns
    ///
    /// The dot corrections for every vector in this cluster
    #[inline]
    pub fn cluster_dot_corrections(&self, cluster_idx: usize) -> &[T] {
        let start = self.offsets[cluster_idx];
        let end = self.offsets[cluster_idx + 1];
        &self.dot_corrections[start..end]
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
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.binary_codes.capacity()
            + self.dist_to_centroid.capacity() * std::mem::size_of::<T>()
            + self.dot_corrections.capacity() * std::mem::size_of::<T>()
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
) -> RaBitQStorage<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + Sum,
{
    let n_bytes = dim.div_ceil(8);

    // Compute centroid norms
    let centroids_norm: Vec<T> = (0..nlist)
        .map(|i| compute_norm(&centroids[i * dim..(i + 1) * dim]))
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

    // Allocate storage
    let mut storage = RaBitQStorage {
        centroids: centroids.to_vec(),
        centroids_norm,
        binary_codes: vec![0u8; n * n_bytes],
        dist_to_centroid: vec![T::zero(); n],
        dot_corrections: vec![T::zero(); n],
        vector_indices: vec![0usize; n],
        offsets: offsets.clone(),
        nlist,
        dim,
        n_bytes,
    };

    // Track insertion position per cluster
    let mut insert_pos = offsets[..nlist].to_vec();

    // Encode and insert each vector
    for vec_idx in 0..n {
        let cluster_idx = assignments[vec_idx];
        let pos = insert_pos[cluster_idx];
        insert_pos[cluster_idx] += 1;

        let vec = &data[vec_idx * dim..(vec_idx + 1) * dim];
        let centroid = &centroids[cluster_idx * dim..(cluster_idx + 1) * dim];

        let (binary, dist, dot_corr) = encoder.encode_vector(vec, centroid);

        let byte_start = pos * n_bytes;
        storage.binary_codes[byte_start..byte_start + n_bytes].copy_from_slice(&binary);
        storage.dist_to_centroid[pos] = dist;
        storage.dot_corrections[pos] = dot_corr;
        storage.vector_indices[pos] = vec_idx;
    }

    storage
}

/////////////////////
// RaBitQQuantiser //
/////////////////////

/// RaBitQ quantiser using CSR storage
///
/// ### Fields
///
/// * `encoder` - The encoder structure
/// * `storage` - The storage structure
pub struct RaBitQQuantiser<T> {
    pub encoder: RaBitQEncoder<T>,
    pub storage: RaBitQStorage<T>,
}

impl<T> RaBitQQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
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
    pub fn new(data: MatRef<T>, metric: &Dist, n_clusters: Option<usize>, seed: usize) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let k = n_clusters
            .unwrap_or_else(|| ((n as f64).sqrt() * 0.5).ceil() as usize)
            .max(1)
            .min(n);

        // Flatten data, normalise for cosine
        let mut data_flat = Vec::with_capacity(n * dim);
        let mut data_norms = Vec::with_capacity(n);

        for i in 0..n {
            let row = data.row(i);
            let vec: Vec<T> = row.iter().cloned().collect();
            let norm = compute_norm(&vec);
            data_norms.push(norm);

            match metric {
                Dist::Cosine => {
                    if norm > T::epsilon() {
                        data_flat.extend(vec.iter().map(|&x| x / norm));
                    } else {
                        data_flat.extend(vec);
                    }
                }
                Dist::Euclidean => {
                    data_flat.extend(vec);
                }
            }
        }

        let cluster_norms = if matches!(metric, Dist::Cosine) {
            vec![T::one(); n]
        } else {
            data_norms
        };

        // Train centroids
        let centroids_flat = train_centroids(
            &data_flat,
            dim,
            n,
            k,
            metric,
            RABITQ_K_MEANS_ITER,
            seed,
            false,
        );

        let centroid_norms: Vec<T> = (0..k)
            .map(|c| {
                let cent = &centroids_flat[c * dim..(c + 1) * dim];
                compute_norm(cent)
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
        let encoder = RaBitQEncoder::new(dim, *metric, seed as u64);

        // Build CSR storage
        let storage = build_rabitq_storage(
            &data_flat,
            dim,
            n,
            &centroids_flat,
            k,
            &assignments,
            &encoder,
        );

        Self { encoder, storage }
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
    pub fn encode_query(&self, query: &[T], cluster_idx: usize) -> RaBitQQuery<T> {
        let centroid = self.storage.centroid(cluster_idx);
        self.encoder.encode_query(query, centroid)
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
    T: Float + FromPrimitive + Sum,
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

    fn sample_data_2d() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.5,
        ]
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Euclidean, 42);
        assert_eq!(encoder.dim, 4);
        assert_eq!(encoder.n_bytes, 1);
        assert_eq!(encoder.rotation.len(), 16);
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 8;
        let encoder = RaBitQEncoder::<f32>::new(dim, Dist::Euclidean, 42);

        // Check R^T * R = I
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0;
                for k in 0..dim {
                    dot += encoder.rotation[i * dim + k] * encoder.rotation[j * dim + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(dot, expected, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_encode_vector_basic() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Euclidean, 42);
        let vec = vec![1.0, 0.0, 0.0, 0.0];
        let centroid = vec![0.0, 0.0, 0.0, 0.0];

        let (binary, dist, correction) = encoder.encode_vector(&vec, &centroid);

        assert_eq!(binary.len(), 1); // 4 dims = 1 byte
        assert_abs_diff_eq!(dist, 1.0, epsilon = 1e-5);
        assert!(correction > 0.0);
    }

    #[test]
    fn test_encode_vector_with_centroid() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Euclidean, 42);
        let vec = vec![2.0, 2.0, 0.0, 0.0];
        let centroid = vec![1.0, 1.0, 0.0, 0.0];

        let (_, dist, _) = encoder.encode_vector(&vec, &centroid);

        let expected_dist = (1.0f32 + 1.0f32).sqrt();
        assert_abs_diff_eq!(dist, expected_dist, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_query_int4_range() {
        let encoder = RaBitQEncoder::<f32>::new(8, Dist::Euclidean, 42);
        let query = vec![1.0; 8];
        let centroid = vec![0.0; 8];

        let encoded = encoder.encode_query(&query, &centroid);

        assert_eq!(encoded.quantised.len(), 8);
        for &val in &encoded.quantised {
            assert!(val <= 15); // int4 max value
        }
        assert_eq!(
            encoded.sum_quantised,
            encoded.quantised.iter().map(|&x| x as u32).sum::<u32>()
        );
    }

    #[test]
    fn test_encode_query_cosine_normalises() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Cosine, 42);
        let query = vec![2.0, 0.0, 0.0, 0.0]; // Will be normalised
        let centroid = vec![0.0; 4];

        let encoded = encoder.encode_query(&query, &centroid);

        // Distance should be 1.0 since normalised query - centroid has norm 1
        assert_abs_diff_eq!(encoded.dist_to_centroid, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_storage_creation() {
        let storage = RaBitQStorage::<f32>::with_capacity(10, 100, 8);
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
        let encoder = RaBitQEncoder::new(dim, Dist::Euclidean, 42);

        let storage =
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder);

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
        let encoder = RaBitQEncoder::new(dim, Dist::Euclidean, 42);

        let storage =
            build_rabitq_storage(&data, dim, n, &centroids, nlist, &assignments, &encoder);

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

        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Euclidean, Some(2), 42);

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.n_vectors(), 6);
        assert_eq!(quantiser.encoder.dim, 2);
    }

    #[test]
    fn test_quantiser_creation_cosine() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Cosine, Some(2), 42);

        assert_eq!(quantiser.n_clusters(), 2);
        assert_eq!(quantiser.encoder.metric, Dist::Cosine);
    }

    #[test]
    fn test_quantiser_encode_query() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);
        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Euclidean, Some(2), 42);

        let query = vec![0.8, 0.2];
        let encoded = quantiser.encode_query(&query, 0);

        assert_eq!(encoded.quantised.len(), 2);
        assert!(encoded.dist_to_centroid >= 0.0);
        assert!(encoded.sum_quantised <= 30); // 2 dims * 15 max
    }

    #[test]
    fn test_quantiser_default_nlist() {
        let data = sample_data_2d();
        let mat = Mat::from_fn(6, 2, |i, j| data[i * 2 + j]);

        let quantiser = RaBitQQuantiser::new(mat.as_ref(), &Dist::Euclidean, None, 42);

        // Should default to 0.5 * sqrt(6) ≈ 1.22, ceiled and clamped
        assert!(quantiser.n_clusters() >= 1);
    }

    #[test]
    fn test_encode_zero_residual() {
        let encoder = RaBitQEncoder::<f32>::new(4, Dist::Euclidean, 42);
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let centroid = vec.clone();

        let (_, dist, _) = encoder.encode_vector(&vec, &centroid);

        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-5);
    }
}
