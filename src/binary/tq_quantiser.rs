//! TurboQuant encoder, query, flat storage, and quantiser.
//!
//! Each unit-normalised data vector is rotated by a fixed random
//! orthogonal matrix and scalar-quantised against the Lloyd-Max codebook
//! for `Beta((d-1)/2, (d-1)/2)`. Codes are stored in bit-plane format:
//! `bits` planes of `dim/8` bytes each per vector. The bit-plane layout
//! is later re-packed for SIMD scoring (see `tq_pack.rs`).

use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::iter::Sum;

use crate::binary::tq_codebook::codebook;
use crate::prelude::*;

//////////////////////
// TurboQuantQuery //
//////////////////////

/// Encoded query for TurboQuant scoring.
///
/// The query is unit-normalised and rotated by the encoder's rotation
/// matrix. The original L2 norm is retained for Euclidean reconstruction.
pub struct TurboQuantQuery<T> {
    /// Rotated, unit-normalised query (length = dim).
    pub q_rot: Vec<T>,
    /// L2 norm of the original query (used for Euclidean reconstruction).
    pub query_norm: T,
}

////////////////////////
// TurboQuantEncoder //
////////////////////////

/// Pure encoding logic for TurboQuant.
pub struct TurboQuantEncoder<T> {
    /// Rotation matrix, dim × dim, row-major.
    pub rotation: Vec<T>,
    /// Lloyd-Max boundaries (length = `2^bits - 1`).
    pub boundaries: Vec<T>,
    /// Lloyd-Max reconstruction levels (length = `2^bits`).
    pub levels: Vec<T>,
    /// Dimensionality.
    pub dim: usize,
    /// Bits per coordinate (2, 3, or 4).
    pub bits: usize,
    /// Bytes per packed vector (`bits * dim / 8`).
    pub bytes_per_vec: usize,
    /// Distance metric.
    pub metric: Dist,
}

impl<T> TurboQuantEncoder<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + SimdDistance,
{
    /// Create encoder with random orthogonal rotation and Lloyd-Max codebook.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality (must be a multiple of 8)
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `metric` - Distance metric
    /// * `seed` - Random seed for the rotation matrix
    pub fn new(dim: usize, bits: usize, metric: Dist, seed: u64) -> Self {
        assert!((2..=4).contains(&bits), "bits must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        let rotation = Self::generate_random_orthogonal(dim, seed);
        let (boundaries, levels) = codebook::<T>(bits, dim);
        let bytes_per_vec = bits * dim / 8;

        Self {
            rotation,
            boundaries,
            levels,
            dim,
            bits,
            bytes_per_vec,
            metric,
        }
    }

    /// Encode a single vector.
    ///
    /// ### Returns
    ///
    /// `(packed_bit_plane_codes, l2_norm)`
    #[inline]
    pub fn encode_vector(&self, vec: &[T]) -> (Vec<u8>, T) {
        let mut packed = vec![0u8; self.bytes_per_vec];
        let norm = self.encode_vector_into(vec, &mut packed);
        (packed, norm)
    }

    /// Encode a single vector into a caller-owned buffer.
    ///
    /// ### Returns
    ///
    /// L2 norm of the input.
    #[inline]
    pub fn encode_vector_into(&self, vec: &[T], out: &mut [u8]) -> T {
        assert_eq!(vec.len(), self.dim, "vector length must equal dim");
        assert_eq!(
            out.len(),
            self.bytes_per_vec,
            "output buffer must be bytes_per_vec"
        );

        let norm = compute_l2_norm(vec);

        let unit: Vec<T> = if norm > T::epsilon() {
            let inv = T::one() / norm;
            vec.iter().map(|&x| x * inv).collect()
        } else {
            vec![T::zero(); self.dim]
        };

        let rotated = self.apply_rotation(&unit);

        out.fill(0);
        let bytes_per_plane = self.dim / 8;
        for d in 0..self.dim {
            // Code = number of boundaries strictly below `rotated[d]`.
            // Boundaries are sorted ascending; linear scan is fine for
            // ≤ 15 boundaries and predictable for the branch predictor.
            let mut code: u8 = 0;
            for &b in &self.boundaries {
                if rotated[d] > b {
                    code += 1;
                }
            }
            // Bit-plane layout: bit p of `code` lands in plane p of byte
            // (d/8) at position 7-(d%8). High-bit-first ordering matches
            // what the SIMD re-pack expects.
            let byte_pos = d / 8;
            let bit_mask = 1u8 << (7 - (d % 8));
            for p in 0..self.bits {
                if code & (1 << p) != 0 {
                    out[p * bytes_per_plane + byte_pos] |= bit_mask;
                }
            }
        }

        norm
    }

    /// Encode a query: unit-normalise, rotate, retain original norm.
    #[inline]
    pub fn encode_query(&self, query: &[T]) -> TurboQuantQuery<T> {
        assert_eq!(query.len(), self.dim, "query length must equal dim");

        let query_norm = compute_l2_norm(query);

        let q_unit: Vec<T> = if query_norm > T::epsilon() {
            let inv = T::one() / query_norm;
            query.iter().map(|&x| x * inv).collect()
        } else {
            query.to_vec()
        };

        let q_rot = self.apply_rotation(&q_unit);

        TurboQuantQuery { q_rot, query_norm }
    }

    /// Apply the rotation matrix to a vector: `out = R · vec`.
    #[inline]
    pub fn apply_rotation(&self, vec: &[T]) -> Vec<T> {
        let mut rotated = vec![T::zero(); self.dim];
        for i in 0..self.dim {
            let row = &self.rotation[i * self.dim..(i + 1) * self.dim];
            rotated[i] = T::dot_simd(row, vec);
        }
        rotated
    }

    /// Generate a deterministic random orthogonal matrix via QR.
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

    /// Memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.rotation.capacity() * std::mem::size_of::<T>()
            + self.boundaries.capacity() * std::mem::size_of::<T>()
            + self.levels.capacity() * std::mem::size_of::<T>()
    }
}

////////////////////////
// TurboQuantStorage //
////////////////////////

/// Flat (non-clustered) storage of TurboQuant-encoded vectors.
///
/// Codes are stored in bit-plane format, vectors in original order.
/// Cluster-aware storage for IVF lives in `tq_ivf.rs`.
pub struct TurboQuantStorage<T> {
    /// Bit-plane packed codes, `n × bytes_per_vec` bytes total.
    pub packed_codes: Vec<u8>,
    /// L2 norms of the original vectors.
    pub norms: Vec<T>,
    /// Dimensionality.
    pub dim: usize,
    /// Bits per coordinate.
    pub bits: usize,
    /// Bytes per packed vector.
    pub bytes_per_vec: usize,
    /// Number of vectors.
    pub n: usize,
}

impl<T: Float + FromPrimitive> TurboQuantStorage<T> {
    /// Get the packed codes for vector `idx`.
    #[inline]
    pub fn vector_packed(&self, idx: usize) -> &[u8] {
        let start = idx * self.bytes_per_vec;
        &self.packed_codes[start..start + self.bytes_per_vec]
    }

    /// Number of stored vectors.
    #[inline]
    pub fn n_vectors(&self) -> usize {
        self.n
    }

    /// Memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.packed_codes.capacity()
            + self.norms.capacity() * std::mem::size_of::<T>()
    }
}

//////////////////////////
// TurboQuantQuantiser //
//////////////////////////

/// TurboQuant quantiser: encoder + flat storage of the encoded data.
pub struct TurboQuantQuantiser<T> {
    /// The TurboQuant encoder
    pub encoder: TurboQuantEncoder<T>,
    /// The TurboQuant decoder
    pub storage: TurboQuantStorage<T>,
}

impl<T> TurboQuantQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance,
{
    /// Build the quantiser by encoding all rows of `data`.
    pub fn new(data: MatRef<T>, metric: &Dist, bits: usize, seed: u64) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        let mut data_flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data_flat.push(data[(i, j)]);
            }
        }

        let encoder = TurboQuantEncoder::new(dim, bits, *metric, seed);
        let bytes_per_vec = encoder.bytes_per_vec;

        let mut packed_codes = vec![0u8; n * bytes_per_vec];
        let mut norms = vec![T::zero(); n];

        packed_codes
            .par_chunks_mut(bytes_per_vec)
            .zip(norms.par_iter_mut())
            .zip(data_flat.par_chunks(dim))
            .for_each(|((packed_slice, norm_out), data_row)| {
                *norm_out = encoder.encode_vector_into(data_row, packed_slice);
            });

        let storage = TurboQuantStorage {
            packed_codes,
            norms,
            dim,
            bits,
            bytes_per_vec,
            n,
        };

        Self { encoder, storage }
    }

    /// Encode a query.
    #[inline]
    pub fn encode_query(&self, query: &[T]) -> TurboQuantQuery<T> {
        self.encoder.encode_query(query)
    }

    /// Number of stored vectors.
    pub fn n_vectors(&self) -> usize {
        self.storage.n
    }

    /// Memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.encoder.memory_usage_bytes() + self.storage.memory_usage_bytes()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_encoder_creation_4bit() {
        let enc = TurboQuantEncoder::<f32>::new(64, 4, Dist::SquaredEuclidean, 42);
        assert_eq!(enc.dim, 64);
        assert_eq!(enc.bits, 4);
        assert_eq!(enc.bytes_per_vec, 32);
        assert_eq!(enc.boundaries.len(), 15);
        assert_eq!(enc.levels.len(), 16);
    }

    #[test]
    fn test_encoder_creation_2bit_3bit() {
        let e2 = TurboQuantEncoder::<f32>::new(64, 2, Dist::SquaredEuclidean, 42);
        assert_eq!(e2.bytes_per_vec, 16);
        assert_eq!(e2.levels.len(), 4);

        let e3 = TurboQuantEncoder::<f32>::new(64, 3, Dist::SquaredEuclidean, 42);
        assert_eq!(e3.bytes_per_vec, 24);
        assert_eq!(e3.levels.len(), 8);
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 16;
        let enc = TurboQuantEncoder::<f64>::new(dim, 4, Dist::SquaredEuclidean, 42);
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f64;
                for k in 0..dim {
                    dot += enc.rotation[i * dim + k] * enc.rotation[j * dim + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(dot, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rotation_deterministic() {
        let e1 = TurboQuantEncoder::<f32>::new(16, 4, Dist::SquaredEuclidean, 42);
        let e2 = TurboQuantEncoder::<f32>::new(16, 4, Dist::SquaredEuclidean, 42);
        assert_eq!(e1.rotation, e2.rotation);
    }

    #[test]
    fn test_encode_vector_norm() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42);
        let v = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (packed, norm) = enc.encode_vector(&v);
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-5);
        assert_eq!(packed.len(), enc.bytes_per_vec);
    }

    #[test]
    fn test_encode_zero_vector() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42);
        let v = vec![0.0_f32; 8];
        let (_, norm) = enc.encode_vector(&v);
        assert_abs_diff_eq!(norm, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_encode_query_normalises() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::Cosine, 42);
        let q = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let eq = enc.encode_query(&q);
        assert_abs_diff_eq!(eq.query_norm, 5.0, epsilon = 1e-5);
        let rot_norm: f32 = eq.q_rot.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(rot_norm, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_query_zero() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42);
        let q = vec![0.0_f32; 8];
        let eq = enc.encode_query(&q);
        assert_abs_diff_eq!(eq.query_norm, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_encode_roundtrip_low_error() {
        // Encode many pseudo-random unit vectors, decode via bit-plane
        // walk + level lookup, check mean cosine similarity in rotated
        // space (rotation is isometric, so it equals original-space cosine).
        let dim = 64;
        let bits = 4;
        let enc = TurboQuantEncoder::<f32>::new(dim, bits, Dist::Cosine, 42);
        let bytes_per_plane = dim / 8;

        let mut total_sim = 0.0f32;
        let n_trials = 50;
        for trial in 0..n_trials {
            let v: Vec<f32> = (0..dim)
                .map(|i| ((i as f32) * 0.123 + 0.7 + trial as f32 * 0.317).sin())
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let unit: Vec<f32> = v.iter().map(|x| x / norm).collect();

            let (packed, _) = enc.encode_vector(&unit);

            let rotated_unit = enc.apply_rotation(&unit);
            let mut decoded_rot = vec![0.0f32; dim];
            for d in 0..dim {
                let byte_pos = d / 8;
                let bit_mask = 1u8 << (7 - (d % 8));
                let mut code = 0u8;
                for p in 0..bits {
                    if packed[p * bytes_per_plane + byte_pos] & bit_mask != 0 {
                        code |= 1 << p;
                    }
                }
                decoded_rot[d] = enc.levels[code as usize];
            }

            let dot: f32 = rotated_unit
                .iter()
                .zip(decoded_rot.iter())
                .map(|(a, b)| a * b)
                .sum();
            let n_d: f32 = decoded_rot.iter().map(|x| x * x).sum::<f32>().sqrt();
            total_sim += dot / n_d;
        }
        let mean_sim = total_sim / n_trials as f32;
        assert!(mean_sim > 0.9, "mean cosine sim {mean_sim} below threshold");
    }

    #[test]
    fn test_storage_via_quantiser() {
        let n = 10;
        let dim = 16;
        let mut data = Mat::<f32>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = (i * dim + j) as f32 * 0.1;
            }
        }
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42);

        assert_eq!(q.n_vectors(), n);
        assert_eq!(q.storage.dim, dim);
        assert_eq!(q.storage.bits, 4);
        assert_eq!(q.storage.bytes_per_vec, 8);
        assert_eq!(q.storage.packed_codes.len(), n * 8);
        assert_eq!(q.storage.norms.len(), n);
    }

    #[test]
    fn test_quantiser_norms_match_input() {
        let n = 5;
        let dim = 16;
        let mut data = Mat::<f32>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = (i * dim + j) as f32 * 0.1;
            }
        }
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42);

        for i in 0..n {
            let row: Vec<f32> = (0..dim).map(|j| data[(i, j)]).collect();
            let expected_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_abs_diff_eq!(q.storage.norms[i], expected_norm, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_vector_packed_slice_length() {
        let n = 4;
        let dim = 16;
        let data = Mat::<f32>::zeros(n, dim);
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42);
        for i in 0..n {
            assert_eq!(q.storage.vector_packed(i).len(), q.storage.bytes_per_vec);
        }
    }

    #[test]
    #[should_panic(expected = "bits must be 2, 3, or 4")]
    fn test_invalid_bits() {
        let _ = TurboQuantEncoder::<f32>::new(8, 5, Dist::SquaredEuclidean, 42);
    }

    #[test]
    #[should_panic(expected = "dim must be a multiple of 8")]
    fn test_invalid_dim() {
        let _ = TurboQuantEncoder::<f32>::new(7, 4, Dist::SquaredEuclidean, 42);
    }
}
