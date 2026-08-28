//! TurboQuant encoder, query, flat storage, and quantiser.
//!
//! Each unit-normalised data vector is rotated by a fixed random orthogonal
//! matrix and scalar-quantised against the Lloyd-Max codebook for
//! `Beta((d-1)/2, (d-1)/2)`. Codes are stored in bit-plane format: `bits`
//! planes of `dim/8` bytes each per vector.

use faer::Mat;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::binary::turboquant::codebook::codebook;
use crate::prelude::*;

/////////////////////
// TurboQuantQuery //
/////////////////////

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

///////////////////////
// TurboQuantEncoder //
///////////////////////

/// Pure encoding logic for TurboQuant.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
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

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for TurboQuantEncoder<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

//////////////////
// Main encoder //
//////////////////

impl<T> TurboQuantEncoder<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + SimdDistance,
{
    /// Create an encoder with a random orthogonal rotation and Lloyd-Max
    /// codebook.
    ///
    /// Generates a deterministic rotation matrix from `seed` via QR
    /// decomposition and loads the Lloyd-Max boundaries and levels for the
    /// given bit width.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality (must be a multiple of 8)
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `metric` - Distance metric
    /// * `seed` - Random seed for the rotation matrix
    ///
    /// ### Returns
    ///
    /// The constructed encoder, or an error if `bits` is out of range or `dim`
    /// is not a multiple of 8.
    pub fn new(dim: usize, bits: usize, metric: Dist, seed: u64) -> Result<Self, AnnSearchErrors> {
        if !(2..=4).contains(&bits) {
            return Err(AnnSearchErrors::TQInvalidBits { n_bits: bits });
        }
        if !dim.is_multiple_of(8) {
            return Err(AnnSearchErrors::TQDimMustBe8Multiple { dims: dim });
        }

        let rotation = Self::generate_random_orthogonal(dim, seed);
        let (boundaries, levels) = codebook::<T>(bits, dim)?;
        let bytes_per_vec = bits * dim / 8;

        Ok(Self {
            rotation,
            boundaries,
            levels,
            dim,
            bits,
            bytes_per_vec,
            metric,
        })
    }

    /// Encode a single vector, allocating the output buffer.
    ///
    /// Convenience wrapper around [`encode_vector_into`] that allocates and
    /// returns the packed code buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length `dim`)
    ///
    /// ### Returns
    ///
    /// `(packed, ‖v‖, correction)` where `packed` holds the bit-plane codes,
    /// `‖v‖` is the raw L2 norm, and `correction = 1 / <u, x̂>` is the
    /// per-vector debias factor.
    #[inline]
    pub fn encode_vector(&self, vec: &[T]) -> Result<(Vec<u8>, T, T), AnnSearchErrors> {
        let mut packed = vec![0u8; self.bytes_per_vec];
        let (norm, correction) = self.encode_vector_into(vec, &mut packed)?;
        Ok((packed, norm, correction))
    }

    /// Encode a single vector into a caller-owned buffer.
    ///
    /// Unit-normalises the input, applies the rotation, and writes bit-plane
    /// packed codes into `out`. Also accumulates `<u, x̂>` to compute the
    /// per-vector debias factor.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length `dim`)
    /// * `out` - Output buffer to write packed codes into (length
    ///   `bytes_per_vec`)
    ///
    /// ### Returns
    ///
    /// `(‖v‖, correction)` where `‖v‖` is the raw L2 norm and
    /// `correction = 1 / <u, x̂>` is the per-vector debias factor.
    #[inline]
    pub fn encode_vector_into(&self, vec: &[T], out: &mut [u8]) -> Result<(T, T), AnnSearchErrors> {
        self.check_dim(vec.len())?;

        if out.len() != self.bytes_per_vec {
            return Err(AnnSearchErrors::TQBufferUnequalBytesPerVec {
                bytes_per_vec: self.bytes_per_vec,
                len: out.len(),
            });
        }

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
        let mut dot_self = T::zero();
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
            // Accumulate <u, x̂> using the level we just chose for this coord.
            dot_self = dot_self + rotated[d] * self.levels[code as usize];
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

        let correction = if norm > T::epsilon() && dot_self > T::epsilon() {
            T::one() / dot_self
        } else {
            T::zero()
        };
        Ok((norm, correction))
    }

    /// Encode a query: unit-normalise, rotate, and retain the original L2 norm.
    ///
    /// The original norm is preserved so callers can reconstruct Euclidean
    /// distances from the approximate inner product scores.
    ///
    /// ### Params
    ///
    /// * `query` - Input query vector (length `dim`)
    ///
    /// ### Returns
    ///
    /// A [`TurboQuantQuery`] holding the rotated unit query and the original
    /// norm.
    #[inline]
    pub fn encode_query(&self, query: &[T]) -> Result<TurboQuantQuery<T>, AnnSearchErrors> {
        self.check_dim(query.len())?;

        let query_norm = compute_l2_norm(query);

        let q_unit: Vec<T> = if query_norm > T::epsilon() {
            let inv = T::one() / query_norm;
            query.iter().map(|&x| x * inv).collect()
        } else {
            query.to_vec()
        };

        let q_rot = self.apply_rotation(&q_unit);

        Ok(TurboQuantQuery { q_rot, query_norm })
    }

    /// Apply the rotation matrix to a vector: `out = R · vec`.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length `dim`)
    ///
    /// ### Returns
    ///
    /// The rotated vector (length `dim`).
    #[inline]
    pub fn apply_rotation(&self, vec: &[T]) -> Vec<T> {
        let mut rotated = vec![T::zero(); self.dim];
        for i in 0..self.dim {
            let row = &self.rotation[i * self.dim..(i + 1) * self.dim];
            rotated[i] = T::dot_simd(row, vec);
        }
        rotated
    }

    /// Generate a deterministic random orthogonal matrix via QR decomposition.
    ///
    /// Fills a `dim × dim` matrix with standard-normal samples seeded by
    /// `seed`, then extracts the Q factor.
    ///
    /// ### Params
    ///
    /// * `dim` - Matrix dimension
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// Row-major flat representation of the `dim × dim` orthogonal matrix
    /// (length `dim * dim`).
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

    /// Heap memory used by this encoder in bytes.
    ///
    /// ### Returns
    ///
    /// Total bytes occupied by the encoder's heap allocations plus its
    /// stack-size footprint.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.rotation.capacity() * std::mem::size_of::<T>()
            + self.boundaries.capacity() * std::mem::size_of::<T>()
            + self.levels.capacity() * std::mem::size_of::<T>()
    }
}

///////////////////////
// TurboQuantStorage //
///////////////////////

/// Flat (non-clustered) storage of TurboQuant-encoded vectors.
///
/// Codes are stored in bit-plane format, vectors in original order.
/// Cluster-aware storage for IVF lives in `tq_ivf.rs`.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct TurboQuantStorage<T> {
    /// Bit-plane packed codes, `n × bytes_per_vec` bytes total.
    pub packed_codes: Vec<u8>,
    /// L2 norms of the original vectors.
    pub norms: Vec<T>,
    /// Per-vector debias factors `1 / <u, x̂>`.
    pub corrections: Vec<T>,
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
    /// Return the packed bit-plane codes for one stored vector.
    ///
    /// ### Params
    ///
    /// * `idx` - Vector index (`0..n`)
    ///
    /// ### Returns
    ///
    /// Byte slice of length `bytes_per_vec` containing the bit-plane codes
    /// for vector `idx`.
    #[inline]
    pub fn vector_packed(&self, idx: usize) -> &[u8] {
        let start = idx * self.bytes_per_vec;
        &self.packed_codes[start..start + self.bytes_per_vec]
    }

    /// Number of stored vectors.
    ///
    /// ### Returns
    ///
    /// The count of encoded vectors held in this storage.
    #[inline]
    pub fn n_vectors(&self) -> usize {
        self.n
    }

    /// Heap memory used by this storage in bytes.
    ///
    /// ### Returns
    ///
    /// Total bytes occupied by the storage's heap allocations plus its
    /// stack-size footprint.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.packed_codes.capacity()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.corrections.capacity() * std::mem::size_of::<T>()
    }
}

/////////////////////////
// TurboQuantQuantiser //
/////////////////////////

/// TurboQuant quantiser: encoder + flat storage of the encoded data.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct TurboQuantQuantiser<T> {
    /// The TurboQuant encoder
    pub encoder: TurboQuantEncoder<T>,
    /// The TurboQuant decoder
    pub storage: TurboQuantStorage<T>,
}

impl<T> TurboQuantQuantiser<T>
where
    T: AnnSearchFloat,
{
    /// Build the quantiser by encoding all rows of `data` in parallel.
    ///
    /// Constructs a [`TurboQuantEncoder`] from the given parameters, then
    /// encodes every row of `data` via Rayon, collecting packed codes, norms,
    /// and debias corrections into flat storage.
    ///
    /// ### Params
    ///
    /// * `data` - Input data as samples x features, see [`AnnMatrix`]
    /// * `metric` - Distance metric
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `seed` - Random seed forwarded to the encoder's rotation matrix
    ///
    /// ### Returns
    ///
    /// The constructed quantiser, or an error if the encoder cannot be built
    /// or any row fails to encode.
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        bits: usize,
        seed: u64,
    ) -> Result<Self, AnnSearchErrors> {
        let (data_flat, n, dim) = data.into_row_major();

        let encoder = TurboQuantEncoder::new(dim, bits, *metric, seed)?;
        let bytes_per_vec = encoder.bytes_per_vec;

        let mut packed_codes = vec![0u8; n * bytes_per_vec];
        let mut norms = vec![T::zero(); n];
        let mut corrections = vec![T::zero(); n];

        packed_codes
            .par_chunks_mut(bytes_per_vec)
            .zip(norms.par_iter_mut())
            .zip(corrections.par_iter_mut())
            .zip(data_flat.par_chunks(dim))
            .try_for_each(
                |(((packed_slice, norm_out), correction_out), data_row)| -> Result<(), AnnSearchErrors> {
                    let (norm, correction) = encoder.encode_vector_into(data_row, packed_slice)?;
                    *norm_out = norm;
                    *correction_out = correction;
                    Ok(())
                },
            )?;

        let storage = TurboQuantStorage {
            packed_codes,
            norms,
            corrections,
            dim,
            bits,
            bytes_per_vec,
            n,
        };

        Ok(Self { encoder, storage })
    }

    /// Encode a query using the quantiser's encoder.
    ///
    /// ### Params
    ///
    /// * `query` - Input query vector (length `dim`)
    ///
    /// ### Returns
    ///
    /// A [`TurboQuantQuery`] holding the rotated unit query and the original
    /// L2 norm.
    #[inline]
    pub fn encode_query(&self, query: &[T]) -> Result<TurboQuantQuery<T>, AnnSearchErrors> {
        self.encoder.encode_query(query)
    }

    /// Number of stored vectors.
    ///
    /// ### Returns
    ///
    /// The count of encoded vectors held in the quantiser's storage.
    pub fn n_vectors(&self) -> usize {
        self.storage.n
    }

    /// Combined heap memory used by the encoder and storage in bytes.
    ///
    /// ### Returns
    ///
    /// Sum of [`TurboQuantEncoder::memory_usage_bytes`] and
    /// [`TurboQuantStorage::memory_usage_bytes`].
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
        let enc = TurboQuantEncoder::<f32>::new(64, 4, Dist::SquaredEuclidean, 42).unwrap();
        assert_eq!(enc.dim, 64);
        assert_eq!(enc.bits, 4);
        assert_eq!(enc.bytes_per_vec, 32);
        assert_eq!(enc.boundaries.len(), 15);
        assert_eq!(enc.levels.len(), 16);
    }

    #[test]
    fn test_encoder_creation_2bit_3bit() {
        let e2 = TurboQuantEncoder::<f32>::new(64, 2, Dist::SquaredEuclidean, 42).unwrap();
        assert_eq!(e2.bytes_per_vec, 16);
        assert_eq!(e2.levels.len(), 4);

        let e3 = TurboQuantEncoder::<f32>::new(64, 3, Dist::SquaredEuclidean, 42).unwrap();
        assert_eq!(e3.bytes_per_vec, 24);
        assert_eq!(e3.levels.len(), 8);
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 16;
        let enc = TurboQuantEncoder::<f64>::new(dim, 4, Dist::SquaredEuclidean, 42).unwrap();
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
        let e1 = TurboQuantEncoder::<f32>::new(16, 4, Dist::SquaredEuclidean, 42).unwrap();
        let e2 = TurboQuantEncoder::<f32>::new(16, 4, Dist::SquaredEuclidean, 42).unwrap();
        assert_eq!(e1.rotation, e2.rotation);
    }

    #[test]
    fn test_encode_vector_norm_and_correction() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42).unwrap();
        let v = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (packed, norm, correction) = enc.encode_vector(&v).unwrap();
        assert_eq!(packed.len(), enc.bytes_per_vec);

        // Raw L2 norm of (3, 4, 0, ..., 0).
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-5);

        // Recompute expected correction = 1 / <u, x̂> by mirroring the encode loop.
        let raw_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = v.iter().map(|x| x / raw_norm).collect();
        let rotated = enc.apply_rotation(&unit);
        let mut dot_self = 0.0f32;
        for d in 0..enc.dim {
            let mut code: u8 = 0;
            for &b in &enc.boundaries {
                if rotated[d] > b {
                    code += 1;
                }
            }
            dot_self += rotated[d] * enc.levels[code as usize];
        }
        let expected = 1.0 / dot_self;
        assert_abs_diff_eq!(correction, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_zero_vector() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42).unwrap();
        let v = vec![0.0_f32; 8];
        let (_, norm, correction) = enc.encode_vector(&v).unwrap();
        assert_abs_diff_eq!(norm, 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(correction, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_encode_query_normalises() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::Cosine, 42).unwrap();
        let q = vec![3.0_f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let eq = enc.encode_query(&q).unwrap();
        assert_abs_diff_eq!(eq.query_norm, 5.0, epsilon = 1e-5);
        let rot_norm: f32 = eq.q_rot.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_abs_diff_eq!(rot_norm, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_encode_query_zero() {
        let enc = TurboQuantEncoder::<f32>::new(8, 4, Dist::SquaredEuclidean, 42).unwrap();
        let q = vec![0.0_f32; 8];
        let eq = enc.encode_query(&q).unwrap();
        assert_abs_diff_eq!(eq.query_norm, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn test_encode_roundtrip_low_error() {
        // Encode many pseudo-random unit vectors, decode via bit-plane
        // walk + level lookup, check mean cosine similarity in rotated
        // space (rotation is isometric, so it equals original-space cosine).
        let dim = 64;
        let bits = 4;
        let enc = TurboQuantEncoder::<f32>::new(dim, bits, Dist::Cosine, 42).unwrap();
        let bytes_per_plane = dim / 8;

        let mut total_sim = 0.0f32;
        let n_trials = 50;
        for trial in 0..n_trials {
            let v: Vec<f32> = (0..dim)
                .map(|i| ((i as f32) * 0.123 + 0.7 + trial as f32 * 0.317).sin())
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let unit: Vec<f32> = v.iter().map(|x| x / norm).collect();

            let (packed, _, _) = enc.encode_vector(&unit).unwrap();

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
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42).unwrap();

        assert_eq!(q.n_vectors(), n);
        assert_eq!(q.storage.dim, dim);
        assert_eq!(q.storage.bits, 4);
        assert_eq!(q.storage.bytes_per_vec, 8);
        assert_eq!(q.storage.packed_codes.len(), n * 8);
        assert_eq!(q.storage.norms.len(), n);
        assert_eq!(q.storage.corrections.len(), n);
    }

    #[test]
    fn test_quantiser_norms_and_corrections_match_recomputation() {
        let n = 5;
        let dim = 16;
        let mut data = Mat::<f32>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = (i * dim + j) as f32 * 0.1;
            }
        }
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42).unwrap();
        let enc = &q.encoder;

        for i in 0..n {
            let row: Vec<f32> = (0..dim).map(|j| data[(i, j)]).collect();
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let unit: Vec<f32> = row.iter().map(|x| x / norm).collect();
            let rotated = enc.apply_rotation(&unit);
            let mut dot_self = 0.0f32;
            for d in 0..dim {
                let mut code: u8 = 0;
                for &b in &enc.boundaries {
                    if rotated[d] > b {
                        code += 1;
                    }
                }
                dot_self += rotated[d] * enc.levels[code as usize];
            }
            let expected_correction = 1.0 / dot_self;
            assert_abs_diff_eq!(q.storage.norms[i], norm, epsilon = 1e-5);
            assert_abs_diff_eq!(
                q.storage.corrections[i],
                expected_correction,
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn test_vector_packed_slice_length() {
        let n = 4;
        let dim = 16;
        let data = Mat::<f32>::zeros(n, dim);
        let q = TurboQuantQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42).unwrap();
        for i in 0..n {
            assert_eq!(q.storage.vector_packed(i).len(), q.storage.bytes_per_vec);
        }
    }

    #[test]
    fn test_invalid_bits() {
        let result = TurboQuantEncoder::<f32>::new(8, 5, Dist::SquaredEuclidean, 42);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::TQInvalidBits { n_bits: 5 })
        ));
    }

    #[test]
    fn test_invalid_dim() {
        let result = TurboQuantEncoder::<f32>::new(7, 4, Dist::SquaredEuclidean, 42);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::TQDimMustBe8Multiple { dims: 7 })
        ));
    }
}
