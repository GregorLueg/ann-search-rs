//! Uniform 8-bit vector storage.
//!
//! Codes, their integer norms and whichever precomputed term the metric needs,
//! sitting on top of [`UniformQuantiser`]. Shared by every index that stores
//! uniformly quantised vectors: the exhaustive scan, the IVF lists and the
//! graph.

use rayon::prelude::*;

use crate::prelude::*;
use crate::quantised::hnsw_quantised::codec::GraphCodec;
use crate::quantised::int_kernels::*;
use crate::quantised::uniform_quant::*;
use crate::utils::validate_dim;

///////////////
// Sq8uQuery //
///////////////

/// Query state for [`Sq8uCodec`]: the query in the same code space as the
/// database, plus whichever precomputed term its metric needs.
pub struct Sq8uQuery<T> {
    /// The query encoded to 8-bit codes.
    code: Vec<u8>,
    /// Squared integer norm of `code`, for the Euclidean path.
    norm: u32,
    /// Offset-correction term, for the cosine path. Zero otherwise.
    offset_dot: T,
}

///////////////
// Sq8uCodec //
///////////////

/// Uniformly quantised 8-bit storage with an integer distance.
///
/// ### Note
///
/// Ranking is exact whilst the code-space squared distance stays inside `T`'s
/// integer range, which for `f32` means `255^2 * dim <= 2^24`, so up to 258
/// dimensions. Past that, distances differing by one least-significant unit out
/// of millions may tie; `f64` covers any realistic dimensionality. PCA and
/// latent spaces sit well inside the `f32` bound.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "")
)]
pub struct Sq8uCodec<T>
where
    T: AnnSearchFloat,
{
    /// The calibrated quantiser.
    quantiser: UniformQuantiser<T>,
    /// Row-major codes, `n * dim` bytes.
    codes: Vec<u8>,
    /// Squared integer code norms, one per vector. Euclidean path only.
    code_norms: Vec<u32>,
    /// Offset-correction terms, one per vector. Cosine path only, else empty.
    offset_dots: Vec<T>,
    /// Number of stored vectors.
    n: usize,
    /// Dimensionality.
    dim: usize,
    /// Metric this codec was built for.
    metric: Dist,
}

impl<T> Sq8uCodec<T>
where
    T: AnnSearchFloat,
{
    /// Calibrate and encode a dataset.
    ///
    /// For cosine the rows are normalised before calibration, so the codec
    /// stores unit vectors and an inner product is the cosine similarity.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major flattened vectors of length `n * dim`
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality
    /// * `metric` - Distance metric; Manhattan is not supported
    /// * `params` - Calibration settings, `None` for the default
    ///
    /// ### Returns
    ///
    /// The encoded codec, or an error on an unsupported metric or bad
    /// calibration settings
    pub fn new(
        data: &[T],
        n: usize,
        dim: usize,
        metric: Dist,
        params: Option<UniformQuantParams>,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        // Cosine is an inner product on unit rows, so normalise before the
        // quantiser sees the data rather than carrying norms through it.
        let owned;
        let data = if metric == Dist::Cosine {
            let mut rows = data[..n * dim].to_vec();
            rows.par_chunks_mut(dim).for_each(|row| {
                let norm = T::calculate_l2_norm(row);
                if norm > T::epsilon() {
                    row.iter_mut().for_each(|x| *x = *x / norm);
                }
            });
            owned = rows;
            &owned[..]
        } else {
            &data[..n * dim]
        };

        let quantiser = UniformQuantiser::train(data, n, dim, params)?;

        let mut codes = vec![0u8; n * dim];
        codes
            .par_chunks_mut(dim)
            .zip(data.par_chunks(dim))
            .for_each(|(out, row)| quantiser.encode_into(row, out));

        let cosine = metric == Dist::Cosine;
        let code_norms: Vec<u32> = if cosine {
            Vec::new()
        } else {
            codes.par_chunks(dim).map(norm_sq_u8).collect()
        };
        let offset_dots: Vec<T> = if cosine {
            codes
                .par_chunks(dim)
                .map(|c| quantiser.offset_dot(c))
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            quantiser,
            codes,
            code_norms,
            offset_dots,
            n,
            dim,
            metric,
        })
    }

    /// Code vector of a stored point.
    ///
    /// ### Params
    ///
    /// * `id` - Index of the stored vector
    ///
    /// ### Returns
    ///
    /// Slice of length `dim`
    #[inline(always)]
    fn code(&self, id: usize) -> &[u8] {
        let start = id * self.dim;

        unsafe { self.codes.get_unchecked(start..start + self.dim) }
    }

    /// The underlying quantiser.
    ///
    /// ### Returns
    ///
    /// Reference to the calibrated quantiser
    pub fn quantiser(&self) -> &UniformQuantiser<T> {
        &self.quantiser
    }

    /// Dequantise a stored vector back to floats.
    ///
    /// Reconstruction, not the original: each coordinate is accurate to half a
    /// code level. Useful where a float vector is needed away from the hot
    /// path, such as scoring a handful of cluster centroids.
    ///
    /// ### Params
    ///
    /// * `id` - Index of the stored vector
    ///
    /// ### Returns
    ///
    /// The reconstructed vector, of length `dim`
    pub fn decode(&self, id: usize) -> Vec<T> {
        self.quantiser.decode(self.code(id))
    }

    /// Bytes held by the codec.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.quantiser.memory_usage_bytes()
            + self.codes.capacity()
            + self.code_norms.capacity() * std::mem::size_of::<u32>()
            + self.offset_dots.capacity() * std::mem::size_of::<T>()
    }
}

/////////////////////////
// GraphCodec for Sq8u //
/////////////////////////

impl<T> GraphCodec<T> for Sq8uCodec<T>
where
    T: AnnSearchFloat,
{
    type Query = Sq8uQuery<T>;

    fn n(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn encode_query(&self, query: &[T]) -> Result<Self::Query, AnnSearchErrors> {
        validate_dim(query.len(), self.dim)?;

        if self.metric == Dist::Cosine {
            let norm = T::calculate_l2_norm(query);
            let unit: Vec<T> = if norm > T::epsilon() {
                query.iter().map(|&x| x / norm).collect()
            } else {
                query.to_vec()
            };
            let mut code = vec![0u8; self.dim];
            self.quantiser.encode_into(&unit, &mut code);
            let offset_dot = self.quantiser.offset_dot(&code);
            Ok(Sq8uQuery {
                code,
                norm: 0,
                offset_dot,
            })
        } else {
            let mut code = vec![0u8; self.dim];
            self.quantiser.encode_into(query, &mut code);
            let norm = norm_sq_u8(&code);
            Ok(Sq8uQuery {
                code,
                norm,
                offset_dot: T::zero(),
            })
        }
    }

    #[inline(always)]
    fn score(&self, query: &Self::Query, id: usize) -> T {
        match self.metric {
            Dist::Cosine => {
                // Negated: the pool orders by smallest, similarity is largest.
                let ip = self.quantiser.inner_product(
                    &query.code,
                    self.code(id),
                    query.offset_dot,
                    self.offset_dots[id],
                );
                ip.neg()
            }
            _ => {
                let d =
                    sq_dist_from_dot(&query.code, self.code(id), query.norm, self.code_norms[id]);
                T::from_i64(d).unwrap()
            }
        }
    }

    #[inline(always)]
    fn score_sym(&self, a: usize, b: usize) -> T {
        match self.metric {
            Dist::Cosine => {
                let ip = self.quantiser.inner_product(
                    self.code(a),
                    self.code(b),
                    self.offset_dots[a],
                    self.offset_dots[b],
                );
                ip.neg()
            }
            _ => {
                let d = sq_dist_from_dot(
                    self.code(a),
                    self.code(b),
                    self.code_norms[a],
                    self.code_norms[b],
                );
                T::from_i64(d).unwrap()
            }
        }
    }

    #[inline]
    fn finalise(&self, score: T) -> T {
        match self.metric {
            // `score` is the negated similarity, and rows are unit length.
            Dist::Cosine => T::one() + score,
            _ => score * self.quantiser.scale_sq(),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Five clusters, each pointing in its own direction in feature space.
    /// Direction matters: a generator that varies only the magnitude leaves
    /// every row parallel once normalised, and cosine cannot separate them.
    fn blobs<T: AnnSearchFloat>(n: usize, dim: usize) -> Vec<T> {
        let mut s = 0x9E3779B9u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..n * dim)
            .map(|i| {
                let cluster = (i / dim) % 5;
                let base = if (i % dim) % 5 == cluster { 1.0 } else { 0.15 };
                T::from_f64(base + next() * 0.05).unwrap()
            })
            .collect()
    }

    fn exact_sq_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn exact_cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (na * nb)
    }

    #[test]
    fn test_manhattan_is_rejected() {
        let (n, dim) = (50, 8);
        let data = blobs::<f32>(n, dim);
        let got = Sq8uCodec::new(&data, n, dim, Dist::Manhattan, None);
        assert!(matches!(
            got,
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_euclidean_score_is_close_to_the_true_distance() {
        let (n, dim) = (300, 16);
        let data = blobs::<f32>(n, dim);
        let codec = Sq8uCodec::new(&data, n, dim, Dist::SquaredEuclidean, None).unwrap();

        // Absolute floor first. A code is accurate to half a level per
        // dimension, so the squared distance carries an error of order
        // `dim * scale^2` however close the pair is. A purely relative bound
        // would be asserting precision the method does not have.
        let scale = codec.quantiser().scale();
        let floor = dim as f32 * scale * scale;

        for i in [0usize, 7, 123, 299] {
            let q = codec.encode_query(&data[i * dim..(i + 1) * dim]).unwrap();
            for j in [1usize, 50, 200] {
                let got = codec.finalise(codec.score(&q, j));
                let want =
                    exact_sq_l2(&data[i * dim..(i + 1) * dim], &data[j * dim..(j + 1) * dim]);
                assert!(
                    (got - want).abs() <= floor + 0.05 * want,
                    "l2 {i}->{j}: got {got}, want {want}, tol {}",
                    floor + 0.05 * want
                );
            }
        }
    }

    #[test]
    fn test_cosine_score_is_close_to_the_true_distance() {
        let (n, dim) = (300, 16);
        let data = blobs::<f32>(n, dim);
        let codec = Sq8uCodec::new(&data, n, dim, Dist::Cosine, None).unwrap();

        // A unit vector's coordinates are of order `1/sqrt(dim)`, so a
        // half-level error per coordinate puts the dot product's error at
        // order `sqrt(dim) * scale`.
        let scale = codec.quantiser().scale();
        let tol = (dim as f32).sqrt() * scale;

        for i in [0usize, 11, 250] {
            let q = codec.encode_query(&data[i * dim..(i + 1) * dim]).unwrap();
            for j in [3usize, 90, 210] {
                let got = codec.finalise(codec.score(&q, j));
                let want =
                    exact_cosine(&data[i * dim..(i + 1) * dim], &data[j * dim..(j + 1) * dim]);
                assert!(
                    (got - want).abs() <= tol,
                    "cosine {i}->{j}: got {got}, want {want}, tol {tol}"
                );
            }
        }
    }

    #[test]
    fn test_query_of_a_stored_vector_scores_itself_lowest() {
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let (n, dim) = (200, 12);
            let data = blobs::<f32>(n, dim);
            let codec = Sq8uCodec::new(&data, n, dim, metric, None).unwrap();

            for i in [0usize, 42, 199] {
                let q = codec.encode_query(&data[i * dim..(i + 1) * dim]).unwrap();
                let self_score = codec.score(&q, i);
                let best =
                    (0..n)
                        .map(|j| codec.score(&q, j))
                        .fold(f32::MAX, |a, b| if b < a { b } else { a });
                // Self need not win outright. Two rows less than a
                // quantisation step apart encode to identical codes, and then
                // nothing can separate them. The tolerance is that step.
                let scale = codec.quantiser().scale();
                let tol = dim as f32 * scale * scale + (dim as f32).sqrt() * scale;
                assert!(
                    self_score <= best + tol,
                    "{metric:?}: self {self_score} vs best {best}, tol {tol}"
                );
            }
        }
    }

    #[test]
    fn test_symmetric_and_asymmetric_scores_agree() {
        // The query is encoded into the same code space as the database, so
        // scoring a stored vector as a query must reproduce the symmetric
        // score exactly. This is what lets one kernel serve build and search.
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let (n, dim) = (150, 10);
            let data = blobs::<f32>(n, dim);
            let codec = Sq8uCodec::new(&data, n, dim, metric, None).unwrap();

            for a in [0usize, 33, 149] {
                let q = codec.encode_query(&data[a * dim..(a + 1) * dim]).unwrap();
                for b in [5usize, 77, 120] {
                    assert_relative_eq!(
                        codec.score(&q, b),
                        codec.score_sym(a, b),
                        max_relative = 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn test_encode_query_rejects_wrong_dimension() {
        let (n, dim) = (40, 8);
        let data = blobs::<f32>(n, dim);
        let codec = Sq8uCodec::new(&data, n, dim, Dist::SquaredEuclidean, None).unwrap();
        assert!(codec.encode_query(&vec![0.0f32; dim + 3]).is_err());
    }

    #[test]
    fn test_memory_is_about_one_byte_per_dimension() {
        let (n, dim) = (10_000, 64);
        let data = blobs::<f32>(n, dim);
        let codec = Sq8uCodec::new(&data, n, dim, Dist::SquaredEuclidean, None).unwrap();

        let float_bytes = n * dim * std::mem::size_of::<f32>();
        let got = codec.memory_usage_bytes();
        // Codes plus a u32 norm per vector, so a shade over a quarter.
        assert!(
            got < float_bytes / 3,
            "codec {got} B vs f32 {float_bytes} B"
        );
    }

    #[test]
    fn test_f64_codec_matches_f32_ranking() {
        let (n, dim) = (200, 12);
        let d32 = blobs::<f32>(n, dim);
        let d64 = blobs::<f64>(n, dim);
        let c32 = Sq8uCodec::new(&d32, n, dim, Dist::SquaredEuclidean, None).unwrap();
        let c64 = Sq8uCodec::new(&d64, n, dim, Dist::SquaredEuclidean, None).unwrap();

        let q32 = c32.encode_query(&d32[..dim]).unwrap();
        let q64 = c64.encode_query(&d64[..dim]).unwrap();

        let mut r32: Vec<usize> = (0..n).collect();
        let mut r64: Vec<usize> = (0..n).collect();
        r32.sort_by(|&a, &b| {
            c32.score(&q32, a)
                .total_cmp(&c32.score(&q32, b))
                .then(a.cmp(&b))
        });
        r64.sort_by(|&a, &b| {
            c64.score(&q64, a)
                .total_cmp(&c64.score(&q64, b))
                .then(a.cmp(&b))
        });
        assert_eq!(r32, r64);
    }
}
