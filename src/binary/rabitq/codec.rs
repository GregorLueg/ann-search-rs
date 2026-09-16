//! RaBitQ+ codec for the quantised graph index.
//!
//! Implements [`GraphCodec`] over multi-bit RaBitQ codes, so
//! [`HnswQuantisedIndex`](crate::quantised::hnsw_quantised::index::HnswQuantisedIndex)
//! serves RaBitQ the same way it already serves uniform scalar quantisation.
//! The index holds no float vectors: a vertex is its sign bits, its magnitude
//! bits and three factors, which is what the RaBitQ reference's HNSW does and
//! the reason that index exists at all.
//!
//! Needs both `binary` and `quantised`: the codes come from the former and the
//! graph shell from the latter.
//!
//! ### Reconstruction without storing anything extra
//!
//! [`GraphCodec::score_sym`] wants a distance between two *stored* vectors, and
//! nothing here stores one. It does not have to. The estimate is affine in the
//! query,
//!
//! ```text
//! est(q) = f_add + ||q - c||^2 + f_rescale * <q, xu_cb>
//! ```
//!
//! and the true distance expands as
//! `||q - c||^2 + ||v - c||^2 - 2 <q - c, v - c>`. Matching the terms linear in
//! `q` gives `v ~= c - (f_rescale / 2) * xu_cb`, so a vertex reconstructs from
//! the factors it already carries.

use rayon::prelude::*;

use crate::binary::rabitq::ex_bits::{encode_ex_bits, excode_bytes, MAX_EX_BITS};
use crate::binary::rabitq::rotator::RotatorKind;
use crate::binary::rabitq::RaBitQEncoder;
use crate::prelude::*;
use crate::quantised::hnsw_quantised::build::GraphBuildParams;
use crate::quantised::hnsw_quantised::codec::GraphCodec;
use crate::quantised::hnsw_quantised::index::HnswQuantisedIndex;
use crate::utils::graph_utils::ThreadLocalSearchState;
use crate::utils::k_means_utils::{assign_all_parallel, sample_vectors, train_centroids};

////////////
// Consts //
////////////

/// Training vectors drawn per centroid, matching [`crate::binary::ivf_rabitq`].
const TRAIN_PER_CENTROID: usize = 256;

/// Upper bound on the k-means training set.
const MAX_TRAIN: usize = 250_000;

/// Factors held per vertex: `f_add`, `f_rescale`, `f_error`.
const FACTORS_PER_VERTEX: usize = 3;

//////////////////////
// RaBitQCodecQuery //
//////////////////////

/// A query prepared once and scored against many vertices.
pub struct RaBitQCodecQuery<T> {
    /// Rotated query, length `padded_dim`
    rotated: Vec<T>,
    /// `sum(rotated)`, the centring correction's only query-dependent part
    sum: T,
    /// `||q - c||^2` per centroid, the whole per-query cost of `nlist > 1`
    g_add: Vec<T>,
}

/////////////////
// RaBitQCodec //
/////////////////

/// Vector storage for a graph index, as RaBitQ+ codes.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct RaBitQCodec<T> {
    /// The rotation the codes were taken in
    encoder: RaBitQEncoder<T>,
    /// Sign bits per vertex, `n * n_bytes`
    sign_codes: Vec<u8>,
    /// Magnitude bits per vertex, `n * ex_bytes`, empty at `ex_bits` 0
    ex_codes: Vec<u8>,
    /// `(f_add, f_rescale, f_error)` per vertex, interleaved
    factors: Vec<T>,
    /// Centroid each vertex was encoded against
    cluster: Vec<u32>,
    /// Rotated centroids, `nlist * padded_dim`
    centroids_rotated: Vec<T>,
    /// Number of centroids
    nlist: usize,
    /// Magnitude bits per coordinate
    ex_bits: usize,
    /// Dimensionality of the data
    dim: usize,
    /// Dimensionality of the rotated frame
    padded_dim: usize,
    /// Number of vectors
    n: usize,
    /// The metric the index answers in
    metric: Dist,
}

impl<T> RaBitQCodec<T>
where
    T: AnnSearchFloat,
{
    /// Encode a dataset.
    ///
    /// The input is consumed for clustering and encoding only; nothing derived
    /// from it beyond the codes and centroids is kept.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Row-major vectors, already normalised when the metric
    ///   is cosine
    /// * `n` - Number of vectors
    /// * `dim` - Number of features
    /// * `metric` - Distance metric the index answers in
    /// * `ex_bits` - Magnitude bits per coordinate, `0..=MAX_EX_BITS`
    /// * `nlist` - Centroid count, `None` picks `sqrt(n)`
    /// * `k_means_params` - Optional k-means settings
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let the
    ///   dimensionality decide
    /// * `seed` - Random seed, for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// The codec, or an error on an unsupported metric or width
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        vectors_flat: &[T],
        n: usize,
        dim: usize,
        metric: Dist,
        ex_bits: usize,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        rotator_kind: Option<RotatorKind>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        if ex_bits > MAX_EX_BITS {
            return Err(AnnSearchErrors::RaBitQInvalidExBits {
                ex_bits,
                max: MAX_EX_BITS,
            });
        }

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1).min(n);

        if verbose {
            println!(
                "  Encoding {n} vectors at {} total bits over {nlist} centroids.",
                ex_bits + 1
            );
        }

        // Clustering runs in the space the walk searches, which for cosine is
        // the already-normalised one, so the metric here is always squared
        // Euclidean.
        let n_train = (TRAIN_PER_CENTROID * nlist).min(MAX_TRAIN).min(n).max(1);
        let (training_data, _) = sample_vectors(vectors_flat, dim, n, n_train, seed);
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &Dist::SquaredEuclidean,
            k_means_params,
            seed,
            verbose,
        )?;

        let vector_norms: Vec<T> = vectors_flat
            .par_chunks(dim)
            .map(|row| compute_l2_norm(row))
            .collect();
        let centroid_norms: Vec<T> = (0..nlist)
            .map(|c| compute_l2_norm(&centroids[c * dim..(c + 1) * dim]))
            .collect();

        let assignments = assign_all_parallel(
            vectors_flat,
            &vector_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &Dist::SquaredEuclidean,
        );

        let encoder = match rotator_kind {
            Some(kind) => {
                RaBitQEncoder::with_rotator_kind(dim, Dist::SquaredEuclidean, kind, seed as u64)?
            }
            None => RaBitQEncoder::new(dim, Dist::SquaredEuclidean, seed as u64),
        };
        let padded_dim = encoder.padded_dim;
        let n_bytes = encoder.n_bytes;
        let ex_bytes = excode_bytes(padded_dim, ex_bits);

        let centroids_rotated: Vec<T> = (0..nlist)
            .flat_map(|c| encoder.apply_rotation(&centroids[c * dim..(c + 1) * dim]))
            .collect();

        let mut sign_codes = vec![0u8; n * n_bytes];
        let mut ex_codes = vec![0u8; n * ex_bytes];
        let mut factors = vec![T::zero(); n * FACTORS_PER_VERTEX];

        sign_codes
            .par_chunks_mut(n_bytes)
            .zip(ex_codes.par_chunks_mut(ex_bytes.max(1)))
            .zip(factors.par_chunks_mut(FACTORS_PER_VERTEX))
            .enumerate()
            .try_for_each(
                |(node, ((sign, ex), factor))| -> Result<(), AnnSearchErrors> {
                    let c = assignments[node];
                    let rotated =
                        encoder.apply_rotation(&vectors_flat[node * dim..(node + 1) * dim]);
                    let centroid = &centroids_rotated[c * padded_dim..(c + 1) * padded_dim];

                    let encoded = encode_ex_bits(&rotated, centroid, ex_bits, None)?;

                    sign.copy_from_slice(&encoded.sign_code);
                    if ex_bytes > 0 {
                        ex[..ex_bytes].copy_from_slice(&encoded.ex_code);
                    }
                    factor[0] = encoded.f_add;
                    factor[1] = encoded.f_rescale;
                    factor[2] = encoded.f_error;

                    Ok(())
                },
            )?;

        Ok(Self {
            encoder,
            sign_codes,
            ex_codes,
            factors,
            cluster: assignments.iter().map(|&c| c as u32).collect(),
            centroids_rotated,
            nlist,
            ex_bits,
            dim,
            padded_dim,
            n,
            metric,
        })
    }

    /// Signed and magnitude inner products of one vertex against a query.
    ///
    /// Reads the packed codes in place; allocating a level buffer per vertex
    /// would dominate a scan that is otherwise a handful of adds.
    ///
    /// ### Params
    ///
    /// * `node` - Vertex index
    /// * `query` - Rotated query, length `padded_dim`
    ///
    /// ### Returns
    ///
    /// `<q, sign>` and `<q, magnitude>`
    #[inline]
    fn inner_products(&self, node: usize, query: &[T]) -> (T, T) {
        let n_bytes = self.encoder.n_bytes;
        let sign = &self.sign_codes[node * n_bytes..(node + 1) * n_bytes];

        let mut ip_sign = T::zero();
        for (b, &byte) in sign.iter().enumerate() {
            let base = b * 8;
            for bit in 0..8 {
                if byte >> bit & 1 == 1 {
                    ip_sign = ip_sign + query[base + bit];
                }
            }
        }

        if self.ex_bits == 0 {
            return (ip_sign, T::zero());
        }

        let ex_bytes = excode_bytes(self.padded_dim, self.ex_bits);
        let ex = &self.ex_codes[node * ex_bytes..(node + 1) * ex_bytes];
        let mask = (1u16 << self.ex_bits) - 1;

        let mut ip_ex = T::zero();
        for (d, &q) in query.iter().enumerate() {
            let start = d * self.ex_bits;
            let byte = start / 8;
            let shift = start % 8;

            // A level can straddle a byte boundary at widths that do not divide
            // eight, so read two bytes and shift the window out.
            let low = ex[byte] as u16;
            let high = if byte + 1 < ex.len() {
                ex[byte + 1] as u16
            } else {
                0
            };
            let level = ((low | (high << 8)) >> shift) & mask;

            if level != 0 {
                ip_ex = ip_ex + q * T::from_u16(level).unwrap_or_else(T::zero);
            }
        }

        (ip_sign, ip_ex)
    }

    /// Estimate the squared distance from a prepared query to one vertex.
    ///
    /// ### Params
    ///
    /// * `node` - Vertex index
    /// * `rotated` - Rotated query, length `padded_dim`
    /// * `sum` - `sum(rotated)`
    /// * `g_add` - `||q - c||^2` for this vertex's centroid
    ///
    /// ### Returns
    ///
    /// The estimated squared Euclidean distance
    #[inline]
    fn estimate(&self, node: usize, rotated: &[T], sum: T, g_add: T) -> T {
        let (ip_sign, ip_ex) = self.inner_products(node, rotated);

        let base = node * FACTORS_PER_VERTEX;
        let f_add = self.factors[base];
        let f_rescale = self.factors[base + 1];

        let step = T::from_f64((1u64 << self.ex_bits) as f64).unwrap_or_else(T::one);
        let cb = T::from_f64(-((1u64 << self.ex_bits) as f64 - 0.5)).unwrap_or_else(T::zero);

        f_add + g_add + f_rescale * (step * ip_sign + ip_ex + cb * sum)
    }

    /// Reconstruct a vertex in the rotated frame.
    ///
    /// See the module header: the estimate is affine in the query, so matching
    /// its query-linear term against the true expansion recovers the vector
    /// from the factors already stored.
    ///
    /// ### Params
    ///
    /// * `node` - Vertex index
    ///
    /// ### Returns
    ///
    /// The reconstructed rotated vector, length `padded_dim`
    fn reconstruct(&self, node: usize) -> Vec<T> {
        let n_bytes = self.encoder.n_bytes;
        let sign = &self.sign_codes[node * n_bytes..(node + 1) * n_bytes];
        let f_rescale = self.factors[node * FACTORS_PER_VERTEX + 1];

        let centroid = self.centroid(self.cluster[node] as usize);
        let two = T::one() + T::one();
        let step = (1u64 << self.ex_bits) as f64;
        let cb = -(step - 0.5);

        let ex_bytes = excode_bytes(self.padded_dim, self.ex_bits);
        let mask = (1u16 << self.ex_bits) - 1;

        (0..self.padded_dim)
            .map(|d| {
                let positive = sign[d / 8] >> (d % 8) & 1 == 1;
                let level = if self.ex_bits == 0 {
                    0u16
                } else {
                    let ex = &self.ex_codes[node * ex_bytes..(node + 1) * ex_bytes];
                    let start = d * self.ex_bits;
                    let byte = start / 8;
                    let low = ex[byte] as u16;
                    let high = if byte + 1 < ex.len() {
                        ex[byte + 1] as u16
                    } else {
                        0
                    };
                    ((low | (high << 8)) >> (start % 8)) & mask
                };

                let xu_cb = level as f64 + if positive { step } else { 0.0 } + cb;
                centroid[d] - (f_rescale / two) * T::from_f64(xu_cb).unwrap_or_else(T::zero)
            })
            .collect()
    }

    /// One rotated centroid.
    ///
    /// ### Params
    ///
    /// * `cluster` - Centroid index
    ///
    /// ### Returns
    ///
    /// The rotated centroid, length `padded_dim`
    #[inline]
    fn centroid(&self, cluster: usize) -> &[T] {
        &self.centroids_rotated[cluster * self.padded_dim..(cluster + 1) * self.padded_dim]
    }

    /// Error bound coefficient of one vertex.
    ///
    /// Multiplied by `||q - c||` it bounds the estimate's absolute error, which
    /// is what a two-tier search gates its promotion on.
    ///
    /// ### Params
    ///
    /// * `node` - Vertex index
    ///
    /// ### Returns
    ///
    /// The coefficient
    #[inline]
    pub fn f_error(&self, node: usize) -> T {
        self.factors[node * FACTORS_PER_VERTEX + 2]
    }

    /// Magnitude bits per coordinate.
    ///
    /// ### Returns
    ///
    /// The width, zero for a sign-only code
    pub fn ex_bits(&self) -> usize {
        self.ex_bits
    }

    /// Number of centroids the codes were taken against.
    ///
    /// ### Returns
    ///
    /// Centroid count
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Bytes held by the codec.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.encoder.memory_usage_bytes()
            + self.sign_codes.capacity()
            + self.ex_codes.capacity()
            + self.factors.capacity() * std::mem::size_of::<T>()
            + self.cluster.capacity() * std::mem::size_of::<u32>()
            + self.centroids_rotated.capacity() * std::mem::size_of::<T>()
    }
}

////////////////
// GraphCodec //
////////////////

impl<T> GraphCodec<T> for RaBitQCodec<T>
where
    T: AnnSearchFloat,
{
    type Query = RaBitQCodecQuery<T>;

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
        if query.len() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: query.len(),
            });
        }

        let normalised: Vec<T> = if self.metric == Dist::Cosine {
            let norm = compute_l2_norm(query);
            if norm > T::epsilon() {
                query.iter().map(|&x| x / norm).collect()
            } else {
                query.to_vec()
            }
        } else {
            query.to_vec()
        };

        let rotated = self.encoder.apply_rotation(&normalised);
        let sum = rotated.iter().fold(T::zero(), |a, &b| a + b);

        // The rotation is orthogonal, so distances taken in the rotated frame
        // are the distances in the original one.
        let g_add: Vec<T> = (0..self.nlist)
            .map(|c| T::euclidean_simd(self.centroid(c), &rotated))
            .collect();

        Ok(RaBitQCodecQuery {
            rotated,
            sum,
            g_add,
        })
    }

    #[inline]
    fn score(&self, query: &Self::Query, id: usize) -> T {
        let g_add = query.g_add[self.cluster[id] as usize];
        self.estimate(id, &query.rotated, query.sum, g_add)
    }

    fn score_sym(&self, a: usize, b: usize) -> T {
        let rotated = self.reconstruct(a);
        let sum = rotated.iter().fold(T::zero(), |acc, &x| acc + x);
        let g_add = T::euclidean_simd(self.centroid(self.cluster[b] as usize), &rotated);
        self.estimate(b, &rotated, sum, g_add)
    }

    #[inline]
    fn finalise(&self, score: T) -> T {
        // The estimate is a squared Euclidean distance, which on unit vectors
        // is twice the cosine distance. A code can estimate slightly negative
        // for a vertex sitting on the query; clamping keeps a reported distance
        // from going below zero.
        let clamped = if score < T::zero() { T::zero() } else { score };
        match self.metric {
            Dist::Cosine => clamped / (T::one() + T::one()),
            _ => clamped,
        }
    }
}

/////////////////////
// HnswRaBitQIndex //
/////////////////////

/// A quantised HNSW over RaBitQ+ codes.
pub type HnswRaBitQIndex<T> = HnswQuantisedIndex<T, RaBitQCodec<T>>;

impl<T> HnswQuantisedIndex<T, RaBitQCodec<T>>
where
    T: AnnSearchFloat + ThreadLocalSearchState,
{
    /// Encode and build in one step.
    ///
    /// The graph is linked on **exact** distances and the float vectors are
    /// dropped afterwards, which is what the RaBitQ reference does. Pruning a
    /// graph against the codec's own approximation would fold the codec's error
    /// into the topology, where no amount of query-time accuracy recovers it.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix, rows are samples
    /// * `m` - Base connectivity; layer 0 gets `2 * m` slots per vertex
    /// * `ef_construction` - Construction beam width
    /// * `metric` - Distance metric; Manhattan is not supported
    /// * `ex_bits` - Magnitude bits per coordinate, `0..=MAX_EX_BITS`. Total
    ///   width is `ex_bits + 1`
    /// * `nlist` - Centroid count, `None` picks `sqrt(n)`
    /// * `k_means_params` - Optional k-means settings
    /// * `seed` - Random seed, for reproducibility
    /// * `verbose` - Whether to print progress
    ///
    /// ### Returns
    ///
    /// The built index, or an error on an unsupported metric or width
    #[allow(clippy::too_many_arguments)]
    pub fn build_rabitq(
        data: impl AnnMatrix<T>,
        m: usize,
        ef_construction: usize,
        metric: Dist,
        ex_bits: usize,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let (mut flat, n, dim) = data.into_row_major();

        // Cosine runs as squared Euclidean on unit vectors, which is the frame
        // the codes are taken in; the conversion back happens in `finalise`.
        if metric == Dist::Cosine {
            flat.par_chunks_mut(dim).for_each(|row| {
                let norm = compute_l2_norm(row);
                if norm > T::epsilon() {
                    row.iter_mut().for_each(|x| *x = *x / norm);
                }
            });
        }

        let codec = RaBitQCodec::encode(
            &flat,
            n,
            dim,
            metric,
            ex_bits,
            nlist,
            k_means_params,
            None,
            seed,
            verbose,
        )?;

        let params = GraphBuildParams::new(m, ef_construction, seed, verbose);
        let (graph, hierarchy) = Self::build_topology(&codec, &params, |a, b| {
            T::euclidean_simd(&flat[a * dim..(a + 1) * dim], &flat[b * dim..(b + 1) * dim])
        });

        drop(flat);

        Ok(Self::from_parts(codec, graph, hierarchy, &params))
    }

    /// Bytes held by the index.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.codec().memory_usage_bytes()
            + self.graph().memory_usage_bytes()
            + self.hierarchy().memory_usage_bytes()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Deterministic pseudo-random floats in `[-0.5, 0.5)`.
    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        }
    }

    /// Clusters whose spread is wider than their separation, so the graph does
    /// not split into one component per cluster.
    fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
        let mut next = rng(seed);
        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| next() * 0.3).collect())
            .collect();
        Mat::from_fn(n, dim, |i, j| centres[i % n_clusters][j] + next())
    }

    fn build(data: &Mat<f32>, metric: Dist, ex_bits: usize) -> HnswRaBitQIndex<f32> {
        HnswRaBitQIndex::build_rabitq(
            data.as_ref(),
            16,
            200,
            metric,
            ex_bits,
            None,
            None,
            42,
            false,
        )
        .unwrap()
    }

    fn brute_force(data: &Mat<f32>, query: &[f32], k: usize, metric: Dist) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..data.nrows())
            .map(|i| {
                let row: Vec<f32> = (0..data.ncols()).map(|j| data[(i, j)]).collect();
                let d = match metric {
                    Dist::Cosine => {
                        let dot: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
                        let na = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let nb = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot / (na * nb)
                    }
                    _ => row.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum(),
                };
                (d, i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn recall(
        index: &HnswRaBitQIndex<f32>,
        data: &Mat<f32>,
        k: usize,
        ef: usize,
        metric: Dist,
    ) -> f64 {
        let (n, dim) = (data.nrows(), data.ncols());
        let n_queries = 60;
        let mut hits = 0usize;
        for q in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|j| data[(q * 37 % n, j)]).collect();
            let truth = brute_force(data, &query, k, metric);
            let (got, _) = index.query(&query, k, ef).unwrap();
            hits += got.iter().filter(|i| truth.contains(i)).count();
        }
        hits as f64 / (n_queries * k) as f64
    }

    #[test]
    fn test_wider_codes_recall_better() {
        // The whole reason the extended code exists: with no float vectors to
        // fall back on, recall has to come from the code width.
        let data = clustered(4000, 128, 20, 7);

        let mut previous = 0.0f64;
        for ex_bits in [0usize, 2, 4, 7] {
            let index = build(&data, Dist::SquaredEuclidean, ex_bits);
            let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
            assert!(
                r >= previous - 0.01,
                "ex_bits {ex_bits} recall {r} fell from {previous}"
            );
            previous = previous.max(r);
        }
        assert!(previous > 0.9, "widest code only reached {previous}");
    }

    #[test]
    fn test_index_holds_no_float_vectors() {
        // The point of the whole design. The codec is checked separately from
        // the whole index because the adjacency is a fixed `n * 2m * 4` bytes
        // that every HNSW pays whatever its storage, and at this size it is
        // half the index on its own.
        let (n, dim) = (8000usize, 128usize);
        let data = clustered(n, dim, 20, 9);
        let index = build(&data, Dist::SquaredEuclidean, 4);

        let floats = n * dim * std::mem::size_of::<f32>();
        let codec = index.codec().memory_usage_bytes();
        let held = index.memory_usage_bytes();

        assert!(
            codec * 3 < floats,
            "codec held {codec} bytes against {floats} of floats"
        );
        assert!(
            held < floats,
            "index held {held} bytes against {floats} of floats"
        );
    }

    #[test]
    fn test_query_finds_self_at_k1() {
        let data = clustered(2000, 64, 20, 3);
        let index = build(&data, Dist::SquaredEuclidean, 6);

        let mut found = 0usize;
        let probes = (0..2000).step_by(97).count();
        for i in (0..2000).step_by(97) {
            let query: Vec<f32> = (0..64).map(|j| data[(i, j)]).collect();
            let (got, _) = index.query(&query, 1, 64).unwrap();
            if got[0] == i {
                found += 1;
            }
        }
        // Distances are estimates, not exact, so a handful of ties may resolve
        // the other way; the walk still has to land on the query itself nearly
        // every time.
        assert!(found * 10 >= probes * 9, "{found} of {probes} found self");
    }

    #[test]
    fn test_cosine_recall_against_brute_force() {
        let data = clustered(4000, 128, 20, 11);
        let index = build(&data, Dist::Cosine, 6);
        let r = recall(&index, &data, 10, 128, Dist::Cosine);
        assert!(r > 0.9, "cosine recall@10 was {r}");
    }

    #[test]
    fn test_reported_distances_are_close_to_the_metric() {
        // Unlike the exact-vector indices these are estimates, so the check is
        // that the codec's scale and metric conversion are right, not that the
        // number is exact.
        let data = clustered(2000, 128, 10, 5);

        for (metric, tolerance) in [(Dist::SquaredEuclidean, 0.08), (Dist::Cosine, 0.08)] {
            let index = build(&data, metric, 7);
            let query: Vec<f32> = (0..128).map(|j| data[(11, j)]).collect();
            let (got, dists) = index.query(&query, 5, 128).unwrap();

            for (&id, &d) in got.iter().zip(&dists) {
                let row: Vec<f32> = (0..128).map(|j| data[(id, j)]).collect();
                let want: f32 = match metric {
                    Dist::Cosine => {
                        let dot: f32 = row.iter().zip(&query).map(|(a, b)| a * b).sum();
                        let na = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let nb = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot / (na * nb)
                    }
                    _ => row.iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum(),
                };
                assert!(
                    (d - want).abs() <= tolerance * want.max(1e-3),
                    "{metric:?}: reported {d} against {want}"
                );
                assert!(d >= 0.0, "{metric:?}: reported a negative distance {d}");
            }
        }
    }

    #[test]
    fn test_manhattan_is_rejected() {
        let data = clustered(200, 32, 4, 31);
        assert!(matches!(
            HnswRaBitQIndex::build_rabitq(
                data.as_ref(),
                16,
                200,
                Dist::Manhattan,
                4,
                None,
                None,
                42,
                false
            ),
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_too_many_bits_is_rejected() {
        let data = clustered(200, 32, 4, 33);
        assert!(matches!(
            HnswRaBitQIndex::build_rabitq(
                data.as_ref(),
                16,
                200,
                Dist::SquaredEuclidean,
                MAX_EX_BITS + 1,
                None,
                None,
                42,
                false
            ),
            Err(AnnSearchErrors::RaBitQInvalidExBits { .. })
        ));
    }

    #[test]
    fn test_single_centroid_builds_and_queries() {
        let data = clustered(2000, 64, 10, 37);
        let index = HnswRaBitQIndex::build_rabitq(
            data.as_ref(),
            16,
            200,
            Dist::SquaredEuclidean,
            6,
            Some(1),
            None,
            42,
            false,
        )
        .unwrap();

        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.85, "recall@10 at nlist 1 was {r}");
    }

    #[test]
    fn test_reconstruct_recovers_the_vector() {
        // `score_sym` leans on this entirely, so it needs checking on its own.
        let (n, dim) = (500usize, 64usize);
        let data = clustered(n, dim, 5, 41);
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push(data[(i, j)]);
            }
        }

        let codec = RaBitQCodec::encode(
            &flat,
            n,
            dim,
            Dist::SquaredEuclidean,
            7,
            None,
            None,
            None,
            42,
            false,
        )
        .unwrap();

        let mut worst = 0.0f32;
        for node in (0..n).step_by(37) {
            let rebuilt = codec.reconstruct(node);
            let rotated = codec
                .encoder
                .apply_rotation(&flat[node * dim..(node + 1) * dim]);

            let err: f32 = rotated
                .iter()
                .zip(&rebuilt)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
            worst = worst.max(err / norm);
        }
        assert!(worst < 0.25, "worst relative reconstruction error {worst}");
    }

    #[test]
    fn test_f64_index_builds_and_queries() {
        let data32 = clustered(600, 32, 6, 43);
        let data: Mat<f64> = Mat::from_fn(600, 32, |i, j| data32[(i, j)] as f64);

        let index: HnswRaBitQIndex<f64> = HnswRaBitQIndex::build_rabitq(
            data.as_ref(),
            16,
            200,
            Dist::SquaredEuclidean,
            6,
            None,
            None,
            42,
            false,
        )
        .unwrap();

        let query: Vec<f64> = (0..32).map(|j| data[(7, j)]).collect();
        let (got, _) = index.query(&query, 5, 64).unwrap();
        assert!(got.contains(&7), "did not retrieve the query row");
    }
}
