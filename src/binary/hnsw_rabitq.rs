//! HNSW over per-vertex RaBitQ codes.
//!
//! The sibling of [`crate::binary::qg::QgIndex`] at the other end of the memory
//! curve. Both replace the exact distance to a neighbour with a one-bit RaBitQ
//! estimate and keep exact distances for the vertices actually expanded; they
//! differ in what the code is taken against.
//!
//! [`QgIndex`](crate::binary::qg::QgIndex) quantises every vertex against each
//! of its in-neighbours, so a vector's code is duplicated once per in-edge and
//! a hop is one contiguous fast-scan sweep. This index quantises every vertex
//! once, against its cluster centroid, so the codes cost `n * padded_dim / 8`
//! bytes however dense the graph is. The estimate is then a per-neighbour
//! random read of `padded_dim / 8` bytes rather than the `dim * 4` an exact
//! distance needs, which is where the saving comes from: the arithmetic per
//! neighbour is comparable to the exact kernel, the memory traffic is not.
//!
//! ### The estimate
//!
//! With `c` the vertex's cluster centroid, `s` its sign vector and `dc` its
//! stored dot correction,
//!
//! ```text
//! ||q - v||^2 = ||q - c||^2 + ||v - c||^2 - 2 ||v - c|| * dc * (<Rq, s> - <Rc, s>)
//! ```
//!
//! Only `<Rq, s>` involves the query. `<Rc, s>` folds into a per-vertex
//! constant at build time and `||q - c||^2` is computed once per cluster per
//! query, so a neighbour costs one signed dot against the rotated query plus a
//! multiply and two adds:
//!
//! ```text
//! est = f_add + g_add[cluster] + f_rescale * <Rq, s>
//! ```
//!
//! ### References
//!
//! Gao et al., "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
//! Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024.

use faer::RowRef;
use faer_traits::ComplexField;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::binary::dist_binary::asymmetric_binary_dot;
use crate::binary::rabitq::RaBitQEncoder;
use crate::binary::rotator::RotatorKind;
use crate::prelude::*;
use crate::utils::graph_utils::ThreadLocalSearchState;
use crate::utils::hnsw_graph::build::{build_hierarchical_graph, GraphBuildParams};
use crate::utils::hnsw_graph::flat_graph::{FlatGraph, HnswHierarchy};
use crate::utils::k_means_utils::{assign_all_parallel, sample_vectors, train_centroids};
use crate::utils::pack_knn_results;
use crate::utils::KnnValidation;

////////////
// Consts //
////////////

/// Sentinel marking an unused neighbour slot.
const SENTINEL: u32 = u32::MAX;

/// Beam width [`KnnValidation`] queries at.
///
/// Wide enough that a recall shortfall is the index's and not the beam's.
const HNSW_RABITQ_VALIDATION_EF: usize = 256;

/// Training vectors drawn per centroid.
///
/// Matches [`crate::binary::ivf_rabitq`]: enough to place a centroid without
/// making the k-means the dominant cost of the build.
const TRAIN_PER_CENTROID: usize = 256;

/// Upper bound on the k-means training set.
const MAX_TRAIN: usize = 250_000;

/////////////
// Helpers //
/////////////

/// Coordinates one nibble of a code covers.
const DIMS_PER_NIBBLE: usize = 4;

/// Entries in one nibble sub-table.
const NIBBLE_VALUES: usize = 1 << DIMS_PER_NIBBLE;

/// Independent accumulator chains in the per-neighbour scan.
///
/// An add has several cycles of latency and can issue more than once per
/// cycle, so a single running sum leaves the core idle most of the time. Four
/// chains cover the latency without needing the table to be any wider.
const ACCUMULATORS: usize = 4;

/// Build the per-query nibble table.
///
/// Every neighbour needs `sum_d sign(code_d) * q_rot[d]`. Done directly that is
/// one branch and one add per coordinate, which costs more than the exact
/// distance kernel it is supposed to replace. Splitting the code into nibbles
/// turns it into `padded_dim / 4` table lookups: sub-table `g` holds, for each
/// of the sixteen sign patterns a nibble can take, the signed sum of the four
/// rotated query coordinates it covers.
///
/// ### Params
///
/// * `q_rot` - Rotated query, length `padded_dim`, a multiple of 8
///
/// ### Returns
///
/// `(padded_dim / 4) * 16` entries, sub-table `g` at `g * 16`
fn build_sign_nibble_table<T>(q_rot: &[T]) -> Vec<T>
where
    T: AnnSearchFloat,
{
    let n_groups = q_rot.len() / DIMS_PER_NIBBLE;
    let mut table = vec![T::zero(); n_groups * NIBBLE_VALUES];

    for g in 0..n_groups {
        let base = g * DIMS_PER_NIBBLE;
        for value in 0..NIBBLE_VALUES {
            let mut acc = T::zero();
            for c in 0..DIMS_PER_NIBBLE {
                // A set bit means `+1`, a clear bit `-1`, matching what
                // `RaBitQEncoder::encode_vector` writes.
                if value & (1 << c) == 0 {
                    acc = acc - q_rot[base + c];
                } else {
                    acc = acc + q_rot[base + c];
                }
            }
            table[g * NIBBLE_VALUES + value] = acc;
        }
    }

    table
}

/// Signed dot product between one code and the prepared query.
///
/// ### Params
///
/// * `table` - Table from [`build_sign_nibble_table`]
/// * `code` - Packed sign bits of one vertex, `padded_dim / 8` bytes
///
/// ### Returns
///
/// `sum_d sign(code_d) * q_rot[d]`
#[inline]
fn signed_dot_nibble<T>(table: &[T], code: &[u8]) -> T
where
    T: AnnSearchFloat,
{
    let mut acc = [T::zero(); ACCUMULATORS];

    for (b, &byte) in code.iter().enumerate() {
        let lo = (b * 2) * NIBBLE_VALUES + (byte & 0x0F) as usize;
        let hi = (b * 2 + 1) * NIBBLE_VALUES + (byte >> 4) as usize;

        // SAFETY: `code.len() * 2` sub-tables exist by construction and a
        // nibble cannot index past one.
        unsafe {
            acc[b % ACCUMULATORS] = acc[b % ACCUMULATORS] + *table.get_unchecked(lo);
            acc[(b + 1) % ACCUMULATORS] = acc[(b + 1) % ACCUMULATORS] + *table.get_unchecked(hi);
        }
    }

    acc.iter().fold(T::zero(), |a, &b| a + b)
}

///////////////////////
// HnswRaBitQIndex   //
///////////////////////

/// HNSW graph whose neighbour screening runs on one-bit RaBitQ codes.
///
/// Cosine is served by normalising the stored vectors at build, so everything
/// internal is squared Euclidean and the reported distance is converted back on
/// the way out. The original vector magnitudes are not kept.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct HnswRaBitQIndex<T> {
    /// Row-major vectors, `n * dim`. Unit length when `metric` is cosine.
    vectors_flat: Vec<T>,
    /// Per-vector norms, all ones for cosine and empty otherwise
    norms: Vec<T>,
    /// The rotation the codes were taken in
    encoder: RaBitQEncoder<T>,
    /// Dense layer-0 adjacency, `n * degree`, sentinel-padded
    graph: FlatGraph,
    /// The layers above 0, used only to seed the base-layer search
    hierarchy: HnswHierarchy,
    /// One code per vertex in node order, `n * n_bytes`
    codes: Vec<u8>,
    /// Per-vertex `(f_add, f_rescale)` pairs, `2 * n`.
    ///
    /// Interleaved rather than held as two arrays: the walk reads both for
    /// every neighbour it screens, and on a working set past cache a second
    /// scattered region costs more than the same bytes read together.
    factors: Vec<T>,
    /// Cluster each vertex was encoded against, `n`
    cluster: Vec<u32>,
    /// Cluster centroids, `nlist * dim`
    centroids: Vec<T>,
    /// Number of centroids
    nlist: usize,
    /// Dimensionality of the data
    dim: usize,
    /// Dimensionality of the rotated frame
    padded_dim: usize,
    /// Number of vectors
    n: usize,
    /// The metric the index answers in
    metric: Dist,
    /// Original row indices
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for HnswRaBitQIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for HnswRaBitQIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////////
// HnswRaBitQIndex //
/////////////////////

impl<T> HnswRaBitQIndex<T>
where
    T: AnnSearchFloat + ComplexField + ThreadLocalSearchState,
{
    /// Build an HNSW index over per-vertex RaBitQ codes.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix, `n` samples by `dim` features
    /// * `metric` - Distance metric, squared Euclidean or cosine
    /// * `m` - HNSW connectivity; layer 0 gets `2 * m` slots per vertex
    /// * `ef_construction` - Beam width during graph construction
    /// * `nlist` - Centroids the codes are taken against. `None` picks
    ///   `sqrt(n)`. One global centroid is legal and cheapest to query.
    /// * `k_means_params` - Optional k-means settings, see
    ///   [`KMeansTrainingParams`]
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let the
    ///   dimensionality decide
    /// * `seed` - Random seed, for reproducibility
    /// * `verbose` - Print build progress
    ///
    /// ### Returns
    ///
    /// The index, or an error on an unsupported metric
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        m: usize,
        ef_construction: usize,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        rotator_kind: Option<RotatorKind>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        // Cosine runs as squared Euclidean on unit vectors: the two order
        // identically and `2 * cosine = squared euclidean` converts back
        // exactly, which keeps the residual geometry the codes assume.
        let normalise = metric == Dist::Cosine;
        if normalise {
            vectors_flat.par_chunks_mut(dim).for_each(|row| {
                let norm = compute_l2_norm(row);
                if norm > T::epsilon() {
                    row.iter_mut().for_each(|x| *x = *x / norm);
                }
            });
        }
        let norms = if normalise {
            vec![T::one(); n]
        } else {
            Vec::new()
        };

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1).min(n);

        if verbose {
            println!(
                "  Building HNSW-RaBitQ index over {} vectors, {} centroids.",
                n.separate_with_underscores(),
                nlist
            );
        }

        // Clustering runs in the space the walk searches, which for cosine is
        // the normalised one, so the metric here is always squared Euclidean.
        let n_train = (TRAIN_PER_CENTROID * nlist).min(MAX_TRAIN).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);
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
            &vectors_flat,
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

        // One rotation per centroid rather than one per vertex. `<Rc, s>` is
        // the only build-time term that needs it and nlist is small.
        let centroids_rotated: Vec<T> = (0..nlist)
            .flat_map(|c| encoder.apply_rotation(&centroids[c * dim..(c + 1) * dim]))
            .collect();

        let mut codes = vec![0u8; n * n_bytes];
        let mut factors = vec![T::zero(); n * 2];
        let two = T::one() + T::one();

        codes
            .par_chunks_mut(n_bytes)
            .zip(factors.par_chunks_mut(2))
            .enumerate()
            .try_for_each(|(node, (code, factor))| -> Result<(), AnnSearchErrors> {
                let c = assignments[node];
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let c_rot = &centroids_rotated[c * padded_dim..(c + 1) * padded_dim];

                let (encoded, v_dist, dot_correction_inv) = encoder
                    .encode_vector(&vectors_flat[node * dim..(node + 1) * dim], centroid)?;
                code.copy_from_slice(&encoded);

                let rescale = (two * v_dist * dot_correction_inv).neg();
                factor[0] =
                    v_dist * v_dist - rescale * asymmetric_binary_dot(c_rot, code, padded_dim);
                factor[1] = rescale;

                Ok(())
            })?;

        // The graph is built on exact distances: the codes screen neighbours at
        // query time, they do not decide the topology.
        let params = GraphBuildParams::new(m, ef_construction, seed, verbose);
        let (graph, hierarchy) = build_hierarchical_graph::<T, _>(n, &params, |a, b| {
            T::euclidean_simd(
                &vectors_flat[a * dim..(a + 1) * dim],
                &vectors_flat[b * dim..(b + 1) * dim],
            )
        });

        Ok(Self {
            vectors_flat,
            norms,
            encoder,
            graph,
            hierarchy,
            codes,
            factors,
            cluster: assignments.iter().map(|&c| c as u32).collect(),
            centroids,
            nlist,
            dim,
            padded_dim,
            n,
            metric,
            original_ids: (0..n).collect(),
        })
    }

    /// Exact distance from a prepared query to one stored vector.
    ///
    /// Always the squared Euclidean distance, which for cosine is twice the
    /// cosine distance because the stored vectors are unit length.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector, normalised if the metric is cosine
    /// * `node` - Index of the stored vector
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline]
    fn exact(&self, query: &[T], node: usize) -> T {
        let start = node * self.dim;
        T::euclidean_simd(&self.vectors_flat[start..start + self.dim], query)
    }

    /// Convert an internal squared distance into the reported one.
    ///
    /// ### Params
    ///
    /// * `d` - Squared Euclidean distance in the stored frame
    ///
    /// ### Returns
    ///
    /// The distance under this index's metric
    #[inline]
    fn report(&self, d: T) -> T {
        match self.metric {
            Dist::Cosine => d / (T::one() + T::one()),
            _ => d,
        }
    }

    /// Query for the `k` nearest neighbours.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector of length `dim`
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width; clamped up to `k`
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`, nearest first
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let k = k.min(self.n);
        let ef = ef_search.max(k).max(1);

        let query: Vec<T> = if self.metric == Dist::Cosine {
            let norm = compute_l2_norm(query_vec);
            if norm > T::epsilon() {
                query_vec.iter().map(|&x| x / norm).collect()
            } else {
                query_vec.to_vec()
            }
        } else {
            query_vec.to_vec()
        };

        // One rotation for the whole walk; `q_sum` hoists the constant term out
        // of the per-neighbour signed dot.
        let q_rot = self.encoder.apply_rotation(&query);
        let sign_table = build_sign_nibble_table(&q_rot);

        // `||q - c||^2` once per cluster rather than once per neighbour. This
        // is the whole per-query cost of having more than one centroid.
        let g_add: Vec<T> = (0..self.nlist)
            .map(|c| T::euclidean_simd(&self.centroids[c * self.dim..(c + 1) * self.dim], &query))
            .collect();

        let entry = self.hierarchy.descend(|node| self.exact(&query, node));

        T::with_search_state(|state| {
            state.reset(self.n);
            state.results.reset(ef);

            state
                .candidates
                .push(Reverse((OrderedFloat(T::neg_infinity()), entry)));

            while let Some(Reverse((est, node))) = state.candidates.pop() {
                // The frontier is ordered on estimates but gated on exact
                // distances. An estimate already worse than the k-th exact
                // distance cannot lead anywhere better than the estimate error,
                // and without the test the walk drains every candidate it ever
                // admitted, paying a full exact distance for each.
                if est.0 > state.results.threshold() {
                    break;
                }
                if state.is_visited(node) {
                    continue;
                }
                state.mark_visited(node);

                state.results.push(self.exact(&query, node), node);
                let threshold = state.results.threshold();

                for &nb in self.graph.neighbours(node) {
                    if nb == SENTINEL {
                        // Valid ids are packed at the front of the row.
                        break;
                    }
                    let nb = nb as usize;
                    if state.is_visited(nb) {
                        continue;
                    }

                    let code = &self.codes[nb * self.encoder.n_bytes..][..self.encoder.n_bytes];
                    let dot = signed_dot_nibble(&sign_table, code);

                    let est = self.factors[nb * 2]
                        + g_add[self.cluster[nb] as usize]
                        + self.factors[nb * 2 + 1] * dot;

                    if est < threshold {
                        state.candidates.push(Reverse((OrderedFloat(est), nb)));
                    }
                }
            }

            state.results.sort();
            let take = k.min(state.results.len());

            Ok((
                state.results.ids()[..take].to_vec(),
                state.results.dists()[..take]
                    .iter()
                    .map(|&d| self.report(d))
                    .collect(),
            ))
        })
    }

    /// Query from a matrix row.
    ///
    /// Takes the contiguous fast path when the row has unit column stride,
    /// otherwise copies into a temporary.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`, nearest first
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        match query_row.try_as_row_major() {
            Some(contiguous) => self.query(contiguous.as_slice(), k, ef_search),
            None => {
                let owned: Vec<T> = (0..query_row.ncols()).map(|j| query_row[j]).collect();
                self.query(&owned, k, ef_search)
            }
        }
    }

    /// Build the full self-kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per sample
    /// * `ef_search` - Beam width
    /// * `return_dist` - Whether to return distances alongside indices
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// The kNN graph, and the distances when asked for
    pub fn generate_knn(
        &self,
        k: usize,
        ef_search: usize,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query(
                    &self.vectors_flat[i * self.dim..(i + 1) * self.dim],
                    k,
                    ef_search,
                )
            })
            .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

        Ok(pack_knn_results(results, return_dist))
    }

    /// Number of vectors.
    ///
    /// ### Returns
    ///
    /// Vector count
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimensionality of the data.
    ///
    /// ### Returns
    ///
    /// Number of features
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimensionality of the rotated frame.
    ///
    /// ### Returns
    ///
    /// Number of coordinates after rotation
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Neighbour slots per vertex on layer 0.
    ///
    /// ### Returns
    ///
    /// Row stride of the base-layer adjacency
    pub fn degree(&self) -> usize {
        self.graph.degree()
    }

    /// Number of centroids the codes were taken against.
    ///
    /// ### Returns
    ///
    /// Centroid count
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// The metric the index answers in.
    ///
    /// ### Returns
    ///
    /// The metric, see [`Dist`]
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Original row indices.
    ///
    /// ### Returns
    ///
    /// The identity permutation unless the index was reordered
    pub fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }

    /// Bytes held by the index.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.encoder.memory_usage_bytes()
            + self.graph.memory_usage_bytes()
            + self.hierarchy.memory_usage_bytes()
            + self.codes.capacity()
            + self.factors.capacity() * std::mem::size_of::<T>()
            + self.cluster.capacity() * std::mem::size_of::<u32>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

////////////////////
// KnnValidation  //
////////////////////

impl<T> KnnValidation<T> for HnswRaBitQIndex<T>
where
    T: AnnSearchFloat + ComplexField + ThreadLocalSearchState,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, HNSW_RABITQ_VALIDATION_EF)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }
}

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> crate::serialise::IndexIo for HnswRaBitQIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "hnsw_rabitq";
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

    /// Clusters whose spread is wider than their separation.
    ///
    /// The overlap is deliberate: well-separated blobs leave the graph with one
    /// component per cluster, which measures the generator rather than the
    /// index.
    fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
        let mut next = rng(seed);
        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| next() * 0.3).collect())
            .collect();
        Mat::from_fn(n, dim, |i, j| centres[i % n_clusters][j] + next())
    }

    fn build(data: &Mat<f32>, metric: Dist, nlist: Option<usize>) -> HnswRaBitQIndex<f32> {
        HnswRaBitQIndex::build(data.as_ref(), metric, 16, 200, nlist, None, None, 42, false)
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
    fn test_query_finds_self_at_k1() {
        let data = clustered(2000, 64, 20, 3);
        let index = build(&data, Dist::SquaredEuclidean, None);

        for i in (0..2000).step_by(97) {
            let query: Vec<f32> = (0..64).map(|j| data[(i, j)]).collect();
            let (got, dists) = index.query(&query, 1, 64).unwrap();
            assert_eq!(got[0], i, "vertex {i} did not find itself");
            approx::assert_abs_diff_eq!(dists[0], 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_recall_against_brute_force() {
        let data = clustered(5000, 128, 25, 7);
        let index = build(&data, Dist::SquaredEuclidean, None);
        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 was {r}");
    }

    #[test]
    fn test_cosine_recall_against_brute_force() {
        let data = clustered(5000, 128, 25, 11);
        let index = build(&data, Dist::Cosine, None);
        let r = recall(&index, &data, 10, 128, Dist::Cosine);
        assert!(r > 0.9, "cosine recall@10 was {r}");
    }

    #[test]
    fn test_single_centroid_still_recalls() {
        // One global centroid is the cheapest query path: no cluster lookup and
        // a single `||q - c||^2`. It costs code sharpness, not correctness.
        let data = clustered(5000, 128, 25, 23);
        let index = build(&data, Dist::SquaredEuclidean, Some(1));
        assert_eq!(index.nlist(), 1);
        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 at nlist 1 was {r}");
    }

    #[test]
    fn test_reported_distances_match_the_metric() {
        // Neighbours are screened on the estimate but scored exactly, so the
        // reported distances carry no quantisation error at all.
        let data = clustered(1000, 64, 10, 5);

        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let index = build(&data, metric, None);
            let query: Vec<f32> = (0..64).map(|j| data[(11, j)]).collect();
            let (got, dists) = index.query(&query, 5, 128).unwrap();

            for (&id, &d) in got.iter().zip(&dists) {
                let row: Vec<f32> = (0..64).map(|j| data[(id, j)]).collect();
                let want: f32 = match metric {
                    Dist::Cosine => {
                        let dot: f32 = row.iter().zip(&query).map(|(a, b)| a * b).sum();
                        let na = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let nb = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot / (na * nb)
                    }
                    _ => row.iter().zip(&query).map(|(a, b)| (a - b) * (a - b)).sum(),
                };
                approx::assert_relative_eq!(d, want, max_relative = 1e-3, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_higher_ef_does_not_reduce_recall() {
        let data = clustered(4000, 128, 20, 13);
        let index = build(&data, Dist::SquaredEuclidean, None);

        let mut best = 0.0f64;
        for ef in [16usize, 32, 64, 128, 256] {
            let r = recall(&index, &data, 10, ef, Dist::SquaredEuclidean);
            assert!(r >= best - 0.02, "ef {ef} dropped recall from {best} to {r}");
            best = best.max(r);
        }
    }

    #[test]
    fn test_k_larger_than_dataset_is_clamped() {
        let data = clustered(50, 32, 4, 29);
        let index = build(&data, Dist::SquaredEuclidean, None);
        let query: Vec<f32> = (0..32).map(|j| data[(3, j)]).collect();

        let (got, dists) = index.query(&query, 500, 64).unwrap();
        assert_eq!(got.len(), 50);
        assert_eq!(dists.len(), 50);

        let mut seen = got.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), got.len(), "duplicate ids in the result");
        assert!(got.iter().all(|&i| i < 50));
    }

    #[test]
    fn test_manhattan_is_rejected() {
        let data = clustered(200, 32, 4, 31);
        let err = HnswRaBitQIndex::build(
            data.as_ref(),
            Dist::Manhattan,
            16,
            200,
            None,
            None,
            None,
            42,
            false,
        );
        assert!(matches!(
            err,
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_query_rejects_wrong_dimension() {
        let data = clustered(200, 32, 4, 37);
        let index = build(&data, Dist::SquaredEuclidean, None);
        let err = index.query(&[0.0f32; 16], 5, 32);
        assert!(matches!(
            err,
            Err(AnnSearchErrors::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_generate_knn_returns_k_per_row() {
        let data = clustered(800, 64, 8, 41);
        let index = build(&data, Dist::SquaredEuclidean, None);

        let (graph, dists) = index.generate_knn(5, 64, true, false).unwrap();
        assert_eq!(graph.len(), 800);
        assert!(graph.iter().all(|row| row.len() == 5));
        assert!(dists.unwrap().iter().all(|row| row.len() == 5));
    }

    #[test]
    fn test_f64_index_builds_and_queries() {
        let data32 = clustered(600, 32, 6, 43);
        let data: Mat<f64> = Mat::from_fn(600, 32, |i, j| data32[(i, j)] as f64);

        let index: HnswRaBitQIndex<f64> =
            HnswRaBitQIndex::build(data.as_ref(), Dist::SquaredEuclidean, 16, 200, None, None, None, 42, false)
                .unwrap();

        let query: Vec<f64> = (0..32).map(|j| data[(7, j)]).collect();
        let (got, dists) = index.query(&query, 1, 64).unwrap();
        assert_eq!(got[0], 7);
        approx::assert_abs_diff_eq!(dists[0], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_hadamard_rotation_path() {
        let data = clustered(1500, 128, 12, 47);
        let index: HnswRaBitQIndex<f32> = HnswRaBitQIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            16,
            200,
            None,
            None,
            Some(RotatorKind::FhtKac),
            42,
            false,
        )
        .unwrap();

        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 under the Hadamard rotation was {r}");
    }

    #[test]
    fn test_codes_cost_one_per_vertex() {
        // The point of this index against the quantised graph: the codes do not
        // grow with the degree, so doubling `m` leaves them untouched.
        let data = clustered(2000, 128, 10, 53);

        let narrow = HnswRaBitQIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            8,
            200,
            None,
            None,
            None,
            42,
            false,
        )
        .unwrap();
        let wide = HnswRaBitQIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            32,
            200,
            None,
            None,
            None,
            42,
            false,
        )
        .unwrap();

        assert_eq!(narrow.degree(), 16);
        assert_eq!(wide.degree(), 64);
        assert_eq!(narrow.codes.len(), wide.codes.len());
        assert_eq!(narrow.codes.len(), 2000 * narrow.encoder.n_bytes);
    }
}
