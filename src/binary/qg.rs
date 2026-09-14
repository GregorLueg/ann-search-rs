//! Quantised graph index: a Vamana graph carrying its neighbours' RaBitQ codes.
//!
//! A graph walk normally pays for one random memory access per neighbour, then
//! one distance kernel per neighbour. This index removes both. Every vertex
//! stores its own neighbours' one-bit RaBitQ codes, quantised against *itself*
//! and pre-transposed into the fast-scan block layout, so one hop is a
//! contiguous read plus one byte-shuffle sweep that estimates all 32 neighbour
//! distances at once.
//!
//! ### Why the table survives the change of centre
//!
//! Quantising each vertex's neighbours against that vertex gives far better
//! one-bit codes than a single global centroid would, but naively it would also
//! mean rebuilding the query's lookup table at every hop. It does not, because
//! the estimate separates:
//!
//! ```text
//! ||q - v||^2 = ||q - c||^2 + ||v - c||^2 - 2 ||v - c|| * dc * (<Rq, s> - <Rc, s>)
//! ```
//!
//! with `c` the current vertex, `s` the neighbour's sign vector and `dc` its
//! stored dot correction. Only `<Rq, s>` involves the query, so the table is
//! built once; `<Rc, s>` folds into a per-neighbour constant at build time, and
//! `||q - c||^2` is the exact distance to the vertex that was just popped,
//! which the walk computes anyway. Rearranged, a lane costs one multiply and
//! one add on top of the shuffle:
//!
//! ```text
//! est = f_add + g_add + f_rescale * <Rq, s>
//! ```
//!
//! ### What this costs
//!
//! Each vector's code is duplicated once per in-edge, so the codes alone are
//! `degree * padded_dim / 8` bytes per vertex, on top of the raw vectors the
//! exact distances still need. The index is bigger than plain HNSW; it trades
//! memory for locality.
//!
//! ### What is not from the paper
//!
//! The graph. SymphonyQG builds its own with random init, repeated
//! search-prune-reverse rounds and a cosine-threshold refill whose purpose is
//! to force exact-degree regularity. [`VamanaIndex`] already produces a
//! fixed-degree graph, so it builds the topology here and the contribution kept
//! is the storage layout and the estimator.
//!
//! ### References
//!
//! Gou et al., "SymphonyQG: Towards Symphonious Integration of Quantization and
//! Graph for Approximate Nearest Neighbor Search", SIGMOD 2025.

use faer::RowRef;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::binary::rabitq::RaBitQEncoder;
use crate::binary::rabitq_fastscan::{
    build_sign_lut, pack_rabitq_blocked, score_sign_block, unpack_rabitq_blocked, BLOCKED_ARCH,
};
use crate::binary::rotator::RotatorKind;
use crate::binary::turboquant::pack::BLOCK;
use crate::cpu::vamana::{VamanaIndex, VamanaState};
use crate::prelude::*;
use crate::utils::graph_utils::ThreadLocalSearchState;
use crate::utils::pack_knn_results;
use crate::utils::KnnValidation;

////////////
// Consts //
////////////

/// Neighbours estimated per fast-scan sweep.
///
/// Fixed by the kernels, which score exactly [`BLOCK`] lanes at a time. The
/// degree bound is a multiple of this so no sweep is wasted on empty lanes.
pub const QG_BATCH: usize = BLOCK;

/// Sentinel marking an unused neighbour slot, as [`VamanaIndex`] writes it.
const SENTINEL: u32 = u32::MAX;

/// Beam width [`KnnValidation`] queries at.
///
/// Wide enough that a recall shortfall is the index's and not the beam's,
/// which is the whole point of the harness.
const QG_VALIDATION_EF: usize = 256;

/// Default degree bound.
///
/// One sweep per hop. Doubling it doubles both the code footprint and the
/// per-hop work, so it wants a reason.
pub const DEFAULT_QG_DEGREE: usize = 32;

/////////////
// QgIndex //
/////////////

/// Quantised graph index over one-bit RaBitQ codes.
///
/// Cosine is served by normalising the stored vectors at build, so everything
/// internal is squared Euclidean and the reported distance is converted back
/// on the way out. The original vector magnitudes are not kept.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct QgIndex<T> {
    /// Row-major vectors, `n * dim`. Unit length when `metric` is cosine.
    vectors_flat: Vec<T>,
    /// Per-vector norms, all ones for cosine and empty otherwise
    norms: Vec<T>,
    /// The rotation the codes were taken in
    encoder: RaBitQEncoder<T>,
    /// Adjacency, `n * degree`, packed at the front and [`SENTINEL`]-padded
    edges: Vec<u32>,
    /// Neighbour codes, `n * n_batches` blocks of `n_bytes * QG_BATCH` bytes
    codes: Vec<u8>,
    /// Per-slot additive term, `n * degree`
    f_add: Vec<T>,
    /// Per-slot multiplier on the table score, `n * degree`
    f_rescale: Vec<T>,
    /// The [`BLOCKED_ARCH`] tag `codes` was packed with
    blocked_arch: u8,
    /// Where every walk starts, the graph's medoid
    entry_point: u32,
    /// Neighbour slots per vertex, a multiple of [`QG_BATCH`]
    degree: usize,
    /// `degree / QG_BATCH`
    n_batches: usize,
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

impl<T> VectorDistance<T> for QgIndex<T>
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

impl<T> DimensionValidation for QgIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////
// Helpers //
/////////////

/// Signed dot product between a one-bit code and a vector.
///
/// A set bit means `+1`, a clear bit `-1`, matching what
/// [`RaBitQEncoder::encode_vector`] writes and what the fast-scan table scores.
///
/// ### Params
///
/// * `code` - Packed sign bits, `vec.len() / 8` bytes
/// * `vec` - Vector in the rotated frame
///
/// ### Returns
///
/// `sum_d sign(code_d) * vec[d]`
#[inline]
fn signed_dot<T>(code: &[u8], vec: &[T]) -> T
where
    T: Float,
{
    let mut acc = T::zero();
    for (d, &x) in vec.iter().enumerate() {
        acc = acc
            + if code[d / 8] & (1u8 << (d % 8)) != 0 {
                x
            } else {
                -x
            };
    }
    acc
}

/// Encode a residual that is already in the rotated frame.
///
/// The rotation is linear and orthogonal, so `R(v - c)` equals `Rv - Rc` and
/// its norm equals `||v - c||`. Rotating every vector once and differencing
/// there therefore gives exactly what
/// [`RaBitQEncoder::encode_vector`] computes, for one rotation per vector
/// instead of one per edge.
///
/// Normalisation drops out too: the sign bits of the unit residual are the
/// sign bits of the residual, and the unit residual's L1 norm is
/// `||res||_1 / ||res||`.
///
/// ### Params
///
/// * `res` - The residual in the rotated frame, `padded_dim` long
/// * `code` - Output sign bits, `padded_dim / 8` bytes, overwritten in full
///
/// ### Returns
///
/// `(||v - c||, inverse dot correction)`, the latter zero where the residual
/// underflowed
#[inline]
fn encode_rotated_residual<T>(res: &[T], code: &mut [u8]) -> (T, T)
where
    T: Float + FromPrimitive,
{
    code.fill(0);

    let mut sum_sq = T::zero();
    let mut l1 = T::zero();
    for (d, &x) in res.iter().enumerate() {
        sum_sq = sum_sq + x * x;
        l1 = l1 + x.abs();
        if x >= T::zero() {
            code[d / 8] |= 1u8 << (d % 8);
        }
    }

    let v_dist = sum_sq.sqrt();

    // `encode_vector` guards on the L1 norm of the *unit* residual, which is
    // `l1 / v_dist`; the same condition written without the divide.
    let floor = T::from_f32(1e-6).unwrap();
    let dot_correction_inv = if v_dist > T::epsilon() && l1 > floor * v_dist {
        v_dist / l1
    } else {
        T::zero()
    };

    (v_dist, dot_correction_inv)
}

impl<T> QgIndex<T>
where
    T: AnnSearchFloat + ComplexField + ThreadLocalSearchState,
    VamanaIndex<T>: VamanaState<T>,
{
    /// Build a quantised graph index.
    ///
    /// The topology comes from [`VamanaIndex`]; this adds the per-vertex
    /// neighbour codes on top of it.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix, `n` samples by `dim` features
    /// * `metric` - Distance metric, squared Euclidean or cosine
    /// * `degree` - Neighbour slots per vertex, a multiple of [`QG_BATCH`]
    /// * `l_build` - Beam width during graph construction
    /// * `l_build_pass1` - Beam width for Vamana's first pass, `None` to reuse
    ///   `l_build`
    /// * `alpha_pass1` - Vamana prune slack, first pass
    /// * `alpha_pass2` - Vamana prune slack, second pass
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let the
    ///   dimensionality decide
    /// * `seed` - Random seed, for reproducibility
    ///
    /// ### Returns
    ///
    /// The index, or an error on an unsupported metric or an invalid degree
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        degree: usize,
        l_build: usize,
        l_build_pass1: Option<usize>,
        alpha_pass1: f32,
        alpha_pass2: f32,
        rotator_kind: Option<RotatorKind>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        if degree == 0 || !degree.is_multiple_of(QG_BATCH) {
            return Err(AnnSearchErrors::QgInvalidDegree {
                degree,
                batch: QG_BATCH,
            });
        }

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

        // The graph is built in the space the walk searches, which for cosine
        // is the normalised one.
        let graph_index = VamanaIndex::build(
            (&vectors_flat[..], n, dim),
            Dist::SquaredEuclidean,
            degree,
            l_build,
            l_build_pass1,
            alpha_pass1,
            alpha_pass2,
            seed,
        );
        let edges = graph_index.graph;
        let entry_point = graph_index.medoid;

        let encoder = match rotator_kind {
            Some(kind) => {
                RaBitQEncoder::with_rotator_kind(dim, Dist::SquaredEuclidean, kind, seed as u64)?
            }
            None => RaBitQEncoder::new(dim, Dist::SquaredEuclidean, seed as u64),
        };
        let padded_dim = encoder.padded_dim;
        let n_bytes = encoder.n_bytes;
        let n_batches = degree / QG_BATCH;

        let mut codes = vec![0u8; n * n_batches * n_bytes * QG_BATCH];
        let mut f_add = vec![T::zero(); n * degree];
        let mut f_rescale = vec![T::zero(); n * degree];

        let block_bytes = n_bytes * QG_BATCH;
        let two = T::one() + T::one();

        // One rotation per vector rather than one per edge. Every residual the
        // encoder needs is a difference of two of these, so this turns
        // `n * degree` transforms into `n`. Freed before the index is returned.
        let mut rotated = vec![T::zero(); n * padded_dim];
        rotated
            .par_chunks_mut(padded_dim)
            .enumerate()
            .for_each(|(i, out)| {
                encoder
                    .rotator
                    .rotate_into(&vectors_flat[i * dim..(i + 1) * dim], out);
            });

        codes
            .par_chunks_mut(n_batches * block_bytes)
            .zip(f_add.par_chunks_mut(degree))
            .zip(f_rescale.par_chunks_mut(degree))
            .enumerate()
            .try_for_each(
                |(node, ((code_block, add_block), rescale_block))| -> Result<(), AnnSearchErrors> {
                    let centre_rot = &rotated[node * padded_dim..(node + 1) * padded_dim];
                    let slots = &edges[node * degree..(node + 1) * degree];

                    let mut raw = vec![0u8; degree * n_bytes];
                    let mut res = vec![T::zero(); padded_dim];

                    for (slot, &nb) in slots.iter().enumerate() {
                        if nb == SENTINEL {
                            // An empty lane must never win. Zero scale makes
                            // the table score irrelevant and the infinity keeps
                            // it out of the beam whatever the exact term is.
                            add_block[slot] = T::infinity();
                            rescale_block[slot] = T::zero();
                            continue;
                        }

                        let nb = nb as usize;
                        let nb_rot = &rotated[nb * padded_dim..(nb + 1) * padded_dim];
                        for d in 0..padded_dim {
                            res[d] = nb_rot[d] - centre_rot[d];
                        }

                        let code = &mut raw[slot * n_bytes..(slot + 1) * n_bytes];
                        let (v_dist, dot_correction_inv) = encode_rotated_residual(&res, code);

                        let rescale = (two * v_dist * dot_correction_inv).neg();
                        rescale_block[slot] = rescale;
                        add_block[slot] = v_dist * v_dist - rescale * signed_dot(code, centre_rot);
                    }

                    for b in 0..n_batches {
                        let packed = pack_rabitq_blocked(
                            &raw[b * QG_BATCH * n_bytes..(b + 1) * QG_BATCH * n_bytes],
                            QG_BATCH,
                            n_bytes,
                        );
                        code_block[b * block_bytes..(b + 1) * block_bytes]
                            .copy_from_slice(&packed.data);
                    }

                    Ok(())
                },
            )?;

        Ok(Self {
            vectors_flat,
            norms,
            encoder,
            edges,
            codes,
            f_add,
            f_rescale,
            blocked_arch: BLOCKED_ARCH,
            entry_point,
            degree,
            n_batches,
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
    /// cosine distance because the stored vectors are unit length. That is the
    /// quantity the estimator's algebra is written in; the conversion back
    /// happens once, on the returned neighbours.
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

        // One rotation and one table for the whole walk. That is the point of
        // folding the per-vertex centre into `f_add` at build time.
        let q_rot = self.encoder.apply_rotation(&query);
        let lut = build_sign_lut(&q_rot)?;

        T::with_search_state(|state| {
            state.reset(self.n);
            state.results.reset(ef);

            let mut lanes = [0.0f32; QG_BATCH];
            state.candidates.push(Reverse((
                OrderedFloat(T::neg_infinity()),
                self.entry_point as usize,
            )));

            while let Some(Reverse((est, node))) = state.candidates.pop() {
                // The beam is ordered on estimates but gated on exact
                // distances, so this compares the two. That is sound for
                // stopping: an estimate already worse than the k-th exact
                // distance cannot lead anywhere better than the estimate
                // error, and without it the walk drains every candidate it
                // ever admitted, paying a full exact distance for each.
                if est.0 > state.results.threshold() {
                    break;
                }
                if state.is_visited(node) {
                    continue;
                }
                state.mark_visited(node);

                // Exact for the vertex itself, which is both what the results
                // carry and the anchor term every neighbour estimate needs.
                let g_add = self.exact(&query, node);
                state.results.push(g_add, node);
                let threshold = state.results.threshold();

                let slots = &self.edges[node * self.degree..(node + 1) * self.degree];
                let add = &self.f_add[node * self.degree..(node + 1) * self.degree];
                let rescale = &self.f_rescale[node * self.degree..(node + 1) * self.degree];

                for b in 0..self.n_batches {
                    score_sign_block(&lut, &self.codes, node * self.n_batches + b, &mut lanes);

                    for lane in 0..QG_BATCH {
                        let slot = b * QG_BATCH + lane;
                        let nb = slots[slot];
                        if nb == SENTINEL {
                            continue;
                        }

                        let est =
                            add[slot] + g_add + rescale[slot] * T::from_f32(lanes[lane]).unwrap();

                        if est < threshold && !state.is_visited(nb as usize) {
                            state
                                .candidates
                                .push(Reverse((OrderedFloat(est), nb as usize)));
                        }
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
    /// Number of coordinates the codes span
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Neighbour slots per vertex.
    ///
    /// ### Returns
    ///
    /// The degree bound
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The metric this index answers in.
    ///
    /// ### Returns
    ///
    /// The metric
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Original row indices.
    ///
    /// ### Returns
    ///
    /// Slice of original indices
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
            + self.edges.capacity() * std::mem::size_of::<u32>()
            + self.codes.capacity()
            + self.f_add.capacity() * std::mem::size_of::<T>()
            + self.f_rescale.capacity() * std::mem::size_of::<T>()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

impl<T> QgIndex<T>
where
    T: AnnSearchFloat,
{
    /// Re-block the neighbour codes for this machine if they were packed on
    /// another.
    ///
    /// A no-op in the common case. The blocked byte order is
    /// architecture-specific, so this runs after a load, before anything reads
    /// the codes.
    pub fn reblock_for_this_arch(&mut self) {
        if self.blocked_arch == BLOCKED_ARCH {
            return;
        }

        let n_bytes = self.encoder.n_bytes;
        let block_bytes = n_bytes * QG_BATCH;
        let src = self.blocked_arch;

        self.codes.par_chunks_mut(block_bytes).for_each(|block| {
            let raw = unpack_rabitq_blocked(block, QG_BATCH, n_bytes, src);
            block.copy_from_slice(&pack_rabitq_blocked(&raw, QG_BATCH, n_bytes).data);
        });

        self.blocked_arch = BLOCKED_ARCH;
    }
}

////////////////////
// KnnValidation  //
////////////////////

impl<T> KnnValidation<T> for QgIndex<T>
where
    T: AnnSearchFloat + ComplexField + ThreadLocalSearchState,
    VamanaIndex<T>: VamanaState<T>,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, QG_VALIDATION_EF)
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
impl<T> crate::serialise::IndexIo for QgIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "qg";

    fn load_aux(&mut self, _dir: &std::path::Path) -> Result<(), AnnSearchErrors> {
        // The blocked byte order is architecture-specific, so a bundle written
        // elsewhere is re-blocked before anything reads the codes.
        self.reblock_for_this_arch();
        Ok(())
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

    /// Clusters whose spread is wider than their separation.
    ///
    /// The overlap is deliberate. Well-separated blobs in high dimensions leave
    /// Vamana's prune with no reason to keep a single cross-cluster edge, so the
    /// graph splits into one component per cluster and *any* index built on it
    /// scores exactly `1 / n_clusters`. That measures the generator, not the
    /// index.
    fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
        let mut next = rng(seed);
        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| next() * 0.3).collect())
            .collect();
        Mat::from_fn(n, dim, |i, j| centres[i % n_clusters][j] + next())
    }

    fn build(data: &Mat<f32>, metric: Dist, degree: usize) -> QgIndex<f32> {
        QgIndex::build(data.as_ref(), metric, degree, 128, None, 1.2, 1.2, None, 42).unwrap()
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

    fn recall(index: &QgIndex<f32>, data: &Mat<f32>, k: usize, ef: usize, metric: Dist) -> f64 {
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
        let index = build(&data, Dist::SquaredEuclidean, 32);

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
        let index = build(&data, Dist::SquaredEuclidean, 32);
        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 was {r}");
    }

    #[test]
    fn test_cosine_recall_against_brute_force() {
        let data = clustered(5000, 128, 25, 11);
        let index = build(&data, Dist::Cosine, 32);
        let r = recall(&index, &data, 10, 128, Dist::Cosine);
        assert!(r > 0.9, "cosine recall@10 was {r}");
    }

    #[test]
    fn test_reported_distances_match_the_metric() {
        // Euclidean reports the squared distance and cosine reports `1 - cos`,
        // which is what every other graph index in the crate returns.
        let data = clustered(1000, 64, 10, 5);

        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let index = build(&data, metric, 32);
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
        let index = build(&data, Dist::SquaredEuclidean, 32);

        let mut best = 0.0f64;
        for ef in [16usize, 32, 64, 128, 256] {
            let r = recall(&index, &data, 10, ef, Dist::SquaredEuclidean);
            assert!(
                r >= best - 0.02,
                "ef {ef} dropped recall from {best} to {r}"
            );
            best = best.max(r);
        }
    }

    #[test]
    fn test_wider_degree_builds_and_searches() {
        let data = clustered(3000, 128, 15, 17);
        let index = build(&data, Dist::SquaredEuclidean, 64);
        assert_eq!(index.degree(), 64);
        assert_eq!(index.n_batches, 2);
        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 at degree 64 was {r}");
    }

    #[test]
    fn test_padding_lanes_never_reach_the_results() {
        // A dataset barely larger than the degree leaves slots sentinel, so
        // every hop scans partly empty lanes.
        let data = clustered(40, 64, 3, 19);
        let index = build(&data, Dist::SquaredEuclidean, 32);

        let query: Vec<f32> = (0..64).map(|j| data[(5, j)]).collect();
        let (got, _) = index.query(&query, 40, 64).unwrap();
        assert!(got.iter().all(|&i| i < 40));
        let mut seen = got.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), got.len(), "a vertex was returned twice");
    }

    #[test]
    fn test_k_larger_than_dataset_is_clamped() {
        let data = clustered(100, 64, 5, 23);
        let index = build(&data, Dist::SquaredEuclidean, 32);
        let query: Vec<f32> = (0..64).map(|j| data[(1, j)]).collect();
        let (got, dists) = index.query(&query, 500, 64).unwrap();
        assert_eq!(got.len(), 100);
        assert_eq!(dists.len(), 100);
    }

    #[test]
    fn test_manhattan_is_rejected() {
        let data = clustered(100, 64, 5, 29);
        assert!(matches!(
            QgIndex::build(
                data.as_ref(),
                Dist::Manhattan,
                32,
                64,
                None,
                1.2,
                1.2,
                None,
                42
            ),
            Err(AnnSearchErrors::DistanceNotSupported(_))
        ));
    }

    #[test]
    fn test_degree_must_be_a_whole_number_of_batches() {
        let data = clustered(100, 64, 5, 31);
        for degree in [0usize, 1, 31, 33, 48] {
            assert!(
                matches!(
                    QgIndex::build(
                        data.as_ref(),
                        Dist::SquaredEuclidean,
                        degree,
                        64,
                        None,
                        1.2,
                        1.2,
                        None,
                        42
                    ),
                    Err(AnnSearchErrors::QgInvalidDegree { .. })
                ),
                "degree {degree} was accepted"
            );
        }
    }

    #[test]
    fn test_query_rejects_wrong_dimension() {
        let data = clustered(200, 64, 5, 37);
        let index = build(&data, Dist::SquaredEuclidean, 32);
        assert!(matches!(
            index.query(&vec![0.0f32; 63], 5, 64),
            Err(AnnSearchErrors::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_generate_knn_returns_k_per_row() {
        let data = clustered(600, 64, 8, 41);
        let index = build(&data, Dist::SquaredEuclidean, 32);
        let (ids, dists) = index.generate_knn(5, 64, true, false).unwrap();

        assert_eq!(ids.len(), 600);
        assert!(ids.iter().all(|row| row.len() == 5));
        assert!(dists.unwrap().iter().all(|row| row.len() == 5));
    }

    #[test]
    fn test_f64_index_builds_and_queries() {
        let data32 = clustered(1000, 64, 10, 43);
        let data = Mat::<f64>::from_fn(1000, 64, |i, j| data32[(i, j)] as f64);
        let index = QgIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            32,
            128,
            None,
            1.2,
            1.2,
            None,
            42,
        )
        .unwrap();

        let query: Vec<f64> = (0..64).map(|j| data[(7, j)]).collect();
        let (got, _) = index.query(&query, 1, 64).unwrap();
        assert_eq!(got[0], 7);
    }

    #[test]
    fn test_hadamard_rotation_path() {
        // Past the auto threshold the encoder pads, so the codes span more
        // coordinates than the data has.
        let data = clustered(3000, 200, 15, 47);
        let index = build(&data, Dist::SquaredEuclidean, 32);
        assert_eq!(index.padded_dim(), 256);
        let r = recall(&index, &data, 10, 128, Dist::SquaredEuclidean);
        assert!(r > 0.9, "recall@10 with the padded rotation was {r}");
    }

    #[test]
    fn test_reblocking_a_foreign_layout_preserves_results() {
        use crate::binary::rabitq_fastscan::pack_rabitq_blocked_for;

        let data = clustered(2000, 128, 12, 53);
        let mut index = build(&data, Dist::SquaredEuclidean, 32);

        let query: Vec<f32> = (0..128).map(|j| data[(21, j)]).collect();
        let before = index.query(&query, 10, 128).unwrap();

        // Pretend the codes arrived from the other architecture: repack them
        // into that order, relabel, and let the load path put them back.
        let n_bytes = index.encoder.n_bytes;
        let block_bytes = n_bytes * QG_BATCH;
        let foreign = if BLOCKED_ARCH == 1 { 0 } else { 1 };
        for block in index.codes.chunks_mut(block_bytes) {
            let raw = unpack_rabitq_blocked(block, QG_BATCH, n_bytes, BLOCKED_ARCH);
            block.copy_from_slice(&pack_rabitq_blocked_for(&raw, QG_BATCH, n_bytes, foreign).data);
        }
        index.blocked_arch = foreign;
        index.reblock_for_this_arch();

        assert_eq!(index.blocked_arch, BLOCKED_ARCH);
        assert_eq!(index.query(&query, 10, 128).unwrap(), before);
    }
}
