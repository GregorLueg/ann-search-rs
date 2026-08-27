//! Navigating Spreading-out Graph (NSG) index.
//!
//! NSG (Fu et al., 2017) builds a directed navigable graph on top of an
//! approximate kNN graph by applying the MRNG (Monotonic Relative Neighbour
//! Graph) pruning rule and patching connectivity via a DFS from a single
//! navigating node. Query time is a greedy beam search from that navigating
//! node.
//!
//! ### Build pipeline
//!
//! 1. Compute the centroid of the data, snap it to the nearest data point
//!    via one beam-search hop on the input kNN graph. That data point is the
//!    navigating node `n_p`.
//! 2. In parallel for every node `v`, run a bounded beam-search from `n_p`
//!    toward `v` on the kNN graph, collect the visited set, union in `v`'s
//!    kNN neighbours, sort ascending by distance to `v`, and apply the MRNG
//!    prune to select at most `R` outgoing edges. After reverse-edge
//!    insertion, top nodes below degree `R` back up with their closest kNN
//!    entries: on clustered data MRNG occlusion alone collapses to a handful
//!    of edges, and the resulting sparse graph dead-ends queries.
//! 3. Sequentially run a DFS from `n_p` on the partial NSG. For each
//!    unreachable node `u`, find its nearest reachable predecessor `t` and
//!    add the edge `t -> u`, evicting `t`'s farthest neighbour if it is at
//!    the degree cap.
//! 4. Flatten the adjacency into a contiguous `Vec<u32>` for cache-friendly
//!    query traversal.
//!
//! ### References
//!
//! Fu, C., Xiang, C., Wang, C. and Cai, D. *Fast Approximate Nearest Neighbor
//! Search With The Navigating Spreading-out Graph.* PVLDB 12(5), 2017.
//! arXiv:1707.00143.

use faer::{RowRef};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::{RefCell, UnsafeCell};
use std::cmp::{Ordering, Reverse};
use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};
use std::time::Instant;
use thousands::*;

use crate::cpu::nndescent::{NNDescent, NNDescentQuery};
use crate::prelude::*;
use crate::utils::dist::{cosine_distance_static, euclidean_distance_static};
use crate::utils::graph_utils::SearchState;
use crate::utils::nndescent_utils::{ApplySortedUpdates, SENTINEL_PID};
use crate::utils::parallelism::StripedLocks;
use crate::utils::*;

///////////
// Conts //
///////////

/// Lower clamp for the number of query-time entry seeds strided evenly
/// across the dataset (on top of the navigating node). Single-entry search
/// from the medoid dead-ends on clustered data at scale: cross-cluster paths
/// outgrow the beam and a small fraction of queries terminates in the wrong
/// region. The seed count resolves to `sqrt(n)` clamped to
/// `[NSG_ENTRY_SEEDS_MIN, NSG_ENTRY_SEEDS_MAX]`; each seed costs one
/// distance evaluation and far-from-query seeds are never expanded. At 100k
/// samples (~316 seeds) this raised recall@15 from 0.97 to 0.999 at ef=50
/// for ~7% query-time cost on clustered cell-embedding data.
const NSG_ENTRY_SEEDS_MIN: usize = 64;

/// Upper clamp for the query-time entry-seed count. See
/// [`NSG_ENTRY_SEEDS_MIN`] for the heuristic. Caps the per-query seeding
/// cost at large `n`, where the graph itself carries more of the
/// navigation.
const NSG_ENTRY_SEEDS_MAX: usize = 512;

///////////////////
// Thread locals //
///////////////////

thread_local! {
    static NSG_BUILD_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1024));
    static NSG_BUILD_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1024));
    static NSG_SEARCH_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1024));
    static NSG_SEARCH_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1024));
}

/// Thread-local state accessors for NSG build and query paths.
///
/// Splits `SearchState<T>` across two epochs so a query cannot clobber a
/// build in progress on the same thread (relevant when a caller builds NSG
/// on one Rayon pool and queries on another).
pub trait NsgState<T> {
    /// Access the thread-local search state used during index construction.
    ///
    /// ### Params
    ///
    /// * `f` - Closure operating on the borrowed cell
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;

    /// Access the thread-local search state used during query traversal.
    ///
    /// ### Params
    ///
    /// * `f` - Closure operating on the borrowed cell
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;
}

impl NsgState<f32> for NsgIndex<f32> {
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        NSG_BUILD_STATE_F32.with(f)
    }

    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        NSG_SEARCH_STATE_F32.with(f)
    }
}

impl NsgState<f64> for NsgIndex<f64> {
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        NSG_BUILD_STATE_F64.with(f)
    }

    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        NSG_SEARCH_STATE_F64.with(f)
    }
}

////////////
// Params //
////////////

/// Build-time parameters for [`NsgIndex`].
///
/// ### Fields
#[derive(Clone, Copy, Debug)]
pub struct NsgBuildParams {
    /// Target out-degree of the NSG graph (`R` in the paper). MRNG pruning
    /// selects at most `r` diversified edges; nodes left below `r` are then
    /// topped up with their closest kNN entries, so `r` is the actual degree
    /// for almost every node. Typical values: 32–70 for SIFT-scale, higher
    /// for harder datasets.
    pub r: usize,
    /// Beam width for the per-node search on the input kNN graph during build
    /// (`L` in the paper). Larger `l_build` yields more candidates per node.
    pub l_build: usize,
    /// Cap on the size of the candidate set `E` before MRNG pruning. Absent
    /// from the paper but standard in reference implementations to bound
    /// per-node prune cost. Roughly 500 works across scales.
    pub c: usize,
    /// Degree of the input kNN graph when NSG builds one internally. Larger
    /// `knn_k` gives better MRNG candidates but slower NN-Descent. Ignored by
    /// `build_from_nndescent`.
    pub knn_k: usize,
}

/// Default implementation of the [NsgBuildParams].
impl Default for NsgBuildParams {
    fn default() -> Self {
        Self {
            r: 32,
            l_build: 100,
            c: 500,
            knn_k: 64,
        }
    }
}

impl NsgBuildParams {
    /// Construct a parameter set with explicit values.
    ///
    /// ### Params
    ///
    /// * `r` - Target out-degree
    /// * `l_build` - Build-time beam width
    /// * `c` - Cap on candidate-set size before pruning
    /// * `knn_k` - Degree of the internal kNN graph
    ///
    /// ### Returns
    ///
    /// A fully-specified [`NsgBuildParams`].
    pub fn new(r: usize, l_build: usize, c: usize, knn_k: usize) -> Self {
        Self {
            r,
            l_build,
            c,
            knn_k,
        }
    }
}

////////////////////////
// Construction graph //
////////////////////////

/// Concurrent build-time graph.
///
/// Each node owns a `Vec<(u32, T)>` of neighbour ids **with cached distances
/// from the source node**. Caching the distances keeps the InterInsert step
/// cheap: an at-capacity re-prune no longer needs to recompute the source's
/// distances to its existing neighbours before running MRNG.
///
/// Concurrent access uses a striped spin-lock (`StripedLocks`). Both the
/// primary MRNG write for source `v` and the InterInsert writes to targets
/// `u ∈ neighbours(v)` acquire `locks.lock_guard(node)` before touching a
/// slot. The DFS connectivity-fix step is sequential and does not need to
/// lock.
struct NsgConstructionGraph<T> {
    /// Per-node adjacency lists as `(neighbour_id, dist_from_source)`.
    /// Length is soft-capped at `r` during MRNG and InterInsert, but the
    /// DFS-fix path may push past `r` to guarantee reachability.
    nodes: Vec<UnsafeCell<Vec<(u32, T)>>>,
    /// Target out-degree cap
    r: usize,
    /// Striped spin-locks for concurrent edge updates during construction
    locks: StripedLocks,
}

unsafe impl<T> Sync for NsgConstructionGraph<T> {}

impl<T: Copy + PartialOrd> NsgConstructionGraph<T> {
    /// Allocate an empty construction graph.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `r` - Target out-degree per node (soft cap)
    /// * `threads` - Expected number of concurrent writers
    ///
    /// ### Returns
    ///
    /// Empty graph with `n` slots pre-allocated to capacity `r`.
    fn new(n: usize, r: usize, threads: usize) -> Self {
        let nodes = (0..n)
            .map(|_| UnsafeCell::new(Vec::with_capacity(r)))
            .collect();
        Self {
            nodes,
            r,
            locks: StripedLocks::new(threads, r),
        }
    }

    /// Read-only view of a node's neighbours.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    ///
    /// ### Safety
    ///
    /// Caller must ensure no concurrent write to `nodes[node_id]`.
    #[inline]
    unsafe fn get_neighbours_slice(&self, node_id: usize) -> &[(u32, T)] {
        &*self.nodes[node_id].get()
    }

    /// Overwrite a node's adjacency list.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `neighbours` - New adjacency list
    ///
    /// ### Safety
    ///
    /// Caller must hold exclusive access to `nodes[node_id]` (via the striped
    /// lock during parallel build, or by construction during the sequential
    /// DFS-fix step).
    #[inline]
    unsafe fn set_neighbours(&self, node_id: usize, neighbours: &[(u32, T)]) {
        let slot = &mut *self.nodes[node_id].get();
        slot.clear();
        slot.extend_from_slice(neighbours);
    }

    /// Unconditionally append a neighbour to a node's adjacency.
    ///
    /// Used by the DFS connectivity fix. Matches the reference NSG
    /// `findroot` semantics: the added edge is required to make `new_neighbour`
    /// reachable, so we accept degree growth beyond `r` rather than silently
    /// drop the patch. The consequent variable per-row degree is absorbed by
    /// `into_flat`, which pads every row to the observed maximum.
    ///
    /// ### Params
    ///
    /// * `node_id` - Source node
    /// * `new_neighbour` - Neighbour id to insert
    /// * `new_dist` - Distance from `node_id` to `new_neighbour`
    ///
    /// ### Safety
    ///
    /// Caller must hold exclusive access to `nodes[node_id]`. In practice
    /// this is called only from the sequential DFS-fix phase.
    #[inline]
    unsafe fn push_neighbour_unchecked(&self, node_id: usize, new_neighbour: u32, new_dist: T) {
        let slot = &mut *self.nodes[node_id].get();
        slot.push((new_neighbour, new_dist));
    }

    /// Consume the construction graph into a flat `Vec<u32>` and the stride.
    ///
    /// The stride is the maximum per-node degree observed across all rows,
    /// floored at `r`. Shorter rows tail-pad with [`u32::MAX`] so the query
    /// path can break at the first sentinel. This handles the DFS-fix nodes
    /// whose degree grew past `r`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(flat adjacency, stride)`. The flat array is
    /// `nodes.len() * stride` entries long.
    fn into_flat(self) -> (Vec<u32>, usize) {
        let n = self.nodes.len();
        let r = self.r;
        let stride = self
            .nodes
            .iter()
            .map(|cell| unsafe { (*cell.get()).len() })
            .max()
            .unwrap_or(0)
            .max(r);

        let mut flat = Vec::with_capacity(n * stride);
        for cell in self.nodes {
            let neighbours = cell.into_inner();
            let deg = neighbours.len();
            for (id, _) in &neighbours {
                flat.push(*id);
            }
            for _ in deg..stride {
                flat.push(u32::MAX);
            }
        }
        (flat, stride)
    }
}

////////////////
// Main index //
////////////////

/// NSG index for approximate nearest neighbour search.
///
/// The final layout is a flat `n * R` `u32` adjacency array plus a single
/// entry-point node id (the *navigating node*). Search is a greedy beam
/// walk over the graph starting from that entry point.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct NsgIndex<T> {
    /// Row-major flattened vectors
    pub vectors_flat: Vec<T>,
    /// Number of vectors
    pub n: usize,
    /// Dimensionality of vectors
    pub dim: usize,
    /// Pre-computed L2 norms (Cosine only; empty otherwise)
    pub norms: Vec<T>,
    /// Distance metric
    pub metric: Dist,
    /// Flat adjacency array of size `n * max_degree`
    pub graph: Vec<u32>,
    /// Row stride in `graph`. Equal to `r` for the vast majority of nodes;
    /// occasionally slightly larger when the DFS connectivity-fix pushed
    /// past `r` to guarantee reachability. Reader-side sentinel scans stop
    /// at [`u32::MAX`], so the stride only affects the length of each row.
    pub max_degree: usize,
    /// Target out-degree cap (`R` in the paper)
    pub r: usize,
    /// Build-time beam width (kept for reference / re-runs)
    pub l_build: usize,
    /// Cap on the candidate-set size used during pruning
    pub c: usize,
    /// Global entry point selected by the build
    pub navigating_node: u32,
    /// Original ids (identity by default)
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for NsgIndex<T>
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

impl<T> DimensionValidation for NsgIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////
// Helpers //
/////////////

/// Compute the centroid (mean vector) of the data.
///
/// Sequential loop; the work is `O(n * dim)` and used exactly once during
/// build.
///
/// ### Params
///
/// * `vectors_flat` - Row-major flat vectors
/// * `n` - Number of vectors
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// Owned mean vector of length `dim`.
fn compute_centroid<T: AnnSearchFloat>(vectors_flat: &[T], n: usize, dim: usize) -> Vec<T> {
    let mut centroid = vec![T::zero(); dim];
    for i in 0..n {
        let base = i * dim;
        for d in 0..dim {
            centroid[d] = centroid[d] + vectors_flat[base + d];
        }
    }
    if n > 0 {
        let inv_n = T::one() / T::from_usize(n).unwrap();
        for d in 0..dim {
            centroid[d] = centroid[d] * inv_n;
        }
    }
    centroid
}

//////////////
// MetricFn //
//////////////

/// Static-dispatch metric selector for the NSG build kernels.
///
/// `collect_and_prune_for_node` and `mrng_prune` monomorphise over this trait
/// so the runtime `Dist` branch is stripped out of the two hottest loops in the
/// build.
trait MetricFn<T: AnnSearchFloat> {
    /// Distance between the hoisted vector `vec_p` (of some node `p`
    /// with pre-fetched norm `norm_p` when Cosine) and the vector at
    /// internal index `q`.
    fn distance_from_vec(idx: &NsgIndex<T>, vec_p: &[T], norm_p: T, q: usize) -> T;
}

/// Squared Euclidean metric.
struct SqEuclidMetric;

impl<T: AnnSearchFloat> MetricFn<T> for SqEuclidMetric {
    /// Calculate squared Euclidean
    ///
    /// ### Params
    ///
    /// * `idx` - The [NsgIndex] index.
    /// * `vec_p` - The query vector.
    /// * `_norm_p` - The norm of the query (unused)
    /// * `q` - Index of the DB vector
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance between idx and vec_p
    #[inline(always)]
    fn distance_from_vec(idx: &NsgIndex<T>, vec_p: &[T], _norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        T::euclidean_simd(vec_p, vec_q)
    }
}

/// Cosine metric (uses `NsgIndex::norms`).
struct CosineMetric;

impl<T: AnnSearchFloat> MetricFn<T> for CosineMetric {
    /// Calculate Cosine distance
    ///
    /// ### Params
    ///
    /// * `idx` - The [NsgIndex] index.
    /// * `vec_p` - The query vector.
    /// * `norm_p` - The norm of the query
    /// * `q` - Index of the DB vector
    ///
    /// ### Returns
    ///
    /// The Cosine distance between idx and vec_p
    #[inline(always)]
    fn distance_from_vec(idx: &NsgIndex<T>, vec_p: &[T], norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        let dot = T::dot_simd(vec_p, vec_q);
        T::one() - (dot / (norm_p * idx.norms[q]))
    }
}

/// Manhattan (L1) metric.
struct ManhattanMetric;

impl<T: AnnSearchFloat> MetricFn<T> for ManhattanMetric {
    /// Calculate Manhattan
    ///
    /// ### Params
    ///
    /// * `idx` - The [NsgIndex] index.
    /// * `vec_p` - The query vector.
    /// * `_norm_p` - The norm of the query (unused)
    /// * `q` - Index of the DB vector
    ///
    /// ### Returns
    ///
    /// The Manhattan distance between idx and vec_p
    #[inline(always)]
    fn distance_from_vec(idx: &NsgIndex<T>, vec_p: &[T], _norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        T::manhattan_simd(vec_p, vec_q)
    }
}

///////////////////
// Distances i,j //
///////////////////

impl<T> NsgIndex<T>
where
    T: AnnSearchFloat,
{
    /// Distance between two indexed points under the index metric.
    ///
    /// Runtime metric branch. Retained for cold paths (nav-node beam, DFS-fix
    /// beam, external-vector callers). Hot build kernels go through
    /// [`MetricFn::distance_from_vec`] instead.
    ///
    /// ### Params
    ///
    /// * `i` - First point index
    /// * `j` - Second point index
    ///
    /// ### Returns
    ///
    /// Distance under the index metric.
    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
            Dist::Manhattan => self.manhattan_distance(i, j),
        }
    }

    /// Distance between an arbitrary external vector and an indexed point.
    ///
    /// ### Params
    ///
    /// * `query` - Arbitrary external vector (length must equal `dim`)
    /// * `idx` - Indexed point
    /// * `query_norm` - Precomputed L2 norm of `query`. Used only for
    ///   Cosine, ignored otherwise.
    ///
    /// ### Returns
    ///
    /// Distance under the index metric.
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance_to_query(idx, query),
            Dist::Cosine => self.cosine_distance_to_query(idx, query, query_norm),
            Dist::Manhattan => self.manhattan_distance_to_query(idx, query),
        }
    }

    /// Distance from an arbitrary external vector to an indexed point,
    /// using a static distance helper. Kept only for centroid resolution
    /// where the "query" is the centroid (no precomputed norm required for
    /// non-cosine metrics).
    #[inline(always)]
    fn distance_external(&self, external: &[T], idx: usize) -> T {
        let start = idx * self.dim;
        let vec_i = &self.vectors_flat[start..start + self.dim];

        match self.metric {
            Dist::SquaredEuclidean => euclidean_distance_static(vec_i, external),
            Dist::Manhattan => T::manhattan_simd(vec_i, external),
            Dist::Cosine => cosine_distance_static(vec_i, external),
        }
    }
}

///////////
// Build //
///////////

impl<T> NsgIndex<T>
where
    T: AnnSearchFloat,
    Self: NsgState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    /// Build an NSG index by first constructing an internal NN-Descent kNN
    /// graph, then refining it via MRNG pruning and DFS connectivity fix.
    ///
    /// The kNN graph is discarded once the NSG is built.
    ///
    /// ### Params
    ///
    /// * `data` - Input matrix (rows are samples, columns are features)
    /// * `metric` - Distance metric
    /// * `params` - Build parameters (see [`NsgBuildParams`])
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built `NsgIndex` on success.
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if verbose {
            println!("Building internal NN-Descent graph (k={})...", params.knn_k);
        }

        let nnd_start = Instant::now();
        let nnd = NNDescent::<T>::new(
            data,
            metric,
            Some(params.knn_k),
            None,
            None,
            None,
            T::from_f64(0.001).unwrap(),
            T::zero(),
            seed,
            verbose,
        )?;
        if verbose {
            println!(
                "NSG timing: NN-Descent build = {:.1} ms",
                nnd_start.elapsed().as_secs_f64() * 1000.0
            );
        }

        Self::build_from_nndescent(&nnd, params, seed, verbose)
    }

    /// Build an NSG index reusing an already-built NN-Descent index as the
    /// input kNN graph.
    ///
    /// ### Params
    ///
    /// * `data` - Original data matrix
    /// * `nnd` - Reference to a pre-built NN-Descent index
    /// * `params` - Build parameters
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built `NsgIndex` on success.
    pub fn build_from_nndescent(
        nnd: &NNDescent<T>,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        Self::build_from_knn(
            nnd.vectors_flat.to_owned(),
            nnd.n,
            nnd.dim,
            nnd.norms.to_owned(),
            nnd.metric(),
            nnd.graph(),
            nnd.k,
            params,
            seed,
            verbose,
        )
    }

    /// Build an NSG index from the slim GPU-built kNN graph.
    ///
    /// ### Params
    ///
    /// * `knn_gpu` - Reference to a pre-built
    ///   [`KnnGraphGpu`](crate::gpu::nndescent_gpu::KnnGraphGpu)
    /// * `params` - Build parameters
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built `NsgIndex` on success.
    #[cfg(feature = "gpu")]
    pub fn build_from_gpu_knn(
        knn_gpu: &crate::gpu::nndescent_gpu::KnnGraphGpu<T>,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        Self::build_from_knn(
            knn_gpu.vectors_flat.to_owned(),
            knn_gpu.n,
            knn_gpu.dim,
            knn_gpu.norms.to_owned(),
            knn_gpu.metric,
            &knn_gpu.knn_graph,
            knn_gpu.k,
            params,
            seed,
            verbose,
        )
    }

    /// Build an NSG index from a pre-existing flat kNN graph.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Row-major flattened vectors owned by the new index
    /// * `n` - Number of samples
    /// * `dim` - Dimensionality
    /// * `norms` - Pre-computed L2 norms for Cosine (empty otherwise)
    /// * `metric` - Distance metric
    /// * `knn_graph` - Flat kNN graph of length `n * knn_k`, sorted per node
    ///   by distance ascending, sentinel-padded
    /// * `knn_k` - Neighbours per node in `knn_graph`
    /// * `params` - Build parameters
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built `NsgIndex` on success.
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_knn(
        vectors_flat: Vec<T>,
        n: usize,
        dim: usize,
        norms: Vec<T>,
        metric: Dist,
        knn_graph: &[(usize, T)],
        knn_k: usize,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        assert_eq!(
            knn_graph.len(),
            n * knn_k,
            "knn_graph length must equal n * knn_k"
        );

        let mut index = Self {
            vectors_flat,
            n,
            dim,
            norms,
            metric,
            graph: Vec::new(),
            max_degree: params.r,
            r: params.r,
            l_build: params.l_build,
            c: params.c,
            navigating_node: 0,
            original_ids: (0..n).collect(),
        };

        // Step 1: pick the navigating node
        let t_nav = Instant::now();
        let np = index.pick_navigating_node(knn_graph, knn_k, params.l_build, seed);
        index.navigating_node = np as u32;
        if verbose {
            println!(
                "Navigating node: {} ({} candidates from centroid search)",
                np, params.l_build
            );
            println!(
                "NSG timing: step 1 (nav node) = {:.1} ms",
                t_nav.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Step 2a: parallel forward MRNG prune per node.
        let threads = rayon::current_num_threads();
        let build_graph: NsgConstructionGraph<T> = NsgConstructionGraph::new(n, params.r, threads);
        let mut forward: Vec<Vec<(u32, T)>> = vec![Vec::new(); n];
        let progress = Arc::new(AtomicUsize::new(0));
        let t_mrng = Instant::now();
        forward.par_iter_mut().enumerate().for_each(|(v, out)| {
            let neighbours = Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                index.collect_and_prune_for_node(
                    v,
                    np,
                    knn_graph,
                    knn_k,
                    params.l_build,
                    params.c,
                    params.r,
                    &mut state,
                )
            });

            // SAFETY: `v` is unique across the parallel iterator so no other
            // thread writes to nodes[v] in this phase.
            unsafe {
                build_graph.set_neighbours(v, &neighbours);
            }
            *out = neighbours;

            if verbose {
                let count = progress.fetch_add(1, Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        "  MRNG pruned {} / {} nodes",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
            }
        });

        if verbose {
            println!(
                "NSG timing: step 2a (MRNG prune) = {:.1} ms",
                t_mrng.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Step 2b: parallel InterInsert (reverse edges).
        let t_inter = Instant::now();
        let bg = &build_graph;
        forward.par_iter().enumerate().for_each(|(v, neighbours)| {
            let build_graph = bg;
            for &(u_id, dist_uv) in neighbours {
                let u = u_id as usize;
                if u == v {
                    continue;
                }
                let _guard = build_graph.locks.lock_guard(u);

                // SAFETY: guard grants exclusive access to nodes[u].
                let (deg, dup) = unsafe {
                    let slot = &*build_graph.nodes[u].get();
                    (slot.len(), slot.iter().any(|&(id, _)| id as usize == v))
                };
                if dup {
                    continue;
                }

                if deg < params.r {
                    unsafe {
                        (*build_graph.nodes[u].get()).push((v as u32, dist_uv));
                    }
                    continue;
                }

                // At capacity: replace u's farthest neighbour with v if v is
                // closer. MRNG's occlusion rule is too aggressive on tightly
                // clustered data.
                unsafe {
                    let slot = &mut *build_graph.nodes[u].get();
                    let mut worst_pos = 0usize;
                    let mut worst_dist = slot[0].1;
                    for i in 1..slot.len() {
                        if slot[i].1 > worst_dist {
                            worst_dist = slot[i].1;
                            worst_pos = i;
                        }
                    }
                    if dist_uv < worst_dist {
                        slot[worst_pos] = (v as u32, dist_uv);
                    }
                }
            }
        });
        drop(forward);
        if verbose {
            println!(
                "NSG timing: step 2b (InterInsert) = {:.1} ms",
                t_inter.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Step 2c: kNN top-up. MRNG occlusion collapses to ~6-8 edges per
        // node on clustered data, so R never binds and the sparse graph
        // dead-ends a subset of queries. Fill every node up to degree R with
        // its closest kNN entries (cached distances, no recompute). Must run
        // after InterInsert so reverse edges don't hit full nodes, and before
        // the DFS fix so the denser graph needs fewer patches.
        let t_topup = Instant::now();
        let bg = &build_graph;
        (0..n).into_par_iter().for_each(|v| {
            // SAFETY: `v` is unique across the parallel iterator and step 2b's
            // rayon scope has completed, so no other thread touches nodes[v].
            unsafe {
                let slot = &mut *bg.nodes[v].get();
                if slot.len() >= params.r {
                    return;
                }
                let base = v * knn_k;
                for &(nbr, dist) in &knn_graph[base..base + knn_k] {
                    if slot.len() >= params.r {
                        break;
                    }
                    if nbr == SENTINEL_PID || nbr >= n || nbr == v {
                        continue;
                    }
                    let nbr_u32 = nbr as u32;
                    if slot.iter().any(|&(id, _)| id == nbr_u32) {
                        continue;
                    }
                    slot.push((nbr_u32, dist));
                }
            }
        });
        if verbose {
            println!(
                "NSG timing: step 2c (kNN top-up) = {:.1} ms",
                t_topup.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Step 3: DFS connectivity
        let t_dfs = Instant::now();
        index.dfs_connectivity_fix(&build_graph, verbose);
        if verbose {
            println!(
                "NSG timing: step 3 (DFS fix) = {:.1} ms",
                t_dfs.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Step 4: flatten
        let t_flat = Instant::now();
        let (flat, stride) = build_graph.into_flat();
        index.graph = flat;
        index.max_degree = stride;
        if verbose {
            println!(
                "NSG timing: step 4 (flatten) = {:.1} ms",
                t_flat.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(index)
    }

    /////////////////////
    // Navigating node //
    /////////////////////

    /// Beam-search the input kNN graph toward the centroid.
    ///
    /// Uses `SearchState` for visited tracking and candidate ordering. Seeds
    /// the beam with `l_build` distinct random entry points (matching the
    /// reference NSG `Init_Center` behaviour).
    ///
    /// ### Params
    ///
    /// * `knn_graph` - Input kNN graph
    /// * `knn_k` - Neighbours per node in `knn_graph`
    /// * `l_build` - Beam width for the search
    /// * `seed` - Seed for the random entry points
    ///
    /// ### Returns
    ///
    /// Index of the data point nearest to the centroid.
    fn pick_navigating_node(
        &self,
        knn_graph: &[(usize, T)],
        knn_k: usize,
        l_build: usize,
        seed: usize,
    ) -> usize {
        let centroid = compute_centroid(&self.vectors_flat, self.n, self.dim);
        let mut rng = SmallRng::seed_from_u64(seed as u64 ^ 0xA5A5_5A5A);

        // Seed the beam with l_build distinct random entries.
        let seed_count = l_build.min(self.n);
        let mut seeds: Vec<usize> = Vec::with_capacity(seed_count);
        {
            let mut chosen = vec![false; self.n];
            while seeds.len() < seed_count {
                let cand = rng.random_range(0..self.n);
                if !chosen[cand] {
                    chosen[cand] = true;
                    seeds.push(cand);
                }
            }
        }

        Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let mut furthest_dist = OrderedFloat(T::infinity());
            for &s in &seeds {
                let d = OrderedFloat(self.distance_external(&centroid, s));
                state.mark_visited(s);
                state.candidates.push(Reverse((d, s)));
                state.working_sorted.insert((d, s), l_build);
            }
            if state.working_sorted.len() >= l_build {
                furthest_dist = state
                    .working_sorted
                    .top()
                    .map(|(d, _)| *d)
                    .unwrap_or(OrderedFloat(T::infinity()));
            }

            while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
                if current_dist > furthest_dist && state.working_sorted.len() >= l_build {
                    break;
                }

                let base = current_id * knn_k;
                for &(nbr, _) in &knn_graph[base..base + knn_k] {
                    if nbr == SENTINEL_PID || nbr >= self.n {
                        continue;
                    }
                    if state.is_visited(nbr) {
                        continue;
                    }
                    state.mark_visited(nbr);

                    let d = OrderedFloat(self.distance_external(&centroid, nbr));

                    if d < furthest_dist || state.working_sorted.len() < l_build {
                        state.candidates.push(Reverse((d, nbr)));
                        if state.working_sorted.insert((d, nbr), l_build)
                            && state.working_sorted.len() >= l_build
                        {
                            furthest_dist = state
                                .working_sorted
                                .top()
                                .map(|(d, _)| *d)
                                .unwrap_or(OrderedFloat(T::infinity()));
                        }
                    }
                }
            }

            state
                .working_sorted
                .data()
                .first()
                .map(|(_, idx)| *idx)
                .unwrap_or_else(|| seeds[0])
        })
    }

    //////////////////////
    // Candidate + MRNG //
    //////////////////////

    /// Collect candidates for node `v` and apply MRNG pruning.
    ///
    /// Dispatches on `self.metric` to a monomorphised
    /// [`Self::collect_and_prune_for_node_impl`] so the two hot inner
    /// distance loops (beam expansion, MRNG occlusion check) run with a
    /// hoisted base vector and a static metric.
    ///
    /// ### Params
    ///
    /// * `v` - Target node whose adjacency is being computed
    /// * `np` - Navigating node
    /// * `knn_graph` - Input kNN graph
    /// * `knn_k` - Neighbours per node in `knn_graph`
    /// * `l_build` - Beam pool width
    /// * `c` - Cap on the candidate-set size before pruning
    /// * `r` - Target out-degree of the resulting adjacency
    /// * `state` - Thread-local search state (reused for visited tracking)
    ///
    /// ### Returns
    ///
    /// Vector of at most `r` `(neighbour_id, dist_from_v)` pairs.
    #[allow(clippy::too_many_arguments)]
    fn collect_and_prune_for_node(
        &self,
        v: usize,
        np: usize,
        knn_graph: &[(usize, T)],
        knn_k: usize,
        l_build: usize,
        c: usize,
        r: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(u32, T)> {
        match self.metric {
            Dist::SquaredEuclidean => self.collect_and_prune_for_node_impl::<SqEuclidMetric>(
                v, np, knn_graph, knn_k, l_build, c, r, state,
            ),
            Dist::Cosine => self.collect_and_prune_for_node_impl::<CosineMetric>(
                v, np, knn_graph, knn_k, l_build, c, r, state,
            ),
            Dist::Manhattan => self.collect_and_prune_for_node_impl::<ManhattanMetric>(
                v, np, knn_graph, knn_k, l_build, c, r, state,
            ),
        }
    }

    /// Monomorphised inner kernel for candidate collection + MRNG prune.
    ///
    /// `M` selects the distance function at compile time. `vec_v` and
    /// `norm_v` are hoisted once at the top so the beam-expansion loop
    /// and the two candidate-injection loops never re-slice the base
    /// vector or re-fetch its norm.
    #[allow(clippy::too_many_arguments)]
    fn collect_and_prune_for_node_impl<M: MetricFn<T>>(
        &self,
        v: usize,
        np: usize,
        knn_graph: &[(usize, T)],
        knn_k: usize,
        l_build: usize,
        c: usize,
        r: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(u32, T)> {
        state.reset(self.n);

        state.scratch_working.clear();

        let dim = self.dim;
        let vec_v = &self.vectors_flat[v * dim..(v + 1) * dim];
        let has_norms = !self.norms.is_empty();
        let norm_v = if has_norms { self.norms[v] } else { T::zero() };

        let mut furthest_dist = OrderedFloat(T::infinity());
        let entry_dist = OrderedFloat(M::distance_from_vec(self, vec_v, norm_v, np));
        state.mark_visited(np);
        state.candidates.push(Reverse((entry_dist, np)));
        state.working_sorted.insert((entry_dist, np), l_build);
        if np != v {
            state.scratch_working.push((entry_dist, np));
        }

        // Seed with np's kNN neighbours as extra beam entries.
        let np_base = np * knn_k;
        for &(nbr, _) in &knn_graph[np_base..np_base + knn_k] {
            if nbr == SENTINEL_PID || nbr >= self.n || state.is_visited(nbr) {
                continue;
            }
            state.mark_visited(nbr);
            let d = OrderedFloat(M::distance_from_vec(self, vec_v, norm_v, nbr));
            state.candidates.push(Reverse((d, nbr)));
            state.working_sorted.insert((d, nbr), l_build);
            if nbr != v {
                state.scratch_working.push((d, nbr));
            }
        }
        if state.working_sorted.len() >= l_build {
            furthest_dist = state
                .working_sorted
                .top()
                .map(|(d, _)| *d)
                .unwrap_or(OrderedFloat(T::infinity()));
        }

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist && state.working_sorted.len() >= l_build {
                break;
            }

            let base = current_id * knn_k;
            for &(nbr, _) in &knn_graph[base..base + knn_k] {
                if nbr == SENTINEL_PID || nbr >= self.n {
                    continue;
                }
                if state.is_visited(nbr) {
                    continue;
                }
                state.mark_visited(nbr);

                let d = OrderedFloat(M::distance_from_vec(self, vec_v, norm_v, nbr));

                if nbr != v {
                    state.scratch_working.push((d, nbr));
                }

                if d < furthest_dist || state.working_sorted.len() < l_build {
                    state.candidates.push(Reverse((d, nbr)));
                    if state.working_sorted.insert((d, nbr), l_build)
                        && state.working_sorted.len() >= l_build
                    {
                        furthest_dist = state
                            .working_sorted
                            .top()
                            .map(|(d, _)| *d)
                            .unwrap_or(OrderedFloat(T::infinity()));
                    }
                }
            }
        }

        // Union in v's kNN neighbours to guarantee NNG ⊂ NSG.
        let vk_base = v * knn_k;
        for &(nbr, _) in &knn_graph[vk_base..vk_base + knn_k] {
            if nbr == SENTINEL_PID || nbr >= self.n || nbr == v {
                continue;
            }
            if !state.is_visited(nbr) {
                state.mark_visited(nbr);
                let d = OrderedFloat(M::distance_from_vec(self, vec_v, norm_v, nbr));
                state.scratch_working.push((d, nbr));
            }
        }

        // Sort candidates ascending by distance to v, cap at C.
        state
            .scratch_working
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        if state.scratch_working.len() > c {
            state.scratch_working.truncate(c);
        }

        self.mrng_prune::<M>(&state.scratch_working, r, v)
    }

    /// MRNG selection on a pre-sorted candidate pool.
    ///
    /// For each candidate `p` (closest to `base` first), accept if no
    /// already-accepted `s` satisfies `d(p, s) < d(p, base)`. Strict `<`
    /// matches the paper's Algorithm 2 and the ZJULearning reference.
    ///
    /// ### Params
    ///
    /// * `candidates` - Pre-sorted ascending by distance to `base`. Not
    ///   modified; `exclude` filters `base` itself.
    /// * `r` - Maximum size of the accepted set
    /// * `exclude` - Node id to skip (typically the base node itself)
    ///
    /// ### Returns
    ///
    /// Accepted neighbours as `(id, dist_from_base)`, order-preserving w.r.t.
    /// `candidates`.
    fn mrng_prune<M: MetricFn<T>>(
        &self,
        candidates: &[(OrderedFloat<T>, usize)],
        r: usize,
        exclude: usize,
    ) -> Vec<(u32, T)> {
        let mut accepted: Vec<(u32, T)> = Vec::with_capacity(r);
        let dim = self.dim;
        let has_norms = !self.norms.is_empty();

        for &(cand_dist, cand_id) in candidates.iter() {
            if cand_id == exclude {
                continue;
            }
            if accepted.len() >= r {
                break;
            }

            let vec_cand = &self.vectors_flat[cand_id * dim..(cand_id + 1) * dim];
            let norm_cand = if has_norms {
                self.norms[cand_id]
            } else {
                T::zero()
            };

            let mut ok = true;
            for &(sel_id, _) in accepted.iter() {
                let d_ps = M::distance_from_vec(self, vec_cand, norm_cand, sel_id as usize);
                if d_ps < cand_dist.0 {
                    ok = false;
                    break;
                }
            }

            if ok {
                accepted.push((cand_id as u32, cand_dist.0));
            }
        }

        accepted
    }

    //////////////////////
    // Connectivity fix //
    //////////////////////

    /// Ensure every node is reachable from `n_p` by forward walk.
    ///
    /// Follows the reference NSG `tree_grow` pattern: interleave DFS with
    /// per-unreachable-node patching. Persistent `reachable` flags carry
    /// across DFS restarts, so patching one unreachable `u` and restarting
    /// DFS from `u` propagates through `u`'s entire subtree in a single
    /// pass — no re-visiting nodes that a patched ancestor already covered.
    ///
    /// The patch itself is unconditional: `t -> u` is always added (via
    /// `push_neighbour_unchecked`), even if `t` already has degree `r`.
    /// Nodes whose degree exceeded `r` after patching are absorbed by
    /// `into_flat`, which widens the stride to accommodate them.
    ///
    /// ### Params
    ///
    /// * `build_graph` - Partial NSG under construction
    /// * `verbose` - Print progress
    fn dfs_connectivity_fix(&self, build_graph: &NsgConstructionGraph<T>, verbose: bool) {
        let mut reachable = vec![false; self.n];
        let mut stack: Vec<usize> = Vec::new();
        let np = self.navigating_node as usize;
        reachable[np] = true;
        stack.push(np);

        let mut patch_count = 0usize;
        let mut cursor = 0usize;

        loop {
            // Drain the DFS stack; each pop expands one node's outgoing
            // edges. Persistent reachable[] means restarts don't retraverse.
            while let Some(node) = stack.pop() {
                // SAFETY: sequential phase.
                let nbrs = unsafe { build_graph.get_neighbours_slice(node) };
                for &(nbr, _) in nbrs {
                    let n_idx = nbr as usize;
                    if !reachable[n_idx] {
                        reachable[n_idx] = true;
                        stack.push(n_idx);
                    }
                }
            }

            // Find the next unreachable node.
            while cursor < self.n && reachable[cursor] {
                cursor += 1;
            }
            if cursor == self.n {
                break;
            }
            let u = cursor;

            let t = self.beam_search_on_partial_graph(build_graph, np, u, self.l_build, &reachable);
            let dist_tu = self.distance(t, u);

            // SAFETY: sequential phase, no concurrent access.
            unsafe {
                build_graph.push_neighbour_unchecked(t, u as u32, dist_tu);
            }

            // Mark u reachable and restart DFS from u so its outgoing
            // subtree flood-fills in the next iteration.
            reachable[u] = true;
            stack.push(u);
            patch_count += 1;
        }

        if verbose && patch_count > 0 {
            println!(
                "DFS: patched {} unreachable nodes",
                patch_count.separate_with_underscores()
            );
        }
    }

    /// Beam-search the partial NSG for the closest reachable node to `u`.
    ///
    /// Runs only during the sequential connectivity-fix step, so borrowing
    /// the thread-local `SearchState` is safe.
    ///
    /// ### Params
    ///
    /// * `build_graph` - Partial NSG under construction
    /// * `np` - Navigating node (entry point)
    /// * `u` - Unreachable target node
    /// * `l_build` - Beam width
    /// * `reachable` - Boolean array marking reachable nodes
    ///
    /// ### Returns
    ///
    /// Node id `t` that is reachable and closest to `u`. Falls back to `np`
    /// if the beam somehow returned nothing reachable (should not happen).
    fn beam_search_on_partial_graph(
        &self,
        build_graph: &NsgConstructionGraph<T>,
        np: usize,
        u: usize,
        l_build: usize,
        reachable: &[bool],
    ) -> usize {
        Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let entry_dist = OrderedFloat(self.distance(np, u));
            state.mark_visited(np);
            state.candidates.push(Reverse((entry_dist, np)));
            state.working_sorted.insert((entry_dist, np), l_build);

            let mut furthest_dist = entry_dist;

            while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
                if current_dist > furthest_dist && state.working_sorted.len() >= l_build {
                    break;
                }

                // SAFETY: sequential connectivity-fix phase.
                let nbrs = unsafe { build_graph.get_neighbours_slice(current_id) };
                for &(nbr, _) in nbrs {
                    let n_idx = nbr as usize;
                    if n_idx >= self.n || state.is_visited(n_idx) {
                        continue;
                    }
                    state.mark_visited(n_idx);
                    let d = OrderedFloat(self.distance(n_idx, u));

                    if d < furthest_dist || state.working_sorted.len() < l_build {
                        state.candidates.push(Reverse((d, n_idx)));
                        if state.working_sorted.insert((d, n_idx), l_build)
                            && state.working_sorted.len() >= l_build
                        {
                            furthest_dist = state
                                .working_sorted
                                .top()
                                .map(|(d, _)| *d)
                                .unwrap_or(OrderedFloat(T::infinity()));
                        }
                    }
                }
            }

            state
                .working_sorted
                .data()
                .iter()
                .find(|(_, id)| reachable[*id])
                .map(|(_, id)| *id)
                .unwrap_or(np)
        })
    }
}

///////////
// Query //
///////////

impl<T> NsgIndex<T>
where
    T: AnnSearchFloat,
    Self: NsgState<T>,
{
    /// Read a node's neighbours from the flat graph.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    ///
    /// ### Returns
    ///
    /// Slice of length `max_degree`. Sentinel entries ([`u32::MAX`]) mark
    /// unused slots; callers should break on the first sentinel.
    #[inline(always)]
    fn get_neighbours_flat(&self, node_id: usize) -> &[u32] {
        let start = node_id * self.max_degree;
        &self.graph[start..start + self.max_degree]
    }

    /// Query the index for `k` nearest neighbours.
    ///
    /// Greedy beam search seeded from the navigating node plus a fixed set
    /// of entry points strided evenly across the dataset. Single-entry search from the medoid dead-ends on clustered
    /// data once cross-cluster paths get long; the strided seeds guarantee
    /// the beam starts near every region of the data. Seeds far from the
    /// query cost one distance evaluation each and are never expanded.
    /// `ef_search` sets the beam width; if `None`, defaults to `100`.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector (length must equal `dim`)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Optional beam width override
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance ascending.
    pub fn query(
        &self,
        query: &[T],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query.len())?;

        let ef = ef_search.unwrap_or(100).max(k);

        Self::with_search_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            let n_seeds = ((self.n as f64).sqrt() as usize)
                .clamp(NSG_ENTRY_SEEDS_MIN, NSG_ENTRY_SEEDS_MAX)
                .min(self.n);
            let step = (self.n / n_seeds).max(1);
            let seeds = std::iter::once(self.navigating_node as usize)
                .chain((0..n_seeds).map(|j| j * step));
            Ok(self.search_with_seeds(query, k, ef, seeds, &mut state))
        })
    }

    /// Beam-search the graph from a set of entry seeds.
    ///
    /// Shared kernel for [`Self::query`] (navigating node + strided global
    /// seeds) and the self-kNN path (the node itself as the sole seed).
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width (already resolved and clamped to `>= k`)
    /// * `seeds` - Entry node ids; duplicates are ignored
    /// * `state` - Thread-local search state
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance ascending.
    fn search_with_seeds(
        &self,
        query: &[T],
        k: usize,
        ef: usize,
        seeds: impl Iterator<Item = usize>,
        state: &mut SearchState<T>,
    ) -> (Vec<usize>, Vec<T>) {
        state.reset(self.n);

        let query_norm = if self.metric == Dist::Cosine {
            T::calculate_l2_norm(query)
        } else {
            T::one()
        };

        for entry_node in seeds {
            if state.is_visited(entry_node) {
                continue;
            }
            state.mark_visited(entry_node);
            let d = OrderedFloat(self.compute_query_distance(query, entry_node, query_norm));
            state.candidates.push(Reverse((d, entry_node)));
            state.working_sorted.insert((d, entry_node), ef);
        }

        let mut furthest_dist = state
            .working_sorted
            .top()
            .map(|(d, _)| *d)
            .unwrap_or(OrderedFloat(T::infinity()));

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist && state.working_sorted.len() >= ef {
                break;
            }

            let neighbours = self.get_neighbours_flat(current_id);
            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    break;
                }
                let n_idx = neighbour as usize;
                if state.is_visited(n_idx) {
                    continue;
                }
                state.mark_visited(n_idx);

                let d = OrderedFloat(self.compute_query_distance(query, n_idx, query_norm));

                if d < furthest_dist || state.working_sorted.len() < ef {
                    state.candidates.push(Reverse((d, n_idx)));
                    if state.working_sorted.insert((d, n_idx), ef)
                        && state.working_sorted.len() >= ef
                    {
                        furthest_dist = state
                            .working_sorted
                            .top()
                            .map(|(d, _)| *d)
                            .unwrap_or(OrderedFloat(T::infinity()));
                    }
                }
            }
        }

        let mut results = state.working_sorted.data().to_vec();
        results.truncate(k);

        let (indices, distances): (Vec<usize>, Vec<T>) = results
            .into_iter()
            .map(|(OrderedFloat(d), id)| (id, d))
            .unzip();

        (indices, distances)
    }

    /// Query using a matrix row reference.
    ///
    /// Optimised path for stride-1 rows; falls back to a copy otherwise.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Optional beam width override
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`.
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, ef_search)
    }

    /// Generate the full kNN graph by querying every internal vector.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per row
    /// * `ef_search` - Optional beam width override
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress every 100_000 samples
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`.
    pub fn generate_knn(
        &self,
        k: usize,
        ef_search: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

                if verbose {
                    let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                // Self-kNN: the node itself is the ideal entry seed. Its
                // adjacency already holds its nearest kNN entries, so the
                // beam converges locally without the global strided seeds.
                let ef = ef_search.unwrap_or(100).max(k);
                Self::with_search_state(|state_cell| {
                    let mut state = state_cell.borrow_mut();
                    self.search_with_seeds(vec, k, ef, std::iter::once(i), &mut state)
                })
            })
            .collect::<Vec<_>>();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }

    /// Memory usage of the index in bytes.
    ///
    /// ### Returns
    ///
    /// Size of vectors + norms + adjacency + owned metadata.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.graph.capacity() * std::mem::size_of::<u32>()
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for NsgIndex<T>
where
    T: AnnSearchFloat,
    Self: NsgState<T>,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, None)
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

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for NsgIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "nsg";
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn simple_matrix() -> Mat<f32> {
        let data = [
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    fn linear_matrix(n: usize, dim: usize) -> Mat<f32> {
        let mut data = vec![0.0_f32; n * dim];
        for i in 0..n {
            data[i * dim] = i as f32 * 0.1;
        }
        Mat::from_fn(n, dim, |i, j| data[i * dim + j])
    }

    fn build_default(mat: &Mat<f32>, metric: Dist) -> NsgIndex<f32> {
        let params = NsgBuildParams::new(16, 40, 100, 20);
        NsgIndex::<f32>::build(mat.as_ref(), metric, params, 42, false).unwrap()
    }

    #[test]
    fn build_euclidean_small() {
        let mat = simple_matrix();
        let _ = build_default(&mat, Dist::SquaredEuclidean);
    }

    #[test]
    fn build_cosine_small() {
        let mat = simple_matrix();
        let _ = build_default(&mat, Dist::Cosine);
    }

    #[test]
    fn navigating_node_in_range() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        assert!((idx.navigating_node as usize) < idx.n);
    }

    #[test]
    fn query_finds_self_euclidean() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = idx.query(&query, 1, None).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn query_finds_self_cosine() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::Cosine);
        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = idx.query(&query, 1, None).unwrap();
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn distances_sorted_ascending() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        let query = vec![0.5_f32, 0.5, 0.0];
        let (_, distances) = idx.query(&query, 4, None).unwrap();
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn query_returns_exactly_k() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        let query = vec![0.0_f32, 0.0, 0.0];
        for k in 1..=5 {
            let (indices, distances) = idx.query(&query, k, None).unwrap();
            assert_eq!(indices.len(), k);
            assert_eq!(distances.len(), k);
        }
    }

    #[test]
    fn no_self_loops() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        for node in 0..idx.n {
            let base = node * idx.max_degree;
            for &nbr in &idx.graph[base..base + idx.max_degree] {
                if nbr == u32::MAX {
                    break;
                }
                assert_ne!(nbr as usize, node);
            }
        }
    }

    #[test]
    fn neighbour_ids_in_range() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        for node in 0..idx.n {
            let base = node * idx.max_degree;
            for &nbr in &idx.graph[base..base + idx.max_degree] {
                if nbr == u32::MAX {
                    break;
                }
                assert!((nbr as usize) < idx.n);
            }
        }
    }

    #[test]
    fn dfs_reachability_holds() {
        let n = 200;
        let dim = 8;
        let mat = linear_matrix(n, dim);
        let params = NsgBuildParams::new(16, 60, 200, 32);
        let idx = NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 42, false)
            .unwrap();

        let mut reachable = vec![false; idx.n];
        let mut stack = Vec::new();
        stack.push(idx.navigating_node as usize);
        reachable[idx.navigating_node as usize] = true;

        while let Some(node) = stack.pop() {
            let base = node * idx.max_degree;
            for &nbr in &idx.graph[base..base + idx.max_degree] {
                if nbr == u32::MAX {
                    break;
                }
                let n_idx = nbr as usize;
                if !reachable[n_idx] {
                    reachable[n_idx] = true;
                    stack.push(n_idx);
                }
            }
        }

        for (i, &r) in reachable.iter().enumerate() {
            assert!(r, "Node {} is unreachable from the navigating node", i);
        }
    }

    #[test]
    fn recall_linear_data() {
        let n = 200;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = NsgBuildParams::new(16, 60, 200, 32);
        let idx = NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 42, false)
            .unwrap();

        let query = vec![0.0_f32; dim];
        let (indices, _) = idx.query(&query, 5, Some(80)).unwrap();
        assert_eq!(indices[0], 0);

        let expected: Vec<usize> = (0..5).collect();
        let found = indices.iter().filter(|&&i| expected.contains(&i)).count();
        assert!(found >= 4, "Expected 4/5 top-5, got {}", found);
    }

    #[test]
    fn recall_validation_high() {
        let n = 500;
        let dim = 8;
        let mut data = vec![0.0_f32; n * dim];
        let mut seed = 0xdead_beef_u64;
        for slot in data.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = ((seed >> 32) as u32 as f32) / (u32::MAX as f32);
        }
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let params = NsgBuildParams::new(24, 80, 300, 48);
        let idx = NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 42, false)
            .unwrap();

        let recall = idx.validate_index(5, 42, Some(50)).unwrap();
        assert!(recall > 0.9, "Recall too low: {}", recall);
    }

    #[test]
    fn reproducible_same_seed() {
        let n = 100;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = NsgBuildParams::new(16, 40, 150, 24);
        let a =
            NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 7, false).unwrap();
        let b =
            NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 7, false).unwrap();
        assert_eq!(a.navigating_node, b.navigating_node);
        assert_eq!(a.graph, b.graph);
    }
}
