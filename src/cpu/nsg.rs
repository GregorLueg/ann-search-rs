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
//!    prune to select at most `R` outgoing edges.
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
//! Search With The Navigating Spreading-out Graph.* PVLDB 12(5), 2019.
//! arXiv:1707.00143.

use faer::{MatRef, RowRef};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::{RefCell, UnsafeCell};
use std::cmp::{Ordering, Reverse};
use std::sync::{atomic::AtomicUsize, Arc};
use thousands::*;

use crate::cpu::nndescent::{NNDescent, NNDescentQuery};
use crate::prelude::*;
use crate::utils::dist::{cosine_distance_static, euclidean_distance_static};
use crate::utils::graph_utils::SearchState;
use crate::utils::nndescent_utils::{ApplySortedUpdates, SENTINEL_PID};
use crate::utils::*;

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
    /// Maximum out-degree of the NSG graph (`R` in the paper). Typical
    /// values: 32–70 for SIFT-scale, higher for harder datasets.
    pub r: usize,
    /// Beam width for the per-node search on the input kNN graph during
    /// build (`L` in the paper). Larger `l_build` yields more candidates
    /// per node.
    pub l_build: usize,
    /// Cap on the size of the candidate set `E` before MRNG pruning.
    /// Absent from the paper but standard in reference implementations
    /// to bound per-node prune cost. Roughly 500 works across scales.
    pub c: usize,
    /// Degree of the input kNN graph when NSG builds one internally.
    /// Larger `knn_k` gives better MRNG candidates but slower NN-Descent.
    /// Ignored by `build_from_nndescent`.
    pub knn_k: usize,
}

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
    /// * `r` - Maximum out-degree
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
/// Each node owns a `Vec<u32>` of length up to `R`. Step 2 of the build
/// writes each node's slot exactly once from a parallel per-node loop, so
/// no synchronisation is required there. Step 3 (DFS connectivity fix) is
/// sequential, so the `add_or_evict_neighbour` helper does not need locks
/// either.
///
/// ### Fields
struct NsgConstructionGraph {
    /// Per-node adjacency lists, each capped at `r` entries
    nodes: Vec<UnsafeCell<Vec<u32>>>,
    /// Hard degree cap
    r: usize,
}

unsafe impl Sync for NsgConstructionGraph {}

impl NsgConstructionGraph {
    /// Allocate an empty construction graph.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `r` - Maximum out-degree per node
    ///
    /// ### Returns
    ///
    /// Empty graph with `n` slots pre-allocated to capacity `r`.
    fn new(n: usize, r: usize) -> Self {
        let nodes = (0..n)
            .map(|_| UnsafeCell::new(Vec::with_capacity(r)))
            .collect();
        Self { nodes, r }
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
    unsafe fn get_neighbours_slice(&self, node_id: usize) -> &[u32] {
        &*self.nodes[node_id].get()
    }

    /// Overwrite a node's adjacency list.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `neighbours` - New adjacency list (must not exceed `R`)
    ///
    /// ### Safety
    ///
    /// Caller must hold exclusive access to `nodes[node_id]`.
    #[inline]
    unsafe fn set_neighbours(&self, node_id: usize, neighbours: &[u32]) {
        let slot = &mut *self.nodes[node_id].get();
        slot.clear();
        slot.extend_from_slice(neighbours);
    }

    /// Add a neighbour or, if the node is at capacity, evict the farthest.
    ///
    /// Distances are supplied so the eviction step needs no external lookup.
    ///
    /// ### Params
    ///
    /// * `node_id` - Source node
    /// * `new_neighbour` - Neighbour id to insert
    /// * `new_dist` - Distance from `node_id` to `new_neighbour`
    /// * `neighbour_dists` - Existing distances (in the same order as the
    ///   current adjacency list). Must have length equal to the current
    ///   degree.
    ///
    /// ### Safety
    ///
    /// Caller must hold exclusive access to `nodes[node_id]`. Only used
    /// during the sequential DFS-fix step.
    #[inline]
    unsafe fn add_or_evict_neighbour<T: Copy + PartialOrd>(
        &self,
        node_id: usize,
        new_neighbour: u32,
        new_dist: T,
        neighbour_dists: &[T],
    ) {
        let slot = &mut *self.nodes[node_id].get();
        if slot.len() < self.r {
            slot.push(new_neighbour);
            return;
        }

        let mut worst_pos = 0usize;
        let mut worst_dist = neighbour_dists[0];
        for (i, &d) in neighbour_dists.iter().enumerate().skip(1) {
            if d > worst_dist {
                worst_dist = d;
                worst_pos = i;
            }
        }

        if new_dist < worst_dist {
            slot[worst_pos] = new_neighbour;
        }
    }

    /// Consume the construction graph into a flat `Vec<u32>`.
    ///
    /// Each node contributes exactly `R` entries, tail-padded with
    /// [`u32::MAX`] sentinels. Reader-friendly for cache-linear beam search.
    ///
    /// ### Returns
    ///
    /// The flattened `n * R` adjacency array.
    fn into_flat(self) -> Vec<u32> {
        let r = self.r;
        let mut flat = Vec::with_capacity(self.nodes.len() * r);
        for cell in self.nodes {
            let neighbours = cell.into_inner();
            let deg = neighbours.len();
            flat.extend_from_slice(&neighbours);
            for _ in deg..r {
                flat.push(u32::MAX);
            }
        }
        flat
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
///
/// ### Fields
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
    /// Flat adjacency array of size `n * R`
    pub graph: Vec<u32>,
    /// Maximum out-degree
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
/// build. Parallelising would not pay for itself on typical sizes.
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

///////////////////
// Distances i,j //
///////////////////

impl<T> NsgIndex<T>
where
    T: AnnSearchFloat,
{
    /// Distance between two indexed points under the index metric.
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
            Dist::SquaredEuclidean | Dist::Manhattan => match self.metric {
                Dist::Manhattan => T::manhattan_simd(vec_i, external),
                _ => euclidean_distance_static(vec_i, external),
            },
            Dist::Cosine => cosine_distance_static(vec_i, external),
        }
    }
}

////////////
// Build  //
////////////

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
        data: MatRef<T>,
        metric: Dist,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if verbose {
            println!(
                "Building internal NN-Descent graph (k={})...",
                params.knn_k
            );
        }

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

        Self::build_from_nndescent(data, &nnd, params, seed, verbose)
    }

    /// Build an NSG index reusing an already-built NN-Descent index as the
    /// input kNN graph.
    ///
    /// The `data` must match the matrix that was originally passed to
    /// [`NNDescent::new`]; NSG re-flattens it for cache locality and does
    /// not touch the vectors owned by `nnd`.
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
        data: MatRef<T>,
        nnd: &NNDescent<T>,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let norms = if metric_needs_norms(nnd.metric()) {
            precompute_norms(&vectors_flat, n, dim)
        } else {
            Vec::new()
        };

        Self::build_from_knn(
            vectors_flat,
            n,
            dim,
            norms,
            nnd.metric(),
            nnd.graph(),
            nnd.k,
            params,
            seed,
            verbose,
        )
    }

    /// Build an NSG index reusing a GPU-built NN-Descent index.
    ///
    /// Downloads the flat kNN graph produced by NN-Descent (before the
    /// CAGRA rank-prune step) and hands it to the standard CPU NSG build.
    /// The CAGRA-optimised graph on the GPU is deliberately not used here
    /// because MRNG pruning wants a true kNN as input, not a navigational
    /// graph.
    ///
    /// ### Params
    ///
    /// * `data` - Original data matrix
    /// * `nnd_gpu` - Reference to a pre-built GPU NN-Descent index
    /// * `params` - Build parameters
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built `NsgIndex` on success.
    #[cfg(feature = "gpu")]
    pub fn build_from_gpu_nndescent<R>(
        data: MatRef<T>,
        nnd_gpu: &crate::gpu::nndescent_gpu::NNDescentGpu<T, R>,
        params: NsgBuildParams,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors>
    where
        R: cubecl::Runtime,
        T: crate::gpu::traits_gpu::AnnSearchGpuFloat,
    {
        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let norms = if metric_needs_norms(nnd_gpu.metric()) {
            precompute_norms(&vectors_flat, n, dim)
        } else {
            Vec::new()
        };

        Self::build_from_knn(
            vectors_flat,
            n,
            dim,
            norms,
            nnd_gpu.metric(),
            nnd_gpu.knn_graph(),
            nnd_gpu.k,
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
            r: params.r,
            l_build: params.l_build,
            c: params.c,
            navigating_node: 0,
            original_ids: (0..n).collect(),
        };

        // Step 1: pick the navigating node
        let np = index.pick_navigating_node(knn_graph, knn_k, params.l_build, seed);
        index.navigating_node = np as u32;
        if verbose {
            println!(
                "Navigating node: {} ({} candidates from centroid search)",
                np, params.l_build
            );
        }

        // Step 2: parallel MRNG prune per node
        let build_graph = NsgConstructionGraph::new(n, params.r);
        let progress = Arc::new(AtomicUsize::new(0));
        (0..n).into_par_iter().for_each(|v| {
            Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                let neighbours = index.collect_and_prune_for_node(
                    v,
                    np,
                    knn_graph,
                    knn_k,
                    params.l_build,
                    params.c,
                    params.r,
                    &mut state,
                );
                // SAFETY: `v` is unique across the parallel iterator so no
                // other thread writes to `nodes[v]`.
                unsafe {
                    build_graph.set_neighbours(v, &neighbours);
                }
            });

            if verbose {
                let count = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count.is_multiple_of(100_000) {
                    println!(
                        "  MRNG pruned {} / {} nodes",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
            }
        });

        // Step 3: DFS connectivity fix (sequential)
        index.dfs_connectivity_fix(&build_graph, verbose);

        // Step 4: flatten
        index.graph = build_graph.into_flat();

        Ok(index)
    }

    ///////////////////////////
    // Navigating node       //
    ///////////////////////////

    /// Beam-search the input kNN graph toward the centroid.
    ///
    /// Uses `SearchState` for visited tracking and candidate ordering. The
    /// entry point is a random node; a single search pass with pool `L` is
    /// enough because the centroid is a synthetic vector and we only need
    /// the closest indexed point.
    ///
    /// ### Params
    ///
    /// * `knn_graph` - Input kNN graph
    /// * `knn_k` - Neighbours per node in `knn_graph`
    /// * `l_build` - Beam width for the search
    /// * `seed` - Seed for the random entry point
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
        let entry = rng.random_range(0..self.n);

        Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let entry_dist = OrderedFloat(self.distance_external(&centroid, entry));
            state.mark_visited(entry);
            state.candidates.push(Reverse((entry_dist, entry)));
            state.working_sorted.insert((entry_dist, entry), l_build);

            let mut furthest_dist = entry_dist;

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
                .unwrap_or(entry)
        })
    }

    ///////////////////////////
    // Candidate + MRNG      //
    ///////////////////////////

    /// Collect candidates for node `v` and apply MRNG pruning.
    ///
    /// Beam-searches the kNN graph from `np` toward `v`, records every
    /// visited node, unions in `v`'s kNN neighbours, sorts by distance
    /// ascending, then greedily accepts candidates that do not violate the
    /// MRNG rule. Returns at most `r` outgoing neighbours.
    ///
    /// ### Params
    ///
    /// * `v` - Target node whose adjacency is being computed
    /// * `np` - Navigating node
    /// * `knn_graph` - Input kNN graph
    /// * `knn_k` - Neighbours per node in `knn_graph`
    /// * `l_build` - Beam pool width
    /// * `c` - Cap on the candidate-set size before pruning
    /// * `r` - Maximum out-degree of the resulting adjacency
    /// * `state` - Thread-local search state (reused for visited tracking)
    ///
    /// ### Returns
    ///
    /// Vector of at most `r` neighbour ids for `v`, sorted by distance.
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
    ) -> Vec<u32> {
        state.reset(self.n);

        // Beam-search from np toward v, collecting the visited set.
        // scratch_working accumulates (dist_to_v, node_id) pairs.
        state.scratch_working.clear();

        let entry_dist = OrderedFloat(self.distance(np, v));
        state.mark_visited(np);
        state.candidates.push(Reverse((entry_dist, np)));
        state.working_sorted.insert((entry_dist, np), l_build);
        if np != v {
            state.scratch_working.push((entry_dist, np));
        }

        let mut furthest_dist = entry_dist;

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

                let d = OrderedFloat(self.distance(nbr, v));

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
                let d = OrderedFloat(self.distance(nbr, v));
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

        // MRNG prune: for each candidate p (closest first), accept iff no
        // already-selected r satisfies delta(p, r) < delta(p, v).
        // scratch_discarded stores (dist_to_v, id) for accepted neighbours
        // — needed by the acceptance predicate.
        state.scratch_discarded.clear();
        let mut accepted: Vec<u32> = Vec::with_capacity(r);

        for &(cand_dist, cand_id) in state.scratch_working.iter() {
            if cand_id == v {
                continue;
            }
            if accepted.len() >= r {
                break;
            }

            let mut ok = true;
            for &(_, sel_id) in state.scratch_discarded.iter() {
                let d_ps = self.distance(cand_id, sel_id);
                // Strict inequality tie-broken by lower id via the outer
                // sort order (candidates in scratch_working are ascending
                // by distance).
                if d_ps < cand_dist.0 {
                    ok = false;
                    break;
                }
            }

            if ok {
                accepted.push(cand_id as u32);
                state.scratch_discarded.push((cand_dist, cand_id));
            }
        }

        accepted
    }

    ///////////////////////////
    // Connectivity fix      //
    ///////////////////////////

    /// DFS from `n_p` on the partial NSG; patch every unreachable node.
    ///
    /// Patches consist of a beam-search on the partial NSG from `n_p`
    /// toward each unreachable node `u`, then adding the edge `t -> u`
    /// where `t` is the nearest reachable point in the pool.
    ///
    /// Each iteration shrinks the unreachable set by at least one node,
    /// so termination is guaranteed.
    ///
    /// ### Params
    ///
    /// * `build_graph` - Partial NSG under construction
    /// * `verbose` - Print progress
    fn dfs_connectivity_fix(&self, build_graph: &NsgConstructionGraph, verbose: bool) {
        let mut reachable = vec![false; self.n];
        let mut stack: Vec<usize> = Vec::new();
        let np = self.navigating_node as usize;
        reachable[np] = true;
        stack.push(np);

        while let Some(node) = stack.pop() {
            // SAFETY: no concurrent writes during the sequential DFS phase.
            let nbrs = unsafe { build_graph.get_neighbours_slice(node) }.to_vec();
            for &nbr in &nbrs {
                let n_idx = nbr as usize;
                if !reachable[n_idx] {
                    reachable[n_idx] = true;
                    stack.push(n_idx);
                }
            }
        }

        let unreachable: Vec<usize> = (0..self.n).filter(|&i| !reachable[i]).collect();
        if verbose && !unreachable.is_empty() {
            println!(
                "DFS: patching {} unreachable nodes",
                unreachable.len().separate_with_underscores()
            );
        }

        for u in unreachable {
            let t = self.beam_search_on_partial_graph(build_graph, np, u, self.l_build, &reachable);
            let dist_tu = self.distance(t, u);

            // SAFETY: sequential phase, no concurrent access.
            let neighbour_dists: Vec<T> = unsafe {
                build_graph
                    .get_neighbours_slice(t)
                    .iter()
                    .map(|&nbr| self.distance(t, nbr as usize))
                    .collect()
            };
            unsafe {
                build_graph.add_or_evict_neighbour(t, u as u32, dist_tu, &neighbour_dists);
            }
            // Adding an edge from a reachable node guarantees u is now
            // reachable in the next connected component scan; we mark it
            // eagerly to avoid recomputing DFS.
            reachable[u] = true;
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
        build_graph: &NsgConstructionGraph,
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
                for &nbr in nbrs {
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
    /// Slice of length `R`. Sentinel entries mark unused slots.
    #[inline(always)]
    fn get_neighbours_flat(&self, node_id: usize) -> &[u32] {
        let start = node_id * self.r;
        &self.graph[start..start + self.r]
    }

    /// Query the index for `k` nearest neighbours.
    ///
    /// Greedy beam search from the navigating node. `ef_search` sets the
    /// beam width; if `None`, defaults to `100`.
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
            state.reset(self.n);

            let query_norm = if self.metric == Dist::Cosine {
                T::calculate_l2_norm(query)
            } else {
                T::one()
            };

            let entry_node = self.navigating_node as usize;
            let entry_dist =
                OrderedFloat(self.compute_query_distance(query, entry_node, query_norm));

            state.mark_visited(entry_node);
            state.candidates.push(Reverse((entry_dist, entry_node)));
            state.working_sorted.insert((entry_dist, entry_node), ef);

            let mut furthest_dist = entry_dist;

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

            Ok((indices, distances))
        })
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

                self.query(vec, k, ef_search)
            })
            .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

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

///////////////////
// Small helpers //
///////////////////

/// Whether a metric requires pre-computed L2 norms in the index.
fn metric_needs_norms(metric: Dist) -> bool {
    matches!(metric, Dist::Cosine)
}

/// Row-wise L2 norms for a flat vector buffer.
fn precompute_norms<T: AnnSearchFloat>(vectors_flat: &[T], n: usize, dim: usize) -> Vec<T> {
    (0..n)
        .map(|i| {
            let start = i * dim;
            let end = start + dim;
            T::calculate_l2_norm(&vectors_flat[start..end])
        })
        .collect()
}

///////////
// Tests //
///////////

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
            let base = node * idx.r;
            for &nbr in &idx.graph[base..base + idx.r] {
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
            let base = node * idx.r;
            for &nbr in &idx.graph[base..base + idx.r] {
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
            let base = node * idx.r;
            for &nbr in &idx.graph[base..base + idx.r] {
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
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        let a = NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 7, false)
            .unwrap();
        let b = NsgIndex::<f32>::build(mat.as_ref(), Dist::SquaredEuclidean, params, 7, false)
            .unwrap();
        assert_eq!(a.navigating_node, b.navigating_node);
        assert_eq!(a.graph, b.graph);
    }
}
