//! Relative NN-Descent (RNN-Descent) index.
//!
//! RNN-Descent (Ono & Matsui, ACM MM 2023) folds the RNG-style pruning of NSG
//! into the NN-Descent update loop so a single pass produces a search-ready
//! graph. Compared to NN-Descent + NSG, no separate kNN graph is materialised.
//!
//! Ono, N. and Matsui, Y. *Relative NN-Descent: A Fast Index Construction for
//! Graph-Based Approximate Nearest Neighbor Search.* ACM MM 2023.
//! arXiv:2310.20419.

use faer::RowRef;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use rdst::RadixSort;
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::{
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
    Arc,
};
use std::time::Instant;
use thousands::*;

use crate::cpu::kd_forest::KdTreeIndex;
use crate::prelude::*;
use crate::utils::graph_utils::SearchState;
use crate::utils::nndescent_utils::{
    find_target_boundaries, ApplySortedUpdates, Neighbour, UnsafeGraphPtr, Update, SENTINEL_PID,
};
use crate::utils::*;

///////////////////
// Thread locals //
///////////////////

thread_local! {
    // UpdateNeighbors per-thread scratch
    static RNN_UPDATE_SCRATCH_F32: RefCell<UpdateScratch<f32>> =
        const { RefCell::new(UpdateScratch::new()) };
    static RNN_UPDATE_SCRATCH_F64: RefCell<UpdateScratch<f64>> =
        const { RefCell::new(UpdateScratch::new()) };

    // ApplySortedUpdates per-thread state (mirrors NN-Descent's pattern)
    static RNN_MERGE_HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static RNN_MERGE_HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static RNN_PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static RNN_SORT_BUF_F32: RefCell<Vec<(f32, usize, bool)>> =
        const { RefCell::new(Vec::new()) };
    static RNN_SORT_BUF_F64: RefCell<Vec<(f64, usize, bool)>> =
        const { RefCell::new(Vec::new()) };

    // Query beam
    static RNN_SEARCH_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1024));
    static RNN_SEARCH_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1024));
}

/// Per-thread scratch for the parallel UpdateNeighbors pass.
///
/// `accepted` holds the greedy-accepted adjacency being built for the
/// current source node; `emitted` collects reverse-insert updates that
/// target other nodes.
pub struct UpdateScratch<T> {
    /// `(dist, pid, is_new)` in ascending distance order
    pub accepted: Vec<(T, usize, bool)>,
    /// Survivors of the RNG prune, ascending. Reused across nodes so the
    /// per-node pass does not allocate.
    pub new_accepted: Vec<(T, usize, bool)>,
    /// Updates to be batched and applied via [`ApplySortedUpdates`]
    pub emitted: Vec<Update<T>>,
}

impl<T> UpdateScratch<T> {
    /// Create an empty scratch.
    ///
    /// ### Returns
    ///
    /// Empty [`UpdateScratch`].
    pub const fn new() -> Self {
        Self {
            accepted: Vec::new(),
            new_accepted: Vec::new(),
            emitted: Vec::new(),
        }
    }
}

impl<T> Default for UpdateScratch<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local state accessors for build-time and query-time paths.
pub trait RnnDescentState<T> {
    /// Access the thread-local UpdateNeighbors scratch.
    ///
    /// ### Params
    ///
    /// * `f` - Closure operating on the borrowed cell
    fn with_update_scratch<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<UpdateScratch<T>>) -> R;

    /// Access the thread-local query beam state.
    ///
    /// ### Params
    ///
    /// * `f` - Closure operating on the borrowed cell
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;
}

impl RnnDescentState<f32> for RnnDescentIndex<f32> {
    fn with_update_scratch<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<UpdateScratch<f32>>) -> R,
    {
        RNN_UPDATE_SCRATCH_F32.with(f)
    }
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        RNN_SEARCH_STATE_F32.with(f)
    }
}

impl RnnDescentState<f64> for RnnDescentIndex<f64> {
    fn with_update_scratch<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<UpdateScratch<f64>>) -> R,
    {
        RNN_UPDATE_SCRATCH_F64.with(f)
    }
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        RNN_SEARCH_STATE_F64.with(f)
    }
}

////////////
// Params //
////////////

/// Build-time parameters for [`RnnDescentIndex`].
///
/// Paper defaults are used by [`RnnDescentBuildParams::default`].
///
/// ### Fields
#[derive(Clone, Copy, Debug)]
pub struct RnnDescentBuildParams {
    /// Initial out-degree in the random seed graph
    pub s: usize,
    /// Maximum per-node adjacency after AddReverseEdges (`R` in the paper)
    pub r: usize,
    /// Outer rounds
    pub t1: usize,
    /// UpdateNeighbors passes per outer round
    pub t2: usize,
}

impl Default for RnnDescentBuildParams {
    fn default() -> Self {
        Self {
            s: 20,
            r: 96,
            t1: 4,
            t2: 15,
        }
    }
}

impl RnnDescentBuildParams {
    /// Construct with explicit values.
    ///
    /// ### Params
    ///
    /// * `s` - Initial out-degree
    /// * `r` - Maximum per-node adjacency
    /// * `t1` - Outer rounds
    /// * `t2` - UpdateNeighbors passes per outer round
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(s: usize, r: usize, t1: usize, t2: usize) -> Self {
        Self { s, r, t1, t2 }
    }
}

/////////////////////
// RnnDescentIndex //
/////////////////////

/// Relative NN-Descent index.
///
/// The final graph is stored as a flat `Vec<u32>` of size `n * R`, sorted by
/// distance ascending within each node's slot. Unused trailing slots hold
/// `SENTINEL_PID`. Build distances are dropped at compaction: the walk needs
/// `d(query, neighbour)` and never the stored `d(node, neighbour)`.
///
/// A small [`KdTreeIndex`] forest is built alongside the graph and used at
/// query time to pick a batch of near-query entry points. This mirrors how
/// NN-Descent seeds its beam search, but with fewer trees since the forest
/// is only queried once per query (not per node during build).
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct RnnDescentIndex<T> {
    /// Row-major flattened vectors
    pub vectors_flat: Vec<T>,
    /// Dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Maximum per-node adjacency
    pub r: usize,
    /// Distance metric
    pub metric: Dist,
    /// Pre-computed L2 norms (Cosine only; empty otherwise)
    pub norms: Vec<T>,
    /// Flat final graph, `n * R` point ids, sentinel-padded.
    ///
    /// Ids only: the walk needs `d(query, neighbour)`, never the stored
    /// `d(node, neighbour)`, so keeping build distances here would double the
    /// bytes touched per hop for nothing. Matches the reference CSR.
    pub graph: Vec<u32>,
    /// Kd-tree forest used to pick query entry points
    pub forest: KdTreeIndex<T>,
    /// Original ids
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for RnnDescentIndex<T>
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

impl<T> DimensionValidation for RnnDescentIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

////////////////////////
// ApplySortedUpdates //
////////////////////////

/// Macro producing [`ApplySortedUpdates`] impls per concrete float type.
///
/// The logic mirrors NN-Descent's implementation: for each target's contiguous
/// update segment, load the target's existing neighbours into a max-heap of
/// size `R`, merge in candidate updates, drop the farthest when the heap
/// overflows, then write back sorted ascending. `is_new` is preserved on
/// existing entries and set to `true` on newly-inserted ones.
macro_rules! impl_apply_sorted_updates {
    ($float:ty, $heap_tls:ident, $sort_buf_tls:ident) => {
        impl ApplySortedUpdates<$float> for RnnDescentIndex<$float> {
            fn apply_sorted_updates(
                &self,
                updates: &[Update<$float>],
                graph: &mut [Neighbour<$float>],
                k: usize,
                updates_count: &AtomicUsize,
            ) {
                if updates.is_empty() {
                    return;
                }

                let boundaries = find_target_boundaries(updates);
                let segments: Vec<(usize, &[Update<$float>])> = boundaries
                    .windows(2)
                    .filter_map(|w| {
                        let start = w[0];
                        let end = w[1];
                        if start < end {
                            Some((updates[start].target as usize, &updates[start..end]))
                        } else {
                            None
                        }
                    })
                    .collect();

                let graph_ptr = UnsafeGraphPtr(graph.as_mut_ptr());

                segments.par_iter().for_each(|&(target, segment)| {
                    #[allow(clippy::redundant_locals)]
                    let graph_ptr = graph_ptr;
                    $heap_tls.with(|heap_cell| {
                        RNN_PID_SET.with(|set_cell| {
                            $sort_buf_tls.with(|sort_cell| {
                                let mut heap = heap_cell.borrow_mut();
                                let mut pid_set = set_cell.borrow_mut();
                                let mut sort_buf = sort_cell.borrow_mut();

                                heap.clear();
                                if pid_set.len() < self.n {
                                    pid_set.resize(self.n, false);
                                }

                                let start_idx = target * k;
                                // SAFETY: each target processed by a unique thread
                                let target_slice = unsafe {
                                    std::slice::from_raw_parts_mut(graph_ptr.0.add(start_idx), k)
                                };

                                let mut edge_updates = 0usize;

                                for n in target_slice.iter() {
                                    if n.is_sentinel() {
                                        continue;
                                    }
                                    let pid = n.pid();
                                    heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                                    pid_set[pid] = true;
                                }

                                for update in segment {
                                    let src = update.source as usize;
                                    if src == target {
                                        continue;
                                    }
                                    if pid_set[src] {
                                        continue;
                                    }

                                    if heap.len() < k {
                                        heap.push((OrderedFloat(update.dist), src, true));
                                        pid_set[src] = true;
                                        edge_updates += 1;
                                    } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                        if update.dist < worst {
                                            if let Some((_, old_pid, _)) = heap.pop() {
                                                pid_set[old_pid] = false;
                                            }
                                            heap.push((OrderedFloat(update.dist), src, true));
                                            pid_set[src] = true;
                                            edge_updates += 1;
                                        }
                                    }
                                }

                                if edge_updates > 0 {
                                    updates_count.fetch_add(edge_updates, AtomicOrdering::Relaxed);

                                    sort_buf.clear();
                                    sort_buf.extend(heap.drain().map(
                                        |(OrderedFloat(d), p, is_new)| {
                                            pid_set[p] = false;
                                            (d, p, is_new)
                                        },
                                    ));
                                    sort_buf
                                        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                                    sort_buf.truncate(k);

                                    for (i, &(d, p, is_new)) in sort_buf.iter().enumerate() {
                                        target_slice[i] = Neighbour::new(p, d, is_new);
                                    }
                                    for i in sort_buf.len()..k {
                                        target_slice[i] =
                                            Neighbour::new(SENTINEL_PID, <$float>::MAX, false);
                                    }
                                } else {
                                    for (_, pid, _) in heap.iter() {
                                        pid_set[*pid] = false;
                                    }
                                }
                            })
                        })
                    })
                });
            }
        }
    };
}

impl_apply_sorted_updates!(f32, RNN_MERGE_HEAP_F32, RNN_SORT_BUF_F32);
impl_apply_sorted_updates!(f64, RNN_MERGE_HEAP_F64, RNN_SORT_BUF_F64);

/////////////
// Helpers //
/////////////

/// Whether a metric requires pre-computed L2 norms.
fn metric_needs_norms(metric: Dist) -> bool {
    matches!(metric, Dist::Cosine)
}

/// Row-wise L2 norms.
fn precompute_norms<T: AnnSearchFloat>(vectors_flat: &[T], n: usize, dim: usize) -> Vec<T> {
    (0..n)
        .map(|i| {
            let start = i * dim;
            let end = start + dim;
            T::calculate_l2_norm(&vectors_flat[start..end])
        })
        .collect()
}

//////////////
// Distance //
//////////////

impl<T> RnnDescentIndex<T>
where
    T: AnnSearchFloat,
{
    /// Distance between two indexed points under the index metric.
    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
            Dist::Manhattan => self.manhattan_distance(i, j),
        }
    }

    /// Distance from external query to an indexed point.
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance_to_query(idx, query),
            Dist::Cosine => self.cosine_distance_to_query(idx, query, query_norm),
            Dist::Manhattan => self.manhattan_distance_to_query(idx, query),
        }
    }
}

///////////
// Build //
///////////

impl<T> RnnDescentIndex<T>
where
    T: AnnSearchFloat,
    Self: RnnDescentState<T> + ApplySortedUpdates<T>,
{
    /// Build a new Relative NN-Descent index.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `metric` - Distance metric
    /// * `params` - Build parameters
    /// * `n_trees` - Kd-forest size for query entry points. `None` picks a
    ///   dataset-scaled default `min(5 + n^0.25 / 2, 16)` — half of
    ///   NN-Descent's rule since the forest is only queried at search time
    ///   (not per build node), so a lower budget is sufficient.
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// The built index on success.
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        params: RnnDescentBuildParams,
        n_trees: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let (vectors_flat, n, dim) = data.into_row_major();
        let norms = if metric_needs_norms(metric) {
            precompute_norms(&vectors_flat, n, dim)
        } else {
            Vec::new()
        };

        let n_trees =
            n_trees.unwrap_or_else(|| ((n as f64).powf(0.25) / 2.0 + 5.0).min(16.0) as usize);

        // Build the entry-point forest once, before the graph. Query time
        // multi-seeds the beam from `forest.query()` results.
        let start = Instant::now();
        let forest = KdTreeIndex::new((&vectors_flat[..], n, dim), n_trees, metric, seed);
        if verbose {
            println!(
                "Built KdForest ({} trees): {:.2?}",
                n_trees,
                start.elapsed()
            );
        }

        let mut index = Self {
            vectors_flat,
            dim,
            n,
            r: params.r,
            metric,
            norms,
            graph: Vec::new(),
            forest,
            original_ids: (0..n).collect(),
        };

        // Working flat graph in Neighbour<T> form during build.
        let sentinel = Neighbour::new(SENTINEL_PID, T::max_value(), false);
        let mut build_graph = vec![sentinel; n * params.r];

        index.initialise_random_graph(&mut build_graph, params.s, params.r, seed, verbose);

        for t1 in 0..params.t1 {
            for t2 in 0..params.t2 {
                let changes = index.update_neighbours_pass(&mut build_graph, params.r);
                if verbose {
                    println!(
                        "  t1={} t2={}: emitted {} reinsert updates",
                        t1 + 1,
                        t2 + 1,
                        changes.separate_with_underscores()
                    );
                }
            }
            if t1 + 1 != params.t1 {
                let changes = index.add_reverse_edges_pass(&mut build_graph, params.r);
                if verbose {
                    println!(
                        "  t1={}: AddReverseEdges applied {} edge updates",
                        t1 + 1,
                        changes.separate_with_underscores()
                    );
                }
            }
        }

        // Flatten to compact ids-only storage.
        // `collect` reuses the Neighbour<T> allocation in place (8 bytes/elem
        // for f32 against 4 for u32), so without the shrink the buffer keeps
        // twice the capacity it needs and the saving is never released.
        index.graph = build_graph.into_iter().map(|n| n.pid() as u32).collect();
        index.graph.shrink_to_fit();

        Ok(index)
    }

    /// Random-seed graph.
    ///
    /// Each node picks `s` distinct random out-neighbours (never itself),
    /// computes distances, sorts ascending, and stores flagged new. Trailing
    /// slots up to `r` are left as sentinels.
    ///
    /// ### Params
    ///
    /// * `graph` - Flat build-time graph (`n * r`)
    /// * `s` - Initial out-degree
    /// * `r` - Slot capacity per node
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    fn initialise_random_graph(
        &self,
        graph: &mut [Neighbour<T>],
        s: usize,
        r: usize,
        seed: usize,
        verbose: bool,
    ) {
        if verbose {
            println!(
                "Random seed graph: {} nodes, S={}, R={}",
                self.n.separate_with_underscores(),
                s,
                r
            );
        }

        let n = self.n;
        let actual_s = s.min(n.saturating_sub(1));
        graph.par_chunks_mut(r).enumerate().for_each(|(u, slot)| {
            let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(u as u64));
            let mut chosen: Vec<usize> = Vec::with_capacity(actual_s);
            while chosen.len() < actual_s {
                let cand = rng.random_range(0..n);
                if cand == u || chosen.contains(&cand) {
                    continue;
                }
                chosen.push(cand);
            }

            let mut with_dist: Vec<(T, usize)> = chosen
                .into_iter()
                .map(|v| (self.distance(u, v), v))
                .collect();
            with_dist.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            for (i, (d, v)) in with_dist.into_iter().enumerate() {
                slot[i] = Neighbour::new(v, d, true);
            }
        });
    }

    /// One UpdateNeighbors pass.
    ///
    /// Parallel over source nodes; each thread accumulates reinsert updates
    /// into a per-thread scratch. After the pass, updates are radix-sorted by
    /// target and applied via [`ApplySortedUpdates`].
    ///
    /// ### Params
    ///
    /// * `graph` - Build-time graph (mutated in place)
    /// * `r` - Slot capacity per node
    ///
    /// ### Returns
    ///
    /// Number of edge changes applied by [`ApplySortedUpdates`] this pass.
    fn update_neighbours_pass(&self, graph: &mut [Neighbour<T>], r: usize) -> usize {
        let counter = AtomicUsize::new(0);
        let graph_ptr = UnsafeGraphPtr(graph.as_mut_ptr());
        let n = self.n;

        let per_thread_updates: Vec<Vec<Update<T>>> = (0..n)
            .into_par_iter()
            .fold(Vec::<Update<T>>::new, |mut local_emits, u| {
                #[allow(clippy::redundant_locals)]
                let graph_ptr = graph_ptr;
                Self::with_update_scratch(|scratch_cell| {
                    let mut scratch_ref = scratch_cell.borrow_mut();
                    let UpdateScratch {
                        accepted,
                        new_accepted,
                        emitted,
                    } = &mut *scratch_ref;
                    accepted.clear();
                    new_accepted.clear();
                    emitted.clear();

                    // Load u's current sorted adjacency.
                    // SAFETY: u is unique across the parallel iterator.
                    let slot = unsafe { std::slice::from_raw_parts_mut(graph_ptr.0.add(u * r), r) };

                    for entry in slot.iter() {
                        if entry.is_sentinel() {
                            break;
                        }
                        accepted.push((entry.dist, entry.pid(), entry.is_new()));
                    }

                    // Greedy-accept in ascending distance order.
                    for &(v_dist, v, v_new) in accepted.iter() {
                        let mut pruned = false;
                        for &(_, w, w_new) in new_accepted.iter() {
                            if !v_new && !w_new {
                                continue;
                            }
                            let d_vw = self.distance(v, w);
                            // RNG prune rule from Alg. 4.
                            if v_dist >= d_vw {
                                emitted.push(Update::new(w as u32, v as u32, d_vw));
                                pruned = true;
                                break;
                            }
                        }
                        if !pruned {
                            new_accepted.push((v_dist, v, false));
                        }
                    }

                    // Write back to u's slot (all surviving edges old).
                    for (i, &(d, pid, _)) in new_accepted.iter().enumerate() {
                        slot[i] = Neighbour::new(pid, d, false);
                    }
                    for i in new_accepted.len()..r {
                        slot[i] = Neighbour::new(SENTINEL_PID, T::max_value(), false);
                    }

                    local_emits.append(emitted);
                });
                local_emits
            })
            .collect();

        let mut all_updates: Vec<Update<T>> = per_thread_updates.into_iter().flatten().collect();
        if !all_updates.is_empty() {
            all_updates.radix_sort_unstable();
            self.apply_sorted_updates(&all_updates, graph, r, &counter);
        }

        counter.load(AtomicOrdering::Relaxed)
    }

    /// One AddReverseEdges pass.
    ///
    /// Emits reverse edges from every current directed edge, then applies them
    /// via [`ApplySortedUpdates`] with per-target cap `R`. Returns the number
    /// of edge changes applied.
    ///
    /// ### Params
    ///
    /// * `graph` - Build-time graph (mutated in place)
    /// * `r` - Slot capacity per node
    ///
    /// ### Returns
    ///
    /// Number of edge changes applied.
    fn add_reverse_edges_pass(&self, graph: &mut [Neighbour<T>], r: usize) -> usize {
        let counter = AtomicUsize::new(0);
        let n = self.n;

        // Re-arm every surviving edge as new. UpdateNeighbors writes all
        // survivors back as old, so without this the `!v_new && !w_new` guard
        // in the prune loop skips essentially every pair from the second outer
        // round onwards and those rounds compute nothing. Matches the reference
        // (RNNDescent.cpp, add_reverse_edges).
        graph.par_iter_mut().for_each(|entry| {
            if !entry.is_sentinel() {
                entry.mark_new();
            }
        });

        // Collect reverse-edge updates in parallel.
        let per_thread: Vec<Vec<Update<T>>> = (0..n)
            .into_par_iter()
            .fold(Vec::<Update<T>>::new, |mut local, u| {
                let base = u * r;
                for entry in &graph[base..base + r] {
                    if entry.is_sentinel() {
                        break;
                    }
                    // Reverse edge: target = entry.pid, source = u.
                    local.push(Update::new(entry.pid() as u32, u as u32, entry.dist));
                }
                local
            })
            .collect();

        let mut updates: Vec<Update<T>> = per_thread.into_iter().flatten().collect();
        if !updates.is_empty() {
            updates.radix_sort_unstable();
            self.apply_sorted_updates(&updates, graph, r, &counter);
        }

        counter.load(AtomicOrdering::Relaxed)
    }

    /// Memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.graph.capacity() * std::mem::size_of::<u32>()
            + self.forest.memory_usage_bytes()
    }
}

///////////
// Query //
///////////

impl<T> RnnDescentIndex<T>
where
    T: AnnSearchFloat,
    Self: RnnDescentState<T>,
{
    /// Read a node's neighbours slice.
    ///
    /// ### Params
    ///
    /// * `node_id` - Source node
    ///
    /// ### Returns
    ///
    /// Slice of `R` point ids.
    #[inline(always)]
    fn get_neighbours_slot(&self, node_id: usize) -> &[u32] {
        let base = node_id * self.r;
        &self.graph[base..base + self.r]
    }

    /// Query the index for `k` nearest neighbours.
    ///
    /// Beam search using [`SearchState`] with pool size `ef` (default 100).
    ///
    /// The paper (Ono & Matsui 2023, Section 4.4, Eq. 4) introduces a
    /// query-time out-degree cap `K` that limits how many of each node's stored
    /// `R` neighbours the walk expands per hop. Since neighbours are stored
    /// sorted ascending by distance from the source, this is a `.take(K)` on
    /// the sorted slot. Typical values from the paper's ablation are
    /// `K = 16-64` even when the graph was built with `R = 96`. Default here is
    /// `min(32, R)`, matching the paper's sweet spot.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Optional beam width override (default 100)
    /// * `k_search` - Optional per-hop out-degree cap (default `min(32, R)`)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted ascending.
    pub fn query(
        &self,
        query: &[T],
        k: usize,
        ef_search: Option<usize>,
        k_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query.len())?;

        let ef = ef_search.unwrap_or(100).max(k);
        let k_hop = k_search.unwrap_or(32).min(self.r).max(1);

        Self::with_search_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let query_norm = if self.metric == Dist::Cosine {
                T::calculate_l2_norm(query)
            } else {
                T::one()
            };

            // Seed the beam from the kd-forest. The forest returns already-near
            // candidates, so we can safely push all of them into
            // `working_sorted`.
            let init_candidates = (ef / 2).max(2 * k).min(self.n);
            let search_k = init_candidates * 3;
            let (seed_ids, seed_dists) =
                self.forest.query(query, init_candidates, Some(search_k))?;

            for (id, d) in seed_ids.iter().zip(seed_dists.iter()) {
                if state.is_visited(*id) {
                    continue;
                }
                state.mark_visited(*id);
                let od = OrderedFloat(*d);
                state.candidates.push(Reverse((od, *id)));
                state.working_sorted.insert((od, *id), ef);
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

                let neighbours = self.get_neighbours_slot(current_id);
                for &pid in neighbours.iter().take(k_hop) {
                    if pid as usize == SENTINEL_PID {
                        break;
                    }
                    let n_idx = pid as usize;
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

    /// Query using a matrix row reference (stride-optimised).
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: Option<usize>,
        k_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search, k_search);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, ef_search, k_search)
    }

    /// Generate the full kNN graph by querying every internal vector.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per row
    /// * `ef_search` - Optional beam width override
    /// * `k_search` - Optional per-hop out-degree cap (default `min(32, R)`)
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
        k_search: Option<usize>,
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
                    let count = counter.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query(vec, k, ef_search, k_search)
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
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for RnnDescentIndex<T>
where
    T: AnnSearchFloat,
    Self: RnnDescentState<T>,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, None, None)
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
impl<T> IndexIo for RnnDescentIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "rnn_descent";
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

    fn build_default(mat: &Mat<f32>, metric: Dist) -> RnnDescentIndex<f32> {
        let params = RnnDescentBuildParams::new(10, 32, 3, 8);
        RnnDescentIndex::<f32>::build(mat.as_ref(), metric, params, None, 42, false).unwrap()
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
    fn query_finds_self_euclidean() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = idx.query(&query, 1, None, None).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn query_finds_self_cosine() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::Cosine);
        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = idx.query(&query, 1, None, None).unwrap();
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn distances_ascending() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        let query = vec![0.5_f32, 0.5, 0.0];
        let (_, distances) = idx.query(&query, 4, None, None).unwrap();
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
            let (indices, distances) = idx.query(&query, k, None, None).unwrap();
            assert_eq!(indices.len(), k);
            assert_eq!(distances.len(), k);
        }
    }

    #[test]
    fn degree_within_r_cap() {
        let n = 200;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = RnnDescentBuildParams::new(20, 32, 3, 8);
        let idx = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            42,
            false,
        )
        .unwrap();

        for node in 0..idx.n {
            let base = node * idx.r;
            let slot = &idx.graph[base..base + idx.r];
            let mut deg = 0;
            for &pid in slot {
                if pid as usize == SENTINEL_PID {
                    break;
                }
                deg += 1;
            }
            assert!(deg <= idx.r);
        }
    }

    #[test]
    fn per_node_neighbours_sorted() {
        let n = 200;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = RnnDescentBuildParams::new(20, 32, 3, 8);
        let idx = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            42,
            false,
        )
        .unwrap();

        // The final graph is ids only, so recompute distances to check ordering.
        for node in 0..idx.n {
            let base = node * idx.r;
            let slot = &idx.graph[base..base + idx.r];
            let mut prev = f32::MIN;
            for &pid in slot {
                if pid as usize == SENTINEL_PID {
                    break;
                }
                let d = idx.distance(node, pid as usize);
                assert!(d >= prev, "Distances not ascending at node {}", node);
                prev = d;
            }
        }
    }

    #[test]
    fn no_self_loops() {
        let mat = simple_matrix();
        let idx = build_default(&mat, Dist::SquaredEuclidean);
        for node in 0..idx.n {
            let base = node * idx.r;
            for &pid in &idx.graph[base..base + idx.r] {
                if pid as usize == SENTINEL_PID {
                    break;
                }
                assert_ne!(pid as usize, node);
            }
        }
    }

    #[test]
    fn recall_linear_data() {
        let n = 200;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = RnnDescentBuildParams::new(20, 48, 3, 8);
        let idx = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            42,
            false,
        )
        .unwrap();

        let query = vec![0.0_f32; dim];
        let (indices, _) = idx.query(&query, 5, Some(80), None).unwrap();
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

        let params = RnnDescentBuildParams::new(20, 64, 4, 10);
        let idx = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            42,
            false,
        )
        .unwrap();

        let recall = idx.validate_index(5, 42, Some(50)).unwrap();
        assert!(recall > 0.85, "Recall too low: {}", recall);
    }

    #[test]
    fn reproducible_same_seed() {
        let n = 100;
        let dim = 4;
        let mat = linear_matrix(n, dim);
        let params = RnnDescentBuildParams::new(15, 32, 3, 6);
        let a = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            7,
            false,
        )
        .unwrap();
        let b = RnnDescentIndex::<f32>::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            params,
            None,
            7,
            false,
        )
        .unwrap();
        assert_eq!(a.graph, b.graph);

        // Same seed → same forest → same query answer.
        let query = vec![0.5_f32; dim];
        let (a_ids, _) = a.query(&query, 3, None, None).unwrap();
        let (b_ids, _) = b.query(&query, 3, None, None).unwrap();
        assert_eq!(a_ids, b_ids);
    }
}
