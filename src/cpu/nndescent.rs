//! NNDescent implementation in ann-search-rs. Uses concepts of the original
//! implementation, PyNNDescent and EFANNA. Leverages Annoy over Kd forest for
//! graph initialisation (when not using Manhattan distance).

use faer::{RowRef};
use fixedbitset::FixedBitSet;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use rdst::RadixSort;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use thousands::*;

use crate::cpu::annoy::*;
use crate::cpu::kd_forest::*;
use crate::prelude::*;
use crate::utils::nndescent_utils::*;
use crate::utils::*;

///////////////////
// Thread locals //
///////////////////

thread_local! {
    static SORTED_F32: RefCell<SortedBuffer<(OrderedFloat<f32>, u32, bool)>> =
        RefCell::new(SortedBuffer::new());
    static SORTED_F64: RefCell<SortedBuffer<(OrderedFloat<f64>, u32, bool)>> =
        RefCell::new(SortedBuffer::new());
    static PID_SET: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };

    /// Scratch buffers reused across all `build_candidates` per-node closures
    /// on this thread, keyed as `(new_temp, old_temp)`.
    static CAND_SCRATCH: RefCell<CandScratch> =
        const { RefCell::new((Vec::new(), Vec::new())) };

    static QUERY_VISITED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
    static QUERY_CANDIDATES_F32: QueryCandF32 =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_CANDIDATES_F64: QueryCandF64 =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_RESULTS_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_RESULTS_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> =
        const { RefCell::new(BinaryHeap::new()) };
}

///////////
// Types //
///////////

/// Type alias for the query candidates for f32
pub type QueryCandF32 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>>;

/// Type alias for the query candidates for f64
pub type QueryCandF64 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>>;

/// Per-thread scratch pair for `build_candidates` phase 1 sampling:
/// `(new_temp, old_temp)` holding `(priority, pid)` entries.
type CandScratch = (Vec<(f64, usize)>, Vec<(f64, usize)>);

///////////////
// MetricFn  //
///////////////

/// Static-dispatch metric selector for the update kernel.
///
/// The inner loops in `generate_updates_for_chunk_impl` call
/// `M::distance_from_vec(idx, vec_p, norm_p, q)` where `M` is one of the
/// zero-sized types below. `vec_p` (and `norm_p` for Cosine) is hoisted
/// once per outer `p` iteration so the inner loop never re-slices the
/// flat vector buffer or re-fetches the cached norm.
/// Monomorphisation strips the runtime `Dist` branch out of the hot path.
trait MetricFn<T: AnnSearchFloat> {
    /// Distance between the hoisted vector `vec_p` (of node `p` with
    /// pre-fetched norm `norm_p` when Cosine) and the vector at internal
    /// index `q`.
    fn distance_from_vec(idx: &NNDescent<T>, vec_p: &[T], norm_p: T, q: usize) -> T;
}

/// Squared Euclidean metric.
struct SqEuclidMetric;
/// Cosine metric (assumes pre-computed norms in `NNDescent::norms`).
struct CosineMetric;
/// Manhattan (L1) metric.
struct ManhattanMetric;

impl<T: AnnSearchFloat> MetricFn<T> for SqEuclidMetric {
    #[inline(always)]
    fn distance_from_vec(idx: &NNDescent<T>, vec_p: &[T], _norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        T::euclidean_simd(vec_p, vec_q)
    }
}
impl<T: AnnSearchFloat> MetricFn<T> for CosineMetric {
    #[inline(always)]
    fn distance_from_vec(idx: &NNDescent<T>, vec_p: &[T], norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        let dot = T::dot_simd(vec_p, vec_q);
        T::one() - (dot / (norm_p * idx.norms[q]))
    }
}
impl<T: AnnSearchFloat> MetricFn<T> for ManhattanMetric {
    #[inline(always)]
    fn distance_from_vec(idx: &NNDescent<T>, vec_p: &[T], _norm_p: T, q: usize) -> T {
        let dim = idx.dim;
        let vec_q = &idx.vectors_flat[q * dim..(q + 1) * dim];
        T::manhattan_simd(vec_p, vec_q)
    }
}

////////////
// Forest //
////////////

/// Wrapper over the two forest types usable as an NNDescent initialiser.
///
/// Annoy gives better init quality but can't handle Manhattan; KdForest
/// handles all three metrics including Manhattan. Selection is automatic
/// based on the metric.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
enum Forest<T> {
    /// Annoy version
    Annoy(AnnoyIndex<T>),
    /// KdForest version
    Kd(KdTreeIndex<T>),
}

impl<T> Forest<T>
where
    T: AnnSearchFloat,
{
    /// Build the appropriate forest for the given metric.
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `n_trees` - The number of trees to use in the forest
    /// * `metric` - The distance metric to use, see [Dist].
    /// * `seed` - The random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// The [Forest].
    fn new(
        data: impl AnnMatrix<T>,
        n_trees: usize,
        metric: Dist,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        match metric {
            Dist::Manhattan => Ok(Forest::Kd(KdTreeIndex::new(data, n_trees, metric, seed))),
            _ => Ok(Forest::Annoy(AnnoyIndex::new(data, n_trees, metric, seed)?)),
        }
    }

    /// Number of trees in the underlying forest. Used to size the per-query
    /// search budget.
    ///
    /// ### Returns
    ///
    /// The number of trees.
    fn n_trees(&self) -> usize {
        match self {
            Forest::Annoy(f) => f.n_trees,
            Forest::Kd(f) => f.n_trees,
        }
    }

    /// Query the forest for approximate nearest neighbours.
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vector
    /// * `k` - Number of neighbours to return
    /// * `search_k` - The budget
    ///
    /// ### Returns
    ///
    /// The `(indices, dist)`
    fn query(
        &self,
        query_vec: &[T],
        k: usize,
        search_k: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        match self {
            Forest::Annoy(f) => f.query(query_vec, k, search_k),
            Forest::Kd(f) => f.query(query_vec, k, search_k),
        }
    }

    /// Memory footprint in bytes.
    ///
    /// ### Returns
    ///
    /// The memory usage
    fn memory_usage_bytes(&self) -> usize {
        match self {
            Forest::Annoy(f) => f.memory_usage_bytes(),
            Forest::Kd(f) => f.memory_usage_bytes(),
        }
    }
}

////////////////////
// NNDescentQuery //
////////////////////

/// Query interface for the NN-Descent index.
pub trait NNDescentQuery<T> {
    /// Internal query dispatch (delegates to metric-specific implementation).
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `query_norm` - Pre-computed L2 norm (Cosine only; ignored for Euclidean)
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width for search
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    fn query_internal(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors>;

    /// Beam search using Euclidean distance.
    fn query_euclidean(
        &self,
        query_vec: &[T],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors>;

    /// Beam search using Manhattan distance.
    fn query_manhattan(
        &self,
        query_vec: &[T],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors>;

    /// Beam search using Cosine distance.
    #[allow(clippy::too_many_arguments)]
    fn query_cosine(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors>;
}

//////////
// Main //
//////////

/// NN-Descent index for approximate nearest neighbour search.
///
/// Builds a k-NN graph via the NN-Descent algorithm, using an Annoy
/// forest for initialisation and beam search for querying.
///
/// ### Flat graph layout
///
/// Both the build-phase graph (`Vec<Neighbour<T>>`) and the final query
/// graph (`Vec<(usize, T)>`) are stored as contiguous 1D arrays of size
/// `n * k`. Node `i`'s neighbours occupy indices `[i*k .. (i+1)*k]`,
/// sorted by distance ascending. Empty trailing slots are filled with
/// sentinel values (`SENTINEL_PID`, `T::MAX`).
///
/// This layout gives better cache locality during graph updates and
/// queries compared to a `Vec<Vec<...>>` and eliminates per-node heap
/// allocations entirely.
///
/// ### Memory-efficient update strategy
///
/// During construction, candidate updates are processed in chunks
/// (~50k source nodes) to bound peak memory to
/// `O(chunk_size * max_candidates)` rather than `O(n * max_candidates)`.
/// Each chunk emits both edge directions, sorts by target, and applies
/// updates lock-free via disjoint pointer writes.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct NNDescent<T> {
    /// Original vectors, flattened row-major
    pub vectors_flat: Vec<T>,
    /// Dimensionality of the vectors
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node in the generated kNN graph
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric of the index
    metric: Dist,
    /// Forest used for graph initialisation and query entry points. Annoy for
    /// Euclidean/Cosine, KdForest for Manhattan.
    forest: Forest<T>,
    /// Flat k-NN graph of size `n * k`
    graph: Vec<(usize, T)>,
    /// Whether construction converged
    converged: bool,
    /// Original indices - for trait purposes
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for NNDescent<T>
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

impl<T> DimensionValidation for NNDescent<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

//////////
// Main //
//////////

impl<T> NNDescent<T>
where
    T: AnnSearchFloat,
    Self: ApplySortedUpdates<T>,
    Self: NNDescentQuery<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build a new NN-Descent index.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `metric` - Distance metric
    /// * `k` - Neighbours per node (default 30)
    /// * `max_candidates` - Max candidates per node per iteration
    /// * `max_iter` - Maximum iterations
    /// * `n_trees` - Annoy/Kd forest size
    /// * `delta` - Convergence threshold (fraction of edges updated)
    /// * `diversify_prob` - Bernoulli probability for the post-descent
    ///   RNG-rule prune over the forward+reverse candidate pool
    ///   (0 disables).
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: Dist,
        k: Option<usize>,
        max_candidates: Option<usize>,
        max_iter: Option<usize>,
        n_trees: Option<usize>,
        delta: T,
        diversify_prob: T,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let (vectors_flat, n, dim) = data.into_row_major();

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        let n_trees = n_trees.unwrap_or_else(|| {
            let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
            calculated.min(32)
        });

        let max_iter = max_iter.unwrap_or_else(|| {
            let calculated = ((n as f64).log2().round()) as usize;
            calculated.max(5)
        });

        let k = k.unwrap_or(30);
        let max_candidates = max_candidates.unwrap_or(k.min(60));

        let start = Instant::now();
        // Feed the forest the buffer we already flattened. Handing it the
        // caller's matrix again would walk the whole thing a second time.
        let forest = Forest::new((&vectors_flat[..], n, dim), n_trees, metric, seed)?;
        if verbose {
            println!("Built forest: {:.2?}", start.elapsed());
        }

        let builder = NNDescent {
            vectors_flat,
            dim,
            n,
            k,
            metric,
            norms,
            graph: Vec::new(),
            converged: false,
            forest,
            original_ids: (0..n).collect(),
        };

        let (build_graph, converged) =
            builder.generate_index(k, max_iter, delta, max_candidates, seed, verbose)?;

        let graph = if diversify_prob > T::zero() {
            builder.diversify_graph(&build_graph, k, diversify_prob, seed)
        } else {
            build_graph
        };

        Ok(NNDescent {
            vectors_flat: builder.vectors_flat,
            dim: builder.dim,
            n: builder.n,
            k,
            metric: builder.metric,
            norms: builder.norms,
            graph,
            converged,
            forest: builder.forest,
            original_ids: (0..n).collect(),
        })
    }

    /// Whether the algorithm converged during construction.
    pub fn index_converged(&self) -> bool {
        self.converged
    }

    /// Distance metric this index was built with.
    ///
    /// ### Returns
    ///
    /// The [`Dist`] metric embedded at build time.
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Borrow the flat kNN graph.
    ///
    /// Layout is `n * k` `(pid, distance)` pairs, sorted by distance
    /// ascending within each node's slot. Unused trailing slots hold
    /// [`SENTINEL_PID`] and `T::MAX`.
    ///
    /// ### Returns
    ///
    /// A slice view over the internal graph.
    pub fn graph(&self) -> &[(usize, T)] {
        &self.graph
    }

    /// Compute chunk size for memory-bounded update processing
    ///
    /// Targets roughly 200 MB of update storage per chunk based on the size
    /// of an `Update<T>` and the expected number of updates per source node.
    /// Clamped to at least 10k nodes (or the full dataset if smaller) and at
    /// most the total number of nodes.
    ///
    /// ### Params
    ///
    /// * `max_candidates` - Maximum candidates sampled per node per iteration
    ///
    /// ### Returns
    ///
    /// Number of source nodes to process per chunk
    fn compute_chunk_size(&self, max_candidates: usize) -> usize {
        const TARGET_BYTES: usize = 200 * 1024 * 1024;
        const BYTES_PER_UPDATE: usize = 24;

        let updates_per_source = max_candidates * 2;
        let bytes_per_source = updates_per_source * BYTES_PER_UPDATE;

        let chunk_size = TARGET_BYTES / bytes_per_source.max(1);
        let min_chunk = 10_000.min(self.n);
        chunk_size.clamp(min_chunk, self.n)
    }

    /// Initialise the flat k-NN graph using the Annoy forest
    ///
    /// Each node queries Annoy for `k+1` candidates, skips the self-match,
    /// and takes the next `k`. Results are marked new so they participate
    /// in the first iteration's local joins. Unused trailing slots are
    /// padded with sentinels.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per node
    ///
    /// ### Returns
    ///
    /// Flat graph of size `n * k` with Annoy-seeded initial neighbours
    fn init_with_forest(&self, k: usize) -> Result<Vec<Neighbour<T>>, AnnSearchErrors> {
        let sentinel = Neighbour::new(SENTINEL_PID, T::max_value(), false);
        let mut graph = vec![sentinel; self.n * k];
        graph.par_chunks_mut(k).enumerate().try_for_each(
            |(i, slot)| -> Result<(), AnnSearchErrors> {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                let search_k = k * self.forest.n_trees();
                let (indices, distances) = self.forest.query(query, k + 1, Some(search_k))?;
                for (j, (idx, dist)) in indices
                    .into_iter()
                    .zip(distances)
                    .skip(1)
                    .take(k)
                    .enumerate()
                {
                    slot[j] = Neighbour::new(idx, dist, true);
                }
                Ok(())
            },
        )?;

        Ok(graph)
    }

    /// Run the main NN-Descent algorithm with chunked updates
    ///
    /// Alternates between building candidate lists (new and old, forward and
    /// reverse) and applying pairwise distance updates back into the graph.
    /// Processes candidates in chunks to bound peak memory. Terminates early
    /// when the fraction of edge updates drops below `delta`.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per node
    /// * `max_iter` - Maximum iterations before giving up
    /// * `delta` - Convergence threshold (fraction of edges updated)
    /// * `max_candidates` - Max candidates sampled per node per iteration
    /// * `seed` - Random seed for per-iteration sampling
    /// * `verbose` - Print progress information
    ///
    /// ### Returns
    ///
    /// Tuple of (flat graph as `(pid, dist)` pairs, converged flag)
    fn generate_index(
        &self,
        k: usize,
        max_iter: usize,
        delta: T,
        max_candidates: usize,
        seed: usize,
        verbose: bool,
    ) -> Result<(Vec<(usize, T)>, bool), AnnSearchErrors> {
        if verbose {
            println!(
                "Running NN-Descent: {} samples, max_candidates={}",
                self.n.separate_with_underscores(),
                max_candidates
            );
        }

        let mut converged = false;

        let start = Instant::now();
        let mut graph = self.init_with_forest(k)?;

        if verbose {
            println!("Queried Annoy index: {:.2?}", start.elapsed());
        }

        let chunk_size = self.compute_chunk_size(max_candidates);
        let n_chunks = self.n.div_ceil(chunk_size);

        if verbose {
            println!(
                " Using chunk size {} ({} chunks) for memory-efficient updates",
                chunk_size.separate_with_underscores(),
                n_chunks
            );
        }

        let mut new_cands = vec![Vec::with_capacity(max_candidates * 2); self.n];
        let mut old_cands = vec![Vec::with_capacity(max_candidates * 2); self.n];
        let mut new_cands_sym = vec![Vec::with_capacity(max_candidates); self.n];
        let mut old_cands_sym = vec![Vec::with_capacity(max_candidates); self.n];

        for iter in 0..max_iter {
            let updates_count = AtomicUsize::new(0);
            let iter_seed = (seed as u64).wrapping_add(iter as u64);

            if verbose {
                println!(" Preparing candidates for iter {}", iter + 1);
            }
            self.build_candidates(
                &graph,
                k,
                max_candidates,
                iter_seed,
                &mut new_cands,
                &mut old_cands,
                &mut new_cands_sym,
                &mut old_cands_sym,
            );

            self.mark_as_old(&mut graph, k, &new_cands);

            if verbose {
                println!(
                    " Processing updates for iter {} ({} chunks)",
                    iter + 1,
                    n_chunks
                );
            }

            for chunk_idx in 0..n_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = (chunk_start + chunk_size).min(self.n);

                let mut chunk_updates = self.generate_updates_for_chunk(
                    &new_cands,
                    &old_cands,
                    &graph,
                    k,
                    chunk_start,
                    chunk_end,
                );

                chunk_updates.radix_sort_unstable();

                self.apply_sorted_updates(&chunk_updates, &mut graph, k, &updates_count);
            }

            let update_count = updates_count.load(Ordering::Relaxed);
            let update_rate = T::from_usize(update_count).unwrap()
                / T::from_usize(self.n * max_candidates).unwrap();

            if verbose {
                println!(
                    "  Iter {}: {} edge updates (rate={:.4})",
                    iter + 1,
                    update_count.separate_with_underscores(),
                    update_rate.to_f64().unwrap(),
                );
            }

            if update_rate < delta {
                if verbose {
                    println!("  Converged after {} iterations", iter + 1);
                }
                converged = true;
                break;
            }
        }

        if verbose {
            println!("Total time: {:.2?}", start.elapsed());
        }

        let res = graph.into_iter().map(|n| (n.pid(), n.dist)).collect();

        Ok((res, converged))
    }

    /// Build candidate lists for the local join step.
    ///
    /// For each node, samples up to `max_candidates` new and old neighbours,
    /// then adds symmetric reverse candidates to ensure connectivity.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph
    /// * `k` - Neighbours per node
    /// * `max_candidates` - Maximum candidates to sample per node
    /// * `iter_seed` - Per-iteration seed for reproducible sampling
    /// * `new_cands` - Output: sampled new (unexplored) neighbours per node
    /// * `old_cands` - Output: sampled old (explored) neighbours per node
    /// * `new_cands_sym` - Output: reverse edges into `new_cands` (cleared and
    ///   repopulated)
    /// * `old_cands_sym` - Output: reverse edges into `old_cands` (cleared and
    ///   repopulated)
    #[allow(clippy::too_many_arguments)]
    fn build_candidates(
        &self,
        graph: &[Neighbour<T>],
        k: usize,
        max_candidates: usize,
        iter_seed: u64,
        new_cands: &mut [Vec<usize>],
        old_cands: &mut [Vec<usize>],
        new_cands_sym: &mut [Vec<usize>],
        old_cands_sym: &mut [Vec<usize>],
    ) {
        // Phase 1: Parallel sampling - each thread writes only to its own node
        let n = self.n;
        new_cands
            .par_iter_mut()
            .zip(old_cands.par_iter_mut())
            .enumerate()
            .for_each(|(i, (new_c, old_c))| {
                CAND_SCRATCH.with(|cell| {
                    let mut b = cell.borrow_mut();
                    let (new_temp, old_temp) = &mut *b;

                    new_c.clear();
                    old_c.clear();
                    new_temp.clear();
                    old_temp.clear();

                    let mut rng = SmallRng::seed_from_u64(iter_seed.wrapping_add(i as u64));
                    let base = i * k;

                    for slot in &graph[base..base + k] {
                        if slot.is_sentinel() {
                            continue;
                        }
                        let j = slot.pid();
                        if j >= n {
                            continue;
                        }

                        let priority = rng.random::<f64>();
                        if slot.is_new() {
                            new_temp.push((priority, j));
                        } else {
                            old_temp.push((priority, j));
                        }
                    }

                    // O(n) partial sort instead of O(n log n) full sort
                    if new_temp.len() > max_candidates {
                        new_temp.select_nth_unstable_by(max_candidates - 1, |a, b| {
                            a.0.partial_cmp(&b.0).unwrap()
                        });
                        new_temp.truncate(max_candidates);
                    }
                    new_c.extend(new_temp.iter().map(|&(_, idx)| idx));

                    if old_temp.len() > max_candidates {
                        old_temp.select_nth_unstable_by(max_candidates - 1, |a, b| {
                            a.0.partial_cmp(&b.0).unwrap()
                        });
                        old_temp.truncate(max_candidates);
                    }
                    old_c.extend(old_temp.iter().map(|&(_, idx)| idx));
                });
            });

        // Phase 2: Symmetric candidates via parallel target-chunk scan.
        //
        // Each thread owns a disjoint slice of `*_sym` and scans all sources
        // once, picking up entries whose target lands in its range. This
        // avoids per-target locking; the cost is that each thread walks the
        // full source list, but that walk is a linear read of small vecs.
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = n.div_ceil(n_threads).max(1);
        new_cands_sym
            .par_chunks_mut(chunk)
            .zip(old_cands_sym.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(ci, (new_sym_chunk, old_sym_chunk))| {
                for v in new_sym_chunk.iter_mut() {
                    v.clear();
                }
                for v in old_sym_chunk.iter_mut() {
                    v.clear();
                }
                let target_start = ci * chunk;
                let new_end = target_start + new_sym_chunk.len();
                let old_end = target_start + old_sym_chunk.len();
                for src_i in 0..n {
                    for &j in &new_cands[src_i] {
                        if j >= target_start && j < new_end {
                            new_sym_chunk[j - target_start].push(src_i);
                        }
                    }
                    for &j in &old_cands[src_i] {
                        if j >= target_start && j < old_end {
                            old_sym_chunk[j - target_start].push(src_i);
                        }
                    }
                }
            });

        // Phase 3: Merge symmetric, sort, dedup (parallel, per-node independent)
        new_cands
            .par_iter_mut()
            .zip(old_cands.par_iter_mut())
            .zip(new_cands_sym.par_iter())
            .zip(old_cands_sym.par_iter())
            .for_each(|(((new_c, old_c), new_sym), old_sym)| {
                new_c.extend_from_slice(new_sym);
                new_c.sort_unstable();
                new_c.dedup();

                old_c.extend_from_slice(old_sym);
                old_c.sort_unstable();
                old_c.dedup();
            });
    }

    /// Mark neighbours as old if they were sampled into the new-candidate list
    ///
    /// After sampling, any neighbour that survived into `new_cands[i]` will
    /// have been "explored" during this iteration's local joins, so flip its
    /// flag so subsequent iterations treat it as old.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph (mutated in place)
    /// * `k` - Neighbours per node
    /// * `new_cands` - Sorted new-candidate lists per node
    fn mark_as_old(&self, graph: &mut [Neighbour<T>], k: usize, new_cands: &[Vec<usize>]) {
        graph
            .par_chunks_mut(k)
            .zip(new_cands.par_iter())
            .for_each(|(slots, new_c)| {
                if new_c.is_empty() {
                    return;
                }
                for slot in slots.iter_mut() {
                    if slot.is_sentinel() {
                        continue;
                    }
                    if slot.is_new() && new_c.binary_search(&slot.pid()).is_ok() {
                        slot.mark_old();
                    }
                }
            });
    }

    /// Generate distance updates from a chunk of source nodes.
    ///
    /// Emits both edge directions `(p, q, d)` and `(q, p, d)` so that
    /// the caller can sort by target and apply lock-free.
    ///
    /// ### Params
    ///
    /// * `new_cands` - New (unexplored) candidate lists per node
    /// * `old_cands` - Old (explored) candidate lists per node
    /// * `graph` - Current flat k-NN graph
    /// * `k` - Neighbours per node
    /// * `chunk_start` - First source node index (inclusive)
    /// * `chunk_end` - Last source node index (exclusive)
    ///
    /// ### Returns
    ///
    /// Unsorted list of `(target, source, distance)` update triples
    fn generate_updates_for_chunk(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Neighbour<T>],
        k: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Vec<Update<T>> {
        match self.metric {
            Dist::SquaredEuclidean => self.generate_updates_for_chunk_impl::<SqEuclidMetric>(
                new_cands,
                old_cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
            Dist::Cosine => self.generate_updates_for_chunk_impl::<CosineMetric>(
                new_cands,
                old_cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
            Dist::Manhattan => self.generate_updates_for_chunk_impl::<ManhattanMetric>(
                new_cands,
                old_cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
        }
    }

    /// Monomorphised inner kernel for chunked update generation.
    ///
    /// `M` selects the distance function at compile time so the branch on
    /// `self.metric` is stripped out of the hot loop entirely.
    fn generate_updates_for_chunk_impl<M: MetricFn<T>>(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Neighbour<T>],
        k: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Vec<Update<T>> {
        let dim = self.dim;
        let has_norms = !self.norms.is_empty();
        (chunk_start..chunk_end)
            .into_par_iter()
            .fold(
                || Vec::with_capacity(16_384),
                |mut updates, i| {
                    let get_threshold = |idx: usize| -> T { graph[idx * k + k - 1].dist };

                    // new-new pairs. Hoist vec_p (and norm_p for Cosine)
                    // once per outer p so the inner q-loop only re-slices
                    // vec_q.
                    for j in 0..new_cands[i].len() {
                        let p = new_cands[i][j];
                        if p >= self.n {
                            continue;
                        }
                        let p_threshold = get_threshold(p);
                        let vec_p = &self.vectors_flat[p * dim..(p + 1) * dim];
                        let norm_p = if has_norms { self.norms[p] } else { T::zero() };

                        for l in (j + 1)..new_cands[i].len() {
                            let q = new_cands[i][l];
                            if q >= self.n || p == q {
                                continue;
                            }
                            let d = M::distance_from_vec(self, vec_p, norm_p, q);
                            if d <= p_threshold || d <= get_threshold(q) {
                                updates.push(Update::new(p as u32, q as u32, d));
                                updates.push(Update::new(q as u32, p as u32, d));
                            }
                        }
                    }

                    // new-old pairs. Same hoist as above.
                    for &p in &new_cands[i] {
                        if p >= self.n {
                            continue;
                        }
                        let p_threshold = get_threshold(p);
                        let vec_p = &self.vectors_flat[p * dim..(p + 1) * dim];
                        let norm_p = if has_norms { self.norms[p] } else { T::zero() };

                        for &q in &old_cands[i] {
                            if q >= self.n || p == q {
                                continue;
                            }
                            let d = M::distance_from_vec(self, vec_p, norm_p, q);
                            if d <= p_threshold || d <= get_threshold(q) {
                                updates.push(Update::new(p as u32, q as u32, d));
                                updates.push(Update::new(q as u32, p as u32, d));
                            }
                        }
                    }

                    updates
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                if a.len() >= b.len() {
                    a.extend_from_slice(&b);
                    a
                } else {
                    b.extend_from_slice(&a);
                    b
                }
            })
    }

    /// Calculate distance between two indexed points under the index metric
    ///
    /// ### Params
    ///
    /// * `i` - Index of first vector
    /// * `j` - Index of second vector
    ///
    /// ### Returns
    ///
    /// Distance value under `self.metric`
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
            Dist::Manhattan => self.manhattan_distance(i, j),
        }
    }

    /// Build reverse adjacency from the directed k-NN graph.
    ///
    /// For each source node `u`, walks its `k` outgoing neighbours in `graph`
    /// and pushes `(u, d(u, v))` into `reverse[v]`. Sentinels are skipped.
    /// Distances are inherited from the forward edge so no extra distance
    /// computation is performed.
    ///
    /// ### Params
    ///
    /// * `graph` - Flat directed k-NN graph, row `i` at `[i*k..(i+1)*k]`,
    ///   sorted ascending, sentinel-padded.
    /// * `k` - Neighbours per node.
    ///
    /// ### Returns
    ///
    /// `Vec<Vec<(usize, T)>>` of length `n`. Entry `v` lists `(u, d(u, v))`
    /// for every source `u` that has `v` in its forward list.
    fn build_reverse_adjacency(
        &self,
        graph: &[(usize, T)],
        k: usize,
    ) -> Vec<Vec<(usize, T)>> {
        let mut reverse: Vec<Vec<(usize, T)>> = (0..self.n).map(|_| Vec::new()).collect();
        for u in 0..self.n {
            let row = &graph[u * k..(u + 1) * k];
            for &(v, d) in row {
                if v == SENTINEL_PID {
                    continue;
                }
                reverse[v].push((u, d));
            }
        }
        reverse
    }

    /// Diversify the graph via probabilistic RNG-rule pruning over the
    /// forward + reverse edge pool.
    ///
    /// For each node `u`, the input pool merges `graph[u]` with all nodes
    /// that have `u` as a forward neighbour, deduplicates by pid, and
    /// sorts ascending by distance. The RNG rule then prunes an entry
    /// `v` from the pool if some already-kept neighbour `w` satisfies
    /// `d(w, v) < d(u, v)` (Bernoulli coin with probability `prune_prob`).
    /// The output row is filled with up to `k` kept entries; short rows
    /// are topped up from the pruned-out tail in distance order so
    /// out-degree does not shrink.
    ///
    /// Because iteration is ascending in `d(u, v)`, every already-kept
    /// `w` satisfies `d(u, w) <= d(u, v)` by construction, so the
    /// classical two-sided RNG rule collapses to the single check
    /// `d(w, v) < d(u, v)`.
    ///
    /// ### Params
    ///
    /// * `graph` - Input flat directed k-NN graph, sentinel-padded.
    /// * `k` - Neighbours per node.
    /// * `prune_prob` - Bernoulli probability of applying the RNG rule
    ///   per candidate/kept pair. Zero disables pruning.
    /// * `seed` - Base RNG seed. Per-node seed is `seed + i`.
    ///
    /// ### Returns
    ///
    /// New flat graph of length `n * k`. Sentinel-padded only when the
    /// merged pool for a row is smaller than `k` (small `n` or
    /// disconnected components).
    fn diversify_graph(
        &self,
        graph: &[(usize, T)],
        k: usize,
        prune_prob: T,
        seed: usize,
    ) -> Vec<(usize, T)> {
        let reverse = self.build_reverse_adjacency(graph, k);
        let mut result = vec![(SENTINEL_PID, T::max_value()); self.n * k];
        let prune_prob_f64 = prune_prob.to_f64().unwrap();

        result
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, out_slot)| {
                let mut pool: Vec<(usize, T)> = Vec::with_capacity(2 * k);
                for &entry in &graph[i * k..(i + 1) * k] {
                    if entry.0 != SENTINEL_PID && entry.0 != i {
                        pool.push(entry);
                    }
                }
                for &entry in &reverse[i] {
                    if entry.0 != i {
                        pool.push(entry);
                    }
                }
                if pool.is_empty() {
                    return;
                }

                // Dedupe by pid, keeping the smallest distance per pid.
                pool.sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0).then_with(|| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                });
                pool.dedup_by_key(|x| x.0);

                // Re-sort by distance ascending for the RNG-rule sweep.
                pool.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(i as u64));
                let mut kept: Vec<(usize, T)> = Vec::with_capacity(k);
                let mut pruned: Vec<(usize, T)> = Vec::new();

                kept.push(pool[0]);

                for &(cand_idx, cand_dist) in &pool[1..] {
                    let mut should_keep = true;
                    for &(kept_idx, _) in &kept {
                        let dist_to_kept = self.distance(cand_idx, kept_idx);
                        if dist_to_kept < cand_dist
                            && rng.random::<f64>() < prune_prob_f64
                        {
                            should_keep = false;
                            break;
                        }
                    }
                    if should_keep {
                        kept.push((cand_idx, cand_dist));
                        if kept.len() == k {
                            break;
                        }
                    } else {
                        pruned.push((cand_idx, cand_dist));
                    }
                }

                // Top up short rows from the pruned tail in distance order.
                if kept.len() < k {
                    for &entry in &pruned {
                        if kept.len() == k {
                            break;
                        }
                        kept.push(entry);
                    }
                }

                for (j, &entry) in kept.iter().enumerate() {
                    out_slot[j] = entry;
                }
            });

        result
    }

    ///////////
    // Query //
    ///////////

    /// Return the neighbours slice for node `idx` from the query graph
    ///
    /// ### Params
    ///
    /// * `idx` - Node index
    ///
    /// ### Returns
    ///
    /// Slice of `k` `(pid, distance)` pairs, possibly containing sentinels
    #[inline]
    fn graph_neighbours(&self, idx: usize) -> &[(usize, T)] {
        &self.graph[idx * self.k..(idx + 1) * self.k]
    }

    /// Query for k nearest neighbours using beam search.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width. Higher values improve recall at the
    ///   cost of latency. Defaults to `max(2*k, 50)` clamped to 200.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let k = k.min(self.n);
        let ef = ef_search.unwrap_or_else(|| (k * 2).clamp(50, 200)).max(k);

        let query_norm = if self.metric == Dist::Cosine {
            query_vec.iter().map(|x| *x * *x).sum::<T>().sqrt()
        } else {
            T::one()
        };

        #[allow(clippy::needless_question_mark)]
        Ok(self.query_internal(query_vec, query_norm, k, ef)?)
    }

    /// Query using a matrix row reference.
    ///
    /// Uses a zero-copy path when stride is 1, otherwise copies to a
    /// temporary vector.
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

    /// Generate a kNN graph by querying every vector in the index.
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        ef_search: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

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

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Index size `in n bytes`
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.forest.memory_usage_bytes();
        total += self.graph.capacity() * std::mem::size_of::<(usize, T)>();

        total
    }
}

///////////////////////////
// Trait implementations //
///////////////////////////

////////////////////////
// ApplySortedUpdates //
////////////////////////

/// Generates the `ApplySortedUpdates` impl for a concrete float type.
///
/// The logic is identical for f32 and f64; only the thread-local storage
/// keys differ.
macro_rules! impl_apply_sorted_updates {
    ($float:ty, $sorted_tls:ident) => {
        impl ApplySortedUpdates<$float> for NNDescent<$float> {
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
                    $sorted_tls.with(|sorted_cell| {
                        PID_SET.with(|set_cell| {
                            let mut sorted = sorted_cell.borrow_mut();
                            let mut pid_set = set_cell.borrow_mut();

                            sorted.clear();
                            if pid_set.len() < self.n {
                                pid_set.grow(self.n);
                            }

                            let start_idx = target * k;

                            // SAFETY: Each thread processes a unique target.
                            // No two threads alias the same slice.
                            let target_slice = unsafe {
                                std::slice::from_raw_parts_mut(graph_ptr.0.add(start_idx), k)
                            };

                            let mut edge_updates = 0usize;

                            // Load current neighbours in ascending distance order.
                            for n in target_slice.iter() {
                                if n.is_sentinel() {
                                    continue;
                                }
                                let pid = n.pid();
                                sorted.insert((OrderedFloat(n.dist), pid as u32, n.is_new()), k);
                                pid_set.insert(pid);
                            }

                            // Merge incoming updates.
                            for update in segment {
                                let src = update.source as usize;
                                if pid_set.contains(src) {
                                    continue;
                                }
                                let evicted_pid: Option<u32> = if sorted.len() == k {
                                    sorted.top().map(|&(_, pid, _)| pid)
                                } else {
                                    None
                                };
                                let entry = (OrderedFloat(update.dist), update.source, true);
                                if sorted.insert(entry, k) {
                                    if let Some(pid) = evicted_pid {
                                        pid_set.remove(pid as usize);
                                    }
                                    pid_set.insert(src);
                                    edge_updates += 1;
                                }
                            }

                            if edge_updates > 0 {
                                updates_count.fetch_add(edge_updates, Ordering::Relaxed);

                                for (i, &(OrderedFloat(d), pid, is_new)) in
                                    sorted.data().iter().enumerate()
                                {
                                    target_slice[i] = Neighbour::new(pid as usize, d, is_new);
                                }
                                for i in sorted.len()..k {
                                    target_slice[i] =
                                        Neighbour::new(SENTINEL_PID, <$float>::MAX, false);
                                }
                            }

                            // Clear pid_set entries left in sorted.
                            for &(_, pid, _) in sorted.data().iter() {
                                pid_set.remove(pid as usize);
                            }
                        })
                    })
                });
            }
        }
    };
}

impl_apply_sorted_updates!(f32, SORTED_F32);
impl_apply_sorted_updates!(f64, SORTED_F64);

////////////////////
// NNDescentQuery //
////////////////////

/// Generates the `NNDescentQuery` impl for a concrete float type.
macro_rules! impl_nndescent_query {
    ($float:ty, $cand_tls:ident, $res_tls:ident) => {
        impl NNDescentQuery<$float> for NNDescent<$float> {
            fn query_internal(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                QUERY_VISITED.with(|visited_cell| {
                    $cand_tls.with(|cand_cell| {
                        $res_tls.with(|res_cell| {
                            let mut visited = visited_cell.borrow_mut();
                            let mut candidates = cand_cell.borrow_mut();
                            let mut results = res_cell.borrow_mut();

                            visited.clear();
                            visited.grow(self.n);
                            candidates.clear();
                            results.clear();

                            match self.metric {
                                Dist::SquaredEuclidean => self.query_euclidean(
                                    query_vec,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                                Dist::Manhattan => self.query_manhattan(
                                    query_vec,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                                Dist::Cosine => self.query_cosine(
                                    query_vec,
                                    query_norm,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                            }
                        })
                    })
                })
            }

            #[inline(always)]
            fn query_euclidean(
                &self,
                query_vec: &[$float],
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                let init_candidates = (ef / 2).max(2 * k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k))?;

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.euclidean_distance_to_query(entry_idx, query_vec);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.euclidean_distance_to_query(nbr_idx, query_vec);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                Ok(final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip())
            }

            #[inline(always)]
            fn query_manhattan(
                &self,
                query_vec: &[$float],
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                let init_candidates = (ef / 2).max(2 * k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k))?;

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.manhattan_distance_to_query(entry_idx, query_vec);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.manhattan_distance_to_query(nbr_idx, query_vec);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                Ok(final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip())
            }

            #[inline(always)]
            fn query_cosine(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                let init_candidates = (ef / 2).max(k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k))?;

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.cosine_distance_to_query(entry_idx, query_vec, query_norm);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.cosine_distance_to_query(nbr_idx, query_vec, query_norm);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                Ok(final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip())
            }
        }
    };
}

impl_nndescent_query!(f32, QUERY_CANDIDATES_F32, QUERY_RESULTS_F32);
impl_nndescent_query!(f64, QUERY_CANDIDATES_F64, QUERY_RESULTS_F64);

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for NNDescent<T>
where
    T: AnnSearchFloat,
    Self: ApplySortedUpdates<T>,
    Self: NNDescentQuery<T>,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        // Default budget
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
impl<T> IndexIo for NNDescent<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "nndescent";
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        let data = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    /// Return the non-sentinel neighbours for node `i`.
    fn neighbours(index: &NNDescent<f32>, i: usize) -> Vec<(usize, f32)> {
        index.graph[i * index.k..(i + 1) * index.k]
            .iter()
            .copied()
            .filter(|&(pid, _)| pid != SENTINEL_PID)
            .collect()
    }

    fn neighbours_f64(index: &NNDescent<f64>, i: usize) -> Vec<(usize, f64)> {
        index.graph[i * index.k..(i + 1) * index.k]
            .iter()
            .copied()
            .filter(|&(pid, _)| pid != SENTINEL_PID)
            .collect()
    }

    #[test]
    fn test_nndescent_build_euclidean() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), 5 * 3);
        for i in 0..5 {
            assert!(neighbours(&index, i).len() <= 3);
        }
    }

    #[test]
    fn test_nndescent_build_cosine() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Cosine,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), 5 * 3);
        assert!(!index.norms.is_empty());
    }

    #[test]
    fn test_nndescent_query() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(50)).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_nndescent_convergence() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(100),
            None,
            0.5,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), 5 * 3);
    }

    #[test]
    fn test_nndescent_reproducibility() {
        let mat = create_simple_matrix();

        let g1 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();
        let g2 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(g1.graph.len(), g2.graph.len());
        for i in 0..g1.n {
            assert_eq!(neighbours(&g1, i).len(), neighbours(&g2, i).len());
        }
    }

    #[test]
    fn test_nndescent_k_parameter() {
        let mat = create_simple_matrix();

        let gk2 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();
        let gk4 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(4),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        for i in 0..5 {
            assert!(neighbours(&gk2, i).len() <= 2);
        }
        for i in 0..5 {
            assert!(neighbours(&gk4, i).len() <= 4);
        }
    }

    #[test]
    fn test_nndescent_larger_dataset() {
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            None,
            Some(15),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), n * 10);
        for i in 0..n {
            let nbrs = neighbours(&index, i);
            assert!(nbrs.len() <= 10);
            assert!(!nbrs.is_empty());
        }
    }

    #[test]
    fn test_nndescent_distance_ordering() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        for i in 0..5 {
            let nbrs = neighbours(&index, i);
            for w in nbrs.windows(2) {
                assert!(w[1].1 >= w[0].1);
            }
        }
    }

    #[test]
    fn test_nndescent_with_f64() {
        let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

        let index = NNDescent::<f64>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), 3 * 2);
        for i in 0..3 {
            assert!(!neighbours_f64(&index, i).is_empty());
        }
    }

    #[test]
    fn test_nndescent_quality() {
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..10 {
            let offset = i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }
        for i in 0..10 {
            let offset = 10.0 + i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            None,
            Some(20),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        let nbrs_0 = neighbours(&index, 0);
        let in_cluster = nbrs_0.iter().filter(|(idx, _)| *idx < 10).count();
        assert!(in_cluster >= 3);

        let nbrs_10 = neighbours(&index, 10);
        let in_cluster_2 = nbrs_10.iter().filter(|(idx, _)| *idx >= 10).count();
        assert!(in_cluster_2 >= 3);
    }

    #[test]
    fn test_nndescent_diversify() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.5,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph.len(), 5 * 3);
        for i in 0..5 {
            assert!(!neighbours(&index, i).is_empty());
        }
    }
}
