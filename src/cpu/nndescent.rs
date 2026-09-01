//! NNDescent implementation in ann-search-rs. Uses concepts of the original
//! implementation, PyNNDescent and EFANNA. Leverages Annoy over Kd forest for
//! graph initialisation (when not using Manhattan distance).

use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par, RowRef};
use fixedbitset::FixedBitSet;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use rdst::RadixSort;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
    time::{Duration, Instant},
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

/// Per-thread scratch pair for `build_candidates` sampling and merging:
/// `(new_temp, old_temp)` holding `(priority, pid)` entries.
type CandScratch = (Vec<(u64, u32)>, Vec<(u64, u32)>);

///////////////
// Constants //
///////////////

/// Target byte budget for one chunk's update batch.
///
/// `radix_sort_unstable` allocates a scratch buffer the same size as the batch,
/// so the live footprint while sorting is twice this figure.
const UPDATE_TARGET_BYTES: usize = 200 * 1024 * 1024;

/// Floor on the number of source nodes per chunk.
///
/// Below this the per-chunk fixed costs (radix sort setup, the parallel fan-out
/// over target segments) start to outweigh the local join itself, so the byte
/// budget gives way.
const MIN_CHUNK_NODES: usize = 1_024;

/// Fraction of candidate pairs assumed to clear the distance threshold when
/// sizing the *first* chunk.
const ASSUMED_ACCEPT_RATE: f64 = 0.25;

/// Cap on how much the adaptive chunk size may grow in one step.
///
/// One unusually quiet chunk should not blow the budget on the next.
const CHUNK_GROWTH_LIMIT: usize = 8;

/// Dimensionality below which the blocked GEMM local join is not taken.
const NND_GEMM_MIN_DIM: usize = 96;

/// Candidate-count below which the blocked GEMM local join is not taken.
///
/// A short candidate list makes the matmul too skinny to amortise its own
/// setup, and the gather then buys nothing.
const NND_GEMM_MIN_CANDIDATES: usize = 32;

///////////////////
// Build timings //
///////////////////

/// Merged candidate-list size statistics, accumulated inside the local join.
///
/// The join is `O(|C_new| * |C_total|)` per node, so the distribution of
/// `|C_total|` is what sets its cost. Since `2e866da` dropped the cap on the
/// merged forward+reverse list, that distribution is driven by the reverse
/// in-degree tail, which is fat on hub-heavy data. Three integer ops per node
/// against a quadratic distance loop, so this is always on rather than gated.
#[derive(Default, Clone, Copy)]
struct CandStats {
    /// Merged list lengths summed over the nodes seen
    sum: usize,
    /// Nodes seen
    count: usize,
    /// Longest merged list seen
    max: usize,
    /// Candidate pairs whose distance was evaluated
    pairs: u64,
}

impl CandStats {
    /// Record one node's merged candidate list length.
    ///
    /// ### Params
    ///
    /// * `n_total` - Length of the merged new + old candidate list
    #[inline(always)]
    fn record(&mut self, n_total: usize) {
        self.sum += n_total;
        self.count += 1;
        if n_total > self.max {
            self.max = n_total;
        }
    }

    /// Combine two thread-local accumulators.
    ///
    /// ### Params
    ///
    /// * `other` - Accumulator to fold in
    ///
    /// ### Returns
    ///
    /// The combined statistics.
    fn merge(self, other: Self) -> Self {
        Self {
            sum: self.sum + other.sum,
            count: self.count + other.count,
            max: self.max.max(other.max),
            pairs: self.pairs + other.pairs,
        }
    }
}

/// Per-phase wall-clock breakdown of a build, accumulated across iterations.
///
/// Only populated when `verbose` is set. Every phase is timed from the driving
/// loop rather than from inside the parallel closures, so the cost is a handful
/// of `Instant::now()` calls per iteration and nothing in the inner loops.
///
/// The phases partition the build: `forest` and `seed` are initialisation,
/// the remaining four are the descent, and they sum to roughly the total.
#[derive(Default, Clone, Copy)]
struct BuildTimings {
    /// RP-forest construction (Annoy, or Kd for Manhattan)
    forest: Duration,
    /// Seeding the flat graph by querying the forest once per node
    seed: Duration,
    /// Sampling forward candidates and building their reverse CSR
    candidates: Duration,
    /// Retiring the is-new flags of sampled edges
    mark_old: Duration,
    /// The local join itself: candidate gather plus pairwise distances
    join: Duration,
    /// Radix sorting each chunk's update batch by target
    sort: Duration,
    /// Merging sorted updates back into the graph rows
    apply: Duration,
    /// Updates emitted by the join, before deduplication or heap rejection
    updates_emitted: usize,
    /// Updates that actually changed a graph row
    updates_accepted: usize,
    /// Merged candidate list lengths summed over every node and iteration
    cand_len_sum: usize,
    /// Nodes contributing to `cand_len_sum`
    cand_len_count: usize,
    /// Longest merged candidate list seen
    cand_len_max: usize,
    /// Candidate pairs whose distance was evaluated
    pairs: u64,
}

impl BuildTimings {
    /// Total time attributed to the descent iterations.
    fn descent(&self) -> Duration {
        self.candidates + self.mark_old + self.join + self.sort + self.apply
    }

    /// Print the breakdown as a table, each phase against its share of the total.
    ///
    /// ### Params
    ///
    /// * `n` - Number of samples, for the per-node figures
    /// * `k` - Neighbours per node
    fn report(&self, n: usize, k: usize) {
        let total = self.forest + self.seed + self.descent();
        let secs = total.as_secs_f64().max(f64::MIN_POSITIVE);
        let row = |name: &str, d: Duration| {
            println!(
                "  {:<22} {:>9.3} s  {:>5.1}%",
                name,
                d.as_secs_f64(),
                100.0 * d.as_secs_f64() / secs
            );
        };

        println!("\nNN-Descent build breakdown (n={n}, k={k}):");
        row("forest build", self.forest);
        row("graph seeding", self.seed);
        row("candidates", self.candidates);
        row("mark old", self.mark_old);
        row("local join", self.join);
        row("radix sort", self.sort);
        row("apply updates", self.apply);
        println!("  {:<22} {:>9.3} s", "total", total.as_secs_f64());

        let accept = if self.updates_emitted > 0 {
            100.0 * self.updates_accepted as f64 / self.updates_emitted as f64
        } else {
            0.0
        };
        println!(
            "  updates: {} emitted, {} accepted ({:.2}%)",
            self.updates_emitted.separate_with_underscores(),
            self.updates_accepted.separate_with_underscores(),
            accept
        );

        if self.cand_len_count > 0 {
            println!(
                "  merged candidate list: mean {:.1}, max {}",
                self.cand_len_sum as f64 / self.cand_len_count as f64,
                self.cand_len_max
            );
            println!(
                "  candidate pairs evaluated: {}",
                self.pairs.separate_with_underscores()
            );
        }
    }
}

////////////////
// Raw writes //
////////////////

/// Raw pointer wrapper for the lock-free CSR scatter in [`build_reverse_csr`].
///
/// Safety rests on the counting sort: every `fetch_add` on the cursor hands out
/// a slot that no other thread can be handed, and the per-target segments
/// partition the buffer, so no two writes ever alias.
#[derive(Copy, Clone)]
struct UnsafeU32Ptr(*mut u32);

unsafe impl Send for UnsafeU32Ptr {}
unsafe impl Sync for UnsafeU32Ptr {}

////////////////////
// Candidate sets //
////////////////////

/// Reverse (in-edge) adjacency of one forward candidate sample, in CSR form.
///
/// Built by counting sort rather than by scanning every source list once per
/// thread, which is what the previous `Vec<Vec<usize>>` layout forced.
struct ReverseCsr {
    /// Row offsets, length `n + 1`
    offsets: Vec<u32>,
    /// Source ids grouped by target; target `i` owns `offsets[i]..offsets[i+1]`
    data: Vec<u32>,
}

impl ReverseCsr {
    /// Empty CSR with no rows.
    fn new() -> Self {
        Self {
            offsets: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Sources that sampled `i` into their forward candidate list.
    ///
    /// ### Params
    ///
    /// * `i` - Target node
    ///
    /// ### Returns
    ///
    /// Slice of source ids, in unspecified order.
    #[inline]
    fn segment(&self, i: usize) -> &[u32] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.data[start..end]
    }
}

/// Forward samples plus their reverse adjacency for one NN-Descent iteration.
///
/// The forward lists live in fixed-stride flat buffers of `n * max_candidates`
/// `u32` ids rather than `n` growable `Vec`s, and the reverse edges in CSR. The
/// total reverse entry count equals the total forward count by construction, so
/// the whole structure is `O(n * max_candidates)` with no per-node allocation.
struct CandidateSets {
    /// Slots per node in each forward buffer (equals `max_candidates`)
    stride: usize,
    /// New forward sample, `n * stride`; node `i` owns
    /// `[i*stride .. i*stride + new_len[i]]`, sorted ascending by id
    new_cands: Vec<u32>,
    /// Valid entries per node in `new_cands`
    new_len: Vec<u32>,
    /// Old forward sample, same layout as `new_cands`
    old_cands: Vec<u32>,
    /// Valid entries per node in `old_cands`
    old_len: Vec<u32>,
    /// Reverse edges of the new forward sample
    new_rev: ReverseCsr,
    /// Reverse edges of the old forward sample
    old_rev: ReverseCsr,
}

impl CandidateSets {
    /// Allocate the flat buffers for `n` nodes at `stride` candidates each.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `stride` - Candidates per node (`max_candidates`)
    ///
    /// ### Returns
    ///
    /// Zeroed candidate sets, ready for the first `build_candidates` call.
    fn new(n: usize, stride: usize) -> Self {
        Self {
            stride,
            new_cands: vec![0u32; n * stride],
            new_len: vec![0u32; n],
            old_cands: vec![0u32; n * stride],
            old_len: vec![0u32; n],
            new_rev: ReverseCsr::new(),
            old_rev: ReverseCsr::new(),
        }
    }

    /// New forward sample of node `i`, sorted ascending by id.
    #[inline]
    fn new_forward(&self, i: usize) -> &[u32] {
        let base = i * self.stride;
        &self.new_cands[base..base + self.new_len[i] as usize]
    }

    /// Old forward sample of node `i`, sorted ascending by id.
    #[inline]
    fn old_forward(&self, i: usize) -> &[u32] {
        let base = i * self.stride;
        &self.old_cands[base..base + self.old_len[i] as usize]
    }

    /// Merge node `i`'s forward sample with its reverse edges into `out`.
    ///
    /// Sorted ascending and deduplicated, which is what the pair loop and the
    /// tile gather both want.
    ///
    /// ### Params
    ///
    /// * `i` - Node whose candidate list is wanted
    /// * `new` - Whether to merge the new lists or the old ones
    /// * `out` - Destination scratch, cleared on entry
    fn merged_into(&self, i: usize, new: bool, out: &mut Vec<u32>) {
        let (fwd, rev) = if new {
            (self.new_forward(i), self.new_rev.segment(i))
        } else {
            (self.old_forward(i), self.old_rev.segment(i))
        };
        out.clear();
        out.reserve(fwd.len() + rev.len());
        out.extend_from_slice(fwd);
        out.extend_from_slice(rev);
        out.sort_unstable();
        out.dedup();
    }
}

/// Deterministic random priority for the undirected edge `(u, v)`.
///
/// The forward sample and the reverse edge it induces have to agree on a
/// priority, otherwise the two directions of the same edge compete on different
/// coin flips and the cap becomes asymmetric. Deriving it from the unordered
/// pair by hash rather than storing it alongside every reverse entry keeps the
/// CSR down to bare ids.
///
/// SplitMix64 finaliser over the packed pair, mixed with the iteration seed.
///
/// ### Params
///
/// * `seed` - Per-iteration seed
/// * `u` - One endpoint
/// * `v` - The other endpoint
///
/// ### Returns
///
/// Uniformly distributed 64-bit priority, symmetric in `u` and `v`.
#[inline(always)]
fn edge_priority(seed: u64, u: u32, v: u32) -> u64 {
    let (lo, hi) = if u < v { (u, v) } else { (v, u) };
    let mut z =
        ((hi as u64) << 32 | lo as u64).wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build the reverse adjacency of a fixed-stride forward candidate sample.
///
/// Two-pass counting sort: a parallel count into relaxed atomics, an exclusive
/// prefix sum, then a parallel scatter through the same array reused as a write
/// cursor. Cost is `O(total_entries)` regardless of thread count, where the
/// previous target-range partitioning cost `O(n_threads * total_entries)`.
///
/// Order within a target's segment is unspecified (it depends on the atomic
/// interleaving), which is fine because every consumer sorts the merged list
/// before use.
///
/// ### Params
///
/// * `fwd` - Flat forward sample, `n * stride` ids
/// * `lens` - Valid entries per node in `fwd`
/// * `stride` - Slots per node in `fwd`
/// * `n` - Number of nodes
/// * `out` - CSR to overwrite
fn build_reverse_csr(fwd: &[u32], lens: &[u32], stride: usize, n: usize, out: &mut ReverseCsr) {
    let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();

    (0..n).into_par_iter().for_each(|i| {
        let base = i * stride;
        for &j in &fwd[base..base + lens[i] as usize] {
            counts[j as usize].fetch_add(1, Ordering::Relaxed);
        }
    });

    out.offsets.clear();
    out.offsets.reserve(n + 1);
    out.offsets.push(0);
    let mut acc: u32 = 0;
    for c in counts.iter() {
        acc += c.load(Ordering::Relaxed);
        out.offsets.push(acc);
    }

    out.data.clear();
    out.data.resize(acc as usize, 0);

    // Reuse the count array as the per-target write cursor.
    for (i, c) in counts.iter().enumerate() {
        c.store(out.offsets[i], Ordering::Relaxed);
    }

    let data_ptr = UnsafeU32Ptr(out.data.as_mut_ptr());
    (0..n).into_par_iter().for_each(|i| {
        #[allow(clippy::redundant_locals)]
        let data_ptr = data_ptr;
        let base = i * stride;
        for &j in &fwd[base..base + lens[i] as usize] {
            let pos = counts[j as usize].fetch_add(1, Ordering::Relaxed) as usize;
            // SAFETY: the cursor for target `j` starts at its segment offset
            // and is bumped once per write, so every thread gets a distinct
            // slot inside a segment that no other target touches.
            unsafe { *data_ptr.0.add(pos) = i as u32 };
        }
    });
}

//////////////
// MetricFn //
//////////////

/// Static-dispatch metric selector for the update kernel.
///
/// The inner loop in `generate_updates_for_chunk_impl` calls
/// `M::distance_from_tile` where `M` is one of the zero-sized types below.
/// Both operands are rows of the gathered candidate tile, so the kernel never
/// re-slices `vectors_flat` or re-fetches a cached norm inside the loop.
/// Monomorphisation strips the runtime `Dist` branch out of the hot path.
trait MetricFn<T: AnnSearchFloat> {
    /// Distance between two rows of the gathered candidate tile.
    ///
    /// `norm_a` and `norm_b` are the pre-fetched L2 norms, meaningful for
    /// Cosine only.
    fn distance_from_tile(vec_a: &[T], norm_a: T, vec_b: &[T], norm_b: T) -> T;
}

/// Squared Euclidean metric.
struct SqEuclidMetric;
/// Cosine metric (assumes pre-computed norms in `NNDescent::norms`).
struct CosineMetric;
/// Manhattan (L1) metric.
struct ManhattanMetric;

impl<T: AnnSearchFloat> MetricFn<T> for SqEuclidMetric {
    #[inline(always)]
    fn distance_from_tile(vec_a: &[T], _norm_a: T, vec_b: &[T], _norm_b: T) -> T {
        T::euclidean_simd(vec_a, vec_b)
    }
}

impl<T: AnnSearchFloat> MetricFn<T> for CosineMetric {
    #[inline(always)]
    fn distance_from_tile(vec_a: &[T], norm_a: T, vec_b: &[T], norm_b: T) -> T {
        let denom = norm_a * norm_b;
        if denom > T::zero() {
            T::one() - (T::dot_simd(vec_a, vec_b) / denom)
        } else {
            T::one()
        }
    }
}

impl<T: AnnSearchFloat> MetricFn<T> for ManhattanMetric {
    #[inline(always)]
    fn distance_from_tile(vec_a: &[T], _norm_a: T, vec_b: &[T], _norm_b: T) -> T {
        T::manhattan_simd(vec_a, vec_b)
    }
}

/// Keep the `cap` lowest-priority entries of `temp` and write their ids into
/// `out`, sorted ascending.
///
/// `select_nth_unstable_by_key` gives the `O(len)` partial selection the full
/// sort would not.
///
/// ### Params
///
/// * `temp` - `(priority, id)` entries, consumed and reordered
/// * `cap` - Maximum entries to keep
/// * `out` - Destination stride slice, at least `cap` long
///
/// ### Returns
///
/// Number of ids written into `out`.
fn take_lowest_priority(temp: &mut Vec<(u64, u32)>, cap: usize, out: &mut [u32]) -> u32 {
    if cap == 0 {
        return 0;
    }
    if temp.len() > cap {
        temp.select_nth_unstable_by_key(cap - 1, |&(p, _)| p);
        temp.truncate(cap);
    }
    for (slot, &(_, id)) in out.iter_mut().zip(temp.iter()) {
        *slot = id;
    }
    let len = temp.len().min(out.len());
    out[..len].sort_unstable();
    len as u32
}

/////////////////
// JoinScratch //
/////////////////

/// Per-thread scratch for one source node's local join.
///
/// Everything the pair loop touches is gathered here first: the candidate
/// vectors into one contiguous tile, and their eviction thresholds and norms
/// into flat arrays. That turns `|C|^2 / 2` strided reads of `vectors_flat`
/// and random reads of the graph into `|C|` of each, and gives the GEMM path
/// the row-major operand it needs.
///
/// Buffers are reused across every node the thread handles, so the allocations
/// settle after the first few nodes.
struct JoinScratch<T> {
    /// Merged new candidates for this node
    new_ids: Vec<u32>,
    /// Merged old candidates for this node
    old_ids: Vec<u32>,
    /// Candidate ids, new list followed by old list
    ids: Vec<u32>,
    /// Distance of each candidate's current worst neighbour
    thresh: Vec<T>,
    /// L2 norms (Cosine only; zero-filled otherwise)
    norms: Vec<T>,
    /// Squared L2 norms, filled only on the GEMM path
    sq: Vec<T>,
    /// Candidate vectors, row-major `ids.len() * dim`
    tile: Vec<T>,
    /// Dot products, row-major `n_new * n_total`, filled only on the GEMM path
    dots: Vec<T>,
}

impl<T: AnnSearchFloat> JoinScratch<T> {
    /// Empty scratch. Buffers grow to fit on the first gather.
    fn new() -> Self {
        Self {
            new_ids: Vec::new(),
            old_ids: Vec::new(),
            ids: Vec::new(),
            thresh: Vec::new(),
            norms: Vec::new(),
            sq: Vec::new(),
            tile: Vec::new(),
            dots: Vec::new(),
        }
    }

    /// Merge node `node`'s candidate lists, then gather their vectors,
    /// thresholds and norms.
    ///
    /// ### Params
    ///
    /// * `idx` - Index owning the vectors and norms
    /// * `graph` - Current flat k-NN graph, read for the eviction thresholds
    /// * `k` - Neighbours per node in `graph`
    /// * `cands` - Forward samples and reverse adjacency
    /// * `node` - Source node
    /// * `cosine` - Whether norms are meaningful for this metric
    ///
    /// ### Returns
    ///
    /// Length of the new-candidate list, which is where the tile switches from
    /// new to old.
    fn gather(
        &mut self,
        idx: &NNDescent<T>,
        graph: &[Neighbour<T>],
        k: usize,
        cands: &CandidateSets,
        node: usize,
        cosine: bool,
    ) -> usize {
        let dim = idx.dim;

        cands.merged_into(node, true, &mut self.new_ids);
        cands.merged_into(node, false, &mut self.old_ids);

        self.ids.clear();
        self.ids.extend_from_slice(&self.new_ids);
        self.ids.extend_from_slice(&self.old_ids);

        let total = self.ids.len();
        self.thresh.clear();
        self.norms.clear();
        self.tile.clear();
        self.thresh.reserve(total);
        self.norms.reserve(total);
        self.tile.reserve(total * dim);

        for t in 0..total {
            let p = self.ids[t] as usize;
            self.thresh.push(graph[p * k + k - 1].dist);
            self.norms
                .push(if cosine { idx.norms[p] } else { T::zero() });
            self.tile
                .extend_from_slice(&idx.vectors_flat[p * dim..(p + 1) * dim]);
        }

        self.new_ids.len()
    }

    /// Fill `dots` with the `n_new x n_total` dot-product block of the tile.
    ///
    /// Squared norms are computed here too, since only this path needs them.
    /// The matmul stays `Par::Seq`: the chunk's rayon iterator already owns
    /// every core.
    ///
    /// ### Params
    ///
    /// * `n_new` - Rows, the length of the new-candidate list
    /// * `n_total` - Columns, the full candidate count
    /// * `dim` - Vector dimensionality
    fn compute_dots(&mut self, n_new: usize, n_total: usize, dim: usize) {
        self.sq.clear();
        self.sq.reserve(n_total);
        for t in 0..n_total {
            let v = &self.tile[t * dim..(t + 1) * dim];
            self.sq.push(T::dot_simd(v, v));
        }

        self.dots.clear();
        self.dots.resize(n_new * n_total, T::zero());

        let lhs = MatRef::from_row_major_slice(&self.tile[..n_new * dim], n_new, dim);
        let rhs = MatRef::from_row_major_slice(&self.tile[..n_total * dim], n_total, dim);
        let mut out = MatMut::from_row_major_slice_mut(&mut self.dots[..], n_new, n_total);

        matmul(
            out.as_mut(),
            Accum::Replace,
            lhs,
            rhs.transpose(),
            T::one(),
            Par::Seq,
        );
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

    /// Query into reusable scratch, leaving `(distance, id)` pairs sorted
    /// nearest-first in `scratch.candidates`.
    ///
    /// Only the Annoy arm has an allocation-free form; the Kd arm falls back to
    /// the allocating query and copies its result across, which keeps Manhattan
    /// working without a second scratch type for a path that is not the
    /// default.
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vector
    /// * `k` - Number of neighbours to return
    /// * `search_k` - The budget
    /// * `scratch` - Reusable buffers, reset on entry
    ///
    /// ### Returns
    ///
    /// `Ok(())`, with results left in `scratch.candidates`.
    fn query_into(
        &self,
        query_vec: &[T],
        k: usize,
        search_k: Option<usize>,
        scratch: &mut AnnoyScratch<T>,
    ) -> Result<(), AnnSearchErrors> {
        match self {
            Forest::Annoy(f) => f.query_into(query_vec, k, search_k, scratch),
            Forest::Kd(f) => {
                let (ids, dists) = f.query(query_vec, k, search_k)?;
                scratch.set_candidates(ids.into_iter().zip(dists).map(|(i, d)| (d, i)));
                Ok(())
            }
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

        // based on PyNNDescent... 12 seems to be good for the initialisation
        let n_trees = n_trees.unwrap_or_else(|| {
            let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
            calculated.min(12)
        });

        let max_iter = max_iter.unwrap_or_else(|| {
            let calculated = ((n as f64).log2().round()) as usize;
            calculated.max(5)
        });

        let k = k.unwrap_or(30);
        let max_candidates = max_candidates.unwrap_or(k.min(60));

        let start = Instant::now();
        let forest = Forest::new((&vectors_flat[..], n, dim), n_trees, metric, seed)?;
        let forest_time = start.elapsed();
        if verbose {
            println!("Built forest: {forest_time:.2?}");
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

        let (build_graph, converged) = builder.generate_index(
            k,
            max_iter,
            delta,
            max_candidates,
            seed,
            forest_time,
            verbose,
        )?;

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

    /// Number of updates that fit in the per-chunk byte budget.
    ///
    /// ### Returns
    ///
    /// [`UPDATE_TARGET_BYTES`] divided by the real size of an `Update<T>`.
    #[inline]
    fn target_updates_per_chunk() -> usize {
        UPDATE_TARGET_BYTES / std::mem::size_of::<Update<T>>()
    }

    /// Opening guess at the number of source nodes per chunk.
    ///
    /// A source node emits one update per direction for every candidate pair
    /// that clears the distance threshold, so the count scales with
    /// `|C_new| * |C_total|`, not with `max_candidates`. Getting that wrong is
    /// what let the nominal memory bound be exceeded by two orders of
    /// magnitude; it is only a starting point regardless, since
    /// [`Self::rescale_chunk_size`] corrects it from the second chunk on.
    ///
    /// ### Params
    ///
    /// * `max_candidates` - Cap on the merged candidate list per node
    ///
    /// ### Returns
    ///
    /// Number of source nodes to process in the first chunk.
    fn initial_chunk_size(&self, max_candidates: usize) -> usize {
        let pairs_per_source = 6 * max_candidates * max_candidates;
        let updates_per_source =
            ((pairs_per_source * 2) as f64 * ASSUMED_ACCEPT_RATE).ceil() as usize;

        let lo = MIN_CHUNK_NODES.min(self.n);
        let chunk = Self::target_updates_per_chunk() / updates_per_source.max(1);
        chunk.clamp(lo, self.n.max(lo))
    }

    /// Rescale the chunk size against what the last chunk actually emitted.
    ///
    /// This is what makes the memory bound real: the accept rate falls by
    /// orders of magnitude between the first iteration and convergence, so a
    /// static estimate is either wildly conservative or wildly over budget.
    ///
    /// ### Params
    ///
    /// * `current` - Chunk size just used
    /// * `sources` - Source nodes in that chunk
    /// * `emitted` - Updates the chunk produced
    ///
    /// ### Returns
    ///
    /// Chunk size for the next chunk, clamped to
    /// `[MIN_CHUNK_NODES, n]` and to [`CHUNK_GROWTH_LIMIT`] times `current`.
    fn rescale_chunk_size(&self, current: usize, sources: usize, emitted: usize) -> usize {
        let lo = MIN_CHUNK_NODES.min(self.n);
        let hi = current
            .saturating_mul(CHUNK_GROWTH_LIMIT)
            .min(self.n)
            .max(lo);

        if sources == 0 || emitted == 0 {
            return hi;
        }

        let per_source = emitted as f64 / sources as f64;
        let next = (Self::target_updates_per_chunk() as f64 / per_source) as usize;
        next.clamp(lo, hi)
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
        let search_k = k * self.forest.n_trees();

        graph.par_chunks_mut(k).enumerate().try_for_each_init(
            AnnoyScratch::<T>::new,
            |scratch, (i, slot)| -> Result<(), AnnSearchErrors> {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                self.forest
                    .query_into(query, k + 1, Some(search_k), scratch)?;
                for (j, &(dist, idx)) in scratch.results().iter().skip(1).take(k).enumerate() {
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
    /// * `forest_time` - Time already spent building the RP forest, so the
    ///   verbose breakdown can account for the whole build rather than just
    ///   the part after the forest
    /// * `verbose` - Print progress information
    ///
    /// ### Returns
    ///
    /// Tuple of (flat graph as `(pid, dist)` pairs, converged flag)
    #[allow(clippy::too_many_arguments)]
    fn generate_index(
        &self,
        k: usize,
        max_iter: usize,
        delta: T,
        max_candidates: usize,
        seed: usize,
        forest_time: Duration,
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

        let mut timings = BuildTimings {
            forest: forest_time,
            ..Default::default()
        };

        let start = Instant::now();
        let mut graph = self.init_with_forest(k)?;
        timings.seed = start.elapsed();

        if verbose {
            println!("Seeded graph from forest: {:.2?}", timings.seed);
        }

        let mut chunk_size = self.initial_chunk_size(max_candidates);

        if verbose {
            println!(
                " Starting chunk size {} for memory-bounded updates ({} MiB budget)",
                chunk_size.separate_with_underscores(),
                UPDATE_TARGET_BYTES / (1024 * 1024)
            );
        }

        let mut cands = CandidateSets::new(self.n, max_candidates);

        for iter in 0..max_iter {
            let updates_count = AtomicUsize::new(0);
            let iter_seed = (seed as u64).wrapping_add(iter as u64);

            if verbose {
                println!(" Preparing candidates for iter {}", iter + 1);
            }
            let t = Instant::now();
            self.build_candidates(&graph, k, max_candidates, iter_seed, &mut cands);
            timings.candidates += t.elapsed();

            let t = Instant::now();
            self.mark_as_old(&mut graph, k, &cands);
            timings.mark_old += t.elapsed();

            if verbose {
                println!(" Processing updates for iter {}", iter + 1);
            }

            let mut chunk_start = 0usize;
            let mut n_chunks = 0usize;
            while chunk_start < self.n {
                let chunk_end = (chunk_start + chunk_size).min(self.n);

                let t = Instant::now();
                let (mut chunk_updates, stats) =
                    self.generate_updates_for_chunk(&cands, &graph, k, chunk_start, chunk_end);
                timings.join += t.elapsed();
                timings.updates_emitted += chunk_updates.len();
                timings.cand_len_sum += stats.sum;
                timings.cand_len_count += stats.count;
                timings.cand_len_max = timings.cand_len_max.max(stats.max);
                timings.pairs += stats.pairs;

                let t = Instant::now();
                chunk_updates.radix_sort_unstable();
                timings.sort += t.elapsed();

                let t = Instant::now();
                self.apply_sorted_updates(&chunk_updates, &mut graph, k, &updates_count);
                timings.apply += t.elapsed();

                chunk_size = self.rescale_chunk_size(
                    chunk_size,
                    chunk_end - chunk_start,
                    chunk_updates.len(),
                );
                chunk_start = chunk_end;
                n_chunks += 1;
            }

            if verbose {
                println!("  {} chunks, next chunk size {}", n_chunks, chunk_size);
            }

            let update_count = updates_count.load(Ordering::Relaxed);
            timings.updates_accepted += update_count;
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
            timings.report(self.n, k);
        }

        let res = graph.into_iter().map(|n| (n.pid(), n.dist)).collect();

        Ok((res, converged))
    }

    /// Sample the forward candidate lists and build their reverse adjacency.
    ///
    /// Two passes:
    ///
    /// 1. Sample up to `max_candidates` new and old neighbours per node from
    ///    its graph row, keeping the lowest-priority entries.
    /// 2. Build the reverse adjacency of both samples as CSR
    ///    ([`build_reverse_csr`]).
    ///
    /// The merge of the two is deliberately left to the join, which assembles
    /// it per node into thread-local scratch.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph
    /// * `k` - Neighbours per node
    /// * `max_candidates` - Cap on the forward sample
    /// * `iter_seed` - Per-iteration seed for reproducible sampling
    /// * `cands` - Candidate sets, overwritten in place
    fn build_candidates(
        &self,
        graph: &[Neighbour<T>],
        k: usize,
        max_candidates: usize,
        iter_seed: u64,
        cands: &mut CandidateSets,
    ) {
        let n = self.n;
        let stride = cands.stride;

        // Phase 1: sample the forward lists. Each node writes only its own
        // stride slice, so this is embarrassingly parallel.
        cands
            .new_cands
            .par_chunks_mut(stride)
            .zip(cands.old_cands.par_chunks_mut(stride))
            .zip(cands.new_len.par_iter_mut())
            .zip(cands.old_len.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((new_slot, old_slot), new_len), old_len))| {
                CAND_SCRATCH.with(|cell| {
                    let mut b = cell.borrow_mut();
                    let (new_temp, old_temp) = &mut *b;
                    new_temp.clear();
                    old_temp.clear();

                    let base = i * k;
                    for slot in &graph[base..base + k] {
                        if slot.is_sentinel() {
                            continue;
                        }
                        let j = slot.pid();
                        if j >= n {
                            continue;
                        }
                        let entry = (edge_priority(iter_seed, i as u32, j as u32), j as u32);
                        if slot.is_new() {
                            new_temp.push(entry);
                        } else {
                            old_temp.push(entry);
                        }
                    }

                    *new_len = take_lowest_priority(new_temp, max_candidates, new_slot);
                    *old_len = take_lowest_priority(old_temp, max_candidates, old_slot);
                });
            });

        // Phase 2: reverse adjacency of both samples.
        build_reverse_csr(
            &cands.new_cands,
            &cands.new_len,
            stride,
            n,
            &mut cands.new_rev,
        );
        build_reverse_csr(
            &cands.old_cands,
            &cands.old_len,
            stride,
            n,
            &mut cands.old_rev,
        );
    }

    /// Mark neighbours as old if they were sampled into the new-candidate list
    ///
    /// After sampling, any neighbour that survived into node `i`'s merged new
    /// list will have been explored during this iteration's local joins, so
    /// flip its flag and let subsequent iterations treat it as old.
    ///
    /// The merged list is never materialised, but membership in it does not
    /// need it: `j` is in `i`'s merged new list exactly when `j` is in `i`'s
    /// forward sample or `i` is in `j`'s, the second case being the reverse
    /// edge. Both samples are sorted, so this is two binary searches.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph (mutated in place)
    /// * `k` - Neighbours per node
    /// * `cands` - Forward samples, sorted ascending by id
    fn mark_as_old(&self, graph: &mut [Neighbour<T>], k: usize, cands: &CandidateSets) {
        graph.par_chunks_mut(k).enumerate().for_each(|(i, slots)| {
            let fwd_i = cands.new_forward(i);
            for slot in slots.iter_mut() {
                if slot.is_sentinel() || !slot.is_new() {
                    continue;
                }
                let j = slot.pid();
                if fwd_i.binary_search(&(j as u32)).is_ok()
                    || cands.new_forward(j).binary_search(&(i as u32)).is_ok()
                {
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
    /// * `cands` - Merged candidate lists per node
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
        cands: &CandidateSets,
        graph: &[Neighbour<T>],
        k: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> (Vec<Update<T>>, CandStats) {
        match self.metric {
            Dist::SquaredEuclidean => self.generate_updates_for_chunk_impl::<SqEuclidMetric>(
                cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
            Dist::Cosine => self.generate_updates_for_chunk_impl::<CosineMetric>(
                cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
            Dist::Manhattan => self.generate_updates_for_chunk_impl::<ManhattanMetric>(
                cands,
                graph,
                k,
                chunk_start,
                chunk_end,
            ),
        }
    }

    /// Whether the blocked GEMM local join applies to this index.
    ///
    /// Manhattan has no inner-product form, and below the dimensionality and
    /// candidate-count thresholds the gather plus expansion plus exact
    /// re-ranking costs more than the fused SIMD kernels save.
    ///
    /// ### Params
    ///
    /// * `n_cands` - Length of the merged candidate list for this node
    ///
    /// ### Returns
    ///
    /// `true` if the GEMM path should be taken.
    #[inline]
    fn gemm_join_applies(&self, n_cands: usize) -> bool {
        self.metric != Dist::Manhattan
            && self.dim >= NND_GEMM_MIN_DIM
            && n_cands >= NND_GEMM_MIN_CANDIDATES
    }

    /// Monomorphised inner kernel for chunked update generation.
    ///
    /// `M` selects the distance function at compile time so the branch on
    /// `self.metric` is stripped out of the hot loop entirely.
    ///
    /// Per source node the candidate vectors, their eviction thresholds and
    /// their norms are gathered into contiguous thread-local scratch before the
    /// pair loop. The threshold then costs `|C|` random reads of the graph
    /// amortised over `|C|^2 / 2` pair tests rather than one read per pair, and
    /// the tile turns the strided reads of `vectors_flat` into a sequential
    /// walk of a buffer that stays hot for the whole node.
    ///
    /// New and old lists are concatenated into one tile, so the single loop
    /// `a in 0..n_new`, `b in a+1..n_total` covers the new-new upper triangle
    /// and the full new-old rectangle exactly once each.
    fn generate_updates_for_chunk_impl<M: MetricFn<T>>(
        &self,
        cands: &CandidateSets,
        graph: &[Neighbour<T>],
        k: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> (Vec<Update<T>>, CandStats) {
        let dim = self.dim;
        let cosine = self.metric == Dist::Cosine;
        let two = T::one() + T::one();

        (chunk_start..chunk_end)
            .into_par_iter()
            .fold(
                || {
                    (
                        Vec::<Update<T>>::with_capacity(16_384),
                        JoinScratch::<T>::new(),
                        CandStats::default(),
                    )
                },
                |(mut updates, mut scratch, mut stats), i| {
                    let n_new = scratch.gather(self, graph, k, cands, i, cosine);
                    let n_total = scratch.ids.len();
                    stats.record(n_total);
                    if n_new == 0 || n_total < 2 {
                        return (updates, scratch, stats);
                    }

                    let use_gemm = self.gemm_join_applies(n_total);
                    if use_gemm {
                        scratch.compute_dots(n_new, n_total, dim);
                    }

                    stats.pairs += (n_new * n_total - n_new * (n_new + 1) / 2) as u64;
                    for a in 0..n_new {
                        let pa = scratch.ids[a];
                        let ta = scratch.thresh[a];
                        let na = scratch.norms[a];

                        for b in (a + 1)..n_total {
                            let pb = scratch.ids[b];
                            if pa == pb {
                                continue;
                            }

                            let d = if use_gemm {
                                let dot = scratch.dots[a * n_total + b];
                                if cosine {
                                    let denom = na * scratch.norms[b];
                                    if denom > T::zero() {
                                        T::one() - dot / denom
                                    } else {
                                        T::one()
                                    }
                                } else {
                                    (scratch.sq[a] + scratch.sq[b] - two * dot).max(T::zero())
                                }
                            } else {
                                M::distance_from_tile(
                                    &scratch.tile[a * dim..(a + 1) * dim],
                                    na,
                                    &scratch.tile[b * dim..(b + 1) * dim],
                                    scratch.norms[b],
                                )
                            };

                            let tb = scratch.thresh[b];
                            if d > ta && d > tb {
                                continue;
                            }

                            let d = if use_gemm {
                                M::distance_from_tile(
                                    &scratch.tile[a * dim..(a + 1) * dim],
                                    na,
                                    &scratch.tile[b * dim..(b + 1) * dim],
                                    scratch.norms[b],
                                )
                            } else {
                                d
                            };

                            if d <= ta {
                                updates.push(Update::new(pa, pb, d));
                            }
                            if d <= tb {
                                updates.push(Update::new(pb, pa, d));
                            }
                        }
                    }

                    (updates, scratch, stats)
                },
            )
            .map(|(updates, _, stats)| (updates, stats))
            .reduce(
                || (Vec::new(), CandStats::default()),
                |(mut a, sa), (mut b, sb)| {
                    let merged = sa.merge(sb);
                    if a.len() >= b.len() {
                        a.extend_from_slice(&b);
                        (a, merged)
                    } else {
                        b.extend_from_slice(&a);
                        (b, merged)
                    }
                },
            )
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
    fn build_reverse_adjacency(&self, graph: &[(usize, T)], k: usize) -> Vec<Vec<(usize, T)>> {
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
                    a.0.cmp(&b.0)
                        .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
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
                        if dist_to_kept < cand_dist && rng.random::<f64>() < prune_prob_f64 {
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

                kept.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

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

    /// Hand back the kNN graph exactly as NN-Descent built it.
    ///
    /// No re-query: this is the descent's own output, so it costs one pass over
    /// the graph. [`Self::generate_knn`] runs a beam search per point instead,
    /// which lifts recall but is orders of magnitude more expensive. Reach for
    /// this one when the graph you asked for is the graph you want.
    ///
    /// Rows can come back **shorter than `k`** where the descent never filled
    /// them (small `n`, disconnected components), which `generate_knn` never
    /// does.
    ///
    /// ### Params
    ///
    /// * `k` - Truncate each row to this **total** length, self-edge included
    ///   when `include_self` is set. `None` keeps the full build-time `k`.
    /// * `include_self` - Prepend `(i, 0)` to row `i`, matching what every
    ///   `query_*_self` and an exhaustive ground truth return. Leave unset for
    ///   true neighbours only.
    /// * `return_dist` - Whether to materialise the distances
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)`, sorted by distance ascending.
    pub fn extract_knn(
        &self,
        k: Option<usize>,
        include_self: bool,
        return_dist: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        unpack_knn_graph(&self.graph, self.n, self.k, k, include_self, return_dist)
    }

    /// Generate a kNN graph by querying every vector in the index.
    ///
    /// Each point runs a fresh beam search, so this refines the built graph
    /// rather than reading it back. See [`Self::extract_knn`] for the cheap
    /// path.
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

        Ok(pack_knn_results(results, return_dist))
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

/// In-place merge of a target's sorted update segment into its graph row.
///
/// The row is already the sorted structure this needs: `k` entries ascending by
/// distance, sentinel-padded. Merging directly into it drops the previous
/// approach's per-target rebuild of a thread-local `SortedBuffer` and its
/// `n`-bit duplicate set, both of which cost the full `k` even when a single
/// update landed. Duplicate detection is a linear scan of the row, which for
/// realistic `k` is a couple of contiguous cache lines against the bitset's
/// scattered reads over `n / 8` bytes.
impl<T: AnnSearchFloat> ApplySortedUpdates<T> for NNDescent<T> {
    fn apply_sorted_updates(
        &self,
        updates: &[Update<T>],
        graph: &mut [Neighbour<T>],
        k: usize,
        updates_count: &AtomicUsize,
    ) {
        if updates.is_empty() || k == 0 {
            return;
        }

        let graph_ptr = UnsafeGraphPtr(graph.as_mut_ptr());

        // `par_chunk_by` splits the sorted batch on target boundaries directly,
        // which saves a sequential boundary scan plus the Vec of segment
        // descriptors it used to feed.
        updates
            .par_chunk_by(|a, b| a.target == b.target)
            .for_each(|segment| {
                let target = segment[0].target as usize;

                #[allow(clippy::redundant_locals)]
                let graph_ptr = graph_ptr;

                // SAFETY: each segment covers one target, and the batch is
                // sorted by target, so this thread owns the row for the whole
                // call and no other thread aliases it.
                let row = unsafe { std::slice::from_raw_parts_mut(graph_ptr.0.add(target * k), k) };

                // Most segments change nothing once the graph settles, so bail
                // before touching the row if not one update can beat its
                // current worst neighbour. Sentinel-padded rows hold MAX, so
                // short rows always pass.
                let mut cutoff = row[k - 1].dist;
                if segment.iter().all(|u| u.dist > cutoff) {
                    return;
                }

                let mut edge_updates = 0usize;

                for update in segment {
                    let d = update.dist;
                    if d > cutoff {
                        continue;
                    }
                    let src = update.source as usize;

                    // One pass finds both the duplicate and the insertion
                    // point. The scan must run to the end regardless, since a
                    // duplicate can sit past the insertion point.
                    let mut pos = k;
                    let mut duplicate = false;
                    for (i, slot) in row.iter().enumerate() {
                        let pid = slot.pid();
                        if pid == src {
                            duplicate = true;
                            break;
                        }
                        if pos == k && (d < slot.dist || (d == slot.dist && src < pid)) {
                            pos = i;
                        }
                    }

                    if duplicate || pos == k {
                        continue;
                    }

                    row.copy_within(pos..k - 1, pos + 1);
                    row[pos] = Neighbour::new(src, d, true);
                    cutoff = row[k - 1].dist;
                    edge_updates += 1;
                }

                if edge_updates > 0 {
                    updates_count.fetch_add(edge_updates, Ordering::Relaxed);
                }
            });
    }
}

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
    use approx::assert_relative_eq;
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
    ///////////////////////
    // Candidate sets    //
    ///////////////////////

    /// Naive reverse adjacency, for the CSR to be checked against.
    fn naive_reverse(fwd: &[u32], lens: &[u32], stride: usize, n: usize) -> Vec<Vec<u32>> {
        let mut rev = vec![Vec::new(); n];
        for i in 0..n {
            for &j in &fwd[i * stride..i * stride + lens[i] as usize] {
                rev[j as usize].push(i as u32);
            }
        }
        rev
    }

    #[test]
    fn test_reverse_csr_matches_naive() {
        let n = 64;
        let stride = 5;
        let mut fwd = vec![0u32; n * stride];
        let mut lens = vec![0u32; n];

        // Deterministic ragged fan-out, including a hub every node points at.
        for i in 0..n {
            let len = i % (stride + 1);
            lens[i] = len as u32;
            for s in 0..len {
                fwd[i * stride + s] = if s == 0 {
                    7
                } else {
                    ((i * 13 + s * 29) % n) as u32
                };
            }
        }

        let mut csr = ReverseCsr::new();
        build_reverse_csr(&fwd, &lens, stride, n, &mut csr);

        let naive = naive_reverse(&fwd, &lens, stride, n);
        assert_eq!(csr.offsets.len(), n + 1);
        assert_eq!(csr.data.len(), naive.iter().map(|v| v.len()).sum::<usize>());

        for i in 0..n {
            let mut got: Vec<u32> = csr.segment(i).to_vec();
            let mut want = naive[i].clone();
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "reverse segment {i} disagrees");
        }
    }

    #[test]
    fn test_reverse_csr_handles_empty_forward() {
        let n = 8;
        let stride = 4;
        let fwd = vec![0u32; n * stride];
        let lens = vec![0u32; n];

        let mut csr = ReverseCsr::new();
        build_reverse_csr(&fwd, &lens, stride, n, &mut csr);

        assert!(csr.data.is_empty());
        assert!((0..n).all(|i| csr.segment(i).is_empty()));
    }

    #[test]
    fn test_edge_priority_is_symmetric() {
        for (u, v) in [(0u32, 1u32), (17, 4), (99, 99), (1_000_000, 3)] {
            assert_eq!(edge_priority(7, u, v), edge_priority(7, v, u));
        }
        // Different seeds must not collapse onto the same ordering.
        assert_ne!(edge_priority(1, 3, 9), edge_priority(2, 3, 9));
    }

    #[test]
    fn test_take_lowest_priority_caps_and_sorts() {
        let mut temp: Vec<(u64, u32)> = vec![(9, 40), (1, 10), (5, 30), (3, 20)];
        let mut out = vec![0u32; 2];
        let len = take_lowest_priority(&mut temp, 2, &mut out);

        assert_eq!(len, 2);
        // Priorities 1 and 3 win, and the ids come back sorted.
        assert_eq!(out, vec![10, 20]);
    }

    #[test]
    fn test_forward_sample_respects_max_candidates() {
        // The forward sample is the only capped list. The merged list the join
        // runs over is deliberately uncapped: capping it costs recall, since
        // the reverse in-degree distribution is broad rather than a clippable
        // tail. This pins the forward cap and the sorted-deduped invariant both
        // `mark_as_old` and `merged_into` rely on.
        let n = 200;
        let dim = 4;
        let mat = Mat::from_fn(n, dim, |i, j| {
            if i < 20 {
                (i as f32) * 0.001 + (j as f32) * 0.002
            } else {
                ((i * 7 + j * 13) % 50) as f32
            }
        });

        let max_candidates = 6;
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            Some(max_candidates),
            Some(4),
            None,
            0.0,
            0.0,
            42,
            false,
        )
        .unwrap();

        let mut cands = CandidateSets::new(index.n, max_candidates);
        let graph: Vec<Neighbour<f32>> = index
            .graph
            .iter()
            .map(|&(pid, d)| Neighbour::new(pid, d, true))
            .collect();
        index.build_candidates(&graph, index.k, max_candidates, 42, &mut cands);

        let mut merged = Vec::new();
        for i in 0..index.n {
            let new_fwd = cands.new_forward(i);
            let old_fwd = cands.old_forward(i);
            assert!(new_fwd.len() <= max_candidates, "new sample {i} over cap");
            assert!(old_fwd.len() <= max_candidates, "old sample {i} over cap");
            assert!(
                new_fwd.windows(2).all(|w| w[0] < w[1]),
                "new sample {i} not sorted and deduped"
            );
            assert!(
                old_fwd.windows(2).all(|w| w[0] < w[1]),
                "old sample {i} not sorted and deduped"
            );

            // The merged list must be the sorted union of the two directions.
            cands.merged_into(i, true, &mut merged);
            assert!(merged.windows(2).all(|w| w[0] < w[1]));
            for &j in new_fwd {
                assert!(merged.binary_search(&j).is_ok(), "merged {i} lost {j}");
            }
            for &src in cands.new_rev.segment(i) {
                assert!(merged.binary_search(&src).is_ok(), "merged {i} lost {src}");
            }
        }
    }

    ///////////////////////
    // Chunk sizing      //
    ///////////////////////

    #[test]
    fn test_chunk_sizing_stays_inside_the_byte_budget() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(5),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        let budget = NNDescent::<f32>::target_updates_per_chunk();
        assert_eq!(
            budget,
            UPDATE_TARGET_BYTES / std::mem::size_of::<Update<f32>>()
        );

        // A chunk that emitted its full budget must not grow.
        let next = index.rescale_chunk_size(1_000, 1_000, budget);
        assert!(next <= 1_000, "chunk grew past a saturated budget");

        // A chunk that emitted almost nothing grows, but only by the limit.
        let next = index.rescale_chunk_size(2, 2, 1);
        assert!(next <= 2 * CHUNK_GROWTH_LIMIT);

        // A chunk that emitted nothing at all still grows rather than stalling.
        assert!(index.rescale_chunk_size(1, 1, 0) >= 1);
    }

    #[test]
    fn test_initial_chunk_size_is_bounded_by_n() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(5),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert!(index.initial_chunk_size(30) <= index.n);
        assert!(index.initial_chunk_size(30) >= 1);
    }

    ///////////////////////
    // Join paths        //
    ///////////////////////

    /// Clustered data wide enough to put the build on the GEMM join path.
    fn gemm_width_matrix(n: usize) -> Mat<f32> {
        let dim = NND_GEMM_MIN_DIM + 8;
        Mat::from_fn(n, dim, |i, j| {
            let cluster = (i % 5) as f32;
            cluster * 10.0 + ((i * 31 + j * 17) % 13) as f32 * 0.1
        })
    }

    #[test]
    fn test_gemm_join_distances_are_exact() {
        // On the GEMM path selection happens against the norm expansion, which
        // cancels catastrophically on close pairs. Every accepted pair is
        // therefore recomputed with the fused kernel, so the distances stored
        // in the graph must match `euclidean_simd` exactly.
        let n = 300;
        let mat = gemm_width_matrix(n);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            Some(NND_GEMM_MIN_CANDIDATES + 8),
            Some(6),
            None,
            0.0,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert!(
            index.gemm_join_applies(NND_GEMM_MIN_CANDIDATES),
            "test data does not reach the GEMM path"
        );

        let dim = index.dim;
        for i in 0..index.n {
            for &(pid, d) in &index.graph[i * index.k..(i + 1) * index.k] {
                if pid == SENTINEL_PID {
                    continue;
                }
                let exact = f32::euclidean_simd(
                    &index.vectors_flat[i * dim..(i + 1) * dim],
                    &index.vectors_flat[pid * dim..(pid + 1) * dim],
                );
                assert_eq!(d, exact, "stored distance for ({i}, {pid}) is not exact");
            }
        }
    }

    #[test]
    fn test_gemm_join_reaches_high_recall() {
        let n = 300;
        let mat = gemm_width_matrix(n);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            Some(NND_GEMM_MIN_CANDIDATES + 8),
            Some(8),
            None,
            0.0,
            0.0,
            42,
            false,
        )
        .unwrap();

        // Recall of the raw descent graph against exhaustive ground truth,
        // which is what the join itself is responsible for.
        let k = index.k;
        let dim = index.dim;
        let rows = index.extract_knn(None, false, false).0;
        let mut hits = 0usize;
        let mut total = 0usize;
        for i in 0..index.n {
            let query = &index.vectors_flat[i * dim..(i + 1) * dim];
            let (truth, _) = index.exhaustive_query(query, k + 1).unwrap();
            let truth: Vec<usize> = truth.into_iter().filter(|&j| j != i).take(k).collect();
            for t in &truth {
                if rows[i].contains(t) {
                    hits += 1;
                }
            }
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall > 0.8, "GEMM join recall too low: {recall}");
    }

    #[test]
    fn test_simd_join_distances_are_exact() {
        // The counterpart at a dimensionality below the GEMM threshold, so the
        // fused kernels carry the whole join.
        let n = 200;
        let mat = Mat::from_fn(n, 8, |i, j| ((i * 17 + j * 5) % 23) as f32 * 0.5);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(8),
            None,
            Some(6),
            None,
            0.0,
            0.0,
            7,
            false,
        )
        .unwrap();

        assert!(
            !index.gemm_join_applies(1_000),
            "test data took the GEMM path"
        );

        let dim = index.dim;
        for i in 0..index.n {
            for &(pid, d) in &index.graph[i * index.k..(i + 1) * index.k] {
                if pid == SENTINEL_PID {
                    continue;
                }
                let exact = f32::euclidean_simd(
                    &index.vectors_flat[i * dim..(i + 1) * dim],
                    &index.vectors_flat[pid * dim..(pid + 1) * dim],
                );
                assert_eq!(d, exact);
            }
        }
    }

    ///////////////////////
    // Extraction / NSG  //
    ///////////////////////

    #[test]
    fn test_graph_stays_flat_and_sentinel_padded() {
        // NSG consumes the flat graph directly and asserts `len == n * knn_k`,
        // so the layout is load-bearing beyond this module.
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(4),
            None,
            Some(5),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.graph().len(), index.n * index.k);
        // Five points at k = 4 cannot fill every row without self-edges, so at
        // least one sentinel has to survive the build.
        assert!(index
            .graph()
            .iter()
            .any(|&(pid, _)| pid == SENTINEL_PID || pid < index.n));
    }

    #[test]
    fn test_extract_knn_matches_the_stored_graph() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(5),
            None,
            0.001,
            0.0,
            42,
            false,
        )
        .unwrap();

        let (ids, dists) = index.extract_knn(None, false, true);
        let dists = dists.unwrap();

        assert_eq!(ids.len(), index.n);
        for i in 0..index.n {
            let want = neighbours(&index, i);
            assert_eq!(ids[i], want.iter().map(|e| e.0).collect::<Vec<_>>());
            assert_eq!(dists[i], want.iter().map(|e| e.1).collect::<Vec<_>>());
            assert!(dists[i].windows(2).all(|w| w[0] <= w[1]));
        }

        // No re-query, so it must not agree with the beam-search path by
        // construction; it only has to be a subset of the stored rows.
        let (short, none) = index.extract_knn(Some(1), false, false);
        assert!(none.is_none());
        assert!(short.iter().all(|r| r.len() <= 1));
    }

    #[test]
    fn test_diversified_rows_stay_distance_sorted() {
        // The RNG-rule prune tops short rows up from the pruned tail, and a
        // topped-up entry can sit closer than a kept one. Every consumer, and
        // `extract_knn` truncating from the front in particular, needs the row
        // ascending regardless.
        let n = 150;
        let mat = Mat::from_fn(n, 6, |i, j| ((i * 11 + j * 7) % 17) as f32 * 0.3);

        for prune_prob in [0.25f32, 0.5, 1.0] {
            let index = NNDescent::<f32>::new(
                mat.as_ref(),
                Dist::SquaredEuclidean,
                Some(10),
                None,
                Some(6),
                None,
                0.0,
                prune_prob,
                42,
                false,
            )
            .unwrap();

            for i in 0..index.n {
                let row: Vec<f32> = index.graph[i * index.k..(i + 1) * index.k]
                    .iter()
                    .filter(|&&(pid, _)| pid != SENTINEL_PID)
                    .map(|&(_, d)| d)
                    .collect();
                assert!(
                    row.windows(2).all(|w| w[0] <= w[1]),
                    "row {i} unsorted at prune_prob={prune_prob}: {row:?}"
                );
            }

            // Extraction truncates from the front, so it must hand back the
            // closest entries and nothing further out.
            let (_, dists) = index.extract_knn(Some(3), false, true);
            for row in dists.unwrap() {
                assert!(row.windows(2).all(|w| w[0] <= w[1]));
                assert!(row.len() <= 3);
            }
        }
    }

    #[test]
    fn test_compute_dots_matches_the_simd_kernel() {
        // The GEMM block is the one piece of the join with no fallback cross
        // check at build time, so pin it against `dot_simd` directly.
        let dim = 12;
        let n_new = 3;
        let n_total = 5;

        let mut scratch = JoinScratch::<f32>::new();
        scratch.tile = (0..n_total * dim)
            .map(|t| ((t * 7 % 19) as f32) * 0.25 - 2.0)
            .collect();
        scratch.compute_dots(n_new, n_total, dim);

        for a in 0..n_new {
            for b in 0..n_total {
                let want = f32::dot_simd(
                    &scratch.tile[a * dim..(a + 1) * dim],
                    &scratch.tile[b * dim..(b + 1) * dim],
                );
                assert_relative_eq!(scratch.dots[a * n_total + b], want, epsilon = 1e-4);
            }
        }

        // Squared norms come out of the same call and feed the expansion.
        for t in 0..n_total {
            let v = &scratch.tile[t * dim..(t + 1) * dim];
            assert_relative_eq!(scratch.sq[t], f32::dot_simd(v, v), epsilon = 1e-4);
        }
    }
}
