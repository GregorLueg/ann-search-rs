//! CAGRA beam search on MLX, for query-time retrieval over a ready
//! navigational graph.
//!
//! Port of [`crate::gpu::cagra_gpu_search`] as one custom Metal kernel
//! ([`BEAM_SOURCE`]). There is no dense algebra to hand to MLX here: the whole
//! search is gathers and selection, so MLX only holds the buffers and
//! dispatches. One SIMD group per query. The beam lives in registers, sorted,
//! slot `b * 32 + lane` in lane `lane`; an insert is a `simd_sum` rank plus a
//! `simd_shuffle_up` shift, so there is no serial thread-0 section and no
//! barrier in the merge. The visited set is a linear-probing hash table in
//! threadgroup memory, written with atomic compare-exchange since every lane
//! inserts at once. Each iteration's unvisited neighbours are compacted, then
//! scored [`lanes_per_neighbour`] lanes to a row: one lane per neighbour on
//! narrow rows, the row split across lanes on wide ones, where a single lane
//! walking a 784-dim row serially left the kernel latency bound.
//!
//! The graph is an input, not built here. Entry points are an input too,
//! either as fixed ids or, via [`CagraSearchMlx::search_routed`], as router
//! candidates the kernel scores and selects from itself. Without either the
//! search falls back to the medoid plus random nodes.

use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::utils::DimensionValidation;

////////////
// Consts //
////////////

/// Beam width (number of active candidates maintained during search)
const BEAM_WIDTH: usize = 16;

/// Maximum beam search iterations before forced termination
const MAX_BEAM_ITERS: usize = 48;

/// Preferred hash table size for visited-node tracking (power of 2). Half
/// the wgpu kernel's 2048: the table is most of the threadgroup memory, and
/// the smaller footprint fits more SIMD groups per core. A node dropped by a
/// reset was already worse than the beam's worst, so results do not change;
/// the cost is re-scoring it, which is why 512 already loses at high dim.
const HASH_SIZE: usize = 1024;

/// Smallest hash table the plan will shrink to before giving up
const MIN_HASH_SIZE: usize = 128;

/// Beam entries expanded per iteration
const EXPAND_PER_ITER: usize = 3;

/// Number of entry points per query
pub const N_ENTRY_POINTS_MLX: usize = 8;

/// Threadgroup memory Metal guarantees per threadgroup on Apple GPUs
pub const MLX_MAX_THREADGROUP_BYTES: usize = 32_768;

/// Queries per dispatch. Bounds a single command's run time; chunks are
/// queued asynchronously, so the host does not wait between them.
const MLX_QUERY_CHUNK: usize = 65_536;

/// Widest padded row (in `float4`s) still scored one lane per neighbour.
/// One lane wins at dim 32 and 64 and loses from dim 128.
const LPN_SERIAL_MAX_DIM4: usize = 16;

/// Widest padded row (in `float4`s) scored by `LPN_MID` lanes.
const LPN_MID_MAX_DIM4: usize = 64;

/// Lanes per neighbour for mid-width rows (best at dim 128).
const LPN_MID: usize = 4;

/// Lanes per neighbour for wide rows (best at dim 512 and 784; 16 and 32
/// were level or slightly worse).
const LPN_WIDE: usize = 8;

/// Marks an empty graph slot, beam slot or output slot.
const SENTINEL: u32 = 0x7FFF_FFFF;

/// Metal body of the beam search. Template args: `DIM4` (padded dim / 4),
/// `DEG` (graph degree), `BW` (beam width), `HASH` (table size, power of 2),
/// `EXPAND` (parents per iteration), `N_ENTRY`, `MAX_ITERS`, `COSINE`,
/// `K_OUT`, `N` (graph size), `LPN` (lanes per scored neighbour, a power of 2
/// up to 32, see [`lanes_per_neighbour`]) and `KEEP` (routed candidates kept,
/// 0 when the search is not routed).
///
/// Semantics follow the wgpu kernel: entries seed the beam, each iteration
/// claims the `EXPAND` best unexpanded beam entries, scores their unvisited
/// neighbours and merges each one whose distance beats the beam's worst, in
/// slot order. The table is cleared and refilled from the beam before an
/// iteration could push it past three quarters full; anything else seen again
/// is worse than the beam's worst and fails the merge check. Stops when every
/// beam entry has been expanded.
///
/// With `KEEP > 0` each query also carries a CSR list of router candidates
/// (`cands`, offsets in `coffs`). They are scored first and only the `KEEP`
/// closest stay, then `entries` go in as usual: the selection the host used to
/// do per query, moved onto the device.
const BEAM_SOURCE: &str = r#"
    constexpr uint SENT = 0x7FFFFFFFu;
    constexpr int BPL = (BW + 31) / 32;
    constexpr int TOTAL = EXPAND * DEG;
    constexpr int SPL = (TOTAL + 31) / 32;
    constexpr uint HMASK = HASH - 1;
    constexpr uint RESET_AT = HASH * 3 / 4;
    constexpr uint NPP = 32u / LPN;

    uint lane = thread_position_in_grid.x;
    uint qi = thread_position_in_grid.y;
    uint sub = lane % LPN;
    uint grp = lane / LPN;

    threadgroup float4 sq[DIM4];
    threadgroup atomic_uint vis[HASH];
    threadgroup uint cand[TOTAL];

    const device float4* q4 = (const device float4*)queries + (ulong)(qbase[0] + qi) * DIM4;
    const device float4* v4 = (const device float4*)vectors;
    for (uint i = lane; i < DIM4; i += 32) sq[i] = q4[i];
    for (uint i = lane; i < HASH; i += 32) atomic_store_explicit(&vis[i], SENT, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qnorm = 1.0f;
    if (COSINE) {
        float s = 0.0f;
        for (uint i = lane; i < DIM4; i += 32) s += dot(sq[i], sq[i]);
        qnorm = sqrt(simd_sum(s));
    }

    // Visited-set insert. NEW is true only for the lane that claimed the slot.
    #define HASH_INSERT(ID, NEW) { \
        uint hs_ = (ID) & HMASK; \
        NEW = false; \
        for (uint ha_ = 0; ha_ < HASH; ) { \
            uint ex_ = SENT; \
            if (atomic_compare_exchange_weak_explicit(&vis[hs_], &ex_, (ID), \
                    memory_order_relaxed, memory_order_relaxed)) { NEW = true; break; } \
            if (ex_ == (ID)) break; \
            if (ex_ != SENT) { hs_ = (hs_ + 1) & HMASK; ha_++; } \
        } \
    }

    // One lane, whole row.
    #define DIST(NODE, OUT) { \
        const device float4* x4_ = v4 + (ulong)(NODE) * DIM4; \
        float s_ = 0.0f; \
        if (COSINE) { \
            for (int i_ = 0; i_ < DIM4; i_++) s_ += dot(sq[i_], x4_[i_]); \
            OUT = 1.0f - s_ / (qnorm * norms[NODE]); \
        } else { \
            for (int i_ = 0; i_ < DIM4; i_++) { float4 d_ = sq[i_] - x4_[i_]; s_ += dot(d_, d_); } \
            OUT = s_; \
        } \
    }

    // LPN lanes, one row: each lane takes every LPN-th float4, then a
    // shuffle-xor tree sums within the group. Must be reached by all lanes.
    #define COOP_DIST(VALID, NODE, OUT) { \
        float a_ = 0.0f; \
        if (VALID) { \
            const device float4* x4_ = v4 + (ulong)(NODE) * DIM4; \
            for (uint i_ = sub; i_ < (uint)DIM4; i_ += LPN) { \
                if (COSINE) a_ += dot(sq[i_], x4_[i_]); \
                else { float4 d_ = sq[i_] - x4_[i_]; a_ += dot(d_, d_); } \
            } \
        } \
        for (uint o_ = LPN / 2; o_ > 0; o_ >>= 1) a_ += simd_shuffle_xor(a_, (ushort)o_); \
        OUT = a_; \
        if (COSINE && (VALID)) OUT = 1.0f - a_ / (qnorm * norms[NODE]); \
    }

    // Beam: slot b * 32 + lane, ascending. Empty slots are INF / SENT and
    // flagged expanded so they are never claimed.
    float bd[BPL];
    uint bi[BPL];
    uint bx[BPL];
    for (int b = 0; b < BPL; b++) { bd[b] = INFINITY; bi[b] = SENT; bx[b] = 1u; }

    #define WORST() simd_shuffle(bd[BPL - 1], (ushort)((BW - 1) & 31))

    // Uniform insert of (D, ID): rank by simd_sum, then shift everything from
    // the rank one slot up, the carry into lane 0 coming from the row below.
    // Rows go high to low so each reads its predecessor before it changes.
    #define BEAM_INSERT(D, ID) { \
        if ((D) < WORST()) { \
            uint pos_ = 0; \
            for (int b = 0; b < BPL; b++) pos_ += (uint)(bd[b] <= (D)); \
            pos_ = simd_sum(pos_); \
            for (int b = BPL - 1; b >= 0; b--) { \
                float ud_ = simd_shuffle_up(bd[b], (ushort)1); \
                uint ui_ = simd_shuffle_up(bi[b], (ushort)1); \
                uint ux_ = simd_shuffle_up(bx[b], (ushort)1); \
                if (b > 0) { \
                    float cd_ = simd_shuffle(bd[b > 0 ? b - 1 : 0], (ushort)31); \
                    uint ci_ = simd_shuffle(bi[b > 0 ? b - 1 : 0], (ushort)31); \
                    uint cx_ = simd_shuffle(bx[b > 0 ? b - 1 : 0], (ushort)31); \
                    if (lane == 0) { ud_ = cd_; ui_ = ci_; ux_ = cx_; } \
                } \
                uint slot_ = (uint)b * 32 + lane; \
                if (slot_ > pos_) { bd[b] = ud_; bi[b] = ui_; bx[b] = ux_; } \
                else if (slot_ == pos_) { bd[b] = (D); bi[b] = (ID); bx[b] = 0u; } \
                if (slot_ >= (uint)BW) { bd[b] = INFINITY; bi[b] = SENT; bx[b] = 1u; } \
            } \
        } \
    }

    // Merge every flagged lane's (D, ID) in lane order.
    #define MERGE_LANES(OK, D, ID) { \
        uint m_ = (uint)(ulong)simd_ballot(OK); \
        while (m_ != 0) { \
            ushort src_ = (ushort)ctz(m_); \
            m_ &= m_ - 1; \
            float dd_ = simd_shuffle((D), src_); \
            uint ii_ = simd_shuffle((ID), src_); \
            BEAM_INSERT(dd_, ii_); \
        } \
    }

    uint hcount = 0;
    if (KEEP > 0) {
        // Score every routed candidate, keep the KEEP closest. They are not
        // marked visited while scored, only once kept, as on the host.
        uint c_lo = coffs[qi];
        uint c_hi = coffs[qi + 1];
        for (uint c0 = c_lo; c0 < c_hi; c0 += NPP) {
            uint c = c0 + grp;
            bool valid = c < c_hi;
            uint id = valid ? cands[c] : 0u;
            valid = valid && id < (uint)N;
            float d;
            COOP_DIST(valid, id, d);
            MERGE_LANES(valid && sub == 0 && d < WORST(), d, id);
        }
        uint kept = 0;
        for (int b = 0; b < BPL; b++) {
            if ((uint)b * 32 + lane >= (uint)KEEP) { bd[b] = INFINITY; bi[b] = SENT; bx[b] = 1u; }
            if (bi[b] != SENT) { bool nw; HASH_INSERT(bi[b], nw); kept++; }
        }
        hcount = simd_sum(kept);
    }

    const device uint* ep = entries + (ulong)qi * N_ENTRY;
    for (int e0 = 0; e0 < N_ENTRY; e0 += 32) {
        uint e = (uint)e0 + lane;
        uint id = SENT;
        float d = INFINITY;
        bool ok = false;
        if (e < (uint)N_ENTRY) {
            id = ep[e];
            if (id < (uint)N) {
                bool nw;
                HASH_INSERT(id, nw);
                if (nw) { DIST(id, d); ok = true; }
            }
        }
        hcount += simd_sum((uint)ok);
        MERGE_LANES(ok, d, id);
    }

    for (int it = 0; it < MAX_ITERS; it++) {
        if (hcount + TOTAL > RESET_AT) {
            for (uint i = lane; i < HASH; i += 32) atomic_store_explicit(&vis[i], SENT, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint c = 0;
            for (int b = 0; b < BPL; b++) {
                if (bi[b] != SENT) { bool nw; HASH_INSERT(bi[b], nw); c++; }
            }
            hcount = simd_sum(c);
        }

        // Claim up to EXPAND unexpanded entries in beam order.
        uint parents[EXPAND];
        uint claimed = 0;
        for (int b = 0; b < BPL; b++) {
            uint m = (uint)(ulong)simd_ballot(bx[b] == 0u);
            while (m != 0 && claimed < (uint)EXPAND) {
                ushort src = (ushort)ctz(m);
                m &= m - 1;
                parents[claimed] = simd_shuffle(bi[b], src);
                if (lane == src) bx[b] = 1u;
                claimed++;
            }
        }
        if (claimed == 0) break;

        // Compact the unvisited neighbours in slot order, so the scoring
        // passes below carry no empty slots.
        uint ncand = 0;
        for (int r = 0; r < SPL; r++) {
            uint s = (uint)r * 32 + lane;
            uint p = s / DEG;
            bool nw = false;
            uint nbr = SENT;
            if (p < claimed) {
                nbr = graph[(ulong)parents[p] * DEG + (s - p * DEG)];
                if (nbr < (uint)N) HASH_INSERT(nbr, nw);
            }
            uint m = (uint)(ulong)simd_ballot(nw);
            if (nw) cand[ncand + popcount(m & ((1u << lane) - 1u))] = nbr;
            ncand += popcount(m);
        }
        hcount += ncand;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c0 = 0; c0 < ncand; c0 += NPP) {
            uint c = c0 + grp;
            bool valid = c < ncand;
            uint id = valid ? cand[c] : 0u;
            float d;
            COOP_DIST(valid, id, d);
            MERGE_LANES(valid && sub == 0 && d < WORST(), d, id);
        }
        // `cand` is rewritten next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint* oi = out_idx + (ulong)qi * K_OUT;
    device float* od = out_dist + (ulong)qi * K_OUT;
    for (int b = 0; b < BPL; b++) {
        uint slot = (uint)b * 32 + lane;
        if (slot < (uint)K_OUT && slot < (uint)BW) { oi[slot] = bi[b]; od[slot] = bd[b]; }
    }
    for (uint s = (uint)BW + lane; s < (uint)K_OUT; s += 32) { oi[s] = SENT; od[s] = INFINITY; }
"#;

////////////
// Params //
////////////

/// Parameters for the CAGRA style beam search on MLX. Mirrors
/// `CagraGpuSearchParams`, which lives behind the `gpu` feature.
#[derive(Clone, Debug)]
pub struct CagraMlxSearchParams {
    /// Optional width of the beam. Defaults to `BEAM_WIDTH`.
    pub beam_width: Option<usize>,
    /// Optional maximum iterations. Good rule of thumb is 3x beam_width.
    /// Defaults to `MAX_BEAM_ITERS`.
    pub max_beam_iters: Option<usize>,
    /// Optional number of entry points. Defaults to `N_ENTRY_POINTS_MLX`.
    pub n_entry_points: Option<usize>,
    /// Number of beam entries expanded per iteration. Defaults to
    /// `EXPAND_PER_ITER`.
    pub expand_per_iter: Option<usize>,
}

impl CagraMlxSearchParams {
    /// Generates a new instance of the search parameters
    ///
    /// ### Params
    ///
    /// * `beam_width` - Beam width for the kNN search
    /// * `max_beam_iters` - Maximum numbers of iterations to do. Rule of thumb
    ///   to be 2 to 3x beam width
    /// * `n_entry_points` - Number of entry points to use in the CAGRA graph.
    /// * `expand_per_iter` - Number of beam entries to expand per iteration.
    ///   Usually something between 1 to 4.
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
    ) -> Self {
        Self {
            beam_width,
            max_beam_iters,
            n_entry_points,
            expand_per_iter,
        }
    }

    /// Pull out the needed values for the beam search
    ///
    /// ### Returns
    ///
    /// Tuple of `(width, iters, n_entry, expand)`
    pub fn get_vals(&self) -> (usize, usize, usize, usize) {
        (
            self.beam_width.unwrap_or(BEAM_WIDTH),
            self.max_beam_iters.unwrap_or(MAX_BEAM_ITERS),
            self.get_n_entry(),
            self.expand_per_iter.unwrap_or(EXPAND_PER_ITER),
        )
    }

    /// Get the number of entry points
    ///
    /// ### Returns
    ///
    /// n_entry
    pub fn get_n_entry(&self) -> usize {
        self.n_entry_points.unwrap_or(N_ENTRY_POINTS_MLX)
    }

    /// Create params with the beam scaled to the requested `k`, as the wgpu
    /// search does.
    ///
    /// ### Params
    ///
    /// * `k_out` - Number of neighbours to return per query
    ///
    /// ### Returns
    ///
    /// Params with beam width and iterations scaled appropriately
    pub fn from_k(k_out: usize) -> Self {
        let beam_width = k_out.max(BEAM_WIDTH) * 2;
        Self::new(Some(beam_width), Some(beam_width * 3), None, None)
    }
}

impl Default for CagraMlxSearchParams {
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

//////////
// Plan //
//////////

/// Lanes that cooperate on one neighbour's distance. Narrow rows keep one
/// lane per neighbour; wide ones split the row so a lane does not walk
/// hundreds of serial loads while the rest of the group waits.
///
/// ### Params
///
/// * `dim_padded` - Dimensionality padded to a multiple of 4
///
/// ### Returns
///
/// A power of 2 dividing 32
pub fn lanes_per_neighbour(dim_padded: usize) -> usize {
    match dim_padded / 4 {
        d if d <= LPN_SERIAL_MAX_DIM4 => 1,
        d if d <= LPN_MID_MAX_DIM4 => LPN_MID,
        _ => LPN_WIDE,
    }
}

/// Size the visited table against the threadgroup memory budget. The staged
/// query (`dim_padded` f32) and the compacted neighbour list (`total_slots`
/// u32) are fixed; the beam lives in registers, so the table is the only
/// elastic term and halves until it fits.
///
/// ### Params
///
/// * `dim_padded` - Dimensionality padded to a multiple of 4
/// * `total_slots` - Neighbour slots per iteration, `expand * degree`
/// * `preferred_hash` - Table size to start from (power of 2)
/// * `max_tg_bytes` - Threadgroup memory budget in bytes
///
/// ### Returns
///
/// The hash table size, or `DimTooHighForSharedMemory` when even
/// `MIN_HASH_SIZE` does not fit next to the fixed terms
pub fn plan_beam_search_threadgroup(
    dim_padded: usize,
    total_slots: usize,
    preferred_hash: usize,
    max_tg_bytes: usize,
) -> Result<usize, AnnSearchErrors> {
    let fixed = dim_padded * size_of::<f32>() + total_slots * size_of::<u32>();
    let mut hash_size = preferred_hash.max(MIN_HASH_SIZE).next_power_of_two();
    while hash_size >= MIN_HASH_SIZE {
        if fixed + hash_size * size_of::<u32>() <= max_tg_bytes {
            return Ok(hash_size);
        }
        hash_size /= 2;
    }
    Err(AnnSearchErrors::DimTooHighForSharedMemory {
        chosen_dim: dim_padded,
        required: fixed + MIN_HASH_SIZE * size_of::<u32>(),
        available: max_tg_bytes,
    })
}

/////////////
// Helpers //
/////////////

/// Copy rows into a buffer with zero-padded columns.
///
/// ### Params
///
/// * `data` - Row-major input, `n * dim`
/// * `n` - Rows
/// * `dim` - Input row length
/// * `dim_padded` - Output row length, at least `dim`
///
/// ### Returns
///
/// Row-major `n * dim_padded` buffer
fn pad_rows(data: &[f32], n: usize, dim: usize, dim_padded: usize) -> Vec<f32> {
    if dim == dim_padded {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; n * dim_padded];
    for (dst, src) in out
        .chunks_exact_mut(dim_padded)
        .zip(data.chunks_exact(dim))
    {
        dst[..dim].copy_from_slice(src);
    }
    out
}

/// Self-query entry points, the scheme `NNDescentGpu::self_query_gpu` uses:
/// the node itself, then `n_entry - 1` strided picks from its row of
/// `rows`, topped up with LCG-random nodes.
///
/// ### Params
///
/// * `rows` - Flat `n * row_len` neighbour ids, `0x7FFFFFFF` in empty slots
///   (the wgpu path passes the NN-Descent kNN graph)
/// * `row_len` - Ids per row
/// * `n` - Number of nodes
/// * `n_entry` - Entry points per node, at least 1
/// * `seed` - Seed for the top-up
///
/// ### Returns
///
/// Flat `[n * n_entry]` node ids
pub fn self_entry_points(
    rows: &[u32],
    row_len: usize,
    n: usize,
    n_entry: usize,
    seed: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(n * n_entry);
    for i in 0..n {
        let valid: Vec<u32> = rows[i * row_len..(i + 1) * row_len]
            .iter()
            .copied()
            .filter(|&p| p != SENTINEL)
            .collect();
        let remaining = n_entry - 1;
        let stride = (valid.len() / remaining.max(1)).max(1);
        let start = out.len();
        out.push(i as u32);
        out.extend((0..remaining).filter_map(|j| valid.get(j * stride).copied()));
        let mut rng_val = (i as u32) ^ (seed as u32);
        while out.len() - start < n_entry {
            rng_val = rng_val.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push(rng_val % n as u32);
        }
    }
    out
}

////////////////////
// CagraSearchMlx //
////////////////////

/// CAGRA beam search over a ready navigational graph, on MLX. f32 only.
///
/// Holds an MLX stream, which is thread affine: build and query on the same
/// thread. The raw handles keep this type `!Send` and `!Sync`.
pub struct CagraSearchMlx {
    /// Number of vectors
    pub n: usize,
    /// Original (unpadded) dimensionality
    pub dim: usize,
    /// Dimensionality padded to a multiple of 4 for `float4` loads
    dim_padded: usize,
    /// Navigational graph degree
    degree: usize,
    /// Distance metric
    metric: Dist,
    /// Default entry point, always first in the fallback entry set
    medoid: u32,
    /// Navigational graph on the host, `n * degree`, for the self-query
    /// fallback entries
    nav_graph: Vec<u32>,
    /// Device vectors `[n, dim_padded]`
    vectors: Array,
    /// Device L2 norms `[n]` for Cosine, a dummy `[1]` otherwise
    norms: Array,
    /// Device graph `[n, degree]`
    graph: Array,
    /// Beam search kernel, see [`BEAM_SOURCE`]
    kernel: MetalKernel,
    /// Stream every op runs on. Declared last so it drops after the arrays.
    stream: Stream,
}

impl DimensionValidation for CagraSearchMlx {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl CagraSearchMlx {
    /// Upload vectors and a navigational graph for searching.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Row-major vectors, `n * dim`, unnormalised
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality
    /// * `metric` - Distance metric. Manhattan is not supported.
    /// * `nav_graph` - Flat `n * degree` neighbour ids, `0x7FFFFFFF` in empty
    ///   slots
    /// * `degree` - Graph degree
    /// * `medoid` - Default entry point
    ///
    /// ### Returns
    ///
    /// The searcher, everything resident on the device
    pub fn new(
        vectors_flat: &[f32],
        n: usize,
        dim: usize,
        metric: Dist,
        nav_graph: Vec<u32>,
        degree: usize,
        medoid: u32,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        install_error_handler();

        let dim_padded = dim.next_multiple_of(4);
        let stream = Stream::default_gpu();
        let vectors = Array::from_f32(
            &pad_rows(vectors_flat, n, dim, dim_padded),
            &[n as i32, dim_padded as i32],
        );
        let norms = if metric == Dist::Cosine {
            let norms: Vec<f32> = vectors_flat
                .chunks_exact(dim)
                .map(f32::calculate_l2_norm)
                .collect();
            Array::from_f32(&norms, &[n as i32])
        } else {
            Array::from_f32(&[0.0], &[1])
        };
        let graph = Array::from_u32(&nav_graph, &[n as i32, degree as i32]);
        eval_all(&[&vectors, &norms, &graph], false)?;

        Ok(Self {
            n,
            dim,
            dim_padded,
            degree,
            metric,
            medoid,
            nav_graph,
            vectors,
            norms,
            graph,
            kernel: MetalKernel::new(
                "cagra_beam_search",
                &[
                    "vectors", "norms", "graph", "queries", "entries", "qbase", "cands",
                    "coffs",
                ],
                &["out_idx", "out_dist"],
                BEAM_SOURCE,
            ),
            stream,
        })
    }

    /// Fallback entry points: the medoid, then random nodes.
    ///
    /// ### Params
    ///
    /// * `n_queries` - Number of queries
    /// * `n_entry` - Entry points per query
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// Flat `[n_queries * n_entry]` node ids
    fn default_entry_points(&self, n_queries: usize, n_entry: usize, seed: usize) -> Vec<u32> {
        let mut rng = SmallRng::seed_from_u64(seed as u64);
        (0..n_queries)
            .flat_map(|_| {
                std::iter::once(self.medoid)
                    .chain((1..n_entry).map(|_| rng.random_range(0..self.n as u32)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Search a batch of queries.
    ///
    /// ### Params
    ///
    /// * `queries_flat` - Row-major queries, `n_queries * dim`
    /// * `n_queries` - Number of queries
    /// * `k` - Neighbours per query
    /// * `query_params` - Beam parameters; `None` scales them to `k`
    /// * `entry_points` - Optional `[n_queries * n_entry]` entry ids; `None`
    ///   uses the medoid plus random nodes
    /// * `seed` - Seed for the fallback entries
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per query, ascending. Unfilled slots are
    /// dropped, so a row can be shorter than `k`.
    pub fn search(
        &self,
        queries_flat: &[f32],
        n_queries: usize,
        k: usize,
        query_params: Option<CagraMlxSearchParams>,
        entry_points: Option<&[u32]>,
        seed: usize,
    ) -> KnnResult<f32> {
        if n_queries == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.check_dim(queries_flat.len() / n_queries)?;
        let params = query_params.unwrap_or_else(|| CagraMlxSearchParams::from_k(k));
        let n_entry = params.get_n_entry();
        let entries = match entry_points {
            Some(e) => {
                assert_eq!(e.len(), n_queries * n_entry, "entry points per query");
                e.to_vec()
            }
            None => self.default_entry_points(n_queries, n_entry, seed),
        };
        let queries = Array::from_f32(
            &pad_rows(queries_flat, n_queries, self.dim, self.dim_padded),
            &[n_queries as i32, self.dim_padded as i32],
        );
        self.run(&queries, n_queries, &entries, n_entry, None, k, &params, HASH_SIZE)
    }

    /// Search a batch of queries seeded from router candidates. Per query the
    /// kernel scores every candidate, keeps the `n_entry - 1` closest and adds
    /// `fixed_entry` (the medoid): the selection `NNDescentGpu` does on the
    /// host, done on the device.
    ///
    /// ### Params
    ///
    /// * `queries_flat` - Row-major queries, `n_queries * dim`
    /// * `n_queries` - Number of queries
    /// * `k` - Neighbours per query
    /// * `query_params` - Beam parameters; `None` scales them to `k`
    /// * `cand_ids` - Candidates of every query, concatenated
    /// * `cand_offsets` - `n_queries + 1` offsets into `cand_ids`
    /// * `fixed_entry` - Entry added to every query, never scored as a
    ///   candidate; leave it out of `cand_ids`
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per query, ascending. Unfilled slots are
    /// dropped, so a row can be shorter than `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_routed(
        &self,
        queries_flat: &[f32],
        n_queries: usize,
        k: usize,
        query_params: Option<CagraMlxSearchParams>,
        cand_ids: &[u32],
        cand_offsets: &[u32],
        fixed_entry: u32,
    ) -> KnnResult<f32> {
        if n_queries == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.check_dim(queries_flat.len() / n_queries)?;
        assert_eq!(cand_offsets.len(), n_queries + 1, "candidate offsets");
        let params = query_params.unwrap_or_else(|| CagraMlxSearchParams::from_k(k));
        let keep = params.get_n_entry().saturating_sub(1);
        let queries = Array::from_f32(
            &pad_rows(queries_flat, n_queries, self.dim, self.dim_padded),
            &[n_queries as i32, self.dim_padded as i32],
        );
        let entries = vec![fixed_entry; n_queries];
        let routed = (keep > 0).then_some((cand_ids, cand_offsets, keep));
        self.run(&queries, n_queries, &entries, 1, routed, k, &params, HASH_SIZE)
    }

    /// Search every indexed vector against the graph (self-kNN).
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector, self included
    /// * `query_params` - Beam parameters; `None` scales them to `k`
    /// * `entry_points` - Optional `[n * n_entry]` entry ids, e.g. from
    ///   [`self_entry_points`] over the NN-Descent kNN graph as the wgpu path
    ///   does; `None` builds them the same way from the navigational graph
    /// * `seed` - Seed for the entry top-up
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per vector, ascending
    pub fn self_search(
        &self,
        k: usize,
        query_params: Option<CagraMlxSearchParams>,
        entry_points: Option<&[u32]>,
        seed: usize,
    ) -> KnnResult<f32> {
        let params = query_params.unwrap_or_else(|| CagraMlxSearchParams::from_k(k));
        let n_entry = params.get_n_entry();
        let entries = match entry_points {
            Some(e) => {
                assert_eq!(e.len(), self.n * n_entry, "entry points per vector");
                e.to_vec()
            }
            None => self_entry_points(&self.nav_graph, self.degree, self.n, n_entry, seed),
        };
        // The device vectors double as the queries: no re-upload.
        self.run(&self.vectors, self.n, &entries, n_entry, None, k, &params, HASH_SIZE)
    }

    /// Queue the beam search in chunks and collect the results.
    ///
    /// ### Params
    ///
    /// * `queries` - Device queries `[n_queries, dim_padded]`
    /// * `n_queries` - Number of queries
    /// * `entries` - Flat `[n_queries * n_entry]` entry ids
    /// * `n_entry` - Entries per query
    /// * `routed` - Optional `(candidate ids, n_queries + 1 offsets, keep)`
    ///   scored on the device before the entries go in
    /// * `k` - Neighbours per query
    /// * `params` - Beam parameters; only width, iterations and expansion
    ///   are read
    /// * `hash_pref` - Preferred visited table size; raised to twice the
    ///   beam plus one iteration's slots, then shrunk to fit by
    ///   [`plan_beam_search_threadgroup`]
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per query, ascending, unfilled slots dropped
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        queries: &Array,
        n_queries: usize,
        entries: &[u32],
        n_entry: usize,
        routed: Option<(&[u32], &[u32], usize)>,
        k: usize,
        params: &CagraMlxSearchParams,
        hash_pref: usize,
    ) -> KnnResult<f32> {
        let (width, iters, _, expand) = params.get_vals();
        if k == 0 || n_queries == 0 {
            return Ok((vec![Vec::new(); n_queries], vec![Vec::new(); n_queries]));
        }
        let expand = expand.max(1);
        let hash_size = plan_beam_search_threadgroup(
            self.dim_padded,
            expand * self.degree,
            hash_pref.max((2 * (width + expand * self.degree)).next_power_of_two()),
            MLX_MAX_THREADGROUP_BYTES,
        )?;
        let lpn = lanes_per_neighbour(self.dim_padded);
        let template = [
            ("DIM4", (self.dim_padded / 4) as i32),
            ("DEG", self.degree as i32),
            ("BW", width.max(1) as i32),
            ("HASH", hash_size as i32),
            ("EXPAND", expand as i32),
            ("N_ENTRY", n_entry as i32),
            ("MAX_ITERS", iters as i32),
            ("COSINE", (self.metric == Dist::Cosine) as i32),
            ("K_OUT", k as i32),
            ("N", self.n as i32),
            ("LPN", lpn as i32),
            ("KEEP", routed.map_or(0, |r| r.2) as i32),
        ];

        // Offsets stay global, so one candidate buffer serves every chunk.
        let cands = match routed {
            Some((ids, _, _)) if !ids.is_empty() => Array::from_u32(ids, &[ids.len() as i32]),
            _ => Array::from_u32(&[SENTINEL], &[1]),
        };
        let mut pending = Vec::with_capacity(n_queries.div_ceil(MLX_QUERY_CHUNK));
        for start in (0..n_queries).step_by(MLX_QUERY_CHUNK) {
            let end = (start + MLX_QUERY_CHUNK).min(n_queries);
            let n_q = end - start;
            let ent = Array::from_u32(
                &entries[start * n_entry..end * n_entry],
                &[n_q as i32, n_entry as i32],
            );
            let qbase = Array::from_u32(&[start as u32], &[1]);
            let coffs = match routed {
                Some((_, offs, _)) => Array::from_u32(&offs[start..=end], &[n_q as i32 + 1]),
                None => Array::from_u32(&[0, 0], &[2]),
            };
            let shape = [n_q as i32, k as i32];
            let mut out = self.kernel.apply(
                &[
                    &self.vectors,
                    &self.norms,
                    &self.graph,
                    queries,
                    &ent,
                    &qbase,
                    &cands,
                    &coffs,
                ],
                &[
                    OutputSpec {
                        shape: &shape,
                        dtype: MLX_UINT32,
                    },
                    OutputSpec {
                        shape: &shape,
                        dtype: MLX_FLOAT32,
                    },
                ],
                [32, n_q as i32, 1],
                [32, 1, 1],
                &template,
                &self.stream,
            )?;
            let dist = out.pop().expect("kernel has two outputs");
            let idx = out.pop().expect("kernel has two outputs");
            eval_all(&[&idx, &dist], true)?;
            pending.push((idx, dist));
        }

        let mut indices = Vec::with_capacity(n_queries);
        let mut distances = Vec::with_capacity(n_queries);
        for (idx, dist) in &pending {
            eval_all(&[idx, dist], false)?;
            for (ir, dr) in idx.as_u32()?.chunks_exact(k).zip(dist.as_f32()?.chunks_exact(k)) {
                let (i, d): (Vec<usize>, Vec<f32>) = ir
                    .iter()
                    .zip(dr)
                    .filter(|(&j, _)| (j as usize) < self.n)
                    // Cosine can round a self-distance to just under zero.
                    .map(|(&j, &v)| (j as usize, v.max(0.0)))
                    .unzip();
                indices.push(i);
                distances.push(d);
            }
        }
        Ok((indices, distances))
    }

    /// Returns the size of the searcher in bytes, host copy plus device copy
    ///
    /// ### Returns
    ///
    /// Number of bytes used
    pub fn memory_usage_bytes(&self) -> usize {
        let host = self.nav_graph.capacity() * size_of::<u32>();
        let norms = if self.metric == Dist::Cosine { self.n } else { 1 };
        let device = (self.n * self.dim_padded + norms) * size_of::<f32>()
            + self.n * self.degree * size_of::<u32>();
        std::mem::size_of_val(self) + host + device
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    /// Brute-force neighbours of each query, by the given metric.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries
    /// * `data` - Row-major database
    /// * `dim` - Dimensionality
    /// * `k` - Neighbours per query
    /// * `cosine` - Cosine instead of squared Euclidean
    ///
    /// ### Returns
    ///
    /// The `k` nearest ids per query, ascending
    pub(super) fn brute_force(
        queries: &[f32],
        data: &[f32],
        dim: usize,
        k: usize,
        cosine: bool,
    ) -> Vec<Vec<usize>> {
        queries
            .chunks_exact(dim)
            .map(|q| {
                let mut d: Vec<(f32, usize)> = data
                    .chunks_exact(dim)
                    .enumerate()
                    .map(|(j, x)| {
                        let v = if cosine {
                            1.0 - f32::dot_simd(q, x)
                                / (f32::calculate_l2_norm(q) * f32::calculate_l2_norm(x))
                        } else {
                            f32::euclidean_simd(q, x)
                        };
                        (v, j)
                    })
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.into_iter().take(k).map(|(_, j)| j).collect()
            })
            .collect()
    }

    /// Exact kNN graph (self excluded), padded with the sentinel.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major vectors
    /// * `dim` - Dimensionality
    /// * `k` - Degree
    /// * `cosine` - Cosine instead of squared Euclidean
    ///
    /// ### Returns
    ///
    /// Flat `n * k` graph
    fn brute_force_graph(data: &[f32], dim: usize, k: usize, cosine: bool) -> Vec<u32> {
        brute_force(data, data, dim, k + 1, cosine)
            .into_iter()
            .enumerate()
            .flat_map(|(i, row)| {
                let mut r: Vec<u32> = row
                    .into_iter()
                    .filter(|&j| j != i)
                    .take(k)
                    .map(|j| j as u32)
                    .collect();
                r.resize(k, SENTINEL);
                r
            })
            .collect()
    }

    /// Mean recall of `found` against `truth`.
    ///
    /// ### Params
    ///
    /// * `truth` - Ground truth per query
    /// * `found` - Result per query
    ///
    /// ### Returns
    ///
    /// Fraction of truth ids found
    pub(super) fn recall(truth: &[Vec<usize>], found: &[Vec<usize>]) -> f64 {
        let hits: usize = truth
            .iter()
            .zip(found)
            .map(|(t, f)| t.iter().filter(|g| f.contains(g)).count())
            .sum();
        hits as f64 / truth.iter().map(Vec::len).sum::<usize>() as f64
    }

    /// Uniform random rows in `[-10, 10)`.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// Row-major data
    fn uniform(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.random_range(-10.0..10.0)).collect()
    }

    #[test]
    fn test_mlx_cagra_plan_shrinks_the_hash_then_errors() {
        for budget in [16_384usize, 32_768, 49_152, 65_536] {
            for dim in [32usize, 128, 512, 1024] {
                let h = plan_beam_search_threadgroup(dim, 90, 2048, budget).unwrap();
                assert!(h.is_power_of_two() && (MIN_HASH_SIZE..=2048).contains(&h));
                assert!(dim * 4 + 90 * 4 + h * 4 <= budget, "dim {dim} budget {budget}");
            }
        }
        assert_eq!(plan_beam_search_threadgroup(128, 90, 2048, 32_768).unwrap(), 2048);
        assert_eq!(plan_beam_search_threadgroup(3072, 90, 2048, 16_384).unwrap(), 512);
        // A 4096-wide row alone fills 16 KiB.
        assert!(plan_beam_search_threadgroup(4096, 90, 2048, 16_384).is_err());
    }

    #[test]
    fn test_mlx_cagra_star_graph() {
        let (n, dim, deg, k) = (50, 32, 10, 5);
        let mut data = vec![0.0f32; n * dim];
        for i in 1..n {
            for j in 0..dim {
                data[i * dim + j] = i as f32 * 0.1 + j as f32 * 0.001;
            }
        }
        let graph = brute_force_graph(&data, dim, deg, false);
        let s =
            CagraSearchMlx::new(&data, n, dim, Dist::SquaredEuclidean, graph, deg, 25).unwrap();
        let (idx, dist) = s
            .search(&vec![0.0; dim], 1, k, Some(CagraMlxSearchParams::default()), None, 42)
            .unwrap();
        assert_eq!(idx[0], vec![0, 1, 2, 3, 4]);
        assert!(dist[0].windows(2).all(|w| w[0] <= w[1]));
    }

    /// Rows on an 8-dim subspace of `dim`, so beam search recall stays high
    /// at any width. The projection is fixed, so base and queries share it.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `seed` - RNG seed for the latent coordinates
    ///
    /// ### Returns
    ///
    /// Row-major data
    fn latent(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let proj = uniform(8, dim, 999);
        let z = uniform(n, 8, seed);
        z.chunks_exact(8)
            .flat_map(|zr| {
                (0..dim)
                    .map(|j| (0..8).map(|l| zr[l] * proj[l * dim + j]).sum::<f32>() / 10.0)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Recall on an exact graph, and the self-query finding every point.
    /// The widths cover one, four and eight lanes per neighbour.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    fn check_recall(metric: Dist) {
        for (dim, lpn) in [(30usize, 1usize), (200, LPN_MID), (300, LPN_WIDE)] {
            assert_eq!(lanes_per_neighbour(dim.next_multiple_of(4)), lpn);
            let (n, deg, k, nq) = (2000, 16, 10, 100);
            let data = latent(n, dim, 123);
            let queries = latent(nq, dim, 9);
            let cosine = metric == Dist::Cosine;
            let graph = brute_force_graph(&data, dim, deg, cosine);
            let s = CagraSearchMlx::new(&data, n, dim, metric, graph, deg, 0).unwrap();

            let (idx, dist) = s.search(&queries, nq, k, None, None, 42).unwrap();
            let r = recall(&brute_force(&queries, &data, dim, k, cosine), &idx);
            assert!(r > 0.9, "{metric:?} dim {dim} recall {r}");
            assert!(dist.iter().all(|d| d.windows(2).all(|w| w[0] <= w[1])));

            let (idx, _) = s.self_search(k, None, None, 42).unwrap();
            assert!(idx.iter().enumerate().all(|(i, row)| row[0] == i));
            let r = recall(&brute_force(&data, &data, dim, k, cosine), &idx);
            assert!(r > 0.9, "{metric:?} dim {dim} self recall {r}");
        }
    }

    #[test]
    fn test_mlx_cagra_recall_euclidean() {
        check_recall(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_mlx_cagra_recall_cosine() {
        check_recall(Dist::Cosine);
    }

    /// Routed search against the same selection done on the host: the
    /// medoid plus the `n_entry - 1` closest candidates as fixed entries.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    /// * `dim` - Dimensionality
    fn check_routed_matches_host(metric: Dist, dim: usize) {
        let (n, deg, k, nq, n_cand) = (2000, 16, 10, 200, 150);
        let cosine = metric == Dist::Cosine;
        let data = latent(n, dim, 3);
        let queries = latent(nq, dim, 4);
        let graph = brute_force_graph(&data, dim, deg, cosine);
        let medoid = 0u32;
        let s = CagraSearchMlx::new(&data, n, dim, metric, graph, deg, medoid).unwrap();
        let n_entry = CagraMlxSearchParams::from_k(k).get_n_entry();

        let mut rng = StdRng::seed_from_u64(5);
        let mut offsets = vec![0u32];
        let mut ids = Vec::new();
        let mut entries = Vec::new();
        for q in queries.chunks_exact(dim) {
            let cands: Vec<u32> = (0..n_cand).map(|_| rng.random_range(1..n as u32)).collect();
            let mut cands_sorted = cands.clone();
            cands_sorted.sort_unstable();
            cands_sorted.dedup();
            let mut scored: Vec<(f32, u32)> = cands_sorted
                .iter()
                .map(|&c| {
                    let x = &data[c as usize * dim..(c as usize + 1) * dim];
                    let d = if cosine {
                        1.0 - f32::dot_simd(q, x)
                            / (f32::calculate_l2_norm(q) * f32::calculate_l2_norm(x))
                    } else {
                        f32::euclidean_simd(q, x)
                    };
                    (d, c)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            entries.push(medoid);
            entries.extend(scored.iter().take(n_entry - 1).map(|&(_, c)| c));
            ids.extend_from_slice(&cands_sorted);
            offsets.push(ids.len() as u32);
        }

        let (host, _) = s.search(&queries, nq, k, None, Some(&entries), 0).unwrap();
        let (dev, _) = s
            .search_routed(&queries, nq, k, None, &ids, &offsets, medoid)
            .unwrap();
        let same = host.iter().zip(&dev).filter(|(a, b)| a == b).count();
        assert!(same as f64 >= 0.99 * nq as f64, "{metric:?} dim {dim}: {same}/{nq} rows identical");
    }

    #[test]
    fn test_mlx_cagra_routed_matches_host_selection() {
        for dim in [30, 300] {
            check_routed_matches_host(Dist::SquaredEuclidean, dim);
            check_routed_matches_host(Dist::Cosine, dim);
        }
    }

    /// A beam that visits far more nodes than the table holds: without the
    /// reset the table fills and recall collapses.
    #[test]
    fn test_mlx_cagra_outgrows_the_hash_table() {
        let (n, dim, deg, k, nq) = (2000, 32, 15, 10, 20);
        let data = uniform(n, dim, 7);
        let queries = uniform(nq, dim, 8);
        let graph = brute_force_graph(&data, dim, deg, false);
        let s =
            CagraSearchMlx::new(&data, n, dim, Dist::SquaredEuclidean, graph, deg, 0).unwrap();
        let params = CagraMlxSearchParams::new(Some(64), Some(192), Some(8), Some(3));
        let entries = s.default_entry_points(nq, 8, 3);
        let q = Array::from_f32(&queries, &[nq as i32, dim as i32]);
        let (idx, _) = s.run(&q, nq, &entries, 8, None, k, &params, 128).unwrap();
        let r = recall(&brute_force(&queries, &data, dim, k, false), &idx);
        assert!(r > 0.85, "recall {r}");
    }

    #[test]
    fn test_mlx_cagra_k_above_beam_and_odd_dim() {
        let (n, dim, deg) = (500, 7, 8);
        let data = uniform(n, dim, 5);
        let graph = brute_force_graph(&data, dim, deg, false);
        let s =
            CagraSearchMlx::new(&data, n, dim, Dist::SquaredEuclidean, graph, deg, 0).unwrap();
        let params = CagraMlxSearchParams::new(Some(8), Some(24), None, None);
        let (idx, _) = s.search(&data[..dim * 3], 3, 12, Some(params), None, 1).unwrap();
        assert!(idx.iter().all(|row| row.len() == 8));
        assert!(s.search(&data[..dim * 2], 1, 5, None, None, 1).is_err());
        assert!(CagraSearchMlx::new(&data, n, dim, Dist::Manhattan, vec![0; n * deg], deg, 0)
            .is_err());
    }
}

/// Cross-check against the wgpu search on the same NN-Descent graph and the
/// same entry points.
#[cfg(all(test, feature = "gpu"))]
mod wgpu_tests {
    use super::tests::{brute_force, recall};
    use super::*;
    use crate::gpu::cagra_gpu_search::CagraGpuSearchParams;
    use crate::gpu::nndescent_gpu::NNDescentGpu;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use rand::rngs::StdRng;

    /// Both searches on one graph, recall within noise of each other.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    fn cross_check(metric: Dist) {
        let (n, dim, k, nq) = (5000, 32, 15, 500);
        let mut rng = StdRng::seed_from_u64(11);
        let data: Vec<f32> = (0..(n + nq) * dim)
            .map(|_| rng.random_range(-1.0..1.0f32))
            .collect();
        let (base, queries) = data.split_at(n * dim);
        let mut index = NNDescentGpu::<f32, WgpuRuntime>::build(
            (base, n, dim),
            metric,
            Some(30),
            None,
            None,
            None,
            None,
            None,
            None,
            42,
            false,
            true,
            WgpuDevice::DefaultDevice,
        )
        .unwrap();
        let s = CagraSearchMlx::new(
            base,
            n,
            dim,
            metric,
            index.nav_graph().to_vec(),
            index.k,
            index.medoid,
        )
        .unwrap();
        let cosine = metric == Dist::Cosine;

        let truth = brute_force(queries, base, dim, k, cosine);
        // Default beam, then a narrow one that leaves recall off the ceiling.
        for (bw, iters, expand) in [(2 * k.max(16), 6 * k.max(16), 3), (16, 24, 1)] {
            let gp = CagraGpuSearchParams::new(Some(bw), Some(iters), None, Some(expand));
            let mp = CagraMlxSearchParams::new(Some(bw), Some(iters), None, Some(expand));
            let entries = index.query_entry_points(queries, nq, gp.get_n_entry());
            let (wgpu_idx, _) = index
                .query_batch_gpu(queries, nq, Some(gp), k, 42)
                .unwrap();
            let (mlx_idx, _) = s
                .search(queries, nq, k, Some(mp), Some(&entries), 42)
                .unwrap();
            let (rw, rm) = (recall(&truth, &wgpu_idx), recall(&truth, &mlx_idx));
            println!("{metric:?} beam {bw} query recall: wgpu {rw:.4}, mlx {rm:.4}");
            assert!((rw - rm).abs() < 0.01, "wgpu {rw} mlx {rm}");
        }

        let knn_rows: Vec<u32> = index
            .knn_graph()
            .iter()
            .map(|&(p, _)| p as u32)
            .collect();
        let self_entries = self_entry_points(&knn_rows, index.k, n, N_ENTRY_POINTS_MLX, 42);
        let (wgpu_self, _) = index.self_query_gpu(k, None, 42).unwrap();
        let (mlx_self, _) = s.self_search(k, None, Some(&self_entries), 42).unwrap();
        let truth = brute_force(base, base, dim, k, cosine);
        let (rw, rm) = (recall(&truth, &wgpu_self), recall(&truth, &mlx_self));
        println!("{metric:?} self recall: wgpu {rw:.4}, mlx {rm:.4}");
        assert!((rw - rm).abs() < 0.01, "wgpu {rw} mlx {rm}");
    }

    #[test]
    fn test_mlx_cagra_matches_wgpu_euclidean() {
        cross_check(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_mlx_cagra_matches_wgpu_cosine() {
        cross_check(Dist::Cosine);
    }
}
