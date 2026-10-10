//! NN-Descent kNN graph construction on MLX, plus the CAGRA graph
//! optimisation that turns it into a navigational graph.
//!
//! A port of [`crate::gpu::nndescent_gpu`]: random init, random-projection
//! forest init ([`crate::mlx::forest_mlx`]), then reverse candidates, local
//! join and proposal merge per iteration. Everything except the forest's
//! projection GEMM is a custom Metal kernel; MLX has no op for graph work.
//!
//! ## Immutability
//!
//! MLX arrays are immutable, and the wgpu loop mutates the graph in place in
//! two kernels: the local join clears the is-new flag on the entries it
//! sampled, and the merge rewrites each row. Here the graph is
//! double-buffered: every merge writes a fresh `[n, build_k]` pair and the
//! previous one is dropped. The flag clearing moved out of the local join into
//! the merge, which recomputes the join's sampling decision for its own
//! forward row (the hash and the per-kind cap over forward entries do not
//! depend on anything the join saw that the merge cannot), so the join only
//! reads the graph. Scatter buffers (reverse edges, proposals) are kernel
//! outputs with `atomic_outputs` and a zero `init_value`, which replaces the
//! wgpu reset kernels. The update counter is a per-node output summed by MLX,
//! downloaded once per iteration: the only host sync in the loop.

use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::cpu::vamana::compute_medoid;
use crate::mlx::ffi::*;
use crate::mlx::forest_mlx::forest_init_mlx;
use crate::prelude::*;
use crate::utils::nndescent_utils::{unpack_knn_graph, SENTINEL_PID};

////////////
// Consts //
////////////

/// Max proposals per node per iteration. Overflow is silently dropped.
pub const MLX_MAX_PROPOSALS: usize = 128;

/// Most new, and separately most old, candidates the local join keeps per node
/// per iteration. Same cap as the wgpu path.
pub const MLX_NND_MAX_CANDIDATES: usize = 60;

/// Default maximum number of NN-Descent iterations
const DEFAULT_MAX_ITERS: usize = 15;

/// Default convergence threshold (fraction of `n * build_k` edges updated)
const DEFAULT_DELTA: f32 = 0.001;

/// Default sampling rate for the local join
const DEFAULT_RHO: f32 = 1.0;

/// Threadgroup memory per threadgroup on every Apple GPU Metal exposes.
pub(crate) const MLX_TG_BYTES: usize = 32 * 1024;

/// SIMD width on Apple GPUs; the per-node cooperative kernels run one SIMD
/// group per node.
pub(crate) const SIMD: i32 = 32;

/// Threads per threadgroup for the one-thread-per-node kernels.
const NODE_TG: i32 = 64;

/// Local-join threadgroup extent along the candidate-`j` axis. 16 x 8 = 128
/// threads, the shape the wgpu sweep picked on the same hardware.
const LOCAL_JOIN_TX: usize = 16;

/// Local-join threadgroup extent along the candidate-`i` axis.
const LOCAL_JOIN_TY: usize = 8;

/// Longest row, in `float4`s, whose staged stride gets one `float4` of
/// bank-conflict padding. Same rule as the wgpu staging plan.
const LOCAL_JOIN_PAD_MAX_LINES: usize = 64;

/// Unfilled-slot marker in the graphs, the low 31 bits all set.
pub const MLX_SENTINEL: u32 = 0x7FFF_FFFF;

///////////////////
// Metal sources //
///////////////////

/// Helpers shared by every NN-Descent kernel. Template ints arrive as `int`,
/// so `COS` is an int flag. Proposal distances travel as their `uint` bit
/// pattern, because an atomic kernel's outputs are all atomic and `uint` is
/// the one atomic type every Apple GPU supports.
pub(crate) const NND_HEADER: &str = r#"
inline uint nnd_xorshift(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline uint nnd_entry_hash(uint node, uint entry, uint seed) {
    return nnd_xorshift(node ^ (entry * 2654435769u) ^ seed);
}

// Squared Euclidean, or the raw dot product under COS, between rows a and b.
// Norms stay with the caller: MLX binds a small input to `constant` rather
// than `device`, so a pointer parameter would pin one address space.
template <int D4, int COS>
inline float nnd_dist(const device float4* v, uint a, uint b) {
    const device float4* pa = v + (ulong)a * D4;
    const device float4* pb = v + (ulong)b * D4;
    float4 acc = 0.0f;
    for (int l = 0; l < D4; l++) {
        float4 x = pa[l];
        float4 y = pb[l];
        if (COS) {
            acc = fma(x, y, acc);
        } else {
            float4 d = x - y;
            acc = fma(d, d, acc);
        }
    }
    return acc.x + acc.y + acc.z + acc.w;
}

// Raw dot product (COS) or squared distance between two staged rows. Four
// accumulators keep independent chains in flight.
template <int D4, int COS>
inline float nnd_pair(const threadgroup float4* a, const threadgroup float4* b) {
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int l = 0;
    for (; l + 4 <= D4; l += 4) {
        if (COS) {
            a0 = fma(a[l], b[l], a0);
            a1 = fma(a[l + 1], b[l + 1], a1);
            a2 = fma(a[l + 2], b[l + 2], a2);
            a3 = fma(a[l + 3], b[l + 3], a3);
        } else {
            float4 d0 = a[l] - b[l];
            float4 d1 = a[l + 1] - b[l + 1];
            float4 d2 = a[l + 2] - b[l + 2];
            float4 d3 = a[l + 3] - b[l + 3];
            a0 = fma(d0, d0, a0);
            a1 = fma(d1, d1, a1);
            a2 = fma(d2, d2, a2);
            a3 = fma(d3, d3, a3);
        }
    }
    for (; l < D4; l++) {
        if (COS) {
            a0 = fma(a[l], b[l], a0);
        } else {
            float4 d = a[l] - b[l];
            a0 = fma(d, d, a0);
        }
    }
    float4 t = (a0 + a1) + (a2 + a3);
    return t.x + t.y + t.z + t.w;
}

// Claim a proposal slot of node `to` and store (other, d). The atomic claim
// is what keeps the index and distance stores of one proposal paired.
inline void nnd_emit(
    device atomic<uint>* p_idx,
    device atomic<uint>* p_dist,
    device atomic<uint>* p_cnt,
    uint to,
    uint other,
    float d,
    uint mp
) {
    uint slot = atomic_fetch_add_explicit(&p_cnt[to], 1u, memory_order_relaxed);
    if (slot < mp) {
        ulong off = (ulong)to * mp + slot;
        atomic_store_explicit(&p_idx[off], other, memory_order_relaxed);
        atomic_store_explicit(&p_dist[off], as_type<uint>(d), memory_order_relaxed);
    }
}
"#;

/// Random graph initialisation, one thread per node. Draws `K` distinct random
/// neighbours, keeps them sorted in registers, writes them all flagged new.
/// `params = [n, seed]`.
const INIT_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    const device float4* v = (const device float4*)vecs;
    uint li[K];
    float ld[K];
    uint rng = nnd_xorshift(node ^ params[1] ^ 0xDEADBEEFu);
    for (int slot = 0; slot < K; slot++) {
        rng = nnd_xorshift(rng);
        // Probe past self and earlier draws. The wgpu init only skips self,
        // and a repeated draw that is a true neighbour never leaves the row.
        uint pid = rng % n;
        for (uint t = 0; t < n; t++) {
            bool clash = pid == node;
            for (int j = 0; j < slot; j++) {
                clash = clash || li[j] == pid;
            }
            if (!clash) {
                break;
            }
            pid = (pid + 1u) % n;
        }
        float d = nnd_dist<D4, COS>(v, node, pid);
        if (COS) {
            d = 1.0f - d / (norms[node] * norms[pid]);
        }
        int pos = slot;
        while (pos > 0 && d < ld[pos - 1]) {
            li[pos] = li[pos - 1];
            ld[pos] = ld[pos - 1];
            pos--;
        }
        li[pos] = pid;
        ld[pos] = d;
    }
    ulong base = (ulong)node * K;
    for (int j = 0; j < K; j++) {
        g_idx_out[base + j] = li[j] | 0x80000000u;
        g_dist_out[base + j] = ld[j];
    }
"#;

/// Scatter forward edges into reverse lists, one thread per node, keeping the
/// forward edge's is-new flag. Atomic outputs. `params = [n, ...]`.
const REVERSE_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    ulong base = (ulong)node * K;
    for (int i = 0; i < K; i++) {
        uint raw = g_idx[base + i];
        uint t = raw & 0x7FFFFFFFu;
        if (t < n && t != node) {
            uint pos = atomic_fetch_add_explicit(&rev_cnt[t], 1u, memory_order_relaxed);
            if (pos < (uint)K) {
                atomic_store_explicit(
                    &rev_idx[(ulong)t * K + pos],
                    node | (raw & 0x80000000u),
                    memory_order_relaxed);
            }
        }
    }
"#;

/// The local join, one threadgroup of `TX * TY` threads per node.
///
/// Loads forward and reverse candidates, compacts them on one thread under the
/// rho sampling and the per-kind cap (forward first), then evaluates every
/// (new, new) and (new, old) pair with the vectors staged in threadgroup
/// memory, `BLOCK` candidates per buffer. One block is the unblocked case;
/// otherwise the block pairs `bi <= bj` cover the upper triangle once, exactly
/// as on the wgpu path. Unlike wgpu it does not clear the sampled is-new
/// flags; the merge recomputes and clears them. Atomic outputs.
/// `params = [n, rho_thresh, iter_seed, cand_cap, ...]`.
const LOCAL_JOIN_SOURCE: &str = r#"
    constexpr uint NT = TX * TY;
    constexpr int MAXC = 2 * K;
    uint node = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    uint rho = params[1];
    uint seed = params[2];
    uint cap = params[3];
    uint tx = tid % TX;
    uint ty = tid / TX;
    const device float4* v = (const device float4*)vecs;

    threadgroup uint s_pids[MAXC];
    threadgroup uint s_new[MAXC];
    threadgroup float s_thr[MAXC];
    threadgroup float s_norm[COS ? MAXC : 1];
    threadgroup float4 buf_a[BUF_A];
    threadgroup float4 buf_b[BUF_B];
    threadgroup uint s_meta[2];

    uint rc = min(rev_cnt[node], (uint)K);
    uint raw_total = K + rc;
    for (uint i = tid; i < raw_total; i += NT) {
        uint e = i < (uint)K ? g_idx[(ulong)node * K + i] : rev_idx[(ulong)node * K + i - K];
        s_pids[i] = e & 0x7FFFFFFFu;
        s_new[i] = e >> 31;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint write = 0, has_new = 0, n_new = 0, n_old = 0;
        for (uint r = 0; r < raw_total; r++) {
            uint pid = s_pids[r];
            if ((nnd_entry_hash(node, r, seed) & 0xFFFFu) < rho && pid < n) {
                uint is_new = s_new[r];
                bool keep = false;
                if (is_new != 0) {
                    if (n_new < cap) { keep = true; n_new++; }
                } else {
                    if (n_old < cap) { keep = true; n_old++; }
                }
                if (keep) {
                    s_pids[write] = pid;
                    s_new[write] = is_new;
                    has_new |= is_new;
                    write++;
                }
            }
        }
        s_meta[0] = write;
        s_meta[1] = has_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = s_meta[0];
    if (total < 2 || s_meta[1] == 0) {
        return;
    }
    for (uint i = tid; i < total; i += NT) {
        uint p = s_pids[i];
        s_thr[i] = g_dist[(ulong)p * K + K - 1];
        if (COS) {
            s_norm[i] = norms[p];
        }
    }

    uint n_blocks = (total + BLOCK - 1) / BLOCK;
    for (uint bi = 0; bi < n_blocks; bi++) {
        uint base_i = bi * BLOCK;
        uint len_i = min((uint)BLOCK, total - base_i);
        // Also orders the metadata writes above before the first pair read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < len_i * D4; idx += NT) {
            uint r = idx / D4;
            uint l = idx - r * D4;
            buf_a[r * ROW + l] = v[(ulong)s_pids[base_i + r] * D4 + l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint ii = ty; ii < len_i; ii += TY) {
            uint ai = base_i + ii;
            uint new_i = s_new[ai];
            uint pi = s_pids[ai];
            float thi = s_thr[ai];
            for (uint jj = ii + 1 + tx; jj < len_i; jj += TX) {
                uint aj = base_i + jj;
                uint pj = s_pids[aj];
                if ((new_i | s_new[aj]) != 0 && pi != pj) {
                    float d = nnd_pair<D4, COS>(buf_a + ii * ROW, buf_a + jj * ROW);
                    if (COS) {
                        d = 1.0f - d / (s_norm[ai] * s_norm[aj]);
                    }
                    if (d < thi) {
                        nnd_emit(p_idx, p_dist, p_cnt, pi, pj, d, MP);
                    }
                    if (d < s_thr[aj]) {
                        nnd_emit(p_idx, p_dist, p_cnt, pj, pi, d, MP);
                    }
                }
            }
        }

        for (uint bj = bi + 1; bj < n_blocks; bj++) {
            uint base_j = bj * BLOCK;
            uint len_j = min((uint)BLOCK, total - base_j);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < len_j * D4; idx += NT) {
                uint r = idx / D4;
                uint l = idx - r * D4;
                buf_b[r * ROW + l] = v[(ulong)s_pids[base_j + r] * D4 + l];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint oi = ty; oi < len_i; oi += TY) {
                uint ai = base_i + oi;
                uint new_i = s_new[ai];
                uint pi = s_pids[ai];
                float thi = s_thr[ai];
                for (uint oj = tx; oj < len_j; oj += TX) {
                    uint aj = base_j + oj;
                    uint pj = s_pids[aj];
                    if ((new_i | s_new[aj]) != 0 && pi != pj) {
                        float d = nnd_pair<D4, COS>(buf_a + oi * ROW, buf_b + oj * ROW);
                        if (COS) {
                            d = 1.0f - d / (s_norm[ai] * s_norm[aj]);
                        }
                        if (d < thi) {
                            nnd_emit(p_idx, p_dist, p_cnt, pi, pj, d, MP);
                        }
                        if (d < s_thr[aj]) {
                            nnd_emit(p_idx, p_dist, p_cnt, pj, pi, d, MP);
                        }
                    }
                }
            }
        }
    }
"#;

/// Proposal merge, one SIMD group per node, writing a fresh row. A port of
/// `merge_proposals_coop`: filter survivors (beat the worst, not self, not in
/// the row, not an earlier proposal's duplicate), rank row entries and
/// survivors in the merged order, write the first `K`. With `params[4] != 0`
/// it first clears the is-new flag on the forward entries the local join
/// sampled, by replaying the join's sampling for the forward row.
/// `changed[node]` is the number of survivors that landed in the row.
/// `params = [n, rho_thresh, iter_seed, cand_cap, clear]`.
const MERGE_SOURCE: &str = r#"
    uint node = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    threadgroup uint r_idx[K];
    threadgroup float r_dist[K];
    threadgroup uint q_idx[MP];
    threadgroup float q_dist[MP];
    threadgroup uint keep[MP];
    threadgroup uint lane_kept[32];

    ulong base = (ulong)node * K;
    ulong pbase = (ulong)node * MP;
    for (uint i = lane; i < (uint)K; i += 32) {
        r_idx[i] = g_idx[base + i];
        r_dist[i] = g_dist[base + i];
    }
    uint pn = min(p_cnt[node], (uint)MP);
    for (uint p = lane; p < pn; p += 32) {
        q_idx[p] = p_idx[pbase + p];
        q_dist[p] = as_type<float>(p_dist[pbase + p]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (params[4] != 0 && lane == 0) {
        uint rho = params[1];
        uint seed = params[2];
        uint cap = params[3];
        uint n_new = 0;
        for (uint r = 0; r < (uint)K; r++) {
            uint e = r_idx[r];
            if ((e >> 31) != 0
                && (e & 0x7FFFFFFFu) < n
                && (nnd_entry_hash(node, r, seed) & 0xFFFFu) < rho
                && n_new < cap) {
                r_idx[r] = e & 0x7FFFFFFFu;
                n_new++;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float worst = r_dist[K - 1];
    for (uint p = lane; p < pn; p += 32) {
        uint cand = q_idx[p];
        uint k = (q_dist[p] < worst && cand != node) ? 1u : 0u;
        if (k) {
            for (uint j = 0; j < (uint)K; j++) {
                if ((r_idx[j] & 0x7FFFFFFFu) == cand) { k = 0; }
            }
            for (uint q = 0; q < p; q++) {
                if (q_idx[q] == cand) { k = 0; }
            }
        }
        keep[p] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row entries: shifted down by every survivor strictly closer.
    uint row_kept = 0;
    for (uint i = lane; i < (uint)K; i += 32) {
        float d = r_dist[i];
        uint r = i;
        for (uint q = 0; q < pn; q++) {
            if (keep[q] && q_dist[q] < d) { r++; }
        }
        if (r < (uint)K) {
            o_idx[base + r] = r_idx[i];
            o_dist[base + r] = d;
            row_kept++;
        }
    }
    // Survivors: after every row entry at or below their distance, and after
    // every earlier survivor at the same distance.
    for (uint p = lane; p < pn; p += 32) {
        if (keep[p]) {
            float d = q_dist[p];
            uint lo = 0, hi = K;
            while (lo < hi) {
                uint mid = (lo + hi) / 2;
                if (r_dist[mid] <= d) { lo = mid + 1; } else { hi = mid; }
            }
            uint r = lo;
            for (uint q = 0; q < pn; q++) {
                if (keep[q] && (q_dist[q] < d || (q_dist[q] == d && q < p))) { r++; }
            }
            if (r < (uint)K) {
                o_idx[base + r] = q_idx[p] | 0x80000000u;
                o_dist[base + r] = d;
            }
        }
    }
    lane_kept[lane] = row_kept;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint kept = 0;
        for (int l = 0; l < 32; l++) { kept += lane_kept[l]; }
        changed[node] = (uint)K - kept;
    }
"#;

/// Two-hop refinement, one SIMD group per node: every neighbour's neighbour
/// that is not already in the row and beats the worst becomes a proposal.
/// Atomic outputs. `params = [n, ...]`.
const TWO_HOP_SOURCE: &str = r#"
    uint node = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    const device float4* v = (const device float4*)vecs;
    threadgroup float4 src[D4];
    threadgroup uint own[K];
    ulong base = (ulong)node * K;
    for (uint l = lane; l < (uint)D4; l += 32) {
        src[l] = v[(ulong)node * D4 + l];
    }
    for (uint i = lane; i < (uint)K; i += 32) {
        own[i] = g_idx[base + i] & 0x7FFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float worst = g_dist[base + K - 1];
    float node_norm = COS ? norms[node] : 1.0f;
    for (uint c = lane; c < (uint)(K * K); c += 32) {
        uint n1 = c / K;
        uint n2 = c - n1 * K;
        uint p1 = own[n1];
        if (p1 >= n) { continue; }
        uint cand = g_idx[(ulong)p1 * K + n2] & 0x7FFFFFFFu;
        if (cand >= n || cand == node) { continue; }
        bool dup = false;
        for (int j = 0; j < K; j++) {
            if (own[j] == cand) { dup = true; }
        }
        if (dup) { continue; }
        const device float4* pc = v + (ulong)cand * D4;
        float4 acc = 0.0f;
        for (int l = 0; l < D4; l++) {
            if (COS) {
                acc = fma(src[l], pc[l], acc);
            } else {
                float4 d = src[l] - pc[l];
                acc = fma(d, d, acc);
            }
        }
        float s = acc.x + acc.y + acc.z + acc.w;
        float d = COS ? 1.0f - s / (node_norm * norms[cand]) : s;
        if (d < worst) {
            nnd_emit(p_idx, p_dist, p_cnt, node, cand, d, MP);
        }
    }
"#;

/// CAGRA step 1, rank-based detour pruning, one SIMD group per node. An edge
/// to the `i`-th neighbour `y` has a detour through every earlier neighbour
/// `z` whose own first `i` entries contain `y`. Thread 0 selection-sorts by
/// `(detours, rank)` and keeps the first `D`. `params = [n, ...]`.
const CAGRA_PRUNE_SOURCE: &str = r#"
    uint node = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    threadgroup uint nb[K];
    threadgroup uint det[K];
    ulong base = (ulong)node * K;
    for (uint i = lane; i < (uint)K; i += 32) {
        nb[i] = g_idx[base + i] & 0x7FFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < (uint)K; i += 32) {
        uint y = nb[i];
        uint detours = 0;
        for (uint j = 0; j < i; j++) {
            uint z = nb[j];
            // Sentinel guard; the wgpu kernel reads past the end on one.
            if (z >= n) { continue; }
            bool found = false;
            for (uint m = 0; m < i; m++) {
                if ((g_idx[(ulong)z * K + m] & 0x7FFFFFFFu) == y) { found = true; }
            }
            if (found) { detours++; }
        }
        det[i] = (detours << 16) | i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint step = 0; step < (uint)D; step++) {
            uint min_val = 0xFFFFFFFFu;
            uint min_idx = 0;
            for (uint s = step; s < (uint)K; s++) {
                uint val = det[s];
                if (val < min_val) { min_val = val; min_idx = s; }
            }
            uint tmp = det[step];
            det[step] = det[min_idx];
            det[min_idx] = tmp;
            pruned[(ulong)node * D + step] = nb[min_val & 0xFFFFu];
        }
    }
"#;

/// CAGRA step 2, reverse edges of the pruned graph, one thread per node.
/// Overflow past `D` is dropped. Atomic outputs. `params = [n, ...]`.
const CAGRA_REVERSE_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    for (int i = 0; i < D; i++) {
        uint t = pruned[(ulong)node * D + i];
        if (t < n) {
            uint pos = atomic_fetch_add_explicit(&rev_cnt[t], 1u, memory_order_relaxed);
            if (pos < (uint)D) {
                atomic_store_explicit(&rev_idx[(ulong)t * D + pos], node, memory_order_relaxed);
            }
        }
    }
"#;

/// CAGRA step 3, one thread per node: up to `D / 2` reverse edges first, then
/// the pruned forward edges, deduplicated, sentinel padded.
/// `params = [n, ...]`.
const CAGRA_MERGE_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = params[0];
    if (node >= n) {
        return;
    }
    ulong base = (ulong)node * D;
    uint take_rev = min(rev_cnt[node], (uint)(D / 2));
    uint count = 0;
    for (uint i = 0; i < take_rev; i++) {
        final_idx[base + count] = rev_idx[base + i];
        count++;
    }
    for (uint j = 0; j < (uint)D && count < (uint)D; j++) {
        uint cand = pruned[base + j];
        bool dup = false;
        for (uint c = 0; c < count; c++) {
            if (final_idx[base + c] == cand) { dup = true; }
        }
        if (!dup) {
            final_idx[base + count] = cand;
            count++;
        }
    }
    for (; count < (uint)D; count++) {
        final_idx[base + count] = 0x7FFFFFFFu;
    }
"#;

////////////////////
// Staging plans  //
////////////////////

/// Threadgroup staging plan for the local join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalJoinPlanMlx {
    /// Candidates staged per vector buffer
    pub block: usize,
    /// Stride between staged rows in `float4`s, with the bank padding
    pub row: usize,
    /// Length of buffer A in `float4`s
    pub buf_a: usize,
    /// Length of buffer B in `float4`s; 1 (never touched) when one block holds
    /// every candidate
    pub buf_b: usize,
}

/// Threadgroup bytes the local join spends on everything except the staged
/// vectors: pids, flags and thresholds per candidate, norms under cosine,
/// two meta words.
///
/// ### Params
///
/// * `build_k` - Working degree; candidates per node are `2 * build_k`
/// * `use_cosine` - Whether the norm buffer is full width
///
/// ### Returns
///
/// Bytes
fn local_join_meta_bytes(build_k: usize, use_cosine: bool) -> usize {
    let max_c = 2 * build_k;
    let norm_len = if use_cosine { max_c } else { 1 };
    max_c * 12 + norm_len * 4 + 8
}

/// Plan how the local join stages candidate vectors in threadgroup memory.
///
/// ### Params
///
/// * `dim_padded` - Padded dimensionality, a multiple of 4
/// * `build_k` - Working degree of the graph
/// * `use_cosine` - Whether the kernel takes its cosine arm
/// * `budget` - Threadgroup bytes available
///
/// ### Returns
///
/// The plan, or `DimTooHighForSharedMemory` when two rows do not fit
pub(crate) fn plan_local_join_mlx(
    dim_padded: usize,
    build_k: usize,
    use_cosine: bool,
    budget: usize,
) -> Result<LocalJoinPlanMlx, AnnSearchErrors> {
    let d4 = dim_padded / 4;
    let row = d4 + usize::from(d4 <= LOCAL_JOIN_PAD_MAX_LINES);
    let row_bytes = row * 16;
    let meta = local_join_meta_bytes(build_k, use_cosine);
    let avail = budget.saturating_sub(meta);
    if avail < 2 * row_bytes {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: meta + 2 * row_bytes,
            available: budget,
        });
    }
    let max_joined = (2 * MLX_NND_MAX_CANDIDATES).min(2 * build_k);
    // The one-float4 dummy buffer B still counts.
    if max_joined * row_bytes + 16 <= avail {
        return Ok(LocalJoinPlanMlx {
            block: max_joined,
            row,
            buf_a: max_joined * row,
            buf_b: 1,
        });
    }
    let block = (avail / (2 * row_bytes)).min(max_joined);
    Ok(LocalJoinPlanMlx {
        block,
        row,
        buf_a: block * row,
        buf_b: block * row,
    })
}

/// Threadgroup bytes of the merge kernel: row and proposals with keep flags,
/// one counter per lane.
///
/// ### Params
///
/// * `build_k` - Working degree
///
/// ### Returns
///
/// Bytes
pub(crate) fn merge_tg_bytes(build_k: usize) -> usize {
    build_k * 8 + MLX_MAX_PROPOSALS * 12 + 32 * 4
}

/// Error unless a footprint fits the threadgroup budget.
///
/// ### Params
///
/// * `bytes` - Footprint
/// * `dim_padded` - Reported in the error
///
/// ### Returns
///
/// `Ok(())` or `DimTooHighForSharedMemory`
pub(crate) fn check_tg_fit(bytes: usize, dim_padded: usize) -> Result<(), AnnSearchErrors> {
    if bytes > MLX_TG_BYTES {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: bytes,
            available: MLX_TG_BYTES,
        });
    }
    Ok(())
}

/////////////
// Helpers //
/////////////

/// Spec of an `[rows, cols]` u32 output.
///
/// ### Params
///
/// * `shape` - The shape, borrowed by the spec
///
/// ### Returns
///
/// The spec
pub(crate) fn u32_out(shape: &[i32]) -> OutputSpec<'_> {
    OutputSpec {
        shape,
        dtype: MLX_UINT32,
    }
}

/// Spec of an f32 output.
///
/// ### Params
///
/// * `shape` - The shape, borrowed by the spec
///
/// ### Returns
///
/// The spec
pub(crate) fn f32_out(shape: &[i32]) -> OutputSpec<'_> {
    OutputSpec {
        shape,
        dtype: MLX_FLOAT32,
    }
}

/// Pop the outputs of a kernel into a fixed-size array.
///
/// ### Params
///
/// * `v` - Outputs as returned by `apply`
///
/// ### Returns
///
/// The outputs in order
pub(crate) fn outs<const N: usize>(v: Vec<Array>) -> [Array; N] {
    v.try_into()
        .unwrap_or_else(|_| unreachable!("kernel output count is fixed"))
}

/// Pad row-major vectors to a multiple of 4 columns with zeros.
///
/// ### Params
///
/// * `data` - Row-major vectors
/// * `n` - Rows
/// * `dim` - Row length
///
/// ### Returns
///
/// `(padded, dim_padded)`
fn pad_rows(data: &[f32], n: usize, dim: usize) -> (Vec<f32>, usize) {
    let dim_padded = dim.next_multiple_of(4);
    if dim_padded == dim {
        return (data.to_vec(), dim);
    }
    let mut out = vec![0.0f32; n * dim_padded];
    out.par_chunks_exact_mut(dim_padded)
        .zip(data.par_chunks_exact(dim))
        .for_each(|(o, i)| o[..dim].copy_from_slice(i));
    (out, dim_padded)
}

/// Compact the `build_k`-wide working graph to `k` neighbours per node: drop
/// self-edges and sentinels, keep the first `k`, sort each row by distance.
///
/// ### Params
///
/// * `graph_idx` - Raw ids with the is-new flag, `n * build_k`
/// * `graph_dist` - Matching distances
/// * `n` - Nodes
/// * `k` - Neighbours to keep
/// * `build_k` - Working degree
///
/// ### Returns
///
/// Flat `n * k` graph, unfilled slots `(SENTINEL_PID, f32::MAX)`
pub fn compact_knn_rows_mlx(
    graph_idx: &[u32],
    graph_dist: &[f32],
    n: usize,
    k: usize,
    build_k: usize,
) -> Vec<(usize, f32)> {
    let mut knn_graph = vec![(SENTINEL_PID, f32::MAX); n * k];
    knn_graph
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, slot)| {
            let mut written = 0;
            for j in 0..build_k {
                if written >= k {
                    break;
                }
                let pid = (graph_idx[i * build_k + j] & MLX_SENTINEL) as usize;
                if pid < n && pid != i {
                    slot[written] = (pid, graph_dist[i * build_k + j]);
                    written += 1;
                }
            }
            slot.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        });
    knn_graph
}

/////////////////
// NND context //
/////////////////

/// Device state and compiled kernels for one NN-Descent build.
///
/// Holds an MLX stream, so it is thread affine and `!Send`.
pub(crate) struct NndMlx {
    /// Padded vectors `[n, dim_padded]`
    pub vecs: Array,
    /// L2 norms `[n]` under cosine, a dummy `[2]` otherwise
    pub norms: Array,
    /// Number of nodes
    pub n: usize,
    /// `dim_padded / 4`
    pub d4: usize,
    /// Working degree
    pub build_k: usize,
    /// Cosine rather than squared Euclidean
    pub use_cosine: bool,
    /// See [`INIT_SOURCE`]
    init_k: MetalKernel,
    /// See [`REVERSE_SOURCE`]
    reverse_k: MetalKernel,
    /// See [`LOCAL_JOIN_SOURCE`]
    join_k: MetalKernel,
    /// See [`MERGE_SOURCE`]
    merge_k: MetalKernel,
    /// See [`TWO_HOP_SOURCE`]
    two_hop_k: MetalKernel,
    /// Stream every op runs on. Declared last so it drops after the arrays.
    pub stream: Stream,
}

/// Graph pair `(idx, dist)`, both `[n, build_k]`.
pub(crate) type GraphPair = (Array, Array);

/// Proposal triple `(idx, dist bits, count)`.
pub(crate) type Proposals = (Array, Array, Array);

impl NndMlx {
    /// Upload the vectors and compile the kernels.
    ///
    /// ### Params
    ///
    /// * `vectors_padded` - Row-major vectors, `dim_padded` a multiple of 4
    /// * `norms` - L2 norms; empty unless `use_cosine`
    /// * `n` - Rows
    /// * `dim_padded` - Padded row length
    /// * `build_k` - Working degree
    /// * `use_cosine` - Cosine rather than squared Euclidean
    ///
    /// ### Returns
    ///
    /// The context
    pub fn new(
        vectors_padded: &[f32],
        norms: &[f32],
        n: usize,
        dim_padded: usize,
        build_k: usize,
        use_cosine: bool,
    ) -> Self {
        // MLX passes a one-element input by value (`constant T&`), not as a
        // pointer, so the dummy needs at least two.
        let norms = if use_cosine {
            Array::from_f32(norms, &[n as i32])
        } else {
            Array::from_f32(&[0.0; 2], &[2])
        };
        let kern = |name: &str, ins: &[&str], outs: &[&str], src: &str, atomic: bool| {
            MetalKernel::with_header(name, ins, outs, NND_HEADER, src, atomic)
        };
        Self {
            vecs: Array::from_f32(vectors_padded, &[n as i32, dim_padded as i32]),
            norms,
            n,
            d4: dim_padded / 4,
            build_k,
            use_cosine,
            init_k: kern(
                "nnd_init",
                &["vecs", "norms", "params"],
                &["g_idx_out", "g_dist_out"],
                INIT_SOURCE,
                false,
            ),
            reverse_k: kern(
                "nnd_reverse",
                &["g_idx", "params"],
                &["rev_idx", "rev_cnt"],
                REVERSE_SOURCE,
                true,
            ),
            join_k: kern(
                "nnd_local_join",
                &[
                    "vecs", "norms", "g_idx", "g_dist", "rev_idx", "rev_cnt", "params",
                ],
                &["p_idx", "p_dist", "p_cnt"],
                LOCAL_JOIN_SOURCE,
                true,
            ),
            merge_k: kern(
                "nnd_merge",
                &["g_idx", "g_dist", "p_idx", "p_dist", "p_cnt", "params"],
                &["o_idx", "o_dist", "changed"],
                MERGE_SOURCE,
                false,
            ),
            two_hop_k: kern(
                "nnd_two_hop",
                &["vecs", "norms", "g_idx", "g_dist", "params"],
                &["p_idx", "p_dist", "p_cnt"],
                TWO_HOP_SOURCE,
                true,
            ),
            stream: Stream::default_gpu(),
        }
    }

    /// Params array `[n, rho_thresh, seed, cand_cap, clear]`.
    ///
    /// ### Params
    ///
    /// * `rho_thresh` - Sampling threshold on the 16-bit hash
    /// * `seed` - Iteration seed (or the init seed)
    /// * `clear` - Whether the merge clears the sampled is-new flags
    ///
    /// ### Returns
    ///
    /// The array
    pub fn params(&self, rho_thresh: u32, seed: u32, clear: bool) -> Array {
        let cap = MLX_NND_MAX_CANDIDATES.min(2 * self.build_k) as u32;
        Array::from_u32(&[self.n as u32, rho_thresh, seed, cap, clear as u32], &[5])
    }

    /// Shape `[n, build_k]` as i32.
    ///
    /// ### Returns
    ///
    /// The shape
    fn graph_shape(&self) -> [i32; 2] {
        [self.n as i32, self.build_k as i32]
    }

    /// Template ints every distance kernel takes.
    ///
    /// ### Returns
    ///
    /// `K`, `D4`, `COS`, `MP`
    fn base_templates(&self) -> [(&'static str, i32); 4] {
        [
            ("K", self.build_k as i32),
            ("D4", self.d4 as i32),
            ("COS", self.use_cosine as i32),
            ("MP", MLX_MAX_PROPOSALS as i32),
        ]
    }

    /// Queue the random graph initialisation.
    ///
    /// ### Params
    ///
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// The lazy graph, sorted per row, every entry flagged new
    pub fn init_random(&self, seed: u32) -> Result<GraphPair, AnnSearchErrors> {
        let shape = self.graph_shape();
        let params = self.params(0, seed, false);
        let [idx, dist] = outs(self.init_k.apply(
            &[&self.vecs, &self.norms, &params],
            &[u32_out(&shape), f32_out(&shape)],
            [self.n as i32, 1, 1],
            [NODE_TG, 1, 1],
            &self.base_templates()[..3],
            &self.stream,
        )?);
        Ok((idx, dist))
    }

    /// Queue the reverse-candidate scatter.
    ///
    /// ### Params
    ///
    /// * `g_idx` - Current graph ids
    /// * `params` - Params array, see [`Self::params`]
    ///
    /// ### Returns
    ///
    /// Lazy `(reverse ids [n, build_k], reverse counts [n])`
    fn reverse(&self, g_idx: &Array, params: &Array) -> Result<(Array, Array), AnnSearchErrors> {
        let shape = self.graph_shape();
        let [ri, rc] = outs(self.reverse_k.apply_zeroed(
            &[g_idx, params],
            &[u32_out(&shape), u32_out(&[self.n as i32])],
            [self.n as i32, 1, 1],
            [NODE_TG, 1, 1],
            &[("K", self.build_k as i32)],
            &self.stream,
        )?);
        Ok((ri, rc))
    }

    /// Queue the local join.
    ///
    /// ### Params
    ///
    /// * `g` - Current graph
    /// * `rev` - Reverse candidates from [`Self::reverse`]
    /// * `params` - Params array, see [`Self::params`]
    /// * `plan` - Threadgroup staging plan
    ///
    /// ### Returns
    ///
    /// Lazy proposals
    fn local_join(
        &self,
        g: &GraphPair,
        rev: &(Array, Array),
        params: &Array,
        plan: &LocalJoinPlanMlx,
    ) -> Result<Proposals, AnnSearchErrors> {
        let p_shape = [self.n as i32, MLX_MAX_PROPOSALS as i32];
        let nt = (LOCAL_JOIN_TX * LOCAL_JOIN_TY) as i32;
        let mut t = self.base_templates().to_vec();
        t.extend([
            ("ROW", plan.row as i32),
            ("BLOCK", plan.block as i32),
            ("BUF_A", plan.buf_a as i32),
            ("BUF_B", plan.buf_b as i32),
            ("TX", LOCAL_JOIN_TX as i32),
            ("TY", LOCAL_JOIN_TY as i32),
        ]);
        let [pi, pd, pc] = outs(self.join_k.apply_zeroed(
            &[&self.vecs, &self.norms, &g.0, &g.1, &rev.0, &rev.1, params],
            &[
                u32_out(&p_shape),
                u32_out(&p_shape),
                u32_out(&[self.n as i32]),
            ],
            [nt, self.n as i32, 1],
            [nt, 1, 1],
            &t,
            &self.stream,
        )?);
        Ok((pi, pd, pc))
    }

    /// Queue the proposal merge into a fresh graph.
    ///
    /// ### Params
    ///
    /// * `g` - Current graph
    /// * `p` - Proposals
    /// * `params` - Params array, see [`Self::params`]
    ///
    /// ### Returns
    ///
    /// Lazy `(new graph, per-node insert count [n])`
    pub fn merge(
        &self,
        g: &GraphPair,
        p: &Proposals,
        params: &Array,
    ) -> Result<(GraphPair, Array), AnnSearchErrors> {
        let shape = self.graph_shape();
        let [oi, od, ch] = outs(self.merge_k.apply(
            &[&g.0, &g.1, &p.0, &p.1, &p.2, params],
            &[u32_out(&shape), f32_out(&shape), u32_out(&[self.n as i32])],
            [SIMD, self.n as i32, 1],
            [SIMD, 1, 1],
            &[("K", self.build_k as i32), ("MP", MLX_MAX_PROPOSALS as i32)],
            &self.stream,
        )?);
        Ok(((oi, od), ch))
    }

    /// Queue one two-hop proposal sweep.
    ///
    /// ### Params
    ///
    /// * `g` - Current graph
    /// * `params` - Params array, see [`Self::params`]
    ///
    /// ### Returns
    ///
    /// Lazy proposals
    fn two_hop(&self, g: &GraphPair, params: &Array) -> Result<Proposals, AnnSearchErrors> {
        let p_shape = [self.n as i32, MLX_MAX_PROPOSALS as i32];
        let [pi, pd, pc] = outs(self.two_hop_k.apply_zeroed(
            &[&self.vecs, &self.norms, &g.0, &g.1, params],
            &[
                u32_out(&p_shape),
                u32_out(&p_shape),
                u32_out(&[self.n as i32]),
            ],
            [SIMD, self.n as i32, 1],
            [SIMD, 1, 1],
            &self.base_templates(),
            &self.stream,
        )?);
        Ok((pi, pd, pc))
    }
}

/////////////////////
// Main NND driver //
/////////////////////

/// Resolved NN-Descent knobs for the MLX build.
#[derive(Clone, Copy, Debug)]
pub struct NnDescentCfgMlx {
    /// Working degree, wider than the returned `k`
    pub build_k: usize,
    /// Iteration cap for the main loop
    pub max_iters: usize,
    /// Random-projection trees used to seed the graph
    pub n_trees: usize,
    /// Convergence threshold as a fraction of `n * build_k` edges updated
    pub delta: f32,
    /// Local-join sampling rate, scaled to the kernel's 16-bit hash domain
    pub rho_thresh: u32,
    /// Two-hop refinement sweeps after the main loop
    pub refine_knn: usize,
    /// RNG seed
    pub seed: usize,
    /// Cosine rather than squared Euclidean
    pub use_cosine: bool,
}

/// Run the device-resident NN-Descent loop on MLX and read the raw graph
/// back. Counterpart of `nndescent_core` on the wgpu path.
///
/// ### Params
///
/// * `vectors_flat` - Row-major vectors, `n` rows of `dim` (unpadded)
/// * `norms` - L2 norms per row; only read under cosine, may be empty
///   otherwise
/// * `n` - Rows
/// * `dim` - Row length
/// * `cfg` - Resolved knobs
/// * `verbose` - Print per-phase progress
///
/// ### Returns
///
/// `(graph_idx, graph_dist, converged)`, both `n * build_k`, rows sorted
/// ascending; ids still carry the is-new flag in bit 31
pub fn nndescent_core_mlx(
    vectors_flat: &[f32],
    norms: &[f32],
    n: usize,
    dim: usize,
    cfg: &NnDescentCfgMlx,
    verbose: bool,
) -> Result<(Vec<u32>, Vec<f32>, bool), AnnSearchErrors> {
    install_error_handler();
    let build_k = cfg.build_k;
    let (vectors_padded, dim_padded) = pad_rows(vectors_flat, n, dim);

    let plan = plan_local_join_mlx(dim_padded, build_k, cfg.use_cosine, MLX_TG_BYTES)?;
    check_tg_fit(merge_tg_bytes(build_k), dim_padded)?;
    if cfg.refine_knn > 0 {
        check_tg_fit(dim_padded * 4 + build_k * 4, dim_padded)?;
    }

    let ctx = NndMlx::new(
        &vectors_padded,
        norms,
        n,
        dim_padded,
        build_k,
        cfg.use_cosine,
    );

    if verbose {
        println!("  Random graph initialisation...");
    }
    let mut g = ctx.init_random(cfg.seed as u32)?;

    g = forest_init_mlx(&ctx, g, dim, cfg.n_trees, cfg.seed, verbose)?;
    // No mark-all-new pass: the random init flags every slot new and the
    // forest merges flag what they insert, so every entry is already new.

    let iter_start = Instant::now();
    let mut converged = false;
    for iter in 0..cfg.max_iters {
        let iter_seed = cfg.seed as u32 ^ (iter as u32).wrapping_mul(0x9E3779B9);
        let params = ctx.params(cfg.rho_thresh, iter_seed, true);
        let rev = ctx.reverse(&g.0, &params)?;
        let props = ctx.local_join(&g, &rev, &params, &plan)?;
        let (next, changed) = ctx.merge(&g, &props, &params)?;
        let total = changed.sum_axis(0, false, &ctx.stream)?;
        eval_all(&[&total, &next.0, &next.1], false)?;
        g = next;

        let updates = total.as_u32()?[0] as f64;
        let rate = updates / (n * build_k) as f64;
        if verbose {
            println!(
                "   Iter {}: {} updates (rate={:.6})",
                iter + 1,
                (updates as usize).separate_with_underscores(),
                rate
            );
        }
        if rate < cfg.delta as f64 {
            if verbose {
                println!("  Converged after {} iterations", iter + 1);
            }
            converged = true;
            break;
        }
    }
    if verbose {
        println!("  NNDescent iterations: {:.2?}", iter_start.elapsed());
    }

    for sweep in 0..cfg.refine_knn {
        let params = ctx.params(0, 0, false);
        let props = ctx.two_hop(&g, &params)?;
        let (next, changed) = ctx.merge(&g, &props, &params)?;
        g = next;
        if verbose {
            let total = changed.sum_axis(0, false, &ctx.stream)?;
            eval_all(&[&total], false)?;
            println!(
                "    2-Hop sweep {}: {} updates",
                sweep + 1,
                total.as_u32()?[0].separate_with_underscores()
            );
        }
    }

    eval_all(&[&g.0, &g.1], false)?;
    Ok((g.0.as_u32()?.to_vec(), g.1.as_f32()?.to_vec(), converged))
}

/////////////////
// KnnGraphMlx //
/////////////////

/// Raw kNN graph built on MLX. Same fields as `KnnGraphGpu`, which lives
/// behind the `gpu` feature.
pub struct KnnGraphMlx {
    /// Original (unpadded) vector data, flattened row-major
    pub vectors_flat: Vec<f32>,
    /// Original embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<f32>,
    /// Distance metric
    pub metric: Dist,
    /// Flat kNN graph of size `n * k`, sorted per row ascending by distance
    pub knn_graph: Vec<(usize, f32)>,
    /// Whether NN-Descent hit the delta convergence threshold
    pub converged: bool,
}

impl KnnGraphMlx {
    /// Hand back the kNN graph as per-node rows.
    ///
    /// ### Params
    ///
    /// * `k` - Truncate each row to this total length, self-edge included
    ///   when `include_self` is set. `None` keeps the build-time `k`.
    /// * `include_self` - Prepend `(i, 0)` to row `i`
    /// * `return_dist` - Whether to materialise the distances
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)`, ascending; sentinel slots dropped
    pub fn extract_knn(
        &self,
        k: Option<usize>,
        include_self: bool,
        return_dist: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>) {
        unpack_knn_graph(
            &self.knn_graph,
            self.n,
            self.k,
            k,
            include_self,
            return_dist,
        )
    }

    /// Returns the CPU-side memory footprint of the graph in bytes.
    ///
    /// ### Returns
    ///
    /// Bytes allocated for this struct and its owned Vecs
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * size_of::<f32>()
            + self.norms.capacity() * size_of::<f32>()
            + self.knn_graph.capacity() * size_of::<(usize, f32)>()
    }
}

/// Build a raw kNN graph with NN-Descent on MLX. Parameters mirror
/// `build_knn_graph_gpu`, minus the device.
///
/// ### Params
///
/// * `data` - Row-major sample matrix
/// * `metric` - Distance metric (Manhattan is rejected)
/// * `k` - Neighbours per node in the returned graph. Defaults to 30
/// * `build_k` - Working degree. Defaults to `max(k, 1.5 * k)`
/// * `max_iters` - Maximum iterations. Defaults to 15
/// * `n_trees` - Forest size. Defaults to `5 + n^0.25`, capped at 20
/// * `delta` - Convergence threshold. Defaults to 0.001
/// * `rho` - Local-join sampling rate. Defaults to 1.0
/// * `refine_knn` - Two-hop sweeps after the loop. Defaults to 0
/// * `seed` - RNG seed
/// * `verbose` - Print per-phase progress
///
/// ### Returns
///
/// Populated [`KnnGraphMlx`]
#[allow(clippy::too_many_arguments)]
pub fn build_knn_graph_mlx(
    data: impl AnnMatrix<f32>,
    metric: Dist,
    k: Option<usize>,
    build_k: Option<usize>,
    max_iters: Option<usize>,
    n_trees: Option<usize>,
    delta: Option<f32>,
    rho: Option<f32>,
    refine_knn: Option<usize>,
    seed: usize,
    verbose: bool,
) -> Result<KnnGraphMlx, AnnSearchErrors> {
    if metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(metric));
    }
    let (vectors_flat, n, dim) = data.into_row_major();
    let k = k.unwrap_or(30);
    let build_k = build_k.unwrap_or((1.5 * k as f32) as usize).max(k);
    let use_cosine = metric == Dist::Cosine;

    let norms: Vec<f32> = if use_cosine {
        vectors_flat
            .par_chunks_exact(dim)
            .map(f32::calculate_l2_norm)
            .collect()
    } else {
        Vec::new()
    };

    if verbose {
        println!(
            "kNN-Graph-MLX: {} vectors, dim={}, k={}, build_k={}",
            n.separate_with_underscores(),
            dim,
            k,
            build_k,
        );
    }
    let start = Instant::now();

    let cfg = NnDescentCfgMlx {
        build_k,
        max_iters: max_iters.unwrap_or(DEFAULT_MAX_ITERS),
        n_trees: n_trees.unwrap_or_else(|| (5 + (n as f64).powf(0.25).round() as usize).min(20)),
        delta: delta.unwrap_or(DEFAULT_DELTA),
        rho_thresh: (rho.unwrap_or(DEFAULT_RHO) * 65535.0) as u32,
        refine_knn: refine_knn.unwrap_or(0),
        seed,
        use_cosine,
    };

    let (graph_idx, graph_dist, converged) =
        nndescent_core_mlx(&vectors_flat, &norms, n, dim, &cfg, verbose)?;
    let knn_graph = compact_knn_rows_mlx(&graph_idx, &graph_dist, n, k, build_k);

    if verbose {
        println!("  Total build time: {:.2?}", start.elapsed());
    }

    Ok(KnnGraphMlx {
        vectors_flat,
        dim,
        n,
        k,
        norms,
        metric,
        knn_graph,
        converged,
    })
}

///////////
// CAGRA //
///////////

/// CAGRA graph optimisation on MLX: rank-based detour pruning from `build_k`
/// to `k`, reverse edges, merge. Same three kernels and semantics as the tail
/// of `NNDescentGpu::build`.
///
/// ### Params
///
/// * `graph_idx` - Raw NN-Descent graph `n * build_k`, rows ascending by
///   distance, is-new flags allowed (as returned by [`nndescent_core_mlx`])
/// * `vectors_flat` - Row-major vectors, for the medoid
/// * `n` - Nodes
/// * `dim` - Row length of `vectors_flat`
/// * `build_k` - Row width of `graph_idx`
/// * `k` - Degree of the navigational graph, at most `build_k`
/// * `metric` - Metric, for the medoid
///
/// ### Returns
///
/// `(nav_graph, medoid)`: `nav_graph` is `n * k` raw node ids with
/// `0x7FFFFFFF` in unfilled slots, the layout `NNDescentGpu`'s beam search
/// reads; `medoid` is the entry point. `MlxGraphShape` when `graph_idx` is
/// not `n * build_k` or `k > build_k`
pub fn cagra_optimise_mlx(
    graph_idx: &[u32],
    vectors_flat: &[f32],
    n: usize,
    dim: usize,
    build_k: usize,
    k: usize,
    metric: Dist,
) -> Result<(Vec<u32>, u32), AnnSearchErrors> {
    install_error_handler();
    if k > build_k || graph_idx.len() != n * build_k {
        return Err(AnnSearchErrors::MlxGraphShape {
            len: graph_idx.len(),
            n,
            build_k,
            k,
        });
    }
    check_tg_fit(2 * build_k * 4, dim)?;

    let s = Stream::default_gpu();
    let kern = |name: &str, ins: &[&str], outs: &[&str], src: &str, atomic: bool| {
        MetalKernel::with_header(name, ins, outs, "", src, atomic)
    };
    let prune_k = kern(
        "cagra_prune",
        &["g_idx", "params"],
        &["pruned"],
        CAGRA_PRUNE_SOURCE,
        false,
    );
    let reverse_k = kern(
        "cagra_reverse",
        &["pruned", "params"],
        &["rev_idx", "rev_cnt"],
        CAGRA_REVERSE_SOURCE,
        true,
    );
    let merge_k = kern(
        "cagra_merge",
        &["pruned", "rev_idx", "rev_cnt", "params"],
        &["final_idx"],
        CAGRA_MERGE_SOURCE,
        false,
    );

    let g = Array::from_u32(graph_idx, &[n as i32, build_k as i32]);
    // Two elements: a one-element input would arrive by value, not pointer.
    let params = Array::from_u32(&[n as u32, 0], &[2]);
    let nav_shape = [n as i32, k as i32];
    let d = [("D", k as i32)];

    let [pruned] = outs(prune_k.apply(
        &[&g, &params],
        &[u32_out(&nav_shape)],
        [SIMD, n as i32, 1],
        [SIMD, 1, 1],
        &[("K", build_k as i32), ("D", k as i32)],
        &s,
    )?);
    let [rev_idx, rev_cnt] = outs(reverse_k.apply_zeroed(
        &[&pruned, &params],
        &[u32_out(&nav_shape), u32_out(&[n as i32])],
        [n as i32, 1, 1],
        [NODE_TG, 1, 1],
        &d,
        &s,
    )?);
    let [nav] = outs(merge_k.apply(
        &[&pruned, &rev_idx, &rev_cnt, &params],
        &[u32_out(&nav_shape)],
        [n as i32, 1, 1],
        [NODE_TG, 1, 1],
        &d,
        &s,
    )?);
    eval_all(&[&nav], false)?;
    let nav_graph = nav.as_u32()?.to_vec();

    let medoid = compute_medoid(vectors_flat, n, dim, metric);
    Ok((nav_graph, medoid))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exhaustive::ExhaustiveIndex;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Clustered data: `n` points around 20 random centres.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// The matrix
    fn clustered(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect())
            .collect();
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..20)).collect();
        Mat::from_fn(n, dim, |i, j| {
            centres[labels[i]][j] + rng.random_range(-1.0f32..1.0)
        })
    }

    /// Recall of `approx` (no self) against exhaustive `k + 1` with the self
    /// hit dropped.
    ///
    /// ### Params
    ///
    /// * `data` - The data
    /// * `metric` - Metric
    /// * `approx` - Neighbour rows under test, no self
    /// * `k` - Neighbours per row
    ///
    /// ### Returns
    ///
    /// Recall in `[0, 1]`
    fn recall(data: &Mat<f32>, metric: Dist, approx: &[Vec<usize>], k: usize) -> f64 {
        let cpu = ExhaustiveIndex::new(data.as_ref(), metric);
        let (truth, _) = cpu.generate_knn(k + 1, false, false).unwrap();
        let mut hits = 0;
        for (i, (t, a)) in truth.iter().zip(approx).enumerate() {
            let t: Vec<usize> = t.iter().copied().filter(|&j| j != i).take(k).collect();
            hits += a.iter().filter(|j| t.contains(j)).count();
        }
        hits as f64 / (approx.len() * k) as f64
    }

    #[test]
    fn test_mlx_plan_local_join_fits_budget() {
        for dim in [4usize, 32, 64, 128, 256, 512, 1024] {
            for build_k in [15usize, 45, 90, 150] {
                for cos in [false, true] {
                    let p = plan_local_join_mlx(dim, build_k, cos, MLX_TG_BYTES).unwrap();
                    let bytes = local_join_meta_bytes(build_k, cos) + (p.buf_a + p.buf_b) * 16;
                    assert!(bytes <= MLX_TG_BYTES, "dim {dim} k {build_k}: {bytes}");
                    assert!(p.block >= 1 && p.block <= 2 * build_k);
                    assert_eq!(p.buf_a, p.block * p.row);
                }
            }
        }
        // dim 32, build_k 45: 90 rows of 9 float4s fit in one block.
        let p = plan_local_join_mlx(32, 45, false, MLX_TG_BYTES).unwrap();
        assert_eq!((p.block, p.buf_b), (90, 1));
        assert!(plan_local_join_mlx(8192, 45, false, MLX_TG_BYTES).is_err());
    }

    #[test]
    fn test_mlx_merge_footprint() {
        assert_eq!(merge_tg_bytes(45), 45 * 8 + 128 * 12 + 128);
        assert!(merge_tg_bytes(150) <= MLX_TG_BYTES);
    }

    #[test]
    fn test_mlx_knn_graph_rejects_manhattan() {
        let data = clustered(100, 8, 1);
        let res = build_knn_graph_mlx(
            data.as_ref(),
            Dist::Manhattan,
            Some(5),
            None,
            None,
            None,
            None,
            None,
            None,
            42,
            false,
        );
        assert!(res.is_err());
    }

    /// Build on MLX and check recall against exhaustive ground truth.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric
    /// * `dim` - Columns (not a multiple of 4 exercises the padding)
    /// * `refine` - Two-hop sweeps
    ///
    /// ### Returns
    ///
    /// The recall
    fn mlx_recall(metric: Dist, dim: usize, refine: usize) -> f64 {
        let (n, k) = (3_000, 15);
        let data = clustered(n, dim, 11);
        let g = build_knn_graph_mlx(
            data.as_ref(),
            metric,
            Some(k),
            None,
            None,
            None,
            None,
            None,
            Some(refine),
            42,
            false,
        )
        .unwrap();
        let (idx, _) = g.extract_knn(None, false, false);
        for (i, row) in idx.iter().enumerate() {
            assert_eq!(row.len(), k);
            assert!(!row.contains(&i));
            assert!(row.iter().all(|&j| j < n));
            let mut ids = row.clone();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), k, "row {i} has duplicates");
        }
        let r = recall(&data, metric, &idx, k);
        println!("MLX recall {metric:?} dim {dim} refine {refine}: {r:.4}");
        r
    }

    #[test]
    fn test_mlx_knn_graph_recall_euclidean() {
        let r = mlx_recall(Dist::SquaredEuclidean, 30, 0);
        assert!(r > 0.98, "recall {r}");
    }

    #[test]
    fn test_mlx_knn_graph_recall_cosine_padded() {
        let r = mlx_recall(Dist::Cosine, 18, 0);
        assert!(r > 0.98, "recall {r}");
    }

    #[test]
    fn test_mlx_knn_graph_two_hop() {
        let r = mlx_recall(Dist::SquaredEuclidean, 30, 1);
        assert!(r > 0.98, "recall {r}");
    }

    #[test]
    fn test_mlx_knn_graph_blocked_staging() {
        // dim 256 at build_k 22 needs the blocked local-join path.
        let p = plan_local_join_mlx(256, 22, false, MLX_TG_BYTES).unwrap();
        assert!(p.buf_b > 1);
        let r = mlx_recall(Dist::SquaredEuclidean, 256, 0);
        assert!(r > 0.95, "recall {r}");
    }

    /// Fraction of nodes reachable from `start` over a nav graph.
    ///
    /// ### Params
    ///
    /// * `nav` - Flat `n * k` graph with sentinels
    /// * `n` - Nodes
    /// * `k` - Degree
    /// * `start` - Entry node
    ///
    /// ### Returns
    ///
    /// Fraction in `[0, 1]`
    fn reachable(nav: &[u32], n: usize, k: usize, start: u32) -> f64 {
        let mut seen = vec![false; n];
        let mut stack = vec![start as usize];
        seen[start as usize] = true;
        let mut count = 1;
        while let Some(u) = stack.pop() {
            for &v in &nav[u * k..(u + 1) * k] {
                if v != MLX_SENTINEL && !seen[v as usize] {
                    seen[v as usize] = true;
                    count += 1;
                    stack.push(v as usize);
                }
            }
        }
        count as f64 / n as f64
    }

    /// Run NN-Descent and CAGRA on MLX and check the nav-graph invariants.
    ///
    /// ### Returns
    ///
    /// `(data, reachability from the medoid)`
    fn mlx_cagra() -> (Mat<f32>, f64) {
        let (n, dim, k) = (3_000, 24, 16);
        let build_k = 24;
        // Uniform rather than clustered: separated clusters give a
        // disconnected kNN graph, and reachability then measures the data.
        let mut rng = StdRng::seed_from_u64(5);
        let data = Mat::from_fn(n, dim, |_, _| rng.random_range(-1.0f32..1.0));
        let (flat, _, _) = data.as_ref().into_row_major();
        let cfg = NnDescentCfgMlx {
            build_k,
            max_iters: 15,
            n_trees: 8,
            delta: 0.001,
            rho_thresh: 65535,
            refine_knn: 0,
            seed: 42,
            use_cosine: false,
        };
        let (g_idx, _, _) = nndescent_core_mlx(&flat, &[], n, dim, &cfg, false).unwrap();
        let (nav, medoid) =
            cagra_optimise_mlx(&g_idx, &flat, n, dim, build_k, k, Dist::SquaredEuclidean).unwrap();
        assert_eq!(nav.len(), n * k);
        assert!((medoid as usize) < n);
        for (i, row) in nav.chunks_exact(k).enumerate() {
            let filled = row.iter().take_while(|&&v| v != MLX_SENTINEL).count();
            assert!(row[filled..].iter().all(|&v| v == MLX_SENTINEL), "row {i}");
            assert!(filled >= k / 2, "row {i} has {filled} edges");
            let mut ids = row[..filled].to_vec();
            assert!(ids.iter().all(|&v| (v as usize) < n && v as usize != i));
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), filled, "row {i} has duplicates");
        }
        let r = reachable(&nav, n, k, medoid);
        println!("MLX CAGRA reachability from the medoid: {r:.4}");
        (data, r)
    }

    #[test]
    fn test_mlx_cagra_invariants() {
        let (_, r) = mlx_cagra();
        assert!(r > 0.99, "reachability {r}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_mlx_knn_graph_matches_wgpu_recall() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (n, dim, k) = (3_000, 30, 15);
        let data = clustered(n, dim, 11);
        let wgpu = crate::gpu::nndescent_gpu::build_knn_graph_gpu::<f32, WgpuRuntime>(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(k),
            None,
            None,
            None,
            None,
            None,
            None,
            42,
            false,
            WgpuDevice::DefaultDevice,
        )
        .unwrap();
        let (w_idx, _) = wgpu.extract_knn(None, false, false);
        let r_wgpu = recall(&data, Dist::SquaredEuclidean, &w_idx, k);
        let r_mlx = mlx_recall(Dist::SquaredEuclidean, dim, 0);
        println!("recall wgpu {r_wgpu:.4} mlx {r_mlx:.4}");
        assert!(r_mlx >= r_wgpu - 0.005, "mlx {r_mlx} vs wgpu {r_wgpu}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_mlx_cagra_reachability_matches_wgpu() {
        use crate::gpu::nndescent_gpu::NNDescentGpu;
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (data, r_mlx) = mlx_cagra();
        let (n, k) = (data.nrows(), 16);
        let idx = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(k),
            Some(24),
            None,
            Some(8),
            None,
            None,
            None,
            42,
            false,
            false,
            WgpuDevice::DefaultDevice,
        )
        .unwrap();
        let r_wgpu = reachable(idx.nav_graph(), n, k, idx.medoid);
        println!("reachability wgpu {r_wgpu:.4} mlx {r_mlx:.4}");
        assert!(r_mlx >= r_wgpu - 0.01, "mlx {r_mlx} vs wgpu {r_wgpu}");
    }
}
