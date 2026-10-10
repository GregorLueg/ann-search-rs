//! Inverted file (IVF) index on MLX.
//!
//! Build: k-means on a subsample and the full assignment both run on MLX
//! (see [`crate::mlx::k_means_mlx`]), then the vectors are laid out cluster by
//! cluster on the device with a CSR offset array.
//!
//! Query, per tile of queries, with no host sync until the results:
//!
//! 1. MLX GEMM scores every query against every centroid.
//! 2. The exhaustive index's row top-k kernel picks each query's probe pool.
//! 3. The queries are ordered by their nearest cluster (MLX `argsort`), so
//!    neighbouring SIMD groups scan the same clusters and share cache.
//! 4. The scan, one of three (see [`IvfScanMlx`]):
//!    - Query-major ([`IVF_SCAN_BODY`]): one SIMD group per query walks its
//!      probed clusters and keeps a SIMD-cooperative top k. No task list and no
//!      candidate buffer, but members are re-read for every query probing
//!      them.
//!    - Cluster-major: the probe lists are inverted on the device (sort keys,
//!      MLX `argsort`, `scatter_add` counts, a one-thread prefix sum), each
//!      cluster's tasks are cut into tiles of 8 queries, every member load
//!      is scored against all 8, and a final kernel merges the
//!      per-(query, probe) top k. Like the wgpu layout but with the task
//!      list built on the device and a candidate buffer of k per probe
//!      rather than every member. The dot products run either on registers
//!      ([`IVF_CLUSTER_TILED_BODY`]) or on `simdgroup_matrix` tiles
//!      ([`IVF_CLUSTER_MMA_BODY`]).
//!
//! The query-major and register-tiled scans compute distances directly, so
//! there is no cancellation; the `simdgroup_matrix` one expands Euclidean.

use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::mlx::exhaustive_mlx::{RowTopK, TopKKernel, TOPK_ROWS_PER_GROUP};
use crate::mlx::ffi::*;
use crate::mlx::k_means_mlx::*;
use crate::prelude::*;
use crate::utils::k_means_utils::sample_vectors;
use crate::utils::DimensionValidation;

////////////
// Consts //
////////////

/// Default cap on queries per tile, as on the wgpu index. The centroid score
/// matrix caps it further, see [`score_tile_rows`].
const IVF_MLX_QUERY_BATCH_SIZE: usize = 100_000;

/// Query rows (SIMD groups) per threadgroup in the query-major scan kernel.
const SCAN_ROWS_PER_GROUP: i32 = 8;

/// Query tasks (SIMD groups) per threadgroup in the cluster-major scans. Each
/// member load is reused by this many queries. The `simdgroup_matrix` scan
/// needs exactly 8.
///
/// Capped by the pipeline's thread limit, which the compiler lowers with the
/// register-resident top-K: 512 threads failed at K = 15..32 on an M1 Max
/// (limit 384) and only fails at eval, so this stays at the 256 threads the
/// other top-k kernels here run with.
const CLUSTER_SCAN_QT: usize = 8;

/// Members per SIMD group in one cluster-major chunk; a chunk is
/// `CLUSTER_SCAN_MEMBERS * QT` members.
const CLUSTER_SCAN_MEMBERS: usize = 32;

/// Cap on the query dimensions staged per block in the cluster-major scans;
/// wider rows are split along the reduction axis.
const CLUSTER_SCAN_MAX_DB: usize = 256;

/// Threadgroup memory on Apple GPUs, the budget the cluster-scan plan is
/// sized against. MLX does not expose the device limit.
const APPLE_THREADGROUP_BYTES: usize = 32 * 1024;

/// Zero rows appended to the device copy of the vectors, one 8x8 tile's
/// worth, so the `simdgroup_matrix` scan never reads past the buffer.
const DB_PAD_ROWS: usize = 8;

/// Threads per threadgroup in the probe-key kernel (one per query).
const PROBE_KEYS_THREADS: usize = 64;

/// Upper bound in bytes on the cluster-major path's per-(query, probe)
/// partial top-k buffers (indices plus distances) for one query tile.
const CLUSTER_PARTIAL_TILE_BYTES: usize = 256 * 1024 * 1024;

/// Query-major cluster scan, a [`TopKKernel`] body. One
/// SIMD group per query: the probed clusters are walked in ascending centroid
/// distance, each lane takes every 32nd member, computes the distance in full
/// (`float4` loads when `VEC4`) and inserts it. Probing stops after
/// `params[0]` (nprobe) clusters once K points are reachable, else carries on
/// down the pool of P, as the wgpu index does. Slots never filled stay at
/// `INFINITY`.
///
/// Template args: `K`, pool width `P`, `DIM`, `VEC4` (`DIM % 4 == 0`) and
/// `COSINE` (`1 - q.x` on unit vectors, else squared Euclidean).
const IVF_SCAN_BODY: &str = r#"
    uint lane = thread_position_in_grid.x;
    uint qi = order[thread_position_in_grid.y];
    ulong orow = qi;
    const device float* qv = q + (ulong)qi * DIM;
    const device uint* prow = probe + (ulong)qi * P;
    uint nprobe = params[0];
    uint reach = 0;
    for (uint p = 0; p < (uint)P; p++) {
        if (p >= nprobe && reach >= (uint)K) break;
        uint c = prow[p];
        uint s = offsets[c];
        uint e = offsets[c + 1];
        reach += e - s;
        for (uint base = s; base < e; base += 32) {
            uint j = base + lane;
            bool ok = j < e;
            const device float* x = db + (ulong)(ok ? j : s) * DIM;
            float acc = 0.0f;
            if (VEC4) {
                const device float4* x4 = (const device float4*)x;
                const device float4* q4 = (const device float4*)qv;
                float4 a4 = float4(0.0f);
                for (int d = 0; d < DIM / 4; d++) {
                    if (COSINE) {
                        a4 = fma(q4[d], x4[d], a4);
                    } else {
                        float4 t = q4[d] - x4[d];
                        a4 = fma(t, t, a4);
                    }
                }
                acc = (a4.x + a4.y) + (a4.z + a4.w);
            } else {
                for (int d = 0; d < DIM; d++) {
                    if (COSINE) {
                        acc = fma(qv[d], x[d], acc);
                    } else {
                        float t = qv[d] - x[d];
                        acc = fma(t, t, acc);
                    }
                }
            }
            float dist = COSINE ? 1.0f - acc : acc;
            SG_OFFER(ok, dist, j);
        }
    }
"#;

/// Cluster-major step 1, one thread per query: apply the probe rule (nprobe
/// clusters, more until K points are reachable) to the pool and emit one sort
/// key per (query, pool slot): the cluster id, or the sentinel `NLIST` for a
/// slot not probed. `eff` is the number of slots probed.
///
/// Template args: `K`, `P`, `NLIST`.
const IVF_PROBE_KEYS_SOURCE: &str = r#"
    uint qi = thread_position_in_grid.x;
    const device uint* prow = probe + (ulong)qi * P;
    uint nprobe = params[0];
    uint reach = 0;
    uint e = P;
    for (uint p = 0; p < (uint)P; p++) {
        if (p >= nprobe && reach >= (uint)K) { e = p; break; }
        uint c = prow[p];
        reach += offsets[c + 1] - offsets[c];
    }
    for (uint p = 0; p < (uint)P; p++) {
        keys[(ulong)qi * P + p] = p < e ? prow[p] : (uint)NLIST;
    }
    eff[qi] = e;
"#;

/// Cluster-major step 3, a single thread: exclusive prefix sums of the
/// per-cluster task counts, into task offsets and offsets of the `QT`-task
/// tiles each cluster is cut into. `NLIST` is a few thousand at most, so the
/// serial loop is cheap.
///
/// Template args: `NLIST`, `QT`.
const IVF_TASK_CSR_SOURCE: &str = r#"
    if (thread_position_in_grid.x != 0) return;
    uint t = 0;
    uint g = 0;
    for (uint c = 0; c < (uint)NLIST; c++) {
        task_off[c] = t;
        tile_off[c] = g;
        uint n = counts[c];
        t += n;
        g += (n + QT - 1) / QT;
    }
    task_off[NLIST] = t;
    tile_off[NLIST] = g;
"#;

/// Metal snippet opening every cluster-major scan: map threadgroup `g` to its
/// cluster `c` by binary search over `tile_off` (surplus threadgroups of the
/// upper-bound grid exit at once), and the tile's first task `tbase`, its task
/// count `nt` and the cluster's member range `[ms, me)`.
const CLUSTER_TILE_LOOKUP: &str = r#"
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint tid = sg * 32 + lane;
    uint g = threadgroup_position_in_grid.y;
    if (g >= tile_off[NLIST]) return;
    uint lo = 0;
    uint hi = NLIST;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (tile_off[mid] <= g) { lo = mid; } else { hi = mid; }
    }
    uint c = lo;
    uint tbase = task_off[c] + (g - tile_off[c]) * QT;
    uint nt = min((uint)QT, task_off[c + 1] - tbase);
    uint ms = offsets[c];
    uint me = offsets[c + 1];
    bool active = sg < nt;
    uint flat = active ? order[tbase + sg] : 0;
    ulong orow = flat;
"#;

/// Register-tiled cluster-major scan, a [`TopKKernel`] body after
/// [`CLUSTER_TILE_LOOKUP`]. The tile's `QT` query
/// blocks are staged in threadgroup memory (`DB` dims at a time); each thread
/// owns one member of a `32 * QT` chunk, reads it from device memory once and
/// scores it against all `QT` queries with `QT` accumulators, so one member
/// load feeds `QT` FMAs (`float4` when `VEC4`). Scores go through threadgroup
/// memory so SIMD group `sg` can run the top-k insert for task `sg`.
///
/// Template args: `K`, `P`, `DIM`, `DB` (a multiple of 4 when `VEC4`), `QT`,
/// `NLIST`, `VEC4`, `COSINE`.
const IVF_CLUSTER_TILED_BODY: &str = r#"
    const uint CHUNK = 32 * QT;
    threadgroup float4 qs4[QT * ((DB + 3) / 4)];
    threadgroup float* qs = (threadgroup float*)qs4;
    threadgroup float sc[QT * 32 * QT];
    for (uint cs = ms; cs < me; cs += CHUNK) {
        uint j = cs + tid;
        bool jv = j < me;
        const device float* x = db + (ulong)(jv ? j : ms) * DIM;
        float acc[QT];
        for (int i = 0; i < QT; i++) { acc[i] = 0.0f; }
        for (uint blk = 0; blk < (uint)DIM; blk += DB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < (uint)(QT * DB); e += CHUNK) {
                uint qq = e / DB;
                uint col = blk + e % DB;
                float v = 0.0f;
                if (qq < nt && col < (uint)DIM) {
                    v = q[(ulong)(order[tbase + qq] / P) * DIM + col];
                }
                qs[e] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint dn = min((uint)DB, (uint)DIM - blk);
            if (VEC4) {
                const device float4* x4 = (const device float4*)(x + blk);
                for (uint d4 = 0; d4 < dn / 4; d4++) {
                    float4 xv = x4[d4];
                    for (int i = 0; i < QT; i++) {
                        float4 qv = qs4[i * (DB / 4) + d4];
                        if (COSINE) {
                            acc[i] += dot(qv, xv);
                        } else {
                            float4 t = qv - xv;
                            acc[i] += dot(t, t);
                        }
                    }
                }
            } else {
                for (uint d = 0; d < dn; d++) {
                    float xv = x[blk + d];
                    for (int i = 0; i < QT; i++) {
                        float qv = qs[i * DB + d];
                        if (COSINE) {
                            acc[i] = fma(qv, xv, acc[i]);
                        } else {
                            float t = qv - xv;
                            acc[i] = fma(t, t, acc[i]);
                        }
                    }
                }
            }
        }
        for (int i = 0; i < QT; i++) {
            sc[i * CHUNK + tid] = jv ? (COSINE ? 1.0f - acc[i] : acc[i]) : INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint m = lane; m < CHUNK; m += 32) {
                float v = sc[sg * CHUNK + m];
                SG_OFFER(true, v, cs + m);
            }
        }
    }
    if (!active) return;
"#;

/// Header for [`IVF_CLUSTER_MMA_BODY`].
const SIMDGROUP_MATRIX_HEADER: &str = "#include <metal_simdgroup_matrix>\n";

/// simdgroup_matrix cluster-major scan, a [`TopKKernel`] body after
/// [`CLUSTER_TILE_LOOKUP`]. `QT` is 8: the tile's 8
/// query blocks are staged in threadgroup memory (zero beyond `DIM`), and
/// SIMD group `sg` multiplies them against 4 transposed 8-member tiles loaded
/// straight from device memory, so a chunk of 256 members costs one 8x8x8
/// `simdgroup_multiply_accumulate` per (member tile, 8 dims). Reads past
/// `DIM` or the cluster land on the next row or the index's zero padding rows
/// and meet a zero query entry or are discarded. Dot products go through
/// threadgroup memory to the per-task top-k insert; Euclidean is
/// `|x|^2 - 2 q.x + |q|^2` from precomputed norms.
///
/// Template args: `K`, `P`, `DIM`, `DB` (a multiple of 8), `QT` (8),
/// `NLIST`, `COSINE`.
const IVF_CLUSTER_MMA_BODY: &str = r#"
    const uint CHUNK = 32 * QT;
    threadgroup float qs[QT * DB];
    threadgroup float sc[QT * 32 * QT];
    float qnorm = active ? qn[flat / P] : 0.0f;
    for (uint cs = ms; cs < me; cs += CHUNK) {
        simdgroup_float8x8 acc[4];
        for (int t = 0; t < 4; t++) { acc[t] = simdgroup_float8x8(0.0f); }
        uint m0 = cs + sg * 32;
        for (uint blk = 0; blk < (uint)DIM; blk += DB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < (uint)(QT * DB); e += CHUNK) {
                uint qq = e / DB;
                uint col = blk + e % DB;
                float v = 0.0f;
                if (qq < nt && col < (uint)DIM) {
                    v = q[(ulong)(order[tbase + qq] / P) * DIM + col];
                }
                qs[e] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint dn = min((uint)DB, (uint)DIM - blk);
            for (uint kk = 0; kk < dn; kk += 8) {
                simdgroup_float8x8 a;
                simdgroup_load(a, qs + kk, DB);
                for (int t = 0; t < 4; t++) {
                    uint mt = m0 + t * 8;
                    if (mt < me) {
                        simdgroup_float8x8 b;
                        simdgroup_load(b, db + (ulong)mt * DIM + blk + kk, DIM, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(acc[t], a, b, acc[t]);
                    }
                }
            }
        }
        for (int t = 0; t < 4; t++) {
            simdgroup_store(acc[t], sc + sg * 32 + t * 8, CHUNK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint m = lane; m < CHUNK; m += 32) {
                uint j = cs + m;
                bool ok = j < me;
                float dotv = sc[sg * CHUNK + m];
                float v = COSINE ? 1.0f - dotv : xn[ok ? j : ms] - 2.0f * dotv + qnorm;
                SG_OFFER(ok, v, j);
            }
        }
    }
    if (!active) return;
"#;

/// Cluster-major step 5, a [`TopKKernel`] body: one SIMD
/// group per query merges the `eff * K` partial candidates of its probed
/// slots.
///
/// Template args: `K`, `P`.
const IVF_PARTIAL_MERGE_BODY: &str = r#"
    uint lane = thread_position_in_grid.x;
    uint qi = thread_position_in_grid.y;
    ulong orow = qi;
    ulong base = (ulong)qi * P * K;
    uint n = eff[qi] * K;
    for (uint i0 = 0; i0 < n; i0 += 32) {
        uint i = i0 + lane;
        bool ok = i < n;
        SG_OFFER(ok, ok ? part_dist[base + i] : INFINITY, ok ? part_idx[base + i] : 0u);
    }
"#;

/// Which scan the IVF MLX query runs. All give the same candidates up to
/// rounding; they differ in memory access pattern and arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IvfScanMlx {
    /// One SIMD group per query walks its probed clusters; members are read
    /// from device memory once per query that probes them. Loses to
    /// `ClusterMajor` as `dim` grows, for want of reuse, but keeps one warm
    /// top-k per query, which the cluster-major scans (a cold top-k per
    /// probe) lack when k approaches the cluster size.
    QueryMajor,
    /// The probe lists are inverted on the device into per-cluster task lists
    /// and each member is scored against up to [`CLUSTER_SCAN_QT`] queries
    /// with register accumulators, see [`IVF_CLUSTER_TILED_BODY`];
    /// per-(query, probe) top-k are merged after.
    ClusterMajor,
    /// Cluster-major with the dot products on `simdgroup_matrix` 8x8 tiles,
    /// see [`IVF_CLUSTER_MMA_BODY`]. Euclidean goes through the
    /// `|x|^2 - 2 q.x + |q|^2` expansion, so near-ties can swap.
    ClusterMatrix,
    /// `ClusterMajor` while `k <= dim`, else `QueryMajor`. Cluster-major
    /// saves member reads, worth more as `dim` grows; it pays a cold top-k
    /// per probed cluster plus a merge over `probes * k` candidates, worth
    /// more as `k` grows. The `k <= dim` line is fitted to a small ablation
    /// and is not tuned for cluster size.
    #[default]
    Auto,
}

/// Threadgroup memory plan for the cluster-major scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClusterScanPlan {
    /// Dimensions staged per block
    pub db: usize,
    /// Tasks (SIMD groups) per threadgroup
    pub qt: usize,
}

/// Threadgroup bytes the cluster-major scans stage: `QT` query blocks of
/// `db` dims (rounded up to `float4`) plus the `QT x 32 QT` score tile.
///
/// ### Params
///
/// * `db` - Dimensions per block
///
/// ### Returns
///
/// Bytes of threadgroup memory
pub(crate) fn cluster_scan_smem_bytes(db: usize) -> usize {
    let qt = CLUSTER_SCAN_QT;
    (qt * db.next_multiple_of(4) + qt * CLUSTER_SCAN_MEMBERS * qt) * size_of::<f32>()
}

/// Size the cluster-major scan's query staging against a threadgroup budget.
///
/// Blocks the reduction axis, so the footprint is independent of `dim`.
///
/// ### Params
///
/// * `dim` - Embedding dimensionality
/// * `smem_bytes` - Threadgroup memory budget
/// * `align` - Block width granularity: 4 for the `float4` kernel, 8 for the
///   `simdgroup_matrix` one
///
/// ### Returns
///
/// The plan, or `None` if not even one `align`-wide block fits
pub(crate) fn plan_cluster_scan(
    dim: usize,
    smem_bytes: usize,
    align: usize,
) -> Option<ClusterScanPlan> {
    let qt = CLUSTER_SCAN_QT;
    let scores = qt * CLUSTER_SCAN_MEMBERS * qt * size_of::<f32>();
    let fit = smem_bytes.checked_sub(scores)? / (qt * size_of::<f32>());
    let cap = fit.min(CLUSTER_SCAN_MAX_DB) / align * align;
    let full = dim.next_multiple_of(align);
    let db = if full <= cap { full } else { cap };
    debug_assert!(db == 0 || cluster_scan_smem_bytes(db) <= smem_bytes);
    (db > 0).then_some(ClusterScanPlan { db, qt })
}

/////////////
// Helpers //
/////////////

/// Scale every row to unit L2 norm in place. Zero rows stay zero.
///
/// ### Params
///
/// * `data` - Row-major vectors
/// * `dim` - Row length
fn normalise_rows(data: &mut [f32], dim: usize) {
    data.par_chunks_exact_mut(dim).for_each(|row| {
        let norm = f32::calculate_l2_norm(row);
        if norm > 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    });
}

/// Lay the vectors out cluster by cluster.
///
/// ### Params
///
/// * `vectors_flat` - Row-major vectors
/// * `dim` - Row length
/// * `assignments` - Cluster per vector
/// * `nlist` - Number of clusters
///
/// ### Returns
///
/// `(reordered vectors, original index per position, CSR offsets)`
fn reorganise_by_cluster(
    vectors_flat: &[f32],
    dim: usize,
    assignments: &[usize],
    nlist: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut offsets = vec![0usize; nlist + 1];
    for &c in assignments {
        offsets[c + 1] += 1;
    }
    for i in 0..nlist {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut original = vec![0usize; assignments.len()];
    for (i, &c) in assignments.iter().enumerate() {
        original[cursor[c]] = i;
        cursor[c] += 1;
    }
    let mut reordered = vec![0f32; vectors_flat.len()];
    reordered
        .par_chunks_exact_mut(dim)
        .zip(original.par_iter())
        .for_each(|(dst, &src)| dst.copy_from_slice(&vectors_flat[src * dim..(src + 1) * dim]));
    (reordered, original, offsets)
}

/////////////////
// IvfIndexMlx //
/////////////////

/// IVF index on MLX. f32 only.
///
/// Holds an MLX stream, which is thread affine: build and query on the same
/// thread. The raw handles keep this type `!Send` and `!Sync`.
pub struct IvfIndexMlx {
    /// Vectors in cluster order on the host, unit-normalised for Cosine. The
    /// self query reads its queries from here.
    vectors_flat: Vec<f32>,
    /// Maps cluster-order position -> original index
    original_indices: Vec<usize>,
    /// CSR offsets per cluster into the cluster-ordered vectors; `nlist + 1`
    cluster_offsets: Vec<usize>,
    /// Prefix sums of the cluster sizes in ascending order; `nlist + 1`. The
    /// smallest `m` with `sorted_size_prefix[m] >= k` is a probe count that
    /// reaches `k` points whichever clusters are probed.
    sorted_size_prefix: Vec<usize>,
    /// Embedding dimensionality
    dim: usize,
    /// Number of samples
    n: usize,
    /// Number of clusters
    nlist: usize,
    /// Distance metric
    metric: Dist,
    /// Cluster-ordered vectors on the device, `[n, dim]`
    db: Array,
    /// CSR offsets on the device, `[nlist + 1]` u32
    offsets: Array,
    /// Centroid GEMM operands for the probe selection
    centroid_ops: CentroidOperands,
    /// Row top-k kernel (probe selection), shared with the exhaustive index
    probe_topk: RowTopK,
    /// Query-major scan kernel, see [`IVF_SCAN_BODY`]
    scan_kernel: TopKKernel,
    /// Which scan the queries run
    scan: IvfScanMlx,
    /// Cluster-major staging plan; `None` falls back to the query-major scan
    cluster_plan: Option<ClusterScanPlan>,
    /// Cluster-major probe-key kernel, see [`IVF_PROBE_KEYS_SOURCE`]
    keys_kernel: MetalKernel,
    /// Cluster-major task CSR kernel, see [`IVF_TASK_CSR_SOURCE`]
    csr_kernel: MetalKernel,
    /// Register-tiled cluster-major scan, see [`IVF_CLUSTER_TILED_BODY`]
    tiled_kernel: TopKKernel,
    /// `simdgroup_matrix` cluster-major scan, see [`IVF_CLUSTER_MMA_BODY`]
    mma_kernel: TopKKernel,
    /// Staging plan for the `simdgroup_matrix` scan (8-wide blocks)
    mma_plan: Option<ClusterScanPlan>,
    /// Squared L2 norm per cluster-ordered vector, `[n]`; the Euclidean
    /// `simdgroup_matrix` scan reads it
    xn: Array,
    /// Cluster-major partial merge kernel, see [`IVF_PARTIAL_MERGE_BODY`]
    merge_kernel: TopKKernel,
    /// Stream every op runs on. Declared last so it drops after the arrays.
    stream: Stream,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl DimensionValidation for IvfIndexMlx {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////////////
// Main implementation //
/////////////////////////

impl IvfIndexMlx {
    /// Build an IVF index on MLX.
    ///
    /// ### Params
    ///
    /// * `data` - Database vectors, samples x features
    /// * `metric` - Distance metric. Manhattan is not supported.
    /// * `nlist` - Number of clusters (defaults to `sqrt(n)`)
    /// * `k_means_params` - Optional [`KMeansTrainingParams`]; `iters`,
    ///   `init` and `balanced` are honoured, `path` is ignored. Every
    ///   iteration runs (no early stop).
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Initialised `IvfIndexMlx`, vectors and centroids resident on the device
    pub fn build(
        data: impl AnnMatrix<f32>,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        install_error_handler();

        let (vectors_flat, n, dim) = data.into_row_major();
        let nlist = nlist
            .unwrap_or((n as f32).sqrt() as usize)
            .clamp(1, n.max(1));
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!("  Generating MLX IVF index with {} Voronoi cells.", nlist);
        }

        let stream = Stream::default_gpu();
        let centroids = train_centroids_mlx(
            &training,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            &stream,
            verbose,
        )?;
        let assignments = assign_all_mlx(&vectors_flat, dim, &centroids, nlist, &metric, &stream)?;

        let (mut vectors_flat, original_indices, cluster_offsets) =
            reorganise_by_cluster(&vectors_flat, dim, &assignments, nlist);
        if metric == Dist::Cosine {
            normalise_rows(&mut vectors_flat, dim);
        }

        let mut sizes: Vec<usize> = cluster_offsets.windows(2).map(|w| w[1] - w[0]).collect();
        sizes.sort_unstable();
        let mut sorted_size_prefix = Vec::with_capacity(nlist + 1);
        sorted_size_prefix.push(0);
        for s in sizes {
            sorted_size_prefix.push(sorted_size_prefix[sorted_size_prefix.len() - 1] + s);
        }

        // Zero rows past the end: the `simdgroup_matrix` scan loads whole 8x8
        // tiles and may run up to 8 rows past the last cluster.
        let mut padded = vectors_flat.clone();
        padded.resize((n + DB_PAD_ROWS) * dim, 0.0);
        let db = Array::from_f32(&padded, &[(n + DB_PAD_ROWS) as i32, dim as i32]);
        drop(padded);
        let xn_host: Vec<f32> = vectors_flat
            .par_chunks_exact(dim)
            .map(|r| f32::dot_simd(r, r))
            .collect();
        let xn = Array::from_f32(&xn_host, &[n as i32]);
        let offsets_u32: Vec<u32> = cluster_offsets.iter().map(|&o| o as u32).collect();
        let offsets = Array::from_u32(&offsets_u32, &[nlist as i32 + 1]);
        let c = Array::from_f32(&centroids, &[nlist as i32, dim as i32]);
        let centroid_ops = CentroidOperands::new(&c, nlist, &metric, &stream)?;
        eval_all(
            &[&db, &offsets, &xn, &centroid_ops.ct, &centroid_ops.add],
            false,
        )?;

        Ok(Self {
            vectors_flat,
            original_indices,
            cluster_offsets,
            sorted_size_prefix,
            dim,
            n,
            nlist,
            metric,
            db,
            offsets,
            centroid_ops,
            probe_topk: RowTopK::new("ivf_probe_topk"),
            scan_kernel: TopKKernel::new(
                "ivf_scan",
                &["q", "db", "offsets", "probe", "order", "params"],
                &["out_idx", "out_dist"],
                "",
                "",
                IVF_SCAN_BODY,
            ),
            scan: IvfScanMlx::default(),
            cluster_plan: plan_cluster_scan(dim, APPLE_THREADGROUP_BYTES, 4),
            mma_plan: plan_cluster_scan(dim, APPLE_THREADGROUP_BYTES, 8),
            keys_kernel: MetalKernel::new(
                "ivf_probe_keys",
                &["probe", "offsets", "params"],
                &["keys", "eff"],
                IVF_PROBE_KEYS_SOURCE,
            ),
            csr_kernel: MetalKernel::new(
                "ivf_task_csr",
                &["counts"],
                &["task_off", "tile_off"],
                IVF_TASK_CSR_SOURCE,
            ),
            tiled_kernel: TopKKernel::new(
                "ivf_cluster_tiled",
                &["q", "db", "offsets", "order", "task_off", "tile_off"],
                &["out_idx", "out_dist"],
                "",
                CLUSTER_TILE_LOOKUP,
                IVF_CLUSTER_TILED_BODY,
            ),
            mma_kernel: TopKKernel::new(
                "ivf_cluster_mma",
                &[
                    "q", "db", "offsets", "order", "task_off", "tile_off", "qn", "xn",
                ],
                &["out_idx", "out_dist"],
                SIMDGROUP_MATRIX_HEADER,
                CLUSTER_TILE_LOOKUP,
                IVF_CLUSTER_MMA_BODY,
            ),
            xn,
            merge_kernel: TopKKernel::new(
                "ivf_partial_merge",
                &["part_idx", "part_dist", "eff"],
                &["out_idx", "out_dist"],
                "",
                "",
                IVF_PARTIAL_MERGE_BODY,
            ),
            stream,
        })
    }

    /// Pick the scan the queries run.
    ///
    /// ### Params
    ///
    /// * `scan` - Query-major (the default) or cluster-major
    ///
    /// ### Returns
    ///
    /// `self` with the scan set
    pub fn with_scan(mut self, scan: IvfScanMlx) -> Self {
        self.scan = scan;
        self
    }

    /// Query the index with a batch of vectors.
    ///
    /// ### Params
    ///
    /// * `query_mat` - Query vectors, samples x features
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    /// * `nquery` - Queries per device tile (defaults to 100k, capped so the
    ///   query x centroid score matrix stays bounded)
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)`, ascending by distance
    pub fn query_batch(
        &self,
        query_mat: impl AnnMatrix<f32>,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        verbose: bool,
    ) -> KnnResult<f32> {
        let (mut queries, n_query, dim_query) = query_mat.into_row_major();
        self.check_dim(dim_query)?;
        if self.metric == Dist::Cosine {
            normalise_rows(&mut queries, dim_query);
        }
        let (indices, dists) =
            self.query_prepared(&queries, n_query, k, nprobe, nquery, verbose)?;
        let indices = indices
            .into_iter()
            .map(|row| row.into_iter().map(|p| self.original_indices[p]).collect())
            .collect();
        Ok((indices, dists))
    }

    /// Generate the kNN graph of the indexed vectors against themselves.
    ///
    /// The queries are the cluster-ordered vectors, so consecutive queries
    /// probe mostly the same clusters.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector, self included
    /// * `nprobe` - Number of clusters to search
    /// * `nquery` - Queries per device tile
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`, one row per vector in the
    /// original order
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<f32> {
        let (idx_reorg, dist_reorg) =
            self.query_prepared(&self.vectors_flat, self.n, k, nprobe, nquery, verbose)?;
        let mut indices = vec![Vec::new(); self.n];
        let mut dists = if return_dist {
            vec![Vec::new(); self.n]
        } else {
            Vec::new()
        };
        for (pos, (row, drow)) in idx_reorg.into_iter().zip(dist_reorg).enumerate() {
            let orig = self.original_indices[pos];
            indices[orig] = row.into_iter().map(|p| self.original_indices[p]).collect();
            if return_dist {
                dists[orig] = drow;
            }
        }
        Ok((indices, return_dist.then_some(dists)))
    }

    /// Clusters that are always enough to reach `k` points, whichever they
    /// are.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours the query must be able to reach
    ///
    /// ### Returns
    ///
    /// Cluster count, at most `nlist`
    fn min_clusters_for_k(&self, k: usize) -> usize {
        self.sorted_size_prefix
            .partition_point(|&reachable| reachable < k)
            .min(self.nlist)
    }

    /// Queue probe selection and cluster scan for one query tile.
    ///
    /// ### Params
    ///
    /// * `q` - Row-major query tile, unit-normalised for Cosine
    /// * `n_q` - Rows in the tile
    /// * `k` - Number of neighbours, at most `n`
    /// * `nprobe` - Clusters to probe before the reachability top-up
    /// * `pool` - Probe pool width, `>= nprobe` and `<= nlist`
    /// * `scan` - Resolved scan, never `Auto`
    ///
    /// ### Returns
    ///
    /// Lazy `(positions as u32, distances)`, both `[n_q, k]`, ascending
    fn queue_tile(
        &self,
        q: &[f32],
        n_q: usize,
        k: usize,
        nprobe: usize,
        pool: usize,
        scan: IvfScanMlx,
    ) -> Result<(Array, Array), AnnSearchErrors> {
        let s = &self.stream;
        let q_host = q;
        let q = Array::from_f32(q, &[n_q as i32, self.dim as i32]);
        let ops = &self.centroid_ops;
        let scores = Array::addmm(&ops.add, &q, &ops.ct, ops.alpha, 1.0, s)?;

        let (probe, _) = self.probe_topk.apply(&scores, n_q, self.nlist, pool, s)?;

        let plan = match scan {
            IvfScanMlx::QueryMajor | IvfScanMlx::Auto => None,
            IvfScanMlx::ClusterMatrix => self.mma_plan,
            IvfScanMlx::ClusterMajor => self.cluster_plan,
        };
        if let Some(plan) = plan {
            let qn: Vec<f32> = q_host
                .par_chunks_exact(self.dim)
                .map(|r| f32::dot_simd(r, r))
                .collect();
            let qn = Array::from_f32(&qn, &[n_q as i32]);
            return self.queue_cluster_major(&q, &qn, &probe, n_q, k, nprobe, pool, scan, plan);
        }

        let order = probe
            .slice(&[0, 0], &[n_q as i32, 1], s)?
            .reshape(&[n_q as i32], s)?
            .argsort_axis(0, s)?;
        let params = Array::from_u32(&[nprobe as u32], &[1]);

        let shape = [n_q as i32, k as i32];
        let mut out = self.scan_kernel.pick(k).apply(
            &[&q, &self.db, &self.offsets, &probe, &order, &params],
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
            [32, SCAN_ROWS_PER_GROUP, 1],
            &[
                ("K", k as i32),
                ("P", pool as i32),
                ("DIM", self.dim as i32),
                ("VEC4", self.dim.is_multiple_of(4) as i32),
                ("COSINE", (self.metric == Dist::Cosine) as i32),
            ],
            s,
        )?;
        let dist = out.pop().expect("kernel has two outputs");
        let idx = out.pop().expect("kernel has two outputs");
        Ok((idx, dist))
    }

    /// Queue the cluster-major scan for one query tile, given its probe pool.
    ///
    /// ### Params
    ///
    /// * `q` - Query tile on the device, `[n_q, dim]`
    /// * `qn` - Squared L2 norm per query, `[n_q]`
    /// * `probe` - Probe pool per query, `[n_q, pool]` u32, ascending
    /// * `n_q` - Rows in the tile
    /// * `k` - Number of neighbours, at most `n`
    /// * `nprobe` - Clusters to probe before the reachability top-up
    /// * `pool` - Probe pool width
    /// * `scan` - `ClusterMajor` or `ClusterMatrix`
    /// * `plan` - Threadgroup staging plan
    ///
    /// ### Returns
    ///
    /// Lazy `(positions as u32, distances)`, both `[n_q, k]`, ascending
    #[allow(clippy::too_many_arguments)]
    fn queue_cluster_major(
        &self,
        q: &Array,
        qn: &Array,
        probe: &Array,
        n_q: usize,
        k: usize,
        nprobe: usize,
        pool: usize,
        scan: IvfScanMlx,
        plan: ClusterScanPlan,
    ) -> Result<(Array, Array), AnnSearchErrors> {
        let s = &self.stream;
        let n_tasks = n_q * pool;
        let nlist = self.nlist as i32;
        let params = Array::from_u32(&[nprobe as u32], &[1]);
        let flat = [n_tasks as i32];
        let rows = [n_q as i32];
        let mut ke = self.keys_kernel.apply(
            &[probe, &self.offsets, &params],
            &[
                OutputSpec {
                    shape: &flat,
                    dtype: MLX_UINT32,
                },
                OutputSpec {
                    shape: &rows,
                    dtype: MLX_UINT32,
                },
            ],
            [n_q as i32, 1, 1],
            [n_q.min(PROBE_KEYS_THREADS) as i32, 1, 1],
            &[("K", k as i32), ("P", pool as i32), ("NLIST", nlist)],
            s,
        )?;
        let eff = ke.pop().expect("kernel has two outputs");
        let keys = ke.pop().expect("kernel has two outputs");

        let order = keys.argsort_axis(0, s)?;
        let ones = Array::ones(&[n_tasks as i32, 1], MLX_UINT32, s)?;
        let counts =
            Array::zeros(&[nlist + 1], MLX_UINT32, s)?.scatter_add_rows(&keys, &ones, s)?;

        let csr_shape = [nlist + 1];
        let mut csr = self.csr_kernel.apply(
            &[&counts],
            &[
                OutputSpec {
                    shape: &csr_shape,
                    dtype: MLX_UINT32,
                },
                OutputSpec {
                    shape: &csr_shape,
                    dtype: MLX_UINT32,
                },
            ],
            [1, 1, 1],
            [1, 1, 1],
            &[("NLIST", nlist), ("QT", plan.qt as i32)],
            s,
        )?;
        let tile_off = csr.pop().expect("kernel has two outputs");
        let task_off = csr.pop().expect("kernel has two outputs");

        // Upper bound on the tiles: every cluster's last tile may be partial.
        let max_tiles = n_tasks.div_ceil(plan.qt) + self.nlist;
        let part_shape = [n_tasks as i32, k as i32];
        let mut inputs = vec![q, &self.db, &self.offsets, &order, &task_off, &tile_off];
        let (kernel, db) = match scan {
            IvfScanMlx::ClusterMatrix => {
                inputs.extend([qn, &self.xn]);
                (&self.mma_kernel, plan.db)
            }
            _ => (&self.tiled_kernel, plan.db),
        };
        let mut part = kernel.pick(k).apply(
            &inputs,
            &[
                OutputSpec {
                    shape: &part_shape,
                    dtype: MLX_UINT32,
                },
                OutputSpec {
                    shape: &part_shape,
                    dtype: MLX_FLOAT32,
                },
            ],
            [32, (plan.qt * max_tiles) as i32, 1],
            [32, plan.qt as i32, 1],
            &[
                ("K", k as i32),
                ("P", pool as i32),
                ("DIM", self.dim as i32),
                ("DB", db as i32),
                ("QT", plan.qt as i32),
                ("NLIST", nlist),
                ("VEC4", self.dim.is_multiple_of(4) as i32),
                ("COSINE", (self.metric == Dist::Cosine) as i32),
            ],
            s,
        )?;
        let part_dist = part.pop().expect("kernel has two outputs");
        let part_idx = part.pop().expect("kernel has two outputs");

        let shape = [n_q as i32, k as i32];
        let mut out = self.merge_kernel.pick(k).apply(
            &[&part_idx, &part_dist, &eff],
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
            [32, TOPK_ROWS_PER_GROUP, 1],
            &[("K", k as i32), ("P", pool as i32)],
            s,
        )?;
        let dist = out.pop().expect("kernel has two outputs");
        let idx = out.pop().expect("kernel has two outputs");
        Ok((idx, dist))
    }

    /// Run the tiled probe + scan over already prepared queries.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, unit-normalised for Cosine
    /// * `n_query` - Number of queries
    /// * `k` - Number of neighbours, clamped to the index size
    /// * `nprobe` - Clusters to search (defaults to `sqrt(nlist)`)
    /// * `nquery` - Queries per device tile
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// `(cluster-order positions, distances)` per query, ascending
    fn query_prepared(
        &self,
        queries: &[f32],
        n_query: usize,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        verbose: bool,
    ) -> KnnResult<f32> {
        let k = k.min(self.n);
        if k == 0 || n_query == 0 {
            return Ok((vec![Vec::new(); n_query], vec![Vec::new(); n_query]));
        }
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .clamp(1, self.nlist);
        let pool = nprobe.max(self.min_clusters_for_k(k)).min(self.nlist);
        let mut tile = nquery
            .unwrap_or(IVF_MLX_QUERY_BATCH_SIZE)
            .min(score_tile_rows(self.nlist));
        let scan = match self.scan {
            IvfScanMlx::Auto if k <= self.dim => IvfScanMlx::ClusterMajor,
            IvfScanMlx::Auto => IvfScanMlx::QueryMajor,
            other => other,
        };
        if scan != IvfScanMlx::QueryMajor {
            let per_query = pool * k * (size_of::<u32>() + size_of::<f32>());
            tile = tile.min(CLUSTER_PARTIAL_TILE_BYTES / per_query);
        }
        let tile = tile.max(1);
        let dim = self.dim;

        let t0 = Instant::now();
        let mut pending = Vec::with_capacity(n_query.div_ceil(tile));
        for start in (0..n_query).step_by(tile) {
            let end = (start + tile).min(n_query);
            let (idx, dist) = self.queue_tile(
                &queries[start * dim..end * dim],
                end - start,
                k,
                nprobe,
                pool,
                scan,
            )?;
            eval_all(&[&idx, &dist], true)?;
            pending.push((idx, dist));
        }
        let t1 = Instant::now();

        let mut all_indices = Vec::with_capacity(n_query);
        let mut all_distances = Vec::with_capacity(n_query);
        for (idx, dist) in &pending {
            eval_all(&[idx, dist], false)?;
            let rows: Vec<(Vec<usize>, Vec<f32>)> = idx
                .as_u32()?
                .par_chunks_exact(k)
                .zip(dist.as_f32()?.par_chunks_exact(k))
                .map(|(ir, dr)| {
                    ir.iter()
                        .zip(dr)
                        .filter(|(_, d)| d.is_finite())
                        // Cosine can round a self-distance to just under zero.
                        .map(|(&j, &d)| (j as usize, d.max(0.0)))
                        .unzip()
                })
                .collect();
            for (ir, dr) in rows {
                all_indices.push(ir);
                all_distances.push(dr);
            }
        }

        if verbose {
            println!(
                "MLX IVF: {} queries in tiles of {}, nprobe {} (pool {}), queue {:.1} ms, wait+copy {:.1} ms",
                n_query.separate_with_underscores(),
                tile,
                nprobe,
                pool,
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }

        Ok((all_indices, all_distances))
    }

    /// Returns the approximate memory footprint of the index.
    ///
    /// ### Returns
    ///
    /// `(host bytes, device bytes)`; the device side is unified memory but a
    /// separate allocation
    pub fn memory_usage_bytes(&self) -> (usize, usize) {
        let host = std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * size_of::<f32>()
            + (self.original_indices.capacity()
                + self.cluster_offsets.capacity()
                + self.sorted_size_prefix.capacity())
                * size_of::<usize>();
        let device = (self.n * self.dim + self.nlist * self.dim + self.nlist) * size_of::<f32>()
            + (self.nlist + 1) * size_of::<u32>();
        (host, device)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exhaustive::ExhaustiveIndex;
    use crate::cpu::ivf::IvfIndex;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Gaussian blobs as a matrix.
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
    fn blobs(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<f32> = (0..30 * dim).map(|_| rng.random_range(-3.0..3.0)).collect();
        Mat::from_fn(n, dim, |i, j| {
            centres[(i % 30) * dim + j] + rng.random_range(-1.0..1.0)
        })
    }

    /// Mean fraction of the true neighbours recovered.
    ///
    /// ### Params
    ///
    /// * `truth` - Ground-truth neighbours
    /// * `approx` - Neighbours under test
    ///
    /// ### Returns
    ///
    /// Recall in `[0, 1]`
    fn recall(truth: &[Vec<usize>], approx: &[Vec<usize>]) -> f64 {
        let hits: usize = truth
            .iter()
            .zip(approx)
            .map(|(t, a)| a.iter().filter(|j| t.contains(j)).count())
            .sum();
        hits as f64 / truth.iter().map(Vec::len).sum::<usize>() as f64
    }

    /// MLX IVF and CPU IVF at the same nlist / nprobe, against exhaustive.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    fn check_against_cpu(metric: Dist) {
        let (n, dim, k, nlist, nprobe) = (4_000, 24, 10, 40, 4);
        let data = blobs(n, dim, 42);
        let queries = blobs(200, dim, 7);
        let params = KMeansTrainingParams::new(10, None, None);

        let mlx =
            IvfIndexMlx::build(data.as_ref(), metric, Some(nlist), Some(params), 1, false).unwrap();
        let cpu =
            IvfIndex::build(data.as_ref(), metric, Some(nlist), Some(params), 1, false).unwrap();
        let exh = ExhaustiveIndex::new(data.as_ref(), metric);

        let (q_flat, nq, _) = queries.as_ref().into_row_major();
        let truth: Vec<Vec<usize>> = exh
            .query_batch(&q_flat, nq, k, None, false)
            .unwrap()
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let cpu_nn: Vec<Vec<usize>> = (0..nq)
            .map(|i| {
                cpu.query(&q_flat[i * dim..(i + 1) * dim], k, Some(nprobe))
                    .unwrap()
                    .0
            })
            .collect();
        let (mlx_nn, mlx_dist) = mlx
            .query_batch(queries.as_ref(), k, Some(nprobe), None, false)
            .unwrap();

        let (r_mlx, r_cpu) = (recall(&truth, &mlx_nn), recall(&truth, &cpu_nn));
        assert!((r_mlx - r_cpu).abs() < 0.05, "mlx {r_mlx} vs cpu {r_cpu}");
        assert!(mlx_dist
            .iter()
            .all(|row| row.len() == k && row.windows(2).all(|w| w[0] <= w[1])));

        // Probing every cluster is exact.
        let (all_nn, _) = mlx
            .query_batch(queries.as_ref(), k, Some(nlist), Some(64), false)
            .unwrap();
        assert!(recall(&truth, &all_nn) > 0.999);
    }

    #[test]
    fn test_mlx_ivf_euclidean_matches_cpu() {
        check_against_cpu(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_mlx_ivf_cosine_matches_cpu() {
        check_against_cpu(Dist::Cosine);
    }

    #[test]
    fn test_cluster_scan_plan_fits_budgets() {
        for budget in [12 * 1024, 16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024] {
            for align in [4, 8] {
                for dim in [1, 3, 4, 30, 32, 100, 128, 129, 256, 768, 1536] {
                    let plan = plan_cluster_scan(dim, budget, align).unwrap();
                    assert!(
                        cluster_scan_smem_bytes(plan.db) <= budget,
                        "{dim} {budget} {align}"
                    );
                    assert_eq!(plan.db % align, 0);
                    assert!(plan.db <= dim.next_multiple_of(align).min(CLUSTER_SCAN_MAX_DB));
                }
            }
        }
        // Whole rows up to the cap at the Apple budget, blocked beyond it.
        let apple = APPLE_THREADGROUP_BYTES;
        assert_eq!(plan_cluster_scan(128, apple, 4).unwrap().db, 128);
        assert_eq!(plan_cluster_scan(30, apple, 8).unwrap().db, 32);
        assert_eq!(plan_cluster_scan(768, apple, 4).unwrap().db, 256);
        assert_eq!(plan_cluster_scan(768, 12 * 1024, 8).unwrap().db, 128);
        // The score tile alone fills 8 KiB.
        assert!(plan_cluster_scan(32, 8 * 1024, 4).is_none());
    }

    /// Every cluster-major scan against the query-major scan over the same
    /// index and queries.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    /// * `dim` - Dimensionality (above `CLUSTER_SCAN_MAX_DB` splits the
    ///   reduction axis)
    fn check_cluster_major(metric: Dist, dim: usize) {
        let (n, k, nlist, nprobe) = (4_000, 10, 40, 4);
        let data = blobs(n, dim, 42);
        let queries = blobs(300, dim, 7);
        let mut index = IvfIndexMlx::build(data.as_ref(), metric, Some(nlist), None, 1, false)
            .unwrap()
            .with_scan(IvfScanMlx::QueryMajor);
        let (qm_nn, qm_dist) = index
            .query_batch(queries.as_ref(), k, Some(nprobe), None, false)
            .unwrap();
        for scan in [IvfScanMlx::ClusterMajor, IvfScanMlx::ClusterMatrix] {
            index = index.with_scan(scan);
            // A small tile so several tiles run.
            let (cm_nn, cm_dist) = index
                .query_batch(queries.as_ref(), k, Some(nprobe), Some(70), false)
                .unwrap();
            let r = recall(&qm_nn, &cm_nn);
            assert!(r > 0.995, "{scan:?} {metric:?} dim {dim}: overlap {r}");
            for (a, b) in qm_dist.iter().zip(&cm_dist) {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b) {
                    assert!(
                        (x - y).abs() <= 1e-3 * x.abs().max(1.0),
                        "{scan:?} {metric:?} dim {dim}: {x} vs {y}"
                    );
                }
            }
            let (self_nn, _) = index
                .generate_knn(k, Some(nlist), None, false, false)
                .unwrap();
            assert!(self_nn.iter().enumerate().all(|(i, row)| row[0] == i));
        }
    }

    #[test]
    fn test_mlx_ivf_cluster_major_matches_query_major() {
        check_cluster_major(Dist::SquaredEuclidean, 24);
        check_cluster_major(Dist::Cosine, 30);
        check_cluster_major(Dist::SquaredEuclidean, 37);
        check_cluster_major(Dist::SquaredEuclidean, 300);
        check_cluster_major(Dist::Cosine, 300);
    }

    /// Every scan probing every cluster, against CPU exhaustive, over k
    /// values on both sides of the 32-slot boundaries.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    fn check_k_sweep(metric: Dist) {
        let (n, dim, nlist) = (3_000, 20, 30);
        let data = blobs(n, dim, 5);
        let queries = blobs(100, dim, 6);
        let exh = ExhaustiveIndex::new(data.as_ref(), metric);
        let (q_flat, nq, _) = queries.as_ref().into_row_major();
        let mut index =
            IvfIndexMlx::build(data.as_ref(), metric, Some(nlist), None, 1, false).unwrap();
        for k in [1, 10, 15, 16, 31, 32, 33, 50, 64, 100] {
            let truth = exh.query_batch(&q_flat, nq, k, None, false).unwrap();
            let truth_nn: Vec<Vec<usize>> = truth.iter().map(|(i, _)| i.clone()).collect();
            for scan in [
                IvfScanMlx::QueryMajor,
                IvfScanMlx::ClusterMajor,
                IvfScanMlx::ClusterMatrix,
                IvfScanMlx::Auto,
            ] {
                index = index.with_scan(scan);
                let (nn, dist) = index
                    .query_batch(queries.as_ref(), k, Some(nlist), None, false)
                    .unwrap();
                let r = recall(&truth_nn, &nn);
                assert!(r > 0.999, "{scan:?} {metric:?} k {k}: recall {r}");
                for (row, (_, td)) in dist.iter().zip(&truth) {
                    assert_eq!(row.len(), k);
                    for (x, y) in row.iter().zip(td) {
                        assert!(
                            (x - y).abs() <= 1e-3 * y.abs().max(1.0),
                            "{scan:?} {metric:?} k {k}: {x} vs {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_mlx_ivf_k_sweep_euclidean() {
        check_k_sweep(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_mlx_ivf_k_sweep_cosine() {
        check_k_sweep(Dist::Cosine);
    }

    #[test]
    fn test_mlx_ivf_cluster_major_reachability_top_up() {
        let data = blobs(200, 8, 9);
        let index = IvfIndexMlx::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(50),
            None,
            2,
            false,
        )
        .unwrap()
        .with_scan(IvfScanMlx::ClusterMajor);
        let (idx, _) = index
            .query_batch(data.as_ref(), 30, Some(1), None, false)
            .unwrap();
        assert!(idx.iter().all(|row| row.len() == 30));
    }

    #[test]
    fn test_mlx_ivf_self_query() {
        let data = blobs(3_000, 20, 3);
        let index = IvfIndexMlx::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(30),
            None,
            5,
            false,
        )
        .unwrap();
        let (idx, dist) = index.generate_knn(5, Some(30), None, true, false).unwrap();
        let dist = dist.unwrap();
        for (i, row) in idx.iter().enumerate() {
            assert_eq!(row[0], i);
            assert!(dist[i][0] < 1e-4);
        }
    }

    #[test]
    fn test_mlx_ivf_reachability_top_up() {
        // nprobe 1 over tiny clusters: the probe pool must widen to reach k.
        let data = blobs(200, 8, 9);
        let index = IvfIndexMlx::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(50),
            None,
            2,
            false,
        )
        .unwrap();
        let (idx, _) = index
            .query_batch(data.as_ref(), 30, Some(1), None, false)
            .unwrap();
        assert!(idx.iter().all(|row| row.len() == 30));
    }

    #[test]
    fn test_mlx_ivf_rejects_manhattan_and_bad_dim() {
        let data = blobs(100, 4, 1);
        assert!(IvfIndexMlx::build(data.as_ref(), Dist::Manhattan, None, None, 1, false).is_err());
        let index = IvfIndexMlx::build(data.as_ref(), Dist::Cosine, None, None, 1, false).unwrap();
        assert!(index
            .query_batch(blobs(2, 5, 2).as_ref(), 3, None, None, false)
            .is_err());
    }
}
