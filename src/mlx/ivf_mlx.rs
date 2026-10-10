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
//! 4. The scan, one of two (see [`IvfScanMlx`]):
//!    - Query-major ([`IVF_SCAN_BODY`]): one SIMD group per query walks its
//!      probed clusters and keeps the top k in registers. No task list and no
//!      candidate buffer, but members are re-read for every query probing
//!      them.
//!    - Cluster-major: the probe lists are inverted on the device (sort keys,
//!      MLX `argsort`, `scatter_add` counts, a one-thread prefix sum), each
//!      cluster's tasks are cut into tiles that stage member blocks in
//!      threadgroup memory for several queries at once, and a final kernel
//!      merges the per-(query, probe) top k. Like the wgpu layout but with
//!      the task list built on the device and a candidate buffer of k per
//!      probe rather than every member.
//!
//! Distances are computed directly, not through the GEMM expansion, so there
//! is no cancellation.

use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::mlx::exhaustive_mlx::{TOPK_ROWS_PER_GROUP, TOPK_SOURCE};
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

/// Query tasks (SIMD groups) per threadgroup in the cluster-major scan. Each
/// staged member block is reused by this many queries.
///
/// Capped by the pipeline's thread limit, which the compiler lowers with the
/// register-resident top-K: 512 threads failed at K = 15..32 on an M1 Max
/// (limit 384) and only fails at eval, so this stays at the 256 threads the
/// other top-k kernels here run with.
const CLUSTER_SCAN_QT: usize = 8;

/// Members staged per block in the cluster-major scan: one per lane.
const CLUSTER_SCAN_MEMBERS: usize = 32;

/// Cap on the dimensions staged per block in the cluster-major scan; wider
/// rows are split along the reduction axis.
const CLUSTER_SCAN_MAX_DB: usize = 128;

/// Threadgroup memory on Apple GPUs, the budget the cluster-scan plan is
/// sized against. MLX does not expose the device limit.
const APPLE_THREADGROUP_BYTES: usize = 32 * 1024;

/// Threads per threadgroup in the probe-key kernel (one per query).
const PROBE_KEYS_THREADS: usize = 64;

/// Upper bound in bytes on the cluster-major path's per-(query, probe)
/// partial top-k buffers (indices plus distances) for one query tile.
const CLUSTER_PARTIAL_TILE_BYTES: usize = 256 * 1024 * 1024;

/// Metal prelude shared by every top-k kernel here: a sorted top-K in
/// registers (`vals`, `ids`, initialised to `INFINITY`) and `TOPK_INSERT`.
const TOPK_REGS: &str = r#"
    float vals[K];
    uint ids[K];
    for (int i = 0; i < K; i++) { vals[i] = INFINITY; ids[i] = 0; }
    #define TOPK_INSERT(V, J) \
        if ((V) < vals[K - 1]) { \
            vals[K - 1] = (V); \
            ids[K - 1] = (J); \
            for (int i = K - 1; i > 0; i--) { \
                if (vals[i] < vals[i - 1]) { \
                    float tv = vals[i]; vals[i] = vals[i - 1]; vals[i - 1] = tv; \
                    uint ti = ids[i]; ids[i] = ids[i - 1]; ids[i - 1] = ti; \
                } \
            } \
        }
"#;

/// Metal epilogue: K rounds of `simd_min` merge the 32 lanes' lists into
/// output row `orow` of `out_dist` / `out_idx`, ascending.
const TOPK_MERGE: &str = r#"
    for (int r = 0; r < K; r++) {
        float m = simd_min(vals[0]);
        uint w = simd_min(vals[0] == m ? lane : 32u);
        if (lane == w) {
            out_dist[orow * K + r] = vals[0];
            out_idx[orow * K + r] = ids[0];
            for (int i = 0; i < K - 1; i++) { vals[i] = vals[i + 1]; ids[i] = ids[i + 1]; }
            vals[K - 1] = INFINITY;
        }
    }
"#;

/// Query-major cluster scan, between [`TOPK_REGS`] and [`TOPK_MERGE`]. One
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
        for (uint j = s + lane; j < e; j += 32) {
            const device float* x = db + (ulong)j * DIM;
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
            TOPK_INSERT(dist, j);
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

/// Cluster-major step 4, between [`TOPK_REGS`] and [`TOPK_MERGE`]. One
/// threadgroup per tile of up to `QT` tasks probing one cluster, one SIMD
/// group per task. The cluster's members are staged 32 at a time (one per
/// lane) in threadgroup memory, `DB` dimensions per block with a padded row
/// stride so the lane-strided reads miss bank conflicts, and every SIMD group
/// scores its query against the staged block. Each task's top K lands in
/// partial row `query * P + slot`; slots no tile touches are never read.
///
/// The tile -> cluster lookup is a binary search over `tile_off`, so the grid
/// can be sized by an upper bound and surplus threadgroups exit at once.
///
/// Template args: `K`, `P`, `DIM`, `DB`, `QT`, `NLIST`, `COSINE`.
const IVF_CLUSTER_SCAN_BODY: &str = r#"
    uint lane = thread_position_in_threadgroup.x;
    uint sg = thread_position_in_threadgroup.y;
    uint g = threadgroup_position_in_grid.y;
    if (g >= tile_off[NLIST]) return;
    uint lo = 0;
    uint hi = NLIST;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (tile_off[mid] <= g) { lo = mid; } else { hi = mid; }
    }
    uint c = lo;
    uint task = task_off[c] + (g - tile_off[c]) * QT + sg;
    bool active = task < task_off[c + 1];
    uint flat = active ? order[task] : 0;
    ulong orow = flat;
    const device float* qv = q + (ulong)(flat / P) * DIM;
    uint ms = offsets[c];
    uint me = offsets[c + 1];
    uint tid = sg * 32 + lane;
    threadgroup float tile[32 * (DB + 1)];
    for (uint cs = ms; cs < me; cs += 32) {
        float acc = 0.0f;
        for (uint blk = 0; blk < (uint)DIM; blk += DB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < 32u * DB; e += 32u * QT) {
                uint m = e / DB;
                uint d = e % DB;
                uint row = cs + m;
                uint col = blk + d;
                tile[m * (DB + 1) + d] =
                    (row < me && col < (uint)DIM) ? db[(ulong)row * DIM + col] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                uint dn = min((uint)DB, (uint)DIM - blk);
                const threadgroup float* xr = tile + lane * (DB + 1);
                for (uint d = 0; d < dn; d++) {
                    float qd = qv[blk + d];
                    if (COSINE) {
                        acc = fma(qd, xr[d], acc);
                    } else {
                        float t = qd - xr[d];
                        acc = fma(t, t, acc);
                    }
                }
            }
        }
        uint j = cs + lane;
        if (active && j < me) {
            float dist = COSINE ? 1.0f - acc : acc;
            TOPK_INSERT(dist, j);
        }
    }
    if (!active) return;
"#;

/// Cluster-major step 5, between [`TOPK_REGS`] and [`TOPK_MERGE`]: one SIMD
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
    for (uint i = lane; i < n; i += 32) {
        TOPK_INSERT(part_dist[base + i], part_idx[base + i]);
    }
"#;

/// Which scan the IVF MLX query runs. Both give the same candidates; they
/// differ in memory access pattern.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IvfScanMlx {
    /// One SIMD group per query walks its probed clusters; members are read
    /// from device memory once per query that probes them.
    #[default]
    QueryMajor,
    /// The probe lists are inverted on the device into per-cluster task lists
    /// and each staged member block is scored against up to
    /// [`CLUSTER_SCAN_QT`] queries; per-(query, probe) top-k are merged after.
    ClusterMajor,
}

/// Threadgroup memory plan for the cluster-major scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClusterScanPlan {
    /// Dimensions staged per block
    pub db: usize,
    /// Tasks (SIMD groups) per threadgroup
    pub qt: usize,
}

/// Threadgroup bytes the cluster-major scan stages for a given block width.
///
/// ### Params
///
/// * `db` - Dimensions per block
///
/// ### Returns
///
/// Bytes of threadgroup memory
pub(crate) fn cluster_scan_smem_bytes(db: usize) -> usize {
    CLUSTER_SCAN_MEMBERS * (db + 1) * size_of::<f32>()
}

/// Size the cluster-major scan's staging against a threadgroup budget.
///
/// Blocks the reduction axis rather than shrinking the member block, so the
/// footprint is independent of `dim`.
///
/// ### Params
///
/// * `dim` - Embedding dimensionality
/// * `smem_bytes` - Threadgroup memory budget
///
/// ### Returns
///
/// The plan, or `None` if not even a 4-wide block fits
pub(crate) fn plan_cluster_scan(dim: usize, smem_bytes: usize) -> Option<ClusterScanPlan> {
    let fit = (smem_bytes / (CLUSTER_SCAN_MEMBERS * size_of::<f32>())).checked_sub(1)?;
    let db = if dim <= fit.min(CLUSTER_SCAN_MAX_DB) {
        dim
    } else {
        fit.min(CLUSTER_SCAN_MAX_DB) / 4 * 4
    };
    debug_assert!(cluster_scan_smem_bytes(db) <= smem_bytes);
    (db > 0).then_some(ClusterScanPlan {
        db,
        qt: CLUSTER_SCAN_QT,
    })
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
    probe_kernel: MetalKernel,
    /// Query-major scan kernel, see [`IVF_SCAN_BODY`]
    scan_kernel: MetalKernel,
    /// Which scan the queries run
    scan: IvfScanMlx,
    /// Cluster-major staging plan; `None` falls back to the query-major scan
    cluster_plan: Option<ClusterScanPlan>,
    /// Cluster-major probe-key kernel, see [`IVF_PROBE_KEYS_SOURCE`]
    keys_kernel: MetalKernel,
    /// Cluster-major task CSR kernel, see [`IVF_TASK_CSR_SOURCE`]
    csr_kernel: MetalKernel,
    /// Cluster-major scan kernel, see [`IVF_CLUSTER_SCAN_BODY`]
    cluster_scan_kernel: MetalKernel,
    /// Cluster-major partial merge kernel, see [`IVF_PARTIAL_MERGE_BODY`]
    merge_kernel: MetalKernel,
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

        let db = Array::from_f32(&vectors_flat, &[n as i32, dim as i32]);
        let offsets_u32: Vec<u32> = cluster_offsets.iter().map(|&o| o as u32).collect();
        let offsets = Array::from_u32(&offsets_u32, &[nlist as i32 + 1]);
        let c = Array::from_f32(&centroids, &[nlist as i32, dim as i32]);
        let centroid_ops = CentroidOperands::new(&c, nlist, &metric, &stream)?;
        eval_all(&[&db, &offsets, &centroid_ops.ct, &centroid_ops.add], false)?;

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
            probe_kernel: MetalKernel::new(
                "ivf_probe_topk",
                &["d"],
                &["out_idx", "out_dist"],
                TOPK_SOURCE,
            ),
            scan_kernel: MetalKernel::new(
                "ivf_scan",
                &["q", "db", "offsets", "probe", "order", "params"],
                &["out_idx", "out_dist"],
                &format!("{TOPK_REGS}{IVF_SCAN_BODY}{TOPK_MERGE}"),
            ),
            scan: IvfScanMlx::default(),
            cluster_plan: plan_cluster_scan(dim, APPLE_THREADGROUP_BYTES),
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
            cluster_scan_kernel: MetalKernel::new(
                "ivf_cluster_scan",
                &["q", "db", "offsets", "order", "task_off", "tile_off"],
                &["out_idx", "out_dist"],
                &format!("{TOPK_REGS}{IVF_CLUSTER_SCAN_BODY}{TOPK_MERGE}"),
            ),
            merge_kernel: MetalKernel::new(
                "ivf_partial_merge",
                &["part_idx", "part_dist", "eff"],
                &["out_idx", "out_dist"],
                &format!("{TOPK_REGS}{IVF_PARTIAL_MERGE_BODY}{TOPK_MERGE}"),
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
    ) -> Result<(Array, Array), AnnSearchErrors> {
        let s = &self.stream;
        let q = Array::from_f32(q, &[n_q as i32, self.dim as i32]);
        let ops = &self.centroid_ops;
        let scores = Array::addmm(&ops.add, &q, &ops.ct, ops.alpha, 1.0, s)?;

        let pool_shape = [n_q as i32, pool as i32];
        let mut probe = self.probe_kernel.apply(
            &[&scores],
            &[
                OutputSpec {
                    shape: &pool_shape,
                    dtype: MLX_UINT32,
                },
                OutputSpec {
                    shape: &pool_shape,
                    dtype: MLX_FLOAT32,
                },
            ],
            [32, n_q as i32, 1],
            [32, TOPK_ROWS_PER_GROUP, 1],
            &[
                ("K", pool as i32),
                ("N", self.nlist as i32),
                ("VEC4", self.nlist.is_multiple_of(4) as i32),
            ],
            s,
        )?;
        probe.truncate(1);
        let probe = probe.pop().expect("kernel has two outputs");

        if let (IvfScanMlx::ClusterMajor, Some(plan)) = (self.scan, self.cluster_plan) {
            return self.queue_cluster_major(&q, &probe, n_q, k, nprobe, pool, plan);
        }

        let order = probe
            .slice(&[0, 0], &[n_q as i32, 1], s)?
            .reshape(&[n_q as i32], s)?
            .argsort_axis(0, s)?;
        let params = Array::from_u32(&[nprobe as u32], &[1]);

        let shape = [n_q as i32, k as i32];
        let mut out = self.scan_kernel.apply(
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
    /// * `probe` - Probe pool per query, `[n_q, pool]` u32, ascending
    /// * `n_q` - Rows in the tile
    /// * `k` - Number of neighbours, at most `n`
    /// * `nprobe` - Clusters to probe before the reachability top-up
    /// * `pool` - Probe pool width
    /// * `plan` - Threadgroup staging plan
    ///
    /// ### Returns
    ///
    /// Lazy `(positions as u32, distances)`, both `[n_q, k]`, ascending
    #[allow(clippy::too_many_arguments)]
    fn queue_cluster_major(
        &self,
        q: &Array,
        probe: &Array,
        n_q: usize,
        k: usize,
        nprobe: usize,
        pool: usize,
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
        let mut part = self.cluster_scan_kernel.apply(
            &[q, &self.db, &self.offsets, &order, &task_off, &tile_off],
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
                ("DB", plan.db as i32),
                ("QT", plan.qt as i32),
                ("NLIST", nlist),
                ("COSINE", (self.metric == Dist::Cosine) as i32),
            ],
            s,
        )?;
        let part_dist = part.pop().expect("kernel has two outputs");
        let part_idx = part.pop().expect("kernel has two outputs");

        let shape = [n_q as i32, k as i32];
        let mut out = self.merge_kernel.apply(
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
        if self.scan == IvfScanMlx::ClusterMajor {
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
        for budget in [4 * 1024, 16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024] {
            for dim in [1, 3, 4, 30, 32, 100, 128, 129, 256, 768, 1536] {
                let plan = plan_cluster_scan(dim, budget).unwrap();
                assert!(cluster_scan_smem_bytes(plan.db) <= budget, "{dim} {budget}");
                assert!(plan.db >= 1 && plan.db <= dim.min(CLUSTER_SCAN_MAX_DB));
                // A split reduction axis keeps float4-aligned blocks.
                assert!(plan.db == dim || plan.db % 4 == 0);
            }
        }
        // Whole rows up to the cap at the Apple budget, blocked beyond it.
        assert_eq!(
            plan_cluster_scan(128, APPLE_THREADGROUP_BYTES).unwrap().db,
            128
        );
        assert_eq!(
            plan_cluster_scan(768, APPLE_THREADGROUP_BYTES).unwrap().db,
            128
        );
        assert_eq!(plan_cluster_scan(128, 4 * 1024).unwrap().db, 28);
        assert!(plan_cluster_scan(32, 64).is_none());
    }

    /// Cluster-major and query-major scans over the same index and queries.
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
        let index = IvfIndexMlx::build(data.as_ref(), metric, Some(nlist), None, 1, false).unwrap();
        let (qm_nn, qm_dist) = index
            .query_batch(queries.as_ref(), k, Some(nprobe), None, false)
            .unwrap();
        let index = index.with_scan(IvfScanMlx::ClusterMajor);
        // A small tile so several tiles run.
        let (cm_nn, cm_dist) = index
            .query_batch(queries.as_ref(), k, Some(nprobe), Some(70), false)
            .unwrap();
        let r = recall(&qm_nn, &cm_nn);
        assert!(r > 0.995, "{metric:?} dim {dim}: overlap {r}");
        for (a, b) in qm_dist.iter().zip(&cm_dist) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= 1e-3 * x.abs().max(1.0), "{x} vs {y}");
            }
        }
        let (self_nn, _) = index
            .generate_knn(k, Some(nlist), None, false, false)
            .unwrap();
        assert!(self_nn.iter().enumerate().all(|(i, row)| row[0] == i));
    }

    #[test]
    fn test_mlx_ivf_cluster_major_matches_query_major() {
        check_cluster_major(Dist::SquaredEuclidean, 24);
        check_cluster_major(Dist::Cosine, 30);
        check_cluster_major(Dist::SquaredEuclidean, 200);
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
