//! Exhaustive (flat) index on MLX.
//!
//! Per query tile MLX's GEMM materialises the `tile x n` distance matrix, then
//! a custom Metal kernel ([`RowTopK`]) pulls the k smallest per row.
//! MLX's own `argpartition` was far too slow on 100k-wide rows, even split
//! into blocks. Tiles are queued asynchronously so the host never waits
//! between them.
//!
//! Euclidean uses the expansion `|q|^2 + |x|^2 - 2 q.x`, so near-ties can swap
//! through cancellation; there is no exact re-rank. Cosine runs on unit
//! vectors as `1 - q.x`.

use rayon::prelude::*;
use std::time::Instant;

use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::utils::DimensionValidation;

////////////
// Consts //
////////////

/// Upper bound in bytes on one query tile's `tile x n` f32 distance matrix.
/// Small tiles drown in per-tile overhead; larger ones gained little warm and
/// cost more on the cold call.
const MLX_DIST_TILE_BYTES: usize = 512 * 1024 * 1024;

/// Rows per threadgroup in the top-k kernel; one SIMD group per row.
pub(crate) const TOPK_ROWS_PER_GROUP: i32 = 8;

/// Largest K served by the per-lane register top-k ([`REG_TOPK_REGS`]); above
/// it the SIMD-cooperative list ([`SG_TOPK_REGS`]) runs. The register list is
/// slightly ahead at small K but collapses once `vals[K]` / `ids[K]` spill
/// (seen from K = 31 on); 17 to 30 were not measured, so the switch sits at
/// the last K known to be safe.
const REG_TOPK_MAX_K: usize = 16;

/// Metal prelude of the per-lane register top-K: each lane keeps a sorted
/// `vals[K]` / `ids[K]` and bubbles new entries in. One compare rejects
/// almost everything once the list is warm. Exposes the same `SG_OFFER`
/// macro as [`SG_TOPK_REGS`] so the kernel bodies are shared.
const REG_TOPK_REGS: &str = r#"
    float vals[K];
    uint ids[K];
    for (int i = 0; i < K; i++) { vals[i] = INFINITY; ids[i] = 0; }
    #define SG_OFFER(OK, D, ID) { \
        float v_ = (D); \
        if ((OK) && v_ < vals[K - 1]) { \
            vals[K - 1] = v_; \
            ids[K - 1] = (ID); \
            for (int i = K - 1; i > 0; i--) { \
                if (vals[i] < vals[i - 1]) { \
                    float tv = vals[i]; vals[i] = vals[i - 1]; vals[i - 1] = tv; \
                    uint ti = ids[i]; ids[i] = ids[i - 1]; ids[i - 1] = ti; \
                } \
            } \
        } \
    }
"#;

/// Metal epilogue of the register top-K: K rounds of `simd_min` merge the 32
/// lanes' lists into row `orow` of `out_dist` / `out_idx`, ascending.
const REG_TOPK_STORE: &str = r#"
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

/// Metal prelude of the SIMD-cooperative top-K. The list is spread over the
/// SIMD group: `SPL = ceil(K / 32)` slots per lane, slot `b` on lane `j`
/// holding element `b * 32 + j`, ascending. `thr` is the K-th value,
/// uniform across the group. Needs `lane` in scope and uniform control flow
/// at every `SG_OFFER`.
///
/// `SG_OFFER(OK, D, ID)` offers one candidate per lane: a ballot keeps the
/// lanes under the threshold and each survivor, lowest lane first, is ranked
/// with `simd_sum` and inserted by a `simd_shuffle_up` shift (the CAGRA
/// beam's scheme), refreshing the threshold. A warm list rejects a whole
/// round of 32 on one ballot, and an insert costs `O(SPL)` SIMD ops rather
/// than an `O(K)` serial bubble per lane.
const SG_TOPK_REGS: &str = r#"
    constexpr int SPL = (K + 31) / 32;
    float tv[SPL];
    uint ti[SPL];
    for (int b = 0; b < SPL; b++) { tv[b] = INFINITY; ti[b] = 0; }
    float thr = INFINITY;
    #define SG_INSERT(D, ID) { \
        uint pos_ = 0; \
        for (int b = 0; b < SPL; b++) pos_ += (uint)(tv[b] <= (D)); \
        pos_ = simd_sum(pos_); \
        for (int b = SPL - 1; b >= 0; b--) { \
            float ud_ = simd_shuffle_up(tv[b], (ushort)1); \
            uint ui_ = simd_shuffle_up(ti[b], (ushort)1); \
            if (b > 0) { \
                float cd_ = simd_shuffle(tv[b > 0 ? b - 1 : 0], (ushort)31); \
                uint ci_ = simd_shuffle(ti[b > 0 ? b - 1 : 0], (ushort)31); \
                if (lane == 0) { ud_ = cd_; ui_ = ci_; } \
            } \
            uint slot_ = (uint)b * 32 + lane; \
            if (slot_ > pos_) { tv[b] = ud_; ti[b] = ui_; } \
            else if (slot_ == pos_) { tv[b] = (D); ti[b] = (ID); } \
        } \
        thr = simd_shuffle(tv[SPL - 1], (ushort)((K - 1) & 31)); \
    }
    #define SG_OFFER(OK, D, ID) { \
        float d_ = (D); \
        uint id_ = (ID); \
        uint m_ = (uint)(ulong)simd_ballot((OK) && d_ < thr); \
        while (m_ != 0) { \
            ushort src_ = (ushort)ctz(m_); \
            m_ &= m_ - 1; \
            float dd_ = simd_shuffle(d_, src_); \
            uint ii_ = simd_shuffle(id_, src_); \
            if (dd_ < thr) SG_INSERT(dd_, ii_); \
        } \
    }
"#;

/// Metal epilogue of the SIMD-cooperative top-K: the list is already sorted,
/// so each lane writes its slots below K to row `orow` of `out_dist` /
/// `out_idx`.
const SG_TOPK_STORE: &str = r#"
    for (int b = 0; b < SPL; b++) {
        uint slot = (uint)b * 32 + lane;
        if (slot < (uint)K) {
            out_dist[orow * K + slot] = tv[b];
            out_idx[orow * K + slot] = ti[b];
        }
    }
"#;

/// Opening of the row top-k: lane, row and the row pointer.
const ROW_TOPK_HEAD: &str = r#"
    uint lane = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    ulong orow = row;
    const device float* drow = d + (ulong)row * N;
"#;

/// Row top-k body. One SIMD group per row, rounds of 32 candidates (128 with
/// `float4` loads, which was the larger of the two original kernel wins).
/// Template args: `K`, row length `N`, `VEC4` (`N % 4 == 0`).
const ROW_TOPK_BODY: &str = r#"
    if (VEC4) {
        const device float4* drow4 = (const device float4*)drow;
        for (uint base = 0; base < N / 4; base += 32) {
            uint j = base + lane;
            bool ok = j < N / 4;
            float4 v = ok ? drow4[j] : float4(INFINITY);
            SG_OFFER(ok, v.x, 4 * j);
            SG_OFFER(ok, v.y, 4 * j + 1);
            SG_OFFER(ok, v.z, 4 * j + 2);
            SG_OFFER(ok, v.w, 4 * j + 3);
        }
        for (uint base = (N / 4) * 4; base < N; base += 32) {
            uint j = base + lane;
            bool ok = j < N;
            SG_OFFER(ok, ok ? drow[j] : INFINITY, j);
        }
    } else {
        for (uint base = 0; base < N; base += 32) {
            uint j = base + lane;
            bool ok = j < N;
            SG_OFFER(ok, ok ? drow[j] : INFINITY, j);
        }
    }
"#;

/// A top-k kernel compiled twice, once per top-K flavour, picked by K.
///
/// The body offers candidates with `SG_OFFER(ok, dist, id)` under uniform
/// control flow and must define `lane` and `orow` (the output row); the
/// flavour supplies the list and the store to `out_idx` / `out_dist`.
pub(crate) struct TopKKernel {
    /// Per-lane register flavour, `K <= REG_TOPK_MAX_K`
    reg: MetalKernel,
    /// SIMD-cooperative flavour
    sg: MetalKernel,
}

impl TopKKernel {
    /// Compile both flavours of `head`, list, `body`, store.
    ///
    /// ### Params
    ///
    /// * `name` - Kernel name prefix
    /// * `inputs` - Input buffer names
    /// * `outputs` - Output buffer names, ending in `out_idx`, `out_dist`
    /// * `header` - Metal header (includes), may be empty
    /// * `head` - Source before the list declaration
    /// * `body` - Source between the list and the store
    ///
    /// ### Returns
    ///
    /// The pair
    pub(crate) fn new(
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        header: &str,
        head: &str,
        body: &str,
    ) -> Self {
        let build = |suffix: &str, regs: &str, store: &str| {
            MetalKernel::with_header(
                &format!("{name}_{suffix}"),
                inputs,
                outputs,
                header,
                &format!("{head}{regs}{body}{store}"),
                false,
            )
        };
        Self {
            reg: build("reg", REG_TOPK_REGS, REG_TOPK_STORE),
            sg: build("sg", SG_TOPK_REGS, SG_TOPK_STORE),
        }
    }

    /// The flavour for a given K.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours kept
    ///
    /// ### Returns
    ///
    /// The kernel to apply
    pub(crate) fn pick(&self, k: usize) -> &MetalKernel {
        if k <= REG_TOPK_MAX_K {
            &self.reg
        } else {
            &self.sg
        }
    }
}

/// Row top-k over a row-major `[rows, n]` matrix, see [`ROW_TOPK_BODY`].
pub(crate) struct RowTopK {
    /// The kernel pair
    kernel: TopKKernel,
}

impl RowTopK {
    /// Register both flavours.
    ///
    /// ### Params
    ///
    /// * `name` - Kernel name prefix
    ///
    /// ### Returns
    ///
    /// The row top-k
    pub(crate) fn new(name: &str) -> Self {
        Self {
            kernel: TopKKernel::new(
                name,
                &["d"],
                &["out_idx", "out_dist"],
                "",
                ROW_TOPK_HEAD,
                ROW_TOPK_BODY,
            ),
        }
    }

    /// Queue the k smallest of every row.
    ///
    /// ### Params
    ///
    /// * `d` - The `[rows, n]` matrix
    /// * `rows` - Number of rows
    /// * `n` - Row length
    /// * `k` - Number kept, at most `n`
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// Lazy `(indices as u32, values)`, both `[rows, k]`, ascending
    pub(crate) fn apply(
        &self,
        d: &Array,
        rows: usize,
        n: usize,
        k: usize,
        s: &Stream,
    ) -> Result<(Array, Array), AnnSearchErrors> {
        let shape = [rows as i32, k as i32];
        let mut out = self.kernel.pick(k).apply(
            &[d],
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
            [32, rows as i32, 1],
            [32, TOPK_ROWS_PER_GROUP, 1],
            &[
                ("K", k as i32),
                ("N", n as i32),
                // float4 rows need 16-byte aligned row starts.
                ("VEC4", n.is_multiple_of(4) as i32),
            ],
            s,
        )?;
        let dist = out.pop().expect("kernel has two outputs");
        let idx = out.pop().expect("kernel has two outputs");
        Ok((idx, dist))
    }
}

////////////////////////
// ExhaustiveIndexMlx //
////////////////////////

/// Exhaustive (brute-force) nearest neighbour index on MLX. f32 only.
///
/// Holds an MLX stream, which is thread affine: build and query on the same
/// thread. The raw handles keep this type `!Send` and `!Sync`.
pub struct ExhaustiveIndexMlx {
    /// Row-major vectors on the host, unit-normalised for Cosine. Kept for
    /// the self-kNN query.
    pub vectors_flat: Vec<f32>,
    /// Embedding dimensionality
    pub dim: usize,
    /// Number of samples
    pub n: usize,
    /// Distance metric the index is configured for
    metric: Dist,
    /// Database on the device as a `[dim, n]` transposed view
    db_t: Array,
    /// Additive GEMM term: `|x|^2` as `[1, n]` for Euclidean, the scalar 1 for
    /// Cosine
    add_term: Array,
    /// Row top-k kernels, see [`RowTopK`]
    topk: RowTopK,
    /// Stream every op runs on. Declared last so it drops after the arrays.
    stream: Stream,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl DimensionValidation for ExhaustiveIndexMlx {
    fn dim(&self) -> usize {
        self.dim
    }
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

/// Squared L2 norm of every row.
///
/// ### Params
///
/// * `data` - Row-major vectors
/// * `dim` - Row length
///
/// ### Returns
///
/// One squared norm per row
fn squared_norms(data: &[f32], dim: usize) -> Vec<f32> {
    data.par_chunks_exact(dim)
        .map(|row| f32::dot_simd(row, row))
        .collect()
}

/////////////////////////
// Main implementation //
/////////////////////////

impl ExhaustiveIndexMlx {
    /// Generate a new exhaustive index on MLX and upload the database.
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Distance metric. Manhattan is not supported.
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index, database resident on the device
    pub fn new(data: impl AnnMatrix<f32>, metric: Dist) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        install_error_handler();

        let (mut vectors_flat, n, dim) = data.into_row_major();
        let stream = Stream::default_gpu();

        let add_term = if metric == Dist::Cosine {
            normalise_rows(&mut vectors_flat, dim);
            Array::scalar_f32(1.0)
        } else {
            Array::from_f32(&squared_norms(&vectors_flat, dim), &[1, n as i32])
        };

        let db = Array::from_f32(&vectors_flat, &[n as i32, dim as i32]);
        let db_t = db.transpose(&stream)?;
        eval_all(&[&db_t, &add_term], false)?;

        Ok(Self {
            vectors_flat,
            dim,
            n,
            metric,
            db_t,
            add_term,
            topk: RowTopK::new("row_topk"),
            stream,
        })
    }

    /// Query the exhaustive index
    ///
    /// ### Params
    ///
    /// * `query_mat` - The samples x features matrix to query
    /// * `k` - Number of neighbours to return, clamped to the index size
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// A tuple of `(Vec<indices>, Vec<distances>)`, ascending by distance
    pub fn query_batch(
        &self,
        query_mat: impl AnnMatrix<f32>,
        k: usize,
        verbose: bool,
    ) -> KnnResult<f32> {
        let (mut queries, n_query, dim_query) = query_mat.into_row_major();
        self.check_dim(dim_query)?;
        if self.metric == Dist::Cosine {
            normalise_rows(&mut queries, dim_query);
        }
        self.query_prepared(&queries, n_query, k, verbose)
    }

    /// Generate the kNN graph of the indexed vectors against themselves.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector, self included
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`, one row per vector
    pub fn generate_knn(&self, k: usize, return_dist: bool, verbose: bool) -> KnnOptionResult<f32> {
        let (indices, distances) = self.query_prepared(&self.vectors_flat, self.n, k, verbose)?;
        Ok((indices, return_dist.then_some(distances)))
    }

    /// Queue GEMM + top-k for one query tile.
    ///
    /// ### Params
    ///
    /// * `q` - Row-major query tile
    /// * `n_q` - Rows in the tile
    /// * `k` - Number of neighbours, at most `n`
    /// * `alpha` - GEMM scale: -2 for Euclidean, -1 for Cosine
    ///
    /// ### Returns
    ///
    /// Lazy `(indices as u32, distances)`, both `[n_q, k]` and ascending
    fn queue_tile(
        &self,
        q: &[f32],
        n_q: usize,
        k: usize,
        alpha: f32,
    ) -> Result<(Array, Array), AnnSearchErrors> {
        let s = &self.stream;
        let q = Array::from_f32(q, &[n_q as i32, self.dim as i32]);
        let d = Array::addmm(&self.add_term, &q, &self.db_t, alpha, 1.0, s)?;
        self.topk.apply(&d, n_q, self.n, k, s)
    }

    /// Run the tiled GEMM + top-k over already prepared queries.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, unit-normalised for Cosine
    /// * `n_query` - Number of queries
    /// * `k` - Number of neighbours, clamped to the index size
    /// * `verbose` - Print the queue versus wait time split
    ///
    /// ### Returns
    ///
    /// A tuple of `(Vec<indices>, Vec<distances>)`, ascending by distance
    fn query_prepared(
        &self,
        queries: &[f32],
        n_query: usize,
        k: usize,
        verbose: bool,
    ) -> KnnResult<f32> {
        let k = k.min(self.n);
        if k == 0 {
            return Ok((vec![Vec::new(); n_query], vec![Vec::new(); n_query]));
        }

        let dim = self.dim;
        let tile = (MLX_DIST_TILE_BYTES / (self.n * size_of::<f32>())).max(1);
        // Euclidean ranks on `|x|^2 - 2 q.x`; `|q|^2` is constant per row, so
        // it only gets added to the k survivors on the host.
        let (alpha, q_sq) = match self.metric {
            Dist::Cosine => (-1.0, Vec::new()),
            _ => (-2.0, squared_norms(queries, dim)),
        };

        let t0 = Instant::now();
        let mut pending = Vec::with_capacity(n_query.div_ceil(tile));
        for start in (0..n_query).step_by(tile) {
            let end = (start + tile).min(n_query);
            let (idx, dist) =
                self.queue_tile(&queries[start * dim..end * dim], end - start, k, alpha)?;
            eval_all(&[&idx, &dist], true)?;
            pending.push((idx, dist));
        }
        let t1 = Instant::now();

        let mut all_indices = Vec::with_capacity(n_query);
        let mut all_distances = Vec::with_capacity(n_query);
        for (idx, dist) in &pending {
            eval_all(&[idx, dist], false)?;
            let start = all_indices.len();
            let rows: Vec<(Vec<usize>, Vec<f32>)> = idx
                .as_u32()?
                .par_chunks_exact(k)
                .zip(dist.as_f32()?.par_chunks_exact(k))
                .enumerate()
                .map(|(i, (ir, dr))| {
                    let shift = q_sq.get(start + i).copied().unwrap_or(0.0);
                    (
                        ir.iter().map(|&j| j as usize).collect(),
                        dr.iter().map(|&v| (v + shift).max(0.0)).collect(),
                    )
                })
                .collect();
            for (ir, dr) in rows {
                all_indices.push(ir);
                all_distances.push(dr);
            }
        }

        if verbose {
            println!(
                "MLX exhaustive: {} queries in tiles of {}, queue {:.1} ms, wait+copy {:.1} ms",
                n_query,
                tile,
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }

        Ok((all_indices, all_distances))
    }

    /// Returns the size of the index in bytes, host copy plus device copy
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        let host = self.vectors_flat.capacity() * size_of::<f32>();
        let device = (self.n * self.dim + self.n) * size_of::<f32>();
        std::mem::size_of_val(self) + host + device
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exhaustive::ExhaustiveIndex;
    use approx::assert_relative_eq;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Random matrix with entries in `[-1, 1)`.
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
    fn random_mat(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Mat::from_fn(n, dim, |_, _| rng.random_range(-1.0..1.0))
    }

    /// Compare the MLX index against the CPU exhaustive index on one metric.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    /// * `k` - Neighbours per query
    fn check_against_cpu(metric: Dist, k: usize) {
        let (n, dim) = (5_000, 24);
        let data = random_mat(n, dim, 42);
        let queries = random_mat(50, dim, 7);

        let mlx = ExhaustiveIndexMlx::new(data.as_ref(), metric).unwrap();
        let cpu = ExhaustiveIndex::new(data.as_ref(), metric);

        let (mlx_idx, mlx_dist) = mlx.query_batch(queries.as_ref(), k, false).unwrap();
        let (q_flat, nq, _) = queries.as_ref().into_row_major();
        let cpu_res = cpu.query_batch(&q_flat, nq, k, None, false).unwrap();

        let mut hits = 0;
        for (i, (cpu_idx, cpu_dist)) in cpu_res.iter().enumerate() {
            assert_eq!(mlx_idx[i].len(), k);
            hits += mlx_idx[i].iter().filter(|j| cpu_idx.contains(j)).count();
            for (a, b) in mlx_dist[i].iter().zip(cpu_dist) {
                assert_relative_eq!(*a, *b, epsilon = 1e-3, max_relative = 1e-3);
            }
            assert!(mlx_dist[i].windows(2).all(|w| w[0] <= w[1]));
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(recall > 0.99, "k {k}: recall {recall}");
    }

    /// The k values the top-k kernels are swept over: both sides of the
    /// 32-slot boundaries and the small-k dispatch.
    const K_SWEEP: [usize; 10] = [1, 10, 15, 16, 31, 32, 33, 50, 64, 100];

    #[test]
    fn test_mlx_exhaustive_euclidean_matches_cpu() {
        for k in K_SWEEP {
            check_against_cpu(Dist::SquaredEuclidean, k);
        }
    }

    #[test]
    fn test_mlx_exhaustive_cosine_matches_cpu() {
        for k in K_SWEEP {
            check_against_cpu(Dist::Cosine, k);
        }
    }

    #[test]
    fn test_mlx_exhaustive_self_query_finds_self() {
        let data = random_mat(500, 16, 3);
        let index = ExhaustiveIndexMlx::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();
        let (idx, dist) = index.generate_knn(1, true, false).unwrap();
        let dist = dist.unwrap();
        for (i, row) in idx.iter().enumerate() {
            assert_eq!(row[0], i);
            assert!(dist[i][0] < 1e-4);
        }
    }

    #[test]
    fn test_mlx_exhaustive_k_larger_than_n_clamps() {
        let data = random_mat(8, 4, 1);
        let index = ExhaustiveIndexMlx::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();
        let (idx, _) = index.query_batch(data.as_ref(), 20, false).unwrap();
        assert!(idx.iter().all(|row| row.len() == 8));
    }

    #[test]
    fn test_mlx_exhaustive_rejects_manhattan_and_bad_dim() {
        let data = random_mat(8, 4, 1);
        assert!(ExhaustiveIndexMlx::new(data.as_ref(), Dist::Manhattan).is_err());
        let index = ExhaustiveIndexMlx::new(data.as_ref(), Dist::Cosine).unwrap();
        assert!(index
            .query_batch(random_mat(2, 5, 2).as_ref(), 3, false)
            .is_err());
    }
}
