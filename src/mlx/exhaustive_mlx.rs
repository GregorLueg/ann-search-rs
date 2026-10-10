//! Exhaustive (flat) index on MLX.
//!
//! Per query tile MLX's GEMM materialises the `tile x n` distance matrix, then
//! a custom Metal kernel ([`TOPK_SOURCE`]) pulls the k smallest per row.
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

/// Metal body of the row top-k. One SIMD group per row: each lane scans a
/// stride-32 slice keeping a sorted top-K in registers (one compare rejects
/// almost everything once the list is warm), then K rounds of `simd_min` merge
/// the 32 lists in ascending order. Template args: `K`, row length `N`, and
/// `VEC4` (rows read as `float4`, needs `N % 4 == 0`), which was the larger of
/// the two kernel wins.
pub(crate) const TOPK_SOURCE: &str = r#"
    uint lane = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    const device float* drow = d + (ulong)row * N;
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
    if (VEC4) {
        const device float4* drow4 = (const device float4*)drow;
        for (uint j = lane; j < N / 4; j += 32) {
            float4 v = drow4[j];
            TOPK_INSERT(v.x, 4 * j);
            TOPK_INSERT(v.y, 4 * j + 1);
            TOPK_INSERT(v.z, 4 * j + 2);
            TOPK_INSERT(v.w, 4 * j + 3);
        }
        for (uint j = (N / 4) * 4 + lane; j < N; j += 32) {
            TOPK_INSERT(drow[j], j);
        }
    } else {
        for (uint j = lane; j < N; j += 32) {
            TOPK_INSERT(drow[j], j);
        }
    }
    for (int r = 0; r < K; r++) {
        float m = simd_min(vals[0]);
        uint w = simd_min(vals[0] == m ? lane : 32u);
        if (lane == w) {
            out_dist[(ulong)row * K + r] = vals[0];
            out_idx[(ulong)row * K + r] = ids[0];
            for (int i = 0; i < K - 1; i++) { vals[i] = vals[i + 1]; ids[i] = ids[i + 1]; }
            vals[K - 1] = INFINITY;
        }
    }
"#;

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
    /// Row top-k kernel, see [`TOPK_SOURCE`]
    topk_kernel: MetalKernel,
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
            topk_kernel: MetalKernel::new(
                "row_topk",
                &["d"],
                &["out_idx", "out_dist"],
                TOPK_SOURCE,
            ),
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
        let shape = [n_q as i32, k as i32];
        let mut out = self.topk_kernel.apply(
            &[&d],
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
            &[
                ("K", k as i32),
                ("N", self.n as i32),
                // float4 rows need 16-byte aligned row starts.
                ("VEC4", self.n.is_multiple_of(4) as i32),
            ],
            s,
        )?;
        let dist = out.pop().expect("kernel has two outputs");
        let idx = out.pop().expect("kernel has two outputs");
        Ok((idx, dist))
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
    fn check_against_cpu(metric: Dist) {
        let (n, dim, k) = (5_000, 24, 10);
        let data = random_mat(n, dim, 42);
        let queries = random_mat(50, dim, 7);

        let mlx = ExhaustiveIndexMlx::new(data.as_ref(), metric).unwrap();
        let cpu = ExhaustiveIndex::new(data.as_ref(), metric);

        let (mlx_idx, mlx_dist) = mlx.query_batch(queries.as_ref(), k, false).unwrap();
        let (q_flat, nq, _) = queries.as_ref().into_row_major();
        let cpu_res = cpu.query_batch(&q_flat, nq, k, None, false).unwrap();

        let mut hits = 0;
        for (i, (cpu_idx, cpu_dist)) in cpu_res.iter().enumerate() {
            hits += mlx_idx[i].iter().filter(|j| cpu_idx.contains(j)).count();
            for (a, b) in mlx_dist[i].iter().zip(cpu_dist) {
                assert_relative_eq!(*a, *b, epsilon = 1e-3, max_relative = 1e-3);
            }
            assert!(mlx_dist[i].windows(2).all(|w| w[0] <= w[1]));
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(recall > 0.99, "recall {recall}");
    }

    #[test]
    fn test_mlx_exhaustive_euclidean_matches_cpu() {
        check_against_cpu(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_mlx_exhaustive_cosine_matches_cpu() {
        check_against_cpu(Dist::Cosine);
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
