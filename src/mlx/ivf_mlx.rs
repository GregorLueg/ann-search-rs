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
//! 4. [`IVF_SCAN_SOURCE`] scans the probed clusters, one SIMD group per
//!    query, and keeps the top k in registers.
//!
//! This is query-major where the wgpu index is cluster-major. The wgpu layout
//! needs a host-built task list (a readback of the probe ids mid-query) and a
//! `[n_queries, max_candidates]` candidate buffer whose first touch was most
//! of its query time; the scan here needs neither. Distances are computed
//! directly, not through the GEMM expansion, so there is no cancellation.

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

/// Query rows (SIMD groups) per threadgroup in the scan kernel.
const SCAN_ROWS_PER_GROUP: i32 = 8;

/// Metal body of the cluster scan. One SIMD group per query: the probed
/// clusters are walked in ascending centroid distance, each lane takes every
/// 32nd member, computes the distance in full (`float4` loads when `VEC4`)
/// and keeps a sorted top-K in registers; K rounds of `simd_min` merge the 32
/// lists. Probing stops after `params[0]` (nprobe) clusters once K points are
/// reachable, else carries on down the pool of P, as the wgpu index does.
/// Slots never filled stay at `INFINITY`.
///
/// Template args: `K`, pool width `P`, `DIM`, `VEC4` (`DIM % 4 == 0`) and
/// `COSINE` (`1 - q.x` on unit vectors, else squared Euclidean).
const IVF_SCAN_SOURCE: &str = r#"
    uint lane = thread_position_in_grid.x;
    uint qi = order[thread_position_in_grid.y];
    const device float* qv = q + (ulong)qi * DIM;
    const device uint* prow = probe + (ulong)qi * P;
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
    for (int r = 0; r < K; r++) {
        float m = simd_min(vals[0]);
        uint w = simd_min(vals[0] == m ? lane : 32u);
        if (lane == w) {
            out_dist[(ulong)qi * K + r] = vals[0];
            out_idx[(ulong)qi * K + r] = ids[0];
            for (int i = 0; i < K - 1; i++) { vals[i] = vals[i + 1]; ids[i] = ids[i + 1]; }
            vals[K - 1] = INFINITY;
        }
    }
"#;

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
    /// Cluster scan kernel, see [`IVF_SCAN_SOURCE`]
    scan_kernel: MetalKernel,
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
                IVF_SCAN_SOURCE,
            ),
            stream,
        })
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
        let tile = nquery
            .unwrap_or(IVF_MLX_QUERY_BATCH_SIZE)
            .min(score_tile_rows(self.nlist))
            .max(1);
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
