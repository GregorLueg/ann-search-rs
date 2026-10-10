//! Queryable NN-Descent + CAGRA index on MLX, the counterpart of
//! `NNDescentGpu`: NN-Descent builds the kNN graph, the CAGRA optimisation
//! turns it into a navigational graph, and [`CagraSearchMlx`] beam-searches
//! it. Query entry points come from the build's forest router, self-query
//! entries from the kNN graph, both exactly as on the wgpu path.

use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::mlx::cagra_mlx::*;
use crate::mlx::nndescent_mlx::*;
use crate::prelude::*;
use crate::utils::dist::cosine_from_dot;
use crate::utils::nndescent_utils::unpack_knn_graph;
use crate::utils::rp_forest::{compact_knn_rows, default_forest_trees, ForestRouter};
use crate::utils::DimensionValidation;

/// Default final degree, as on the wgpu path.
const DEFAULT_K: usize = 30;

/// Default maximum number of NN-Descent iterations
const DEFAULT_MAX_ITERS: usize = 15;

/// Default convergence threshold (fraction of `n * build_k` edges updated)
const DEFAULT_DELTA: f32 = 0.001;

/// Default sampling rate for the local join
const DEFAULT_RHO: f32 = 1.0;

/// Router candidates gathered per entry point before scoring.
const ROUTER_OVERSAMPLE: usize = 4;

////////////////////////
// NNDescentIndexMlx //
////////////////////////

/// NN-Descent kNN graph with a CAGRA navigational graph, searched on MLX.
/// f32 only.
///
/// Holds MLX streams, which are thread affine: build and query on the same
/// thread. The raw handles keep this type `!Send` and `!Sync`.
pub struct NNDescentIndexMlx {
    /// Original (unpadded) vector data, flattened row-major
    pub vectors_flat: Vec<f32>,
    /// Original embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node (final CAGRA degree)
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<f32>,
    /// Distance metric
    metric: Dist,
    /// The medoid of the graph as entry point
    pub medoid: u32,
    /// True kNN graph of size `n * k`, sorted by distance per row, before
    /// CAGRA pruning
    knn_graph: Vec<(usize, f32)>,
    /// Whether NN-Descent hit the delta threshold
    converged: bool,
    /// Forest router for query entry points
    router: ForestRouter<f32>,
    /// Beam search over the navigational graph, device resident
    searcher: CagraSearchMlx,
}

impl DimensionValidation for NNDescentIndexMlx {
    fn dim(&self) -> usize {
        self.dim
    }
}

impl NNDescentIndexMlx {
    /// Build the kNN graph via NN-Descent, then the CAGRA navigational graph.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `metric` - Distance metric; Manhattan is rejected
    /// * `k` - Final neighbours per node (default 30)
    /// * `build_k` - Internal NN-Descent degree before CAGRA pruning.
    ///   Defaults to `1.5 * k`, at least `k`
    /// * `max_iters` - Maximum NN-Descent iterations (default 15)
    /// * `n_trees` - Forest size for the init. Defaults to `5 + n^0.25`,
    ///   capped at 20
    /// * `delta` - Convergence threshold as fraction of `n * build_k`
    ///   (default 0.001)
    /// * `rho` - Sampling rate for the local join (default 1.0, no sampling)
    /// * `refine_knn` - Two-hop sweeps after the main loop (default 0)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// The index, vectors and navigational graph resident on the device
    #[allow(clippy::too_many_arguments)]
    pub fn build(
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
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        let (vectors_flat, n, dim) = data.into_row_major();
        let k = k.unwrap_or(DEFAULT_K);
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
                "NNDescent-MLX: {} vectors, dim={}, k={}, build_k={}",
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
            n_trees: n_trees.unwrap_or_else(|| default_forest_trees(n)),
            delta: delta.unwrap_or(DEFAULT_DELTA),
            rho_thresh: (rho.unwrap_or(DEFAULT_RHO) * 65535.0) as u32,
            refine_knn: refine_knn.unwrap_or(0),
            seed,
            use_cosine,
        };
        let out = nndescent_core_mlx(&vectors_flat, &norms, n, dim, &cfg, verbose)?;
        let knn_graph = compact_knn_rows(&out.graph_idx, &out.graph_dist, n, k, build_k);

        let cagra_start = Instant::now();
        let (nav_graph, medoid) =
            cagra_optimise_mlx(&out.graph_idx, &vectors_flat, n, dim, build_k, k, metric)?;
        if verbose {
            println!("  CAGRA optimisation: {:.2?}", cagra_start.elapsed());
        }

        let searcher = CagraSearchMlx::new(&vectors_flat, n, dim, metric, nav_graph, k, medoid)?;
        if verbose {
            println!("  Total build time: {:.2?}", start.elapsed());
        }

        Ok(Self {
            vectors_flat,
            dim,
            n,
            k,
            norms,
            metric,
            medoid,
            knn_graph,
            converged: out.converged,
            router: out.router,
            searcher,
        })
    }

    /// Batch query via beam search on the navigational graph.
    ///
    /// Entry points per query: the medoid, then the closest of the router's
    /// leaf candidates, padded with node 0, as `NNDescentGpu` does.
    ///
    /// ### Params
    ///
    /// * `queries_flat` - Flattened query vectors, row-major `[n_queries, dim]`
    /// * `n_queries` - Number of query vectors
    /// * `query_params` - Optional beam parameters; `None` scales them to `k`
    /// * `k` - Number of neighbours to return per query
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per query, sorted by distance ascending
    pub fn query_batch(
        &self,
        queries_flat: &[f32],
        n_queries: usize,
        query_params: Option<CagraMlxSearchParams>,
        k: usize,
        seed: usize,
    ) -> KnnResult<f32> {
        if n_queries == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.check_dim(queries_flat.len() / n_queries)?;
        let query_params = query_params.unwrap_or_else(|| CagraMlxSearchParams::from_k(k));
        let n_entry = query_params.get_n_entry();
        // Plain host fields only: the MLX handles in `self` are not `Sync`.
        let (medoid, dim, metric) = (self.medoid, self.dim, self.metric);
        let (router, vectors, norms) = (&self.router, &self.vectors_flat, &self.norms);

        let entries: Vec<u32> = (0..n_queries)
            .into_par_iter()
            .flat_map_iter(|i| {
                let query = &queries_flat[i * dim..(i + 1) * dim];
                let q_norm = if metric == Dist::Cosine {
                    f32::calculate_l2_norm(query)
                } else {
                    1.0
                };
                // Score each candidate once, then select.
                let mut scored: Vec<(f32, usize)> = router
                    .find_entry_points(query, n_entry * ROUTER_OVERSAMPLE)
                    .into_iter()
                    .filter(|&c| c != medoid as usize)
                    .map(|c| {
                        let row = &vectors[c * dim..(c + 1) * dim];
                        let d = match metric {
                            Dist::Cosine => {
                                cosine_from_dot(f32::dot_simd(query, row), q_norm * norms[c])
                            }
                            _ => f32::euclidean_simd(query, row),
                        };
                        (d, c)
                    })
                    .collect();
                let by_dist = |a: &(f32, usize), b: &(f32, usize)| a.0.total_cmp(&b.0);
                let take = (n_entry - 1).min(scored.len());
                if take < scored.len() {
                    scored.select_nth_unstable_by(take, by_dist);
                    scored.truncate(take);
                }
                scored.sort_unstable_by(by_dist);

                let mut e = Vec::with_capacity(n_entry);
                e.push(medoid);
                e.extend(scored.into_iter().map(|(_, c)| c as u32));
                e.resize(n_entry, 0);
                e.into_iter()
            })
            .collect();

        self.searcher.search(
            queries_flat,
            n_queries,
            k,
            Some(query_params),
            Some(&entries),
            seed,
        )
    }

    /// Self-query: beam search for every indexed vector. Entries are the node
    /// itself plus strided picks from its kNN row, as `NNDescentGpu` does.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector, self included
    /// * `query_params` - Optional beam parameters; `None` scales them to `k`
    /// * `seed` - Random seed for the entry top-up
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per vector, sorted by distance ascending
    pub fn self_query(
        &self,
        k: usize,
        query_params: Option<CagraMlxSearchParams>,
        seed: usize,
    ) -> KnnResult<f32> {
        let query_params = query_params.unwrap_or_else(|| CagraMlxSearchParams::from_k(k));
        let n_entry = query_params.get_n_entry();
        // `SENTINEL_PID` narrows to the `0x7FFFFFFF` the helper skips.
        let rows: Vec<u32> = self.knn_graph.iter().map(|&(p, _)| p as u32).collect();
        let entries = self_entry_points(&rows, self.k, self.n, n_entry, seed);
        self.searcher
            .self_search(k, Some(query_params), Some(&entries), seed)
    }

    /// Distance metric this index was built with.
    ///
    /// ### Returns
    ///
    /// The metric
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Whether NN-Descent reached the convergence threshold.
    ///
    /// ### Returns
    ///
    /// `true` if the update rate fell below `delta` before `max_iters`
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Borrow the flat kNN graph, `n * k` `(pid, distance)` pairs, before
    /// CAGRA pruning.
    ///
    /// ### Returns
    ///
    /// The raw NN-Descent graph
    pub fn knn_graph(&self) -> &[(usize, f32)] {
        &self.knn_graph
    }

    /// Extract the kNN graph as index/distance vectors.
    ///
    /// ### Params
    ///
    /// * `k` - Truncate each row to this total length, self-edge included
    ///   when `include_self` is set. `None` keeps the build-time `k`.
    /// * `include_self` - Prepend `(i, 0)` to row `i`
    /// * `return_dist` - Whether to include distances
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)`, ascending; sentinels dropped
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

    /// Size of the index in bytes: host copies plus the searcher's host and
    /// device copies. The router is not counted.
    ///
    /// ### Returns
    ///
    /// Number of bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * size_of::<f32>()
            + self.norms.capacity() * size_of::<f32>()
            + self.knn_graph.capacity() * size_of::<(usize, f32)>()
            + self.searcher.memory_usage_bytes()
    }
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

    /// Points around 20 random centres.
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
            .map(|_| (0..dim).map(|_| rng.random_range(-3.0..3.0)).collect())
            .collect();
        let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..20)).collect();
        Mat::from_fn(n, dim, |i, j| {
            centres[labels[i]][j] + rng.random_range(-1.0f32..1.0)
        })
    }

    /// Mean recall of `approx` against exhaustive `truth`.
    ///
    /// ### Params
    ///
    /// * `truth` - Ground truth rows
    /// * `approx` - Rows under test
    ///
    /// ### Returns
    ///
    /// Recall in `[0, 1]`
    fn recall(truth: &[Vec<usize>], approx: &[Vec<usize>]) -> f64 {
        let k = truth[0].len();
        let hits: usize = truth
            .iter()
            .zip(approx)
            .map(|(t, a)| a.iter().filter(|j| t.contains(j)).count())
            .sum();
        hits as f64 / (truth.len() * k) as f64
    }

    /// Build an MLX index and measure external and self recall.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric
    ///
    /// ### Returns
    ///
    /// `(data, queries, external recall, self recall)`
    fn mlx_recalls(metric: Dist) -> (Mat<f32>, Mat<f32>, f64, f64) {
        let (n, dim, k) = (3_000, 24, 10);
        let data = clustered(n, dim, 3);
        let queries = clustered(200, dim, 3);
        let index = NNDescentIndexMlx::build(
            data.as_ref(),
            metric,
            Some(16),
            None,
            None,
            None,
            None,
            None,
            None,
            42,
            false,
        )
        .unwrap();
        let cpu = ExhaustiveIndex::new(data.as_ref(), metric);
        let (q_flat, nq, _) = queries.as_ref().into_row_major();

        let truth: Vec<Vec<usize>> = cpu
            .query_batch(&q_flat, nq, k, None, false)
            .unwrap()
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let (idx, dist) = index.query_batch(&q_flat, nq, None, k, 42).unwrap();
        for d in &dist {
            assert!(d.windows(2).all(|w| w[0] <= w[1]));
        }
        let r_ext = recall(&truth, &idx);

        let (self_truth, _) = cpu.generate_knn(k, false, false).unwrap();
        let (self_idx, _) = index.self_query(k, None, 42).unwrap();
        let r_self = recall(&self_truth, &self_idx);
        println!("MLX {metric:?}: query recall {r_ext:.4}, self recall {r_self:.4}");
        (data, queries, r_ext, r_self)
    }

    #[test]
    fn test_mlx_nndescent_index_euclidean() {
        let (_, _, r_ext, r_self) = mlx_recalls(Dist::SquaredEuclidean);
        assert!(r_ext > 0.95, "query recall {r_ext}");
        assert!(r_self > 0.95, "self recall {r_self}");
    }

    #[test]
    fn test_mlx_nndescent_index_cosine() {
        let (_, _, r_ext, r_self) = mlx_recalls(Dist::Cosine);
        assert!(r_ext > 0.95, "query recall {r_ext}");
        assert!(r_self > 0.95, "self recall {r_self}");
    }

    #[test]
    fn test_mlx_nndescent_index_rejects_bad_input() {
        let data = clustered(100, 8, 1);
        let build = |m| {
            NNDescentIndexMlx::build(
                data.as_ref(),
                m,
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                42,
                false,
            )
        };
        assert!(build(Dist::Manhattan).is_err());
        let index = build(Dist::SquaredEuclidean).unwrap();
        assert!(index.query_batch(&[0.0; 9], 1, None, 3, 42).is_err());
        let (idx, _) = index.extract_knn(None, false, false);
        assert!(idx.iter().all(|r| r.len() == 5));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_mlx_nndescent_index_matches_wgpu() {
        use crate::gpu::nndescent_gpu::NNDescentGpu;
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (data, queries, r_ext, r_self) = mlx_recalls(Dist::SquaredEuclidean);
        let k = 10;
        let mut gpu = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(16),
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
        let cpu = ExhaustiveIndex::new(data.as_ref(), Dist::SquaredEuclidean);
        let (q_flat, nq, _) = queries.as_ref().into_row_major();
        let truth: Vec<Vec<usize>> = cpu
            .query_batch(&q_flat, nq, k, None, false)
            .unwrap()
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        let (g_idx, _) = gpu.query_batch_gpu(&q_flat, nq, None, k, 42).unwrap();
        let (self_truth, _) = cpu.generate_knn(k, false, false).unwrap();
        let (g_self, _) = gpu.self_query_gpu(k, None, 42).unwrap();
        let (g_ext, g_slf) = (recall(&truth, &g_idx), recall(&self_truth, &g_self));
        println!("wgpu: query recall {g_ext:.4}, self recall {g_slf:.4}");
        assert!(r_ext >= g_ext - 0.01, "mlx {r_ext} vs wgpu {g_ext}");
        assert!(r_self >= g_slf - 0.01, "mlx {r_self} vs wgpu {g_slf}");
    }
}
