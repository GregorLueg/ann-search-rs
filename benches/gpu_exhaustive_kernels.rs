//! GPU kernel microbenchmarks using CubeCL's Benchmark trait. This bench is for
//! the Kernel used in the exhaustive search
//!
//! Run with: cargo bench --bench gpu_exhaustive_kernels --features gpu
//!
//! Note: no `#![allow(dead_code)]` here on purpose. It previously masked the
//! fact that `TopkCoalescedBench` was defined but never constructed, so the
//! coalesced-vs-serial top-k comparison the runner advertised never ran.

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::gpu::dist_gpu::*;
use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::*;
use ann_search_rs::utils::dist::Dist;

/// Shared config so all benchmarks use identical data
#[derive(Clone)]
struct BenchConfig {
    n_queries: usize,
    n_db: usize,
    dim: usize,
    k: usize,
    /// Distance metric. Cosine carries the larger shared-memory footprint that
    /// `pick_wg_y` was sized around, so it must be swept alongside Euclidean.
    metric: Dist,
}

impl BenchConfig {
    /// Euclidean config, the common case.
    fn euclidean(n_queries: usize, n_db: usize, dim: usize, k: usize) -> Self {
        Self {
            n_queries,
            n_db,
            dim,
            k,
            metric: Dist::SquaredEuclidean,
        }
    }

    /// Cosine config.
    fn cosine(n_queries: usize, n_db: usize, dim: usize, k: usize) -> Self {
        Self {
            n_queries,
            n_db,
            dim,
            k,
            metric: Dist::Cosine,
        }
    }

    /// Short metric tag for benchmark names.
    fn tag(&self) -> &'static str {
        match self.metric {
            Dist::Cosine => "cos",
            _ => "l2",
        }
    }
}

/// Deterministic synthetic query data, matching the original bench generator.
fn make_queries(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|i| ((i * 13 + 7) % 29) as f32 * 0.1).collect()
}

/// Deterministic synthetic database data, matching the original bench generator.
fn make_db(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|i| ((i * 17 + 3) % 31) as f32 * 0.1).collect()
}

/// Row-wise L2 norms, needed by the cosine kernel.
fn l2_norms(flat: &[f32], dim: usize) -> Vec<f32> {
    flat.chunks_exact(dim)
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

// ──────────────────────────────────────────────
// 1. Distance kernel only (euclidean_tiled)
// ──────────────────────────────────────────────

struct DistanceBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

struct DistanceInput<R: Runtime> {
    query_gpu: GpuTensor<R, f32>,
    db_gpu: GpuTensor<R, f32>,
    distances_gpu: GpuTensor<R, f32>,
    /// Query norms, populated for cosine only
    query_norms_gpu: Option<GpuTensor<R, f32>>,
    /// Database norms, populated for cosine only
    db_norms_gpu: Option<GpuTensor<R, f32>>,
}

impl<R: Runtime> Clone for DistanceInput<R> {
    fn clone(&self) -> Self {
        Self {
            query_gpu: self.query_gpu.clone(),
            db_gpu: self.db_gpu.clone(),
            distances_gpu: self.distances_gpu.clone(),
            query_norms_gpu: self.query_norms_gpu.clone(),
            db_norms_gpu: self.db_norms_gpu.clone(),
        }
    }
}

impl<R: Runtime> Benchmark for DistanceBench<R> {
    type Input = DistanceInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let dim = self.cfg.dim;
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;

        let queries = make_queries(nq, dim);
        let db = make_db(ndb, dim);

        let query_gpu = GpuTensor::<R, f32>::from_slice(&queries, vec![nq, dim], &self.client);
        let db_gpu = GpuTensor::<R, f32>::from_slice(&db, vec![ndb, dim], &self.client);
        let distances_gpu = GpuTensor::<R, f32>::empty(vec![nq, ndb], &self.client);

        let (query_norms_gpu, db_norms_gpu) = if self.cfg.metric == Dist::Cosine {
            let qn = l2_norms(&queries, dim);
            let dn = l2_norms(&db, dim);
            (
                Some(GpuTensor::<R, f32>::from_slice(&qn, vec![nq], &self.client)),
                Some(GpuTensor::<R, f32>::from_slice(&dn, vec![ndb], &self.client)),
            )
        } else {
            (None, None)
        };

        DistanceInput {
            query_gpu,
            db_gpu,
            distances_gpu,
            query_norms_gpu,
            db_norms_gpu,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let dim_lines = self.cfg.dim / LINE_SIZE;
        let vec_size = LINE_SIZE;

        // Use the production heuristic rather than a hardcoded 4, so the
        // shared-memory budget tiers actually get exercised by the sweep.
        let wg_y = pick_wg_y(self.cfg.dim).expect("dim outside pick_wg_y table");

        let grid_x = (ndb as u32).div_ceil(WORKGROUP_SIZE_X);
        let (grid_y, grid_z) = grid_2d((nq as u32).div_ceil(wg_y));

        match self.cfg.metric {
            Dist::Cosine => unsafe {
                cosine_tiled::launch_unchecked::<f32, R>(
                    &self.client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, wg_y),
                    vec_size,
                    input.query_gpu.into_tensor_arg(),
                    input.db_gpu.into_tensor_arg(),
                    input
                        .query_norms_gpu
                        .as_ref()
                        .expect("cosine needs query norms")
                        .into_tensor_arg(),
                    input
                        .db_norms_gpu
                        .as_ref()
                        .expect("cosine needs db norms")
                        .into_tensor_arg(),
                    input.distances_gpu.into_tensor_arg(),
                    0u32,
                    ndb as u32,
                    nq as u32,
                    ndb as u32,
                    dim_lines,
                    wg_y,
                );
            },
            _ => unsafe {
                euclidean_tiled::launch_unchecked::<f32, R>(
                    &self.client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, wg_y),
                    vec_size,
                    input.query_gpu.into_tensor_arg(),
                    input.db_gpu.into_tensor_arg(),
                    input.distances_gpu.into_tensor_arg(),
                    0u32,
                    ndb as u32,
                    nq as u32,
                    ndb as u32,
                    dim_lines,
                    wg_y,
                );
            },
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "distance_only_{}q_{}db_{}d_{}",
            self.cfg.n_queries,
            self.cfg.n_db,
            self.cfg.dim,
            self.cfg.tag()
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// 1b. Register-tiled distance kernel
// ──────────────────────────────────────────────

struct DistanceRegBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for DistanceRegBench<R> {
    type Input = DistanceInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        DistanceBench {
            cfg: self.cfg.clone(),
            client: self.client.clone(),
        }
        .prepare()
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let dim_lines = self.cfg.dim / LINE_SIZE;
        let vec_size = LINE_SIZE;

        let wg_y = pick_wg_y(self.cfg.dim).expect("dim outside pick_wg_y table");
        if !tile_fits(wg_y) {
            return Err(format!("wg_y {wg_y} not divisible by TILE_Q {TILE_Q}"));
        }
        let threads_y = wg_y / TILE_Q as u32;

        let grid_x = (ndb as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);
        let (grid_y, grid_z) = grid_2d((nq as u32).div_ceil(wg_y));

        match self.cfg.metric {
            Dist::Cosine => unsafe {
                cosine_tiled_reg::launch_unchecked::<f32, R>(
                    &self.client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, threads_y),
                    vec_size,
                    input.query_gpu.into_tensor_arg(),
                    input.db_gpu.into_tensor_arg(),
                    input
                        .query_norms_gpu
                        .as_ref()
                        .expect("cosine needs query norms")
                        .into_tensor_arg(),
                    input
                        .db_norms_gpu
                        .as_ref()
                        .expect("cosine needs db norms")
                        .into_tensor_arg(),
                    input.distances_gpu.into_tensor_arg(),
                    0u32,
                    ndb as u32,
                    nq as u32,
                    ndb as u32,
                    dim_lines,
                    wg_y,
                    TILE_D,
                    TILE_Q,
                );
            },
            _ => unsafe {
                euclidean_tiled_reg::launch_unchecked::<f32, R>(
                    &self.client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, threads_y),
                    vec_size,
                    input.query_gpu.into_tensor_arg(),
                    input.db_gpu.into_tensor_arg(),
                    input.distances_gpu.into_tensor_arg(),
                    0u32,
                    ndb as u32,
                    nq as u32,
                    ndb as u32,
                    dim_lines,
                    wg_y,
                    TILE_D,
                    TILE_Q,
                );
            },
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "distance_reg{}x{}_{}q_{}db_{}d_{}",
            TILE_D,
            TILE_Q,
            self.cfg.n_queries,
            self.cfg.n_db,
            self.cfg.dim,
            self.cfg.tag()
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// 2. extract_topk only (pre-filled distance matrix)
// ──────────────────────────────────────────────

struct TopkBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

struct TopkInput<R: Runtime> {
    distances_gpu: GpuTensor<R, f32>,
    topk_dists: GpuTensor<R, f32>,
    topk_indices: GpuTensor<R, u32>,
}

impl<R: Runtime> Clone for TopkInput<R> {
    fn clone(&self) -> Self {
        Self {
            distances_gpu: self.distances_gpu.clone(),
            topk_dists: self.topk_dists.clone(),
            topk_indices: self.topk_indices.clone(),
        }
    }
}

impl<R: Runtime> Benchmark for TopkBench<R> {
    type Input = TopkInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let k = self.cfg.k;

        let dists: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 7 + 13) % 1000) as f32 * 0.01)
            .collect();

        let distances_gpu = GpuTensor::<R, f32>::from_slice(&dists, vec![nq, ndb], &self.client);
        let topk_dists = GpuTensor::<R, f32>::empty(vec![nq, k], &self.client);
        let topk_indices = GpuTensor::<R, u32>::empty(vec![nq, k], &self.client);

        let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let init_gy = (nq as u32).div_ceil(4);
        unsafe {
            init_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(init_gx, init_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                topk_dists.clone().into_tensor_arg(),
                topk_indices.clone().into_tensor_arg(),
                4,
            );
        }
        future::block_on(self.client.sync()).expect("sync failed");

        TopkInput {
            distances_gpu,
            topk_dists,
            topk_indices,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;

        let extract_grid = (nq as u32).div_ceil(WORKGROUP_SIZE_X);
        unsafe {
            extract_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(extract_grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.distances_gpu.into_tensor_arg(),
                input.topk_dists.into_tensor_arg(),
                input.topk_indices.into_tensor_arg(),
                0u32,
                ndb as u32,
                self.cfg.k as u32,
                self.cfg.k,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "topk_only_{}q_{}db_k{}",
            self.cfg.n_queries, self.cfg.n_db, self.cfg.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// 3. Full pipeline (distance + topk, chunked)
// ──────────────────────────────────────────────

struct FullPipelineBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
    device: R::Device,
}

#[derive(Clone)]
struct PipelineInput {
    queries: Vec<f32>,
    db: Vec<f32>,
    /// Empty unless the config is cosine
    query_norms: Vec<f32>,
    /// Empty unless the config is cosine
    db_norms: Vec<f32>,
}

impl<R: Runtime> Benchmark for FullPipelineBench<R> {
    type Input = PipelineInput;
    type Output = (Vec<Vec<usize>>, Vec<Vec<f32>>);

    fn prepare(&self) -> Self::Input {
        let dim = self.cfg.dim;

        let queries = make_queries(self.cfg.n_queries, dim);
        let db = make_db(self.cfg.n_db, dim);

        let (query_norms, db_norms) = if self.cfg.metric == Dist::Cosine {
            (l2_norms(&queries, dim), l2_norms(&db, dim))
        } else {
            (Vec::new(), Vec::new())
        };

        PipelineInput {
            queries,
            db,
            query_norms,
            db_norms,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let qb = BatchData::new(&input.queries, &input.query_norms, self.cfg.n_queries);
        let dbb = BatchData::new(&input.db, &input.db_norms, self.cfg.n_db);

        let result = query_batch_gpu::<f32, R>(
            self.cfg.k,
            &qb,
            &dbb,
            self.cfg.dim,
            &self.cfg.metric,
            self.device.clone(),
            false,
        )
        .map_err(|e| e.to_string())?;

        Ok(result)
    }

    fn name(&self) -> String {
        format!(
            "full_pipeline_{}q_{}db_{}d_k{}_{}",
            self.cfg.n_queries,
            self.cfg.n_db,
            self.cfg.dim,
            self.cfg.k,
            self.cfg.tag()
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// 4. TopK Coalesced kernel
// ──────────────────────────────────────────────

struct TopkCoalescedBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

impl<R: Runtime> Clone for TopkCoalescedBench<R> {
    fn clone(&self) -> Self {
        Self {
            cfg: self.cfg.clone(),
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> Benchmark for TopkCoalescedBench<R> {
    type Input = TopkInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let k = self.cfg.k;

        let dists: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 7 + 13) % 1000) as f32 * 0.01)
            .collect();

        let distances_gpu = GpuTensor::<R, f32>::from_slice(&dists, vec![nq, ndb], &self.client);
        let topk_dists = GpuTensor::<R, f32>::empty(vec![nq, k], &self.client);
        let topk_indices = GpuTensor::<R, u32>::empty(vec![nq, k], &self.client);

        let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let init_gy = (nq as u32).div_ceil(4);
        unsafe {
            init_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(init_gx, init_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                topk_dists.clone().into_tensor_arg(),
                topk_indices.clone().into_tensor_arg(),
                4,
            );
        }
        future::block_on(self.client.sync()).expect("sync failed");

        TopkInput {
            distances_gpu,
            topk_dists,
            topk_indices,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let k = self.cfg.k;

        let grid = nq as u32; // one workgroup per query
        let merge = plan_topk_merge(
            k,
            size_of::<f32>(),
            self.client.properties().hardware.max_shared_memory_size,
        )
        .expect("k too large for the device shared-memory budget");
        unsafe {
            extract_topk_coalesced::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new_2d(32, 1),
                input.distances_gpu.into_tensor_arg(),
                input.topk_dists.into_tensor_arg(),
                input.topk_indices.into_tensor_arg(),
                0u32,
                ndb as u32,
                ndb as u32,
                k as u32,
                k,
                merge.group,
                merge.single_round,
                merge.slots,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "topk_coalesced_{}q_{}db_k{}",
            self.cfg.n_queries, self.cfg.n_db, self.cfg.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// Runner
// ──────────────────────────────────────────────

/// Print the device ceilings that silently gate `launch_unchecked`.
///
/// The per-binding limit is the one that bites: a wave can fit the total VRAM
/// budget while a single tensor busts `max_page_size`.
fn report_device_limits<R: Runtime>(client: &ComputeClient<R>) {
    let props = client.properties();
    let hw = &props.hardware;
    println!("====== Device ======");
    println!("runtime          : {}", R::name(client));
    println!(
        "plane size       : {} .. {}{}",
        hw.plane_size_min,
        hw.plane_size_max,
        if hw.plane_size_min == WORKGROUP_SIZE_X && hw.plane_size_max == WORKGROUP_SIZE_X {
            "  (== WORKGROUP_SIZE_X, plane primitives viable)"
        } else {
            "  (!= WORKGROUP_SIZE_X, plane primitives NOT viable)"
        }
    );
    println!("max shared mem   : {} bytes", hw.max_shared_memory_size);
    println!("max binding size : {} bytes", props.memory.max_page_size);
    println!("max cube count   : {:?}", hw.max_cube_count);
    println!("max cube dim     : {:?}", hw.max_cube_dim);
    println!("max units/cube   : {}", hw.max_units_per_cube);
    println!();
}

/// Validate the top-k kernels once, before any timing runs.
///
/// Every launch in this crate is `launch_unchecked`. A dispatch that busts a
/// device limit does no work, returns zeros, reports no error, and looks like
/// an enormous speedup. So we check the output is real before believing any
/// number, and we cross-check the two top-k kernels against each other, which
/// is the comparison the runner previously claimed to make but never did.
///
/// ### Params
///
/// * `cfg` - Config to validate against
/// * `client` - GPU compute client
///
/// ### Returns
///
/// Panics with a diagnostic if either kernel produced implausible output or
/// the two disagree.
fn validate_topk<R: Runtime>(cfg: &BenchConfig, client: &ComputeClient<R>) {
    let serial = TopkBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };
    let coalesced = TopkCoalescedBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };

    let s_in = serial.prepare();
    serial.execute(s_in.clone()).expect("serial topk failed");
    serial.sync();
    let s_dists = s_in.topk_dists.read(client).expect("read failed");

    let c_in = coalesced.prepare();
    coalesced.execute(c_in.clone()).expect("coalesced topk failed");
    coalesced.sync();
    let c_dists = c_in.topk_dists.read(client).expect("read failed");

    let k = cfg.k;

    // A kernel that did nothing leaves the init_topk sentinel in place.
    let untouched = s_dists.iter().filter(|d| **d >= f32::MAX).count();
    assert_eq!(
        untouched, 0,
        "serial topk left {untouched} sentinel entries: the kernel almost certainly did no work"
    );
    let untouched_c = c_dists.iter().filter(|d| **d >= f32::MAX).count();
    assert_eq!(
        untouched_c, 0,
        "coalesced topk left {untouched_c} sentinel entries: the kernel almost certainly did no work"
    );

    // Each row must be sorted ascending.
    for (row, chunk) in s_dists.chunks_exact(k).enumerate() {
        assert!(
            chunk.windows(2).all(|w| w[0] <= w[1]),
            "serial topk row {row} is not sorted: {chunk:?}"
        );
    }

    // The two kernels must agree on the selected distances. Indices may differ
    // on ties, distances must not.
    for (row, (s, c)) in s_dists.chunks_exact(k).zip(c_dists.chunks_exact(k)).enumerate() {
        for (j, (sd, cd)) in s.iter().zip(c.iter()).enumerate() {
            assert!(
                (sd - cd).abs() <= 1e-6 * sd.abs().max(1.0),
                "topk kernels disagree at row {row} slot {j}: serial={sd} coalesced={cd}"
            );
        }
    }

    println!("topk validation passed ({} queries, k={})", cfg.n_queries, k);
}

/// Cross-check the register-tiled distance kernel against the 1x1 kernel.
///
/// Both write the same `[n_queries, n_db]` matrix, so every entry must agree.
/// A tiling bug shows up as a block-structured mismatch, and an out-of-range
/// dispatch shows up as a block of exact zeros.
///
/// ### Params
///
/// * `cfg` - Config to validate against
/// * `client` - GPU compute client
fn validate_distance_reg<R: Runtime>(cfg: &BenchConfig, client: &ComputeClient<R>) {
    let base = DistanceBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };
    let tiled = DistanceRegBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };

    let b_in = base.prepare();
    base.execute(b_in.clone()).expect("base distance failed");
    base.sync();
    let b = b_in.distances_gpu.read(client).expect("read failed");

    let t_in = tiled.prepare();
    match tiled.execute(t_in.clone()) {
        Ok(()) => {}
        Err(e) => {
            println!("distance_reg skipped for dim={}: {e}", cfg.dim);
            return;
        }
    }
    tiled.sync();
    let t = t_in.distances_gpu.read(client).expect("read failed");

    assert_eq!(b.len(), t.len(), "distance matrices differ in length");

    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    let mut mismatches = 0usize;
    for (i, (bv, tv)) in b.iter().zip(t.iter()).enumerate() {
        let tol = 1e-3 * bv.abs().max(1.0);
        let d = (bv - tv).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
        if d > tol {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches,
        0,
        "distance_reg disagrees with the 1x1 kernel in {mismatches}/{} entries; \
         worst delta {worst} at flat index {worst_at} (row {}, col {})",
        b.len(),
        worst_at / cfg.n_db,
        worst_at % cfg.n_db
    );

    // A kernel that did no work returns zeros everywhere.
    let nonzero = t.iter().filter(|v| **v != 0.0).count();
    assert!(
        nonzero > t.len() / 2,
        "distance_reg produced {nonzero}/{} non-zero entries: the dispatch almost \
         certainly did no work",
        t.len()
    );

    println!(
        "distance_reg validation passed (dim={}, worst delta {worst:.3e})",
        cfg.dim
    );
}

fn run_exhaustive_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    report_device_limits::<R>(&client);

    // ── Kernel-level benchmarks (single chunk sizes) ──
    //
    // dim is swept across the `pick_wg_y` tiers (32/64 -> wg_y 32, 128 -> 32,
    // 512 -> 8) so the shared-memory budget table added in 5ab27ef is actually
    // exercised, and cosine is included because it carries the larger footprint
    // that table was sized around.
    let kernel_configs = vec![
        BenchConfig::euclidean(8192, 16_384, 32, 15),
        BenchConfig::euclidean(8192, 16_384, 64, 15),
        BenchConfig::euclidean(8192, 16_384, 128, 15),
        BenchConfig::euclidean(8192, 16_384, 512, 15),
        BenchConfig::cosine(8192, 16_384, 128, 15),
    ];

    println!("====== Correctness gate ======");
    validate_topk::<R>(&kernel_configs[0], &client);
    for cfg in &kernel_configs {
        validate_distance_reg::<R>(cfg, &client);
    }
    println!();

    println!("====== Kernel-level (single chunk) ======");
    for cfg in kernel_configs {
        println!(
            "\n--- {}q x {}db, dim={}, k={}, {} ---\n",
            cfg.n_queries,
            cfg.n_db,
            cfg.dim,
            cfg.k,
            cfg.tag()
        );

        let dist_bench = DistanceBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };
        let dist_reg_bench = DistanceRegBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };
        let topk_bench = TopkBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };
        let topk_coalesced_bench = TopkCoalescedBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };

        println!("{}", dist_bench.name());
        println!("{:?}", dist_bench.run(TimingMethod::System));

        println!("{}", dist_reg_bench.name());
        println!("{:?}", dist_reg_bench.run(TimingMethod::System));

        println!("{}", topk_bench.name());
        println!("{:?}", topk_bench.run(TimingMethod::System));

        println!("{}", topk_coalesced_bench.name());
        println!("{:?}", topk_coalesced_bench.run(TimingMethod::System));
    }

    // ── Pipeline-level benchmarks (realistic workloads) ──

    let pipeline_configs = vec![
        // Self-query patterns (kNN graph generation)
        BenchConfig::euclidean(50_000, 50_000, 32, 15),
        // Cross-batch pattern
        BenchConfig::euclidean(25_000, 50_000, 32, 15),
        // Higher dim
        BenchConfig::euclidean(50_000, 50_000, 64, 15),
        BenchConfig::euclidean(50_000, 50_000, 128, 15),
        BenchConfig::cosine(50_000, 50_000, 128, 15),
    ];

    println!("\n====== Full pipeline ======");
    for cfg in pipeline_configs {
        println!(
            "\n--- {}q x {}db, dim={}, k={}, {} ---\n",
            cfg.n_queries,
            cfg.n_db,
            cfg.dim,
            cfg.k,
            cfg.tag()
        );

        let full_bench = FullPipelineBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
            device: device.clone(),
        };

        println!("{}", full_bench.name());
        println!("{:?}", full_bench.run(TimingMethod::System));
    }
}

fn main() {
    run_exhaustive_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
