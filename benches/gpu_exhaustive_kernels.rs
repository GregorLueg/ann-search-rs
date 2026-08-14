//! GPU kernel microbenchmarks using CubeCL's Benchmark trait. This bench is for
//! the Kernel used in the exhaustive search
//!
//! Run with: cargo bench --bench gpu_exhaustive_kernels --features gpu
//!
//! Note: no `#![allow(dead_code)]` here on purpose. It previously masked a
//! bench struct that was defined but never constructed, so a comparison the
//! runner advertised never actually ran. Let the compiler complain instead.

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::gpu::dist_gpu::*;
use ann_search_rs::gpu::topk_gpu::{
    radix_select_smem_bytes, radix_select_topk, radix_select_usable, SURV_ARRAYS_DENSE,
};
use ann_search_rs::gpu::*;
use ann_search_rs::utils::dist::Dist;
use cubecl_utils_rs::prelude::*;

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
    (0..n * dim)
        .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
        .collect()
}

/// Deterministic synthetic database data, matching the original bench generator.
fn make_db(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
        .collect()
}

/// Row-wise L2 norms, needed by the cosine kernel.
fn l2_norms(flat: &[f32], dim: usize) -> Vec<f32> {
    flat.chunks_exact(dim)
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// Seed the top-k output buffers with the `f32::MAX` / `0` sentinel.
///
/// Called from the timed `execute`, not from `prepare`, and that placement is
/// deliberate. `Benchmark::run` calls `prepare` once and clones the input for
/// every iteration, so seeding there leaves iterations 2..n reducing into an
/// already-populated top-k, where the `dist < local[k - 1]` acceptance test
/// rejects nearly every candidate. Those iterations do far less work than the
/// first, and the gap widens with `k`, which flattens exactly the curve a `k`
/// sweep is trying to measure.
///
/// The cost is `O(n_queries * k)` writes against the reducer's
/// `O(n_queries * n_db)` reads, and it is paid identically by every arm.
///
/// ### Params
///
/// * `client` - Compute client
/// * `dists` - Top-k distance buffer `[n_queries, k]`
/// * `indices` - Top-k index buffer `[n_queries, k]`
/// * `nq` - Number of queries
/// * `k` - Number of neighbours
fn seed_topk<R: Runtime>(
    client: &ComputeClient<R>,
    dists: &GpuTensor<R, f32>,
    indices: &GpuTensor<R, u32>,
    nq: usize,
    k: usize,
) {
    let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
    let init_gy = (nq as u32).div_ceil(4);
    unsafe {
        init_topk::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(init_gx, init_gy, 1),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
            dists.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            4,
        );
    }
}

/// Allocate the shared top-k bench input: a synthetic distance matrix plus the
/// two output buffers.
///
/// The distance values are a deterministic sawtooth in `[0, 10)`. It is not a
/// real distance matrix, it only has to be reducible.
///
/// ### Params
///
/// * `client` - Compute client
/// * `nq` - Number of queries
/// * `ndb` - Number of database candidates per query
/// * `k` - Number of neighbours
///
/// ### Returns
///
/// The populated [`TopkInput`].
fn make_topk_input<R: Runtime>(
    client: &ComputeClient<R>,
    nq: usize,
    ndb: usize,
    k: usize,
) -> TopkInput<R> {
    // High-entropy values on purpose. The original generator cycled through
    // only 1000 distinct distances, so at large `n_db` every value had dozens of
    // exact duplicates. That is not what a real distance matrix looks like, and
    // it biases the comparison in both directions at once: ties are rejected by
    // the serial kernel's strict `<` acceptance test, so they cheapen it, while
    // they force the radix descent into its position tie-break passes.
    let dists: Vec<f32> = (0..nq * ndb)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2_654_435_761) % 16_777_213;
            h as f32 * 1e-4
        })
        .collect();

    TopkInput {
        distances_gpu: GpuTensor::<R, f32>::from_slice(&dists, vec![nq, ndb], client)
            .expect("GPU allocation exceeds the device binding limit"),
        topk_dists: GpuTensor::<R, f32>::empty(vec![nq, k], client)
            .expect("GPU allocation exceeds the device binding limit"),
        topk_indices: GpuTensor::<R, u32>::empty(vec![nq, k], client)
            .expect("GPU allocation exceeds the device binding limit"),
    }
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

        let query_gpu = GpuTensor::<R, f32>::from_slice(&queries, vec![nq, dim], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let db_gpu = GpuTensor::<R, f32>::from_slice(&db, vec![ndb, dim], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let distances_gpu = GpuTensor::<R, f32>::empty(vec![nq, ndb], &self.client)
            .expect("GPU allocation exceeds the device binding limit");

        let (query_norms_gpu, db_norms_gpu) = if self.cfg.metric == Dist::Cosine {
            let qn = l2_norms(&queries, dim);
            let dn = l2_norms(&db, dim);
            (
                Some(
                    GpuTensor::<R, f32>::from_slice(&qn, vec![nq], &self.client)
                        .expect("GPU allocation exceeds the device binding limit"),
                ),
                Some(
                    GpuTensor::<R, f32>::from_slice(&dn, vec![ndb], &self.client)
                        .expect("GPU allocation exceeds the device binding limit"),
                ),
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
        let limits = GpuLimits::from_client(&self.client);
        let wg_y = pick_wg_y(self.cfg.dim, size_of::<f32>(), &limits)
            .expect("dim outside pick_wg_y table");

        let grid_x = (ndb as u32).div_ceil(WORKGROUP_SIZE_X);
        let (grid_y, grid_z) =
            grid_2d((nq as u32).div_ceil(wg_y), &limits).expect("grid too large");

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

        let limits = GpuLimits::from_client(&self.client);
        let wg_y = pick_wg_y(self.cfg.dim, size_of::<f32>(), &limits)
            .expect("dim outside pick_wg_y table");
        if !tile_fits(wg_y) {
            return Err(format!("wg_y {wg_y} not divisible by TILE_Q {TILE_Q}"));
        }
        let threads_y = wg_y / TILE_Q as u32;

        let grid_x = (ndb as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);
        let (grid_y, grid_z) =
            grid_2d((nq as u32).div_ceil(wg_y), &limits).expect("grid too large");

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
        make_topk_input(&self.client, self.cfg.n_queries, self.cfg.n_db, self.cfg.k)
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;

        seed_topk(
            &self.client,
            &input.topk_dists,
            &input.topk_indices,
            nq,
            self.cfg.k,
        );

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
// 2b. Radix-select top-k
// ──────────────────────────────────────────────

/// Times `radix_select_topk`, the replacement for `extract_topk`.
///
/// Same input, same sentinel seeding inside the timed region, so the two arms
/// are directly comparable.
struct TopkRadixBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for TopkRadixBench<R> {
    type Input = TopkInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        make_topk_input(&self.client, self.cfg.n_queries, self.cfg.n_db, self.cfg.k)
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let k = self.cfg.k;

        seed_topk(&self.client, &input.topk_dists, &input.topk_indices, nq, k);

        let limits = GpuLimits::from_client(&self.client);
        let wg = WORKGROUP_SIZE_X as usize;
        if !radix_select_usable(&self.client, k, size_of::<f32>(), wg, &limits) {
            return Err(format!("radix select unusable at k={k}"));
        }
        let (gx, gy) = grid_2d(nq as u32, &limits).map_err(|e| e.to_string())?;

        unsafe {
            radix_select_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.distances_gpu.into_tensor_arg(),
                input.topk_dists.into_tensor_arg(),
                input.topk_indices.into_tensor_arg(),
                0u32,
                ndb as u32,
                k as u32,
                k,
                wg,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "topk_radix_{}q_{}db_k{}",
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
    // The top-k arms are gated per configuration inside `run_topk_pair`, which
    // covers every `k` in the sweep rather than one sample of it.
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
        println!("{}", dist_bench.name());
        println!("{:?}", dist_bench.run(TimingMethod::System));

        println!("{}", dist_reg_bench.name());
        println!("{:?}", dist_reg_bench.run(TimingMethod::System));
    }

    // ── Top-k reducer sweeps ──
    //
    // The reducers ignore dim and metric entirely, so running them inside the
    // loop above measured k=15 five times over and never touched the chunked
    // merge arm. k and n_db are the two axes that do move them, and they are
    // swept separately: without the n_db axis, "cost rises with k" cannot be
    // told apart from "cost rises with work".

    println!("\n====== Top-k reducers: k sweep (8192q x 16384db) ======");
    // Fine granularity below 50: radix select is linear in n_db where the serial
    // kernel is sublinear, so the crossover is expected somewhere in the teens
    // to thirties and needs resolving rather than guessing.
    for k in [10usize, 15, 20, 25, 30, 40, 50, 100, 150, 250] {
        run_topk_pair::<R>(&BenchConfig::euclidean(8192, 16_384, 32, k), &client);
    }

    // Two axes rather than one. Radix select is linear in n_db while the serial
    // kernel is sublinear in it, so radix's worst corner is small k with large
    // n_db, and that corner is what decides whether a dispatch threshold is
    // needed at all.
    println!("\n====== Top-k reducers: n_db x k corners (4096q) ======");
    for ndb in [4096usize, 16_384, 65_536] {
        for k in [10usize, 50] {
            run_topk_pair::<R>(&BenchConfig::euclidean(4096, ndb, 32, k), &client);
        }
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

    // Sweep `k` at one shape to locate the dispatch crossover. This has to be
    // measured end to end rather than on the isolated reducer: with n_db=50000
    // the pipeline runs four DB chunks, and `extract_topk` becomes nearly free
    // after the first because a tight running top-k rejects almost every
    // candidate at its `dist < local[k - 1]` test. Radix select pays full price
    // on every chunk. The isolated bench only ever measures the cold chunk, so
    // it reports radix winning at every `k` while the pipeline does not.
    let crossover_configs: Vec<BenchConfig> = [15usize, 30, 50, 75, 100, 150]
        .into_iter()
        .map(|k| BenchConfig::euclidean(50_000, 50_000, 32, k))
        .collect();

    println!("\n====== Full pipeline ======");
    for cfg in pipeline_configs.into_iter().chain(crossover_configs) {
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

/// Gate the serial and radix arms against each other at this exact config
/// before either is timed.
///
/// Every launch here is `launch_unchecked`, so a kernel that busts a device
/// limit does no work, returns whatever was in the buffer and reports no error.
/// Radix select is the fast arm, so an unguarded timing is precisely where that
/// failure would look like a triumph.
///
/// ### Params
///
/// * `cfg` - Configuration under test
/// * `client` - Compute client
fn validate_topk_arms<R: Runtime>(cfg: &BenchConfig, client: &ComputeClient<R>) {
    let serial = TopkBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };
    let radix = TopkRadixBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };

    let s_in = serial.prepare();
    serial.execute(s_in.clone()).expect("serial topk failed");
    serial.sync();
    let sd = s_in.topk_dists.read(client).expect("read failed");
    let si = s_in.topk_indices.read(client).expect("read failed");

    let r_in = radix.prepare();
    radix.execute(r_in.clone()).expect("radix topk failed");
    radix.sync();
    let rd = r_in.topk_dists.read(client).expect("read failed");
    let ri = r_in.topk_indices.read(client).expect("read failed");

    let untouched = rd.iter().filter(|d| **d >= f32::MAX).count();
    assert_eq!(
        untouched, 0,
        "radix k={}: {untouched} sentinels survived, the kernel almost certainly did no work",
        cfg.k
    );

    // Bit-exact on distances and equal on indices: the radix kernel selects
    // under the same total order the serial one produces, so there is no tie
    // slack to allow for.
    for (slot, ((a, b), (x, y))) in sd
        .iter()
        .zip(rd.iter())
        .zip(si.iter().zip(ri.iter()))
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "k={} slot {slot}: serial={a} radix={b}",
            cfg.k
        );
        assert_eq!(
            x, y,
            "k={} slot {slot}: serial idx={x} radix idx={y}",
            cfg.k
        );
    }
}

/// Run the top-k reducers on one config, printing the shared-memory budget the
/// radix arm is working against alongside the timings.
///
/// Residency is the quantity the high-k diagnosis predicts falls off a cliff, so
/// it is printed next to the timing rather than reconstructed afterwards. It
/// describes the radix arm only: the serial `extract_topk` uses no shared
/// memory at all and is bounded by its per-thread register arrays instead.
///
/// ### Params
///
/// * `cfg` - Configuration to run
/// * `client` - Compute client
fn run_topk_pair<R: Runtime>(cfg: &BenchConfig, client: &ComputeClient<R>) {
    let limits = GpuLimits::from_client(client);
    let radix_smem = radix_select_smem_bytes(cfg.k, WORKGROUP_SIZE_X as usize, SURV_ARRAYS_DENSE);

    println!(
        "\n--- k={} | {}q x {}db | radix smem={}B resident={} ---\n",
        cfg.k,
        cfg.n_queries,
        cfg.n_db,
        radix_smem,
        resident_workgroups(radix_smem, &limits),
    );

    validate_topk_arms::<R>(cfg, client);

    let serial = TopkBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };
    let radix = TopkRadixBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };

    println!("{}", serial.name());
    println!("{:?}", serial.run(TimingMethod::System));

    println!("{}", radix.name());
    println!("{:?}", radix.run(TimingMethod::System));
}

fn main() {
    run_exhaustive_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
