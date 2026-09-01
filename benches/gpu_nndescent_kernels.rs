//! GPU kernel microbenchmarks for the NNDescent local join, using CubeCL's
//! `Benchmark` trait.
//!
//! Run with: cargo bench --bench gpu_nndescent_kernels --features gpu
//!
//! The kernel under test is [`local_join_shared`], which stages candidate
//! vectors in shared memory and evaluates the candidate upper triangle. Its
//! cost is a step function of `plan_local_join_staging`'s `block`, so the sweep
//! is over `(dim, build_k)` and every timing is printed next to the staging
//! decision that produced it.
//!
//! Note: no `#![allow(dead_code)]` here on purpose, matching
//! `gpu_exhaustive_kernels`. A bench arm that is defined and never constructed
//! should be a compiler warning, not a silent gap in the comparison.

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::gpu::forest_gpu::{gpu_forest_init, mark_all_new};
use ann_search_rs::gpu::nndescent_gpu::{
    build_reverse_candidates, init_random_graph, local_join_shared, reset_proposals, MAX_PROPOSALS,
};
use ann_search_rs::gpu::*;
use ann_search_rs::utils::dist::Dist;
use cubecl_utils_rs::prelude::*;

/// Sampling rate the production NNDescent loop defaults to. Held fixed across
/// the sweep so `total_cands` stays comparable between cells.
const RHO: f32 = 0.5;

/// Clusters in the synthetic data. Local join cost depends on how often a
/// proposal beats the incumbent, which on uniform noise is unrepresentatively
/// low, so the generator produces clustered data like the real workloads.
const N_CLUSTERS: usize = 25;

/// Points in the microbenchmark.
///
/// The kernel launches one cube per node, so per-node cost is what the sweep is
/// after and `n` only scales wall-clock. 25k is already far more cubes than the
/// device has cores, so it saturates without spending 20 minutes at dim 1024.
/// End-to-end behaviour at 100k and 1M is the examples' job, not this bench's.
const N_POINTS: usize = 25_000;

/// Shared config so every arm runs on identical data.
#[derive(Clone)]
struct BenchConfig {
    /// Number of points in the graph
    n: usize,
    /// Unpadded embedding dimensionality
    dim: usize,
    /// Working degree of the NNDescent graph
    build_k: usize,
    /// Distance metric. Cosine carries the extra `shared_norms` staging.
    metric: Dist,
    /// Overrides the staging plan's unroll depth, for sweeping the knee.
    /// `None` uses whatever the production heuristic picks.
    unroll: Option<usize>,
    /// Overrides the staging plan's cube shape, for sweeping occupancy.
    cube: Option<(u32, u32)>,
}

impl BenchConfig {
    /// Euclidean config, the common case.
    fn euclidean(n: usize, dim: usize, build_k: usize) -> Self {
        Self {
            n,
            dim,
            build_k,
            metric: Dist::SquaredEuclidean,
            unroll: None,
            cube: None,
        }
    }

    /// Cosine config.
    fn cosine(n: usize, dim: usize, build_k: usize) -> Self {
        Self {
            n,
            dim,
            build_k,
            metric: Dist::Cosine,
            unroll: None,
            cube: None,
        }
    }

    /// Same config with the pair-loop unroll depth pinned.
    fn with_unroll(mut self, unroll: usize) -> Self {
        self.unroll = Some(unroll);
        self
    }

    /// Same config with the cube shape pinned, for the occupancy sweep.
    fn with_cube(mut self, x: u32, y: u32) -> Self {
        self.cube = Some((x, y));
        self
    }

    /// Short metric tag for benchmark names.
    fn tag(&self) -> &'static str {
        match self.metric {
            Dist::Cosine => "cos",
            _ => "l2",
        }
    }

    /// Whether the kernel takes its cosine arm.
    fn use_cosine(&self) -> bool {
        self.metric == Dist::Cosine
    }
}

/// Deterministic clustered synthetic data, row-major `[n, dim]`.
///
/// Each point is a cluster centre plus a per-coordinate jitter, both derived
/// from a multiplicative hash so the data is reproducible without an RNG
/// dependency and has no repeating period at these sizes.
///
/// ### Params
///
/// * `n` - Number of points
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// Flattened row-major matrix of length `n * dim`.
fn make_clustered(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let cluster = i % N_CLUSTERS;
        for d in 0..dim {
            let ch =
                ((cluster * 7919 + d * 104_729) as u64).wrapping_mul(2_654_435_761) % 16_777_213;
            let jh = ((i * 31 + d * 6151) as u64).wrapping_mul(2_246_822_519) % 16_777_213;
            let centre = (ch as f32 / 16_777_213.0) * 20.0 - 10.0;
            let jitter = (jh as f32 / 16_777_213.0) * 2.0 - 1.0;
            out[i * dim + d] = centre + jitter;
        }
    }
    out
}

/// Row-wise L2 norms, needed by the cosine arm.
///
/// ### Params
///
/// * `flat` - Row-major matrix
/// * `dim` - Row length
///
/// ### Returns
///
/// One norm per row.
fn l2_norms(flat: &[f32], dim: usize) -> Vec<f32> {
    flat.chunks_exact(dim)
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// Device-resident state one local-join launch reads and writes.
struct LocalJoinInput<R: Runtime> {
    /// Padded vectors `[n, dim_padded]`
    vectors: GpuTensor<R, f32>,
    /// L2 norms `[n]`, or a one-element dummy on the euclidean path
    norms: GpuTensor<R, f32>,
    /// kNN graph indices `[n, build_k]` with the IS_NEW flag in the MSB
    graph_idx: GpuTensor<R, u32>,
    /// kNN graph distances `[n, build_k]`
    graph_dist: GpuTensor<R, f32>,
    /// Reverse edges `[n, build_k]`
    reverse_idx: GpuTensor<R, u32>,
    /// Valid reverse edges per node `[n]`
    reverse_count: GpuTensor<R, u32>,
    /// Proposal indices `[n, MAX_PROPOSALS]`
    prop_idx: GpuTensor<R, u32>,
    /// Proposal distances `[n, MAX_PROPOSALS]`
    prop_dist: GpuTensor<R, f32>,
    /// Proposals received per node `[n]`
    prop_count: GpuTensor<R, u32>,
    /// Global update accumulator `[1]`
    update_counter: GpuTensor<R, u32>,
}

impl<R: Runtime> Clone for LocalJoinInput<R> {
    fn clone(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            norms: self.norms.clone(),
            graph_idx: self.graph_idx.clone(),
            graph_dist: self.graph_dist.clone(),
            reverse_idx: self.reverse_idx.clone(),
            reverse_count: self.reverse_count.clone(),
            prop_idx: self.prop_idx.clone(),
            prop_dist: self.prop_dist.clone(),
            prop_count: self.prop_count.clone(),
            update_counter: self.update_counter.clone(),
        }
    }
}

struct LocalJoinBench<R: Runtime> {
    cfg: BenchConfig,
    client: ComputeClient<R>,
}

impl<R: Runtime> LocalJoinBench<R> {
    /// Padded dimensionality the kernel actually indexes.
    fn dim_padded(&self) -> usize {
        self.cfg.dim.next_multiple_of(LINE_SIZE)
    }

    /// Staging plan for this config, i.e. the thing the sweep is measuring.
    fn staging(&self) -> LocalJoinStaging {
        let limits = GpuLimits::from_client(&self.client);
        plan_local_join_staging(
            self.dim_padded(),
            self.cfg.build_k * 2,
            size_of::<f32>(),
            self.cfg.use_cosine(),
            &limits,
        )
        .expect("dim too high for the shared-memory budget")
    }
}

impl<R: Runtime> Benchmark for LocalJoinBench<R> {
    type Input = LocalJoinInput<R>;
    type Output = ();

    /// Build a realistic iteration-1 graph state.
    ///
    /// Random init, GPU forest refinement, mark-all-new, then one reverse-edge
    /// build. Skipping the forest step would leave a near-random graph, where
    /// almost every proposal is accepted and the atomic traffic is nothing like
    /// production. Skipping `mark_all_new` or the reverse-edge build would send
    /// the kernel straight to its `has_new == 0` early exit and measure nothing.
    fn prepare(&self) -> Self::Input {
        let n = self.cfg.n;
        let dim = self.cfg.dim;
        let dim_padded = self.dim_padded();
        let dim_vec = dim_padded / LINE_SIZE;
        let build_k = self.cfg.build_k;
        let use_cosine = self.cfg.use_cosine();
        let limits = GpuLimits::from_client(&self.client);

        let flat = make_clustered(n, dim);
        let padded = if dim_padded != dim {
            pad_vectors(&flat, n, dim, dim_padded)
        } else {
            flat.clone()
        };

        let vectors = GpuTensor::<R, f32>::from_slice(&padded, vec![n, dim_padded], &self.client)
            .expect("GPU allocation exceeds the device binding limit");

        let norms = if use_cosine {
            GpuTensor::<R, f32>::from_slice(&l2_norms(&flat, dim), vec![n], &self.client)
        } else {
            GpuTensor::<R, f32>::from_slice(&[0.0f32], vec![1], &self.client)
        }
        .expect("GPU allocation exceeds the device binding limit");

        let graph_idx = GpuTensor::<R, u32>::from_slice(
            &vec![0x7FFF_FFFFu32; n * build_k],
            vec![n, build_k],
            &self.client,
        )
        .expect("GPU allocation exceeds the device binding limit");
        let graph_dist = GpuTensor::<R, f32>::from_slice(
            &vec![f32::MAX; n * build_k],
            vec![n, build_k],
            &self.client,
        )
        .expect("GPU allocation exceeds the device binding limit");

        let prop_idx = GpuTensor::<R, u32>::empty(vec![n, MAX_PROPOSALS], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let prop_dist = GpuTensor::<R, f32>::empty(vec![n, MAX_PROPOSALS], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let prop_count = GpuTensor::<R, u32>::empty(vec![n], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let update_counter = GpuTensor::<R, u32>::empty(vec![1], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let reverse_idx = GpuTensor::<R, u32>::empty(vec![n, build_k], &self.client)
            .expect("GPU allocation exceeds the device binding limit");
        let reverse_count = GpuTensor::<R, u32>::empty(vec![n], &self.client)
            .expect("GPU allocation exceeds the device binding limit");

        let (grid_n_x, grid_n_y) =
            grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X), &limits).expect("grid too large");

        unsafe {
            init_random_graph::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                LINE_SIZE,
                vectors.clone().into_tensor_arg(),
                norms.clone().into_tensor_arg(),
                graph_idx.clone().into_tensor_arg(),
                graph_dist.clone().into_tensor_arg(),
                n as u32,
                42u32,
                use_cosine,
                dim_vec,
                build_k,
            );
        }

        let n_trees = (5 + ((n as f64).powf(0.25)).round() as usize).min(20);
        gpu_forest_init(
            &vectors,
            &norms,
            &graph_idx,
            &graph_dist,
            &prop_idx,
            &prop_dist,
            &prop_count,
            &update_counter,
            n,
            dim,
            dim_padded,
            n_trees,
            42,
            use_cosine,
            false,
            &self.client,
        )
        .expect("forest init failed");

        unsafe {
            mark_all_new::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(
                    grid_2d((n * build_k) as u32 / WORKGROUP_SIZE_X + 1, &limits)
                        .expect("grid too large")
                        .0,
                    grid_2d((n * build_k) as u32 / WORKGROUP_SIZE_X + 1, &limits)
                        .expect("grid too large")
                        .1,
                    1,
                ),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx.clone().into_tensor_arg(),
                (n * build_k) as u32,
            );

            // Reverse edges are read-only for the local join and are not
            // rebuilt between iterations here, so they are staged once.
            reset_proposals::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                reverse_count.clone().into_tensor_arg(),
                update_counter.clone().into_tensor_arg(),
                n as u32,
            );

            build_reverse_candidates::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx.clone().into_tensor_arg(),
                reverse_idx.clone().into_tensor_arg(),
                reverse_count.clone().into_tensor_arg(),
                n as u32,
                build_k as u32,
            );
        }

        future::block_on(self.client.sync()).expect("sync failed");

        LocalJoinInput {
            vectors,
            norms,
            graph_idx,
            graph_dist,
            reverse_idx,
            reverse_count,
            prop_idx,
            prop_dist,
            prop_count,
            update_counter,
        }
    }

    /// One reset plus one local join.
    ///
    /// The reset is inside the timed region deliberately. `Benchmark::run`
    /// calls `prepare` once and clones the input for every iteration, and
    /// `GpuTensor::clone` is a handle clone, so without it iterations 2..n find
    /// `prop_count` already at the `MAX_PROPOSALS` cap and skip nearly every
    /// buffer write. Cost is `O(n)` writes against the join's
    /// `O(n * total_cands^2 * dim)`, and it is paid identically by every arm.
    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let n = self.cfg.n;
        let dim_vec = self.dim_padded() / LINE_SIZE;
        let limits = GpuLimits::from_client(&self.client);
        let staging = self.staging();

        let (grid_n_x, grid_n_y) =
            grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X), &limits).expect("grid too large");
        let (cubes_x, cubes_y) = grid_2d(n as u32, &limits).expect("grid too large");
        let (cube_x, cube_y) = self.cfg.cube.unwrap_or((staging.cube_x, staging.cube_y));

        unsafe {
            reset_proposals::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.prop_count.clone().into_tensor_arg(),
                input.update_counter.clone().into_tensor_arg(),
                n as u32,
            );

            local_join_shared::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(cube_x, cube_y),
                LINE_SIZE,
                input.vectors.into_tensor_arg(),
                input.norms.into_tensor_arg(),
                input.graph_idx.into_tensor_arg(),
                input.graph_dist.into_tensor_arg(),
                input.reverse_idx.into_tensor_arg(),
                input.reverse_count.into_tensor_arg(),
                input.prop_idx.into_tensor_arg(),
                input.prop_dist.into_tensor_arg(),
                input.prop_count.into_tensor_arg(),
                n as u32,
                (RHO * 65535.0) as u32,
                0x9E37_79B9u32,
                MAX_PROPOSALS as u32,
                self.cfg.use_cosine(),
                dim_vec,
                staging.row_lines,
                self.cfg.build_k,
                staging.block,
                staging.single_block,
                staging.buf_a_lines,
                staging.buf_b_lines,
                staging.norm_buf_len,
                self.cfg
                    .unroll
                    .map(|u| u.min(dim_vec).max(1))
                    .unwrap_or(staging.line_unroll),
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        let unroll = self
            .cfg
            .unroll
            .map(|u| u.to_string())
            .unwrap_or_else(|| "auto".to_string());
        let cube = self
            .cfg
            .cube
            .map(|(x, y)| format!("{x}x{y}"))
            .unwrap_or_else(|| "auto".to_string());
        format!(
            "local_join_{}n_{}d_bk{}_{}_u{}_c{}",
            self.cfg.n,
            self.cfg.dim,
            self.cfg.build_k,
            self.cfg.tag(),
            unroll,
            cube
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

/// Run one local join and return the total proposals emitted.
///
/// Every launch in this crate is `launch_unchecked`, so a dispatch that busts a
/// device limit does no work, returns zeros and reports no error. This kernel's
/// documented failure mode is exactly that: it reports success while updating
/// nothing. An implausibly fast arm is the signature, so no timing here is
/// reported without this number next to it.
///
/// ### Params
///
/// * `bench` - Configured benchmark arm
///
/// ### Returns
///
/// Sum of `prop_count` across all nodes.
fn checksum<R: Runtime>(bench: &LocalJoinBench<R>) -> u64 {
    let input = bench.prepare();
    bench.execute(input.clone()).expect("local join failed");
    bench.sync();
    input
        .prop_count
        .read(&bench.client)
        .expect("read failed")
        .iter()
        .map(|c| *c as u64)
        .sum()
}

/// Run one config, printing the staging decision and the occupancy numbers
/// alongside the timing.
///
/// The three walls from the profiling notes need all three of these together:
/// `slab_fill` says how much of the cube the pair slab can even keep busy,
/// residency says how much latency the cube can hide, and the estimated
/// arithmetic intensity says which ceiling is in reach. Reconstructing them
/// after the fact from a table of timings does not work, so they are printed
/// inline.
///
/// ### Params
///
/// * `cfg` - Configuration to run
/// * `client` - Compute client
fn run_local_join<R: Runtime>(cfg: &BenchConfig, client: &ComputeClient<R>) {
    let bench = LocalJoinBench::<R> {
        cfg: cfg.clone(),
        client: client.clone(),
    };

    let limits = GpuLimits::from_client(client);
    let staging = bench.staging();
    let max_cands = cfg.build_k * 2;
    let dim_padded = bench.dim_padded();

    let footprint = local_join_smem_bytes(&staging, max_cands, size_of::<f32>());

    let n_blocks = max_cands.div_ceil(staging.block);

    // Upper bound: rho sampling keeps roughly half of k + rev_k, and rev_k is
    // clamped to k. Labelled an estimate because the real count is runtime data.
    let est_cands = (max_cands as f64 * RHO as f64).round();
    let est_pairs = cfg.n as f64 * est_cands * (est_cands - 1.0) / 2.0;
    let est_gflop = est_pairs * dim_padded as f64 * 2.0 / 1e9;

    let threads = (staging.cube_x * staging.cube_y) as f64;
    let slab = (staging.block * staging.block) as f64;
    let lane_fill = (slab / threads).min(1.0);

    println!(
        "\n--- dim={} (padded {}) build_k={} {} | block={} blocks={} single={} \
         cube={}x{} smem={}B resident={} slab_fill={:.0}% est={:.2} GFLOP ---",
        cfg.dim,
        dim_padded,
        cfg.build_k,
        cfg.tag(),
        staging.block,
        n_blocks,
        staging.single_block,
        staging.cube_x,
        staging.cube_y,
        footprint,
        resident_workgroups(footprint, &limits),
        lane_fill * 100.0,
        est_gflop,
    );

    let emitted = checksum(&bench);
    assert!(
        emitted > 0,
        "dim={} build_k={}: zero proposals emitted, the kernel almost certainly did no work",
        cfg.dim,
        cfg.build_k
    );
    println!("proposals emitted: {emitted}");

    println!("{}", bench.name());
    // Display, not Debug: it reports median, min and max together. Shader
    // compilation is not in these samples, since `Benchmark::run` warms up
    // first, but the spread still matters. A wide min-to-max gap here means
    // the machine was not quiet, and every number in the run is then suspect.
    match bench.run(TimingMethod::System) {
        Ok(durations) => println!("{durations}"),
        Err(e) => panic!("bench run failed: {e}"),
    }
}

/// The sweep.
///
/// `dim 64` is the single-block control: it is the only width at `build_k = 45`
/// where `plan_local_join_staging` does not block, so it is what turns "the
/// blocked path is slower" into a ratio rather than an assertion.
///
/// ### Params
///
/// * `device` - Target device
fn run_nndescent_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let n = N_POINTS;

    // Unroll sweep. Kept in the suite rather than deleted after the fact: the
    // knee has moved between kernels in this crate before, and 16 regressing
    // ~2x while 8 wins is exactly the register-pressure edge a retune on other
    // hardware would need to re-find. Set `SWEEP_UNROLL=1` to run it.
    if std::env::var("SWEEP_UNROLL").is_ok() {
        println!("\n########## pair-loop unroll sweep ##########");
        for dim in [64usize, 128, 1024] {
            for u in [1usize, 2, 4, 8, 16] {
                run_local_join::<R>(&BenchConfig::euclidean(n, dim, 45).with_unroll(u), &client);
            }
        }
    }

    // Cube-width sweep. Shared memory is per cube, so a wider cube costs no
    // footprint and buys resident threads, but the pair slab is only
    // `block * block` slots wide and runs out first. Set `SWEEP_CUBE=1`.
    if std::env::var("SWEEP_CUBE").is_ok() {
        println!("\n########## cube-width sweep ##########");
        for dim in [128usize, 256, 512, 1024] {
            for (x, y) in [(8u32, 4u32), (8, 8), (16, 8), (16, 16)] {
                run_local_join::<R>(&BenchConfig::euclidean(n, dim, 45).with_cube(x, y), &client);
            }
        }
    }

    println!("\n########## main sweep ##########");
    for build_k in [45usize, 64] {
        for dim in [64usize, 128, 256, 512, 1024] {
            run_local_join::<R>(&BenchConfig::euclidean(n, dim, build_k), &client);
        }
    }

    // Cosine carries the larger metadata footprint, so it shifts `block` and
    // has to be swept rather than assumed to track euclidean.
    run_local_join::<R>(&BenchConfig::cosine(n, 128, 45), &client);
}

fn main() {
    run_nndescent_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
