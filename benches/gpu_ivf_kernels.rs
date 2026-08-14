//! GPU kernel microbenchmarks using CubeCL's Benchmark trait. This bench covers
//! the IVF-GPU pipeline (build, cross-batch query, self-kNN) and the isolated
//! IVF top-k reducer, serial against radix select.
//!
//! Run with: cargo bench --bench gpu_ivf_kernels --features gpu

use ann_search_rs::prelude::KMeansTrainingParams;
use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::gpu::dist_gpu::{init_topk, reduce_ivf_topk};
use ann_search_rs::gpu::topk_gpu::{
    radix_select_ivf_topk, radix_select_smem_bytes, SURV_ARRAYS_IVF,
};
use ann_search_rs::gpu::WORKGROUP_SIZE_X;
use ann_search_rs::utils::dist::Dist;
use cubecl_utils_rs::prelude::*;

// ──────────────────────────────────────────────
// IVF Pipeline benchmarks
// ──────────────────────────────────────────────

use ann_search_rs::gpu::ivf_gpu::IvfIndexGpu;
use faer::Mat;

struct IvfBuildBench<R: Runtime> {
    n: usize,
    dim: usize,
    nlist: usize,
    metric: Dist,
    device: R::Device,
}

#[derive(Clone)]
struct IvfBuildInput {
    data: Vec<f32>,
    n: usize,
    dim: usize,
}

fn default_k_means_params() -> Option<KMeansTrainingParams> {
    Some(KMeansTrainingParams::new(10, None, None))
}

impl<R: Runtime> Benchmark for IvfBuildBench<R> {
    type Input = IvfBuildInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        IvfBuildInput {
            data: (0..self.n * self.dim)
                .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
                .collect(),
            n: self.n,
            dim: self.dim,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let mat = Mat::from_fn(input.n, input.dim, |i, j| input.data[i * input.dim + j]);
        // Propagate rather than discard: a silently failing build would
        // otherwise report as an implausibly fast one.
        IvfIndexGpu::<f32, R>::build(
            mat.as_ref(),
            self.metric,
            Some(self.nlist),
            default_k_means_params(),
            42,
            false,
            self.device.clone(),
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn name(&self) -> String {
        format!("ivf_build_{}n_{}d_{}lists", self.n, self.dim, self.nlist)
    }

    fn sync(&self) {
        // Build includes its own syncs
    }
}

struct IvfQueryBench<R: Runtime> {
    index: std::sync::Arc<IvfIndexGpu<f32, R>>,
    n_queries: usize,
    dim: usize,
    k: usize,
    nprobe: usize,
    client: ComputeClient<R>,
}

#[derive(Clone)]
struct IvfQueryInput {
    queries: Vec<f32>,
}

impl<R: Runtime> Benchmark for IvfQueryBench<R> {
    type Input = IvfQueryInput;
    type Output = (Vec<Vec<usize>>, Vec<Vec<f32>>);

    fn prepare(&self) -> Self::Input {
        IvfQueryInput {
            queries: (0..self.n_queries * self.dim)
                .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
                .collect(),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let mat = Mat::from_fn(self.n_queries, self.dim, |i, j| {
            input.queries[i * self.dim + j]
        });
        let result = self
            .index
            .query_batch(mat.as_ref(), self.k, Some(self.nprobe), None, false)
            .unwrap();
        Ok(result)
    }

    fn name(&self) -> String {
        format!(
            "ivf_query_{}q_{}probe_k{}",
            self.n_queries, self.nprobe, self.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

struct IvfKnnBench<R: Runtime> {
    index: std::sync::Arc<IvfIndexGpu<f32, R>>,
    n: usize, // store separately since index.n is private
    k: usize,
    nprobe: usize,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for IvfKnnBench<R> {
    type Input = ();
    type Output = (Vec<Vec<usize>>, Option<Vec<Vec<f32>>>);

    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _input: Self::Input) -> Result<Self::Output, String> {
        let result = self
            .index
            .generate_knn(self.k, Some(self.nprobe), None, true, false)
            .unwrap();
        Ok(result)
    }

    fn name(&self) -> String {
        format!("ivf_knn_{}n_{}probe_k{}", self.n, self.nprobe, self.k)
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// IVF top-k reducer, isolated
// ──────────────────────────────────────────────

/// Times the serial `reduce_ivf_topk` fallback on its own.
///
/// This is the arm radix select replaced, and it is still the path taken for
/// non-f32 elements and runtimes without `u32` atomics, so it stays worth
/// measuring. It inserts straight into global memory and therefore needs the
/// sentinel seeding that the radix path does not.
struct IvfReduceBench<R: Runtime> {
    /// Number of queries, one workgroup each
    n_queries: usize,
    /// Candidate slots per query
    max_candidates: usize,
    /// Neighbours to select
    k: usize,
    client: ComputeClient<R>,
}

struct IvfReduceInput<R: Runtime> {
    candidate_dists: GpuTensor<R, f32>,
    candidate_indices: GpuTensor<R, u32>,
    candidates_per_query: GpuTensor<R, u32>,
    out_dists: GpuTensor<R, f32>,
    out_indices: GpuTensor<R, u32>,
}

impl<R: Runtime> Clone for IvfReduceInput<R> {
    fn clone(&self) -> Self {
        Self {
            candidate_dists: self.candidate_dists.clone(),
            candidate_indices: self.candidate_indices.clone(),
            candidates_per_query: self.candidates_per_query.clone(),
            out_dists: self.out_dists.clone(),
            out_indices: self.out_indices.clone(),
        }
    }
}

impl<R: Runtime> Benchmark for IvfReduceBench<R> {
    type Input = IvfReduceInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let (nq, mc) = (self.n_queries, self.max_candidates);

        // Deterministic sawtooth, same generator as the exhaustive top-k bench.
        let dists: Vec<f32> = (0..nq * mc)
            .map(|i| ((i * 7 + 13) % 1000) as f32 * 0.01)
            .collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32).collect();
        let counts: Vec<u32> = vec![mc as u32; nq];

        IvfReduceInput {
            candidate_dists: GpuTensor::<R, f32>::from_slice(&dists, vec![nq, mc], &self.client)
                .expect("GPU allocation exceeds the device binding limit"),
            candidate_indices: GpuTensor::<R, u32>::from_slice(
                &indices,
                vec![nq, mc],
                &self.client,
            )
            .expect("GPU allocation exceeds the device binding limit"),
            candidates_per_query: GpuTensor::<R, u32>::from_slice(&counts, vec![nq], &self.client)
                .expect("GPU allocation exceeds the device binding limit"),
            out_dists: GpuTensor::<R, f32>::empty(vec![nq, self.k], &self.client)
                .expect("GPU allocation exceeds the device binding limit"),
            out_indices: GpuTensor::<R, u32>::empty(vec![nq, self.k], &self.client)
                .expect("GPU allocation exceeds the device binding limit"),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        // Unlike the radix reducer this one inserts directly into the output
        // buffer, so the sentinel seeding is part of its cost and belongs in the
        // timed region.
        let limits = GpuLimits::from_client(&self.client);
        let init_gx = (self.k as u32).div_ceil(WORKGROUP_SIZE_X);
        let (init_gy, init_gz) =
            grid_2d((self.n_queries as u32).div_ceil(4), &limits).map_err(|e| e.to_string())?;
        unsafe {
            init_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(init_gx, init_gy, init_gz),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                input.out_dists.clone().into_tensor_arg(),
                input.out_indices.clone().into_tensor_arg(),
                4,
            );
        }

        let (gx, gy) = grid_2d((self.n_queries as u32).div_ceil(WORKGROUP_SIZE_X), &limits)
            .map_err(|e| e.to_string())?;
        unsafe {
            reduce_ivf_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.candidate_dists.into_tensor_arg(),
                input.candidate_indices.into_tensor_arg(),
                input.candidates_per_query.into_tensor_arg(),
                input.out_dists.into_tensor_arg(),
                input.out_indices.into_tensor_arg(),
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "ivf_serial_{}q_{}cand_k{}",
            self.n_queries, self.max_candidates, self.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

/// Times `radix_select_ivf_topk`, the replacement, on identical data.
struct RadixIvfReduceBench<R: Runtime> {
    /// Number of queries, one workgroup each
    n_queries: usize,
    /// Candidate slots per query
    max_candidates: usize,
    /// Neighbours to select
    k: usize,
    client: ComputeClient<R>,
}

impl<R: Runtime> Benchmark for RadixIvfReduceBench<R> {
    type Input = IvfReduceInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        IvfReduceBench {
            n_queries: self.n_queries,
            max_candidates: self.max_candidates,
            k: self.k,
            client: self.client.clone(),
        }
        .prepare()
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        // This one writes every slot unconditionally, so no sentinel seeding is
        // needed and repeated iterations do identical work.
        let limits = GpuLimits::from_client(&self.client);
        let (gx, gy) = grid_2d(self.n_queries as u32, &limits).map_err(|e| e.to_string())?;

        unsafe {
            radix_select_ivf_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.candidate_dists.into_tensor_arg(),
                input.candidate_indices.into_tensor_arg(),
                input.candidates_per_query.into_tensor_arg(),
                input.out_dists.into_tensor_arg(),
                input.out_indices.into_tensor_arg(),
                self.k as u32,
                self.k,
                WORKGROUP_SIZE_X as usize,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "ivf_radix_{}q_{}cand_k{}",
            self.n_queries, self.max_candidates, self.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

/// Run the isolated IVF reducer once and check it did real work before timing.
///
/// The output is seeded with the `f32::MAX` sentinel first, so a
/// `launch_unchecked` that busted a device limit shows up as surviving sentinels
/// rather than as an implausibly fast run. Rows are also checked for ascending
/// order, which the reducer guarantees.
///
/// ### Params
///
/// * `bench` - Configured reducer bench, either arm
/// * `n_queries` - Query count
/// * `k` - Neighbours
/// * `client` - Compute client
///
/// ### Returns
///
/// The selected distances, so callers can cross-check the two arms.
fn validate_ivf_reduce<R: Runtime, B>(
    bench: &B,
    n_queries: usize,
    k: usize,
    client: &ComputeClient<R>,
) -> Vec<f32>
where
    B: Benchmark<Input = IvfReduceInput<R>, Output = ()>,
{
    let limits = GpuLimits::from_client(client);
    let input = bench.prepare();

    let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
    let (init_gy, init_gz) = grid_2d((n_queries as u32).div_ceil(4), &limits).unwrap();
    unsafe {
        init_topk::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(init_gx, init_gy, init_gz),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
            input.out_dists.clone().into_tensor_arg(),
            input.out_indices.clone().into_tensor_arg(),
            4,
        );
    }

    bench.execute(input.clone()).expect("ivf reduce failed");
    bench.sync();

    let dists = input.out_dists.read(client).expect("read failed");

    let untouched = dists.iter().filter(|d| **d >= f32::MAX).count();
    assert_eq!(
        untouched,
        0,
        "{}: {untouched} sentinel entries survived, the kernel almost certainly did no work",
        bench.name()
    );

    for (row, chunk) in dists.chunks_exact(k).enumerate() {
        assert!(
            chunk.windows(2).all(|w| w[0] <= w[1]),
            "{}: row {row} is not sorted ascending",
            bench.name()
        );
    }

    dists
}

/// Print the device ceilings that silently gate `launch_unchecked`.
fn report_device_limits<R: Runtime>(client: &ComputeClient<R>) {
    let props = client.properties();
    let hw = &props.hardware;
    println!("====== Device ======");
    println!("runtime          : {}", R::name(client));
    println!(
        "plane size       : {} .. {}",
        hw.plane_size_min, hw.plane_size_max
    );
    println!("max shared mem   : {} bytes", hw.max_shared_memory_size);
    println!("max binding size : {} bytes", props.memory.max_page_size);
    println!("max cube count   : {:?}", hw.max_cube_count);
    println!();
}

/// Assert the IVF query returned a full result set before trusting any timing.
///
/// Two failure modes this catches. A `launch_unchecked` that busted a device
/// limit returns zeros and reports no error. And short result rows are the exact
/// regression fixed in `a7d88a7`: probing exactly `nprobe` cells can reach fewer
/// than `k` vectors, so `nprobe` has to act as a floor rather than a ceiling.
///
/// ### Params
///
/// * `indices` - Per-query neighbour lists
/// * `k` - Requested neighbour count
/// * `label` - Context for the panic message
fn validate_knn(indices: &[Vec<usize>], k: usize, label: &str) {
    let short = indices.iter().filter(|row| row.len() < k).count();
    assert_eq!(
        short,
        0,
        "{label}: {short}/{} queries returned fewer than k={k} neighbours",
        indices.len()
    );
}

fn run_ivf_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    report_device_limits::<R>(&client);

    // ── Isolated reducer sweep ──
    //
    // Runs first, and deliberately outside the index loop: the reducer sees only
    // (n_queries, max_candidates, k) and nothing about dim, nlist or the metric.
    // A realistic nprobe over a 50k index with ~224 lists reaches a few thousand
    // candidates per query, so 2048 is the shape being modelled.
    let limits = GpuLimits::from_client(&client);
    println!("====== IVF reducer, isolated (8192q x 2048 candidates) ======");
    for k in [15usize, 30, 50, 100, 150, 250] {
        let bench = IvfReduceBench::<R> {
            n_queries: 8192,
            max_candidates: 2048,
            k,
            client: client.clone(),
        };

        let radix = RadixIvfReduceBench::<R> {
            n_queries: 8192,
            max_candidates: 2048,
            k,
            client: client.clone(),
        };

        // Both arms are gated before either is timed, and then cross-checked
        // against each other. A radix kernel that silently did nothing would
        // otherwise report as an enormous speedup, which is exactly what a
        // busted `launch_unchecked` looks like.
        let serial_dists = validate_ivf_reduce::<R, _>(&bench, 8192, k, &client);
        let radix_dists = validate_ivf_reduce::<R, _>(&radix, 8192, k, &client);
        assert_eq!(
            serial_dists.len(),
            radix_dists.len(),
            "k={k}: arms returned different shapes"
        );
        for (slot, (m, r)) in serial_dists.iter().zip(radix_dists.iter()).enumerate() {
            // Distances must be bit-exact. Indices are deliberately not compared:
            // the synthetic sawtooth repeats every 1000 values, so ties are
            // everywhere and the two arms may pick different ids for them.
            assert_eq!(
                m.to_bits(),
                r.to_bits(),
                "k={k}: arms disagree at slot {slot}, serial={m} radix={r}"
            );
        }

        println!(
            "\n--- k={k} | radix smem={}B resident={} ---\n",
            radix_select_smem_bytes(k, WORKGROUP_SIZE_X as usize, SURV_ARRAYS_IVF),
            resident_workgroups(
                radix_select_smem_bytes(k, WORKGROUP_SIZE_X as usize, SURV_ARRAYS_IVF),
                &limits
            ),
        );
        println!("{}", bench.name());
        println!("{:?}", bench.run(TimingMethod::System));

        println!("{}", radix.name());
        println!("{:?}", radix.run(TimingMethod::System));
    }
    println!();

    // dim is swept across the `pick_wg_y` tiers rather than staying at 32/64,
    // and cosine is included because it carries the larger shared-memory
    // footprint that table was sized around.
    let db_sizes = vec![
        (50_000usize, 32usize, 224usize, Dist::SquaredEuclidean), // sqrt(50k) ~ 224
        (100_000, 32, 316, Dist::SquaredEuclidean),               // sqrt(100k) ~ 316
        (50_000, 64, 224, Dist::SquaredEuclidean),
        (50_000, 128, 224, Dist::SquaredEuclidean),
        (50_000, 512, 224, Dist::SquaredEuclidean),
        (50_000, 128, 224, Dist::Cosine),
    ];

    for (n, dim, nlist, metric) in db_sizes {
        println!(
            "\n====== IVF: {}n, dim={}, {} lists, {:?} ======",
            n, dim, nlist, metric
        );

        // Time the build, then reuse the same index for the query benchmarks.
        let build_bench = IvfBuildBench::<R> {
            n,
            dim,
            nlist,
            metric,
            device: device.clone(),
        };
        println!("{}", build_bench.name());
        println!("{:?}", build_bench.run(TimingMethod::System));

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = std::sync::Arc::new(
            IvfIndexGpu::<f32, R>::build(
                mat.as_ref(),
                metric,
                Some(nlist),
                default_k_means_params(),
                42,
                false,
                device.clone(),
            )
            .unwrap(),
        );

        let nprobe_values = vec![
            ((nlist as f32).sqrt() as usize).max(1),
            ((nlist as f32).sqrt() as usize * 2).min(nlist),
        ];

        for nprobe in &nprobe_values {
            // Cross-batch query
            let query_bench = IvfQueryBench::<R> {
                index: index.clone(),
                n_queries: 10_000,
                dim,
                k: 15,
                nprobe: *nprobe,
                client: client.clone(),
            };

            // Correctness gate before the timed run.
            let probe = query_bench.prepare();
            let (idx, _) = query_bench.execute(probe).expect("ivf query failed");
            validate_knn(&idx, 15, &query_bench.name());

            println!("{}", query_bench.name());
            println!("{:?}", query_bench.run(TimingMethod::System));

            // Self-query (kNN graph)
            let knn_bench = IvfKnnBench::<R> {
                index: index.clone(),
                k: 15,
                n,
                nprobe: *nprobe,
                client: client.clone(),
            };
            println!("{}", knn_bench.name());
            println!("{:?}", knn_bench.run(TimingMethod::System));
        }
    }
}

fn main() {
    run_ivf_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
