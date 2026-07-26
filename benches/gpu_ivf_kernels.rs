//! GPU kernel microbenchmarks using CubeCL's Benchmark trait. This bench covers
//! the IVF-GPU pipeline (build, cross-batch query, self-kNN).
//!
//! Run with: cargo bench --bench gpu_ivf_kernels --features gpu

use ann_search_rs::prelude::KMeansTrainingParams;
use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl::prelude::*;

use ann_search_rs::utils::dist::Dist;

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

/// Print the device ceilings that silently gate `launch_unchecked`.
fn report_device_limits<R: Runtime>(client: &ComputeClient<R>) {
    let props = client.properties();
    let hw = &props.hardware;
    println!("====== Device ======");
    println!("runtime          : {}", R::name(client));
    println!("plane size       : {} .. {}", hw.plane_size_min, hw.plane_size_max);
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
        short, 0,
        "{label}: {short}/{} queries returned fewer than k={k} neighbours",
        indices.len()
    );
}

fn run_ivf_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    report_device_limits::<R>(&client);

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
