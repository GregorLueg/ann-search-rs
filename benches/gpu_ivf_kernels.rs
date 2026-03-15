//! GPU kernel microbenchmarks using CubeCL's Benchmark trait.
//!
//! Run with: cargo bench --bench gpu_exhaustive_kernels --features gpu

#![allow(dead_code)]

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
        let _index = IvfIndexGpu::<f32, R>::build(
            mat.as_ref(),
            self.metric,
            Some(self.nlist),
            Some(10),
            42,
            false,
            self.device.clone(),
        );
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
            .query_batch(mat.as_ref(), self.k, Some(self.nprobe), None, false);
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
            .generate_knn(self.k, Some(self.nprobe), None, true, false);
        Ok(result)
    }

    fn name(&self) -> String {
        format!("ivf_knn_{}n_{}probe_k{}", self.n, self.nprobe, self.k)
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

fn run_ivf_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let db_sizes = vec![
        (50_000usize, 32usize, 224usize), // sqrt(50k) ~ 224
        (100_000, 32, 316),               // sqrt(100k) ~ 316
        (50_000, 64, 224),
    ];

    for (n, dim, nlist) in db_sizes {
        println!("\n====== IVF: {}n, dim={}, {} lists ======", n, dim, nlist);

        // Build the index once, share across query benchmarks
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = std::sync::Arc::new(IvfIndexGpu::<f32, R>::build(
            mat.as_ref(),
            Dist::Euclidean,
            Some(nlist),
            Some(10),
            42,
            false,
            device.clone(),
        ));

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
