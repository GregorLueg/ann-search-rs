//! GPU kernel microbenchmarks using CubeCL's Benchmark trait.
//!
//! Run with: cargo bench --bench gpu_exhaustive_kernels --features gpu

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
}

impl<R: Runtime> Clone for DistanceInput<R> {
    fn clone(&self) -> Self {
        Self {
            query_gpu: self.query_gpu.clone(),
            db_gpu: self.db_gpu.clone(),
            distances_gpu: self.distances_gpu.clone(),
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

        let queries: Vec<f32> = (0..nq * dim)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
            .collect();
        let db: Vec<f32> = (0..ndb * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();

        let query_gpu = GpuTensor::<R, f32>::from_slice(&queries, vec![nq, dim], &self.client);
        let db_gpu = GpuTensor::<R, f32>::from_slice(&db, vec![ndb, dim], &self.client);
        let distances_gpu = GpuTensor::<R, f32>::empty(vec![nq, ndb], &self.client);

        DistanceInput {
            query_gpu,
            db_gpu,
            distances_gpu,
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;
        let dim_lines = self.cfg.dim / LINE_SIZE as usize;
        let vec_size = LINE_SIZE as usize;

        let grid_x = (ndb as u32).div_ceil(WORKGROUP_SIZE_X);
        let grid_y = (nq as u32).div_ceil(WORKGROUP_SIZE_Y);

        unsafe {
            let _ = euclidean_tiled::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(grid_x, grid_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                input.query_gpu.into_tensor_arg(vec_size),
                input.db_gpu.into_tensor_arg(vec_size),
                input.distances_gpu.into_tensor_arg(1),
                ScalarArg { elem: 0u32 },
                ScalarArg { elem: ndb as u32 },
                ScalarArg { elem: nq as u32 },
                ScalarArg { elem: ndb as u32 },
                dim_lines,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "distance_only_{}q_{}db_{}d",
            self.cfg.n_queries, self.cfg.n_db, self.cfg.dim
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
        let init_gy = (nq as u32).div_ceil(WORKGROUP_SIZE_Y);
        unsafe {
            let _ = init_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(init_gx, init_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                topk_dists.clone().into_tensor_arg(1),
                topk_indices.clone().into_tensor_arg(1),
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
            let _ = extract_topk::launch_unchecked::<f32, R>(
                &self.client,
                CubeCount::Static(extract_grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                input.distances_gpu.into_tensor_arg(1),
                input.topk_dists.into_tensor_arg(1),
                input.topk_indices.into_tensor_arg(1),
                ScalarArg { elem: 0u32 },
                ScalarArg { elem: ndb as u32 },
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
}

impl<R: Runtime> Benchmark for FullPipelineBench<R> {
    type Input = PipelineInput;
    type Output = (Vec<Vec<usize>>, Vec<Vec<f32>>);

    fn prepare(&self) -> Self::Input {
        let dim = self.cfg.dim;
        let nq = self.cfg.n_queries;
        let ndb = self.cfg.n_db;

        PipelineInput {
            queries: (0..nq * dim)
                .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
                .collect(),
            db: (0..ndb * dim)
                .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
                .collect(),
        }
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let qb = BatchData::new(&input.queries, &[], self.cfg.n_queries);
        let dbb = BatchData::new(&input.db, &[], self.cfg.n_db);

        let result = query_batch_gpu::<f32, R>(
            self.cfg.k,
            &qb,
            &dbb,
            self.cfg.dim,
            &Dist::Euclidean,
            self.device.clone(),
            false,
        );

        Ok(result)
    }

    fn name(&self) -> String {
        format!(
            "full_pipeline_{}q_{}db_{}d_k{}",
            self.cfg.n_queries, self.cfg.n_db, self.cfg.dim, self.cfg.k
        )
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("sync failed");
    }
}

// ──────────────────────────────────────────────
// Runner
// ──────────────────────────────────────────────

fn run_suite<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    // ── Kernel-level benchmarks (single chunk sizes) ──

    let kernel_configs = vec![
        BenchConfig {
            n_queries: 8192,
            n_db: 16_384,
            dim: 32,
            k: 15,
        },
        BenchConfig {
            n_queries: 8192,
            n_db: 16_384,
            dim: 64,
            k: 15,
        },
    ];

    println!("====== Kernel-level (single chunk) ======");
    for cfg in kernel_configs {
        println!(
            "\n--- {}q x {}db, dim={}, k={} ---\n",
            cfg.n_queries, cfg.n_db, cfg.dim, cfg.k
        );

        let dist_bench = DistanceBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };
        let topk_bench = TopkBench::<R> {
            cfg: cfg.clone(),
            client: client.clone(),
        };

        println!("{}", dist_bench.name());
        println!("{:?}", dist_bench.run(TimingMethod::System));

        println!("{}", topk_bench.name());
        println!("{:?}", topk_bench.run(TimingMethod::System));
    }

    // ── Pipeline-level benchmarks (realistic workloads) ──

    let pipeline_configs = vec![
        // Self-query patterns (kNN graph generation)
        BenchConfig {
            n_queries: 50_000,
            n_db: 50_000,
            dim: 32,
            k: 15,
        },
        BenchConfig {
            n_queries: 100_000,
            n_db: 100_000,
            dim: 32,
            k: 15,
        },
        BenchConfig {
            n_queries: 150_000,
            n_db: 150_000,
            dim: 32,
            k: 15,
        },
        // Cross-batch pattern
        BenchConfig {
            n_queries: 50_000,
            n_db: 150_000,
            dim: 32,
            k: 15,
        },
        // Higher dim
        BenchConfig {
            n_queries: 50_000,
            n_db: 50_000,
            dim: 64,
            k: 15,
        },
        BenchConfig {
            n_queries: 100_000,
            n_db: 100_000,
            dim: 64,
            k: 15,
        },
    ];

    println!("\n====== Full pipeline (realistic) ======");
    for cfg in pipeline_configs {
        println!(
            "\n--- {}q x {}db, dim={}, k={} ---\n",
            cfg.n_queries, cfg.n_db, cfg.dim, cfg.k
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
    run_suite::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
