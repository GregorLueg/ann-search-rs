//! GPU-accelerated version of k-means clustering. Borrows heavily from the
//! Flash-KMeans approach of Yang et al., arXiv, and ports over what wgpu and
//! CubeCL can express.
//!
//! Re-uses the GPU infrastructure from `cubecl-utils-rs` (tensors, grid
//! decomposition, device limits) and the CPU k-means utilities in
//! [`crate::utils::k_means_utils`] to avoid code duplication: initialisation
//! for the random path, the balancing policy constants, and the whole CPU
//! fallback taken when the runtime has no u32 atomics.
//!
//! ### Mixed precision
//!
//! The data-touching kernels are generic on a storage type `S` and an
//! accumulator type `A`. With `S == A` (the default) the kernel runs at native
//! precision. With `S == half::f16, A == f32` the input data is held on the
//! GPU at half precision but every cast, distance, and reduction happens in
//! fp32, so accumulation drift is bounded by the fp32 path. This is opt-in
//! via `KMeansGpuParams::quantise_to_f16` and requires a wgpu adapter that
//! exposes the `shader-f16` feature.

#![allow(missing_docs)] // complains about cubecl macros...

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use faer::Mat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::iter::Sum;
use std::time::Instant;

use crate::gpu::topk_gpu::atomic_u32_add_supported;
use crate::gpu::{WORKGROUP_128, WORKGROUP_SIZE_X};
use crate::prelude::*;
use crate::utils::k_means_utils::*;

////////////
// Consts //
////////////

/// Points consumed per unrolled step in the segmented centroid reduction.
const SEGMENT_UNROLL: usize = 8;

/// Shared-memory length of [`segmented_centroid_update`], in elements of the
/// accumulator type.
const SEGMENTED_SMEM_LEN: usize = WORKGROUP_128 as usize;

/// Centroids scored per unrolled step in the assignment kernels.
const CENTROID_UNROLL: usize = 8;

/// Oversampling factor for [`kmeans_parallel_init_gpu`]. Each round samples
/// `k * KMEANSPP_OVERSAMPLING` candidates. Matches the upstream CPU
/// `kmeans_parallel_init` so the two produce comparable initialisations.
const KMEANSPP_OVERSAMPLING: usize = 2;

////////////
// Params //
////////////

/// GPU k-means parameters.
///
/// The GPU counterpart of [`KMeansTrainingParams`], with two differences: there
/// is no `LloydPath` because the device has only one path, and `fixed` /
/// `quantise_to_f16` have no CPU equivalent.
///
/// That second difference matters when the runtime has no `u32` atomics and
/// [`k_means_gpu_flat`] falls back to the CPU: `iters`, `init`, `balanced` and
/// `seed` carry across, but `fixed` and `quantise_to_f16` have nothing to map
/// onto and are dropped. A caller asking for a fixed 50 iterations may get
/// fewer, because every CPU Lloyd path stops early on centroid drift.
#[derive(Clone, Copy, Debug)]
pub struct KMeansGpuParams {
    /// Maximum number of Lloyd's iterations.
    pub iters: usize,
    /// Optional initialisation strategy. `None` picks based on `n_centroids`.
    pub init: Option<KMeansInit>,
    /// Fixed number of iterations.
    pub fixed: bool,
    /// Hold the data buffer on the GPU at fp16. Centroids and accumulators
    /// stay at the caller's precision. Halves data-buffer memory and improves
    /// effective bandwidth on the assignment kernels for large `dim`. Requires
    /// `shader-f16` on the wgpu adapter; the caller type `T` should be f32.
    pub quantise_to_f16: bool,
    /// Reseed starved centroids each iteration, RAFT-style.
    pub balanced: bool,
}

impl KMeansGpuParams {
    /// New params.
    ///
    /// Balancing is off. Chain [`KMeansGpuParams::with_balancing`] to turn it
    /// on.
    ///
    /// ### Params
    ///
    /// * `iters` - Number of iterations
    /// * `init` - Optional initialisation
    /// * `fixed` - Shall the algorithm be run for a fixed set of iterations
    ///   or shall convergence be checked.
    /// * `quantise_to_f16` - Quantise the data buffer to fp16 on the GPU
    pub fn new(iters: usize, init: Option<KMeansInit>, fixed: bool, quantise_to_f16: bool) -> Self {
        Self {
            iters,
            init,
            fixed,
            quantise_to_f16,
            balanced: false,
        }
    }

    /// Turn size balancing on or off
    ///
    /// See [`crate::utils::k_means_utils::adjust_centers`] for what balancing
    /// does and what it does not promise.
    ///
    /// ### Params
    ///
    /// * `balanced` - Whether to reseed starved centroids each iteration
    ///
    /// ### Returns
    ///
    /// `self` with the flag set, for chaining off `new` or `default`.
    pub fn with_balancing(mut self, balanced: bool) -> Self {
        self.balanced = balanced;
        self
    }
}

/// Default implementation for [KMeansGpuParams]
impl Default for KMeansGpuParams {
    fn default() -> Self {
        Self::new(50, None, true, false)
    }
}

/////////////
// Kernels //
/////////////

//////////////////
// Flash Assign //
//////////////////

/// One-thread-per-point Euclidean argmin, vectorised, centroids read straight
/// from global memory.
///
/// There is deliberately no shared-memory centroid tile. Every thread in a
/// workgroup reads the same centroid element at the same time, so the value is
/// workgroup-uniform and the hardware already broadcasts it from cache;
/// staging it buys nothing and costs two barriers per tile plus the shared
/// memory that occupancy wants.
///
/// The point is held in registers as `dim_lines` vectors rather than exploded
/// to `dim_lines * LINE_SIZE` scalars. Squared differences accumulate in a
/// vector and are reduced horizontally once per centroid.
///
/// ### Type parameters
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision; all distance arithmetic runs in `A`
/// * `N` - Vectorisation width
///
/// ### Params
///
/// * `data` - Input data vectors `[n, dim]` in storage precision `S`
/// * `centroids` - Centroid matrix `[k, dim]` in accumulator precision `A`
/// * `assignments` - Output hard assignment indices `[n]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of clusters
/// * `dim_lines` - Vectorised dimension (`dim / LINE_SIZE`); comptime
/// * `wg_size` - Workgroup size; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> point
///   index
#[cube(launch_unchecked)]
pub fn flash_assign_euclidean_vec<S: Float, A: Float, N: Size>(
    data: &Tensor<Vector<S, N>>,
    centroids: &Tensor<Vector<A, N>>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] wg_size: u32,
) {
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if point_idx >= n_samples {
        terminate!();
    }
    let p_base = point_idx as usize * dim_lines;

    // Staged in registers as vectors. The cast is a no-op when `S == A`.
    let mut p = Array::<Vector<A, N>>::new(dim_lines);
    for i in 0..dim_lines {
        let sl = data[p_base + i];
        let mut pv = Vector::<A, N>::empty();
        #[unroll]
        for lane in 0..LINE_SIZE {
            pv[lane] = A::cast_from(sl[lane]);
        }
        p[i] = pv;
    }

    let mut best_dist = A::new(f32::MAX);
    let mut best_idx = 0u32;

    let cu = CENTROID_UNROLL as u32;
    let mut accs = Array::<Vector<A, N>>::new(CENTROID_UNROLL);

    let mut c = 0u32;
    while c + cu <= k {
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            accs[u] = Vector::<A, N>::new(A::new(0.0));
        }
        for i in 0..dim_lines {
            let pv = p[i];
            #[unroll]
            for u in 0..CENTROID_UNROLL {
                let diff = pv - centroids[(c as usize + u) * dim_lines + i];
                accs[u] += diff * diff;
            }
        }
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            let av = accs[u];
            let mut sum = A::new(0.0);
            #[unroll]
            for lane in 0..LINE_SIZE {
                sum += av[lane];
            }
            if sum < best_dist {
                best_dist = sum;
                best_idx = c + u as u32;
            }
        }
        c += cu;
    }

    // Tail for `k` not divisible by CENTROID_UNROLL.
    while c < k {
        let cbase = c as usize * dim_lines;
        let mut acc = Vector::<A, N>::new(A::new(0.0));
        for i in 0..dim_lines {
            let diff = p[i] - centroids[cbase + i];
            acc += diff * diff;
        }
        let mut sum = A::new(0.0);
        #[unroll]
        for lane in 0..LINE_SIZE {
            sum += acc[lane];
        }
        if sum < best_dist {
            best_dist = sum;
            best_idx = c;
        }
        c += 1u32;
    }

    assignments[point_idx as usize] = best_idx;
}

/// Cosine analogue of [`fn@flash_assign_euclidean_vec`]. Uses precomputed L2
/// norms; minimises `1 - dot(x, c) / (||x|| * ||c||)`.
///
/// ### Type parameters
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision; all distance arithmetic runs in `A`
/// * `N` - Vectorisation width
///
/// ### Params
///
/// * `data` - Input data vectors `[n, dim]` in storage precision `S`
/// * `centroids` - Centroid matrix `[k, dim]` in accumulator precision `A`
/// * `point_norms` - Precomputed L2 norms of each data point `[n]`
/// * `centroid_norms` - Precomputed L2 norms of each centroid `[k]`
/// * `assignments` - Output hard assignment indices `[n]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of clusters
/// * `dim_lines` - Vectorised dimension (`dim / LINE_SIZE`); comptime
/// * `wg_size` - Workgroup size; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> point
///   index
#[cube(launch_unchecked)]
pub fn flash_assign_cosine_vec<S: Float, A: Float, N: Size>(
    data: &Tensor<Vector<S, N>>,
    centroids: &Tensor<Vector<A, N>>,
    point_norms: &Tensor<A>,
    centroid_norms: &Tensor<A>,
    assignments: &mut Tensor<u32>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] wg_size: u32,
) {
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if point_idx >= n_samples {
        terminate!();
    }
    let p_base = point_idx as usize * dim_lines;

    let mut p = Array::<Vector<A, N>>::new(dim_lines);
    for i in 0..dim_lines {
        let sl = data[p_base + i];
        let mut pv = Vector::<A, N>::empty();
        #[unroll]
        for lane in 0..LINE_SIZE {
            pv[lane] = A::cast_from(sl[lane]);
        }
        p[i] = pv;
    }
    let pnorm = point_norms[point_idx as usize];

    let mut best_dist = A::new(f32::MAX);
    let mut best_idx = 0u32;

    let cu = CENTROID_UNROLL as u32;
    let mut accs = Array::<Vector<A, N>>::new(CENTROID_UNROLL);

    let mut c = 0u32;
    while c + cu <= k {
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            accs[u] = Vector::<A, N>::new(A::new(0.0));
        }
        for i in 0..dim_lines {
            let pv = p[i];
            #[unroll]
            for u in 0..CENTROID_UNROLL {
                accs[u] += pv * centroids[(c as usize + u) * dim_lines + i];
            }
        }
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            let av = accs[u];
            let mut dot = A::new(0.0);
            #[unroll]
            for lane in 0..LINE_SIZE {
                dot += av[lane];
            }
            let dist = A::new(1.0) - dot / (pnorm * centroid_norms[c as usize + u]);
            if dist < best_dist {
                best_dist = dist;
                best_idx = c + u as u32;
            }
        }
        c += cu;
    }

    // Tail for `k` not divisible by CENTROID_UNROLL.
    while c < k {
        let cbase = c as usize * dim_lines;
        let mut acc = Vector::<A, N>::new(A::new(0.0));
        for i in 0..dim_lines {
            acc += p[i] * centroids[cbase + i];
        }
        let mut dot = A::new(0.0);
        #[unroll]
        for lane in 0..LINE_SIZE {
            dot += acc[lane];
        }
        let dist = A::new(1.0) - dot / (pnorm * centroid_norms[c as usize]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
        c += 1u32;
    }

    assignments[point_idx as usize] = best_idx;
}

/// Dispatch the appropriate assignment kernel and write results into a
/// device-resident buffer. Selects between [`fn@flash_assign_euclidean_vec`]
/// and [`fn@flash_assign_cosine_vec`] based on `metric`.
///
/// ### Type params
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision for distance arithmetic
/// * `R` - CubeCL runtime
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Input data `[n, dim]` in storage precision `S`
/// * `cent_gpu` - Centroid matrix `[k, dim]` in accumulator precision `A`
/// * `pnorm_gpu` - Precomputed point L2 norms `[n]`; only used for cosine
/// * `cnorm_gpu` - Precomputed centroid L2 norms `[k]`; only used for cosine
/// * `assign_gpu` - Output assignment indices `[n]`
/// * `n` - Number of data points
/// * `k` - Number of clusters
/// * `dim` - Vector dimension, **padded** to a multiple of `LINE_SIZE`; it is
///   divided by the line size to get the vectorised loop bound
/// * `metric` - Distance metric; `Manhattan` is not supported
///
/// ### Returns
///
/// `Ok(())`; assignments are written into `assign_gpu` in place. `CubeclUtils`
/// if the grid is over the device limit.
#[allow(clippy::too_many_arguments)]
fn flash_assign_device<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    cent_gpu: &GpuTensor<R, A>,
    pnorm_gpu: &GpuTensor<R, A>,
    cnorm_gpu: &GpuTensor<R, A>,
    assign_gpu: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
) -> Result<(), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    let n_workgroups = (n as u32).div_ceil(WORKGROUP_128);
    let (gx, gy) = grid_2d(n_workgroups, &limits)?;
    let count = CubeCount::Static(gx, gy, 1);
    let cdim = CubeDim::new_1d(WORKGROUP_128);

    match *metric {
        Dist::SquaredEuclidean => unsafe {
            flash_assign_euclidean_vec::launch_unchecked::<S, A, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cent_gpu.clone().into_tensor_arg(),
                assign_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
                WORKGROUP_128,
            );
        },
        Dist::Cosine => unsafe {
            flash_assign_cosine_vec::launch_unchecked::<S, A, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cent_gpu.clone().into_tensor_arg(),
                pnorm_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                assign_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
                WORKGROUP_128,
            );
        },
        Dist::Manhattan => unreachable!("Manhattan distance is not supported!"),
    }

    Ok(())
}

/////////////////////
// k-means || init //
/////////////////////

/// Squared Euclidean distance from each point to its nearest candidate.
///
/// Same traversal as [`fn@flash_assign_euclidean_vec`], but it keeps the
/// distance rather than the index. This is the D² pass of k-means||, which
/// dominates that algorithm: it runs `ln(k) + 1` times against a candidate set
/// growing by `2k` each round, so at n = 1e6, k = 100 it is n * dim * 2005
/// fused multiply-adds in total.
///
/// ### Type parameters
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision
/// * `N` - Vectorisation width
///
/// ### Params
///
/// * `data` - Input data vectors `[n, dim]` in storage precision `S`
/// * `cands` - Candidate centres `[k, dim]` in accumulator precision `A`
/// * `out` - Output nearest-candidate distances `[n]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of candidates
/// * `dim_lines` - Vectorised dimension (`dim / LINE_SIZE`); comptime
/// * `wg_size` - Workgroup size; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> point
///   index
#[cube(launch_unchecked)]
pub fn min_dist_euclidean_vec<S: Float, A: Float, N: Size>(
    data: &Tensor<Vector<S, N>>,
    cands: &Tensor<Vector<A, N>>,
    out: &mut Tensor<A>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] wg_size: u32,
) {
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if point_idx >= n_samples {
        terminate!();
    }
    let p_base = point_idx as usize * dim_lines;

    let mut p = Array::<Vector<A, N>>::new(dim_lines);
    for i in 0..dim_lines {
        let sl = data[p_base + i];
        let mut pv = Vector::<A, N>::empty();
        #[unroll]
        for lane in 0..LINE_SIZE {
            pv[lane] = A::cast_from(sl[lane]);
        }
        p[i] = pv;
    }

    let mut best = A::new(f32::MAX);

    let cu = CENTROID_UNROLL as u32;
    let mut accs = Array::<Vector<A, N>>::new(CENTROID_UNROLL);

    let mut c = 0u32;
    while c + cu <= k {
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            accs[u] = Vector::<A, N>::new(A::new(0.0));
        }
        for i in 0..dim_lines {
            let pv = p[i];
            #[unroll]
            for u in 0..CENTROID_UNROLL {
                let diff = pv - cands[(c as usize + u) * dim_lines + i];
                accs[u] += diff * diff;
            }
        }
        #[unroll]
        for u in 0..CENTROID_UNROLL {
            let av = accs[u];
            let mut sum = A::new(0.0);
            #[unroll]
            for lane in 0..LINE_SIZE {
                sum += av[lane];
            }
            if sum < best {
                best = sum;
            }
        }
        c += cu;
    }

    while c < k {
        let cbase = c as usize * dim_lines;
        let mut acc = Vector::<A, N>::new(A::new(0.0));
        for i in 0..dim_lines {
            let diff = p[i] - cands[cbase + i];
            acc += diff * diff;
        }
        let mut sum = A::new(0.0);
        #[unroll]
        for lane in 0..LINE_SIZE {
            sum += acc[lane];
        }
        if sum < best {
            best = sum;
        }
        c += 1u32;
    }

    out[point_idx as usize] = best;
}

/// Cosine analogue of [`fn@min_dist_euclidean_vec`].
///
/// ### Type parameters
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision
/// * `N` - Vectorisation width
///
/// ### Params
///
/// * `data` - Input data vectors `[n, dim]` in storage precision `S`
/// * `cands` - Candidate centres `[k, dim]` in accumulator precision `A`
/// * `point_norms` - Precomputed L2 norms of each data point `[n]`
/// * `cand_norms` - Precomputed L2 norms of each candidate `[k]`
/// * `out` - Output nearest-candidate distances `[n]`
/// * `n_samples` - Total number of data points
/// * `k` - Number of candidates
/// * `dim_lines` - Vectorised dimension (`dim / LINE_SIZE`); comptime
/// * `wg_size` - Workgroup size; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> point
///   index
#[cube(launch_unchecked)]
pub fn min_dist_cosine_vec<S: Float, A: Float, N: Size>(
    data: &Tensor<Vector<S, N>>,
    cands: &Tensor<Vector<A, N>>,
    point_norms: &Tensor<A>,
    cand_norms: &Tensor<A>,
    out: &mut Tensor<A>,
    n_samples: u32,
    k: u32,
    #[comptime] dim_lines: usize,
    #[comptime] wg_size: u32,
) {
    let point_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if point_idx >= n_samples {
        terminate!();
    }
    let p_base = point_idx as usize * dim_lines;

    let mut p = Array::<Vector<A, N>>::new(dim_lines);
    for i in 0..dim_lines {
        let sl = data[p_base + i];
        let mut pv = Vector::<A, N>::empty();
        #[unroll]
        for lane in 0..LINE_SIZE {
            pv[lane] = A::cast_from(sl[lane]);
        }
        p[i] = pv;
    }
    let pnorm = point_norms[point_idx as usize];

    let mut best = A::new(f32::MAX);

    let mut c = 0u32;
    while c < k {
        let cbase = c as usize * dim_lines;
        let mut acc = Vector::<A, N>::new(A::new(0.0));
        for i in 0..dim_lines {
            acc += p[i] * cands[cbase + i];
        }
        let mut dot = A::new(0.0);
        #[unroll]
        for lane in 0..LINE_SIZE {
            dot += acc[lane];
        }
        let dist = A::new(1.0) - dot / (pnorm * cand_norms[c as usize]);
        if dist < best {
            best = dist;
        }
        c += 1u32;
    }

    out[point_idx as usize] = best;
}

/// Dispatch the appropriate nearest-candidate distance kernel.
///
/// ### Type params
///
/// * `S` - Storage precision of the data buffer
/// * `A` - Accumulator precision
/// * `R` - CubeCL runtime
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Input data `[n, dim]`
/// * `cand_gpu` - Candidate centres `[k, dim]`
/// * `pnorm_gpu` - Point L2 norms `[n]`; only used for cosine
/// * `cnorm_gpu` - Candidate L2 norms `[k]`; only used for cosine
/// * `out_gpu` - Output distances `[n]`
/// * `n` - Number of data points
/// * `k` - Number of candidates
/// * `dim` - Padded vector dimension
/// * `metric` - Distance metric; `Manhattan` is not supported
///
/// ### Returns
///
/// `Ok(())`; distances are written into `out_gpu` in place. `CubeclUtils` if
/// the grid is over the device limit.
#[allow(clippy::too_many_arguments)]
fn min_dist_device<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    cand_gpu: &GpuTensor<R, A>,
    pnorm_gpu: &GpuTensor<R, A>,
    cnorm_gpu: &GpuTensor<R, A>,
    out_gpu: &GpuTensor<R, A>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
) -> Result<(), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_128), &limits)?;
    let count = CubeCount::Static(gx, gy, 1);
    let cdim = CubeDim::new_1d(WORKGROUP_128);

    match *metric {
        Dist::SquaredEuclidean => unsafe {
            min_dist_euclidean_vec::launch_unchecked::<S, A, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cand_gpu.clone().into_tensor_arg(),
                out_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
                WORKGROUP_128,
            );
        },
        Dist::Cosine => unsafe {
            min_dist_cosine_vec::launch_unchecked::<S, A, R>(
                client,
                count,
                cdim,
                vec_size,
                data_gpu.clone().into_tensor_arg(),
                cand_gpu.clone().into_tensor_arg(),
                pnorm_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                out_gpu.clone().into_tensor_arg(),
                n as u32,
                k as u32,
                dim_lines,
                WORKGROUP_128,
            );
        },
        Dist::Manhattan => unreachable!("Manhattan distance is not supported!"),
    }

    Ok(())
}

/// Sample `count` indices proportional to `weights` via a prefix sum and
/// binary search.
///
/// The CPU [`kmeans_parallel_init`] rescans the whole weight vector per
/// sample, which is `O(count * n)`; at n = 1e6 and k = 200 that is 2e8
/// sequential steps per round. One prefix sum plus a binary search per sample
/// is `O(n + count * log n)`.
///
/// ### Params
///
/// * `weights` - Non-negative sampling weights `[n]`
/// * `count` - Number of indices to draw
/// * `rng` - Random source
///
/// ### Returns
///
/// `count` indices, with replacement. Empty if the weights sum to zero, which
/// means every point coincides with an existing candidate.
fn sample_proportional<T>(weights: &[T], count: usize, rng: &mut StdRng) -> Vec<usize>
where
    T: num_traits::Float,
{
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut running = 0.0f64;
    for &w in weights {
        running += w.to_f64().unwrap_or(0.0).max(0.0);
        cumulative.push(running);
    }
    if running <= 0.0 {
        return Vec::new();
    }

    (0..count)
        .map(|_| {
            let threshold = rng.random::<f64>() * running;
            cumulative
                .partition_point(|&c| c < threshold)
                .min(weights.len() - 1)
        })
        .collect()
}

/// k-means++ over a small candidate set, on the host.
///
/// Final stage of k-means||. The candidate set is `O(k log k)` rows, so this
/// is cheap next to the D² passes and stays on the CPU.
///
/// ### Params
///
/// * `cands` - Candidate centres `[n_cand, dim]`, flattened row-major
/// * `cand_norms` - L2 norms of the candidates `[n_cand]`
/// * `dim` - Padded vector dimension
/// * `k` - Target centroid count
/// * `metric` - Distance metric
/// * `rng` - Random source
///
/// ### Returns
///
/// Exactly `k * dim` centroid values.
///
/// ### Note
///
/// Returning fewer than `k` is not an option even when the candidate set is
/// smaller: every caller sizes a `[k, dim]` tensor from this. Degenerate data
/// reaches that branch for real -- under cosine, a dataset whose rows are all
/// the same direction has every D^2 weight at zero, so `sample_proportional`
/// draws nothing and the candidate set never grows past the first point. Any
/// choice of centroid is as good as any other there, so the shortfall is
/// topped up by cycling the candidates.
fn kmeans_pp_on_candidates<T>(
    cands: &[T],
    cand_norms: &[T],
    dim: usize,
    k: usize,
    metric: &Dist,
    rng: &mut StdRng,
) -> Vec<T>
where
    T: num_traits::Float + AnnSearchFloat,
{
    let n_cand = cands.len() / dim;
    if n_cand == 0 {
        return vec![T::zero(); k * dim];
    }
    if n_cand <= k {
        let mut centroids = cands[..n_cand * dim].to_vec();
        for i in n_cand..k {
            let src = (i % n_cand) * dim;
            centroids.extend_from_slice(&cands[src..src + dim]);
        }
        return centroids;
    }

    let mut centroids = Vec::with_capacity(k * dim);
    let first = rng.random_range(0..n_cand);
    centroids.extend_from_slice(&cands[first * dim..(first + 1) * dim]);
    let mut cent_norms = vec![cand_norms[first]];

    let mut distances = vec![T::infinity(); n_cand];

    for _ in 1..k {
        let latest = &centroids[centroids.len() - dim..];
        let latest_norm = *cent_norms
            .last()
            .expect("at least one centroid by construction");

        for (i, dist) in distances.iter_mut().enumerate() {
            let v = &cands[i * dim..(i + 1) * dim];
            let d = match metric {
                Dist::Cosine => {
                    let denom = cand_norms[i] * latest_norm;
                    if denom > T::zero() {
                        T::one() - T::dot_simd(v, latest) / denom
                    } else {
                        T::one()
                    }
                }
                // `euclidean_simd` already returns the squared distance, so
                // this is the D^2 the k-means++ weighting wants. Squaring it
                // again skews the sampling toward far candidates.
                _ => T::euclidean_simd(v, latest),
            };
            if d < *dist {
                *dist = d;
            }
        }

        let picked = sample_proportional(&distances, 1, rng);
        let idx = match picked.first() {
            Some(&i) => i,
            // All candidates coincide with a chosen centroid; nothing left to
            // separate, so fall back to a uniform draw.
            None => rng.random_range(0..n_cand),
        };
        centroids.extend_from_slice(&cands[idx * dim..(idx + 1) * dim]);
        cent_norms.push(cand_norms[idx]);
    }

    centroids
}

/// GPU-accelerated k-means|| initialisation.
///
/// Mirrors the CPU [`kmeans_parallel_init`], but runs the D² pass on the
/// device against data already resident there. That pass is the whole
/// cost of the algorithm: `ln(k) + 1` rounds against a candidate set growing
/// by `2k` per round.
///
/// Only the distances come back to the host each round (`n` floats), where the
/// weighted sample and the final k-means++ over the candidates are cheap.
///
/// ### Type parameters
///
/// * `S` - Storage precision of the device data buffer
/// * `A` - Accumulator and centroid precision
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Data already on device `[n, dim]` in `S`
/// * `data_host` - The same data on the host `[n, dim]` in `A`, used to gather
///   sampled candidate rows
/// * `norms_host` - L2 norms of every point `[n]`; only read for cosine
/// * `pnorm_gpu` - Point L2 norms on device `[n]`; only read for cosine
/// * `n` - Number of data points
/// * `dim` - Padded vector dimension
/// * `k` - Target centroid count
/// * `metric` - Distance metric
/// * `seed` - Random seed
///
/// ### Returns
///
/// `k * dim` initial centroid values.
///
/// ### Errors
///
/// * `CubeclUtils` if the D² dispatch grid is over the device limit, a
///   candidate upload busts the per-binding size limit, or the distance
///   read-back fails.
#[allow(clippy::too_many_arguments)]
fn kmeans_parallel_init_gpu<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    data_host: &[A],
    norms_host: &[A],
    pnorm_gpu: &GpuTensor<R, A>,
    n: usize,
    dim: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Result<Vec<A>, AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement + num_traits::Float + AnnSearchFloat,
{
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let n_rounds = ((k as f64).ln() + 1.0) as usize;

    let first = rng.random_range(0..n);
    let mut cands: Vec<A> = data_host[first * dim..(first + 1) * dim].to_vec();
    let mut cand_norms: Vec<A> = vec![norms_host[first]];

    let dist_gpu = GpuTensor::<R, A>::empty(vec![n], client)?;

    for _ in 0..n_rounds {
        let n_cand = cand_norms.len();
        let cand_gpu = GpuTensor::<R, A>::from_slice(&cands, vec![n_cand, dim], client)?;
        let cnorm_gpu = GpuTensor::<R, A>::from_slice(&cand_norms, vec![n_cand], client)?;

        min_dist_device::<S, A, R>(
            client, data_gpu, &cand_gpu, pnorm_gpu, &cnorm_gpu, &dist_gpu, n, n_cand, dim, metric,
        )?;

        let distances = dist_gpu.clone().read(client)?;
        let picked = sample_proportional(&distances, k * KMEANSPP_OVERSAMPLING, &mut rng);
        if picked.is_empty() {
            break;
        }
        for idx in picked {
            cands.extend_from_slice(&data_host[idx * dim..(idx + 1) * dim]);
            cand_norms.push(norms_host[idx]);
        }
    }

    Ok(kmeans_pp_on_candidates(
        &cands,
        &cand_norms,
        dim,
        k,
        metric,
        &mut rng,
    ))
}

//////////////////////////
// Privatised histogram //
//////////////////////////

/// Privatised histogram of cluster sizes.
///
/// Each workgroup maintains its own row in `privatised_counts [cube_count, k]`,
/// first zeroing it cooperatively and then counting the points from its stripe.
/// This avoids inter-workgroup atomic contention on a shared row. The
/// privatised rows are reduced by a subsequent `scan_columns_and_sum` pass.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`
/// * `privatised_counts` - Atomic per-workgroup cluster counters
///   `[cube_count * k]`, zeroed and filled in place
/// * `n` - Total number of data points
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> workgroup (row) index
#[cube(launch_unchecked)]
pub fn histogram_clusters_privatised(
    assignments: &Tensor<u32>,
    privatised_counts: &mut Tensor<Atomic<u32>>,
    n: u32,
    k: u32,
) {
    let r = CUBE_POS_X;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_SIZE_X;

    // 1. Cooperative inline zeroing: threads clear their chunk of the row.
    let mut c = tx;
    while c < k {
        let idx = r * k + c;
        Atomic::fetch_and(&privatised_counts[idx as usize], 0u32);
        c += wg;
    }
    sync_cube();

    // 2. Standard privatised counting pass.
    let total_threads = wg * CUBE_COUNT_X;
    let mut i = r * wg + tx;
    while i < n {
        let chunk_c = assignments[i as usize];
        let idx = r * k + chunk_c;
        Atomic::fetch_add(&privatised_counts[idx as usize], 1u32);
        i += total_threads;
    }
}

/// Column-wise prefix scan of the privatised count matrix.
///
/// For each cluster column `c`, serialises over all workgroup rows: replaces
/// each `privatised_counts[r, c]` with the running exclusive prefix sum and
/// writes the column total into `counts[c]`. This transforms the privatised
/// rows into per-workgroup exclusive write offsets for use by
/// `scatter_csr_privatised`.
///
/// ### Params
///
/// * `privatised_counts` - Privatised count matrix `[cube_count * k]`;
///   overwritten in place with exclusive row-prefix sums
/// * `counts` - Output total cluster sizes `[k]`
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroup rows
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> cluster column index
#[cube(launch_unchecked)]
pub fn scan_columns_and_sum(
    privatised_counts: &mut Tensor<u32>,
    counts: &mut Tensor<u32>,
    k: u32,
    cube_count: u32,
) {
    let c = ABSOLUTE_POS_X;
    if c >= k {
        terminate!();
    }

    let mut acc = 0u32;
    let mut r = 0u32;
    while r < cube_count {
        let idx = r * k + c;
        let val = privatised_counts[idx as usize];
        privatised_counts[idx as usize] = acc;
        acc += val;
        r += 1u32;
    }
    counts[c as usize] = acc;
}

/// Exclusive prefix sum of cluster counts into a `k + 1` offset array.
///
/// Single-thread serial scan. Writes global segment start positions;
/// `offsets[k]` equals the total number of points.
///
/// ### Params
///
/// * `counts` - Cluster sizes `[k]`
/// * `offsets` - Output exclusive prefix sums `[k + 1]`
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * Single cube, single thread (`UNIT_POS_X == 0`)
#[cube(launch_unchecked)]
pub fn exclusive_scan_offsets_2(counts: &Tensor<u32>, offsets: &mut Tensor<u32>, k: u32) {
    if UNIT_POS_X == 0u32 {
        offsets[0] = 0u32;
        let mut acc = 0u32;
        let mut c = 0u32;
        while c < k {
            acc += counts[c as usize];
            offsets[(c + 1u32) as usize] = acc;
            c += 1u32;
        }
    }
}

/// Add global segment offsets into the privatised prefix-sum rows.
///
/// Each element `privatised_counts[r * k + c]` holds a workgroup-local
/// exclusive offset after `scan_columns_and_sum`; adding `offsets[c]`
/// converts it to an absolute write position into the CSR index array, ready
/// for `scatter_csr_privatised`.
///
/// ### Params
///
/// * `privatised_counts` - Privatised prefix-sum matrix `[cube_count * k]`;
///   updated in place to global write cursors
/// * `offsets` - Global exclusive prefix sums `[k + 1]` from
///   `exclusive_scan_offsets_2`
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroup rows
#[cube(launch_unchecked)]
pub fn merge_offsets_to_cursors(
    privatised_counts: &mut Tensor<u32>,
    offsets: &Tensor<u32>,
    k: u32,
    cube_count: u32,
) {
    let idx = ABSOLUTE_POS_X as usize;
    let total_elements = (cube_count * k) as usize;
    if idx < total_elements {
        let c = (idx as u32) % k;
        privatised_counts[idx] += offsets[c as usize];
    }
}

/// Contention-free scatter of point indices into CSR order.
///
/// Each workgroup handles the same stripe of points it counted in
/// `histogram_clusters_privatised`, atomically advancing its own row's cursor
/// for each point. Cursors from different workgroups target non-overlapping
/// slots, so atomic contention is negligible compared with a naive single-row
/// approach. Within-segment order is non-deterministic, which is acceptable
/// because the downstream centroid summation is order-independent.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]`
/// * `privatised_cursors` - Atomic per-workgroup write cursors
///   `[cube_count * k]`, seeded by `merge_offsets_to_cursors` and atomically
///   advanced in place
/// * `all_indices` - Output point indices in CSR order `[n]`
/// * `n` - Total number of data points
/// * `k` - Number of clusters
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> workgroup (row) index
#[cube(launch_unchecked)]
pub fn scatter_csr_privatised(
    assignments: &Tensor<u32>,
    privatised_cursors: &mut Tensor<Atomic<u32>>,
    all_indices: &mut Tensor<u32>,
    n: u32,
    k: u32,
) {
    let r = CUBE_POS_X;
    let tx = UNIT_POS_X;
    let wg = WORKGROUP_SIZE_X;

    let total_threads = wg * CUBE_COUNT_X;
    let mut i = r * wg + tx;

    while i < n {
        let c = assignments[i as usize];
        let idx = r * k + c;
        let pos = Atomic::fetch_add(&privatised_cursors[idx as usize], 1u32);
        all_indices[pos as usize] = i;
        i += total_threads;
    }
}

/// Build a CSR layout from hard assignments using the privatised histogram
/// pipeline.
///
/// Runs five kernels in sequence: privatised histogram, column scan, global
/// prefix scan, cursor merging, and contention-free scatter. No host readback
/// occurs between stages. Pre-allocated scratch buffers are passed in so that
/// the driver loop can reuse them across iterations without extra allocation.
///
/// ### Params
///
/// * `assignments` - Hard assignment indices `[n]` already on device
/// * `n` - Number of data points
/// * `k` - Number of clusters
/// * `cube_count` - Number of workgroups used for the privatised histogram
/// * `privatised_counts` - Pre-allocated scratch buffer `[cube_count * k]`
/// * `counts` - Pre-allocated cluster size buffer `[k]`
/// * `offsets` - Pre-allocated offset buffer `[k + 1]`
/// * `all_indices` - Pre-allocated output index buffer `[n]`; cluster `c`
///   occupies `all_indices[offsets[c]..offsets[c+1]]` on exit
/// * `client` - CubeCL compute client for the target device
///
/// ### Returns
///
/// `Ok(())`; the CSR layout lands in `offsets` and `all_indices`.
/// `CubeclUtils` if any of the five dispatches is over the device's cube-count
/// limit. The merge pass is the one that reaches it first: its grid is
/// `cube_count * k / 256`, which grows in both terms.
#[allow(clippy::too_many_arguments)]
pub fn build_csr_gpu_privatised<R>(
    assignments: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    cube_count: usize,
    privatised_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
    client: &ComputeClient<R>,
) -> Result<(), AnnSearchErrors>
where
    R: Runtime,
{
    let limits = GpuLimits::from_client(client);
    let total_elements = cube_count * k;

    let hist_count = checked_cube_count(
        "histogram_clusters_privatised",
        cube_count as u32,
        1,
        1,
        &limits,
    )?;
    let scan_count = checked_cube_count(
        "scan_columns_and_sum",
        k.div_ceil(256) as u32,
        1,
        1,
        &limits,
    )?;
    let merge_count = checked_cube_count(
        "merge_offsets_to_cursors",
        total_elements.div_ceil(256) as u32,
        1,
        1,
        &limits,
    )?;
    let scatter_count =
        checked_cube_count("scatter_csr_privatised", cube_count as u32, 1, 1, &limits)?;

    unsafe {
        histogram_clusters_privatised::launch_unchecked::<R>(
            client,
            hist_count,
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatised_counts.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );

        scan_columns_and_sum::launch_unchecked::<R>(
            client,
            scan_count,
            CubeDim::new_1d(256),
            privatised_counts.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            k as u32,
            cube_count as u32,
        );

        exclusive_scan_offsets_2::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            k as u32,
        );

        merge_offsets_to_cursors::launch_unchecked::<R>(
            client,
            merge_count,
            CubeDim::new_1d(256),
            privatised_counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            k as u32,
            cube_count as u32,
        );

        scatter_csr_privatised::launch_unchecked::<R>(
            client,
            scatter_count,
            CubeDim::new_1d(WORKGROUP_SIZE_X),
            assignments.clone().into_tensor_arg(),
            privatised_counts.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            n as u32,
            k as u32,
        );
    }

    Ok(())
}

/// Recompute centroids as the mean of assigned points via segmented reduction
/// over the CSR layout.
///
/// One workgroup per cluster. The workgroup is split into `n_sub` sub-stripes
/// of `dim_eff` threads: thread `tx` owns output dimensions
/// `tx % dim_eff, tx % dim_eff + dim_eff, ...` and, within each, the points
/// `tx / dim_eff, tx / dim_eff + n_sub, ...` of the cluster's segment. The
/// sub-stripes are reduced through shared memory at the end of every
/// dimension. With one workgroup per cluster there are only `k` of them, so
/// the sub-stripes are where the parallelism has to come from for a
/// low-dimensional dataset.
///
/// Data is loaded from storage precision `S` and cast to accumulator
/// precision `A` element-by-element; the running sum and the divide-by-count
/// both happen in `A`, so accumulation drift is bounded by the `A` path even
/// when `S` is fp16. Atomic-free.
///
/// ### Type parameters
///
/// * `S` - Storage type of the data buffer
/// * `A` - Accumulator and centroid type
///
/// ### Params
///
/// * `data` - Flattened data points `[n, dim]` in `S`
/// * `all_indices` - Point indices in CSR order `[n]` from
///   `build_csr_gpu_privatised`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from
///   `build_csr_gpu_privatised`; cluster `c` occupies
///   `all_indices[offsets[c]..offsets[c+1]]`
/// * `centroids` - Centroid vectors `[k, dim]` in `A`, updated in place;
///   empty clusters retain their prior value
/// * `k` - Number of clusters
/// * `dim` - Embedding dimensionality (comptime)
/// * `dim_eff` - Dimensions covered per sub-stripe, `min(dim, WORKGROUP_128)`;
///   comptime
/// * `n_sub` - Sub-stripes per workgroup, `WORKGROUP_128 / dim_eff`; comptime.
///   The division truncates, so `n_sub * dim_eff` can be under the workgroup
///   size and the surplus threads store into `SEGMENTED_SMEM_LEN` slots that
///   are never read back
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> cluster index
/// * `UNIT_POS_X % dim_eff` -> dimension stride offset within the centroid row
/// * `UNIT_POS_X / dim_eff` -> point sub-stripe within the cluster's segment
#[cube(launch_unchecked)]
pub fn segmented_centroid_update<S: Float, A: Float>(
    data: &Tensor<S>,
    all_indices: &Tensor<u32>,
    offsets: &Tensor<u32>,
    centroids: &mut Tensor<A>,
    k: u32,
    #[comptime] dim: usize,
    #[comptime] dim_eff: usize,
    #[comptime] n_sub: usize,
) {
    let cluster = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if cluster >= k {
        terminate!();
    }

    let tx = UNIT_POS_X as usize;

    let seg_start = offsets[cluster as usize];
    let seg_end = offsets[(cluster + 1u32) as usize];
    let count = seg_end - seg_start;

    if count == 0u32 {
        terminate!();
    }

    let inv_count = A::new(1.0) / A::cast_from(count);
    let cent_base = cluster as usize * dim;

    // Thread `tx` owns output dimension `e0` and point sub-stripe `sub`. The
    // sub-stripes are what make this kernel wide: with one workgroup per
    // cluster there are only `k` of them, so the parallelism has to come from
    // inside the workgroup.
    let e0 = tx % dim_eff;
    let sub = (tx / dim_eff) as u32;
    let subs = n_sub as u32;

    let mut s_part = SharedMemory::<A>::new(SEGMENTED_SMEM_LEN);

    let stride = (n_sub * SEGMENT_UNROLL) as u32;

    let mut e = e0;
    while e < dim {
        let mut acc = A::new(0.0);
        let mut p = sub;

        // Unrolled body: every index load is hoisted above the data load that
        // consumes it, so `SEGMENT_UNROLL` of each are in flight rather than
        // one. This kernel is latency bound, and outstanding loads per thread
        // is the lever that moves it.
        let mut idx = Array::<u32>::new(SEGMENT_UNROLL);
        while p + stride <= count {
            #[unroll]
            for u in 0..SEGMENT_UNROLL {
                idx[u] = all_indices[(seg_start + p + (u as u32) * subs) as usize];
            }
            #[unroll]
            for u in 0..SEGMENT_UNROLL {
                acc += A::cast_from(data[idx[u] as usize * dim + e]);
            }
            p += stride;
        }

        // Tail.
        while p < count {
            let global = all_indices[(seg_start + p) as usize];
            acc += A::cast_from(data[global as usize * dim + e]);
            p += subs;
        }

        s_part[sub as usize * dim_eff + e0] = acc;
        if comptime!(n_sub > 1) {
            sync_cube();
        }
        if sub == 0u32 {
            let mut total = A::new(0.0);
            #[unroll]
            for s in 0..n_sub {
                total += s_part[s * dim_eff + e0];
            }
            centroids[cent_base + e] = total * inv_count;
        }
        if comptime!(n_sub > 1) {
            sync_cube();
        }

        e += dim_eff;
    }
}

/// Recompute centroids in place from a CSR layout.
///
/// Wraps [`fn@segmented_centroid_update`] and derives its `dim_eff` / `n_sub`
/// split from `dim` and the workgroup width. Empty clusters retain their
/// current value.
///
/// ### Type parameters
///
/// * `S` - Storage type of the data buffer
/// * `A` - Accumulator and centroid type
///
/// ### Params
///
/// * `data` - Data points already on device `[n, dim]` in `S`
/// * `all_indices` - Point indices in CSR order `[n]` from
///   `build_csr_gpu_privatised`
/// * `offsets` - Exclusive prefix sums `[k + 1]` from
///   `build_csr_gpu_privatised`
/// * `centroids` - Centroid vectors `[k, dim]` in `A`, updated in place
/// * `k` - Number of clusters
/// * `dim` - Embedding dimensionality
/// * `client` - CubeCL compute client for the target device
///
/// ### Returns
///
/// `Ok(())`; `centroids` is updated in place. `CubeclUtils` if the grid is over
/// the device limit.
pub fn segmented_update<R, S, A>(
    data: &GpuTensor<R, S>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    centroids: &GpuTensor<R, A>,
    k: usize,
    dim: usize,
    client: &ComputeClient<R>,
) -> Result<(), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(k as u32, &limits)?;
    let dim_eff = dim.min(WORKGROUP_128 as usize);
    let n_sub = WORKGROUP_128 as usize / dim_eff;
    unsafe {
        segmented_centroid_update::launch_unchecked::<S, A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            data.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            centroids.clone().into_tensor_arg(),
            k as u32,
            dim,
            dim_eff,
            n_sub,
        );
    }

    Ok(())
}

/// Count how many points changed cluster between two assignment snapshots.
///
/// One thread per point; atomically increments `changed[0]` when a point's
/// current assignment differs from its previous one. `changed` must be
/// zero-initialised before launch. This is the convergence signal for the
/// driver: once the partition stops changing the centroids are stable, and
/// unlike centroid drift it reaches zero exactly (the cosine path's drift
/// never does, due to fp32 renormalisation noise).
///
/// ### Params
///
/// * `curr` - Current assignment indices `[n]`
/// * `prev` - Previous assignment indices `[n]`
/// * `changed` - Single-element atomic counter `[1]`, incremented in place
/// * `n` - Total number of data points
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
pub fn count_changed(
    curr: &Tensor<u32>,
    prev: &Tensor<u32>,
    changed: &mut Tensor<Atomic<u32>>,
    n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i >= n {
        terminate!();
    }
    if curr[i as usize] != prev[i as usize] {
        Atomic::fetch_add(&changed[0], 1u32);
    }
}

/// Per-centroid L2 norm of the centroid matrix.
///
/// One workgroup per centroid; thread 0 accumulates the sum of squares across
/// the full row and writes its square root. Kept device-side so that cosine
/// assignment never requires a host readback of the centroid matrix between
/// iterations. Operates entirely in the centroid precision (`A` in the driver).
///
/// ### Params
///
/// * `centroids` - Centroid vectors `[k, dim]`
/// * `norms` - Output L2 norms `[k]`, written by thread 0 of each workgroup
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> centroid index
/// * Only `UNIT_POS_X == 0` writes output
#[cube(launch_unchecked)]
pub fn centroid_norms_l2<F: Float>(
    centroids: &Tensor<F>,
    norms: &mut Tensor<F>,
    k: u32,
    #[comptime] dim: usize,
) {
    let c = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if c >= k {
        terminate!();
    }
    if UNIT_POS_X == 0u32 {
        let base = c as usize * dim;
        let mut acc = F::new(0.0);
        for e in 0..dim {
            let v = centroids[base + e];
            acc += v * v;
        }
        norms[c as usize] = F::sqrt(acc);
    }
}

/// Pull starved centroids toward a donor point, in place on device.
///
/// Device half of the balancing step described on
/// [`crate::utils::k_means_utils::adjust_centers`]. Each entry names a centroid
/// to rescue, a *position in the CSR* to draw the donor point from, and the
/// weight the old centroid keeps. Resolving the position to a point id happens
/// here rather than on the host, which is what keeps the balancing readback
/// down to `k + 1` offsets instead of the whole `n`-long assignment vector.
///
/// ### Params
///
/// * `data` - Data points `[n, dim]` in storage precision `S`
/// * `all_indices` - CSR point ids from `build_csr_gpu_privatised`
/// * `fix_target` - Centroid index to reseed, one per entry
/// * `fix_pos` - Index into `all_indices` naming the donor point
/// * `fix_weight` - Weight the existing centroid keeps, `min(count, pullback)`
/// * `centroids` - Centroid vectors `[k, dim]` in `A`, updated in place
/// * `n_fix` - Number of entries; the caller skips the launch when zero
/// * `dim` - Embedding dimensionality (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> entry index
/// * `UNIT_POS_X` -> dimension stride offset within the centroid row
#[cube(launch_unchecked)]
pub fn balance_reseed_centroids<S: Float, A: Float>(
    data: &Tensor<S>,
    all_indices: &Tensor<u32>,
    fix_target: &Tensor<u32>,
    fix_pos: &Tensor<u32>,
    fix_weight: &Tensor<A>,
    centroids: &mut Tensor<A>,
    n_fix: u32,
    #[comptime] dim: usize,
) {
    let f = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if f >= n_fix {
        terminate!();
    }

    let target = fix_target[f as usize] as usize;
    let donor = all_indices[fix_pos[f as usize] as usize] as usize;
    let w = fix_weight[f as usize];
    let denom = w + A::new(1.0);

    let c_base = target * dim;
    let d_base = donor * dim;

    let mut e = UNIT_POS_X as usize;
    while e < dim {
        let x = A::cast_from(data[d_base + e]);
        centroids[c_base + e] = (centroids[c_base + e] * w + x) / denom;
        e += CUBE_DIM_X as usize;
    }
}

/// Reseed starved centroids after a Lloyd step, RAFT `adjust_centers` style.
///
/// Host driver for [`balance_reseed_centroids`]. Reads only `offsets` back from
/// the device (`k + 1` u32s), derives the cluster sizes from it, and picks a
/// donor as a position inside an over-full cluster's CSR segment. Donor
/// clusters are round-robined so several starved centroids do not all land in
/// the same neighbourhood.
///
/// Deliberately never downloads the assignments: at the subsample sizes the
/// clustered NN-Descent trains on, an `n`-long readback every iteration would
/// cost more than the whole balancing step.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Data points `[n, dim]` in storage precision `S`
/// * `all_indices` - CSR point ids from the current iteration
/// * `offsets` - CSR offsets `[k + 1]` from the current iteration
/// * `cent_gpu` - Centroid vectors `[k, dim]` in `A`, updated in place
/// * `cnorm_gpu` - Centroid L2 norms `[k]`, refreshed for cosine
/// * `n` - Number of data points
/// * `k` - Number of clusters
/// * `dim` - Padded embedding dimensionality
/// * `metric` - Distance metric
/// * `seed` - Seeds the donor round-robin; vary it per iteration
///
/// ### Returns
///
/// Number of centroids reseeded. Zero means the partition was balanced enough
/// to leave alone, and no kernel was launched.
#[allow(clippy::too_many_arguments)]
fn balance_centroids<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    all_indices: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    cent_gpu: &GpuTensor<R, A>,
    cnorm_gpu: &GpuTensor<R, A>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
    seed: usize,
) -> Result<usize, AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement + num_traits::Float,
{
    if k < 2 || n == 0 {
        return Ok(0);
    }

    let offs = offsets.clone().read(client)?;
    let counts: Vec<usize> = (0..k).map(|c| (offs[c + 1] - offs[c]) as usize).collect();

    let average = n as f64 / k as f64;
    let floor = average * BALANCE_THRESHOLD;

    let donors: Vec<usize> = (0..k).filter(|&c| counts[c] as f64 > average).collect();
    if donors.is_empty() {
        return Ok(0);
    }

    let mut fix_target: Vec<u32> = Vec::new();
    let mut fix_pos: Vec<u32> = Vec::new();
    let mut fix_weight: Vec<A> = Vec::new();
    let mut cursor = seed;

    for c in 0..k {
        if counts[c] as f64 > floor {
            continue;
        }
        cursor = cursor.wrapping_add(BALANCE_DONOR_STRIDE);
        let b = donors[cursor % donors.len()];
        let w = counts[c].min(BALANCE_PULLBACK);

        fix_target.push(c as u32);
        fix_pos.push((offs[b] as usize + cursor % counts[b]) as u32);
        fix_weight.push(<A as num_traits::NumCast>::from(w).unwrap());
    }

    let n_fix = fix_target.len();
    if n_fix == 0 {
        return Ok(0);
    }

    let target_gpu = GpuTensor::<R, u32>::from_slice(&fix_target, vec![n_fix], client)?;
    let pos_gpu = GpuTensor::<R, u32>::from_slice(&fix_pos, vec![n_fix], client)?;
    let weight_gpu = GpuTensor::<R, A>::from_slice(&fix_weight, vec![n_fix], client)?;

    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(n_fix as u32, &limits)?;

    unsafe {
        balance_reseed_centroids::launch_unchecked::<S, A, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(WORKGROUP_128),
            data_gpu.clone().into_tensor_arg(),
            all_indices.clone().into_tensor_arg(),
            target_gpu.into_tensor_arg(),
            pos_gpu.into_tensor_arg(),
            weight_gpu.into_tensor_arg(),
            cent_gpu.clone().into_tensor_arg(),
            n_fix as u32,
            dim,
        );
    }

    // The reseed moved centroids, so cosine norms taken before it are stale.
    if *metric == Dist::Cosine {
        let (nx, ny) = grid_2d(k as u32, &limits)?;
        unsafe {
            centroid_norms_l2::launch_unchecked::<A, R>(
                client,
                CubeCount::Static(nx, ny, 1),
                CubeDim::new_1d(1),
                cent_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                k as u32,
                dim,
            );
        }
    }

    Ok(n_fix)
}

/// One Lloyd's iteration entirely on device: assign, rebuild CSR, update
/// centroids.
///
/// Runs `flash_assign_device` to compute hard assignments,
/// `build_csr_gpu_privatised` to sort point indices into cluster segments, and
/// `segmented_update` to recompute each centroid as the mean of its assigned
/// points. For cosine distance, centroid norms are refreshed in place via
/// `centroid_norms_l2` so that `cnorm_gpu` is consistent with `cent_gpu` on
/// exit. Pre-allocated scratch buffers are passed in to avoid per-iteration
/// allocation.
///
/// ### Type parameters
///
/// * `S` - Storage type of the data buffer
/// * `A` - Accumulator type; also the type of centroids and norms
///
/// ### Params
///
/// * `client` - CubeCL compute client for the target device
/// * `data_gpu` - Data points already on device `[n, dim]` in `S`
/// * `cent_gpu` - Centroid vectors `[k, dim]` in `A`, updated in place
/// * `pnorm_gpu` - Pre-computed point L2 norms `[n]` in `A`; ignored for
///   Euclidean
/// * `cnorm_gpu` - Pre-computed centroid L2 norms `[k]` in `A`, refreshed on
///   exit for cosine; ignored for Euclidean
/// * `assign_gpu` - Output assignment indices `[n]`, overwritten each call
/// * `n` - Number of data points
/// * `k` - Number of centroids
/// * `dim` - Embedding dimensionality (must be a multiple of `LINE_SIZE`)
/// * `metric` - Distance metric (`SquaredEuclidean` or `Cosine`)
/// * `cube_count` - Number of workgroups used for the privatised histogram
/// * `privatised_counts` - Pre-allocated scratch buffer `[cube_count * k]`
/// * `counts` - Pre-allocated cluster size buffer `[k]`
/// * `offsets` - Pre-allocated offset buffer `[k + 1]`
/// * `all_indices` - Pre-allocated CSR index buffer `[n]`
///
/// ### Returns
///
/// `Ok(())`; `cent_gpu`, `assign_gpu` and the CSR scratch are updated in place.
/// `CubeclUtils` if any dispatch is over the device limit.
#[allow(clippy::too_many_arguments)]
fn lloyd_step<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    cent_gpu: &GpuTensor<R, A>,
    pnorm_gpu: &GpuTensor<R, A>,
    cnorm_gpu: &GpuTensor<R, A>,
    assign_gpu: &GpuTensor<R, u32>,
    n: usize,
    k: usize,
    dim: usize,
    metric: &Dist,
    cube_count: usize,
    privatised_counts: &GpuTensor<R, u32>,
    counts: &GpuTensor<R, u32>,
    offsets: &GpuTensor<R, u32>,
    all_indices: &GpuTensor<R, u32>,
) -> Result<(), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement,
{
    flash_assign_device::<S, A, R>(
        client, data_gpu, cent_gpu, pnorm_gpu, cnorm_gpu, assign_gpu, n, k, dim, metric,
    )?;

    build_csr_gpu_privatised::<R>(
        assign_gpu,
        n,
        k,
        cube_count,
        privatised_counts,
        counts,
        offsets,
        all_indices,
        client,
    )?;

    segmented_update::<R, S, A>(data_gpu, all_indices, offsets, cent_gpu, k, dim, client)?;

    if *metric == Dist::Cosine {
        let limits = GpuLimits::from_client(client);
        let (gx, gy) = grid_2d(k as u32, &limits)?;
        unsafe {
            centroid_norms_l2::launch_unchecked::<A, R>(
                client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d(1),
                cent_gpu.clone().into_tensor_arg(),
                cnorm_gpu.clone().into_tensor_arg(),
                k as u32,
                dim,
            );
        }
    }

    Ok(())
}

/// Run the device-resident Lloyd's loop and read back the final state.
///
/// All assignment-side buffers (`assign_gpu`, the CSR scratch) are allocated
/// internally. The caller provides the data tensor in storage precision `S`
/// and the centroid plus norm tensors in accumulator precision `A`; the loop
/// returns the final assignments and the centroid buffer in `A`.
///
/// Bifurcates on `params.fixed`: fixed mode submits every iteration back-to-
/// back with no readback; assignment-checked mode allocates an alternate
/// assignment buffer, double-buffers between the two, and reads back a
/// single-element change counter each iteration to test convergence.
///
/// ### Type parameters
///
/// * `S` - Storage type of the data buffer (`T` on the unquantised path,
///   `half::f16` on the quantised path)
/// * `A` - Accumulator and centroid type (`T` on both paths)
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Input data `[n, dim_padded]` in storage precision `S`
/// * `cent_gpu` - Centroid matrix `[n_centroids, dim_padded]` in `A`, updated
///   in place across the loop
/// * `pnorm_gpu` - Point L2 norms `[n]`; only read for cosine
/// * `cnorm_gpu` - Centroid L2 norms `[n_centroids]`; only read for cosine
/// * `n` - Number of data points
/// * `n_centroids` - Number of clusters
/// * `dim_padded` - Padded vector dimension, a multiple of `LINE_SIZE`
/// * `dist` - Distance metric
/// * `params` - Iteration count and the fixed / assignment-checked switch
/// * `seed` - Seeds the donor round-robin when `params.balanced` is set
/// * `verbose` - Print per-iteration convergence progress to stdout
///
/// ### Returns
///
/// `(assignments, centroids)` read back from the device: one cluster index per
/// point, and the final `[n_centroids, dim_padded]` centroid buffer flattened
/// row-major. `CubeclUtils` if any dispatch or allocation busts a device limit.
#[allow(clippy::too_many_arguments)]
fn run_kmeans_loop<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    cent_gpu: &GpuTensor<R, A>,
    pnorm_gpu: &GpuTensor<R, A>,
    cnorm_gpu: &GpuTensor<R, A>,
    n: usize,
    n_centroids: usize,
    dim_padded: usize,
    dist: &Dist,
    params: &KMeansGpuParams,
    seed: usize,
    verbose: bool,
) -> Result<(Vec<usize>, Vec<A>), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement + num_traits::Float,
{
    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], client)?;

    let cube_count = 512usize;
    let privatised_counts = GpuTensor::<R, u32>::empty(vec![cube_count, n_centroids], client)?;
    let counts = GpuTensor::<R, u32>::empty(vec![n_centroids], client)?;
    let offsets = GpuTensor::<R, u32>::empty(vec![n_centroids + 1], client)?;
    let all_indices = GpuTensor::<R, u32>::empty(vec![n], client)?;

    if params.fixed {
        if verbose {
            println!(
                "    Dispatching the {:?} iters to the GPU kernel.",
                params.iters
            )
        }
        for iter in 0..params.iters {
            lloyd_step::<S, A, R>(
                client,
                data_gpu,
                cent_gpu,
                pnorm_gpu,
                cnorm_gpu,
                &assign_gpu,
                n,
                n_centroids,
                dim_padded,
                dist,
                cube_count,
                &privatised_counts,
                &counts,
                &offsets,
                &all_indices,
            )?;

            if params.balanced {
                balance_centroids::<S, A, R>(
                    client,
                    data_gpu,
                    &all_indices,
                    &offsets,
                    cent_gpu,
                    cnorm_gpu,
                    n,
                    n_centroids,
                    dim_padded,
                    dist,
                    seed.wrapping_add(iter),
                )?;
            }
        }
    } else {
        // A change floor to absorb near-equidistant points that flip between
        // equally good centroids without stalling termination.
        let change_floor = (n / 10_000).max(1) as u32;

        let limits = GpuLimits::from_client(client);
        let assign_alt_gpu = GpuTensor::<R, u32>::empty(vec![n], client)?;
        let (cnt_gx, cnt_gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X), &limits)?;

        for iter in 0..params.iters {
            // `cur` receives this iteration's assignments; `prev` holds last.
            let (cur, prev) = if iter % 2 == 0 {
                (&assign_gpu, &assign_alt_gpu)
            } else {
                (&assign_alt_gpu, &assign_gpu)
            };

            lloyd_step::<S, A, R>(
                client,
                data_gpu,
                cent_gpu,
                pnorm_gpu,
                cnorm_gpu,
                cur,
                n,
                n_centroids,
                dim_padded,
                dist,
                cube_count,
                &privatised_counts,
                &counts,
                &offsets,
                &all_indices,
            )?;

            if params.balanced {
                balance_centroids::<S, A, R>(
                    client,
                    data_gpu,
                    &all_indices,
                    &offsets,
                    cent_gpu,
                    cnorm_gpu,
                    n,
                    n_centroids,
                    dim_padded,
                    dist,
                    seed.wrapping_add(iter),
                )?;
            }

            // Zeroed single-element atomic counter, recreated per iter.
            let changed_gpu = GpuTensor::<R, u32>::from_slice(&[0u32], vec![1], client)?;
            unsafe {
                count_changed::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(cnt_gx, cnt_gy, 1),
                    CubeDim::new_1d(WORKGROUP_SIZE_X),
                    cur.clone().into_tensor_arg(),
                    prev.clone().into_tensor_arg(),
                    changed_gpu.clone().into_tensor_arg(),
                    n as u32,
                );
            }
            let changed = changed_gpu.read(client)?[0];

            // Iteration 0 has no meaningful `prev` (uninitialised buffer), so
            // only test convergence from the second iteration onward.
            if iter > 0 && changed <= change_floor {
                if verbose {
                    println!(
                        "    Converged at iteration {} ({} assignments changed)",
                        iter + 1,
                        changed
                    );
                }
                break;
            }
            if verbose && (iter + 1) % 10 == 0 {
                println!(
                    "    Iteration {} complete ({} assignments changed)",
                    iter + 1,
                    changed
                );
            }
        }
    }

    // Final assignment against the converged centroids. cnorm_gpu is already
    // consistent with cent_gpu (refreshed at the end of the last lloyd_step).
    flash_assign_device::<S, A, R>(
        client,
        data_gpu,
        cent_gpu,
        pnorm_gpu,
        cnorm_gpu,
        &assign_gpu,
        n,
        n_centroids,
        dim_padded,
        dist,
    )?;

    let assignments: Vec<usize> = assign_gpu
        .read(client)?
        .into_iter()
        .map(|x| x as usize)
        .collect();

    let final_cents = cent_gpu.clone().read(client)?;

    Ok((assignments, final_cents))
}

/// Initialise the centroids, then run the device-resident Lloyd's loop.
///
/// Split out of [`k_means_clusters_gpu`] so both storage precisions share it.
/// Initialisation runs after the data upload because the k-means|| path needs
/// the data already on the device.
///
/// ### Type parameters
///
/// * `S` - Storage precision of the device data buffer
/// * `A` - Accumulator and centroid precision
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `data_gpu` - Data already on device `[n, dim_padded]` in `S`
/// * `data_host` - The same data on the host, in `A`
/// * `norms_host` - Point L2 norms `[n]`; empty when neither the metric nor
///   the initialisation needs them
/// * `pnorm_gpu` - Point L2 norms on device; a single dummy element when the
///   metric is not cosine
/// * `n` - Number of data points
/// * `n_centroids` - Number of centroids
/// * `dim_padded` - Padded vector dimension
/// * `dist` - Distance metric
/// * `params` - GPU k-means parameters
/// * `init_method` - Resolved initialisation strategy
/// * `seed` - Random seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(assignments, final centroid buffer in A)`
///
/// ### Errors
///
/// * `CubeclUtils` if a dispatch or an allocation busts a device limit, or a
///   read-back fails during initialisation or the loop.
#[allow(clippy::too_many_arguments)]
fn init_and_run<S, A, R>(
    client: &ComputeClient<R>,
    data_gpu: &GpuTensor<R, S>,
    data_host: &[A],
    norms_host: &[A],
    pnorm_gpu: &GpuTensor<R, A>,
    n: usize,
    n_centroids: usize,
    dim_padded: usize,
    dist: &Dist,
    params: &KMeansGpuParams,
    init_method: KMeansInit,
    seed: usize,
    verbose: bool,
) -> Result<(Vec<usize>, Vec<A>), AnnSearchErrors>
where
    R: Runtime,
    S: Float + cubecl::CubeElement,
    A: Float + cubecl::CubeElement + num_traits::Float + AnnSearchFloat,
{
    let centroids = match init_method {
        KMeansInit::Random => {
            if verbose {
                println!("  Initialising centroids via fast random selection");
            }
            fast_random_init(data_host, dim_padded, n, n_centroids, seed)
        }
        KMeansInit::KMeansParallel => {
            if verbose {
                println!("  Initialising centroids via k-means|| (GPU)");
            }
            kmeans_parallel_init_gpu::<S, A, R>(
                client,
                data_gpu,
                data_host,
                norms_host,
                pnorm_gpu,
                n,
                dim_padded,
                n_centroids,
                dist,
                seed,
            )?
        }
    };

    let cent_gpu =
        GpuTensor::<R, A>::from_slice(&centroids, vec![n_centroids, dim_padded], client)?;

    let cnorm_gpu = if *dist == Dist::Cosine {
        let cnorms: Vec<A> = (0..n_centroids)
            .map(|c| A::calculate_l2_norm(&centroids[c * dim_padded..(c + 1) * dim_padded]))
            .collect();
        GpuTensor::<R, A>::from_slice(&cnorms, vec![n_centroids], client)?
    } else {
        GpuTensor::<R, A>::from_slice(&[A::one()], vec![1], client)?
    };

    run_kmeans_loop::<S, A, R>(
        client,
        data_gpu,
        &cent_gpu,
        pnorm_gpu,
        &cnorm_gpu,
        n,
        n_centroids,
        dim_padded,
        dist,
        params,
        seed,
        verbose,
    )
}

//////////
// Main //
//////////

/// Generate k-means clusters on the GPU.
///
/// Device-resident Lloyd's loop: FlashAssign for assignment, privatised
/// counting-sort CSR plus segmented reduction for the update. Initialisation
/// runs after the upload: the random path reuses the host [`fast_random_init`],
/// the k-means|| path runs its D² sweeps on device via
/// `kmeans_parallel_init_gpu` and only brings the distances back.
///
/// With `params.fixed` the loop submits every iteration back-to-back and never
/// synchronises. Otherwise convergence is detected via assignment stability:
/// the loop stops once the number of points changing cluster between
/// iterations drops to a small floor. This terminates the cosine path too,
/// where fp32 renormalisation noise keeps per-centroid drift pinned at a tiny
/// non-zero value indefinitely. A small non-zero floor absorbs
/// near-equidistant points that flip between equally good centroids without
/// stalling termination.
///
/// On a runtime without u32 atomics the privatised histogram silently does
/// nothing, so the whole build falls back to the CPU k-means. See
/// [`k_means_gpu_flat`], which is where that check lives.
///
/// ### Mixed precision
///
/// With `params.quantise_to_f16 == true` the data buffer is converted to
/// `half::f16` host-side and uploaded as fp16; centroids, norms, distance
/// accumulators and centroid sums all stay in `T`. The kernels cast back to
/// `T` element-by-element on load, so accumulation drift is bounded by the
/// `T` path. Requires `shader-f16` on the wgpu adapter; pre-normalise or
/// rescale wide-range inputs before opting in.
///
/// ### Params
///
/// * `data` - The data to cluster, samples x features
/// * `dist` - Distance metric, `"euclidean"` or `"cosine"`; unknown strings
///   default to squared Euclidean
/// * `n_centroids` - Number of centroids
/// * `kmeans_params` - Optional [KMeansGpuParams]
/// * `seed` - Seed for reproducible initialisation
/// * `device` - CubeCL runtime device
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(centroid matrix of shape n_centroids x dim, assignments)`
///
/// ### Errors
///
/// * `CubeclUtils` if a dispatch grid is over the device's cube-count limit
///   or an upload busts the per-binding size limit, or a read-back fails.
#[allow(clippy::too_many_arguments)]
pub fn k_means_clusters_gpu<T, R>(
    data: impl AnnMatrix<T>,
    dist: &str,
    n_centroids: usize,
    kmeans_params: Option<KMeansGpuParams>,
    seed: usize,
    device: R::Device,
    verbose: bool,
) -> Result<(Mat<T>, Vec<usize>), AnnSearchErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + num_traits::Float + Sum + AnnSearchFloat,
{
    let metric = parse_ann_dist(dist).unwrap_or_else(|| {
        println!(
            "Unknown string provided ({:?}). Defaulting to Squared Euclidean",
            dist
        );
        Dist::default()
    });

    let (data_flat, n, dim) = data.into_row_major();
    let client = R::client(&device);

    let (centroids, assignments) = k_means_gpu_flat::<T, R>(
        &data_flat,
        dim,
        n,
        n_centroids,
        &metric,
        kmeans_params,
        seed,
        &client,
        verbose,
    )?;

    let centroid_mat = Mat::from_fn(n_centroids, dim, |i, j| centroids[i * dim + j]);

    Ok((centroid_mat, assignments))
}

/// Run the CPU k-means and return the same pair the device path does.
///
/// Taken when the runtime cannot serve the device loop, currently only for the
/// missing-atomics case in [`k_means_gpu_flat`]. `KMeansGpuParams` carries no
/// `LloydPath`, so the CPU side picks its own via its usual heuristic.
///
/// ### Params
///
/// * `data` - Flattened row-major data, `n * dim`, unpadded
/// * `dim` - Embedding dimensionality
/// * `n` - Number of data points
/// * `n_centroids` - Number of centroids
/// * `metric` - Distance metric
/// * `params` - GPU parameters, mapped onto [`KMeansTrainingParams`]
/// * `seed` - Random seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(centroids, assignments)` in the same layout as [`k_means_gpu_flat`].
#[allow(clippy::too_many_arguments)]
fn kmeans_cpu_fallback<T>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    params: &KMeansGpuParams,
    seed: usize,
    verbose: bool,
) -> Result<(Vec<T>, Vec<usize>), AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let cpu_params =
        KMeansTrainingParams::new(params.iters, params.init, None).with_balancing(params.balanced);

    let centroids = train_centroids(
        data,
        dim,
        n,
        n_centroids,
        metric,
        Some(cpu_params),
        seed,
        verbose,
    )?;

    let (data_norms, centroid_norms) = if *metric == Dist::Cosine {
        (
            (0..n)
                .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                .collect(),
            (0..n_centroids)
                .map(|c| T::calculate_l2_norm(&centroids[c * dim..(c + 1) * dim]))
                .collect(),
        )
    } else {
        (vec![T::one(); n], vec![T::one(); n_centroids])
    };

    let assignments = assign_all_parallel(
        data,
        &data_norms,
        dim,
        n,
        &centroids,
        &centroid_norms,
        n_centroids,
        metric,
    );

    Ok((centroids, assignments))
}

/// Train k-means centroids on the GPU from a flat row-major buffer.
///
/// Flat-buffer counterpart to [`k_means_clusters_gpu`], shaped like the CPU
/// [`train_centroids`] so in-crate callers need not round-trip through `faer`.
/// The client is borrowed rather than built from a device, so a caller running
/// many k-means in a loop pays for one device context and reuses its buffer
/// pool instead of re-faulting the allocations on every call.
///
/// ### Params
///
/// * `data` - Flattened row-major data, `n * dim`, unpadded
/// * `dim` - Embedding dimensionality
/// * `n` - Number of data points
/// * `n_centroids` - Number of centroids to train
/// * `metric` - Distance metric; `Manhattan` is rejected
/// * `params` - Optional [`KMeansGpuParams`]; `None` takes the defaults
/// * `seed` - Seed for reproducible initialisation
/// * `client` - CubeCL compute client, borrowed for the call
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// `(centroids, assignments)` where `centroids` is `n_centroids * dim`
/// row-major and **unpadded**, matching the CPU [`train_centroids`] layout, and
/// `assignments` holds the final cluster index of each of the `n` points.
///
/// ### Errors
///
/// * `DistanceNotSupported` for `Manhattan`
/// * `CubeclUtils` if a dispatch grid is over the device's cube-count limit
///   or an upload busts the per-binding size limit, or a read-back fails
#[allow(clippy::too_many_arguments)]
pub fn k_means_gpu_flat<T, R>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    params: Option<KMeansGpuParams>,
    seed: usize,
    client: &ComputeClient<R>,
    verbose: bool,
) -> Result<(Vec<T>, Vec<usize>), AnnSearchErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + num_traits::Float + Sum + AnnSearchFloat,
{
    let start = Instant::now();

    let params = params.unwrap_or_default();
    if *metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(*metric));
    }

    // `fast_random_init` draws `n_centroids` distinct rows, so fewer rows than
    // centroids indexes past the end of its shuffled list. `train_centroids`
    // guards the CPU path; this entry point reaches the same seeding directly,
    // and `resolve_init` picks the random path for every `n_centroids > 200`.
    if n_centroids > n {
        return Err(AnnSearchErrors::TooFewSamplesForCentroids {
            n_centroids,
            n_samples: n,
        });
    }

    // The Lloyd loop rebuilds a CSR every iteration through a privatised
    // histogram, which is entirely `Atomic::fetch_add`. On a runtime without
    // u32 atomics -- the CubeCL CPU runtime is the one that matters here, since
    // the GPU indices' CI tests run on it -- those launches do nothing and
    // report nothing, so every cluster comes back empty and the centroids never
    // move. See `atomic_u32_add_supported` for why that failure is silent.
    // Falling back to the CPU k-means keeps the answer correct; only the
    // acceleration is lost.
    if !atomic_u32_add_supported(client) {
        if verbose {
            println!("  Device has no u32 atomics; running k-means on the CPU instead.");
        }
        return kmeans_cpu_fallback(data, dim, n, n_centroids, metric, &params, seed, verbose);
    }

    let dim_padded = dim.next_multiple_of(LINE_SIZE);
    let data_padded = if dim_padded != dim {
        pad_vectors(data, n, dim, dim_padded)
    } else {
        data.to_vec()
    };

    let init_method = params.init.unwrap_or(if n_centroids > 200 {
        KMeansInit::Random
    } else {
        KMeansInit::KMeansParallel
    });

    if verbose {
        println!("  Moving data to GPU.");
    }

    // Norms are needed by the cosine assignment kernel and by the k-means||
    // initialisation, so compute them once. They depend only on the data, so
    // they can be uploaded before the centroids exist.
    let need_norms = *metric == Dist::Cosine || init_method == KMeansInit::KMeansParallel;
    let norms_host: Vec<T> = if need_norms {
        (0..n)
            .map(|i| T::calculate_l2_norm(&data_padded[i * dim_padded..(i + 1) * dim_padded]))
            .collect()
    } else {
        Vec::new()
    };
    let pnorm_gpu = if *metric == Dist::Cosine {
        GpuTensor::<R, T>::from_slice(&norms_host, vec![n], client)?
    } else {
        GpuTensor::<R, T>::from_slice(&[T::one()], vec![1], client)?
    };

    if verbose {
        println!(
            "  Running Lloyd's iterations (GPU, {}, {})",
            if params.fixed {
                "fixed"
            } else {
                "assignment-checked"
            },
            if params.quantise_to_f16 {
                "fp16 data / fp32 accum"
            } else {
                "native precision"
            }
        );
    }

    // Bifurcation point. Both branches share the initialisation, centroid,
    // norm and CSR machinery; they differ only in the storage precision of
    // `data_gpu`, which propagates through the generic kernels.
    let (assignments, final_cents) = if params.quantise_to_f16 {
        let data_f16: Vec<half::f16> = data_padded
            .iter()
            .map(|x| half::f16::from_f32(x.to_f32().unwrap_or(0.0)))
            .collect();
        let data_gpu =
            GpuTensor::<R, half::f16>::from_slice(&data_f16, vec![n, dim_padded], client)?;

        if verbose {
            println!("  ... moved data to GPU (fp16): {:.2?}", start.elapsed());
        }

        init_and_run::<half::f16, T, R>(
            client,
            &data_gpu,
            &data_padded,
            &norms_host,
            &pnorm_gpu,
            n,
            n_centroids,
            dim_padded,
            metric,
            &params,
            init_method,
            seed,
            verbose,
        )?
    } else {
        let data_gpu = GpuTensor::<R, T>::from_slice(&data_padded, vec![n, dim_padded], client)?;

        if verbose {
            println!("  ... moved data to GPU: {:.2?}", start.elapsed());
        }

        init_and_run::<T, T, R>(
            client,
            &data_gpu,
            &data_padded,
            &norms_host,
            &pnorm_gpu,
            n,
            n_centroids,
            dim_padded,
            metric,
            &params,
            init_method,
            seed,
            verbose,
        )?
    };

    // Strip the padding so the returned buffer matches the CPU contract.
    let centroids: Vec<T> = if dim_padded != dim {
        (0..n_centroids)
            .flat_map(|c| final_cents[c * dim_padded..c * dim_padded + dim].to_vec())
            .collect()
    } else {
        final_cents
    };

    if verbose {
        println!(
            "Finished GPU-accelerated k-means in {:.2?}.",
            start.elapsed()
        );
    }

    Ok((centroids, assignments))
}

/// Train k-means centroids on the GPU, discarding the training assignments.
///
/// Drop-in GPU counterpart to the CPU [`train_centroids`], for callers that
/// train on a subsample and then assign the full dataset separately via
/// [`assign_all_gpu`] -- which is what the IVF and clustered NN-Descent builds
/// both do.
///
/// ### Params
///
/// * `data` - Flattened row-major training data, `n * dim`, unpadded
/// * `dim` - Embedding dimensionality
/// * `n` - Number of training points
/// * `n_centroids` - Number of centroids to train
/// * `metric` - Distance metric; `Manhattan` is rejected
/// * `params` - Optional [`KMeansGpuParams`]
/// * `seed` - Seed for reproducible initialisation
/// * `client` - CubeCL compute client, borrowed for the call
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Flat `n_centroids * dim` row-major centroid buffer, unpadded.
#[allow(clippy::too_many_arguments)]
pub fn train_centroids_gpu<T, R>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    params: Option<KMeansGpuParams>,
    seed: usize,
    client: &ComputeClient<R>,
    verbose: bool,
) -> Result<Vec<T>, AnnSearchErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + num_traits::Float + Sum + AnnSearchFloat,
{
    k_means_gpu_flat::<T, R>(
        data,
        dim,
        n,
        n_centroids,
        metric,
        params,
        seed,
        client,
        verbose,
    )
    .map(|(centroids, _)| centroids)
}

/// Assign every point to its nearest centroid on the GPU.
///
/// GPU counterpart to the CPU [`assign_all_parallel`]. One thread per point
/// through the FlashAssign kernel, so the cost is a single pass over the data
/// rather than the `O(n * n_centroids * dim)` host scan the IVF build used to
/// pay after training on a subsample.
///
/// ### Params
///
/// * `data` - Flattened row-major data, `n * dim`, unpadded
/// * `dim` - Embedding dimensionality
/// * `n` - Number of data points
/// * `centroids` - Flattened row-major centroids, `n_centroids * dim`, unpadded
/// * `n_centroids` - Number of centroids
/// * `metric` - Distance metric; `Manhattan` is rejected
/// * `client` - CubeCL compute client, borrowed for the call
///
/// ### Returns
///
/// Cluster index per point, length `n`.
///
/// ### Errors
///
/// * `DistanceNotSupported` for `Manhattan`
/// * `CubeclUtils` if the dispatch or an upload busts a device limit
#[allow(clippy::too_many_arguments)]
pub fn assign_all_gpu<T, R>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    n_centroids: usize,
    metric: &Dist,
    client: &ComputeClient<R>,
) -> Result<Vec<usize>, AnnSearchErrors>
where
    R: Runtime,
    T: cubecl::prelude::Float + cubecl::CubeElement + num_traits::Float + Sum + AnnSearchFloat,
{
    if *metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(*metric));
    }

    let dim_padded = dim.next_multiple_of(LINE_SIZE);
    let data_padded = if dim_padded != dim {
        pad_vectors(data, n, dim, dim_padded)
    } else {
        data.to_vec()
    };
    let cents_padded = if dim_padded != dim {
        pad_vectors(centroids, n_centroids, dim, dim_padded)
    } else {
        centroids.to_vec()
    };

    let data_gpu = GpuTensor::<R, T>::from_slice(&data_padded, vec![n, dim_padded], client)?;
    let cent_gpu =
        GpuTensor::<R, T>::from_slice(&cents_padded, vec![n_centroids, dim_padded], client)?;

    let (pnorm_gpu, cnorm_gpu) = if *metric == Dist::Cosine {
        let pnorms: Vec<T> = (0..n)
            .map(|i| T::calculate_l2_norm(&data_padded[i * dim_padded..(i + 1) * dim_padded]))
            .collect();
        let cnorms: Vec<T> = (0..n_centroids)
            .map(|c| T::calculate_l2_norm(&cents_padded[c * dim_padded..(c + 1) * dim_padded]))
            .collect();
        (
            GpuTensor::<R, T>::from_slice(&pnorms, vec![n], client)?,
            GpuTensor::<R, T>::from_slice(&cnorms, vec![n_centroids], client)?,
        )
    } else {
        (
            GpuTensor::<R, T>::from_slice(&[T::one()], vec![1], client)?,
            GpuTensor::<R, T>::from_slice(&[T::one()], vec![1], client)?,
        )
    };

    let assign_gpu = GpuTensor::<R, u32>::empty(vec![n], client)?;

    flash_assign_device::<T, T, R>(
        client,
        &data_gpu,
        &cent_gpu,
        &pnorm_gpu,
        &cnorm_gpu,
        &assign_gpu,
        n,
        n_centroids,
        dim_padded,
        metric,
    )?;

    let raw = assign_gpu.read(client)?;
    Ok(raw.into_iter().map(|c| c as usize).collect())
}

///////////
// Tests //
///////////

/// Tests that run on the CubeCL CPU runtime, so they need no GPU and stay on
/// the plain `gpu` feature. The IVF GPU build drives this k-means on
/// `CpuRuntime` in CI, which is a different code path from the wgpu tests
/// below and has caught real out-of-bounds writes that wgpu silently absorbed.
#[cfg(all(test, feature = "gpu-cpu"))]
mod cpu_runtime_tests {
    use super::*;
    use cubecl::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_kmeans_gpu_cpu_runtime_small_n_many_clusters() {
        // The shape the IVF GPU tests build: 50 points into 20 cells, so most
        // clusters hold one or two points and several histogram cubes see none.
        let (n, dim, k) = (50usize, 4usize, 20usize);
        let data: Vec<f32> = (0..n * dim)
            .map(|idx| ((idx / dim) + (idx % dim)) as f32 / 10.0)
            .collect();

        let client = CpuRuntime::client(&CpuDevice);
        let params = KMeansGpuParams::new(10, None, true, false);

        let (centroids, assignments) = k_means_gpu_flat::<f32, CpuRuntime>(
            &data,
            dim,
            n,
            k,
            &Dist::SquaredEuclidean,
            Some(params),
            42,
            &client,
            false,
        )
        .unwrap();

        assert_eq!(centroids.len(), k * dim);
        assert_eq!(assignments.len(), n);
        assert!(assignments.iter().all(|&c| c < k));
    }

    #[test]
    fn test_kmeans_gpu_cpu_runtime_small_n_many_clusters_random_init() {
        let (n, dim, k) = (50usize, 4usize, 20usize);
        let data: Vec<f32> = (0..n * dim)
            .map(|idx| ((idx / dim) + (idx % dim)) as f32 / 10.0)
            .collect();

        let client = CpuRuntime::client(&CpuDevice);
        let params = KMeansGpuParams::new(10, Some(KMeansInit::Random), true, false);

        let (centroids, assignments) = k_means_gpu_flat::<f32, CpuRuntime>(
            &data,
            dim,
            n,
            k,
            &Dist::SquaredEuclidean,
            Some(params),
            42,
            &client,
            false,
        )
        .unwrap();

        assert_eq!(centroids.len(), k * dim);
        assert_eq!(assignments.len(), n);
    }

    /// Pins the reason [`kmeans_cpu_fallback`] exists.
    ///
    /// The CSR stage of the Lloyd loop is all `Atomic::fetch_add`, and this
    /// runtime has no u32 atomics, so those launches silently no-op: every
    /// cluster reads back empty, the centroids never move, and
    /// `scatter_csr_privatised` then writes through uninitialised cursors,
    /// which segfaults rather than merely producing a wrong answer. If this
    /// assertion ever flips because CubeCL grew CPU atomics, the fallback can
    /// go -- but delete it deliberately, having re-run the CSR on this runtime,
    /// not because it looks redundant.
    #[test]
    fn test_cpu_runtime_has_no_u32_atomics() {
        let client = CpuRuntime::client(&CpuDevice);
        assert!(
            !atomic_u32_add_supported(&client),
            "CPU runtime reports u32 atomics; re-check whether the k-means CPU \
             fallback is still needed"
        );
    }

    /// Degenerate cosine data through the full entry point.
    ///
    /// Note this does **not** exercise `kmeans_pp_on_candidates`: this runtime
    /// has no u32 atomics, so `k_means_gpu_flat` takes the CPU fallback before
    /// reaching it. `test_kmeans_pp_pads_to_k_from_a_single_candidate` covers
    /// the padding directly. What this does cover is that the entry point
    /// returns a correctly shaped result on data that defeats the D^2
    /// weighting.
    #[test]
    fn test_kmeans_gpu_cpu_runtime_degenerate_cosine() {
        // Every row is the same direction, so under cosine every D^2 weight is
        // zero and k-means|| draws no candidates at all.
        let (n, dim, k) = (40usize, 4usize, 5usize);
        let data: Vec<f32> = (0..n * dim).map(|idx| (idx / dim) as f32 + 1.0).collect();

        let client = CpuRuntime::client(&CpuDevice);
        let params = KMeansGpuParams::new(10, Some(KMeansInit::KMeansParallel), true, false);

        let (centroids, assignments) = k_means_gpu_flat::<f32, CpuRuntime>(
            &data,
            dim,
            n,
            k,
            &Dist::Cosine,
            Some(params),
            42,
            &client,
            false,
        )
        .unwrap();

        assert_eq!(centroids.len(), k * dim);
        assert_eq!(assignments.len(), n);
    }

    /// The k-means++ reduction must always emit exactly `k` centroids, even
    /// when the candidate set is smaller than `k`. That happens for real on
    /// degenerate data, where every D^2 weight is zero and the sampler draws
    /// nothing, leaving a single candidate. Returning short would leave the
    /// caller sizing a `[k, dim]` tensor from an undersized buffer.
    ///
    /// The padding is byte-copies of existing candidates, so the partition
    /// genuinely collapses: with one candidate and `k = 5` all five centroids
    /// coincide and four cells stay empty. That is the honest outcome for data
    /// with nothing to separate, but it is worth pinning so it stays a shape
    /// guarantee rather than an accident.
    #[test]
    fn test_kmeans_pp_pads_to_k_from_a_single_candidate() {
        let (dim, k) = (4usize, 5usize);
        let cands = vec![1.0f32, 2.0, 3.0, 4.0];
        let cand_norms = vec![f32::sqrt(30.0)];
        let mut rng = StdRng::seed_from_u64(7);

        let got = kmeans_pp_on_candidates(
            &cands,
            &cand_norms,
            dim,
            k,
            &Dist::SquaredEuclidean,
            &mut rng,
        );

        assert_eq!(got.len(), k * dim);
        for c in 0..k {
            assert_eq!(&got[c * dim..(c + 1) * dim], &cands[..dim]);
        }
    }

    #[test]
    fn test_assign_all_gpu_cpu_runtime_matches_cpu_assignment() {
        // `assign_all_gpu` replaced `assign_all_parallel` in the IVF build, so
        // the two must agree exactly on a case with unambiguous nearest
        // centroids.
        let (n, dim, k) = (64usize, 8usize, 4usize);
        let data: Vec<f32> = (0..n * dim)
            .map(|idx| (idx / dim % k) as f32 * 100.0 + (idx % dim) as f32 * 0.01)
            .collect();
        let centroids: Vec<f32> = (0..k * dim)
            .map(|idx| (idx / dim) as f32 * 100.0 + (idx % dim) as f32 * 0.01)
            .collect();

        let client = CpuRuntime::client(&CpuDevice);
        let gpu = assign_all_gpu::<f32, CpuRuntime>(
            &data,
            dim,
            n,
            &centroids,
            k,
            &Dist::SquaredEuclidean,
            &client,
        )
        .unwrap();

        let norms: Vec<f32> = (0..k * dim).step_by(dim).map(|_| 1.0).collect();
        let cpu = assign_all_parallel(
            &data,
            &vec![1.0f32; n],
            dim,
            n,
            &centroids,
            &norms,
            k,
            &Dist::SquaredEuclidean,
        );

        assert_eq!(gpu, cpu);
    }
}

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod tests {
    use super::*;
    use crate::utils::dist::compute_l2_norm;
    use approx::assert_relative_eq;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    // Wraps flash_assign_device into a Vec<usize> result for test convenience.
    fn run_assign(
        data: &[f32],
        dim: usize,
        n: usize,
        cents: &[f32],
        k: usize,
        metric: &Dist,
        device: &WgpuDevice,
    ) -> Vec<usize> {
        let client = WgpuRuntime::client(device);
        let data_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(data, vec![n, dim], &client).unwrap();
        let cent_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(cents, vec![k, dim], &client).unwrap();
        let assign_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client).unwrap();

        let pnorm_gpu = if *metric == Dist::Cosine {
            let v: Vec<f32> = (0..n)
                .map(|i| compute_l2_norm(&data[i * dim..(i + 1) * dim]))
                .collect();
            GpuTensor::<WgpuRuntime, f32>::from_slice(&v, vec![n], &client).unwrap()
        } else {
            GpuTensor::<WgpuRuntime, f32>::from_slice(&[1.0f32], vec![1], &client).unwrap()
        };
        let cnorm_gpu = if *metric == Dist::Cosine {
            let v: Vec<f32> = (0..k)
                .map(|c| compute_l2_norm(&cents[c * dim..(c + 1) * dim]))
                .collect();
            GpuTensor::<WgpuRuntime, f32>::from_slice(&v, vec![k], &client).unwrap()
        } else {
            GpuTensor::<WgpuRuntime, f32>::from_slice(&[1.0f32], vec![1], &client).unwrap()
        };

        flash_assign_device(
            &client,
            &data_gpu,
            &cent_gpu,
            &pnorm_gpu,
            &cnorm_gpu,
            &assign_gpu,
            n,
            k,
            dim,
            metric,
        )
        .unwrap();

        assign_gpu
            .read(&client)
            .unwrap()
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }

    // Wraps build_csr_gpu_privatised + segmented_update.
    fn run_update(
        data: &[f32],
        assignments: &[u32],
        init_cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
        device: &WgpuDevice,
    ) -> Vec<f32> {
        let client = WgpuRuntime::client(device);
        let cube_count = 64usize;

        let data_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(data, vec![n, dim], &client).unwrap();
        let assign_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(assignments, vec![n], &client).unwrap();
        let cent_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(init_cents, vec![k, dim], &client).unwrap();
        let privatised_counts = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; cube_count * k],
            vec![cube_count * k],
            &client,
        )
        .unwrap();
        let counts =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k], vec![k], &client).unwrap();
        let offsets =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k + 1], vec![k + 1], &client)
                .unwrap();
        let all_indices = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client).unwrap();

        build_csr_gpu_privatised::<WgpuRuntime>(
            &assign_gpu,
            n,
            k,
            cube_count,
            &privatised_counts,
            &counts,
            &offsets,
            &all_indices,
            &client,
        )
        .unwrap();
        segmented_update::<WgpuRuntime, f32, f32>(
            &data_gpu,
            &all_indices,
            &offsets,
            &cent_gpu,
            k,
            dim,
            &client,
        )
        .unwrap();

        cent_gpu.read(&client).unwrap()
    }

    /////////////
    // Helpers //
    /////////////

    fn cpu_assign_euclidean(
        data: &[f32],
        cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                (0..k)
                    .map(|c| {
                        (0..dim)
                            .map(|j| (data[i * dim + j] - cents[c * dim + j]).powi(2))
                            .sum::<f32>()
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect()
    }

    fn cpu_assign_cosine(
        data: &[f32],
        cents: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<usize> {
        (0..n)
            .map(|i| {
                let pnorm = compute_l2_norm(&data[i * dim..(i + 1) * dim]);
                (0..k)
                    .map(|c| {
                        let cnorm = compute_l2_norm(&cents[c * dim..(c + 1) * dim]);
                        let dot: f32 = (0..dim)
                            .map(|j| data[i * dim + j] * cents[c * dim + j])
                            .sum();
                        1.0 - dot / (pnorm * cnorm)
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect()
    }

    fn cpu_centroid_means(
        data: &[f32],
        assignments: &[usize],
        init: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c * dim + j] += data[i * dim + j];
            }
        }
        let mut out = init.to_vec();
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    out[c * dim + j] = sums[c * dim + j] / counts[c] as f32;
                }
            }
        }
        out
    }

    //////////
    // Tests //
    //////////

    /// Exact parity: squared-Euclidean assignment must match the host argmin.
    #[test]
    fn test_assign_euclidean() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (200, 16, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.2)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 11 + 5) % 19) as f32 * 0.2)
            .collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::SquaredEuclidean, &device);
        assert_eq!(got, cpu_assign_euclidean(&data, &cents, n, k, dim));
    }

    /// Same exact gate for cosine, which normalises inside the kernel.
    #[test]
    fn test_assign_cosine() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (120, 8, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 1) % 11) as f32 + 0.5)
            .collect();
        let cents: Vec<f32> = (0..k * dim)
            .map(|i| ((i * 13 + 3) % 17) as f32 + 0.5)
            .collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::Cosine, &device);
        assert_eq!(got, cpu_assign_cosine(&data, &cents, n, k, dim));
    }

    // n not a multiple of the workgroup size to exercise the inactive-thread guard.
    #[test]
    fn test_assign_ragged_n() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (137, 5, 8);
        let data: Vec<f32> = (0..n * dim).map(|i| ((i * 5 + 2) % 13) as f32).collect();
        let cents: Vec<f32> = (0..k * dim).map(|i| ((i * 9 + 4) % 17) as f32).collect();

        let got = run_assign(&data, dim, n, &cents, k, &Dist::SquaredEuclidean, &device);
        assert_eq!(got.len(), n);
        assert_eq!(got, cpu_assign_euclidean(&data, &cents, n, k, dim));
    }

    /// Privatised CSR build must match `build_csr_layout`, ordering aside.
    #[test]
    fn test_csr_privatised_matches_cpu() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let assignments = vec![0u32, 1, 0, 2, 1, 0, 2, 2];
        let (n, k, cube_count) = (assignments.len(), 3usize, 64usize);

        let assign_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&assignments, vec![n], &client).unwrap();
        let privatised_counts = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; cube_count * k],
            vec![cube_count * k],
            &client,
        )
        .unwrap();
        let counts =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k], vec![k], &client).unwrap();
        let offsets =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; k + 1], vec![k + 1], &client)
                .unwrap();
        let all_indices = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client).unwrap();

        build_csr_gpu_privatised::<WgpuRuntime>(
            &assign_gpu,
            n,
            k,
            cube_count,
            &privatised_counts,
            &counts,
            &offsets,
            &all_indices,
            &client,
        )
        .unwrap();

        let idx = all_indices.read(&client).unwrap();
        let off = offsets.read(&client).unwrap();

        let (cpu_idx, cpu_off) = crate::utils::k_means_utils::build_csr_layout(
            assignments.iter().map(|&c| c as usize).collect(),
            n,
            k,
        );

        assert_eq!(off.iter().map(|&o| o as usize).collect::<Vec<_>>(), cpu_off);

        for c in 0..k {
            let s = off[c] as usize;
            let e = off[c + 1] as usize;
            let got: std::collections::HashSet<u32> = idx[s..e].iter().copied().collect();
            let want: std::collections::HashSet<u32> = cpu_idx[cpu_off[c]..cpu_off[c + 1]]
                .iter()
                .map(|&i| i as u32)
                .collect();
            assert_eq!(got, want, "cluster {} membership mismatch", c);
        }
    }

    /// Segmented centroid update against host means, on a well behaved dim.
    #[test]
    fn test_segmented_update() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (300, 8, 32);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.3)
            .collect();
        let assignments: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let init = vec![-1.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);
        let want = cpu_centroid_means(
            &data,
            &assignments.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            &init,
            n,
            k,
            dim,
        );

        for j in 0..k * dim {
            assert!(
                (got[j] - want[j]).abs() < 1e-3,
                "elem {}: {} != {}",
                j,
                got[j],
                want[j]
            );
        }
    }

    // Every thread stores into `s_part`, so SEGMENTED_SMEM_LEN has to cover the
    // whole mapping and not just the `n_sub * dim_eff` slots the reduction
    // reads. Sizing it to the latter is what the kernel did originally, and it
    // overran for any `dim_eff` not dividing the workgroup: `n_sub` truncates,
    // the leftover threads get `sub == n_sub`, and their index collapses to
    // `tx`. Under lavapipe shared memory is an ordinary host allocation, so
    // that is a real out-of-bounds write and not a harmless scribble on
    // on-chip scratch.
    #[test]
    fn test_segmented_update_shared_memory_covers_workgroup() {
        let wg = WORKGROUP_128 as usize;

        // The threshold is real, not hypothetical: no_pcs = 10 in
        // test_fast_cluster_gpu.R pads to 12, which does not divide 128 and so
        // leaves the reduction reading fewer slots than the kernel writes.
        let dim_eff = 10usize.next_multiple_of(LINE_SIZE).min(wg);
        assert!((wg / dim_eff) * dim_eff < wg);

        // Raw dims a caller can supply, padded here exactly as
        // `k_means_clusters_gpu` pads them.
        for dim in [5usize, 8, 10, 15, 32, 48, 50, 65, 100, 130] {
            let dim_eff = dim.next_multiple_of(LINE_SIZE).min(wg);
            for tx in 0..wg {
                let idx = (tx / dim_eff) * dim_eff + tx % dim_eff;
                assert!(
                    idx < SEGMENTED_SMEM_LEN,
                    "dim = {dim}: thread {tx} stores at {idx} in a {SEGMENTED_SMEM_LEN} \
                     element buffer"
                );
            }
        }
    }

    // Loudest reachable shape. dim_eff = 68 leaves n_sub = 1, so 60 of the 128
    // threads write 240 bytes past a 272 byte allocation, once per workgroup.
    // The assertion is expected to pass regardless: the reduction only reads
    // `s in 0..n_sub`, so the corrupted slots never reach the output. Only the
    // allocator can see this.
    #[test]
    fn test_segmented_update_ragged_dim() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (300, 64, 68);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.3)
            .collect();
        let assignments: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let init = vec![-1.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);
        let want = cpu_centroid_means(
            &data,
            &assignments.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            &init,
            n,
            k,
            dim,
        );

        for j in 0..k * dim {
            assert!(
                (got[j] - want[j]).abs() < 1e-3,
                "elem {}: {} != {}",
                j,
                got[j],
                want[j]
            );
        }
    }

    /// `dim` above `WORKGROUP_128` and not a multiple of it, which is where the
    /// dimension loop in `segmented_centroid_update` stops having a uniform
    /// trip count: at 200, threads below 72 run it twice and the rest once.
    /// The barriers are comptime-gated off for exactly this case. Nothing else
    /// in the suite launches this kernel above `dim = 68`, and 200 is an
    /// ordinary number of principal components.
    #[test]
    fn test_segmented_update_dim_above_workgroup_not_a_multiple() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (200, 8, 200);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.25)
            .collect();
        let assignments: Vec<u32> = (0..n).map(|i| (i % k) as u32).collect();
        let init = vec![-1.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);
        let want = cpu_centroid_means(
            &data,
            &assignments.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            &init,
            n,
            k,
            dim,
        );

        for j in 0..k * dim {
            assert!(
                (got[j] - want[j]).abs() < 1e-3,
                "elem {}: {} != {}",
                j,
                got[j],
                want[j]
            );
        }
    }

    // Regression shape, found downstream: it aborted a lavapipe CI runner.
    // no_pcs = 10 pads to 12, so 8 threads overrun by 8 elements per workgroup
    // per Lloyd iteration. Many small corruption events rather than the few
    // large ones above. Shape assertions only; an abort is the signal.
    #[test]
    fn test_kmeans_padded_dim_not_divisor_of_workgroup() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (600, 16, 10);
        let mat = Mat::<f32>::from_fn(n, dim, |i, j| ((i * 31 + j * 17) % 100) as f32 * 0.01);
        let params = KMeansGpuParams::new(10, None, true, false);

        let (cents, assignments) = k_means_clusters_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            "euclidean",
            k,
            Some(params),
            42,
            device.clone(),
            false,
        )
        .unwrap();

        assert_eq!(cents.nrows(), k);
        assert_eq!(cents.ncols(), dim);
        assert_eq!(assignments.len(), n);
    }

    // The other update tests use n=300 and n=6, small enough that the
    // unrolled body runs at most once per thread. This one has segments long
    // enough for the unrolled path to dominate and the tail to be a genuine
    // remainder, and a dim that is not a multiple of the sub-stripe count so
    // the outer dimension loop runs more than once.
    #[test]
    // Heavy: n = 20000 with a host reference over the same.
    fn test_segmented_update_long_segments() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (20_000, 4, 48);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 31 + 17) % 97) as f32 * 0.05)
            .collect();
        // Uneven cluster sizes so the segments are not all the same length.
        let assignments: Vec<u32> = (0..n).map(|i| ((i * i + i / 7) % k) as u32).collect();
        let init = vec![-5.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);
        let want = cpu_centroid_means(
            &data,
            &assignments.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            &init,
            n,
            k,
            dim,
        );

        for j in 0..k * dim {
            assert_relative_eq!(got[j], want[j], max_relative = 1e-4);
        }
    }

    // Well-separated blobs: k-means|| initialisation must recover them
    // exactly, which is the property that would be lost by falling back to
    // random initialisation.
    #[test]
    // Heavy: full k-means|| init plus 20 Lloyd iterations over n = 2000.
    fn test_kmeans_parallel_init_gpu_recovers_blobs() {
        let Some(device) = try_device() else { return };
        let (n_blobs, per_blob, dim) = (8usize, 250usize, 16usize);
        let n = n_blobs * per_blob;

        // Each blob sits on its own axis-aligned corner, far apart relative
        // to the within-blob jitter.
        let data: Vec<f32> = (0..n * dim)
            .map(|idx| {
                let point = idx / dim;
                let d = idx % dim;
                let blob = point / per_blob;
                let centre = if d % n_blobs == blob { 50.0 } else { 0.0 };
                centre + (((point * 31 + d * 17) % 7) as f32) * 0.1
            })
            .collect();
        let mat = Mat::<f32>::from_fn(n, dim, |i, j| data[i * dim + j]);

        let params = KMeansGpuParams::new(20, Some(KMeansInit::KMeansParallel), true, false);
        let (_, assignments) = k_means_clusters_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            "euclidean",
            n_blobs,
            Some(params),
            7,
            device,
            false,
        )
        .unwrap();

        // Every blob must land wholly inside one cluster, and no two blobs
        // may share one.
        let mut blob_to_cluster = vec![usize::MAX; n_blobs];
        for (point, &cluster) in assignments.iter().enumerate() {
            let blob = point / per_blob;
            if blob_to_cluster[blob] == usize::MAX {
                blob_to_cluster[blob] = cluster;
            }
            assert_eq!(
                blob_to_cluster[blob], cluster,
                "blob {} split across clusters",
                blob
            );
        }
        let distinct: std::collections::HashSet<usize> = blob_to_cluster.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            n_blobs,
            "blobs collapsed into shared clusters"
        );
    }

    /// k-means|| seeding must give identical assignments for the same seed.
    #[test]
    fn test_kmeans_parallel_init_gpu_deterministic() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (600, 5, 8);
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 13 + 5) % 41) as f32 * 0.25)
            .collect();
        let mat = Mat::<f32>::from_fn(n, dim, |i, j| data[i * dim + j]);
        let params = KMeansGpuParams::new(10, Some(KMeansInit::KMeansParallel), true, false);

        let run = || {
            k_means_clusters_gpu::<f32, WgpuRuntime>(
                mat.as_ref(),
                "cosine",
                k,
                Some(params),
                99,
                device.clone(),
                false,
            )
            .unwrap()
            .1
        };

        assert_eq!(run(), run(), "same seed gave different assignments");
    }

    // Cluster 1 gets no points; its centroid must survive untouched.
    #[test]
    fn test_segmented_update_empty_cluster() {
        let Some(device) = try_device() else { return };
        let (n, k, dim) = (6, 3, 4);
        let data: Vec<f32> = (0..n * dim).map(|i| (i + 1) as f32).collect();
        let assignments = vec![0u32, 0, 2, 2, 0, 2];
        let init = vec![7.0f32; k * dim];

        let got = run_update(&data, &assignments, &init, n, k, dim, &device);

        for j in 0..dim {
            assert!(
                (got[dim + j] - 7.0).abs() < 1e-6,
                "empty cluster overwritten at dim {}",
                j
            );
        }
    }

    // Device mirror of `test_balanced_training_evens_out_a_skewed_partition`.
    // Same construction, same reason the outliers are there: k-means|| seeds
    // onto them, plain Lloyd's leaves three singleton clusters and one holding
    // the whole blob, so the assertion has something real to detect.
    #[test]
    fn test_balanced_kmeans_gpu_evens_out_a_skewed_partition() {
        let Some(device) = try_device() else { return };
        let (n, dim, k) = (203usize, 2usize, 4usize);

        let mut data = Vec::with_capacity(n * dim);
        for i in 0..200 {
            data.push((i % 20) as f32 * 0.01);
            data.push((i / 20) as f32 * 0.01);
        }
        for i in 0..3 {
            data.push(1000.0 * (i + 1) as f32);
            data.push(-1000.0 * (i + 1) as f32);
        }
        let mat = Mat::<f32>::from_fn(n, dim, |i, j| data[i * dim + j]);

        let largest_cluster = |params: KMeansGpuParams| {
            let (_, assignments) = k_means_clusters_gpu::<f32, WgpuRuntime>(
                mat.as_ref(),
                "euclidean",
                k,
                Some(params),
                7,
                device.clone(),
                false,
            )
            .unwrap();
            let mut counts = vec![0usize; k];
            for &a in &assignments {
                counts[a] += 1;
            }
            *counts.iter().max().unwrap()
        };

        let params = KMeansGpuParams::new(30, Some(KMeansInit::KMeansParallel), true, false);
        let plain = largest_cluster(params);
        let balanced = largest_cluster(params.with_balancing(true));

        assert_eq!(plain, 200, "the unbalanced baseline is not the skewed case");
        assert!(
            balanced < plain,
            "balancing did not shrink the largest cluster: {} vs {}",
            balanced,
            plain
        );
    }
}
