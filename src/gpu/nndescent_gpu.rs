//! GPU-accelerated NNDescent kNN graph construction via CubeCL.
//!
//! All vector data remains GPU-resident throughout construction. The host
//! loop only downloads a single u32 convergence counter per iteration.

use cubecl::frontend::{Atomic, CubePrimitive, Float};
use cubecl::prelude::*;
use faer::MatRef;
use rayon::prelude::*;
use std::iter::Sum;
use std::time::Instant;
use thousands::*;

use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;
use crate::utils::*;

///////////
// Const //
///////////

/// Max proposals per node per iteration. Overflow is silently dropped.
const MAX_PROPOSALS: usize = 128;
/// Default maximum number of NNDescent iterations
const DEFAULT_MAX_ITERS: usize = 15;
/// Default convergence threshold (fraction of k*n edges updated)
const DEFAULT_DELTA: f32 = 0.001;
/// Default sampling rate for the local join
const DEFAULT_RHO: f32 = 0.5;

////////////////////
// Kernel helpers //
////////////////////

/// Simple xorshift32 PRNG for deterministic per-thread random decisions.
///
/// ### Params
///
/// * `state` - Current PRNG state
///
/// ### Returns
///
/// Next PRNG state
#[cube]
fn xorshift(state: u32) -> u32 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// Deterministic hash for per-entry rho sampling decisions.
///
/// Same (node, entry, seed) triple always produces the same result, so no local
/// storage is needed for the participation decision.
///
/// ### Params
///
/// * `node` - Source node index
/// * `entry` - Neighbour slot index within the node's adjacency list
/// * `seed` - Per-iteration seed
///
/// ### Returns
///
/// A pseudo-random u32 for use in sampling decisions
#[cube]
fn entry_hash(node: u32, entry: u32, seed: u32) -> u32 {
    xorshift(node ^ (entry * 2654435769u32) ^ seed)
}

/// Squared Euclidean distance between vectors at indices `a` and `b`.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
///
/// ### Returns
///
/// Squared Euclidean distance between the two vectors
#[cube]
fn dist_sq_euclidean<F: Float + CubePrimitive>(vectors: &Tensor<Line<F>>, a: u32, b: u32) -> F {
    let dim_lines = vectors.shape(1);
    let stride = vectors.stride(0);
    let off_a = a as usize * stride;
    let off_b = b as usize * stride;
    let mut sum = F::new(0.0);

    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let diff = va - vb;
        let sq = diff * diff;
        sum += sq[0];
        sum += sq[1];
        sum += sq[2];
        sum += sq[3];
    }
    sum
}

/// Cosine distance (1 - cosine similarity) between vectors at `a` and `b`.
///
/// Requires pre-computed L2 norms.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms, one per row
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
///
/// ### Returns
///
/// Cosine distance in the range [0, 2]
#[cube]
fn dist_cosine<F: Float>(vectors: &Tensor<Line<F>>, norms: &Tensor<F>, a: u32, b: u32) -> F {
    let dim_lines = vectors.shape(1);
    let stride = vectors.stride(0);
    let off_a = a as usize * stride;
    let off_b = b as usize * stride;
    let mut dot = F::new(0.0);

    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let prod = va * vb;
        dot += prod[0];
        dot += prod[1];
        dot += prod[2];
        dot += prod[3];
    }
    F::new(1.0) - dot / (norms[a as usize] * norms[b as usize])
}

/////////////
// Kernels //
/////////////

/// Initialise the kNN graph with random neighbours.
///
/// One thread per node. Generates k random neighbours, computes distances,
/// and maintains a sorted (ascending by distance) list via insertion.
/// All entries are flagged as new (MSB set).
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms (ignored when `use_cosine` is false)
/// * `n` - Number of vectors
/// * `seed` - Random seed for neighbour generation
/// * `use_cosine` - Whether to use cosine distance instead of squared Euclidean
///
/// Writes an initialised sorted kNN graph into `graph_idx` and `graph_dist`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn init_random_graph<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    n: u32,
    seed: u32,
    #[comptime] use_cosine: bool,
) {
    let node = ABSOLUTE_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    let mut rng = xorshift(node ^ seed ^ 0xDEADBEEFu32);

    for slot in 0..k {
        // Generate random PID, reject self-loops
        rng = xorshift(rng);
        let mut pid = rng % n;
        if pid == node {
            pid = (pid + 1u32) % n;
        }

        // Compute distance
        let dist = if use_cosine {
            dist_cosine(vectors, norms, node, pid)
        } else {
            dist_sq_euclidean(vectors, node, pid)
        };

        // Sorted insertion into slots [0..slot].
        // Find the first position where dist < existing, scanning left to right.
        let mut insert_pos = slot;
        for j in 0..slot {
            if dist < graph_dist[base + j] && insert_pos == slot {
                insert_pos = j;
            }
        }

        // Shift right from insert_pos to slot-1
        for j in 0..slot {
            let src = slot - 1 - j;
            let dst = slot - j;
            if src >= insert_pos {
                graph_idx[base + dst] = graph_idx[base + src];
                graph_dist[base + dst] = graph_dist[base + src];
            }
        }

        graph_idx[base + insert_pos] = pid | is_new_bit;
        graph_dist[base + insert_pos] = dist;
    }
}

/// Zero out proposal counts and the global update counter.
///
/// One thread per node. Thread 0 additionally resets the update counter.
///
/// ### Params
///
/// * `n` - Number of nodes
///
/// Zeroes all entries in `prop_count` and resets `update_counter[0]` to zero.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn reset_proposals(prop_count: &mut Tensor<u32>, update_counter: &mut Tensor<u32>, n: u32) {
    let idx = ABSOLUTE_POS_X;
    if idx < n {
        prop_count[idx as usize] = 0u32;
    }
    if idx == 0u32 {
        update_counter[0usize] = 0u32;
    }
}

/// Core NNDescent local join kernel.
///
/// One thread per source node. For each source node S, iterates over all
/// pairs (i, j) of S's k neighbours where at least one is flagged "new",
/// both pass the rho sampling check, and i < j. Computes the distance
/// between the pair and proposes each as a candidate to the other via
/// atomic slot allocation in the proposal buffers.
///
/// The rho sampling uses a deterministic hash so no per-entry storage
/// is needed; the participation decision for entry `i` of node `S` is
/// recomputed identically each time it is checked.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms (ignored when `use_cosine` is false)
/// * `graph_idx` - Current kNN graph indices, MSB encodes the is-new flag
/// * `graph_dist` - Current kNN graph distances, sorted ascending per row
/// * `n` - Number of nodes
/// * `rho_thresh` - Sampling threshold in [0, 65535]; entry participates if
///   hash < threshold
/// * `iter_seed` - Per-iteration seed for the sampling hash
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
/// * `use_cosine` - Whether to use cosine distance instead of squared Euclidean
///   (comptime)
///
/// Appends candidate (index, distance) pairs into `prop_idx`, `prop_dist`,
/// and increments `prop_count` atomically for each target node.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> source node index
#[cube(launch_unchecked)]
fn local_join<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph_idx: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n: u32,
    rho_thresh: u32,
    iter_seed: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
) {
    let node = ABSOLUTE_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let pid_mask = 0x7FFFFFFFu32;
    let is_new_bit = 1u32 << 31;
    let graph_base = node as usize * k;

    for i in 0..k {
        let hash_i = entry_hash(node, i as u32, iter_seed);
        if (hash_i & 0xFFFFu32) < rho_thresh {
            let entry_i = graph_idx[graph_base + i];
            let pid_i = entry_i & pid_mask;
            let is_new_i = entry_i >= is_new_bit;

            for j in 0..k {
                if j > i {
                    let hash_j = entry_hash(node, j as u32, iter_seed);
                    if (hash_j & 0xFFFFu32) < rho_thresh {
                        let entry_j = graph_idx[graph_base + j];
                        let pid_j = entry_j & pid_mask;
                        let is_new_j = entry_j >= is_new_bit;

                        if (is_new_i || is_new_j) && pid_i != pid_j {
                            let dist = if use_cosine {
                                dist_cosine(vectors, norms, pid_i, pid_j)
                            } else {
                                dist_sq_euclidean(vectors, pid_i, pid_j)
                            };

                            let slot_i = prop_count[pid_i as usize].fetch_add(1u32);
                            if slot_i < max_proposals {
                                let off =
                                    pid_i as usize * (max_proposals as usize) + (slot_i as usize);
                                prop_idx[off] = pid_j;
                                prop_dist[off] = dist;
                            }

                            let slot_j = prop_count[pid_j as usize].fetch_add(1u32);
                            if slot_j < max_proposals {
                                let off =
                                    pid_j as usize * (max_proposals as usize) + (slot_j as usize);
                                prop_idx[off] = pid_i;
                                prop_dist[off] = dist;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Merge proposals into the sorted kNN graph.
///
/// One thread per node. For each node:
/// 1. Clears the IS_NEW flag on all existing neighbours (marks old).
/// 2. Iterates over received proposals (up to MAX_PROPOSALS).
/// 3. Skips duplicates already in the graph.
/// 4. Inserts improvements into the sorted list, flagged as new.
/// 5. Atomically accumulates the total improvement count.
///
/// ### Params
///
/// * `prop_idx` - Proposal candidate indices, row-major with `max_proposals`
///   columns
/// * `prop_dist` - Proposal candidate distances, matching layout to `prop_idx`
/// * `prop_count` - Number of valid proposals received per node
/// * `n` - Number of nodes
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
///
/// Updates `graph_idx` and `graph_dist` in place with any improvements, flags
/// inserted entries as new, and accumulates the total improvement count into
/// `update_counter[0]`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn merge_proposals<F: Float>(
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    prop_idx: &Tensor<u32>,
    prop_dist: &Tensor<F>,
    prop_count: &Tensor<u32>,
    update_counter: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] max_proposals: u32,
) {
    let node = ABSOLUTE_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let pid_mask = 0x7FFFFFFFu32;
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    // Clear new flags on all existing entries
    for j in 0..k {
        graph_idx[base + j] = graph_idx[base + j] & pid_mask;
    }

    // Read how many proposals this node received (capped at max_proposals)
    let raw_count = prop_count[node as usize];
    let prop_base = node as usize * max_proposals as usize;
    let mut improvements = 0u32;

    // Fixed loop bound (comptime); guard with runtime count
    for p in 0..max_proposals {
        if p < raw_count {
            let candidate = prop_idx[prop_base + p as usize];
            let dist = prop_dist[prop_base + p as usize];

            // Only process if better than current worst
            if dist < graph_dist[base + k - 1] {
                // Check for duplicates
                let mut exists: u32 = 0u32;
                for j in 0..k {
                    if (graph_idx[base + j] & pid_mask) == candidate {
                        exists = 1u32;
                    }
                }

                // Reject duplicates and self-loops
                if exists == 0u32 && candidate != node {
                    // Find insertion point (first slot where dist < current)
                    let mut insert_pos = k - 1;
                    for j in 0..k {
                        if dist < graph_dist[base + j] && insert_pos == k - 1 {
                            insert_pos = j;
                        }
                    }

                    // Shift right from insert_pos to k-2
                    for j in 0..k - 1 {
                        let src = k - 2 - j;
                        let dst = k - 1 - j;
                        if src >= insert_pos {
                            graph_idx[base + dst] = graph_idx[base + src];
                            graph_dist[base + dst] = graph_dist[base + src];
                        }
                    }

                    // Insert with new flag
                    graph_idx[base + insert_pos] = candidate | is_new_bit;
                    graph_dist[base + insert_pos] = dist;
                    improvements += 1u32;
                }
            }
        }
    }

    if improvements > 0u32 {
        update_counter[0usize].fetch_add(improvements);
    }
}

/////////////
// Helpers //
/////////////

/// Pad vectors to `dim_padded` by appending zeros to each row.
///
/// ### Params
///
/// * `flat` - Flattened row-major vector data with `dim` features per row
/// * `n` - Number of vectors
/// * `dim` - Original feature dimensionality
/// * `dim_padded` - Target dimensionality, must be >= `dim`
///
/// ### Returns
///
/// Flattened row-major buffer of size `n * dim_padded` with each row zero-padded
/// to `dim_padded`
fn pad_vectors<T: Float>(flat: &[T], n: usize, dim: usize, dim_padded: usize) -> Vec<T> {
    let mut padded = vec![T::zero(); n * dim_padded];
    for i in 0..n {
        let src = &flat[i * dim..(i + 1) * dim];
        let dst = &mut padded[i * dim_padded..i * dim_padded + dim];
        dst.copy_from_slice(src);
    }
    padded
}

//////////////////
// NNDescentGpu //
//////////////////

/// GPU-accelerated NNDescent kNN graph builder.
///
/// Builds a k-NN graph entirely on the GPU, downloading only a single u32
/// convergence counter per iteration. The final graph is downloaded once
/// at the end.
pub struct NNDescentGpu<T: Float, R: Runtime> {
    /// Original (unpadded) vector data, flattened row-major
    pub vectors_flat: Vec<T>,
    /// Original embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric
    pub metric: Dist,
    /// Flat kNN graph of size `n * k`, sorted by distance per row
    pub graph: Vec<(usize, T)>,
    /// Whether construction hit the delta threshold
    pub converged: bool,
    /// CubeCL runtime device
    _device: R::Device,
}

impl<T, R> NNDescentGpu<T, R>
where
    R: Runtime,
    T: num_traits::Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + num_traits::FromPrimitive
        + SimdDistance,
{
    /// Build a kNN graph on the GPU via NNDescent.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features). Dimensions will be
    ///   padded to the next multiple of LINE_SIZE if necessary.
    /// * `metric` - Distance metric
    /// * `k` - Neighbours per node (default 30)
    /// * `max_iters` - Maximum NNDescent iterations (default 15)
    /// * `delta` - Convergence threshold as fraction of n*k (default 0.001)
    /// * `rho` - Sampling rate for the local join (default 0.5)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `device` - CubeCL runtime device
    ///
    /// ### Returns
    ///
    /// Initialised struct with the completed kNN graph
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        k: Option<usize>,
        max_iters: Option<usize>,
        delta: Option<f32>,
        rho: Option<f32>,
        seed: usize,
        verbose: bool,
        device: R::Device,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let k = k.unwrap_or(30);
        let max_iters = max_iters.unwrap_or(DEFAULT_MAX_ITERS);
        let delta = delta.unwrap_or(DEFAULT_DELTA);
        let rho = rho.unwrap_or(DEFAULT_RHO);
        let rho_thresh = (rho * 65535.0) as u32;

        // pad dim to next multiple of LINE_SIZE
        let line = LINE_SIZE as usize;
        let dim_padded = dim.next_multiple_of(line);
        let dim_vec = dim_padded / line;

        let vectors_padded = if dim_padded != dim {
            pad_vectors(&vectors_flat, n, dim, dim_padded)
        } else {
            vectors_flat.clone()
        };

        let norms = if metric == Dist::Cosine {
            (0..n)
                .into_par_iter()
                .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
                .collect()
        } else {
            Vec::new()
        };

        if verbose {
            println!(
                "NNDescent-GPU: {} vectors, dim={} (padded to {}), k={}",
                n.separate_with_underscores(),
                dim,
                dim_padded,
                k
            );
        }

        let client = R::client(&device);
        let use_cosine = metric == Dist::Cosine;

        // upload vectors (stays resident for the entire build)
        let vectors_gpu = GpuTensor::<R, T>::from_slice(&vectors_padded, vec![n, dim_vec], &client);

        // norms tensor (dummy scalar if Euclidean to avoid Option in kernel args)
        let norms_gpu = if use_cosine {
            GpuTensor::<R, T>::from_slice(&norms, vec![n], &client)
        } else {
            GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)
        };

        // graph buffers on GPU
        let graph_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);
        let graph_dist_gpu = GpuTensor::<R, T>::empty(vec![n, k], &client);

        // proposal buffers on GPU
        let max_prop = MAX_PROPOSALS;
        let prop_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, max_prop], &client);
        let prop_dist_gpu = GpuTensor::<R, T>::empty(vec![n, max_prop], &client);
        let prop_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

        // convergence counter (single u32)
        let update_counter_gpu = GpuTensor::<R, u32>::empty(vec![1], &client);

        let grid_n = (n as u32).div_ceil(WORKGROUP_SIZE_X);

        // generate the random graph
        let start = Instant::now();

        unsafe {
            let _ = init_random_graph::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.clone().into_tensor_arg(line),
                norms_gpu.clone().into_tensor_arg(1),
                graph_idx_gpu.clone().into_tensor_arg(1),
                graph_dist_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                ScalarArg { elem: seed as u32 },
                use_cosine,
            );
        }

        if verbose {
            println!("  Random init: {:.2?}", start.elapsed());
        }

        // nndescent iterations

        let iter_start = Instant::now();
        let mut converged = false;

        for iter in 0..max_iters {
            // Reset proposal counts and update counter
            unsafe {
                let _ = reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    prop_count_gpu.clone().into_tensor_arg(1),
                    update_counter_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                );
            }

            // local join: enumerate pairs, compute distances, write proposals
            let iter_seed = seed as u32 ^ (iter as u32).wrapping_mul(0x9E3779B9u32);
            unsafe {
                let _ = local_join::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    vectors_gpu.clone().into_tensor_arg(line),
                    norms_gpu.clone().into_tensor_arg(1),
                    graph_idx_gpu.clone().into_tensor_arg(1),
                    graph_dist_gpu.clone().into_tensor_arg(1),
                    prop_idx_gpu.clone().into_tensor_arg(1),
                    prop_dist_gpu.clone().into_tensor_arg(1),
                    prop_count_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    ScalarArg { elem: rho_thresh },
                    ScalarArg { elem: iter_seed },
                    MAX_PROPOSALS as u32,
                    use_cosine,
                );
            }

            // merge proposals into the graph
            unsafe {
                let _ = merge_proposals::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    graph_idx_gpu.clone().into_tensor_arg(1),
                    graph_dist_gpu.clone().into_tensor_arg(1),
                    prop_idx_gpu.clone().into_tensor_arg(1),
                    prop_dist_gpu.clone().into_tensor_arg(1),
                    prop_count_gpu.clone().into_tensor_arg(1),
                    update_counter_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    MAX_PROPOSALS as u32,
                );
            }

            // download single u32 to check convergence
            let counter_data = update_counter_gpu.clone().read(&client);
            let updates = counter_data[0] as f64;
            let rate = updates / (n * k) as f64;

            if verbose {
                println!(
                    "  Iter {}: {} updates (rate={:.6})",
                    iter + 1,
                    (updates as usize).separate_with_underscores(),
                    rate
                );
            }

            if rate < delta as f64 {
                if verbose {
                    println!("  Converged after {} iterations", iter + 1);
                }
                converged = true;
                break;
            }
        }

        if verbose {
            println!("  NNDescent iterations: {:.2?}", iter_start.elapsed());
        }

        // pull final graph from GPU

        let final_idx = graph_idx_gpu.read(&client);
        let final_dist = graph_dist_gpu.read(&client);

        let pid_mask = 0x7FFFFFFFu32;
        let mut graph = Vec::with_capacity(n * k);
        for i in 0..n * k {
            let pid = (final_idx[i] & pid_mask) as usize;
            graph.push((pid, final_dist[i]));
        }

        if verbose {
            println!("  Total build time: {:.2?}", start.elapsed());
        }

        Self {
            vectors_flat,
            dim,
            n,
            k,
            norms,
            metric,
            graph,
            converged,
            _device: device,
        }
    }

    /// Returns the neighbours of node `i` as a slice of `(index, distance)`.
    pub fn neighbours(&self, i: usize) -> &[(usize, T)] {
        &self.graph[i * self.k..(i + 1) * self.k]
    }

    /// Whether the algorithm converged during construction.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the size of the struct in bytes (CPU side only).
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.graph.capacity() * std::mem::size_of::<(usize, T)>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;
    use faer::Mat;

    #[test]
    fn test_nndescent_gpu_basic() {
        let device = WgpuDevice::DefaultDevice;

        // 20 vectors, 4 dimensions (divisible by LINE_SIZE)
        let data = Mat::from_fn(20, 4, |i, j| ((i * 3 + j) as f32) / 10.0);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(5),
            Some(10),
            Some(0.001),
            Some(0.5),
            42,
            false,
            device,
        );

        assert_eq!(index.graph.len(), 20 * 5);
        for i in 0..20 {
            let nbrs = index.neighbours(i);
            assert_eq!(nbrs.len(), 5);
            // distances should be sorted ascending
            for w in nbrs.windows(2) {
                assert!(w[1].1 >= w[0].1);
            }
            // no self-loops
            for &(pid, _) in nbrs {
                assert_ne!(pid, i);
            }
        }
    }

    #[test]
    fn test_nndescent_gpu_cosine() {
        let device = WgpuDevice::DefaultDevice;

        let data = Mat::from_fn(16, 4, |i, _| (i as f32) + 1.0);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::Cosine,
            Some(3),
            Some(5),
            Some(0.01),
            Some(0.5),
            42,
            false,
            device,
        );

        assert_eq!(index.graph.len(), 16 * 3);
        assert!(!index.norms.is_empty());
    }

    #[test]
    fn test_nndescent_gpu_padded_dim() {
        let device = WgpuDevice::DefaultDevice;

        // dim=3 requires padding to 4
        let data = Mat::from_fn(12, 3, |i, j| (i + j) as f32);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(3),
            Some(5),
            None,
            None,
            42,
            false,
            device,
        );

        assert_eq!(index.dim, 3);
        assert_eq!(index.graph.len(), 12 * 3);
    }
}
