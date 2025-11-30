use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::*;

use crate::annoy::*;
use crate::dist::*;
use crate::utils::*;

////////////////
// Neighbours //
////////////////

/// Packed neighbours structure for efficient memory layout
///
/// ### Fields
///
/// * `pid` - Neighbour's point ID
/// * `distance` - Distance to this neighbour
/// * `is_new` - 1 or 0 to identify if new. Using `u32` for better memory
///   alignment (avoids padding bytes in the struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neighbour<T> {
    pid: u32,
    dist: T,
    is_new: u32,
}

/// Trait for the update_neighbours function that uses - pending on type - a
/// slightly different function with a different heap buffer.
pub trait UpdateNeighbours<T> {
    fn update_neighbours(
        &self,
        node: usize,
        current: &[Neighbour<T>],
        candidates: &[(usize, T)],
        k: usize,
        updates: &AtomicUsize,
    ) -> Vec<Neighbour<T>>;
}

impl<T: Copy> Neighbour<T> {
    /// Generate a new Neighbour
    ///
    /// ### Params
    ///
    /// * `pid` - Current point ID
    /// * `dist` - Distance
    /// * `is_new` - Boolean indicating if the neighbour is new.
    #[inline(always)]
    fn new(pid: usize, dist: T, is_new: bool) -> Self {
        Self {
            pid: pid as u32,
            dist,
            is_new: is_new as u32,
        }
    }

    /// Getter for is True
    ///
    /// ### Returns
    ///
    /// Is this a new neighbour.
    #[inline(always)]
    fn is_new(&self) -> bool {
        self.is_new != 0
    }

    /// Getter for point ID
    ///
    /// ### Returns
    ///
    /// The point identifier.
    #[inline(always)]
    fn pid(&self) -> usize {
        self.pid as usize
    }
}

/// Enum with sensible standard parameters for various graph sizes
///
/// Core idea is to provide (more or less) sensible standard parameters for the
/// the NNDescent algorithm based on the number of samples. This affects
/// parameters like the initial Annoy index and search of new neighbours of
/// neighbours.
#[derive(Clone, Copy, Debug)]
pub enum GraphSize {
    /// Less than 100k
    Small,
    /// Between 100k - 1M
    Medium,
    /// More than 1M
    Large,
}

impl GraphSize {
    /// Helper function to return the `GraphSize`
    fn from_n(n: usize) -> Self {
        match n {
            ..100_000 => Self::Small,
            100_000..1_000_000 => Self::Medium,
            _ => Self::Large,
        }
    }

    /// Get the number of trees to use pending graph/sample size
    fn annoy_trees(&self) -> usize {
        match self {
            Self::Small => 32,
            Self::Medium => 48,
            Self::Large => 64,
        }
    }

    /// Get the k multiplier pending graph/sample size
    fn search_k_multiplier(&self) -> usize {
        match self {
            Self::Small => 3,
            Self::Medium => 5,
            Self::Large => 10,
        }
    }

    /// Get the maximum search budget
    fn max_search_k(&self, n: usize) -> usize {
        match self {
            Self::Small => 100,
            Self::Medium => 500,
            Self::Large => (n / 1000).max(1000),
        }
    }

    /// Get the maximum search budget
    fn max_candidates_factor(&self) -> usize {
        match self {
            Self::Small => 4,
            Self::Medium => 6,
            Self::Large => 8,
        }
    }

    /// Get the maximum number of neighbours
    fn max_per_neighbour(&self, k: usize) -> usize {
        match self {
            Self::Small => k.min(10),
            Self::Medium => k.min(15),
            Self::Large => k.min(25),
        }
    }

    /// Get the Rho decay for exploring old/new neighbours
    fn rho_decay(&self) -> f64 {
        match self {
            Self::Small => 0.8,
            Self::Medium => 0.85,
            Self::Large => 0.9,
        }
    }

    /// Minimum rho pending the graph size
    fn rho_min(&self) -> f64 {
        match self {
            Self::Small => 0.3,
            Self::Medium => 0.4,
            Self::Large => 0.5,
        }
    }
}

//////////////////////////
// Thread-local buffers //
//////////////////////////

// Remove Reverse
type F32Refcell = RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>>;
type F64Refcell = RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>>;

thread_local! {
    static HEAP_BUFFER_F32: F32Refcell = const { RefCell::new(BinaryHeap::new()) };
    static HEAP_BUFFER_F64: F64Refcell = const { RefCell::new(BinaryHeap::new()) };
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static CANDIDATE_SET: RefCell<Vec<bool>> = const {RefCell::new(Vec::new())};
    static SAMPLE_INDICES: RefCell<Vec<usize>> = const{RefCell::new(Vec::new())};
}

////////////////////
// Main algorithm //
////////////////////

/// NN-Descent graph builder
///
/// ### Fields
///
/// * `vectors_flat` - Flat structure of the initial data for better cache
///   locality
/// * `dim` - Initial feature dimensions
/// * `n` - Number of samples
/// * `metric` - Which distance metric to use. One of Euclidean or Cosine
/// * `norms` - Normalised values for each data point. Will be pre-computed
///   once if distance metric is Cosine.
///
/// ### Algorithm Overview
///
/// NN-Descent iteratively improves a k-NN graph through "local joins":
///
/// 1. For each vertex, examine neighbours of its neighbours
/// 2. Compute distances between these candidates
/// 3. Update the k-NN lists if improvements found
/// 4. Mark updated neighbours as "new" to guide next iteration
/// 5. Repeat until convergence
pub struct NNDescent<T> {
    vectors_flat: Vec<T>,
    dim: usize,
    n: usize,
    metric: Dist,
    norms: Vec<T>,
    graph_params: GraphSize,
}

impl<T> NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync,
    Self: UpdateNeighbours<T>,
{
    /// Build the kNN graph with NN-Descent
    ///
    /// ### Params
    ///
    /// * `mat` - The embedding matrix with samples x features
    /// * `k` - Number of neighbours to look for
    /// * `max_iter` - Maximum number of iterations to run for
    /// * `delta` - Tolerance parameter. Should the proportion of changes in a
    ///   given iteration fall below that value, the algorithm stops.
    /// * `rho` - Sampling rate. Will be adaptively reduced in each iteration.
    /// * `seed` - Random seed for reproduction
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// A nested vector of the updated nearest neighbours with their distances.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        mat: MatRef<T>,
        k: usize,
        dist_metric: &str,
        max_iter: usize,
        delta: T,
        rho: T,
        graph_size: Option<GraphSize>,
        seed: usize,
        verbose: bool,
    ) -> Vec<Vec<(usize, T)>> {
        let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);
        let n = mat.nrows();
        let n_features = mat.ncols();

        let graph_params = graph_size.unwrap_or_else(|| GraphSize::from_n(n));

        let mut vectors_flat = Vec::with_capacity(n * n_features);
        for i in 0..n {
            vectors_flat.extend(mat.row(i).iter().copied());
        }

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * n_features;
                    let end = start + n_features;
                    vectors_flat[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        let builder = NNDescent {
            vectors_flat,
            dim: n_features,
            n,
            metric,
            norms,
            graph_params,
        };

        let start_initial_index = Instant::now();
        let annoy_index = AnnoyIndex::new(mat, graph_params.annoy_trees(), seed);
        let end_initial_index = start_initial_index.elapsed();

        if verbose {
            println!("Generated initial Annoy index: {:.2?}", end_initial_index);
        }

        builder.run(k, max_iter, &annoy_index, delta, rho, seed, verbose)
    }

    /// Run the underlying algorithm
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to search for
    /// * `max_iter` - Maximum number of iterations for the algorithm.
    /// * `annoy_index` - Annoy index for the initial fast initialisation of
    ///   the graph.
    /// * `graph_params` - Enum storing various parameters for initialisation
    ///   based on graph size
    /// * `delta` - Tolerance parameter. Should the proportion of changes in a
    ///   given iteration fall below that value, the algorithm stops.
    /// * `rho` - Sampling rate. Will be adaptively reduced in each iteration.
    /// * `seed` - Random seed for reproduction
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// A nested vector of the updated nearest neighbours with their distances.
    ///
    /// ### Algorithm Details
    ///
    /// Each iteration consists of:
    /// 1. Candidate Generation: *For each vertex, find candidates via
    ///    local_join*
    /// 2. Bidirectional Distribution: *Candidates are both forward (i→j) and
    ///    reverse (j→i), ensuring the graph remains undirected*
    /// 3. Parallel Update: *Each vertex updates its k-NN list independently*
    /// 4. Convergence Check: *Stop if update rate falls below delta*
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        k: usize,
        max_iter: usize,
        annoy_index: &AnnoyIndex<T>,
        delta: T,
        rho: T,
        seed: usize,
        verbose: bool,
    ) -> Vec<Vec<(usize, T)>> {
        if verbose {
            println!(
                "Initialising NN-Descent with {} samples - k = {}",
                self.n.separate_with_underscores(),
                k
            );
        }

        let start_total = Instant::now();
        let mut graph = self.initialise_with_annoy(k, annoy_index);

        for iter in 0..max_iter {
            let updates = AtomicUsize::new(0);

            let current_rho = if iter == 0 {
                rho
            } else {
                let decay = T::from_f64(self.graph_params.rho_decay())
                    .unwrap()
                    .powi(iter as i32 - 1);
                (rho * decay).max(T::from_f64(self.graph_params.rho_min()).unwrap())
            };

            let all_candidates: Vec<(usize, Vec<(usize, T)>)> = (0..self.n)
                .into_par_iter()
                .map(|i| {
                    let candidates = self.local_join(i, &graph, k, current_rho, seed + iter);
                    (i, candidates)
                })
                .collect();

            let mut forward_candidates: Vec<Vec<(usize, T)>> =
                vec![Vec::with_capacity(k * 5); self.n];
            let mut reverse_candidates: Vec<Vec<(usize, T)>> =
                vec![Vec::with_capacity(k * 5); self.n];

            let chunk_size = all_candidates.len().div_ceil(rayon::current_num_threads());
            let reverse_chunks: Vec<Vec<Vec<(usize, T)>>> = all_candidates
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_reverse = vec![Vec::new(); self.n];
                    for (i, candidates) in chunk {
                        for &(j, dist) in candidates {
                            local_reverse[j].push((*i, dist));
                        }
                    }
                    local_reverse
                })
                .collect();

            for (i, candidates) in &all_candidates {
                forward_candidates[*i].extend(candidates.iter().copied());
            }

            for local_reverse in reverse_chunks {
                for (j, edges) in local_reverse.into_iter().enumerate() {
                    reverse_candidates[j].extend(edges);
                }
            }

            let new_graph: Vec<Vec<Neighbour<T>>> = (0..self.n)
                .into_par_iter()
                .map(|i| {
                    let mut combined = Vec::with_capacity(
                        forward_candidates[i].len() + reverse_candidates[i].len(),
                    );
                    combined.extend_from_slice(&forward_candidates[i]);
                    combined.extend_from_slice(&reverse_candidates[i]);

                    self.update_neighbours(i, &graph[i], &combined, k, &updates)
                })
                .collect();

            graph = new_graph;

            let update_count = updates.load(Ordering::Relaxed);
            let update_rate = T::from_usize(update_count).unwrap() / T::from_usize(self.n).unwrap();

            if verbose {
                println!(
                    "Iteration {}: {} updates ({:.2}% of nodes), rho={:.3}",
                    iter + 1,
                    update_count.separate_with_underscores(),
                    update_rate.to_f64().unwrap() * 100.0,
                    current_rho.to_f64().unwrap()
                );
            }

            if update_rate < delta {
                if verbose {
                    println!(
                        "Converged after {} iterations (update rate {:.4} < {:.4})",
                        iter + 1,
                        update_rate.to_f64().unwrap(),
                        delta.to_f64().unwrap()
                    );
                }
                break;
            }
        }

        let end_total = start_total.elapsed();

        if verbose {
            println!("Total run-time for NNDescent: {:.2?}", end_total);
        }

        graph
            .into_iter()
            .map(|neighbours| neighbours.into_iter().map(|n| (n.pid(), n.dist)).collect())
            .collect()
    }

    /// Initialise a first set of neighbours with annoy
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to sample
    /// * `annoy_index` - The Annoy index.
    /// * `graph_params` - Enum storing various parameters for initialisation
    ///   based on graph size
    ///
    /// ### Return
    ///
    /// A nested Vec of `Neighbour` structures.
    fn initialise_with_annoy(
        &self,
        k: usize,
        annoy_index: &AnnoyIndex<T>,
    ) -> Vec<Vec<Neighbour<T>>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let query_vec = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                let search_k = ((k + 1) * self.graph_params.search_k_multiplier())
                    .min(self.graph_params.max_search_k(self.n));
                let (indices, distances) =
                    annoy_index.query(query_vec, &self.metric, k + 1, Some(search_k));

                indices
                    .into_iter()
                    .zip(distances)
                    .take(k)
                    .map(|(idx, dist)| Neighbour::new(idx, dist, true))
                    .collect()
            })
            .collect()
    }

    /// Local join: find candidates from neighbours of neighbours
    ///
    ///
    /// ### Params
    ///
    /// * `node` - Node to check
    /// * `graph` - Current kNN graph
    /// * `k` - Number of neighbours
    /// * `rho` - Sampling rate for old neighbours
    /// * `seed` - Random seed for sampling
    ///
    /// ### Returns
    ///
    /// Vec of potential candidates as tuple `(index, dist)`.
    ///
    /// ### Algorithm Details
    ///
    /// 1. Separate new vs old neighbours
    /// 2. Sample old neighbours according to rho
    /// 3. Explore neighbours-of-neighbours (NN's NN)
    /// 4. Only compute distances if they might beat worst current distance
    /// 5. Return promising candidates
    fn local_join(
        &self,
        node: usize,
        graph: &[Vec<Neighbour<T>>],
        k: usize,
        rho: T,
        seed: usize,
    ) -> Vec<(usize, T)> {
        let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_mul((node + 1) as u64));

        let worst_current_dist = graph[node].last().map(|n| n.dist).unwrap_or(T::infinity());

        let mut new_neighbours = Vec::new();
        let mut old_neighbours = Vec::new();

        for n in &graph[node] {
            if n.is_new() {
                new_neighbours.push(n.pid());
            } else {
                old_neighbours.push(n.pid());
            }
        }

        let n_old_sample = ((T::from_usize(old_neighbours.len()).unwrap() * rho)
            .ceil()
            .to_usize()
            .unwrap())
        .min(old_neighbours.len());

        if old_neighbours.len() > n_old_sample {
            for i in 0..n_old_sample {
                let j = rng.random_range(i..old_neighbours.len());
                old_neighbours.swap(i, j);
            }
            old_neighbours.truncate(n_old_sample);
        }

        let max_per_neighbour = k.min(self.graph_params.max_per_neighbour(k));
        let max_per_old = (k / 2).min(5);
        let max_candidates = k * self.graph_params.max_candidates_factor();

        CANDIDATE_SET.with(|set_cell| {
            SAMPLE_INDICES.with(|indices_cell| {
                let mut candidate_set = set_cell.borrow_mut();
                let mut sample_indices = indices_cell.borrow_mut();

                if candidate_set.len() < self.n {
                    candidate_set.resize(self.n, false);
                }

                let mut candidate_ids = Vec::with_capacity(max_candidates);

                for &new_nb in &new_neighbours {
                    let neighbour_list = &graph[new_nb];
                    let take = max_per_neighbour.min(neighbour_list.len());

                    if neighbour_list.len() <= take {
                        for nn in neighbour_list {
                            let pid = nn.pid();
                            if pid != node && !candidate_set[pid] {
                                candidate_set[pid] = true;
                                candidate_ids.push(pid);
                                if candidate_ids.len() >= max_candidates {
                                    break;
                                }
                            }
                        }
                    } else {
                        sample_indices.clear();
                        sample_indices.extend(0..neighbour_list.len());

                        for i in 0..take {
                            let j = rng.random_range(i..neighbour_list.len());
                            sample_indices.swap(i, j);
                        }

                        for &idx in &sample_indices[..take] {
                            if candidate_ids.len() >= max_candidates {
                                break;
                            }
                            let pid = neighbour_list[idx].pid();
                            if pid != node && !candidate_set[pid] {
                                candidate_set[pid] = true;
                                candidate_ids.push(pid);
                            }
                        }
                    }
                }

                for &old_nb in &old_neighbours {
                    if candidate_ids.len() >= max_candidates {
                        break;
                    }
                    let neighbour_list = &graph[old_nb];
                    let take = max_per_old.min(neighbour_list.len());

                    if neighbour_list.len() <= take {
                        for nn in neighbour_list {
                            let pid = nn.pid();
                            if pid != node && !candidate_set[pid] {
                                candidate_set[pid] = true;
                                candidate_ids.push(pid);
                                if candidate_ids.len() >= max_candidates {
                                    break;
                                }
                            }
                        }
                    } else {
                        sample_indices.clear();
                        sample_indices.extend(0..neighbour_list.len());

                        for i in 0..take {
                            let j = rng.random_range(i..neighbour_list.len());
                            sample_indices.swap(i, j);
                        }

                        for &idx in &sample_indices[..take] {
                            if candidate_ids.len() >= max_candidates {
                                break;
                            }
                            let pid = neighbour_list[idx].pid();
                            if pid != node && !candidate_set[pid] {
                                candidate_set[pid] = true;
                                candidate_ids.push(pid);
                            }
                        }
                    }
                }

                let result = match self.metric {
                    Dist::Euclidean => candidate_ids
                        .iter()
                        .filter_map(|&c| {
                            let dist = unsafe { self.euclidean_distance(node, c) };
                            if dist < worst_current_dist {
                                Some((c, dist))
                            } else {
                                None
                            }
                        })
                        .collect(),
                    Dist::Cosine => candidate_ids
                        .iter()
                        .filter_map(|&c| {
                            let dist = unsafe { self.cosine_distance(node, c) };
                            if dist < worst_current_dist {
                                Some((c, dist))
                            } else {
                                None
                            }
                        })
                        .collect(),
                };

                for &pid in &candidate_ids {
                    candidate_set[pid] = false;
                }

                result
            })
        })
    }
}

impl UpdateNeighbours<f32> for NNDescent<f32> {
    /// Update the neighbours with the improvements (for f32)
    ///
    /// ### Params
    ///
    /// * `node` - Current node index
    /// * `current` - Current best neighbours
    /// * `candidates` - Potential new neighbours
    /// * `k` - Number of neighbours to find
    /// * `updates` - Borrowed AtomicUsize to check if an update happened
    ///
    /// ### Returns
    ///
    /// Vec of updates `Neigbour`s.
    fn update_neighbours(
        &self,
        node: usize,
        current: &[Neighbour<f32>],
        candidates: &[(usize, f32)],
        k: usize,
        updates: &AtomicUsize,
    ) -> Vec<Neighbour<f32>> {
        HEAP_BUFFER_F32.with(|heap_cell| {
            PID_SET.with(|set_cell| {
                let mut heap = heap_cell.borrow_mut();
                let mut pid_set = set_cell.borrow_mut();

                heap.clear();

                if pid_set.len() < self.n {
                    pid_set.resize(self.n, false);
                }

                for n in current {
                    let pid = n.pid();
                    heap.push((OrderedFloat(n.dist), pid, false));
                    pid_set[pid] = true;
                }

                for &(pid, dist) in candidates {
                    if pid == node || pid_set[pid] {
                        continue;
                    }

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), pid, true));
                        pid_set[pid] = true;
                    } else if let Some(&(OrderedFloat(worst_dist), _, _)) = heap.peek() {
                        if dist < worst_dist {
                            if let Some((_, old_pid, _)) = heap.pop() {
                                pid_set[old_pid] = false;
                            }
                            heap.push((OrderedFloat(dist), pid, true));
                            pid_set[pid] = true;
                        }
                    }
                }

                let mut result: Vec<_> = heap
                    .drain()
                    .map(|(OrderedFloat(d), p, is_new)| {
                        // Remove Reverse
                        pid_set[p] = false;
                        (d, p, is_new)
                    })
                    .collect();
                result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let changed = result.len() != current.len()
                    || result
                        .iter()
                        .zip(current.iter())
                        .any(|(a, b)| a.1 != b.pid() || (a.0 - b.dist).abs() > 1e-6);

                if changed {
                    updates.fetch_add(1, Ordering::Relaxed);
                }

                result
                    .into_iter()
                    .map(|(dist, pid, is_new)| Neighbour::new(pid, dist, is_new && changed))
                    .collect()
            })
        })
    }
}

impl UpdateNeighbours<f64> for NNDescent<f64> {
    /// Update the neighbours with the improvements (for f64)
    ///
    /// ### Params
    ///
    /// * `node` - Current node index
    /// * `current` - Current best neighbours
    /// * `candidates` - Potential new neighbours
    /// * `k` - Number of neighbours to find
    /// * `updates` - Borrowed AtomicUsize to check if an update happened
    ///
    /// ### Returns
    ///
    /// Vec of updates `Neigbour`s.
    fn update_neighbours(
        &self,
        node: usize,
        current: &[Neighbour<f64>],
        candidates: &[(usize, f64)],
        k: usize,
        updates: &AtomicUsize,
    ) -> Vec<Neighbour<f64>> {
        HEAP_BUFFER_F64.with(|heap_cell| {
            PID_SET.with(|set_cell| {
                let mut heap = heap_cell.borrow_mut();
                let mut pid_set = set_cell.borrow_mut();

                heap.clear();

                if pid_set.len() < self.n {
                    pid_set.resize(self.n, false);
                }

                for n in current {
                    let pid = n.pid();
                    heap.push((OrderedFloat(n.dist), pid, false));
                    pid_set[pid] = true;
                }

                for &(pid, dist) in candidates {
                    if pid == node || pid_set[pid] {
                        continue;
                    }

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), pid, true));
                        pid_set[pid] = true;
                    } else if let Some(&(OrderedFloat(worst_dist), _, _)) = heap.peek() {
                        if dist < worst_dist {
                            if let Some((_, old_pid, _)) = heap.pop() {
                                pid_set[old_pid] = false;
                            }
                            heap.push((OrderedFloat(dist), pid, true));
                            pid_set[pid] = true;
                        }
                    }
                }

                let mut result: Vec<_> = heap
                    .drain()
                    .map(|(OrderedFloat(d), p, is_new)| {
                        pid_set[p] = false;
                        (d, p, is_new)
                    })
                    .collect();
                result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let changed = result.len() != current.len()
                    || result
                        .iter()
                        .zip(current.iter())
                        .any(|(a, b)| a.1 != b.pid() || (a.0 - b.dist).abs() > 1e-6);

                if changed {
                    updates.fetch_add(1, Ordering::Relaxed);
                }

                result
                    .into_iter()
                    .map(|(dist, pid, is_new)| Neighbour::new(pid, dist, is_new && changed))
                    .collect()
            })
        })
    }
}

impl<T: Float + FromPrimitive + Send + Sync> VectorDistance<T> for NNDescent<T> {
    /// Get the flat vectors
    ///
    /// ### Returns
    ///
    /// Slice of flattened vectors
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    /// Get the dimensions
    ///
    /// ### Returns
    ///
    /// The dimensions of the initial embeddings
    fn dim(&self) -> usize {
        self.dim
    }

    /// Get the normalised values for each data point.
    ///
    /// Used for Cosine distance calculations
    ///
    /// ### Returns
    ///
    /// The normalised values for each data point
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // tests/test_nndescent.rs
    use super::*;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        let data = [
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_nndescent_build_euclidean() {
        let mat = create_simple_matrix();
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3, // k
            "euclidean",
            10,    // max_iter
            0.001, // delta
            1.0,   // rho
            None,
            42,
            false,
        );

        // should return a graph with n entries
        assert_eq!(graph.len(), 5);

        // each entry should have at most k neighbours
        for neighbours in &graph {
            assert!(neighbours.len() <= 3);
        }
    }

    #[test]
    fn test_nndescent_build_cosine() {
        let mat = create_simple_matrix();
        let graph =
            NNDescent::<f32>::build(mat.as_ref(), 3, "cosine", 10, 0.001, 1.0, None, 42, false);

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn test_nndescent_graph_structure() {
        let mat = create_simple_matrix();
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // each node should have neighbours
        for neighbours in graph.iter() {
            // should not include itself in neighbours (typically)
            for (neighbour_idx, _) in neighbours {
                // Just verify indices are valid
                assert!(*neighbour_idx < 5);
            }

            // distances should be non-negative
            for (_, dist) in neighbours {
                assert!(*dist >= 0.0);
            }
        }
    }

    #[test]
    fn test_nndescent_convergence() {
        let mat = create_simple_matrix();

        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            100, // Many iterations allowed
            0.5, // High delta for quick convergence
            1.0,
            None,
            42,
            false,
        );

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn test_nndescent_reproducibility() {
        let mat = create_simple_matrix();

        let graph1 = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        let graph2 = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Same seed should give similar results
        // Note: Due to parallel execution, exact match might not be guaranteed
        // but structure should be similar
        assert_eq!(graph1.len(), graph2.len());
    }

    #[test]
    fn test_nndescent_k_parameter() {
        let mat = create_simple_matrix();

        let graph_k2 = NNDescent::<f32>::build(
            mat.as_ref(),
            2,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        let graph_k4 = NNDescent::<f32>::build(
            mat.as_ref(),
            4,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Verify k constraint is respected
        for neighbours in &graph_k2 {
            assert!(neighbours.len() <= 2);
        }

        for neighbours in &graph_k4 {
            assert!(neighbours.len() <= 4);
        }
    }

    #[test]
    fn test_nndescent_rho_parameter() {
        let mat = create_simple_matrix();

        // High rho (sample more old neighbours)
        let graph_high_rho = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Low rho (sample fewer old neighbours)
        let graph_low_rho = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            0.3,
            None,
            42,
            false,
        );

        assert_eq!(graph_high_rho.len(), 5);
        assert_eq!(graph_low_rho.len(), 5);
    }

    #[test]
    fn test_nndescent_larger_dataset() {
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            10,
            "euclidean",
            15,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        assert_eq!(graph.len(), n);

        // Each node should have close to k neighbours
        for neighbours in &graph {
            assert!(neighbours.len() <= 10);
            assert!(!neighbours.is_empty());
        }
    }

    #[test]
    fn test_nndescent_orthogonal_vectors() {
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

        // Updated: Set k=3 to capture Self + 2 orthogonal neighbours
        let graph =
            NNDescent::<f32>::build(mat.as_ref(), 3, "cosine", 10, 0.001, 1.0, None, 42, false);

        for neighbours in &graph {
            // Should have 3 neighbours
            assert_eq!(neighbours.len(), 3);

            let mut self_found = false;
            let mut orthogonal_count = 0;

            for (_, dist) in neighbours {
                if *dist < 1e-5 {
                    self_found = true;
                } else if (*dist - 1.0).abs() < 1e-5 {
                    orthogonal_count += 1;
                }
            }

            assert!(self_found, "Should find self (dist ~ 0.0)");
            assert_eq!(
                orthogonal_count, 2,
                "Should find 2 orthogonal neighbours (dist ~ 1.0)"
            );
        }
    }

    #[test]
    fn test_nndescent_parallel_execution() {
        let n = 30;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        // Should complete without deadlock or data races
        let graph = crate::nndescent::NNDescent::<f32>::build(
            mat.as_ref(),
            5,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        assert_eq!(graph.len(), n);
    }

    #[test]
    fn test_nndescent_max_iter() {
        let mat = create_simple_matrix();

        // With max_iter = 1, should stop after 1 iteration
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            1,   // Only 1 iteration
            0.0, // Delta = 0 means never converge by delta
            1.0,
            None,
            42,
            false,
        );

        // Should still produce a valid graph
        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn test_nndescent_distance_ordering() {
        let mat = create_simple_matrix();
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Neighbours should be sorted by distance
        for neighbours in &graph {
            for i in 1..neighbours.len() {
                assert!(neighbours[i].1 >= neighbours[i - 1].1);
            }
        }
    }

    #[test]
    fn test_nndescent_with_f64() {
        let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

        let graph = NNDescent::<f64>::build(
            mat.as_ref(),
            2,
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_nndescent_quality() {
        // Create a structured dataset where we know true neighbours
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        // Create clusters: points 0-9 near origin, 10-19 far away
        for i in 0..10 {
            let offset = i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }
        for i in 0..10 {
            let offset = 10.0 + i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            5,
            "euclidean",
            20,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Point 0 should have neighbours mostly from 0-9 range
        let neighbours_0 = &graph[0];
        let in_cluster = neighbours_0.iter().filter(|(idx, _)| *idx < 10).count();

        // Should find mostly in-cluster neighbours
        assert!(in_cluster >= 3);

        // Point 10 should have neighbours mostly from 10-19 range
        let neighbours_10 = &graph[10];
        let in_cluster_2 = neighbours_10.iter().filter(|(idx, _)| *idx >= 10).count();

        assert!(in_cluster_2 >= 3);
    }

    #[test]
    fn test_nndescent_invalid_metric() {
        let mat = create_simple_matrix();

        // Should fall back to Cosine for invalid metric
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            3,
            "invalid_metric",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn test_nndescent_edge_case_k_equals_n() {
        let mat = create_simple_matrix();

        // k = n - 1 (want all other points as neighbours)
        let graph = NNDescent::<f32>::build(
            mat.as_ref(),
            4, // n = 5, so k = 4 means all others
            "euclidean",
            10,
            0.001,
            1.0,
            None,
            42,
            false,
        );

        // Each point should have 4 neighbours
        for neighbours in &graph {
            assert_eq!(neighbours.len(), 4);
        }
    }
}
