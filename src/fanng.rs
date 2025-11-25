use faer::MatRef;
use num_traits::Float;
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::dist::VectorDistance;
use crate::utils::*;

/// Structure to contain FANNG parameters
///
/// ### Fields
///
/// * `max_degree` - Maximum edges per vertex after truncation. Paper recommends
///   25-32 for SIFT data. Lower values (15-20) are faster for construction and
///   queries but may reduce recall. Higher values (30-40) improve recall at the
///   cost of memory and query time.
/// * `batch_size` - Number of random vertex pairs processed in parallel during
///   traverse-add. Larger batches (50-100) improve parallelism but require more
///   memory. Smaller batches (10-20) are more sequential but use less memory.
///   This will not affect graph quality much.
/// * `traverse_add_multiplier` - Controls traverse-add iterations as
///   `multiplier × n` where n is dataset size.
/// * `refinement_neighbour_no` - Approximate neighbours found per vertex during
///   parallel refinement. Higher -> better recall.
/// * `refinement_max_calc` - Distance calculation budget per vertex during
///   refinement neighbour search. Higher values find better neighbours but
///   increase construction time. Should be roughly half of
///   `refinement_neighbour_no`.
#[derive(Clone, Debug)]
pub struct FanngParams {
    pub max_degree: usize,
    pub batch_size: usize,
    pub traverse_add_multiplier: usize,
    pub refinement_neighbour_no: usize,
    pub refinement_max_calc: usize,
}

impl FanngParams {
    /// Generate new Fanng parameters
    ///
    /// ### Params
    ///
    /// * `max_degree` -  Maximum edges per vertex after truncation.
    /// * `batch_size` - Number of random vertex pairs processed in parallel
    ///   during traverse-add.
    /// * `traverse_add_multiplier` - Controls traverse-add iterations as
    ///   `multiplier × n` where n is dataset size.
    /// * `refinement_neighbour_no` - Approximate neighbours found per vertex
    ///   during parallel refinement. Higher -> better recall.
    /// * `refinement_max_calc` - Distance calculation budget per vertex during
    ///   refinement neighbour search.
    pub fn new(
        max_degree: usize,
        batch_size: usize,
        traverse_add_multiplier: usize,
        refinement_neighbour_no: usize,
        refinement_max_calc: usize,
    ) -> Self {
        Self {
            max_degree,
            batch_size,
            traverse_add_multiplier,
            refinement_neighbour_no,
            refinement_max_calc,
        }
    }

    /// Fast parameters for testing (10x faster build, slightly lower recall)
    ///
    /// Suitable for unit tests and rapid prototyping.
    pub fn fast() -> Self {
        Self {
            max_degree: 15,
            batch_size: 20,
            traverse_add_multiplier: 5,
            refinement_neighbour_no: 100,
            refinement_max_calc: 50,
        }
    }

    /// Balanced parameters (good quality, reasonable build time)
    ///
    /// Recommended starting point for most applications.
    pub fn balanced() -> Self {
        Self {
            max_degree: 25,
            batch_size: 50,
            traverse_add_multiplier: 25,
            refinement_neighbour_no: 500,
            refinement_max_calc: 250,
        }
    }
}

impl Default for FanngParams {
    /// Production-quality parameters as recommended in the paper
    ///
    /// Based on Harwood & Drummond (2016):
    /// - max_degree: 30 (optimal for SIFT, section 3.7)
    /// - traverse_add_multiplier: 50 (50N iterations, section 3.6)
    /// - refinement_neighbour_no: 1000 (section 3.6)
    /// - refinement_max_calc: 500 (maintains 2:1 ratio)
    /// - batch_size: 100 (efficient parallelism)
    fn default() -> Self {
        Self {
            max_degree: 30,
            batch_size: 100,
            traverse_add_multiplier: 50,
            refinement_neighbour_no: 1000,
            refinement_max_calc: 500,
        }
    }
}

/// Structure fo the FANNG algorithm
pub struct Fanng<T>
where
    T: Float,
{
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub norms: Vec<T>,
    pub dist: Dist,
    pub graph: Vec<Vec<usize>>,
    pub start_vertices: Vec<usize>,
    pub max_degree: usize,
}

impl<T: Float> VectorDistance<T> for Fanng<T> {
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

    /// Get the normalised values for each data point
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

impl<T> Fanng<T>
where
    T: Float + Send + Sync,
{
    /// Initialise a new FANNG index
    ///
    /// ### Params
    ///
    /// * `mat` - Embedding matrix (rows = samples, cols = features)
    /// * `dist_metric` - "euclidean" or "cosine"
    /// * `max_degree` - Maximum number of edges per node
    /// * `batch_size` - Batch size for the AddTraverse phase of the index
    ///   generation.
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Initialised self.
    ///
    /// ### Implementation Details
    ///
    /// **Phase 1: Random Initialisation**
    /// - Each vertex gets 8 random outgoing edges
    ///
    /// **Phase 2: Traverse-Add (50N iterations)**
    /// - Randomly select vertex pairs (v1, v2)
    /// - Attempt downhill search from v1 to v2
    /// - If search fails, add edge from last reached vertex to v2 (with occlusion)
    /// - Continues until 90% success rate
    ///
    /// **Phase 3: Parallel Refinement**
    /// - For each vertex, find ~1000 approximate neighbours
    /// - Rebuild edges using occlusion rule on neighbour list
    ///
    /// **Phase 4: Truncation**
    /// - Limit each vertex to max_degree edges (keeping nearest)
    ///
    /// **Phase 5: Find Start Vertex**
    /// - Select vertex nearest to dataset centroid
    pub fn new(
        mat: MatRef<T>,
        dist_metric: &str,
        fanng_params: &FanngParams,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);
        let n = mat.nrows();
        let dim = mat.ncols();

        // Flatten matrix for cache locality
        let mut vectors_flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            vectors_flat.extend(mat.row(i).iter().copied());
        }

        // Compute norms for cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
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

        let mut fanng = Self {
            vectors_flat,
            dim,
            norms,
            dist: metric,
            graph: vec![Vec::new(); n],
            start_vertices: Vec::new(),
            max_degree: fanng_params.max_degree,
        };

        if verbose {
            println!("1.) Initialising a first set of random edges.")
        }

        fanng.random_init(8, seed);

        if verbose {
            println!(
                "2.) Using the TraverseAdd algorithm in batches of {}",
                fanng_params.batch_size
            )
        }

        fanng.batch_traverse_add(
            n * fanng_params.traverse_add_multiplier,
            fanng_params.batch_size,
            seed,
        );

        fanng.parallel_refinement(
            fanng_params.refinement_neighbour_no,
            fanng_params.refinement_max_calc,
        );

        fanng.truncate_graph();

        fanng.find_start_vertex();

        fanng
    }

    /// K-NN search
    ///
    /// Search the generated index
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `k` - Number of nearest neighbours to return
    /// * `max_calcs` - Maximum number of distance calculations before terminating
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) of k nearest neighbours
    ///
    /// ### Algorithm Details
    ///
    /// Uses a priority queue to explore vertices in order of distance to query:
    ///
    /// 1. Start at centroid vertex
    /// 2. Pop closest unexplored vertex from queue
    /// 3. Explore its edges, computing distances to neighbours
    /// 4. Add neighbours to queue with their distances as priority
    /// 5. Track k nearest neighbours seen so far
    /// 6. Continue until max_calcs budget exhausted
    ///
    /// The backtracking behaviour comes from re-adding vertices to the queue
    /// with their remaining unexplored edges, using the vertex's distance as
    /// priority. This ensures closer vertices are explored more thoroughly.
    /// K-NN search with multiple start vertices
    ///
    /// Searches from multiple entry points to handle disconnected graph components.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `k` - Number of nearest neighbours to return
    /// * `max_calcs` - Maximum number of distance calculations before terminating
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) of k nearest neighbours
    ///
    /// ### Algorithm Details
    ///
    /// 1. Initializes search from all start vertices simultaneously
    /// 2. Uses priority queue to explore vertices in order of distance
    /// 3. Tracks k nearest neighbours across all search paths
    /// 4. Continues until budget exhausted
    pub fn search_k(&self, query: &[T], k: usize, max_calcs: usize) -> (Vec<usize>, Vec<T>) {
        let mut visited = FxHashSet::default();
        let mut calcs = 0;

        // Candidates: min-heap (explore closest first)
        let mut candidates = BinaryHeap::new();

        // Result: MAX-heap (so we can pop the FARTHEST when > k)
        let mut result = BinaryHeap::new();

        let query_norm = if self.dist == Dist::Cosine {
            query
                .iter()
                .map(|x| *x * *x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        // Initialize from ALL start vertices
        for &start_vertex in &self.start_vertices {
            if visited.contains(&start_vertex) {
                continue;
            }

            let start_dist = self.compute_query_distance(query, start_vertex, query_norm);
            candidates.push(Reverse((OrderedFloat(start_dist), start_vertex)));
            result.push((OrderedFloat(start_dist), start_vertex));
            visited.insert(start_vertex);
            calcs += 1;

            if calcs >= max_calcs {
                break;
            }
        }

        let mut worst_dist = OrderedFloat(T::infinity());
        if result.len() >= k {
            if let Some(&(w, _)) = result.peek() {
                worst_dist = w;
            }
        }

        while let Some(Reverse((OrderedFloat(cand_dist), current))) = candidates.pop() {
            if calcs >= max_calcs {
                break;
            }

            // Prune: if candidate is farther than worst k-th result, skip
            if result.len() >= k && cand_dist > worst_dist.0 {
                continue;
            }

            // Explore ALL neighbours
            for &neighbour in &self.graph[current] {
                if visited.contains(&neighbour) {
                    continue;
                }
                visited.insert(neighbour);
                calcs += 1;

                let dist = self.compute_query_distance(query, neighbour, query_norm);

                // Add if closer than worst or we don't have k yet
                if result.len() < k || dist < worst_dist.0 {
                    candidates.push(Reverse((OrderedFloat(dist), neighbour)));
                    result.push((OrderedFloat(dist), neighbour));

                    if result.len() > k {
                        result.pop();
                    }

                    if result.len() >= k {
                        if let Some(&(w, _)) = result.peek() {
                            worst_dist = w;
                        }
                    }
                }

                if calcs >= max_calcs {
                    break;
                }
            }
        }

        // Convert to sorted vector
        let sorted_result: Vec<_> = result.into_sorted_vec();

        let (indices, distances): (Vec<_>, Vec<_>) = sorted_result
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (idx, dist))
            .unzip();

        (indices, distances)
    }

    /// Initialise graph with random edges
    ///
    /// Creates initial graph structure by giving each vertex k random outgoing
    /// edges to other vertices. This provides a starting point for the
    /// traverse-add refinement phase.
    ///
    /// ### Params
    ///
    /// * `k` - Number of random edges per vertex (typically 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Implementation Details
    ///
    /// Uses parallel iteration with per-vertex RNG seeding to ensure:
    /// * Reproducibility across runs
    /// * Thread-safe parallel execution
    /// * No duplicate edges from same vertex
    fn random_init(&mut self, k: usize, seed: usize) {
        let n = self.graph.len();

        self.graph
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, edges)| {
                let mut local_rng = SmallRng::seed_from_u64((seed + 2 * i) as u64);
                let mut targets = FxHashSet::default();

                while targets.len() < k.min(n - 1) {
                    let j = local_rng.random_range(0..n);
                    if j != i {
                        targets.insert(j);
                    }
                }

                *edges = targets.into_iter().collect();
            })
    }

    /// Traverse add function (in batches)
    ///
    /// Instead of using a purely sequential approach, this one generates new
    /// edges in batches and then tests them.
    ///
    /// ### Params
    ///
    /// * `total_iters` - Total number of iterations to run the algorithm for
    ///   (typically `50 * n`)
    /// * `batch_size` - Size of the batches to test.
    /// * `seed` - Random seed for reproducibility.
    ///
    /// ### Algorithm Details
    ///
    /// 1. Generate batch of random vertex pairs (v1, v2)
    /// 2. In parallel, attempt downhill search from v1 to v2
    /// 3. For failed searches, add edge from reached vertex to target with
    ///    occlusion
    /// 4. Continue until 90% success rate or total_iters reached
    ///
    /// The 90% success criterion indicates the graph has sufficient
    /// connectivity for most searches to succeed via downhill paths.
    fn batch_traverse_add(&mut self, total_iters: usize, batch_size: usize, seed: usize) {
        let n = self.graph.len();
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut successes = 0;
        let mut total = 0;

        while successes as f64 / (total.max(1) as f64) < 0.9 && total < total_iters {
            let pairs = (0..batch_size)
                .map(|_| {
                    let v1 = rng.random_range(0..n);
                    let v2 = rng.random_range(0..n);
                    (v1, v2)
                })
                .collect::<Vec<(usize, usize)>>();

            // execute this in parallel
            let res: Vec<(usize, usize, usize)> = pairs
                .par_iter()
                .map(|&(v1, v2)| {
                    let reached = self.naive_downhill_search(v1, v2);
                    (v1, v2, reached)
                })
                .collect();

            // this needs to be sequential
            for (_, v2, reached) in res {
                total += 1;
                if reached == v2 {
                    successes += 1;
                } else {
                    self.add_edge_with_occlusion(reached, v2);
                }
            }
        }
    }

    /// Graph refinement
    ///
    /// Rebuilds all edges in parallel using approximate nearest neighbour
    /// lists. This phase improves graph quality by replacing random/heuristic
    /// edges with edges to actual nearby neighbours.
    ///
    /// ### Params
    ///
    /// * `neighbour_count` - Number of approximate neighbours to find per
    ///   vertex (typically 1000)
    /// * `max_calcs` - Distance calculation budget for finding each neighbour
    ///   list (typically 500)
    ///
    /// ### Algorithm Details
    ///
    /// For each vertex v in parallel:
    /// 1. Use current graph to find ~neighbour_count approximate neighbours of
    ///    v
    /// 2. Rebuild v's outgoing edges from this neighbour list using occlusion
    ///    rule
    /// 3. Replace v's edges with the new edge set
    ///
    /// This phase is fully parallelisable since each vertex's edges are rebuilt
    /// independently using the (read-only) current graph state.
    fn parallel_refinement(&mut self, neighbour_count: usize, max_calcs: usize) {
        let n = self.graph.len();

        let new_graphs: Vec<Vec<usize>> = (0..n)
            .into_par_iter()
            .map(|i| {
                // find approximate neighbors using current graph
                let neighbors = self.find_approx_neighbour(i, neighbour_count, max_calcs);

                // rebuild edges with occlusion rule
                self.rebuild_edges_from_neighbours(i, &neighbors)
            })
            .collect();

        self.graph = new_graphs;
    }

    /// Truncate graph to maximum degree
    ///
    /// Limits each vertex to at most max_degree outgoing edges, keeping the
    /// nearest neighbours. This improves query efficiency by reducing the
    /// number of edges explored per vertex during search.
    ///
    /// ### Implementation Details
    ///
    /// For vertices exceeding max_degree:
    /// 1. Compute distances to all neighbours
    /// 2. Sort neighbours by distance
    /// 3. Keep only the max_degree nearest neighbours
    fn truncate_graph(&mut self) {
        for i in 0..self.graph.len() {
            if self.graph[i].len() > self.max_degree {
                let vertex_idx = i;
                // index shenanigans to avoid borrow checker unhappiness...
                let mut edges_with_dist: Vec<_> = self.graph[i]
                    .iter()
                    .map(|&to| (to, unsafe { self.distance(vertex_idx, to) }))
                    .collect();
                edges_with_dist.sort_by_key(|&(_, dist)| OrderedFloat(dist));
                self.graph[i] = edges_with_dist
                    .into_iter()
                    .take(self.max_degree)
                    .map(|(to, _)| to)
                    .collect();
            }
        }
    }

    /// Find multiple start vertices for robust search
    ///
    /// Selects diverse entry points to handle potentially disconnected graph
    /// components or clustered data. Uses a k-means++ style initialisation:
    ///
    /// 1. First vertex is nearest to centroid
    /// 2. Subsequent vertices are chosen to be far from already-selected
    ///    vertices
    ///
    /// ### Algorithm Details
    ///
    /// Having multiple start vertices significantly improves recall on
    /// clustered data where the graph may have weak inter-cluster connectivity.
    fn find_start_vertex(&mut self, num_starts: usize) {
        let n = self.graph.len();
        let num_starts = num_starts.min(n);

        // Compute centroid
        let mut centroid = vec![T::zero(); self.dim];
        for i in 0..n {
            for j in 0..self.dim {
                centroid[j] = centroid[j] + self.vectors_flat[i * self.dim + j];
            }
        }
        let n_float = T::from(n).unwrap();
        for c in &mut centroid {
            *c = *c / n_float;
        }

        // Find first start vertex (nearest to centroid)
        let mut best = 0;
        let mut best_dist = T::infinity();
        for i in 0..n {
            let mut dist = T::zero();
            for j in 0..self.dim {
                let diff = self.vectors_flat[i * self.dim + j] - centroid[j];
                dist = dist + diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }

        self.start_vertices = vec![best];

        // Find additional start vertices using farthest-first traversal
        // (similar to k-means++ initialization)
        for _ in 1..num_starts {
            let mut best_vertex = 0;
            let mut best_min_dist = T::zero();

            // For each candidate vertex, find its distance to nearest start vertex
            for i in 0..n {
                if self.start_vertices.contains(&i) {
                    continue;
                }

                // Find minimum distance to any existing start vertex
                let mut min_dist = T::infinity();
                for &start in &self.start_vertices {
                    let dist = unsafe { self.distance(i, start) };
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }

                // Keep the vertex that is farthest from its nearest start vertex
                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_vertex = i;
                }
            }

            self.start_vertices.push(best_vertex);
        }
    }

    /// Naive downhill search
    ///
    /// Simple greedy search that always follows the edge leading to the nearest
    /// neighbour of the target. Terminates when no neighbour is closer than the
    /// current vertex (local minimum).
    ///
    /// ### Params
    ///
    /// * `start` - Starting vertex index
    /// * `target` - Target vertex index
    ///
    /// ### Returns
    ///
    /// Index of the vertex reached (target if successful, otherwise local
    /// minimum)
    ///
    /// ### Implementation Details
    ///
    /// At each vertex:
    /// 1. Evaluate distance from all neighbours to target
    /// 2. Move to neighbour with minimum distance to target
    /// 3. If no neighbour is closer than current, return current
    ///
    /// This is used during graph construction to identify missing edges.
    fn naive_downhill_search(&self, start: usize, target: usize) -> usize {
        let mut current = start;

        loop {
            let mut best = current;
            let mut best_dist = unsafe { self.distance(current, target) };

            for &neighbor in &self.graph[current] {
                let dist = unsafe { self.distance(neighbor, target) };
                if dist < best_dist {
                    best = neighbor;
                    best_dist = dist;
                }
            }

            if best == current {
                return current;
            }
            current = best;
        }
    }

    /// Add edge with occlusion pruning
    ///
    /// Adds an edge from 'from' to 'to' if it's not occluded by existing edges,
    /// and removes any existing edges that become occluded by the new edge.
    ///
    /// ### Params
    ///
    /// * `from` - Source vertex index
    /// * `to` - Target vertex index
    ///
    /// ### Occlusion Rule
    ///
    /// Edge (from, to) is occluded by existing edge (from, existing) if:
    /// * d(from, existing) < d(from, to) AND
    /// * d(existing, to) < d(from, to)
    ///
    /// This ensures the new edge is unnecessary because 'existing' provides a
    /// shorter path to 'to' (or nearby).
    ///
    /// ### Implementation Details
    ///
    /// 1. Check if any existing edge occludes the new edge (if so, return)
    /// 2. Compute distances for all existing edges
    /// 3. Rebuild edge list, keeping only edges not occluded by new edge
    /// 4. Add new edge
    fn add_edge_with_occlusion(&mut self, from: usize, to: usize) {
        let dist_from_to = unsafe { self.distance(from, to) };

        // Check if new edge is occluded by existing edges
        for &existing_to in &self.graph[from] {
            let dist_from_existing = unsafe { self.distance(from, existing_to) };
            let dist_existing_to = unsafe { self.distance(existing_to, to) };

            if dist_from_existing < dist_from_to && dist_existing_to < dist_from_to {
                return; // Occluded
            }
        }

        // calculate temporary the distances and store them to avoid
        // the borrow checker...
        let temp_distances: Vec<(usize, T, T)> = self.graph[from]
            .iter()
            .map(|&existing_to| {
                let dist_from_existing = unsafe { self.distance(from, existing_to) };
                let dist_to_existing = unsafe { self.distance(to, existing_to) };
                (existing_to, dist_from_existing, dist_to_existing)
            })
            .collect();

        // clear the vector
        self.graph[from].clear();

        // rebuild the edges and keep only edges that are not occluded by the
        // new edges
        for (existing_to, dist_from_existing, dist_to_existing) in temp_distances {
            let is_occluded_by_new_edge =
                dist_from_to < dist_from_existing && dist_to_existing < dist_from_existing;

            if !is_occluded_by_new_edge {
                self.graph[from].push(existing_to);
            }
        }

        self.graph[from].push(to);
    }

    /// Find approximate neighbours using current graph
    ///
    /// Performs a search starting from 'vertex' using the current graph
    /// structure to find approximate nearest neighbours of that vertex.
    ///
    /// ### Params
    ///
    /// * `vertex` - Vertex index to find neighbours for
    /// * `k` - Number of neighbours to find
    /// * `max_calcs` - Distance calculation budget
    ///
    /// ### Returns
    ///
    /// Vector of up to k approximate neighbour indices, sorted by distance
    ///
    /// ### Implementation Details
    ///
    /// Uses the same backtracking search as query search, but with the vertex
    /// itself as both start point and query target. This efficiently explores
    /// the local neighbourhood around the vertex.
    fn find_approx_neighbour(&self, vertex: usize, k: usize, max_calcs: usize) -> Vec<usize> {
        let mut pq = BinaryHeap::new();
        let mut visited = FxHashSet::default();
        let mut calcs = 0;

        pq.push((Reverse(OrderedFloat(T::zero())), vertex, 0));
        visited.insert(vertex);

        let mut nearest = BinaryHeap::new();

        while !pq.is_empty() && calcs < max_calcs {
            let (Reverse(OrderedFloat(_)), current, edge_idx) = pq.pop().unwrap();

            if edge_idx < self.graph[current].len() {
                let neighbour = self.graph[current][edge_idx];

                if !visited.contains(&neighbour) {
                    visited.insert(neighbour);
                    calcs += 1;

                    let dist = unsafe { self.distance(vertex, neighbour) };
                    nearest.push(Reverse((OrderedFloat(dist), neighbour)));

                    if nearest.len() > k {
                        nearest.pop();
                    }

                    // add neighbor's edges to explore
                    pq.push((Reverse(OrderedFloat(dist)), neighbour, 0));
                }

                // re-add current vertex with next edge
                pq.push((Reverse(OrderedFloat(T::zero())), current, edge_idx + 1));
            }
        }

        nearest
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((_, idx))| idx)
            .collect()
    }

    /// Rebuild edges from neighbour list using occlusion
    ///
    /// Constructs a new edge list for 'vertex' by applying the occlusion rule
    /// to a provided list of candidate neighbours.
    ///
    /// ### Params
    ///
    /// * `vertex` - Vertex index to build edges for
    /// * `neighbours` - Candidate neighbour indices (typically from approximate
    ///   search)
    ///
    /// ### Returns
    ///
    /// Vector of vertex indices representing the new edge list
    ///
    /// ### Algorithm Details
    ///
    /// For each candidate neighbour in order:
    ///
    /// 1. Skip if it's the vertex itself
    /// 2. Check if occluded by any already-added edge
    /// 3. If not occluded, add to edge list
    ///
    /// This produces a minimal edge set that efficiently spans the local
    /// neighbourhood.
    fn rebuild_edges_from_neighbours(&self, vertex: usize, neighbours: &[usize]) -> Vec<usize> {
        let mut edges = Vec::new();

        for &neighbour in neighbours {
            if neighbour == vertex {
                continue;
            }

            let dist_v_n = unsafe { self.distance(vertex, neighbour) };
            let mut occluded = false;

            // Check if occluded by existing edges
            for &existing in &edges {
                let dist_v_e = unsafe { self.distance(vertex, existing) };
                let dist_e_n = unsafe { self.distance(existing, neighbour) };

                if dist_v_e < dist_v_n && dist_e_n < dist_v_n {
                    occluded = true;
                    break;
                }
            }

            if !occluded {
                edges.push(neighbour);
            }
        }

        edges
    }

    /// Distance between two dataset vectors
    ///
    /// This is added via the `VectorDistance` trait
    ///
    /// ### Params
    ///
    /// * `i` - Point i
    /// * `j` - Point j
    ///
    /// ### Returns
    ///
    /// The desired distance
    ///
    /// ### Safety
    ///
    /// Does not do bound checks for performance
    #[inline(always)]
    unsafe fn distance(&self, i: usize, j: usize) -> T {
        match self.dist {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    /// Compute distance between query and database vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `idx` - Database vector index
    /// * `query_norm` - Precomputed norm of query (for cosine)
    ///
    /// ### Returns
    ///
    /// Distance value:
    /// * Euclidean: squared Euclidean distance
    /// * Cosine: 1 - (dot product / (norm_query × norm_idx))
    ///
    /// ### Implementation Details
    ///
    /// Uses unsafe pointer arithmetic with 4-way loop unrolling for
    /// SIMD-friendly code. The compiler can often vectorise these patterns.
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.dist {
            Dist::Euclidean => {
                let ptr_idx = unsafe { self.vectors_flat.as_ptr().add(idx * self.dim) };
                let ptr_query = query.as_ptr();

                let mut sum = T::zero();
                let mut k = 0;

                unsafe {
                    while k + 4 <= self.dim {
                        let d0 = *ptr_idx.add(k) - *ptr_query.add(k);
                        let d1 = *ptr_idx.add(k + 1) - *ptr_query.add(k + 1);
                        let d2 = *ptr_idx.add(k + 2) - *ptr_query.add(k + 2);
                        let d3 = *ptr_idx.add(k + 3) - *ptr_query.add(k + 3);

                        sum = sum + d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                        k += 4;
                    }

                    while k < self.dim {
                        let diff = *ptr_idx.add(k) - *ptr_query.add(k);
                        sum = sum + diff * diff;
                        k += 1;
                    }
                }

                sum
            }
            Dist::Cosine => {
                let ptr_idx = unsafe { self.vectors_flat.as_ptr().add(idx * self.dim) };
                let ptr_query = query.as_ptr();

                let mut dot = T::zero();
                let mut k = 0;

                unsafe {
                    while k + 4 <= self.dim {
                        dot = dot
                            + *ptr_idx.add(k) * *ptr_query.add(k)
                            + *ptr_idx.add(k + 1) * *ptr_query.add(k + 1)
                            + *ptr_idx.add(k + 2) * *ptr_query.add(k + 2)
                            + *ptr_idx.add(k + 3) * *ptr_query.add(k + 3);
                        k += 4;
                    }

                    while k < self.dim {
                        dot = dot + *ptr_idx.add(k) * *ptr_query.add(k);
                        k += 1;
                    }

                    T::one() - (dot / (query_norm * self.norms[idx]))
                }
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use num_traits::FromPrimitive;

    // Fast test parameters
    fn test_params() -> FanngParams {
        FanngParams::new(
            15,  // max_degree
            20,  // batch_size
            5,   // traverse_add_multiplier (was 50)
            100, // refinement_neighbour_no (was 1000)
            50,  // refinement_max_calc (was 500)
        )
    }

    // Fast test parameters
    fn test_params_aggressive() -> FanngParams {
        FanngParams::new(
            15,  // max_degree
            25,  // batch_size
            10,  // traverse_add_multiplier (was 50)
            100, // refinement_neighbour_no (was 1000)
            50,  // refinement_max_calc (was 500)
        )
    }

    fn create_distributed_data<T: Float + FromPrimitive>() -> Mat<T> {
        use rand::Rng;
        use rand::SeedableRng;

        let n = 200; // Reduced from 500
        let dims = 20; // Reduced from 25
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut data = Vec::new();

        for _ in 0..n {
            for _ in 0..dims {
                let val = rng.random_range(0.0..10.0);
                data.push(T::from_f64(val).unwrap());
            }
        }

        Mat::from_fn(n, dims, |i, j| data[i * dims + j])
    }

    #[test]
    fn test_fanng_creation() {
        let mat = create_distributed_data::<f32>();
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        assert_eq!(index.dim(), 20);
        assert_eq!(index.graph.len(), 200);
        assert_eq!(index.max_degree, 15);
    }

    #[test]
    fn test_fanng_with_aggressive_params() {
        let mat = create_distributed_data::<f32>();
        let index = Fanng::new(
            mat.as_ref(),
            "euclidean",
            &test_params_aggressive(),
            42,
            false,
        );

        let query = mat.row(0).iter().cloned().collect::<Vec<f32>>();
        let (indices, distances) = index.search_k(&query, 1, 200);

        assert!(
            distances[0] < 0.1,
            "Should find point 0, got idx={} with distance {}",
            indices[0],
            distances[0]
        );
    }

    #[test]
    fn test_fanng_cosine_distance() {
        let mat = create_distributed_data();
        let index = Fanng::new(mat.as_ref(), "cosine", &test_params(), 42, false);

        let query: Vec<f32> = (0..20).map(|j| mat[(0, j)]).collect();
        let (indices, distances) = index.search_k(&query, 5, 200);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0, "Should find itself first");
        assert!(distances[0] < 0.01, "Distance to self should be ~0");
    }

    #[test]
    fn test_fanng_max_calcs_limit() {
        let mat = create_distributed_data();
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = mat.row(10).iter().cloned().collect();

        let (indices1, _) = index.search_k(&query, 5, 50);
        let (indices2, _) = index.search_k(&query, 5, 500);

        assert_eq!(indices1.len(), 5);
        assert_eq!(indices2.len(), 5);

        assert!(
            indices2.contains(&10),
            "Should find query point with high budget"
        );
    }

    #[test]
    fn test_fanng_reproducibility() {
        let mat = create_distributed_data();

        let index1 = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);
        let index2 = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = mat.row(5).iter().cloned().collect();
        let (indices1, _) = index1.search_k(&query, 5, 200);
        let (indices2, _) = index2.search_k(&query, 5, 200);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_fanng_different_seeds() {
        let mat = create_distributed_data();

        let index1 = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);
        let index2 = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 123, false);

        let query: Vec<f32> = mat.row(7).iter().cloned().collect();
        let (indices1, _) = index1.search_k(&query, 5, 200);
        let (indices2, _) = index2.search_k(&query, 5, 200);

        assert_eq!(indices1.len(), 5);
        assert_eq!(indices2.len(), 5);

        assert!(indices1.contains(&7));
        assert!(indices2.contains(&7));
    }

    #[test]
    fn test_fanng_returns_distances() {
        let mat = create_distributed_data();
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = mat.row(15).iter().cloned().collect();
        let (indices, distances) = index.search_k(&query, 5, 300);

        assert_eq!(indices.len(), distances.len());
        assert_eq!(indices.len(), 5);

        for &dist in &distances {
            assert!(dist >= 0.0);
        }

        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "Distances not sorted at index {}: {} < {}",
                i,
                distances[i],
                distances[i - 1]
            );
        }

        assert_eq!(indices[0], 15);
        assert!(distances[0] < 0.01);
    }

    #[test]
    fn test_fanng_larger_dataset() {
        let n = 400; // Reduced from 1000
        let dim = 15; // Reduced from 20
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 100.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, distances) = index.search_k(&query, 10, 500);

        assert_eq!(indices.len(), 10);

        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "Distances not sorted at {}: {} < {}",
                i,
                distances[i],
                distances[i - 1]
            );
        }

        assert!(indices.contains(&0));
    }

    #[test]
    fn test_fanng_high_recall() {
        let mat = create_distributed_data();
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = mat.row(20).iter().cloned().collect();
        let (indices, _) = index.search_k(&query, 10, 500);

        assert_eq!(indices.len(), 10);
        assert!(indices.contains(&20), "Should find query point");
    }

    #[test]
    fn test_fanng_edge_case_k_larger_than_dataset() {
        let mat = create_distributed_data();
        let index = Fanng::new(mat.as_ref(), "euclidean", &test_params(), 42, false);

        let query: Vec<f32> = mat.row(0).iter().cloned().collect();
        let (indices, _) = index.search_k(&query, 300, 1000);

        assert!(indices.len() <= 200);
        assert!(indices.len() >= 50);
    }

    #[test]
    fn test_fanng_batch_size_variation() {
        let mat = create_distributed_data();

        let params1 = FanngParams::new(15, 5, 5, 100, 50);
        let params2 = FanngParams::new(15, 20, 5, 100, 50);

        let index1 = Fanng::new(mat.as_ref(), "euclidean", &params1, 42, false);
        let index2 = Fanng::new(mat.as_ref(), "euclidean", &params2, 42, false);

        let query: Vec<f32> = mat.row(12).iter().cloned().collect();
        let (indices1, _) = index1.search_k(&query, 5, 300);
        let (indices2, _) = index2.search_k(&query, 5, 300);

        assert_eq!(indices1.len(), 5);
        assert_eq!(indices2.len(), 5);

        assert!(indices1.contains(&12));
        assert!(indices2.contains(&12));
    }

    #[test]
    fn test_fanng_max_degree_respected() {
        let mat = create_distributed_data::<f32>();
        let max_degree = 12;
        let params = FanngParams::new(max_degree, 20, 5, 100, 50);
        let index = Fanng::new(mat.as_ref(), "euclidean", &params, 42, false);

        for edges in &index.graph {
            assert!(
                edges.len() <= max_degree,
                "Vertex has {} edges, max is {}",
                edges.len(),
                max_degree
            );
        }
    }
}
