use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::dist::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

pub type NeighbourUpdates<T> = Vec<(usize, Vec<(OrderedFloat<T>, usize)>)>;

/// Search state for HNSW queries
///
/// ### Fields
///
/// * `visited` - Boolean indicating if node was visited
/// * `candidates` - Candidates for the search
/// * `working` - Working set for distance calculations
pub struct SearchState<T> {
    visited: Vec<bool>,
    candidates: BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
    working: Vec<(OrderedFloat<T>, usize)>,
}

impl<T> SearchState<T>
where
    T: Float,
{
    /// Generate a new search state
    ///
    /// ### Params
    ///
    /// * `capacity` - The pre-allocated capacity of the search state
    ///
    /// ### Returns
    ///
    /// Self
    fn new(capacity: usize) -> Self {
        Self {
            visited: vec![false; capacity],
            candidates: BinaryHeap::with_capacity(capacity),
            working: Vec::with_capacity(capacity),
        }
    }

    /// Reset the search state
    ///
    /// ### Params
    ///
    /// * `n` - Number to reset to
    fn reset(&mut self, n: usize) {
        if self.visited.len() < n {
            self.visited.resize(n, false);
        }
        self.visited.fill(false);
        self.candidates.clear();
        self.working.clear();
    }
}

/// Computed connections for a node during parallel construction
///
/// ### Fields
///
/// * `node` - The node being inserted
/// * `connections` - Direct connections from this node to its neighbours
struct ConnectionUpdate<T> {
    node: usize,
    connections: Vec<(OrderedFloat<T>, usize)>,
}

//////////////////////////
// Thread-local buffers //
//////////////////////////

thread_local! {
    static SEARCH_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static SEARCH_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
}

/// Trait for accessing type-specific thread-local search state
pub trait HnswState<T> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<T>>) -> R;

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<T>>) -> R;
}

impl HnswState<f32> for HnswIndex<f32> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<f32>>) -> R,
    {
        SEARCH_STATE_F32.with(f)
    }

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<f32>>) -> R,
    {
        BUILD_STATE_F32.with(f)
    }
}

impl HnswState<f64> for HnswIndex<f64> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<f64>>) -> R,
    {
        SEARCH_STATE_F64.with(f)
    }

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&RefCell<SearchState<f64>>) -> R,
    {
        BUILD_STATE_F64.with(f)
    }
}

////////////////////
// Main structure //
////////////////////

/// HNSW (Hierarchical Navigable Small World) graph index
///
/// ### Fields
///
/// * `vectors_flat` - Flat structure of vectors for cache locality
/// * `dim` - Feature dimensions
/// * `n` - Number of samples
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `norms` - Precomputed norms for Cosine distance
/// * `layer_assignments` - Which layer each node belongs to (0 = base layer)
/// * `neighbours_flat` - All neighbour PIDs in flat array
/// * `neighbour_offsets` - Starting index of each node's neighbours
/// * `entry_point` - Node ID with highest layer
/// * `max_layer` - Highest layer in the graph
/// * `m` - Number of connections per layer (M parameter)
/// * `ef_construction` - Size of dynamic candidate list during construction
/// * `extend_candidates` - Whether to extend candidate set in heuristic
///   (Algorithm 4)
/// * `keep_pruned` - Whether to keep pruned connections if space available
///
/// ### Algorithm Overview
///
/// HNSW builds a hierarchical graph where:
/// 1. Each node is randomly assigned to layers (exponential distribution)
/// 2. Higher layers are sparser and enable fast coarse search
/// 3. Lower layers are denser and provide precise results
/// 4. Search starts at top layer and descends through layers
/// 5. Each layer refines the search, narrowing candidates
/// 6. Heuristic selection prevents clustered connections (Algorithm 4 from
///    paper)
pub struct HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub metric: Dist,
    pub norms: Vec<T>,
    pub layer_assignments: Vec<u8>,
    pub neighbours_flat: Vec<u32>,
    pub neighbour_offsets: Vec<usize>,
    pub entry_point: u32,
    pub max_layer: u8,
    pub m: usize,
    pub ef_construction: usize,
    pub extend_candidates: bool,
    pub keep_pruned: bool,
}

/// Implement the needed function for VectorDistance for HnswIndex
impl<T: Float + FromPrimitive + Send + Sync> VectorDistance<T> for HnswIndex<T> {
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

impl<T> HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync,
    Self: HnswState<T>,
{
    /// Build HNSW index
    ///
    /// ### Params
    ///
    /// * `mat` - Embedding matrix (rows = samples, cols = features)
    /// * `m` - Number of bidirectional connections per layer (typical: 16-32)
    /// * `ef_construction` - Size of candidate list during construction
    ///   (typical: 100-200)
    /// * `dist_metric` - "euclidean" or "cosine"
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built HnswIndex ready for querying
    pub fn build(
        mat: MatRef<T>,
        m: usize,
        ef_construction: usize,
        dist_metric: &str,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);
        let n = mat.nrows();
        let dim = mat.ncols();

        if verbose {
            println!("Building HNSW index with {} nodes, M={}", n, m);
        }

        let start_total = Instant::now();

        // flatten matrix for cache locality
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

        // assign layers using exponential distribution
        let ml = T::one() / T::from_usize(m).unwrap().ln();
        let mut rng = SmallRng::seed_from_u64(seed as u64);

        let layer_assignments: Vec<u8> = (0..n)
            .map(|_| {
                let uniform: f64 = rng.random();
                let uniform_t = T::from_f64(uniform).unwrap();
                ((-uniform_t.ln() * ml).floor().to_u8().unwrap()).min(15)
            })
            .collect();

        let max_layer = *layer_assignments.iter().max().unwrap_or(&0);
        let entry_point = layer_assignments
            .iter()
            .position(|&l| l == max_layer)
            .unwrap_or(0) as u32;

        if verbose {
            println!("Max layer: {}, Entry point: {}", max_layer, entry_point);
        }

        // allocate neighbour storage
        let total_capacity: usize = layer_assignments
            .iter()
            .map(|&layer| if layer == 0 { m * 2 } else { m })
            .sum();
        let neighbours_flat = vec![u32::MAX; total_capacity];

        let mut neighbour_offsets = Vec::with_capacity(n);
        let mut offset = 0;
        for &layer in &layer_assignments {
            neighbour_offsets.push(offset);
            offset += if layer == 0 { m * 2 } else { m };
        }

        let mut index = HnswIndex {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            layer_assignments,
            neighbours_flat,
            neighbour_offsets,
            entry_point,
            max_layer,
            m,
            ef_construction,
            extend_candidates: false,
            keep_pruned: true,
        };

        index.build_graph(verbose);

        let end_total = start_total.elapsed();
        if verbose {
            println!("Total HNSW build time: {:.2?}", end_total);
        }

        index
    }

    /// Build the graph structure layer by layer
    ///
    /// Automatically chooses between parallel and sequential construction
    /// based on layer size. Layers with > 1000 nodes use parallel construction.
    ///
    /// ### Params
    ///
    /// * `verbose` - Print progress information
    ///
    /// ### Algorithm Details
    ///
    /// Constructs the hierarchical graph by:
    /// 1. Grouping nodes by their assigned layers
    /// 2. Building each layer from top to bottom
    /// 3. Using parallel insertion for large layers (> 1000 nodes)
    /// 4. Using sequential insertion for small layers
    /// 5. Skipping entry point (already placed)
    fn build_graph(&mut self, verbose: bool) {
        let nodes_by_layer: Vec<Vec<usize>> = (0..=self.max_layer)
            .map(|layer| {
                (0..self.n)
                    .filter(|&i| self.layer_assignments[i] >= layer)
                    .collect()
            })
            .collect();

        for layer in (0..=self.max_layer).rev() {
            let start = Instant::now();
            let layer_nodes = &nodes_by_layer[layer as usize];

            if verbose {
                println!("Building layer {} with {} nodes", layer, layer_nodes.len());
            }

            if layer_nodes.len() > 1000 {
                self.build_layer_parallel(layer, layer_nodes);
            } else {
                self.build_layer_sequential(layer, layer_nodes);
            }

            if verbose {
                println!("  Layer {} built in {:.2?}", layer, start.elapsed());
            }
        }
    }

    /// Build a layer using parallel construction
    ///
    /// Processes nodes in batches to balance parallelism with memory
    /// efficiency. Each batch is processed in two phases:
    ///
    /// 1. Parallel phase: Each thread computes connections independently
    /// 2. Sequential phase: All connection updates are applied without races
    ///
    /// ### Params
    ///
    /// * `layer` - Layer to build
    /// * `nodes` - Nodes to insert at this layer
    fn build_layer_parallel(&self, layer: u8, nodes: &[usize]) {
        const BATCH_SIZE: usize = 1000;

        for chunk in nodes.chunks(BATCH_SIZE) {
            // Phase 1: PARALLEL - Compute OUTGOING connections only
            let updates: Vec<ConnectionUpdate<T>> = chunk
                .par_iter()
                .map(|&node| {
                    Self::with_build_state(|state_cell| {
                        let mut state = state_cell.borrow_mut();
                        self.compute_node_connections(node, layer, &mut state)
                    })
                })
                .collect();

            // Phase 2: SEQUENTIAL - Apply outgoing AND compute/apply incoming
            for update in updates {
                // 1. Connect Node -> Neighbours
                self.connect_neighbours(update.node, &update.connections, layer);

                // 2. Connect Neighbours -> Node (Pruning with current state)
                for &(_, neighbour_id) in &update.connections {
                    self.prune_and_connect_heuristic(neighbour_id, update.node, layer);
                }
            }
        }
    }

    /// Build a layer using sequential construction
    ///
    /// More efficient for small layers where parallelism overhead exceeds benefits.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer to build
    /// * `nodes` - Nodes to insert at this layer
    fn build_layer_sequential(&self, layer: u8, nodes: &[usize]) {
        let mut state = SearchState::new(self.n);

        for &node in nodes.iter() {
            self.insert_node_with_state(node, layer, &mut state);
        }
    }

    /// Insert a single node at a given layer (sequential version)
    ///
    /// ### Params
    ///
    /// * `node` - Node index to insert
    /// * `insert_layer` - Layer to insert at (and all layers below)
    /// * `state` - Mutable search state
    ///
    /// ### Algorithm Details
    ///
    /// 1. Start from entry point in top layer
    /// 2. Greedily descend through layers above insert_layer
    /// 3. At insert_layer and below:
    ///    - Search for nearest neighbours
    ///    - Select neighbours using heuristic (Algorithm 4)
    ///    - Connect node to selected neighbours
    ///    - Update bidirectional links with heuristic pruning
    fn insert_node_with_state(&self, node: usize, insert_layer: u8, state: &mut SearchState<T>) {
        state.reset(self.n);

        let mut entry_points = vec![(OrderedFloat(T::zero()), self.entry_point as usize)];

        for layer in (insert_layer + 1..=self.max_layer).rev() {
            state.reset(self.n);
            entry_points = self.search_layer(node, layer, &entry_points, 1, state);
        }

        for layer in (0..=insert_layer).rev() {
            state.reset(self.n);
            let ef = self.ef_construction;

            let candidates = self.search_layer(node, layer, &entry_points, ef, state);

            let selected = self.select_heuristic(node, &candidates, layer);

            self.connect_neighbours(node, &selected, layer);

            for &(_, neighbour_id) in &selected {
                self.prune_and_connect_heuristic(neighbour_id, node, layer);
            }

            entry_points = selected;
        }
    }

    /// Compute connections for a node (parallel version, read-only)
    ///
    /// This method is thread-safe and can be called in parallel. It only reads
    /// from shared data structures and returns connection updates to be applied
    /// sequentially.
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `insert_layer` - Layer to insert at
    /// * `state` - Mutable search state (thread-local)
    ///
    /// ### Returns
    ///
    /// ConnectionUpdate containing all changes to be applied
    fn compute_node_connections(
        &self,
        node: usize,
        insert_layer: u8,
        state: &mut SearchState<T>,
    ) -> ConnectionUpdate<T> {
        state.reset(self.n);

        let mut entry_points = vec![(OrderedFloat(T::zero()), self.entry_point as usize)];

        for layer in (insert_layer + 1..=self.max_layer).rev() {
            state.reset(self.n);
            entry_points = self.search_layer(node, layer, &entry_points, 1, state);
        }

        state.reset(self.n);
        let candidates = self.search_layer(
            node,
            insert_layer,
            &entry_points,
            self.ef_construction,
            state,
        );

        let selected = self.select_heuristic(node, &candidates, insert_layer);

        ConnectionUpdate {
            node,
            connections: selected,
        }
    }

    /// Select neighbours using heuristic (Algorithm 4 from paper)
    ///
    /// The heuristic filters candidates to avoid clustered connections. A
    /// candidate is selected if it's closer to the query node than to any
    /// already-selected neighbour, which helps create connections that bridge
    /// different clusters.
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `candidates` - Initial candidate neighbours (sorted by distance)
    /// * `layer` - Current layer
    ///
    /// ### Returns
    ///
    /// Filtered list of neighbours (at most M or M*2)
    ///
    /// ### Algorithm Details
    ///
    /// 1. Optionally extend candidates by considering neighbours of candidates
    /// 2. For each candidate, check if it's closer to query than to any result
    /// 3. If yes, add to result set; if no, add to discarded set
    /// 4. Optionally add back discarded connections if space available
    fn select_heuristic(
        &self,
        node: usize,
        candidates: &[(OrderedFloat<T>, usize)],
        layer: u8,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
        let mut working = Vec::with_capacity(candidates.len() * 2);
        let mut visited = vec![false; self.n];

        for &candidate in candidates {
            if candidate.1 != node && !visited[candidate.1] {
                working.push(candidate);
                visited[candidate.1] = true;
            }
        }

        if self.extend_candidates {
            let max_check = if layer == 0 { self.m * 2 } else { self.m };

            for &(_, cand_id) in candidates {
                if cand_id == node {
                    continue;
                }

                let offset = self.neighbour_offsets[cand_id];

                for i in 0..max_check {
                    let neighbour = self.neighbours_flat[offset + i];
                    if neighbour == u32::MAX {
                        break;
                    }

                    let neighbour_id = neighbour as usize;
                    if neighbour_id == node || visited[neighbour_id] {
                        continue;
                    }

                    visited[neighbour_id] = true;

                    let dist = unsafe {
                        OrderedFloat(match self.metric {
                            Dist::Euclidean => self.euclidean_distance(node, neighbour_id),
                            Dist::Cosine => self.cosine_distance(node, neighbour_id),
                        })
                    };

                    working.push((dist, neighbour_id));
                }
            }

            working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }

        let mut result = Vec::with_capacity(max_neighbours);
        let mut discarded = Vec::with_capacity(working.len());

        for candidate in working {
            if result.len() >= max_neighbours {
                break;
            }

            let (cand_dist, cand_id) = candidate;

            if cand_id == node {
                continue;
            }

            let mut closer_to_query = true;
            for &(_, result_id) in &result {
                let dist_to_result = unsafe {
                    OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(cand_id, result_id),
                        Dist::Cosine => self.cosine_distance(cand_id, result_id),
                    })
                };

                if dist_to_result < cand_dist {
                    closer_to_query = false;
                    break;
                }
            }

            if closer_to_query {
                result.push(candidate);
            } else {
                discarded.push(candidate);
            }
        }

        if self.keep_pruned {
            for candidate in discarded {
                if result.len() >= max_neighbours || candidate.1 == node {
                    break;
                }
                result.push(candidate);
            }
        }

        result
    }

    /// Compute updated neighbour list for an existing node
    ///
    /// Used during parallel construction to compute what a neighbour's
    /// connections should be after adding a new node, without modifying
    /// shared data.
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Existing neighbour to update
    /// * `new_node` - New node to add as neighbour
    /// * `layer` - Current layer
    ///
    /// ### Returns
    ///
    /// Updated list of connections for the neighbour
    fn compute_neighbour_update(
        &self,
        neighbour_id: usize,
        new_node: usize,
        layer: u8,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        if neighbour_id == new_node {
            let offset = self.neighbour_offsets[neighbour_id];
            let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

            let neighbours_slice = unsafe {
                std::slice::from_raw_parts(
                    self.neighbours_flat.as_ptr().add(offset),
                    max_neighbours,
                )
            };
            return neighbours_slice
                .iter()
                .take_while(|&&n| n != u32::MAX)
                .map(|&n| {
                    let dist = unsafe {
                        OrderedFloat(match self.metric {
                            Dist::Euclidean => self.euclidean_distance(neighbour_id, n as usize),
                            Dist::Cosine => self.cosine_distance(neighbour_id, n as usize),
                        })
                    };
                    (dist, n as usize)
                })
                .collect();
        }

        let offset = self.neighbour_offsets[neighbour_id];
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        let neighbours_slice = unsafe {
            std::slice::from_raw_parts(self.neighbours_flat.as_ptr().add(offset), max_neighbours)
        };

        let mut curr: Vec<(OrderedFloat<T>, usize)> = neighbours_slice
            .iter()
            .take_while(|&&n| n != u32::MAX)
            .filter(|&&n| {
                let nid = n as usize;
                nid != new_node && nid != neighbour_id
            })
            .map(|&n| {
                let dist = unsafe {
                    OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(neighbour_id, n as usize),
                        Dist::Cosine => self.cosine_distance(neighbour_id, n as usize),
                    })
                };
                (dist, n as usize)
            })
            .collect();

        let new_dist = unsafe {
            OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, new_node),
                Dist::Cosine => self.cosine_distance(neighbour_id, new_node),
            })
        };

        curr.push((new_dist, new_node));
        curr.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        self.select_heuristic(neighbour_id, &curr, layer)
    }

    /// Prune and add bidirectional connection using heuristic
    ///
    /// Sequential version.
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Existing neighbour to update
    /// * `new_node` - New node to add as neighbour
    /// * `layer` - Current layer
    fn prune_and_connect_heuristic(&self, neighbour_id: usize, new_node: usize, layer: u8) {
        let update = self.compute_neighbour_update(neighbour_id, new_node, layer);
        self.write_neighbour_connections(neighbour_id, &update, layer);
    }

    /// Write neighbour connections to shared storage
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Node whose connections to update
    /// * `connections` - New connections
    /// * `layer` - Current layer
    fn write_neighbour_connections(
        &self,
        neighbour_id: usize,
        connections: &[(OrderedFloat<T>, usize)],
        layer: u8,
    ) {
        let offset = self.neighbour_offsets[neighbour_id];
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        let neighbours_slice_mut = unsafe {
            std::slice::from_raw_parts_mut(
                self.neighbours_flat.as_ptr().add(offset) as *mut u32,
                max_neighbours,
            )
        };

        for (i, &(_, node_id)) in connections.iter().enumerate() {
            neighbours_slice_mut[i] = node_id as u32;
        }

        #[allow(clippy::needless_range_loop)]
        for i in connections.len()..max_neighbours {
            neighbours_slice_mut[i] = u32::MAX;
        }
    }

    /// Search a single layer for nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query_node` - Node we are searching from
    /// * `layer` - Current layer
    /// * `entry_points` - Starting points for search
    /// * `ef` - Size of candidate list
    /// * `state` - Mutable search state
    ///
    /// ### Returns
    ///
    /// List of (distance, node_id) pairs, sorted by distance
    ///
    /// ### Algorithm Details
    ///
    /// Uses greedy best-first search:
    /// 1. Maintain a dynamic candidate list of size ef
    /// 2. Expand closest unvisited candidates
    /// 3. Track furthest candidate to prune worse results
    /// 4. Stop when all promising candidates exhausted
    fn search_layer(
        &self,
        query_node: usize,
        layer: u8,
        entry_points: &[(OrderedFloat<T>, usize)],
        ef: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working.clear();
        state.candidates.clear();

        for &(dist, pid) in entry_points {
            if !state.visited[pid] {
                state.visited[pid] = true;
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

        // The furthest distance must be Infinity until we have collected 'ef'
        // candidates. if I initialise it to the max of entry_points (which
        // might be 0 if entry point is self), I will immediately prune
        // searching if the next neighbour is > 0.
        let mut furthest_dist = if state.working.len() >= ef {
            state
                .working
                .iter()
                .map(|(d, _)| *d)
                .max()
                .unwrap_or(OrderedFloat(T::infinity()))
        } else {
            OrderedFloat(T::infinity())
        };

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist {
                break;
            }

            let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
            let offset = self.neighbour_offsets[current_id];

            for i in 0..max_neighbours {
                let neighbour = self.neighbours_flat[offset + i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;
                if state.visited[neighbour_id] {
                    continue;
                }

                state.visited[neighbour_id] = true;

                let dist = unsafe {
                    OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(query_node, neighbour_id),
                        Dist::Cosine => self.cosine_distance(query_node, neighbour_id),
                    })
                };

                if dist < furthest_dist || state.working.len() < ef {
                    state.candidates.push(Reverse((dist, neighbour_id)));
                    state.working.push((dist, neighbour_id));

                    if state.working.len() > ef {
                        state.working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                        state.working.pop();
                        furthest_dist = state.working.last().map(|(d, _)| *d).unwrap();
                    } else if state.working.len() == ef {
                        furthest_dist = state
                            .working
                            .iter()
                            .map(|(d, _)| *d)
                            .max()
                            .unwrap_or(OrderedFloat(T::infinity()));
                    }
                }
            }
        }

        state.working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        state.working.clone()
    }

    /// Connect node to its selected neighbours (one-directional)
    ///
    /// ### Params
    ///
    /// * `node` - Node to update
    /// * `neighbours` - List of neighbours to connect to
    /// * `layer` - Current layer
    fn connect_neighbours(&self, node: usize, neighbours: &[(OrderedFloat<T>, usize)], layer: u8) {
        let offset = self.neighbour_offsets[node];
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        let neighbours_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.neighbours_flat.as_ptr().add(offset) as *mut u32,
                max_neighbours,
            )
        };

        let mut write_idx = 0;
        for &(_, neighbour_id) in neighbours.iter() {
            if write_idx >= max_neighbours {
                break;
            }
            if neighbour_id != node {
                neighbours_slice[write_idx] = neighbour_id as u32;
                write_idx += 1;
            }
        }

        for i in write_idx..max_neighbours {
            neighbours_slice[i] = u32::MAX;
        }
    }

    /// Query the index for k nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Size of candidate list (higher = better recall, slower)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    ///
    /// ### Algorithm Details
    ///
    /// 1. Start from entry point at top layer
    /// 2. Greedily descend through layers (ef=1 for speed)
    /// 3. At base layer, expand search with ef_search
    /// 4. Return top k results
    pub fn query(&self, query: &[T], k: usize, ef_search: usize) -> (Vec<usize>, Vec<T>) {
        assert_eq!(query.len(), self.dim);

        Self::with_search_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let query_norm = if self.metric == Dist::Cosine {
                query
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            } else {
                T::one()
            };

            let entry_dist =
                self.compute_query_distance(query, self.entry_point as usize, query_norm);
            let mut entry_points = vec![(OrderedFloat(entry_dist), self.entry_point as usize)];

            for layer in (1..=self.max_layer).rev() {
                state.reset(self.n);
                entry_points =
                    self.search_layer_query(query, query_norm, layer, &entry_points, 1, &mut state);
            }

            state.reset(self.n);
            let mut candidates = self.search_layer_query(
                query,
                query_norm,
                0,
                &entry_points,
                ef_search.max(k),
                &mut state,
            );

            candidates.truncate(k);

            let (indices, distances): (Vec<usize>, Vec<T>) = candidates
                .into_iter()
                .map(|(OrderedFloat(d), id)| (id, d))
                .unzip();

            (indices, distances)
        })
    }

    /// Search layer for query vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `query_norm` - Precomputed norm of query (for cosine)
    /// * `layer` - Current layer
    /// * `entry_points` - Starting points for search
    /// * `ef` - Size of candidate list
    /// * `state` - Mutable search state
    ///
    /// ### Returns
    ///
    /// List of (distance, node_id) pairs, sorted by distance
    fn search_layer_query(
        &self,
        query: &[T],
        query_norm: T,
        layer: u8,
        entry_points: &[(OrderedFloat<T>, usize)],
        ef: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working.clear();
        state.candidates.clear();

        for &(dist, pid) in entry_points {
            if !state.visited[pid] {
                state.visited[pid] = true;
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

        // same fix as search_layer. initialise furthest_dist to Infinity if
        // working set < ef.
        let mut furthest_dist = if state.working.len() >= ef {
            state
                .working
                .iter()
                .map(|(d, _)| *d)
                .max()
                .unwrap_or(OrderedFloat(T::infinity()))
        } else {
            OrderedFloat(T::infinity())
        };

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist {
                break;
            }

            let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
            let offset = self.neighbour_offsets[current_id];

            for i in 0..max_neighbours {
                let neighbour = self.neighbours_flat[offset + i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;
                if state.visited[neighbour_id] {
                    continue;
                }

                state.visited[neighbour_id] = true;

                let dist =
                    OrderedFloat(self.compute_query_distance(query, neighbour_id, query_norm));

                if dist < furthest_dist || state.working.len() < ef {
                    state.candidates.push(Reverse((dist, neighbour_id)));
                    state.working.push((dist, neighbour_id));

                    if state.working.len() > ef {
                        state.working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                        state.working.pop();
                        furthest_dist = state.working.last().map(|(d, _)| *d).unwrap();
                    } else if state.working.len() == ef {
                        furthest_dist = state
                            .working
                            .iter()
                            .map(|(d, _)| *d)
                            .max()
                            .unwrap_or(OrderedFloat(T::infinity()));
                    }
                }
            }
        }

        state.working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        state.working.clone()
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
    /// Distance value (Euclidean squared or Cosine distance)
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.metric {
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
    use approx::assert_relative_eq;
    use faer::Mat;
    use std::sync::Arc;
    use std::thread;

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
    fn test_hnsw_build_euclidean() {
        let mat = create_simple_matrix();
        let _ = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);
    }

    #[test]
    fn test_hnsw_build_cosine() {
        let mat = create_simple_matrix();
        let _ = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "cosine", 42, false);
    }

    #[test]
    fn test_hnsw_query_finds_self() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, 50);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hnsw_query_euclidean() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_hnsw_query_cosine() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "cosine", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hnsw_parallel_build() {
        let n = 2000;
        let dim = 10;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, 50);

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_hnsw_thread_safety() {
        let mat = create_simple_matrix();
        let index = Arc::new(HnswIndex::<f32>::build(
            mat.as_ref(),
            16,
            100,
            "euclidean",
            42,
            false,
        ));

        let mut handles = vec![];
        for _ in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let query = vec![0.5, 0.5, 0.0];
                let (indices, _) = index_clone.query(&query, 3, 50);
                assert_eq!(indices.len(), 3);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_hnsw_reproducibility() {
        let mat = create_simple_matrix();

        let index1 = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);
        let index2 = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, 50);
        let (indices2, _) = index2.query(&query, 3, 50);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_hnsw_recall() {
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        data.extend_from_slice(&[0.0, 0.0, 0.0]);

        for i in 1..n {
            let dist = i as f32 * 0.1;
            data.extend_from_slice(&[dist, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query = vec![0.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, 100);

        assert_eq!(indices[0], 0);

        let expected_neighbours: Vec<usize> = (0..5).collect();
        let found_correct = indices
            .iter()
            .filter(|&&idx| expected_neighbours.contains(&idx))
            .count();

        assert!(found_correct >= 4);
    }
}
