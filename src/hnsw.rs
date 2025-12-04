use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use parking_lot::RwLock;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::time::Instant;
use thousands::*;

use crate::dist::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

pub type NeighbourUpdates<T> = Vec<(usize, Vec<(OrderedFloat<T>, usize)>)>;

/// Search state for HNSW queries and construction
///
/// Optimised for low-allocation via epoch-based visitation checks and
/// pre-allocated scratch buffers.
///
/// ### Fields
///
/// * `visited` - Stores the "visit epoch" ID for each node.
/// * `visit_id` - The current epoch ID. Incrementing this effectively "clears"
///   `visited` in O(1).
/// * `candidates` - Min-heap for candidates.
/// * `working` - Working set for distance calculations.
/// * `scratch_working` - Reuse buffer for heuristic selection.
/// * `scratch_discarded` - Reuse buffer for heuristic selection.
pub struct SearchState<T> {
    visited: Vec<usize>,
    visit_id: usize,
    candidates: BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
    working: Vec<(OrderedFloat<T>, usize)>,
    scratch_working: Vec<(OrderedFloat<T>, usize)>,
    scratch_discarded: Vec<(OrderedFloat<T>, usize)>,
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
    /// Returns `Self`.
    fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; capacity],
            visit_id: 1, // Start at 1, 0 is reserved for "never visited"
            candidates: BinaryHeap::with_capacity(capacity),
            working: Vec::with_capacity(capacity),
            scratch_working: Vec::with_capacity(capacity),
            scratch_discarded: Vec::with_capacity(capacity),
        }
    }

    /// Reset the search state for a new query
    ///
    /// This increments the `visit_id` (Epoch), avoiding the O(N) memory
    /// clearing cost associated with standard boolean vectors (prior
    /// implementation was based on this).
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes in the index (to ensure capacity)
    fn reset(&mut self, n: usize) {
        if self.visited.len() < n {
            self.visited.resize(n, 0);
        }

        self.visit_id = self.visit_id.wrapping_add(1);
        if self.visit_id == 0 {
            self.visited.fill(0);
            self.visit_id = 1;
        }

        self.candidates.clear();
        self.working.clear();
        self.scratch_working.clear();
        self.scratch_discarded.clear();
    }

    /// Check if a node has been visited in the current epoch
    ///
    /// ### Params
    ///
    /// * `node` - Node index to check
    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        self.visited[node] == self.visit_id
    }

    /// Mark a node as visited in the current epoch
    ///
    /// ### Params
    ///
    /// * `node` - Node index to mark
    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        self.visited[node] = self.visit_id;
    }
}

/// Construction-time neighbour storage with concurrent write support
///
/// During construction, each node's neighbours are stored in a separate
/// RwLock<Vec>, enabling parallel insertions without batching or synch
/// barriers.
struct ConstructionGraph<T> {
    /// Each node's neighbours stored separately for concurrent access
    nodes: Vec<RwLock<Vec<u32>>>,
    /// Maximum neighbours per node (M or M*2 depending on layer)
    max_neighbours: Vec<usize>,
    /// Phantom data to tie the lifetime to T - new one...
    _phantom: PhantomData<T>,
}

impl<T> ConstructionGraph<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    /// Create a new construction graph
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `layer_assignments` - Layer for each node
    /// * `m` - M parameter
    fn new(n: usize, layer_assignments: &[u8], m: usize) -> Self {
        let nodes = (0..n).map(|_| RwLock::new(Vec::new())).collect();

        let max_neighbours = layer_assignments
            .iter()
            .map(|&layer| if layer == 0 { m * 2 } else { m })
            .collect();

        Self {
            nodes,
            max_neighbours,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get current neighbours for a node
    ///
    /// Initialises the read lock.
    ///
    /// ### Params
    ///
    /// * `node_id` - The node idx.
    fn get_neighbours(&self, node_id: usize) -> Vec<u32> {
        self.nodes[node_id].read().clone()
    }

    /// Write new neighbours for a node (write lock)
    ///
    /// ### Params
    ///
    /// * `node_id` - The node idx.
    /// * `neighbours` - Slice of tuples of OrderedFloat and indices of the
    ///   neighbours.
    fn set_neighbours(&self, node_id: usize, neighbours: &[(OrderedFloat<T>, usize)]) {
        let mut guard = self.nodes[node_id].write();
        guard.clear();

        for &(_, neighbour_id) in neighbours.iter().take(self.max_neighbours[node_id]) {
            if neighbour_id != node_id {
                guard.push(neighbour_id as u32);
            }
        }

        // Pad with INVALID (u32: MAX)
        while guard.len() < self.max_neighbours[node_id] {
            guard.push(u32::MAX);
        }
    }

    /// Finalise construction and convert to flat layout
    ///
    /// Consumes the construction graph and produces the flat representation
    /// optimised for subsequent query performance.
    ///
    /// ### Returns
    ///
    /// Tuple of (neighbours_flat, neighbour_offsets)
    fn into_flat(self) -> (Vec<u32>, Vec<usize>) {
        let mut neighbours_flat = Vec::new();
        let mut neighbour_offsets = Vec::new();

        for node in self.nodes {
            neighbour_offsets.push(neighbours_flat.len());
            let neighbours = node.into_inner();
            neighbours_flat.extend(neighbours);
        }

        (neighbours_flat, neighbour_offsets)
    }
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
///
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
    pub norms: Vec<T>,
    metric: Dist,
    layer_assignments: Vec<u8>,
    neighbours_flat: Vec<u32>,
    neighbour_offsets: Vec<usize>,
    entry_point: u32,
    max_layer: u8,
    m: usize,
    ef_construction: usize,
    extend_candidates: bool,
    keep_pruned: bool,
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
            println!(
                "Building HNSW index with {} nodes, M = {}",
                n.separate_with_underscores(),
                m
            );
        }

        let start_total = Instant::now();

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

        // Assign layers using exponential distribution
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

        // Create construction graph
        let construction_graph = ConstructionGraph::new(n, &layer_assignments, m);

        let mut index = HnswIndex {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            layer_assignments,
            neighbours_flat: Vec::new(),   // Filled after construction
            neighbour_offsets: Vec::new(), // Filled after construction
            entry_point,
            max_layer,
            m,
            ef_construction,
            extend_candidates: false,
            keep_pruned: true,
        };

        index.build_graph(&construction_graph, verbose);

        // Convert construction graph to flat layout
        let (neighbours_flat, neighbour_offsets) = construction_graph.into_flat();
        index.neighbours_flat = neighbours_flat;
        index.neighbour_offsets = neighbour_offsets;

        let end_total = start_total.elapsed();
        if verbose {
            println!("Total HNSW build time: {:.2?}", end_total);
        }

        index
    }

    /// Build the graph structure layer by layer
    ///
    /// This function uses parallelism under the hood via the
    /// `ConstructionGraph` structure that has RwLocks to avoid race conditions
    /// (which were a bane while writing this...).
    ///
    /// ### Params
    ///
    /// * `graph` - Construction graph with RwLock support
    /// * `verbose` - Print progress information
    fn build_graph(&self, graph: &ConstructionGraph<T>, verbose: bool) {
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
                println!(
                    "Building layer {} with {} nodes",
                    layer,
                    layer_nodes.len().separate_with_underscores()
                );
            }

            self.build_layer_parallel(layer, layer_nodes, graph);

            if verbose {
                println!("  Layer {} built in {:.2?}", layer, start.elapsed());
            }
        }
    }

    /// Build a layer with true parallel construction (no batching)
    ///
    /// All nodes are processed in parallel with concurrent writes to the
    /// construction graph. No synchronisation barriers needed.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer to build
    /// * `nodes` - Nodes to insert at this layer
    /// * `graph` - Construction graph with concurrent write support
    fn build_layer_parallel(&self, layer: u8, nodes: &[usize], graph: &ConstructionGraph<T>) {
        nodes.par_iter().for_each(|&node| {
            // Compute outgoing connections (thread-local state)
            let connections = Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                self.compute_node_connections(node, layer, graph, &mut state)
            });

            // Write node -> neighbours (exclusive access to this node)
            graph.set_neighbours(node, &connections);

            // Update neighbours -> node (concurrent, each thread may write to different neighbours)
            for &(_, neighbour_id) in &connections {
                self.prune_and_connect_concurrent(neighbour_id, node, layer, graph);
            }
        });
    }

    /// Compute connections for a node (reads from construction graph)
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `insert_layer` - Layer to insert at
    /// * `graph` - Construction graph
    /// * `state` - Thread-local search state
    ///
    /// ### Returns
    ///
    /// Selected connections for this node
    fn compute_node_connections(
        &self,
        node: usize,
        insert_layer: u8,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.reset(self.n);

        let mut entry_points = vec![(OrderedFloat(T::zero()), self.entry_point as usize)];

        // descend through upper layers
        for layer in (insert_layer + 1..=self.max_layer).rev() {
            state.reset(self.n);
            entry_points = self.search_layer(node, layer, &entry_points, 1, graph, state);
        }

        // search at insert layer
        state.reset(self.n);
        let ef = if insert_layer == 0 {
            self.ef_construction / 2 // Lower ef for dense layer 0
        } else {
            self.ef_construction
        };

        let candidates = self.search_layer(node, insert_layer, &entry_points, ef, graph, state);

        self.select_heuristic(node, &candidates, insert_layer, state)
    }

    /// Prune and connect with concurrent access (race-condition free)
    ///
    /// Atomically updates a neighbour's connection list by adding a new node,
    /// then pruning using the heuristic selection algorithm. The entire
    /// read-modify-write sequence is protected by a single write lock to
    /// prevent lost updates.
    ///
    /// ### Thread Safety
    ///
    /// Multiple threads may call this concurrently on different neighbours.
    /// If multiple threads target the same neighbour, the write lock serialises
    /// access, ensuring each update is applied atomically.
    ///
    /// ### Algorithm
    ///
    /// 1. Acquire exclusive write lock on neighbour's connection list
    /// 2. Read current connections (while holding lock)
    /// 3. Compute distances to all current connections + new node
    /// 4. Sort candidates by distance
    /// 5. Apply heuristic selection (using thread-local state, no locks)
    /// 6. Write final selection back (still holding lock)
    /// 7. Release lock
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Existing neighbour to update
    /// * `new_node` - New node to add as a potential neighbour
    /// * `layer` - Current layer (determines max connections: M or M*2)
    /// * `graph` - Construction graph with RwLock-protected connection lists
    fn prune_and_connect_concurrent(
        &self,
        neighbour_id: usize,
        new_node: usize,
        layer: u8,
        graph: &ConstructionGraph<T>,
    ) {
        // CRITICAL: acquire write lock FIRST and hold throughout entire ops.
        // this prevents race conditions where another thread updates the same
        // neighbour. Windows otherwise runs into issues...
        let mut guard = graph.nodes[neighbour_id].write();

        let current = guard.clone();

        let mut candidates: Vec<(OrderedFloat<T>, usize)> = Vec::with_capacity(current.len() + 1);

        // add all valid existing neighbours with their distances
        for &neighbour in current.iter() {
            if neighbour == u32::MAX {
                break; // Hit padding, stop
            }

            let neighbour_node_id = neighbour as usize;

            // compute distance from neighbour_id to this existing connection
            let dist = OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, neighbour_node_id),
                Dist::Cosine => self.cosine_distance(neighbour_id, neighbour_node_id),
            });

            candidates.push((dist, neighbour_node_id));
        }

        // add the new node as a candidate (if it's not self-connection)
        if new_node != neighbour_id {
            let dist = OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, new_node),
                Dist::Cosine => self.cosine_distance(neighbour_id, new_node),
            });

            candidates.push((dist, new_node));
        }

        // sort candidates by distance (nearest first)
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // apply heuristic selection to choose best subset of candidates
        // sses thread-local state (no locks acquired here - critical!)
        let selected = Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            self.select_heuristic(neighbour_id, &candidates, layer, &mut state)
        });

        // write back the final selection (still holding write lock)
        let max_neighbours = graph.max_neighbours[neighbour_id];
        guard.clear();

        for &(_, node_id) in selected.iter().take(max_neighbours) {
            if node_id != neighbour_id {
                guard.push(node_id as u32);
            }
        }

        // pad with INVALID markers to fixed size
        while guard.len() < max_neighbours {
            guard.push(u32::MAX);
        }

        // write lock automatically released when guard drops here <<<---
    }

    /// Search layer using construction graph
    ///
    /// ### Params
    ///
    /// * `query_node` - Node we're searching from
    /// * `layer` - Current layer
    /// * `entry_points` - Starting points
    /// * `ef` - Candidate list size
    /// * `graph` - Construction graph
    /// * `state` - Search state
    ///
    /// ### Returns
    ///
    /// Sorted list of nearest neighbours
    fn search_layer(
        &self,
        query_node: usize,
        layer: u8,
        entry_points: &[(OrderedFloat<T>, usize)],
        ef: usize,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working.clear();
        state.candidates.clear();

        for &(dist, pid) in entry_points {
            if !state.is_visited(pid) {
                state.mark_visited(pid);
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

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

            // read neighbours from construction graph
            let neighbours = graph.get_neighbours(current_id);

            for i in 0..max_neighbours {
                if i >= neighbours.len() {
                    break;
                }

                let neighbour = neighbours[i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;

                if self.layer_assignments[neighbour_id] < layer {
                    continue;
                }

                if state.is_visited(neighbour_id) {
                    continue;
                }

                state.mark_visited(neighbour_id);

                let dist = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(query_node, neighbour_id),
                    Dist::Cosine => self.cosine_distance(query_node, neighbour_id),
                });

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

    /// Select neighbours using heuristic (Algorithm 4 from paper)
    ///
    /// The heuristic filters candidates to avoid clustered connections. A
    /// candidate is selected if it's closer to the query node than to any
    /// already-selected neighbour.
    ///
    /// **Optimisation**: Uses `SearchState` scratch buffers to avoid
    /// allocation.
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `candidates` - Initial candidate neighbours (sorted by distance)
    /// * `layer` - Current layer
    /// * `state` - SearchState providing scratch buffers
    ///
    /// ### Returns
    ///
    /// Filtered list of neighbours (at most M or M*2)
    fn select_heuristic(
        &self,
        node: usize,
        candidates: &[(OrderedFloat<T>, usize)],
        layer: u8,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        // reuse the scratch buffers from the state
        state.scratch_working.clear();
        state.scratch_discarded.clear();

        // increment visit generation effectively resetting "visited" for this
        // local op
        state.visit_id = state.visit_id.wrapping_add(1);
        if state.visit_id == 0 {
            state.visited.fill(0);
            state.visit_id = 1;
        }

        // 1. filter candidates and mark them as visited
        for &candidate in candidates {
            if candidate.1 != node && !state.is_visited(candidate.1) {
                state.scratch_working.push(candidate);
                state.mark_visited(candidate.1);
            }
        }

        // 2. extend candidates (if enabled)
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
                    if neighbour_id == node || state.is_visited(neighbour_id) {
                        continue;
                    }

                    state.mark_visited(neighbour_id);

                    let dist = OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(node, neighbour_id),
                        Dist::Cosine => self.cosine_distance(node, neighbour_id),
                    });

                    state.scratch_working.push((dist, neighbour_id));
                }
            }

            state.scratch_working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }

        // 3. heuristic Selection
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
        let mut result = Vec::with_capacity(max_neighbours);

        for candidate in state.scratch_working.iter() {
            if result.len() >= max_neighbours {
                break;
            }

            let (cand_dist, cand_id) = *candidate;

            if cand_id == node {
                continue;
            }

            let mut closer_to_query = true;
            for &(_, result_id) in &result {
                let dist_to_result = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(cand_id, result_id),
                    Dist::Cosine => self.cosine_distance(cand_id, result_id),
                });

                if dist_to_result < cand_dist {
                    closer_to_query = false;
                    break;
                }
            }

            if closer_to_query {
                result.push((cand_dist, cand_id));
            } else if self.keep_pruned {
                state.scratch_discarded.push((cand_dist, cand_id));
            }
        }

        // 4. add back pruned connections if we have space
        if self.keep_pruned {
            for &candidate in state.scratch_discarded.iter() {
                if result.len() >= max_neighbours {
                    break;
                }
                result.push(candidate);
            }
        }

        result
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
    #[inline]
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
    #[inline]
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
            if !state.is_visited(pid) {
                state.mark_visited(pid);
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

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

            // FIX: Use node's assigned layer, not search layer
            let node_layer = self.layer_assignments[current_id];
            let max_neighbours = if node_layer == 0 { self.m * 2 } else { self.m };
            let offset = self.neighbour_offsets[current_id];

            for i in 0..max_neighbours {
                let neighbour = self.neighbours_flat[offset + i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;

                // FIX: Only visit nodes that exist at this layer
                if self.layer_assignments[neighbour_id] < layer {
                    continue;
                }

                if state.is_visited(neighbour_id) {
                    continue;
                }

                state.mark_visited(neighbour_id);

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
            Dist::Euclidean => self.euclidean_distance_to_query(idx, query),
            Dist::Cosine => self.cosine_distance_to_query(idx, query, query_norm),
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

    #[test]
    fn test_hnsw_small_m_layer_bug() {
        // This test catches a bug where nodes assigned to higher layers have
        // fewer neighbours stored (m) but layer 0 search incorrectly tries to
        // read 2*m neighbours, causing an index out of bounds panic.
        //
        // The bug occurs when:
        // 1. Node is assigned to layer > 0, so stores m neighbours
        // 2. During layer 0 search, code calculates max_neighbours = m * 2
        // 3. Attempts to read beyond actual storage

        let n = 929; // Exact size that triggered the bug
        let dim = 15;
        let m = 16; // Small m that makes the bug more likely
        let ef_construction = 200;

        // Create dataset similar to PCA embeddings
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push(((i * dim + j) as f32).sin() * 0.1);
            }
        }
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        // Build with parameters that triggered the original bug
        let index = HnswIndex::<f32>::build(mat.as_ref(), m, ef_construction, "cosine", 42, false);

        // Query with ef_search = 100 (also from original failing case)
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 15, 100);

        assert_eq!(indices.len(), 15);
        assert_eq!(distances.len(), 15);

        // Verify results are valid
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1], "Distances not sorted");
        }

        for &idx in &indices {
            assert!(idx < n, "Invalid index returned");
        }
    }

    #[test]
    fn test_hnsw_varying_m_values() {
        // Test multiple small m values to ensure robustness
        let n = 500;
        let dim = 10;

        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        for m in [8, 12, 16, 20] {
            let index = HnswIndex::<f32>::build(mat.as_ref(), m, 200, "euclidean", 42, false);

            let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
            let (indices, _) = index.query(&query, 10, 50);

            assert_eq!(indices.len(), 10, "Failed with m = {}", m);
        }
    }

    // This was needed to reproduce the nasty bug
    // #[test]
    // #[should_panic(expected = "index out of bounds")]
    // fn test_hnsw_query_layer_bug() {
    //     // Exact reproduction of R failure case
    //     let n = 929;
    //     let dim = 15;
    //     let m = 16;
    //     let ef_construction = 200;

    //     // Create data matching PCA output
    //     let mut data = Vec::with_capacity(n * dim);
    //     for i in 0..n {
    //         for j in 0..dim {
    //             data.push(((i * 37 + j * 13) as f32).sin() * 0.5);
    //         }
    //     }
    //     let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

    //     // Build with EXACT parameters from failing R call
    //     let index = HnswIndex::<f32>::build(
    //         mat.as_ref(),
    //         m,
    //         ef_construction,
    //         "cosine", // Default from R
    //         42,
    //         false,
    //     );

    //     // Query ALL nodes with k=16 (k=15 + 1 for self-removal)
    //     // This ensures we traverse all parts of the graph
    //     for i in 0..n {
    //         let query: Vec<f32> = mat.row(i).iter().copied().collect();
    //         let _ = index.query(&query, 16, 100);
    //     }
    // }
}
