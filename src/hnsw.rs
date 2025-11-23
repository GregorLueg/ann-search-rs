use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::dist::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

/// Search state for HNSW queries
///
/// ### Fields
///
/// * `visited` - Boolean indicating if node was visited
/// * `candidates` - Candidates for the search
/// * `working` -
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

    /// Reset the searach state
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
/// * `max_neighbours_per_node` - Maximum neighbours per node (M or M*2)
/// * `entry_point` - Node ID with highest layer
/// * `max_layer` - Highest layer in the graph
/// * `m` - Number of connections per layer (M parameter)
/// * `ef_construction` - Size of dynamic candidate list during construction
///
/// ### Algorithm Overview
///
/// HNSW builds a hierarchical graph where:
/// 1. Each node is randomly assigned to layers (exponential distribution)
/// 2. Higher layers are sparser and enable fast coarse search
/// 3. Lower layers are denser and provide precise results
/// 4. Search starts at top layer and descends through layers
/// 5. Each layer refines the search, narrowing candidates
pub struct HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    vectors_flat: Vec<T>,
    dim: usize,
    n: usize,
    metric: Dist,
    norms: Vec<T>,
    layer_assignments: Vec<u8>,
    neighbours_flat: Vec<u32>,
    neighbour_offsets: Vec<usize>,
    entry_point: u32,
    max_layer: u8,
    m: usize,
    ef_construction: usize,
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
                ((-uniform_t.ln() * ml).floor().to_u8().unwrap()).min(15) // cap at 15 layers
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
        // base layer (0) has M * 2 neighbours, upper layers have M neighbours
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
        };

        // Build layers from top to bottom
        index.build_graph(verbose);

        let end_total = start_total.elapsed();
        if verbose {
            println!("Total HNSW build time: {:.2?}", end_total);
        }

        index
    }

    /// Build the graph structure layer by layer
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
    /// 3. Using parallel insertion for efficiency
    /// 4. Skipping entry point (already placed)
    fn build_graph(&mut self, verbose: bool) {
        let nodes_by_layer: Vec<Vec<usize>> = (0..=self.max_layer)
            .map(|layer| {
                (0..self.n)
                    .filter(|&i| self.layer_assignments[i] >= layer)
                    .collect()
            })
            .collect();

        // insert nodes layer by layer, from top to bottom
        for layer in (0..=self.max_layer).rev() {
            let start = Instant::now();
            let layer_nodes = &nodes_by_layer[layer as usize];

            if verbose {
                println!("Building layer {} with {} nodes", layer, layer_nodes.len());
            }

            // Skip entry point at max layer, insert everything else
            layer_nodes
                .iter()
                .filter(|&&node| !(node == self.entry_point as usize && layer == self.max_layer))
                .for_each(|&node| {
                    self.insert_node(node, layer);
                });

            if verbose {
                println!("  Layer {} built in {:.2?}", layer, start.elapsed(),);
            }
        }
    }

    /// Insert a single node at a given layer
    ///
    /// ### Params
    ///
    /// * `node` - Node index to insert
    /// * `insert_layer` - Layer to insert at (and all layers below)
    ///
    /// ### Algorithm Details
    ///
    /// 1. Start from entry point in top layer
    /// 2. Greedily descend through layers above insert_layer
    /// 3. At insert_layer and below:
    ///    - Search for nearest neighbours
    ///    - Connect node to selected neighbours
    ///    - Update bidirectional links
    ///    - Prune neighbours if needed
    fn insert_node(&self, node: usize, insert_layer: u8) {
        Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            // find entry points by searching from top
            let mut entry_points = vec![(OrderedFloat(T::zero()), self.entry_point as usize)];

            // traverse from top layer to insert_layer
            for layer in (insert_layer + 1..=self.max_layer).rev() {
                state.reset(self.n); // ADD THIS LINE
                entry_points = self.search_layer(node, layer, &entry_points, 1, &mut state);
            }

            // search and insert at target layer and below
            for layer in (0..=insert_layer).rev() {
                state.reset(self.n); // ADD THIS LINE
                let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
                let ef = self.ef_construction;

                let mut candidates = self.search_layer(node, layer, &entry_points, ef, &mut state);

                candidates.truncate(max_neighbours);
                self.connect_neighbours(node, &candidates, layer);

                for &(_, neighbour_id) in &candidates {
                    self.prune_and_connect(neighbour_id, node, layer);
                }

                entry_points = candidates;
            }
        });
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
    ///
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

        // initialise with entry points
        for &(dist, pid) in entry_points {
            if !state.visited[pid] {
                state.visited[pid] = true;
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

        let mut furthest_dist = entry_points
            .iter()
            .map(|(d, _)| *d)
            .max()
            .unwrap_or(OrderedFloat(T::infinity()));

        // greedy search
        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist {
                break;
            }

            // check neighbours
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
                        // Remove furthest
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

        for (i, &(_, neighbour_id)) in neighbours.iter().take(max_neighbours).enumerate() {
            neighbours_slice[i] = neighbour_id as u32;
        }
    }

    /// Prune and add bidirectional connection from neighbour to node
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Existing neighbour to update
    /// * `new_node` - New node to add as neighbour
    /// * `layer` - Current layer
    ///
    /// ### Algorithm Details
    ///
    /// 1. Read current neighbours of neighbour_id
    /// 2. Add new_node to the list
    /// 3. Sort by distance and keep best M (or M*2) connections
    /// 4. Write back updated neighbour list
    fn prune_and_connect(&self, neighbour_id: usize, new_node: usize, layer: u8) {
        let offset = self.neighbour_offsets[neighbour_id];
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        // read current neighbours
        let neighbours_slice = unsafe {
            std::slice::from_raw_parts(self.neighbours_flat.as_ptr().add(offset), max_neighbours)
        };

        let mut curr: Vec<(OrderedFloat<T>, usize)> = neighbours_slice
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

        // Add new connection
        let new_dist = unsafe {
            OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, new_node),
                Dist::Cosine => self.cosine_distance(neighbour_id, new_node),
            })
        };

        curr.push((new_dist, new_node));
        curr.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        curr.truncate(max_neighbours);

        // write back
        let neighbours_slice_mut = unsafe {
            std::slice::from_raw_parts_mut(
                self.neighbours_flat.as_ptr().add(offset) as *mut u32,
                max_neighbours,
            )
        };

        for (i, &(_, node_id)) in curr.iter().enumerate() {
            neighbours_slice_mut[i] = node_id as u32;
        }

        // fill rest with sentinel
        #[allow(clippy::needless_range_loop)]
        for i in curr.len()..max_neighbours {
            neighbours_slice_mut[i] = u32::MAX;
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

            // Compute query norm if needed
            let query_norm = if self.metric == Dist::Cosine {
                query
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            } else {
                T::one()
            };

            // Start from entry point
            let entry_dist =
                self.compute_query_distance(query, self.entry_point as usize, query_norm);
            let mut entry_points = vec![(OrderedFloat(entry_dist), self.entry_point as usize)];

            // Traverse layers from top to bottom
            for layer in (1..=self.max_layer).rev() {
                state.reset(self.n); // ADD THIS LINE
                entry_points =
                    self.search_layer_query(query, query_norm, layer, &entry_points, 1, &mut state);
            }

            // Search base layer with ef_search
            state.reset(self.n); // ADD THIS LINE
            let mut candidates = self.search_layer_query(
                query,
                query_norm,
                0,
                &entry_points,
                ef_search.max(k),
                &mut state,
            );

            // Return top k
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

        // Initialise with entry points
        for &(dist, pid) in entry_points {
            if !state.visited[pid] {
                state.visited[pid] = true;
                state.candidates.push(Reverse((dist, pid)));
                state.working.push((dist, pid));
            }
        }

        let mut furthest_dist = entry_points
            .iter()
            .map(|(d, _)| *d)
            .max()
            .unwrap_or(OrderedFloat(T::infinity()));

        // Greedy search
        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist {
                break;
            }

            // Examine neighbours
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
        // 5 points in 3D space
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
        let _ = HnswIndex::<f32>::build(
            mat.as_ref(),
            16,  // m
            100, // ef_construction
            "euclidean",
            42,
            false,
        );
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

        // Query with point 0
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

        // Should find point 0 first
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Results should be sorted
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
    fn test_hnsw_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10, 50);

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_hnsw_ef_search_impact() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![0.9, 0.1, 0.0];

        // Higher ef_search should maintain or improve results
        let (indices1, _) = index.query(&query, 3, 10);
        let (indices2, _) = index.query(&query, 3, 100);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_hnsw_m_parameter() {
        let mat = create_simple_matrix();

        let index_small = HnswIndex::<f32>::build(mat.as_ref(), 4, 100, "euclidean", 42, false);

        // Larger M
        let index_large = HnswIndex::<f32>::build(mat.as_ref(), 32, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _) = index_small.query(&query, 3, 50);
        let (indices2, _) = index_large.query(&query, 3, 50);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
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
    fn test_hnsw_larger_dataset() {
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, 50);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_hnsw_orthogonal_vectors() {
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "cosine", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Orthogonal vectors should have cosine distance ≈ 1
        assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hnsw_parallel_build() {
        let n = 50;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        // Build should use parallel insertion
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, 50);

        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_hnsw_invalid_metric() {
        let mat = create_simple_matrix();

        // Should fall back to Cosine for invalid metric
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "invalid_metric", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 1, 50);

        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn test_hnsw_with_f64() {
        let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mat = Mat::from_fn(2, 3, |i, j| data[i * 3 + j]);

        let index = HnswIndex::<f64>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0_f64, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hnsw_recall() {
        // Create a dataset where we know the true neighbours
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        // Point 0 is at origin
        data.extend_from_slice(&[0.0, 0.0, 0.0]);

        // Points 1-19 at increasing distances
        for i in 1..n {
            let dist = i as f32 * 0.1;
            data.extend_from_slice(&[dist, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query = vec![0.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, 100);

        // Should find point 0 and its 4 closest neighbours (1, 2, 3, 4)
        assert_eq!(indices[0], 0);

        // Check that we found some of the true nearest neighbours
        let expected_neighbours: Vec<usize> = (0..5).collect();
        let found_correct = indices
            .iter()
            .filter(|&&idx| expected_neighbours.contains(&idx))
            .count();

        // Should have high recall (at least 4 out of 5)
        assert!(found_correct >= 4);
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

        // Query from multiple threads
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
}
