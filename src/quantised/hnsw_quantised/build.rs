//! Codec-generic HNSW construction.
//!
//! The same incremental insert as [`crate::cpu::hnsw`], with the metric
//! replaced by a distance closure so the graph can be built in whatever space
//! it is going to be searched in. It shares the crate-internal construction
//! graph, which owns the striped locks and the sentinel slot layout, and
//! reimplements only the driver on top.

use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::time::Instant;
use thousands::*;

use crate::prelude::*;
use crate::quantised::hnsw_quantised::flat_graph::*;
use crate::utils::graph_utils::*;

////////////
// Consts //
////////////

/// Highest layer a node may be assigned to. Matches `cpu::hnsw`: layers are
/// `u8` and the draw is geometric with ratio `1/M`, so past this is unreachable.
const MAX_LAYER: usize = 15;

/// Level buckets at or below this size insert sequentially. The top buckets
/// hold the navigation highway, and inserting those concurrently means they
/// cannot see each other.
const PARALLEL_BUCKET_THRESHOLD: usize = 100;

/// Initial capacity of a thread's reusable search state.
const SEARCH_STATE_CAPACITY: usize = 1000;

//////////////////////
// GraphBuildParams //
//////////////////////

/// Construction settings for [`build_hierarchical_graph`].
#[derive(Clone, Copy, Debug)]
pub struct GraphBuildParams {
    /// Base connectivity. Layer 0 holds `2 * m` slots, upper layers `m`.
    pub m: usize,
    /// Beam width during construction. Higher is a better graph and a slower
    /// build.
    pub ef_construction: usize,
    /// Seed for the layer draw and the within-bucket permutation.
    pub seed: usize,
    /// Print progress per layer.
    pub verbose: bool,
}

impl GraphBuildParams {
    /// Create construction settings.
    ///
    /// ### Params
    ///
    /// * `m` - Base connectivity parameter
    /// * `ef_construction` - Construction beam width
    /// * `seed` - Random seed
    /// * `verbose` - Whether to print progress
    ///
    /// ### Returns
    ///
    /// The parameter struct
    pub fn new(m: usize, ef_construction: usize, seed: usize, verbose: bool) -> Self {
        Self {
            m,
            ef_construction,
            seed,
            verbose,
        }
    }
}

impl Default for GraphBuildParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            seed: 42,
            verbose: false,
        }
    }
}

/////////////
// Builder //
/////////////

/// Build a hierarchical graph under an arbitrary distance.
///
/// Nodes are bucketed by their own top layer and inserted highest bucket first,
/// so a node descending through the layers above its own always reads a
/// completed bucket. Within a bucket the order is permuted, because node ids
/// follow dataset order and generators emit whole populations in blocks.
///
/// ### Params
///
/// * `n` - Number of nodes
/// * `params` - Construction settings
/// * `dist` - Symmetric distance between two node ids, smaller is nearer
///
/// ### Returns
///
/// The dense base layer and the hierarchy above it
pub fn build_hierarchical_graph<T, F>(
    n: usize,
    params: &GraphBuildParams,
    dist: F,
) -> (FlatGraph, HnswHierarchy)
where
    T: AnnSearchFloat,
    F: Fn(usize, usize) -> T + Sync,
{
    let m = params.m.max(1);
    let ml = 1.0 / (m as f64).ln();
    let mut rng = SmallRng::seed_from_u64(params.seed as u64);

    let levels: Vec<u8> = (0..n)
        .map(|_| {
            let uniform: f64 = rng.random();
            let level = (-uniform.ln() * ml).floor();
            if level.is_finite() {
                (level as usize).min(MAX_LAYER) as u8
            } else {
                MAX_LAYER as u8
            }
        })
        .collect();

    let max_layer = levels.iter().copied().fold(0u8, u8::max);
    let entry_point = levels.iter().position(|&l| l == max_layer).unwrap_or(0) as u32;

    if params.verbose {
        println!(
            "Building quantised HNSW over {} nodes, M = {}, max layer {}",
            n.separate_with_underscores(),
            m,
            max_layer
        );
    }

    let threads = rayon::current_num_threads();
    let graph = ConstructionGraph::new(n, &levels, m, threads);

    let ctx = InsertContext {
        n,
        m,
        ef_construction: params.ef_construction,
        max_layer,
        entry_point: entry_point as usize,
        levels: &levels,
    };

    // The entry point acquires its edges through reverse links only: it is the
    // first node any insertion descends to, so inserting it against an empty
    // graph would be a no-op.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); max_layer as usize + 1];
    for node in (0..n).filter(|&id| id != entry_point as usize) {
        buckets[levels[node] as usize].push(node);
    }

    let start = Instant::now();
    let mut rng = SmallRng::seed_from_u64(params.seed as u64);

    for layer in (0..=max_layer as usize).rev() {
        let bucket = &mut buckets[layer];
        bucket.shuffle(&mut rng);

        let bucket_start = Instant::now();
        let parallel = bucket.len() > PARALLEL_BUCKET_THRESHOLD;

        if parallel {
            bucket.par_iter().for_each_init(
                || SearchState::new(SEARCH_STATE_CAPACITY),
                |state, &node| ctx.insert_node(node, &graph, state, &dist),
            );
        } else {
            let mut state = SearchState::new(SEARCH_STATE_CAPACITY);
            for &node in bucket.iter() {
                ctx.insert_node(node, &graph, &mut state, &dist);
            }
        }

        if params.verbose {
            println!(
                "  Layer {}: {} nodes ({}) in {:.2?}",
                layer,
                bucket.len().separate_with_underscores(),
                if parallel { "parallel" } else { "sequential" },
                bucket_start.elapsed()
            );
        }
    }

    if params.verbose {
        println!("  Total build in {:.2?}", start.elapsed());
    }

    let (nodes, block_offsets, levels) = graph.into_flat();
    split_construction_layout(&nodes, &block_offsets, levels, m, entry_point)
}

/// The invariant parts of an insertion, so the hot path does not thread eight
/// arguments through three functions.
struct InsertContext<'a> {
    /// Number of nodes.
    n: usize,
    /// Base connectivity parameter.
    m: usize,
    /// Construction beam width.
    ef_construction: usize,
    /// Highest layer present.
    max_layer: u8,
    /// Node every descent starts from.
    entry_point: usize,
    /// Top layer of each node.
    levels: &'a [u8],
}

impl InsertContext<'_> {
    /// Slots available to a node at one layer.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// `2 * m` at layer 0, `m` above
    #[inline]
    fn max_neighbours(&self, layer: u8) -> usize {
        if layer == 0 {
            self.m * 2
        } else {
            self.m
        }
    }

    /// Insert one node across every layer it belongs to.
    ///
    /// Greedily descends from the entry point through the layers above the
    /// node's own, then at each of its own layers runs an `ef_construction`
    /// search, prunes, and writes both the forward and the reverse links.
    ///
    /// ### Params
    ///
    /// * `node` - Node to insert
    /// * `graph` - Construction graph to write into
    /// * `state` - Reusable search state
    /// * `dist` - Symmetric distance closure
    fn insert_node<T, F>(
        &self,
        node: usize,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
        dist: &F,
    ) where
        T: AnnSearchFloat,
        F: Fn(usize, usize) -> T + Sync,
    {
        let node_level = self.levels[node];
        let mut current = self.entry_point;
        let mut current_dist = OrderedFloat(dist(node, current));

        for layer in (node_level + 1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbours = unsafe { graph.get_neighbours_slice(current, layer) };
                for &neighbour in neighbours {
                    if neighbour == u32::MAX {
                        break;
                    }
                    let neighbour = neighbour as usize;
                    let d = OrderedFloat(dist(node, neighbour));
                    if d < current_dist {
                        current = neighbour;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
        }

        for layer in (0..=node_level).rev() {
            state.reset(self.n);
            self.search_layer(node, layer, current, graph, state, dist);

            state.results.sort();
            state.scratch_working.clear();
            let (dists, ids) = (state.results.dists(), state.results.ids());
            for i in 0..dists.len() {
                if ids[i] != node {
                    state.scratch_working.push((OrderedFloat(dists[i]), ids[i]));
                }
            }

            let selected = self.select_neighbours(layer, state, dist);
            graph.set_neighbours(node, layer, &selected);

            for &(_, neighbour_id) in &selected {
                if neighbour_id != node && graph.node_level(neighbour_id) >= layer {
                    graph.add_neighbour_with_pruning(neighbour_id, layer, node, dist);
                }
            }

            if let Some(&(_, closest)) = selected.first() {
                current = closest;
            }
        }
    }

    /// Beam search at one layer during construction.
    ///
    /// Leaves the `ef_construction` closest nodes in `state.results`,
    /// heap-ordered rather than sorted.
    ///
    /// ### Params
    ///
    /// * `query_node` - Node being inserted
    /// * `layer` - Layer to search
    /// * `entry_node` - Starting point
    /// * `graph` - Construction graph
    /// * `state` - Reusable search state
    /// * `dist` - Symmetric distance closure
    fn search_layer<T, F>(
        &self,
        query_node: usize,
        layer: u8,
        entry_node: usize,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
        dist: &F,
    ) where
        T: AnnSearchFloat,
        F: Fn(usize, usize) -> T + Sync,
    {
        state.results.reset(self.ef_construction);
        state.candidates.clear();

        let entry_dist = dist(query_node, entry_node);
        state.mark_visited(entry_node);
        state
            .candidates
            .push(Reverse((OrderedFloat(entry_dist), entry_node)));
        state.results.push(entry_dist, entry_node);

        // `threshold()` is infinity until the heap fills, so the "not yet
        // full" case needs no separate arm: nothing is farther than infinity.
        let mut furthest = state.results.threshold();

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist.0 > furthest {
                break;
            }

            // SAFETY: benign race. Stale or torn reads are acceptable during
            // construction search, same rationale as `cpu::hnsw` and Vamana:
            // every slot is always either a valid id or a sentinel.
            let neighbours = unsafe { graph.get_neighbours_slice(current_id, layer) };

            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    continue;
                }
                let neighbour_id = neighbour as usize;
                if state.is_visited(neighbour_id) {
                    continue;
                }
                state.mark_visited(neighbour_id);

                let d = dist(query_node, neighbour_id);
                if d < furthest {
                    state
                        .candidates
                        .push(Reverse((OrderedFloat(d), neighbour_id)));
                    state.results.push(d, neighbour_id);
                    furthest = state.results.threshold();
                }
            }
        }
    }

    /// The HNSW diversity heuristic (Algorithm 4).
    ///
    /// Keeps a candidate only when no already-selected neighbour is closer to
    /// it than the query node is. Rejects are dropped rather than used to fill
    /// the remaining slots: the slack is load-bearing, and filling it saturates
    /// every list and pushes each reverse link through the quadratic pruning
    /// path.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer being built
    /// * `state` - Search state whose `scratch_working` holds the candidates,
    ///   ascending by distance and with the query node already dropped
    /// * `dist` - Symmetric distance closure
    ///
    /// ### Returns
    ///
    /// The pruned neighbour list
    fn select_neighbours<T, F>(
        &self,
        layer: u8,
        state: &SearchState<T>,
        dist: &F,
    ) -> Vec<(OrderedFloat<T>, usize)>
    where
        T: AnnSearchFloat,
        F: Fn(usize, usize) -> T + Sync,
    {
        let max_neighbours = self.max_neighbours(layer);
        let mut result = Vec::with_capacity(max_neighbours);

        for &(cand_dist, cand_id) in &state.scratch_working {
            if result.len() >= max_neighbours {
                break;
            }
            let dominated = result
                .iter()
                .any(|&(_, selected_id)| OrderedFloat(dist(cand_id, selected_id)) < cand_dist);
            if !dominated {
                result.push((cand_dist, cand_id));
            }
        }

        result
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on a line, so the nearest neighbour of `i` is `i +/- 1`.
    fn line(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn line_dist(data: &[f32]) -> impl Fn(usize, usize) -> f32 + Sync + '_ {
        move |a, b| (data[a] - data[b]) * (data[a] - data[b])
    }

    #[test]
    fn test_build_produces_a_graph_of_the_right_shape() {
        let n = 500;
        let data = line(n);
        let params = GraphBuildParams::new(8, 100, 42, false);
        let (graph, hierarchy) = build_hierarchical_graph(n, &params, line_dist(&data));

        assert_eq!(graph.n(), n);
        assert_eq!(graph.degree(), 16);
        assert!(
            hierarchy.max_level() >= 1,
            "hierarchy collapsed to one layer"
        );
        assert!((hierarchy.entry_point() as usize) < n);
    }

    #[test]
    fn test_every_node_gets_at_least_one_base_edge() {
        let n = 400;
        let data = line(n);
        let params = GraphBuildParams::new(8, 100, 7, false);
        let (graph, _) = build_hierarchical_graph(n, &params, line_dist(&data));

        for node in 0..n {
            let deg = graph
                .neighbours(node)
                .iter()
                .take_while(|&&id| id != u32::MAX)
                .count();
            assert!(deg > 0, "node {node} is isolated");
        }
    }

    #[test]
    fn test_base_degree_stays_under_budget() {
        // The diversity heuristic must leave layer-0 lists short of 2*m on
        // average; a saturated mean means rejects are being used as filler.
        let (n, m) = (400, 4);
        let data = line(n);
        let params = GraphBuildParams::new(m, 100, 42, false);
        let (graph, _) = build_hierarchical_graph(n, &params, line_dist(&data));

        let degrees: Vec<usize> = (0..n)
            .map(|node| {
                graph
                    .neighbours(node)
                    .iter()
                    .take_while(|&&id| id != u32::MAX)
                    .count()
            })
            .collect();

        assert!(degrees.iter().all(|&d| d <= m * 2));
        let mean = degrees.iter().sum::<usize>() as f64 / n as f64;
        assert!(mean < (m * 2) as f64, "mean base degree {mean}");
    }

    #[test]
    fn test_edges_are_local_on_a_line() {
        // On a 1-D line a good graph connects each node to nearby ids. This is
        // the cheapest signal that the distance closure is actually driving
        // construction rather than being ignored.
        let n = 300;
        let data = line(n);
        let params = GraphBuildParams::new(8, 200, 42, false);
        let (graph, _) = build_hierarchical_graph(n, &params, line_dist(&data));

        let mut total = 0usize;
        let mut gaps = 0usize;
        for node in 0..n {
            for &nb in graph.neighbours(node) {
                if nb == u32::MAX {
                    break;
                }
                gaps += (nb as isize - node as isize).unsigned_abs();
                total += 1;
            }
        }
        let mean_gap = gaps as f64 / total as f64;
        assert!(mean_gap < 20.0, "mean neighbour id gap {mean_gap}");
    }

    #[test]
    fn test_hierarchy_lists_only_hold_nodes_above_the_base() {
        let n = 600;
        let data = line(n);
        let params = GraphBuildParams::new(8, 100, 3, false);
        let (_, hierarchy) = build_hierarchical_graph(n, &params, line_dist(&data));

        for node in 0..n {
            for level in 1..=hierarchy.max_level() {
                let slots = hierarchy.neighbours(node, level);
                if hierarchy.level(node) < level {
                    assert!(slots.is_empty());
                    continue;
                }
                for &nb in slots {
                    if nb == u32::MAX {
                        break;
                    }
                    assert!(
                        hierarchy.level(nb as usize) >= level,
                        "node {nb} appears at level {level} it does not reach"
                    );
                }
            }
        }
    }

    #[test]
    fn test_small_input_builds() {
        for n in [1usize, 2, 5, 101] {
            let data = line(n);
            let params = GraphBuildParams::new(4, 20, 1, false);
            let (graph, _) = build_hierarchical_graph(n, &params, line_dist(&data));
            assert_eq!(graph.n(), n);
        }
    }
}
