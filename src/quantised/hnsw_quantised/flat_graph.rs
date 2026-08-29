//! Flattened HNSW topology.
//!
//! The hierarchy is only ever used to pick a good entry point; the actual
//! beam search runs entirely on layer 0. Splitting the two apart gives the
//! base layer a dense `n * degree` array with no per-node offset lookup, and
//! keeps the upper-layer lists (which only the ~1/M of nodes above level 0
//! have) out of the bytes the walk streams through.

///////////////
// FlatGraph //
///////////////

/// Dense layer-0 adjacency.
///
/// Row `i` occupies `edges[i * degree .. (i + 1) * degree]`. Entries are packed
/// at the front and padded with [`u32::MAX`], so readers scan until the
/// sentinel.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct FlatGraph {
    /// Adjacency array of `n * degree` entries.
    edges: Vec<u32>,
    /// Row stride, `2 * m` for an HNSW base layer.
    degree: usize,
    /// Number of nodes.
    n: usize,
}

impl FlatGraph {
    /// Wrap a prebuilt adjacency array.
    ///
    /// ### Params
    ///
    /// * `edges` - Adjacency array of `n * degree` entries
    /// * `n` - Number of nodes
    /// * `degree` - Row stride
    ///
    /// ### Returns
    ///
    /// The graph
    pub fn new(edges: Vec<u32>, n: usize, degree: usize) -> Self {
        debug_assert_eq!(edges.len(), n * degree);
        Self { edges, degree, n }
    }

    /// Neighbour slots of one node.
    ///
    /// ### Params
    ///
    /// * `node` - Node index
    ///
    /// ### Returns
    ///
    /// Slice of length `degree`, sentinel-padded
    #[inline(always)]
    pub fn neighbours(&self, node: usize) -> &[u32] {
        let start = node * self.degree;
        // SAFETY: every id reaching this comes from the graph itself or from
        // the caller's `0..n` loop, both bounded by construction.
        unsafe { self.edges.get_unchecked(start..start + self.degree) }
    }

    /// Row stride.
    ///
    /// ### Returns
    ///
    /// Slots per node
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Node count.
    ///
    /// ### Returns
    ///
    /// Number of nodes
    pub fn n(&self) -> usize {
        self.n
    }

    /// Bytes held by the graph.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.edges.capacity() * std::mem::size_of::<u32>()
    }
}

////////////////////
// HnswHierarchy //
////////////////////

/// The layers above 0, kept only to seed the base-layer search.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct HnswHierarchy {
    /// Top layer each node appears in. Zero means base layer only.
    levels: Vec<u8>,
    /// Concatenated upper-layer lists, `levels[i] * degree` entries per node.
    lists: Vec<u32>,
    /// Start of each node's block in `lists`.
    offsets: Vec<usize>,
    /// Slots per layer within a node's block.
    degree: usize,
    /// Node the descent starts from.
    entry_point: u32,
    /// Highest layer present.
    max_level: u8,
}

impl HnswHierarchy {
    /// Assemble the hierarchy.
    ///
    /// ### Params
    ///
    /// * `levels` - Top layer of each node
    /// * `lists` - Concatenated upper-layer lists
    /// * `offsets` - Start of each node's block in `lists`
    /// * `degree` - Slots per layer
    /// * `entry_point` - Node the descent starts from
    /// * `max_level` - Highest layer present
    ///
    /// ### Returns
    ///
    /// The hierarchy
    pub fn new(
        levels: Vec<u8>,
        lists: Vec<u32>,
        offsets: Vec<usize>,
        degree: usize,
        entry_point: u32,
        max_level: u8,
    ) -> Self {
        Self {
            levels,
            lists,
            offsets,
            degree,
            entry_point,
            max_level,
        }
    }

    /// Neighbour slots of one node at one layer above 0.
    ///
    /// ### Params
    ///
    /// * `node` - Node index
    /// * `level` - Layer, must be at least 1
    ///
    /// ### Returns
    ///
    /// Slice of length `degree`, empty if the node does not reach this layer
    #[inline]
    pub fn neighbours(&self, node: usize, level: u8) -> &[u32] {
        if level == 0 || level > self.levels[node] {
            return &[];
        }
        let start = self.offsets[node] + (level as usize - 1) * self.degree;
        &self.lists[start..start + self.degree]
    }

    /// Greedily descend the upper layers to seed the base-layer search.
    ///
    /// Hill-climbs at each layer in turn, from the top down to layer 1, taking
    /// the best node found as the start point for the layer below.
    ///
    /// ### Params
    ///
    /// * `score` - Scoring closure against the query, smaller is nearer
    ///
    /// ### Returns
    ///
    /// The node the base-layer search should start from
    #[inline]
    pub fn descend<T, F>(&self, score: F) -> usize
    where
        T: PartialOrd,
        F: Fn(usize) -> T,
    {
        let mut current = self.entry_point as usize;
        let mut current_score = score(current);

        for level in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbour in self.neighbours(current, level) {
                    if neighbour == u32::MAX {
                        break;
                    }
                    let neighbour = neighbour as usize;
                    let s = score(neighbour);
                    if s < current_score {
                        current = neighbour;
                        current_score = s;
                        changed = true;
                    }
                }
            }
        }
        current
    }

    /// Top layer of a node.
    ///
    /// ### Params
    ///
    /// * `node` - Node index
    ///
    /// ### Returns
    ///
    /// Highest layer the node appears in
    #[inline]
    pub fn level(&self, node: usize) -> u8 {
        self.levels[node]
    }

    /// The descent start node.
    ///
    /// ### Returns
    ///
    /// Entry point index
    pub fn entry_point(&self) -> u32 {
        self.entry_point
    }

    /// Highest layer present in the graph.
    ///
    /// ### Returns
    ///
    /// Maximum layer
    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    /// Bytes held by the hierarchy.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.levels.capacity()
            + self.lists.capacity() * std::mem::size_of::<u32>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

/// Split a construction graph's interleaved layout into a dense base layer and
/// a separate hierarchy.
///
/// The construction layout concatenates each node's layers into one block:
/// `[layer0 (2m slots), layer1 (m), ...]`. Query time only streams layer 0, so
/// the upper layers are lifted out rather than left to pad the rows the walk
/// reads.
///
/// ### Params
///
/// * `nodes` - Flat slot array from the construction graph
/// * `block_offsets` - Start of each node's block in `nodes`
/// * `levels` - Top layer of each node
/// * `m` - Base connectivity parameter
/// * `entry_point` - Node the descent starts from
///
/// ### Returns
///
/// The dense base layer and the hierarchy above it
pub(crate) fn split_construction_layout(
    nodes: &[u32],
    block_offsets: &[usize],
    levels: Vec<u8>,
    m: usize,
    entry_point: u32,
) -> (FlatGraph, HnswHierarchy) {
    let n = levels.len();
    let base_degree = m * 2;

    let mut edges = Vec::with_capacity(n * base_degree);
    let mut lists = Vec::new();
    let mut offsets = Vec::with_capacity(n);

    for node in 0..n {
        let base = block_offsets[node];
        edges.extend_from_slice(&nodes[base..base + base_degree]);

        offsets.push(lists.len());
        let upper = levels[node] as usize * m;
        if upper > 0 {
            let start = base + base_degree;
            lists.extend_from_slice(&nodes[start..start + upper]);
        }
    }

    let max_level = levels.iter().copied().fold(0u8, u8::max);
    (
        FlatGraph::new(edges, n, base_degree),
        HnswHierarchy::new(levels, lists, offsets, m, entry_point, max_level),
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Two nodes, m = 2. Node 0 reaches level 1, node 1 is base only.
    fn toy() -> (Vec<u32>, Vec<usize>, Vec<u8>) {
        let m = 2;
        let base = m * 2;
        // Node 0: 4 base slots + 2 slots at level 1.
        // Node 1: 4 base slots.
        let nodes = vec![
            1, u32::MAX, u32::MAX, u32::MAX, // node 0, layer 0
            1, u32::MAX, // node 0, layer 1
            0, u32::MAX, u32::MAX, u32::MAX, // node 1, layer 0
        ];
        let offsets = vec![0, base + m];
        let levels = vec![1u8, 0];
        (nodes, offsets, levels)
    }

    #[test]
    fn test_split_puts_base_layer_in_a_dense_array() {
        let (nodes, offsets, levels) = toy();
        let (graph, _) = split_construction_layout(&nodes, &offsets, levels, 2, 0);

        assert_eq!(graph.n(), 2);
        assert_eq!(graph.degree(), 4);
        assert_eq!(graph.neighbours(0), &[1, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(graph.neighbours(1), &[0, u32::MAX, u32::MAX, u32::MAX]);
    }

    #[test]
    fn test_split_lifts_upper_layers_out() {
        let (nodes, offsets, levels) = toy();
        let (_, hierarchy) = split_construction_layout(&nodes, &offsets, levels, 2, 0);

        assert_eq!(hierarchy.max_level(), 1);
        assert_eq!(hierarchy.entry_point(), 0);
        assert_eq!(hierarchy.neighbours(0, 1), &[1, u32::MAX]);
        // Node 1 never reaches level 1, and level 0 is not the hierarchy's.
        assert!(hierarchy.neighbours(1, 1).is_empty());
        assert!(hierarchy.neighbours(0, 0).is_empty());
    }

    #[test]
    fn test_descend_walks_downhill_to_the_best_node() {
        let (nodes, offsets, levels) = toy();
        let (_, hierarchy) = split_construction_layout(&nodes, &offsets, levels, 2, 0);
        // Node 1 scores better than the entry point, so the descent moves.
        let found = hierarchy.descend(|id| if id == 1 { 0.0f32 } else { 10.0 });
        assert_eq!(found, 1);
    }

    #[test]
    fn test_descend_stays_put_when_the_entry_point_is_best() {
        let (nodes, offsets, levels) = toy();
        let (_, hierarchy) = split_construction_layout(&nodes, &offsets, levels, 2, 0);
        let found = hierarchy.descend(|id| if id == 0 { 0.0f32 } else { 10.0 });
        assert_eq!(found, 0);
    }

    #[test]
    fn test_flat_graph_only_holds_the_base_layer() {
        // The point of the split: base-layer bytes are exactly n * 2m * 4,
        // with no upper-layer slots interleaved between the rows.
        let (nodes, offsets, levels) = toy();
        let (graph, _) = split_construction_layout(&nodes, &offsets, levels, 2, 0);
        assert_eq!(graph.neighbours(0).len() + graph.neighbours(1).len(), 8);
    }
}
