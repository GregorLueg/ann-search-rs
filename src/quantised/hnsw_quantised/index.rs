//! The quantised HNSW index.
//!
//! Holds a codec, a dense base layer and the hierarchy above it. Both the
//! build and the search run entirely on codec scores, so the graph is
//! constructed in the space it is searched in.

use faer::RowRef;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thousands::*;

use crate::prelude::*;
use crate::quantised::hnsw_quantised::build::*;
use crate::quantised::hnsw_quantised::codec::*;
use crate::quantised::hnsw_quantised::flat_graph::*;
use crate::quantised::sq8u_codec::*;
use crate::quantised::uniform_quant::*;
use crate::utils::graph_utils::*;
use crate::utils::pack_knn_results;

////////////////////////
// HnswQuantisedIndex //
////////////////////////

/// HNSW over quantised vectors.
///
/// The hierarchy seeds the entry point, then one beam search runs over the
/// dense base layer. Every distance goes through the codec, so swapping the
/// codec swaps the storage and the arithmetic without touching the graph.
pub struct HnswQuantisedIndex<T, C>
where
    T: AnnSearchFloat,
    C: GraphCodec<T>,
{
    /// Vector storage and distance arithmetic.
    codec: C,
    /// Dense layer-0 adjacency.
    graph: FlatGraph,
    /// Layers above 0, used only to seed the base-layer search.
    hierarchy: HnswHierarchy,
    /// Number of vectors.
    n: usize,
    /// Dimensionality.
    dim: usize,
    /// Distance metric.
    metric: Dist,
    /// Base connectivity the graph was built with.
    m: usize,
    /// Construction beam width the graph was built with.
    ef_construction: usize,
    /// Original ids, identity by default. Kept for the trait surface.
    original_ids: Vec<usize>,
    /// Ties the index to its float type.
    _phantom: PhantomData<T>,
}

impl<T, C> HnswQuantisedIndex<T, C>
where
    T: AnnSearchFloat + ThreadLocalSearchState,
    C: GraphCodec<T>,
{
    /// Build an index over an already-encoded codec.
    ///
    /// ### Params
    ///
    /// * `codec` - The encoded vectors
    /// * `params` - Construction settings
    ///
    /// ### Returns
    ///
    /// The built index
    pub fn from_codec(codec: C, params: &GraphBuildParams) -> Self {
        let (n, dim, metric) = (codec.n(), codec.dim(), codec.metric());

        let start = Instant::now();
        let (graph, hierarchy) =
            build_hierarchical_graph::<T, _>(n, params, |a, b| codec.score_sym(a, b));

        if params.verbose {
            println!(
                "Quantised HNSW over {} nodes built in {:.2?}",
                n.separate_with_underscores(),
                start.elapsed()
            );
        }

        Self {
            codec,
            graph,
            hierarchy,
            n,
            dim,
            metric,
            m: params.m,
            ef_construction: params.ef_construction,
            original_ids: (0..n).collect(),
            _phantom: PhantomData,
        }
    }

    /// Query for the `k` nearest neighbours.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector, must match the index dimensionality
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width; higher is better recall and slower
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`, nearest first
    pub fn query(
        &self,
        query: &[T],
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let encoded = self.codec.encode_query(query)?;

        T::with_search_state(|state| {
            state.reset(self.n);

            let entry = self
                .hierarchy
                .descend(|id| OrderedFloat(self.codec.score(&encoded, id)));

            self.search_base_layer(&encoded, entry, ef_search.max(k).max(1), state);

            state.results.sort();
            let (dists, ids) = (state.results.dists(), state.results.ids());
            let take = k.min(dists.len());

            let indices = ids[..take].to_vec();
            let distances = dists[..take]
                .iter()
                .map(|&d| self.codec.finalise(d))
                .collect();

            Ok((indices, distances))
        })
    }

    /// Beam search over the dense base layer.
    ///
    /// Leaves the `ef` best candidates in `state.results`, heap-ordered.
    ///
    /// ### Params
    ///
    /// * `encoded` - Prepared query
    /// * `entry_node` - Starting node from the hierarchy descent
    /// * `ef` - Beam width
    /// * `state` - Reusable search state, already reset
    fn search_base_layer(
        &self,
        encoded: &C::Query,
        entry_node: usize,
        ef: usize,
        state: &mut SearchState<T>,
    ) {
        state.results.reset(ef);
        state.candidates.clear();

        let entry_dist = self.codec.score(encoded, entry_node);
        state.mark_visited(entry_node);
        state
            .candidates
            .push(Reverse((OrderedFloat(entry_dist), entry_node)));
        state.results.push(entry_dist, entry_node);

        // Infinity until the heap fills, so no separate "not yet full" arm.
        let mut furthest = state.results.threshold();

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist.0 > furthest {
                break;
            }

            for &neighbour in self.graph.neighbours(current_id) {
                if neighbour == u32::MAX {
                    break;
                }
                let neighbour_id = neighbour as usize;

                if state.is_visited(neighbour_id) {
                    continue;
                }
                state.mark_visited(neighbour_id);

                let d = self.codec.score(encoded, neighbour_id);
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

    /// Query from a matrix row.
    ///
    /// Takes the contiguous fast path when the row has unit column stride,
    /// otherwise copies into a temporary.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`, nearest first
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search);
        }
        let owned: Vec<T> = query_row.iter().cloned().collect();
        self.query(&owned, k, ef_search)
    }

    /// Number of stored vectors.
    ///
    /// ### Returns
    ///
    /// Vector count
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimensionality.
    ///
    /// ### Returns
    ///
    /// Number of features
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric.
    ///
    /// ### Returns
    ///
    /// The metric, see [`Dist`]
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Base connectivity the graph was built with.
    ///
    /// ### Returns
    ///
    /// The `m` parameter
    pub fn m(&self) -> usize {
        self.m
    }

    /// Construction beam width the graph was built with.
    ///
    /// ### Returns
    ///
    /// The `ef_construction` parameter
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// The codec.
    ///
    /// ### Returns
    ///
    /// Reference to the vector storage
    pub fn codec(&self) -> &C {
        &self.codec
    }

    /// Original ids of the stored vectors.
    ///
    /// ### Returns
    ///
    /// Identity mapping unless the index was reordered
    pub fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }

    /// Build the full self-kNN graph over the stored vectors.
    ///
    /// Each row goes through [`Self::query_stored`], which descends the
    /// hierarchy from the entry point like any other query rather than
    /// starting at the point's own node. Results therefore include the point
    /// itself, so `k` here means the same thing it does everywhere else.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector
    /// * `ef_search` - Beam width
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Whether to print progress
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        ef_search: usize,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }
                self.query_stored(i, k, ef_search)
            })
            .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

        Ok(pack_knn_results(results, return_dist))
    }

    /// Query using a stored vector as the query.
    ///
    /// Skips the query encode: the stored code already lives in the same code
    /// space, so the symmetric score is exactly what the asymmetric one would
    /// compute.
    ///
    /// ### Params
    ///
    /// * `id` - Index of the stored vector to query with
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`, nearest first
    pub fn query_stored(
        &self,
        id: usize,
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        T::with_search_state(|state| {
            state.reset(self.n);

            let entry = self
                .hierarchy
                .descend(|other| OrderedFloat(self.codec.score_sym(id, other)));

            self.search_base_layer_stored(id, entry, ef_search.max(k).max(1), state);

            state.results.sort();
            let (dists, ids) = (state.results.dists(), state.results.ids());
            let take = k.min(dists.len());

            Ok((
                ids[..take].to_vec(),
                dists[..take]
                    .iter()
                    .map(|&d| self.codec.finalise(d))
                    .collect(),
            ))
        })
    }

    /// Beam search over the base layer using a stored vector as the query.
    ///
    /// ### Params
    ///
    /// * `id` - Index of the stored query vector
    /// * `entry_node` - Starting node
    /// * `ef` - Beam width
    /// * `state` - Reusable search state, already reset
    fn search_base_layer_stored(
        &self,
        id: usize,
        entry_node: usize,
        ef: usize,
        state: &mut SearchState<T>,
    ) {
        state.results.reset(ef);
        state.candidates.clear();

        let entry_dist = self.codec.score_sym(id, entry_node);
        state.mark_visited(entry_node);
        state
            .candidates
            .push(Reverse((OrderedFloat(entry_dist), entry_node)));
        state.results.push(entry_dist, entry_node);

        let mut furthest = state.results.threshold();

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist.0 > furthest {
                break;
            }

            for &neighbour in self.graph.neighbours(current_id) {
                if neighbour == u32::MAX {
                    break;
                }
                let neighbour_id = neighbour as usize;

                if state.is_visited(neighbour_id) {
                    continue;
                }
                state.mark_visited(neighbour_id);

                let d = self.codec.score_sym(id, neighbour_id);
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
}

/////////////////////
// HnswSq8uIndex //
/////////////////////

/// The quantised HNSW over 8-bit uniformly quantised storage.
pub type HnswSq8uIndex<T> = HnswQuantisedIndex<T, Sq8uCodec<T>>;

impl<T> HnswQuantisedIndex<T, Sq8uCodec<T>>
where
    T: AnnSearchFloat + ThreadLocalSearchState,
{
    /// Calibrate, encode and build in one step.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix, rows are samples
    /// * `m` - Base connectivity parameter
    /// * `ef_construction` - Construction beam width
    /// * `metric` - Distance metric; Manhattan is not supported
    /// * `seed` - Random seed
    /// * `quant_params` - Calibration settings, `None` for the default
    /// * `verbose` - Whether to print progress
    ///
    /// ### Returns
    ///
    /// The built index, or an error on an unsupported metric or bad
    /// calibration settings
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        m: usize,
        ef_construction: usize,
        metric: &Dist,
        seed: usize,
        quant_params: Option<UniformQuantParams>,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        let (flat, n, dim) = data.into_row_major();

        let start = Instant::now();
        let codec = Sq8uCodec::new(&flat, n, dim, *metric, quant_params)?;
        if verbose {
            println!(
                "Encoded {} vectors of dimension {} in {:.2?}",
                n.separate_with_underscores(),
                dim,
                start.elapsed()
            );
        }
        // The float vectors are not retained: everything downstream of here
        // works on codes. Re-ranking against the originals is the caller's
        // job until the refiner lands.
        drop(flat);

        let params = GraphBuildParams::new(m, ef_construction, seed, verbose);
        Ok(Self::from_codec(codec, &params))
    }

    /// Bytes held by the index.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.codec.memory_usage_bytes()
            + self.graph.memory_usage_bytes()
            + self.hierarchy.memory_usage_bytes()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Well-separated clusters, each pointing in its own direction so both
    /// metrics can tell them apart.
    fn clustered(n: usize, dim: usize, n_clusters: usize) -> Vec<f32> {
        let mut s = 0xDEADBEEFu64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..n * dim)
            .map(|i| {
                let cluster = (i / dim) % n_clusters;
                let base = if (i % dim) % n_clusters == cluster {
                    1.0
                } else {
                    0.1
                };
                (base + next() * 0.15) as f32
            })
            .collect()
    }

    fn brute_force(data: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let d: f32 = data[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (d, i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn matrix(data: &[f32], n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| data[i * dim + j])
    }

    #[test]
    fn test_build_and_query_finds_self() {
        let (n, dim) = (1000, 16);
        let data = clustered(n, dim, 5);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            16,
            100,
            &Dist::SquaredEuclidean,
            42,
            None,
            false,
        )
        .unwrap();

        for i in [0usize, 250, 999] {
            let (ids, dists) = index.query(&data[i * dim..(i + 1) * dim], 5, 64).unwrap();
            assert_eq!(ids.len(), 5);
            assert_eq!(ids[0], i, "query {i} did not retrieve itself first");
            assert!(dists[0] <= dists[1]);
        }
    }

    #[test]
    fn test_recall_against_brute_force() {
        // The number that matters. A quantised graph should still find most of
        // the true neighbours; anything near chance means the codec and the
        // graph disagree about distance.
        let (n, dim, k) = (2000, 24, 10);
        let data = clustered(n, dim, 8);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            16,
            200,
            &Dist::SquaredEuclidean,
            7,
            None,
            false,
        )
        .unwrap();

        let mut hits = 0usize;
        let queries = 100;
        for i in 0..queries {
            let q = &data[i * dim..(i + 1) * dim];
            let truth = brute_force(&data, n, dim, q, k);
            let (got, _) = index.query(q, k, 128).unwrap();
            hits += got.iter().filter(|id| truth.contains(id)).count();
        }
        let recall = hits as f64 / (queries * k) as f64;
        assert!(recall > 0.85, "recall@{k} was {recall}");
    }

    /// Points with genuinely varied directions. The clustered fixture puts
    /// every cluster member within one quantisation step of its neighbours in
    /// angle, which measures the codec's error floor rather than the index.
    fn varied_directions(n: usize, dim: usize) -> Vec<f32> {
        let mut s = 0x51ED2701u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
        };
        (0..n * dim).map(|_| next()).collect()
    }

    fn cosine_brute_force(
        data: &[f32],
        n: usize,
        dim: usize,
        query: &[f32],
        k: usize,
    ) -> Vec<usize> {
        let nb: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let row = &data[i * dim..(i + 1) * dim];
                let dot: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
                let na: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                (1.0 - dot / (na * nb), i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn test_cosine_recall_against_brute_force() {
        let (n, dim, k) = (2000, 24, 10);
        let data = varied_directions(n, dim);
        let mat = matrix(&data, n, dim);
        let index =
            HnswSq8uIndex::<f32>::build(mat.as_ref(), 16, 200, &Dist::Cosine, 7, None, false)
                .unwrap();

        let mut hits = 0usize;
        let queries = 100;
        for i in 0..queries {
            let q = &data[i * dim..(i + 1) * dim];
            let truth = cosine_brute_force(&data, n, dim, q, k);
            let (got, _) = index.query(q, k, 128).unwrap();
            hits += got.iter().filter(|id| truth.contains(id)).count();
        }
        let recall = hits as f64 / (queries * k) as f64;
        assert!(recall > 0.85, "cosine recall@{k} was {recall}");
    }

    #[test]
    fn test_graph_recovers_what_an_exhaustive_codec_scan_finds() {
        // Separates the index from the codec. Quantisation error is the
        // codec's business and is bounded in its own tests; what the graph
        // owes is finding whatever the codec considers nearest. On near-
        // parallel clustered data this stays at 1.0 whilst end-to-end recall
        // against exact distances drops to ~0.76, and that gap is the
        // quantisation floor, not a graph defect.
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let (n, dim, k) = (1500, 24, 10);
            let data = clustered(n, dim, 6);
            let mat = matrix(&data, n, dim);
            let index = HnswSq8uIndex::<f32>::build(mat.as_ref(), 16, 200, &metric, 7, None, false)
                .unwrap();

            let mut hits = 0usize;
            let queries = 60;
            for i in 0..queries {
                let q = &data[i * dim..(i + 1) * dim];
                let enc = index.codec().encode_query(q).unwrap();
                let mut scored: Vec<(f32, usize)> =
                    (0..n).map(|j| (index.codec().score(&enc, j), j)).collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let truth: Vec<usize> = scored.into_iter().take(k).map(|(_, j)| j).collect();

                let (got, _) = index.query(q, k, 128).unwrap();
                hits += got.iter().filter(|id| truth.contains(id)).count();
            }
            let recall = hits as f64 / (queries * k) as f64;
            assert!(recall > 0.98, "{metric:?}: graph recall@{k} was {recall}");
        }
    }

    #[test]
    fn test_higher_ef_search_does_not_reduce_recall() {
        let (n, dim, k) = (1500, 20, 10);
        let data = clustered(n, dim, 6);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            16,
            200,
            &Dist::SquaredEuclidean,
            3,
            None,
            false,
        )
        .unwrap();

        let recall_at = |ef: usize| {
            let mut hits = 0usize;
            for i in 0..50 {
                let q = &data[i * dim..(i + 1) * dim];
                let truth = brute_force(&data, n, dim, q, k);
                let (got, _) = index.query(q, k, ef).unwrap();
                hits += got.iter().filter(|id| truth.contains(id)).count();
            }
            hits as f64 / (50 * k) as f64
        };

        let low = recall_at(16);
        let high = recall_at(256);
        assert!(
            high >= low,
            "recall fell from {low} at ef=16 to {high} at ef=256"
        );
    }

    #[test]
    fn test_generate_knn_returns_k_per_row() {
        let (n, dim, k) = (600, 16, 8);
        let data = clustered(n, dim, 4);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            12,
            100,
            &Dist::SquaredEuclidean,
            11,
            None,
            false,
        )
        .unwrap();

        let (ids, dists) = index.generate_knn(k, 64, true, false).unwrap();
        assert_eq!(ids.len(), n);
        assert!(ids.iter().all(|row| row.len() == k));
        let dists = dists.unwrap();
        assert!(dists.iter().all(|row| row.len() == k));
        // A stored vector is its own nearest neighbour.
        for (i, row) in ids.iter().enumerate() {
            assert_eq!(row[0], i, "row {i} did not find itself");
        }
    }

    #[test]
    fn test_query_stored_matches_query_with_the_same_vector() {
        let (n, dim) = (800, 16);
        let data = clustered(n, dim, 5);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            16,
            150,
            &Dist::SquaredEuclidean,
            5,
            None,
            false,
        )
        .unwrap();

        for i in [0usize, 77, 400] {
            let (a, _) = index.query(&data[i * dim..(i + 1) * dim], 10, 100).unwrap();
            let (b, _) = index.query_stored(i, 10, 100).unwrap();
            assert_eq!(a, b, "stored and encoded queries diverged at {i}");
        }
    }

    #[test]
    fn test_memory_is_well_under_the_float_index() {
        let (n, dim) = (20_000, 32);
        let data = clustered(n, dim, 8);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            16,
            100,
            &Dist::SquaredEuclidean,
            1,
            None,
            false,
        )
        .unwrap();

        let vectors_as_f32 = n * dim * std::mem::size_of::<f32>();
        let graph_bytes = n * 2 * 16 * std::mem::size_of::<u32>();
        // Codes are a quarter of the f32 vectors; the graph is unchanged, so
        // the saving is bounded by how much of the index was vectors.
        assert!(index.memory_usage_bytes() < vectors_as_f32 + graph_bytes);
    }

    #[test]
    fn test_manhattan_is_rejected_at_build() {
        let (n, dim) = (100, 8);
        let data = clustered(n, dim, 3);
        let mat = matrix(&data, n, dim);
        let got =
            HnswSq8uIndex::<f32>::build(mat.as_ref(), 8, 50, &Dist::Manhattan, 1, None, false);
        assert!(matches!(
            got,
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_query_rejects_wrong_dimension() {
        let (n, dim) = (200, 12);
        let data = clustered(n, dim, 4);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            8,
            50,
            &Dist::SquaredEuclidean,
            1,
            None,
            false,
        )
        .unwrap();
        assert!(index.query(&vec![0.0f32; dim + 1], 5, 32).is_err());
    }

    #[test]
    fn test_k_larger_than_dataset_is_clamped() {
        let (n, dim) = (40, 8);
        let data = clustered(n, dim, 2);
        let mat = matrix(&data, n, dim);
        let index = HnswSq8uIndex::<f32>::build(
            mat.as_ref(),
            8,
            50,
            &Dist::SquaredEuclidean,
            1,
            None,
            false,
        )
        .unwrap();
        let (ids, _) = index.query(&data[..dim], 500, 500).unwrap();
        assert!(ids.len() <= n);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_f64_index_builds_and_queries() {
        let (n, dim) = (500, 16);
        let data32 = clustered(n, dim, 4);
        let data: Vec<f64> = data32.iter().map(|&x| x as f64).collect();
        let mat = Mat::<f64>::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = HnswSq8uIndex::<f64>::build(
            mat.as_ref(),
            12,
            100,
            &Dist::SquaredEuclidean,
            9,
            None,
            false,
        )
        .unwrap();

        let (ids, _) = index.query(&data[..dim], 5, 64).unwrap();
        assert_eq!(ids[0], 0);
    }
}
