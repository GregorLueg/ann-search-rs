#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod dist;
pub mod exhaustive;
pub mod fanng;
pub mod hnsw;
pub mod nndescent;
pub mod synthetic;
pub mod utils;

use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::annoy::*;
use crate::exhaustive::*;
use crate::fanng::*;
use crate::hnsw::*;
use crate::nndescent::*;
use crate::utils::*;

///////////
// Annoy //
///////////

/// Build an Annoy index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `n_trees` - Number of trees to use to build the index
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// The `AnnoyIndex`.
pub fn build_annoy_index<T>(
    mat: MatRef<T>,
    dist_metric: String,
    n_trees: usize,
    seed: usize,
) -> AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let ann_dist = parse_ann_dist(&dist_metric).unwrap_or_default();

    AnnoyIndex::new(mat, n_trees, ann_dist, seed)
}

/// Helper function to query a given Annoy index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
///
/// * `k` - Number of neighbours to return
/// * `index` - The AnnoyIndex to query.
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `search_budget` - Search budget per tree
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_annoy_index<T>(
    query_mat: MatRef<T>,
    index: &AnnoyIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let (neighbors, dists) = index.query_row(query_mat.row(i), k, search_budget);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                (neighbors, dists)
            })
            .collect();
        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let (neighbors, _) = index.query_row(query_mat.row(i), k, search_budget);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                neighbors
            })
            .collect();
        (indices, None)
    }
}

//////////
// HNSW //
//////////

/// Build an HNSW index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions.
/// * `m` - Number of bidirectional connections per layer.
/// * `ef_construction` - Size of candidate list during construction.
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// The `HnswIndex`.
pub fn build_hnsw_index<T>(
    mat: MatRef<T>,
    m: usize,
    ef_construction: usize,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> HnswIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
    HnswIndex<T>: HnswState<T>,
{
    HnswIndex::build(mat, m, ef_construction, dist_metric, seed, verbose)
}

/// Helper function to query a given HNSW index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built HNSW index
/// * `k` - Number of neighbours to return
/// * `ef_search` - Size of candidate list during search (higher = better
///   recall, slower)
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
///
/// ### Note
///
/// The distance metric is determined at index build time and cannot be changed
/// during querying.
pub fn query_hnsw_index<T>(
    query_mat: MatRef<T>,
    index: &HnswIndex<T>,
    k: usize,
    ef_search: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
    HnswIndex<T>: HnswState<T>,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query_vec: Vec<T> = query_mat.row(i).iter().copied().collect();
                let (neighbours, dists) = index.query(&query_vec, k, ef_search);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                (neighbours, dists)
            })
            .collect();

        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query_vec: Vec<T> = query_mat.row(i).iter().copied().collect();
                let (neighbours, _) = index.query(&query_vec, k, ef_search);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                neighbours
            })
            .collect();

        (indices, None)
    }
}

///////////////
// NNDescent //
///////////////

/// Build an NNDescent index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions.
/// * `k` - Number of neighbours for the k-NN graph.
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `max_iter` - Maximum iterations for the algorithm.
/// * `delta` - Early stop criterium for the algorithm.
/// * `rho` - Sampling rate for the old neighbours. Will adaptively decrease
///   over time.
/// * `diversify_prob` - Probability of pruning redundant edges (1.0 = always prune)
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Controls verbosity of the algorithm
///
/// ### Return
///
/// The `NNDescent` index.
#[allow(clippy::too_many_arguments)]
pub fn build_nndescent_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    delta: T,
    diversify_prob: T,
    k: Option<usize>,
    max_iter: Option<usize>,
    max_candidates: Option<usize>,
    n_tree: Option<usize>,
    seed: usize,
    verbose: bool,
) -> NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
    NNDescent<T>: UpdateNeighbours<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);
    NNDescent::new(
        mat,
        metric,
        k,
        max_candidates,
        max_iter,
        n_tree,
        delta,
        diversify_prob,
        seed,
        verbose,
    )
}

/// Helper function to query a given NNDescent index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built NNDescent index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
///
/// ### Note
///
/// The distance metric is determined at index build time and cannot be changed
/// during querying.
pub fn query_nndescent_index<T>(
    query_mat: MatRef<T>,
    index: &NNDescent<T>,
    k: usize,
    ef_search: Option<usize>,
    epsilon: Option<T>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    NNDescent<T>: UpdateNeighbours<T>,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query_vec: Vec<T> = query_mat.row(i).iter().copied().collect();
                let result = index.query(&query_vec, k, ef_search, epsilon);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }
                result
            })
            .collect();

        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query_vec: Vec<T> = query_mat.row(i).iter().copied().collect();
                let (indices, _) = index.query(&query_vec, k, ef_search, epsilon);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }
                indices
            })
            .collect();

        (indices, None)
    }
}

///////////
// FANNG //
///////////

/// Build a FANNG index from an embedding matrix
///
/// ### Params
///
/// * `mat` - Embedding matrix (rows = samples, cols = features)
/// * `dist_metric` - Distance metric: "euclidean" or "cosine"
/// * `faang_params` - Optional FANNG parameters (uses default if None)
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress updates during construction
///
/// ### Returns
///
/// Constructed FANNG index ready for querying
pub fn build_fanng_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    faang_params: Option<FanngParams>,
    seed: usize,
    verbose: bool,
) -> Fanng<T>
where
    T: Float + Send + Sync,
{
    let faang_params = faang_params.unwrap_or_default();

    Fanng::new(mat, dist_metric, &faang_params, seed, verbose)
}

/// Query a FANNG index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The pre-built FANNG index
/// * `k` - Number of neighbours to return
/// * `max_calcs` - Maximum number of distance calculations per query (controls
///   recall/speed trade-off)
/// * `no_shortcuts` - How many of the random indices selected in the graph
///   shall be explored.
/// * `return_dist` - Whether to return distances between points
/// * `verbose` - Print progress updates
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_fanng_index<T>(
    query_mat: MatRef<T>,
    index: &Fanng<T>,
    k: usize,
    max_calcs: usize,
    no_shortcuts: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + Send + Sync,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query: Vec<T> = query_mat.row(i).iter().copied().collect();
                let (indices, distances) = index.search_k(&query, k, max_calcs, no_shortcuts);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                (indices, distances)
            })
            .collect();

        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query: Vec<T> = query_mat.row(i).iter().copied().collect();
                let (indices, _) = index.search_k(&query, k, max_calcs, no_shortcuts);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }

                indices
            })
            .collect();

        (indices, None)
    }
}

////////////////
// Exhaustive //
////////////////

/// Build an exhaustive index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine"
///
/// ### Returns
///
/// The initialised `ExhausiveIndex`
pub fn build_exhaustive_index<T>(mat: MatRef<T>, dist_metric: &str) -> ExhaustiveIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);
    ExhaustiveIndex::new(mat, metric)
}

/// Query the exhaustive index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The exhaustive index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_index<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let result = index.query_row(query_mat.row(i), k);
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }
                result
            })
            .collect();
        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let (neighbors, _) = index.query_row(query_mat.row(i), k);
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            " Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            n_samples.separate_with_underscores()
                        );
                    }
                }
                neighbors
            })
            .collect();
        (indices, None)
    }
}
