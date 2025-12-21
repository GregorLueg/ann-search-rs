#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod exhaustive;
pub mod hnsw;
pub mod ivf;
pub mod lsh;
pub mod nndescent;
pub mod quantised;
pub mod synthetic;
pub mod utils;

use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::{
    iter::Sum,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use thousands::*;

use crate::annoy::*;
use crate::exhaustive::*;
use crate::hnsw::*;
use crate::ivf::*;
use crate::lsh::*;
use crate::nndescent::*;
use crate::quantised::ivf_sq8::*;
use crate::utils::dist::*;

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
/// * `k` - Number of neighbours to return
/// * `index` - The AnnoyIndex to query.
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
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
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
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
/// * `ef_search` -
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
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let query_vec: Vec<T> = query_mat.row(i).iter().copied().collect();
                let result = index.query(&query_vec, k, ef_search);

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
                let (indices, _) = index.query(&query_vec, k, ef_search);

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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
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

/////////
// LSH //
/////////

/// Build the LSH index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine"
/// * `num_tables` - Number of HashMaps to use (usually something 20 to 100)
/// * `bits_per_hash` - How many bits per hash. Lower values (8) usually yield
///   better Recall with higher query time; higher values (16) have worse Recall
///   but faster query time
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// The ready LSH index for querying
pub fn build_lsh_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    num_tables: usize,
    bits_per_hash: usize,
    seed: usize,
) -> LSHIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    LSHIndex<T>: LSHQuery<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
    LSHIndex::new(mat, metric, num_tables, bits_per_hash, seed)
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
pub fn query_lsh_index<T>(
    query_mat: MatRef<T>,
    index: &LSHIndex<T>,
    k: usize,
    max_candidates: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    LSHIndex<T>: LSHQuery<T>,
{
    let n_samples = query_mat.nrows();
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>, bool)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let result = index.query_row(query_mat.row(i), k, max_candidates);
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

        let mut random: usize = 0;
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(results.len());
        let mut distances: Vec<Vec<T>> = Vec::with_capacity(results.len());
        for (idx, dist, rnd) in results {
            if rnd {
                random += 1;
            };
            indices.push(idx);
            distances.push(dist);
        }

        if (random as f32) / (n_samples as f32) >= 0.01 {
            println!("More than 1% of samples were not represented in the buckets.");
            println!("Please verify underlying data");
        }
        (indices, Some(distances))
    } else {
        let results: Vec<(Vec<usize>, _, bool)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let result = index.query_row(query_mat.row(i), k, max_candidates);
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

        let mut random: usize = 0;
        let mut indices: Vec<Vec<usize>> = Vec::with_capacity(results.len());
        for (idx, _, rnd) in results {
            if rnd {
                random += 1;
            };
            indices.push(idx);
        }

        if (random as f32) / (n_samples as f32) >= 0.01 {
            println!("More than 1% of samples were not represented in the buckets.");
            println!("Please verify underlying data");
        }
        (indices, None)
    }
}

/////////
// IVF //
/////////

/// Build an IVF index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `nlist` - Number of clusters to create
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfIndex`.
pub fn build_ivf_index<T>(
    mat: MatRef<T>,
    nlist: usize,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let n = mat.nrows();
    let dim = mat.ncols();

    let mut vectors_flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors_flat.extend(mat.row(i).iter().cloned());
    }

    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    let norms = if ann_dist == Dist::Cosine {
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

    IvfIndex::build(
        vectors_flat,
        dim,
        n,
        norms,
        ann_dist,
        nlist,
        max_iters,
        seed,
        verbose,
    )
}

/// Helper function to query a given IVF index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built IVF index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to min(nlist/10, 10))
///   Higher values improve recall at the cost of speed
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
pub fn query_ivf_index<T>(
    query_mat: MatRef<T>,
    index: &IvfIndex<T>,
    k: usize,
    nprobe: Option<usize>,
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
                let (neighbours, dists) = index.query_row(query_mat.row(i), k, nprobe);

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
                let (neighbours, _) = index.query_row(query_mat.row(i), k, nprobe);

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

/////////////
// IVF-SQ8 //
/////////////

/// Build an IVF-SQ8 index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `nlist` - Number of clusters to create
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfSq8Index`.
///
/// ### Note
///
/// Currently only supports Euclidean distance.
pub fn build_ivf_sq8_index<T>(
    mat: MatRef<T>,
    nlist: usize,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let n = mat.nrows();
    let dim = mat.ncols();
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    let mut vectors_flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors_flat.extend(mat.row(i).iter().cloned());
    }

    IvfSq8Index::build(
        vectors_flat,
        dim,
        n,
        nlist,
        ann_dist,
        max_iters,
        seed,
        verbose,
    )
}

/// Helper function to query a given IVF-SQ8 index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built IVF-SQ8 index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to 20% of nlist)
///   Higher values improve recall at the cost of speed
/// * `return_dist` - Shall the inner product scores be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional inner_product_scores)`
pub fn query_ivf_sq8_index<T>(
    query_mat: MatRef<T>,
    index: &IvfSq8Index<T>,
    k: usize,
    nprobe: Option<usize>,
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
                let (neighbours, scores) = index.query_row(query_mat.row(i), k, nprobe);
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
                (neighbours, scores)
            })
            .collect();
        let (indices, scores) = results.into_iter().unzip();
        (indices, Some(scores))
    } else {
        let indices: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let (neighbours, _) = index.query_row(query_mat.row(i), k, nprobe);
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
