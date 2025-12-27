#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod exhaustive;
pub mod gpu;
pub mod hnsw;
pub mod ivf;
pub mod lsh;
pub mod nndescent;
pub mod quantised;
pub mod utils;

use cubecl::prelude::*;
use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::ops::AddAssign;
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
use crate::gpu::exhaustive_gpu::*;
use crate::gpu::ivf_gpu::*;
use crate::hnsw::*;
use crate::ivf::*;
use crate::lsh::*;
use crate::nndescent::*;
use crate::quantised::{ivf_opq::*, ivf_pq::*, ivf_sq8::*};

use crate::utils::dist::*;

////////////
// Helper //
////////////

/// Helper function to execute parallel queries across samples
///
/// ### Params
///
/// * `n_samples` - Number of samples to query
/// * `return_dist` - Whether to return distances alongside indices
/// * `verbose` - Print progress information every 100,000 samples
/// * `query_fn` - Closure that takes a sample index and returns (indices,
///   distances)
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
fn query_parallel<T, F>(
    n_samples: usize,
    return_dist: bool,
    verbose: bool,
    query_fn: F,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Send,
    F: Fn(usize) -> (Vec<usize>, Vec<T>) + Sync,
{
    let counter = Arc::new(AtomicUsize::new(0));

    let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let result = query_fn(i);
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
            result
        })
        .collect();

    if return_dist {
        let (indices, distances) = results.into_iter().unzip();
        (indices, Some(distances))
    } else {
        let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
        (indices, None)
    }
}

/// Helper function to execute parallel queries with boolean flags
///
/// ### Params
///
/// * `n_samples` - Number of samples to query
/// * `return_dist` - Whether to return distances alongside indices
/// * `verbose` - Print progress information every 100,000 samples
/// * `query_fn` - Closure that takes a sample index and returns (indices,
///   distances, flag)
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
///
/// ### Note
///
/// This variant tracks boolean flags returned by the query function. If more
/// than 1% of queries return true flags, a warning is printed. Used primarily
/// for LSH queries where the flag indicates samples not represented in hash
/// buckets.
fn query_parallel_with_flags<T, F>(
    n_samples: usize,
    return_dist: bool,
    verbose: bool,
    query_fn: F,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Send,
    F: Fn(usize) -> (Vec<usize>, Vec<T>, bool) + Sync,
{
    let counter = Arc::new(AtomicUsize::new(0));

    let results: Vec<(Vec<usize>, Vec<T>, bool)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let result = query_fn(i);
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
        }
        indices.push(idx);
        distances.push(dist);
    }

    if (random as f32) / (n_samples as f32) >= 0.01 {
        println!("More than 1% of samples were not represented in the buckets.");
        println!("Please verify underlying data");
    }

    if return_dist {
        (indices, Some(distances))
    } else {
        (indices, None)
    }
}

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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, search_budget)
    })
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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
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
    query_parallel_with_flags(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, max_candidates)
    })
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
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfIndex::build(mat, ann_dist, nlist, max_iters, seed, verbose)
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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

///////////////
// Quantised //
///////////////

/////////////
// IVF-SQ8 //
/////////////

/// Build an IVF-SQ8 index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `nlist` - Optional number of cells to create. If not provided, defaults
///   to `sqrt(n)`.
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
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfSq8Index::build(mat, nlist, ann_dist, max_iters, seed, verbose)
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
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

////////////
// IVF-PQ //
////////////

/// Build an IVF-PQ index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `nlist` - Number of IVF clusters to create
/// * `m` - Number of subspaces for product quantisation (dim must be divisible
///   by m)
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `dist_metric` - Distance metric ("euclidean" or "cosine")
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfPqIndex`.
#[allow(clippy::too_many_arguments)]
pub fn build_ivf_pq_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    m: usize,
    max_iters: Option<usize>,
    n_pq_centroids: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfPqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfPqIndex::build(
        mat,
        nlist,
        m,
        ann_dist,
        max_iters,
        n_pq_centroids,
        seed,
        verbose,
    )
}

/// Helper function to query a given IVF-PQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built IVF-PQ index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to 15% of nlist)
///   Higher values improve recall at the cost of speed
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_ivf_pq_index<T>(
    query_mat: MatRef<T>,
    index: &IvfPqIndex<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

/////////////
// IVF-OPQ //
/////////////

/// Build an IVF-OPQ index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `nlist` - Number of IVF clusters to create
/// * `m` - Number of subspaces for product quantisation (dim must be divisible
///   by m)
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `dist_metric` - Distance metric ("euclidean" or "cosine")
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfOpqIndex`.
#[allow(clippy::too_many_arguments)]
pub fn build_ivf_opq_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    m: usize,
    max_iters: Option<usize>,
    n_opq_centroids: Option<usize>,
    n_opq_iter: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfOpqIndex::build(
        mat,
        nlist,
        m,
        ann_dist,
        max_iters,
        n_opq_iter,
        n_opq_centroids,
        seed,
        verbose,
    )
}

/// Helper function to query a given IVF-OPQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built IVF-OPQ index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to 15% of nlist)
///   Higher values improve recall at the cost of speed
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_ivf_opq_index<T>(
    query_mat: MatRef<T>,
    index: &IvfOpqIndex<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

/////////
// GPU //
/////////

////////////////////
// Exhaustive GPU //
////////////////////

/// Build an exhaustive GPU index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine"
/// * `device` - The GPU device to use
///
/// ### Returns
///
/// The initialised `ExhaustiveIndexGpu`
pub fn build_exhaustive_index_gpu<T, R>(
    mat: MatRef<T>,
    dist_metric: &str,
    device: R::Device,
) -> ExhaustiveIndexGpu<T, R>
where
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive,
    R: Runtime,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
    ExhaustiveIndexGpu::new(mat, metric, device)
}

/// Query the exhaustive GPU index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The exhaustive GPU index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_index_gpu<T, R>(
    query_mat: MatRef<T>,
    index: &ExhaustiveIndexGpu<T, R>,
    k: usize,
    return_dist: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive,
    R: Runtime,
{
    let (indices, distances) = index.query_batch(query_mat, k);

    if return_dist {
        (indices, Some(distances))
    } else {
        (indices, None)
    }
}

//////////////
// IVF GPU //
//////////////

/// Build an IVF index with batched GPU acceleration
///
/// ### Params
///
/// * `mat` - Data matrix [samples, features]
/// * `nlist` - Number of clusters (defaults to √n)
/// * `max_iters` - K-means iterations (defaults to 30)
/// * `dist_metric` - "euclidean" or "cosine"
/// * `seed` - Random seed
/// * `verbose` - Print progress
/// * `device` - GPU device
pub fn build_ivf_index_gpu<T, R>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
    device: R::Device,
) -> IvfIndexGpu<T, R>
where
    R: Runtime,
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + Send + Sync,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    IvfIndexGpu::build(mat, ann_dist, nlist, max_iters, seed, verbose, device)
}

/// Query an IVF batched index
///
/// ### Params
///
/// * `query_mat` - Query matrix [samples, features]
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Clusters to search (defaults to √nlist)
/// * `return_dist` - Return distances
/// * `_verbose` - Unused
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_ivf_index_gpu<T, R>(
    query_mat: MatRef<T>,
    index: &IvfIndexGpu<T, R>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    _verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    R: Runtime,
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + Send + Sync,
{
    let (indices, distances) = index.query_batch(query_mat, k, nprobe);

    if return_dist {
        (indices, Some(distances))
    } else {
        (indices, None)
    }
}
