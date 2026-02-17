#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod ball_tree;
pub mod exhaustive;
pub mod hnsw;
pub mod ivf;
pub mod lsh;
pub mod nndescent;
pub mod prelude;
pub mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "quantised")]
pub mod quantised;

#[cfg(feature = "binary")]
pub mod binary;

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

#[cfg(feature = "gpu")]
use cubecl::prelude::*;
#[cfg(feature = "quantised")]
use std::ops::AddAssign;

#[cfg(feature = "binary")]
use bytemuck::Pod;
#[cfg(feature = "binary")]
use faer_traits::ComplexField;
#[cfg(feature = "binary")]
use std::path::Path;

use crate::annoy::*;
use crate::ball_tree::*;
use crate::exhaustive::*;
use crate::hnsw::*;
use crate::ivf::*;
use crate::lsh::*;
use crate::nndescent::*;
use crate::prelude::*;

#[cfg(feature = "binary")]
use crate::binary::{exhaustive_binary::*, exhaustive_rabitq::*, ivf_binary::*, ivf_rabitq::*};
#[cfg(feature = "gpu")]
use crate::gpu::{exhaustive_gpu::*, ivf_gpu::*};
#[cfg(feature = "quantised")]
use crate::quantised::{
    exhaustive_bf16::*, exhaustive_opq::*, exhaustive_pq::*, exhaustive_sq8::*, ivf_bf16::*,
    ivf_opq::*, ivf_pq::*, ivf_sq8::*,
};

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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
    ExhaustiveIndex::new(mat, metric)
}

/// Helper function to query a given exhaustive index
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

/// Helper function to self query an exhaustive index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - The exhaustive index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_self<T>(
    index: &ExhaustiveIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, return_dist, verbose)
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, search_budget)
    })
}

/// Helper function to self query the Annoy index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
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
pub fn query_annoy_self<T>(
    index: &AnnoyIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, search_budget, return_dist, verbose)
}

//////////////
// BallTree //
//////////////

/// Build a BallTree index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `dist_metric` - Distance metric to use
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// The `BallTreeIndex`.
pub fn build_balltree_index<T>(mat: MatRef<T>, dist_metric: String, seed: usize) -> BallTreeIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let ann_dist = parse_ann_dist(&dist_metric).unwrap_or_default();
    BallTreeIndex::new(mat, ann_dist, seed)
}

/// Helper function to query a given BallTree index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `k` - Number of neighbours to return
/// * `index` - The BallTreeIndex to query
/// * `search_budget` - Search budget (number of items to examine)
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_balltree_index<T>(
    query_mat: MatRef<T>,
    index: &BallTreeIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, search_budget)
    })
}

/// Helper function to self query the BallTree index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `k` - Number of neighbours to return
/// * `index` - The BallTreeIndex to query
/// * `search_budget` - Search budget (number of items to examine)
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_balltree_self<T>(
    index: &BallTreeIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, search_budget, return_dist, verbose)
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
    HnswIndex<T>: HnswState<T>,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
}

/// Helper function to self query the HNSW index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `k` - Number of neighbours to return
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
pub fn query_hnsw_self<T>(
    index: &HnswIndex<T>,
    k: usize,
    ef_search: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
    HnswIndex<T>: HnswState<T>,
{
    index.generate_knn(k, ef_search, return_dist, verbose)
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

/// Helper function to self query an IVF index
///
/// This function will generate a full kNN graph based on the internal data. To
/// accelerate the process, it will leverage the information on the Voronoi
/// cells under the hood and query nearby cells per given internal vector.
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
pub fn query_ivf_self<T>(
    index: &IvfIndex<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
    LSHIndex::new(mat, metric, num_tables, bits_per_hash, seed)
}

/// Helper function to query a given LSH index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The LSH index
/// * `k` - Number of neighbours to return
/// * `max_candidates` - Optional number to limit the candidate selection per
///   given table. Makes the querying faster at cost of Recall.
/// * `nprobe` - Number of additional buckets to probe per table. Will identify
///   the closest hash tables and use bit flipping to investigate these.
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
    n_probe: usize,
    max_candidates: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel_with_flags(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, max_candidates, n_probe)
    })
}

/// Helper function to self query an LSH index
///
/// ### Params
///
/// * `index` - The LSH index
/// * `k` - Number of neighbours to return
/// * `max_candidates` - Optional number to limit the candidate selection per
///   given table. Makes the querying faster at cost of Recall.
/// * `n_probe` - Optional number of additional buckets to probe per table. Will
///   identify the closest hash tables and use bit flipping to investigate
///   these. Defaults to half the number of bits.
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_lsh_self<T>(
    index: &LSHIndex<T>,
    k: usize,
    n_probe: Option<usize>,
    max_candidates: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let n_probe = n_probe.unwrap_or(index.num_bits() / 2);

    index.generate_knn(k, max_candidates, n_probe, return_dist, verbose)
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
    T: Float + FromPrimitive + Send + Sync + Sum + SimdDistance,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
}

/// Helper function to self query the NNDescent index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
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
pub fn query_nndescent_self<T>(
    index: &NNDescent<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
{
    index.generate_knn(k, ef_search, return_dist, verbose)
}

///////////////
// Quantised //
///////////////

/////////////////////
// Exhaustive-BF16 //
/////////////////////

#[cfg(feature = "quantised")]
/// Build an Exhaustive-BF16 index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `dist_metric` - Distance metric to use
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustiveIndexBf16`.
pub fn build_exhaustive_bf16_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    verbose: bool,
) -> ExhaustiveIndexBf16<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    if verbose {
        println!(
            "Building exhaustive BF16 index with {} samples",
            mat.nrows()
        );
    }
    ExhaustiveIndexBf16::new(mat, ann_dist)
}

#[cfg(feature = "quantised")]
/// Helper function to query a given Exhaustive-BF16 index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built Exhaustive-BF16 index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_bf16_index<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveIndexBf16<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a given Exhaustive-BF16 index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - Reference to the built Exhaustive-BF16 index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_bf16_self<T>(
    index: &ExhaustiveIndexBf16<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    index.generate_knn(k, return_dist, verbose)
}

////////////////////
// Exhaustive-SQ8 //
////////////////////

#[cfg(feature = "quantised")]
/// Build an Exhaustive-SQ8 index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `dist_metric` - Distance metric to use
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustiveSq8Index`.
pub fn build_exhaustive_sq8_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    verbose: bool,
) -> ExhaustiveSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    if verbose {
        println!("Building exhaustive SQ8 index with {} samples", mat.nrows());
    }
    ExhaustiveSq8Index::new(mat, ann_dist)
}

#[cfg(feature = "quantised")]
/// Helper function to query a given Exhaustive-SQ8 index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built Exhaustive-SQ8 index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_sq8_index<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveSq8Index<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a given Exhaustive-SQ8 index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - Reference to the built Exhaustive-SQ8 index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_sq8_self<T>(
    index: &ExhaustiveSq8Index<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, return_dist, verbose)
}

////////////////////
// Exhaustive-PQ //
////////////////////

#[cfg(feature = "quantised")]
/// Build an Exhaustive-PQ index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `m` - Number of subspaces for product quantisation (dim must be divisible
///   by m)
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `n_pq_centroids` - Number of centroids per subspace (defaults to 256 if None)
/// * `dist_metric` - Distance metric ("euclidean" or "cosine")
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustivePqIndex`.
#[allow(clippy::too_many_arguments)]
pub fn build_exhaustive_pq_index<T>(
    mat: MatRef<T>,
    m: usize,
    max_iters: Option<usize>,
    n_pq_centroids: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> ExhaustivePqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    ExhaustivePqIndex::build(mat, m, ann_dist, max_iters, n_pq_centroids, seed, verbose)
}

#[cfg(feature = "quantised")]
/// Helper function to query a given Exhaustive-PQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built Exhaustive-PQ index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_pq_index<T>(
    query_mat: MatRef<T>,
    index: &ExhaustivePqIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query an Exhaustive-PQ index
///
/// This function will generate a full kNN graph based on the internal data. To
/// note, during quantisation information is lost, hence, the quality of the
/// graph is reduced compared to other indices.
///
/// ### Params
///
/// * `index` - Reference to the built Exhaustive-PQ index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_pq_index_self<T>(
    index: &ExhaustivePqIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, return_dist, verbose)
}

////////////////////
// Exhaustive-OPQ //
////////////////////

#[cfg(feature = "quantised")]
/// Build an Exhaustive-OPQ index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `m` - Number of subspaces for product quantisation (dim must be divisible
///   by m)
/// * `max_iters` - Maximum k-means iterations (defaults to 30 if None)
/// * `n_pq_centroids` - Number of centroids per subspace (defaults to 256 if None)
/// * `dist_metric` - Distance metric ("euclidean" or "cosine")
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustivePqIndex`.
#[allow(clippy::too_many_arguments)]
pub fn build_exhaustive_opq_index<T>(
    mat: MatRef<T>,
    m: usize,
    max_iters: Option<usize>,
    n_pq_centroids: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> ExhaustiveOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + AddAssign,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    ExhaustiveOpqIndex::build(mat, m, ann_dist, max_iters, n_pq_centroids, seed, verbose)
}

#[cfg(feature = "quantised")]
/// Helper function to query a given Exhaustive-OPQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built Exhaustive-PQ index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_opq_index<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveOpqIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + AddAssign,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query an Exhaustive-OPQ index
///
/// This function will generate a full kNN graph based on the internal data. To
/// note, during quantisation information is lost, hence, the quality of the
/// graph is reduced compared to other indices.
///
/// ### Params
///
/// * `index` - Reference to the built Exhaustive-PQ index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_exhaustive_opq_index_self<T>(
    index: &ExhaustiveOpqIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + AddAssign,
{
    index.generate_knn(k, return_dist, verbose)
}

//////////////
// IVF-BF16 //
//////////////

#[cfg(feature = "quantised")]
/// Build an IVF-BF16 index
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
/// The `IvfIndexBf16`.
pub fn build_ivf_bf16_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfIndexBf16<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfIndexBf16::build(mat, ann_dist, nlist, max_iters, seed, verbose)
}

#[cfg(feature = "quantised")]
/// Helper function to query a given IVF-BF16 index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - Reference to the built IVF-BF16 index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to 20% of nlist)
///   Higher values improve recall at the cost of speed
/// * `return_dist` - Shall the inner product scores be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional inner_product_scores)`
pub fn query_ivf_bf16_index<T>(
    query_mat: MatRef<T>,
    index: &IvfIndexBf16<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a given IVF-SQ8 index
///
/// This function will generate a full kNN graph based on the internal data. To
/// accelerate the process, it will leverage the internally quantised vectors
/// and the information on the Voronoi cells under the hood and query nearby
/// cells per given internal vector.
///
/// ### Params
///
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
pub fn query_ivf_bf16_self<T>(
    index: &IvfIndexBf16<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + Bf16Compatible,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
}

/////////////
// IVF-SQ8 //
/////////////

#[cfg(feature = "quantised")]
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
pub fn build_ivf_sq8_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> IvfSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    IvfSq8Index::build(mat, nlist, ann_dist, max_iters, seed, verbose)
}

#[cfg(feature = "quantised")]
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
/// A tuple of `(knn_indices, optional distances)`
pub fn query_ivf_sq8_index<T>(
    query_mat: MatRef<T>,
    index: &IvfSq8Index<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a given IVF-SQ8 index
///
/// This function will generate a full kNN graph based on the internal data. To
/// accelerate the process, it will leverage the internally quantised vectors
/// and the information on the Voronoi cells under the hood and query nearby
/// cells per given internal vector.
///
/// ### Params
///
/// * `index` - Reference to the built IVF-SQ8 index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of clusters to search (defaults to 20% of nlist)
///   Higher values improve recall at the cost of speed
/// * `return_dist` - Shall the inner product scores be returned
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_ivf_sq8_self<T>(
    index: &IvfSq8Index<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
}

////////////
// IVF-PQ //
////////////

#[cfg(feature = "quantised")]
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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

#[cfg(feature = "quantised")]
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a IVF-PQ index
///
/// This function will generate a full kNN graph based on the internal data. To
/// note, during quantisation information is lost, hence, the quality of the
/// graph is reduced compared to other indices.
///
/// ### Params
///
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
pub fn query_ivf_pq_index_self<T>(
    index: &IvfPqIndex<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
}

/////////////
// IVF-OPQ //
/////////////

#[cfg(feature = "quantised")]
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign + SimdDistance,
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

#[cfg(feature = "quantised")]
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign + SimdDistance,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, nprobe)
    })
}

#[cfg(feature = "quantised")]
/// Helper function to self query a IVF-OPQ index
///
/// This function will generate a full kNN graph based on the internal data. To
/// note, during quantisation information is lost, hence, the quality of the
/// graph is reduced compared to other indices.
///
/// ### Params
///
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
pub fn query_ivf_opq_index_self<T>(
    index: &IvfOpqIndex<T>,
    k: usize,
    nprobe: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign + SimdDistance,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
}

/////////
// GPU //
/////////

////////////////////
// Exhaustive GPU //
////////////////////

#[cfg(feature = "gpu")]
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
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + SimdDistance,
    R: Runtime,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_default();
    ExhaustiveIndexGpu::new(mat, metric, device)
}

#[cfg(feature = "gpu")]
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
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + SimdDistance,
    R: Runtime,
{
    let (indices, distances) = index.query_batch(query_mat, k, verbose);

    if return_dist {
        (indices, Some(distances))
    } else {
        (indices, None)
    }
}

#[cfg(feature = "gpu")]
/// Query the exhaustive GPU index itself
///
/// This function will generate a full kNN graph based on the internal data.
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
pub fn query_exhaustive_index_gpu_self<T, R>(
    index: &ExhaustiveIndexGpu<T, R>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + SimdDistance,
    R: Runtime,
{
    index.generate_knn(k, return_dist, verbose)
}

//////////////
// IVF GPU //
//////////////

#[cfg(feature = "gpu")]
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
    T: Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + FromPrimitive
        + Send
        + Sync
        + SimdDistance,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    IvfIndexGpu::build(mat, ann_dist, nlist, max_iters, seed, verbose, device)
}

#[cfg(feature = "gpu")]
/// Query an IVF GPU index
///
/// ### Params
///
/// * `query_mat` - Query matrix [samples, features]
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Clusters to search (defaults to √nlist)
/// * `nquery` - Number of queries to load into the GPU.
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_ivf_index_gpu<T, R>(
    query_mat: MatRef<T>,
    index: &IvfIndexGpu<T, R>,
    k: usize,
    nprobe: Option<usize>,
    nquery: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    R: Runtime,
    T: Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + FromPrimitive
        + Send
        + Sync
        + SimdDistance,
{
    let (indices, distances) = index.query_batch(query_mat, k, nprobe, nquery, verbose);

    if return_dist {
        (indices, Some(distances))
    } else {
        (indices, None)
    }
}

#[cfg(feature = "gpu")]
/// Query an IVF GPU index itself
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `query_mat` - Query matrix [samples, features]
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Clusters to search (defaults to √nlist)
/// * `nquery` - Number of queries to load into the GPU.
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_ivf_index_gpu_self<T, R>(
    index: &IvfIndexGpu<T, R>,
    k: usize,
    nprobe: Option<usize>,
    nquery: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    R: Runtime,
    T: Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + FromPrimitive
        + Send
        + Sync
        + SimdDistance,
{
    index.generate_knn(k, nprobe, nquery, return_dist, verbose)
}

////////////
// Binary //
////////////

///////////////////////
// Exhaustive Binary //
///////////////////////

#[cfg(feature = "binary")]
/// Build an exhaustive binary index
///
/// This one can be only used for Cosine distance. There is no good hash
/// function that translates Euclidean distance to Hamming distance!
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `n_bits` - Number of bits per binary code (must be multiple of 8)
/// * `seed` - Random seed for binariser
/// * `binary_init` - Initialisation method ("itq" or "random")
/// * `metric` - Distance metric for reranking (when save_store is true)
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store is
///   true)
///
/// ### Returns
///
/// The initialised `ExhaustiveIndexBinary`
pub fn build_exhaustive_index_binary<T>(
    mat: MatRef<T>,
    n_bits: usize,
    seed: usize,
    binary_init: &str,
    metric: &str,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
) -> std::io::Result<ExhaustiveIndexBinary<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    let metric = parse_ann_dist(metric).unwrap_or_default();

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        ExhaustiveIndexBinary::new_with_vector_store(mat, binary_init, n_bits, metric, seed, path)
    } else {
        Ok(ExhaustiveIndexBinary::new(mat, binary_init, n_bits, seed))
    }
}

#[cfg(feature = "binary")]
/// Helper function to query a given exhaustive binary index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The exhaustive binary index
/// * `k` - Number of neighbours to return
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank
///   is true)
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)` where distances are Hamming (u32 converted to T) or exact distances (T)
pub fn query_exhaustive_index_binary<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveIndexBinary<T>,
    k: usize,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    if rerank {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row_reranking(query_mat.row(i), k, rerank_factor)
        })
    } else {
        let (indices, dist) = if index.use_asymmetric() {
            // path where asymmetric queries are sensible
            query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                index.query_row_asymmetric(query_mat.row(i), k, rerank_factor)
            })
        } else {
            // path where asymmetric queries are not sensible/possible
            let (indices, distances_u32) =
                query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                    index.query_row(query_mat.row(i), k)
                });
            let distances_t = distances_u32.map(|dists| {
                dists
                    .into_iter()
                    .map(|v| v.into_iter().map(|d| T::from_u32(d).unwrap()).collect())
                    .collect()
            });

            (indices, distances_t)
        };

        (indices, dist)
    }
}

#[cfg(feature = "binary")]
/// Query an exhaustive binary index against itself
///
/// Generates a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `rerank_factor` - Multiplier for candidate set (only used if vector store
///   available)
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_exhaustive_index_binary_self<T>(
    index: &ExhaustiveIndexBinary<T>,
    k: usize,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    index.generate_knn(k, rerank_factor, return_dist, verbose)
}

////////////////
// IVF Binary //
////////////////

#[cfg(feature = "binary")]
/// Build an IVF index with binary quantisation
///
/// ### Params
///
/// * `mat` - Data matrix [samples, features]
/// * `binarisation_init` - "itq" or "random"
/// * `n_bits` - Number of bits per code (multiple of 8)
/// * `nlist` - Number of clusters (defaults to √n)
/// * `max_iters` - K-means iterations (defaults to 30)
/// * `dist_metric` - "euclidean" or "cosine"
/// * `seed` - Random seed
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store
///   is true)
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Built IVF binary index
#[allow(clippy::too_many_arguments)]
pub fn build_ivf_index_binary<T>(
    mat: MatRef<T>,
    binarisation_init: &str,
    n_bits: usize,
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
    verbose: bool,
) -> std::io::Result<IvfIndexBinary<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        IvfIndexBinary::build_with_vector_store(
            mat,
            binarisation_init,
            n_bits,
            ann_dist,
            nlist,
            max_iters,
            seed,
            verbose,
            path,
        )
    } else {
        Ok(IvfIndexBinary::build(
            mat,
            binarisation_init,
            n_bits,
            ann_dist,
            nlist,
            max_iters,
            seed,
            verbose,
        ))
    }
}

#[cfg(feature = "binary")]
/// Query an IVF binary index
///
/// ### Params
///
/// * `query_mat` - Query matrix [samples, features]
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Clusters to search (defaults to √nlist)
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank
///   is true)
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
#[allow(clippy::too_many_arguments)]
pub fn query_ivf_index_binary<T>(
    query_mat: MatRef<T>,
    index: &IvfIndexBinary<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    if rerank {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row_reranking(query_mat.row(i), k, nprobe, rerank_factor)
        })
    } else {
        let (indices, dist) = if index.use_asymmetric() {
            query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                index.query_row_asymmetric(query_mat.row(i), k, nprobe, rerank_factor)
            })
        } else {
            let (indices, distances_u32) =
                query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                    index.query_row(query_mat.row(i), k, nprobe)
                });
            let distances_t = distances_u32.map(|dists| {
                dists
                    .into_iter()
                    .map(|v| v.into_iter().map(|d| T::from_u32(d).unwrap()).collect())
                    .collect()
            });
            (indices, distances_t)
        };
        (indices, dist)
    }
}

#[cfg(feature = "binary")]
/// Query an IVF binary index against itself
///
/// Generates a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Clusters to search (defaults to √nlist)
/// * `rerank_factor` - Multiplier for candidate set (only used if vector store available)
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_ivf_index_binary_self<T>(
    index: &IvfIndexBinary<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    index.generate_knn(k, nprobe, rerank_factor, return_dist, verbose)
}

///////////////////////
// Exhaustive RaBitQ //
///////////////////////

#[cfg(feature = "binary")]
/// Build an exhaustive RaBitQ index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `n_clust_rabitq` - Number of clusters (None for automatic)
/// * `dist_metric` - "euclidean" or "cosine"
/// * `seed` - Random seed
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store is
///   true)
///
/// ### Returns
///
/// The initialised `ExhaustiveIndexRaBitQ`
pub fn build_exhaustive_index_rabitq<T>(
    mat: MatRef<T>,
    n_clust_rabitq: Option<usize>,
    dist_metric: &str,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
) -> std::io::Result<ExhaustiveIndexRaBitQ<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        ExhaustiveIndexRaBitQ::new_with_vector_store(mat, &ann_dist, n_clust_rabitq, seed, path)
    } else {
        Ok(ExhaustiveIndexRaBitQ::new(
            mat,
            &ann_dist,
            n_clust_rabitq,
            seed,
        ))
    }
}

#[cfg(feature = "binary")]
/// Helper function to query a given exhaustive RaBitQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The exhaustive RaBitQ index
/// * `k` - Number of neighbours to return
/// * `n_probe` - Number of clusters to search (None for default 20%)
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank is true)
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
#[allow(clippy::too_many_arguments)]
pub fn query_exhaustive_index_rabitq<T>(
    query_mat: MatRef<T>,
    index: &ExhaustiveIndexRaBitQ<T>,
    k: usize,
    n_probe: Option<usize>,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    if rerank {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row_reranking(query_mat.row(i), k, n_probe, rerank_factor)
        })
    } else {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row(query_mat.row(i), k, n_probe)
        })
    }
}

#[cfg(feature = "binary")]
/// Query an exhaustive RaBitQ index against itself
///
/// Generates a full kNN graph based on the internal data.
/// Requires vector store to be available (use save_store=true when building).
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `n_probe` - Number of clusters to search (None for default 20%)
/// * `rerank_factor` - Multiplier for candidate set size
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_exhaustive_index_rabitq_self<T>(
    index: &ExhaustiveIndexRaBitQ<T>,
    k: usize,
    n_probe: Option<usize>,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    index.generate_knn(k, n_probe, rerank_factor, return_dist, verbose)
}

/////////////////
// IVF-RaBitQ  //
/////////////////

#[cfg(feature = "binary")]
/// Build an IVF-RaBitQ index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `nlist` - Number of IVF cells (None for sqrt(n))
/// * `max_iters` - K-means iterations (None for 30)
/// * `dist_metric` - "euclidean" or "cosine"
/// * `seed` - Random seed
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store is
///   true)
/// * `verbose` - Print progress during build
///
/// ### Returns
///
/// The initialised `IvfIndexRaBitQ`
#[allow(clippy::too_many_arguments)]
pub fn build_ivf_index_rabitq<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    max_iters: Option<usize>,
    dist_metric: &str,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
    verbose: bool,
) -> std::io::Result<IvfIndexRaBitQ<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    let ann_dist = parse_ann_dist(dist_metric).unwrap_or_default();
    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        IvfIndexRaBitQ::build_with_vector_store(
            mat, ann_dist, nlist, max_iters, seed, verbose, path,
        )
    } else {
        Ok(IvfIndexRaBitQ::build(
            mat, ann_dist, nlist, max_iters, seed, verbose,
        ))
    }
}

#[cfg(feature = "binary")]
/// Helper function to query a given IVF-RaBitQ index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The IVF-RaBitQ index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of IVF cells to probe (None for sqrt(nlist))
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank is true)
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
#[allow(clippy::too_many_arguments)]
pub fn query_ivf_index_rabitq<T>(
    query_mat: MatRef<T>,
    index: &IvfIndexRaBitQ<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    if rerank {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row_reranking(query_mat.row(i), k, nprobe, rerank_factor)
        })
    } else {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row(query_mat.row(i), k, nprobe)
        })
    }
}

#[cfg(feature = "binary")]
/// Query an IVF-RaBitQ index against itself
///
/// Generates a full kNN graph based on the internal data.
/// Requires vector store to be available (use save_store=true when building).
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `nprobe` - Number of IVF cells to probe (None for sqrt(nlist))
/// * `rerank_factor` - Multiplier for candidate set size
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_ivf_index_rabitq_self<T>(
    index: &IvfIndexRaBitQ<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    index.generate_knn(k, nprobe, rerank_factor, return_dist, verbose)
}
