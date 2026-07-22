//! Optimised vector searches in Rust originally designed for single cell
//! applications, but now as additionally GPU-accelerated, quantised (with
//! binary indices) vector searches leveraging Rust's performance under the
//! hood.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
#![allow(clippy::needless_range_loop)] // I want these loops!
#![warn(missing_docs)]

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

// MiMalloc for better allocations
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod cpu;
pub mod errors;
pub mod prelude;
pub mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "quantised")]
pub mod quantised;

#[cfg(feature = "binary")]
pub mod binary;

use faer::MatRef;
use rayon::prelude::*;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use thousands::*;

#[cfg(feature = "gpu")]
use cubecl::prelude::*;
#[cfg(feature = "quantised")]
use std::ops::AddAssign;

#[cfg(feature = "binary")]
use bytemuck::Pod;
#[cfg(feature = "binary")]
use std::path::Path;

use crate::cpu::{
    annoy::*, ball_tree::*, exhaustive::*, hnsw::*, ivf::*, kd_forest::*, kmknn::*, lsh::*,
    nndescent::*, nsg::*, rnn_descent::*, vamana::*,
};
use crate::prelude::*;
use crate::utils::nndescent_utils::ApplySortedUpdates;

#[cfg(feature = "binary")]
use crate::binary::{
    exhaustive_binary::*, exhaustive_rabitq::*, exhaustive_tq::*, ivf_binary::*, ivf_rabitq::*,
    ivf_tq::*,
};
#[cfg(feature = "gpu")]
use crate::gpu::{exhaustive_gpu::*, ivf_gpu::*, nndescent_gpu::*};
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
) -> KnnOptionResult<T>
where
    T: Send,
    F: Fn(usize) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> + Sync,
{
    let counter = Arc::new(AtomicUsize::new(0));

    let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let result = query_fn(i)?;
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
            Ok(result)
        })
        .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

    if return_dist {
        let (indices, distances) = results.into_iter().unzip();
        Ok((indices, Some(distances)))
    } else {
        let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
        Ok((indices, None))
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
) -> KnnOptionResult<T>
where
    T: Send,
    F: Fn(usize) -> Result<(Vec<usize>, Vec<T>, bool), AnnSearchErrors> + Sync,
{
    let counter = Arc::new(AtomicUsize::new(0));

    let results: Vec<(Vec<usize>, Vec<T>, bool)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let result = query_fn(i)?;
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
            Ok(result)
        })
        .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

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
        Ok((indices, Some(distances)))
    } else {
        Ok((indices, None))
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
/// * `dist_metric` - Distance metric: "euclidean", "cosine" or "manhattan"
///
/// ### Returns
///
/// The initialised `ExhausiveIndex`
pub fn build_exhaustive_index<T>(mat: MatRef<T>, dist_metric: &str) -> ExhaustiveIndex<T>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    index.generate_knn(k, return_dist, verbose)
}

///////////
// kMkNN //
///////////

/// Build a kMkNN index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `nlist` - Optional number of clusters. Defaults to sqrt(n).
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print build progress
///
/// ### Returns
///
/// The initialised `KmknnIndex`
pub fn build_kmknn_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    seed: usize,
    verbose: bool,
) -> Result<KmknnIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
    KmknnIndex::build(mat, metric, nlist, k_means_params, seed, verbose)
}

/// Helper function to query a given kMkNN index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The kMkNN index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_kmknn_index<T>(
    query_mat: MatRef<T>,
    index: &KmknnIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k)
    })
}

/// Helper function to self query a kMkNN index
///
/// Generates a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - The kMkNN index
/// * `k` - Number of neighbours to return
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_kmknn_self<T>(
    index: &KmknnIndex<T>,
    k: usize,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `n_trees` - Number of trees to use to build the index
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// The `AnnoyIndex`.
pub fn build_annoy_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    n_trees: usize,
    seed: usize,
) -> Result<AnnoyIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    AnnoyIndex::new(mat, n_trees, metric, seed)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `seed` - Random seed for reproducibility
///
/// ### Return
///
/// The `BallTreeIndex`.
pub fn build_balltree_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    seed: usize,
) -> Result<BallTreeIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
    BallTreeIndex::new(mat, metric, seed)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `dist_metric` - Distance metric: "euclidean", "cosine" or "manhatten".
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
    T: AnnSearchFloat,
    HnswIndex<T>: HnswState<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    HnswIndex::build(mat, m, ef_construction, &metric, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfIndex`.
pub fn build_ivf_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<IvfIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    IvfIndex::build(mat, metric, nlist, k_means_params, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    index.generate_knn(k, nprobe, return_dist, verbose)
}

////////////
// KdTree //
////////////

/// Build a Kd-Tree forest index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows represent the samples, columns represent
///   the embedding dimensions
/// * `dist_metric` - Distance metric: "euclidean", "cosine" or "manhatten".
/// * `n_trees` - Number of trees to use to build the index
/// * `seed` - Random seed for reproducibility
/// * `overlap` - Spill-tree overlap fraction. If None, uses the default
///   (5%). If Some(0.0), builds a standard Kd-tree without overlap.
///
/// ### Return
///
/// The `KdTreeIndex`.
pub fn build_kd_tree_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    n_trees: usize,
    seed: usize,
) -> KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    KdTreeIndex::new(mat, n_trees, metric, seed)
}

/// Helper function to query a given Kd-Tree index
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples x features
/// * `index` - The KdTreeIndex to query
/// * `k` - Number of neighbours to return
/// * `search_budget` - Search budget (total items to examine)
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_kd_tree_index<T>(
    query_mat: MatRef<T>,
    index: &KdTreeIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, search_budget)
    })
}

/// Helper function to self query the Kd-Tree index
///
/// This function will generate a full kNN graph based on the internal data.
///
/// ### Params
///
/// * `index` - The KdTreeIndex to query
/// * `k` - Number of neighbours to return
/// * `search_budget` - Search budget (total items to examine)
/// * `return_dist` - Shall the distances between the different points be
///   returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_kd_tree_self<T>(
    index: &KdTreeIndex<T>,
    k: usize,
    search_budget: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    index.generate_knn(k, search_budget, return_dist, verbose)
}

/////////
// LSH //
/////////

/// Build the LSH index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
) -> Result<LSHIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    let n_probe = n_probe.unwrap_or(index.num_bits() / 2);

    Ok(index.generate_knn(k, max_candidates, n_probe, return_dist, verbose))
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
/// * `dist_metric` - Distance metric: "euclidean", "cosine" or "manhatten".
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
) -> Result<NNDescent<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    NNDescent<T>: ApplySortedUpdates<T>,
    NNDescent<T>: NNDescentQuery<T>,
{
    index.generate_knn(k, ef_search, return_dist, verbose)
}

////////////
// Vamana //
////////////

/// Build a Vamana index
///
/// ### Params
///
/// * `mat` - The data matrix. Rows are samples, columns are dimensions.
/// * `r` - Maximum out-degree (edges per node).
/// * `l_build` - Beam width during construction.
/// * `alpha_pass1` - Pruning alpha for pass 1 (typically 1.0).
/// * `alpha_pass2` - Pruning alpha for pass 2 (typically 1.2–1.5).
/// * `dist_metric` - Distance metric: "euclidean", "cosine" or "manhatten".
/// * `seed` - Random seed for reproducibility.
///
/// ### Returns
///
/// The built `VamanaIndex`.
pub fn build_vamana_index<T>(
    mat: MatRef<T>,
    r: usize,
    l_build: usize,
    alpha_pass1: f32,
    alpha_pass2: f32,
    dist_metric: &str,
    seed: usize,
) -> VamanaIndex<T>
where
    T: AnnSearchFloat,
    VamanaIndex<T>: VamanaState<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    VamanaIndex::build(mat, metric, r, l_build, alpha_pass1, alpha_pass2, seed)
}

/// Query a Vamana index with an external query matrix
///
/// ### Params
///
/// * `query_mat` - Query matrix (samples × features).
/// * `index` - Reference to the built index.
/// * `k` - Number of neighbours to return.
/// * `ef_search` - Optional beam width override. Defaults to 100 inside the
///   index if `None`.
/// * `return_dist` - Whether to return distances.
/// * `verbose` - Print progress every 100,000 samples.
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_vamana_index<T>(
    query_mat: MatRef<T>,
    index: &VamanaIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    VamanaIndex<T>: VamanaState<T>,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
}

/// Self-query a Vamana index to generate a full kNN graph
///
/// ### Params
///
/// * `index` - Reference to the built index.
/// * `k` - Number of neighbours to return.
/// * `ef_search` - Optional beam width override.
/// * `return_dist` - Whether to return distances.
/// * `verbose` - Print progress every 100,000 samples.
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_vamana_self<T>(
    index: &VamanaIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    VamanaIndex<T>: VamanaState<T>,
{
    index.generate_knn(k, ef_search, return_dist, verbose)
}

/////////
// NSG //
/////////

/// Build an NSG index.
///
/// NSG (Navigating Spreading-out Graph) is a directed navigable graph
/// obtained by MRNG-pruning an approximate kNN graph and patching
/// connectivity via a DFS from a single navigating node. This entry point
/// builds the input kNN graph internally via NN-Descent.
///
/// ### Params
///
/// * `mat` - Data matrix (rows are samples, columns are dimensions)
/// * `r` - Maximum out-degree of the NSG graph
/// * `l_build` - Beam width for the per-node candidate search on the input
///   kNN graph
/// * `c` - Cap on the candidate-set size before MRNG pruning
/// * `knn_k` - Degree of the internal kNN graph (fed to NN-Descent)
/// * `dist_metric` - Distance metric: `"euclidean"`, `"cosine"`, or
///   `"manhattan"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress
///
/// ### Returns
///
/// The built [`NsgIndex`] on success.
#[allow(clippy::too_many_arguments)]
pub fn build_nsg_index<T>(
    mat: MatRef<T>,
    r: usize,
    l_build: usize,
    c: usize,
    knn_k: usize,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<NsgIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
    NsgIndex<T>: NsgState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    let params = NsgBuildParams::new(r, l_build, c, knn_k);
    NsgIndex::build(mat, metric, params, seed, verbose)
}

/// Build an NSG index reusing an already-built NN-Descent index.
///
/// Avoids the cost of a second NN-Descent build when the caller already has
/// one. The `mat` argument must be the same matrix that was fed to
/// [`NNDescent::new`]; NSG re-flattens it into its own storage.
///
/// ### Params
///
/// * `mat` - Original data matrix
/// * `nndescent_idx` - Reference to a pre-built NN-Descent index
/// * `r` - Maximum out-degree of the NSG graph
/// * `l_build` - Beam width for the per-node candidate search
/// * `c` - Cap on the candidate-set size before MRNG pruning
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress
///
/// ### Returns
///
/// The built [`NsgIndex`] on success.
#[allow(clippy::too_many_arguments)]
pub fn build_nsg_from_knn_index<T>(
    mat: MatRef<T>,
    nndescent_idx: &NNDescent<T>,
    r: usize,
    l_build: usize,
    c: usize,
    seed: usize,
    verbose: bool,
) -> Result<NsgIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
    NsgIndex<T>: NsgState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let params = NsgBuildParams::new(r, l_build, c, nndescent_idx.k);
    NsgIndex::build_from_nndescent(mat, nndescent_idx, params, seed, verbose)
}

/// Query an NSG index with an external query matrix.
///
/// ### Params
///
/// * `query_mat` - Query matrix (samples x features)
/// * `index` - Reference to the built index
/// * `k` - Number of neighbours to return
/// * `ef_search` - Optional beam width override. Defaults to `100` if `None`
/// * `return_dist` - Whether to return distances
/// * `verbose` - Print progress every 100_000 samples
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_nsg_index<T>(
    query_mat: MatRef<T>,
    index: &NsgIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    NsgIndex<T>: NsgState<T>,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
}

/// Self-query an NSG index to generate a full kNN graph.
///
/// ### Params
///
/// * `index` - Reference to the built index
/// * `k` - Number of neighbours to return
/// * `ef_search` - Optional beam width override
/// * `return_dist` - Whether to return distances
/// * `verbose` - Print progress every 100_000 samples
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_nsg_self<T>(
    index: &NsgIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    NsgIndex<T>: NsgState<T>,
{
    index.generate_knn(k, ef_search, return_dist, verbose)
}

/////////////////////////
// Relative NN-Descent //
/////////////////////////

/// Build a Relative NN-Descent (RNN-Descent) index.
///
/// Folds RNG-style pruning into the NN-Descent update loop so one pass
/// produces a search-ready graph. Paper defaults: `s=20`, `r=96`, `t1=4`,
/// `t2=15`.
///
/// ### Params
///
/// * `mat` - Data matrix (rows are samples, columns are dimensions)
/// * `s` - Initial out-degree of the random seed graph
/// * `r` - Maximum per-node adjacency
/// * `t1` - Outer rounds
/// * `t2` - UpdateNeighbors passes per outer round
/// * `dist_metric` - Distance metric: `"euclidean"`, `"cosine"`, or
///   `"manhattan"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress
///
/// ### Returns
///
/// The built [`RnnDescentIndex`] on success.
#[allow(clippy::too_many_arguments)]
pub fn build_rnn_descent_index<T>(
    mat: MatRef<T>,
    s: usize,
    r: usize,
    t1: usize,
    t2: usize,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<RnnDescentIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
    RnnDescentIndex<T>: RnnDescentState<T> + ApplySortedUpdates<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    let params = RnnDescentBuildParams::new(s, r, t1, t2);
    RnnDescentIndex::build(mat, metric, params, seed, verbose)
}

/// Query an RNN-Descent index with an external query matrix.
///
/// ### Params
///
/// * `query_mat` - Query matrix (samples x features)
/// * `index` - Reference to the built index
/// * `k` - Number of neighbours to return
/// * `ef_search` - Optional beam width override (defaults to `100`)
/// * `return_dist` - Whether to return distances
/// * `verbose` - Print progress every 100_000 samples
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_rnn_descent_index<T>(
    query_mat: MatRef<T>,
    index: &RnnDescentIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    RnnDescentIndex<T>: RnnDescentState<T>,
{
    query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
        index.query_row(query_mat.row(i), k, ef_search)
    })
}

/// Self-query an RNN-Descent index for a full kNN graph.
///
/// ### Params
///
/// * `index` - Reference to the built index
/// * `k` - Number of neighbours to return
/// * `ef_search` - Optional beam width override
/// * `return_dist` - Whether to return distances
/// * `verbose` - Print progress every 100_000 samples
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`.
pub fn query_rnn_descent_self<T>(
    index: &RnnDescentIndex<T>,
    k: usize,
    ef_search: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
    RnnDescentIndex<T>: RnnDescentState<T>,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustiveIndexBf16`.
pub fn build_exhaustive_bf16_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    verbose: bool,
) -> Result<ExhaustiveIndexBf16<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if verbose {
        println!(
            "Building exhaustive BF16 index with {} samples",
            mat.nrows()
        );
    }
    ExhaustiveIndexBf16::new(mat, metric)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Bf16Compatible,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    Ok(index.generate_knn(k, return_dist, verbose))
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `ExhaustiveSq8Index`.
pub fn build_exhaustive_sq8_index<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    verbose: bool,
) -> Result<ExhaustiveSq8Index<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if verbose {
        println!("Building exhaustive SQ8 index with {} samples", mat.nrows());
    }
    ExhaustiveSq8Index::new(mat, metric)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    Ok(index.generate_knn(k, return_dist, verbose))
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
) -> Result<ExhaustivePqIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    ExhaustivePqIndex::build(mat, m, metric, max_iters, n_pq_centroids, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
) -> Result<ExhaustiveOpqIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + AddAssign,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
    ExhaustiveOpqIndex::build(mat, m, metric, max_iters, n_pq_centroids, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + AddAssign,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + AddAssign,
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfIndexBf16`.
pub fn build_ivf_bf16_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<IvfIndexBf16<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    IvfIndexBf16::build(mat, metric, nlist, k_means_params, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Bf16Compatible,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    Ok(index.generate_knn(k, nprobe, return_dist, verbose))
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information during index construction
///
/// ### Return
///
/// The `IvfSq8Index`.
pub fn build_ivf_sq8_index<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<IvfSq8Index<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    IvfSq8Index::build(mat, nlist, metric, k_means_params, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    Ok(index.generate_knn(k, nprobe, return_dist, verbose))
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
    k_means_params: Option<KMeansTrainingParams>,
    n_pq_centroids: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<IvfPqIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    IvfPqIndex::build(
        mat,
        nlist,
        m,
        metric,
        k_means_params,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
    k_means_params: Option<KMeansTrainingParams>,
    n_opq_centroids: Option<usize>,
    n_opq_iter: Option<usize>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
) -> Result<IvfOpqIndex<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + AddAssign,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    IvfOpqIndex::build(
        mat,
        nlist,
        m,
        metric,
        k_means_params,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + AddAssign,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + AddAssign,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `device` - The GPU device to use
///
/// ### Returns
///
/// The initialised `ExhaustiveIndexGpu`
pub fn build_exhaustive_index_gpu<T, R>(
    mat: MatRef<T>,
    dist_metric: &str,
    device: R::Device,
) -> Result<ExhaustiveIndexGpu<T, R>, AnnSearchErrors>
where
    T: AnnSearchGpuFloat + AnnSearchFloat,
    R: Runtime,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
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
) -> KnnOptionResult<T>
where
    T: AnnSearchGpuFloat + AnnSearchFloat,
    R: Runtime,
{
    let (indices, distances) = index.query_batch(query_mat, k, verbose)?;

    if return_dist {
        Ok((indices, Some(distances)))
    } else {
        Ok((indices, None))
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
) -> KnnOptionResult<T>
where
    T: AnnSearchGpuFloat + AnnSearchFloat,
    R: Runtime,
{
    let res = index.generate_knn(k, return_dist, verbose)?;

    Ok(res)
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `seed` - Random seed
/// * `verbose` - Print progress
/// * `device` - GPU device
pub fn build_ivf_index_gpu<T, R>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    verbose: bool,
    device: R::Device,
) -> Result<IvfIndexGpu<T, R>, AnnSearchErrors>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
    IvfIndexGpu::build(mat, metric, nlist, k_means_params, seed, verbose, device)
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
) -> KnnOptionResult<T>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    let (indices, distances) = index.query_batch(query_mat, k, nprobe, nquery, verbose)?;

    if return_dist {
        Ok((indices, Some(distances)))
    } else {
        Ok((indices, None))
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
) -> KnnOptionResult<T>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    index.generate_knn(k, nprobe, nquery, return_dist, verbose)
}

///////////////////
// NNDescent GPU //
///////////////////

#[cfg(feature = "gpu")]
/// Build an NNDescent index with GPU-accelerated graph construction
/// and CAGRA optimisation.
///
/// ### Params
///
/// * `mat` - Data matrix [samples, features]
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
/// * `k` - Final neighbours per node (default 30)
/// * `build_k` - Internal NNDescent degree before CAGRA pruning (default 2*k)
/// * `max_iters` - Maximum NNDescent iterations (default 15)
/// * `n_trees` - Annoy forest size (default auto)
/// * `delta` - Convergence threshold (default 0.001)
/// * `rho` - Sampling rate (default 0.5)
/// * `seed` - Random seed
/// * `verbose` - Print progress
/// * `device` - GPU device
#[allow(clippy::too_many_arguments)]
pub fn build_nndescent_index_gpu<T, R>(
    mat: MatRef<T>,
    dist_metric: &str,
    k: Option<usize>,
    build_k: Option<usize>,
    max_iters: Option<usize>,
    n_trees: Option<usize>,
    delta: Option<f32>,
    rho: Option<f32>,
    refine_knn: Option<usize>,
    seed: usize,
    verbose: bool,
    retain_gpu: bool,
    device: R::Device,
) -> Result<NNDescentGpu<T, R>, AnnSearchErrors>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    NNDescentGpu::build(
        mat, metric, k, build_k, max_iters, n_trees, delta, rho, refine_knn, seed, verbose,
        retain_gpu, device,
    )
}

#[cfg(feature = "gpu")]
/// Query an NNDescent GPU index.
///
/// ### Params
///
/// * `query_mat` - Query matrix [samples, features]
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `ef_search` - Beam width (default auto)
/// * `query_params` - Optional GPU beam search parameters
/// * `return_dist` - Return distances
/// * `verbose` - Print progress
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_nndescent_index_gpu<T, R>(
    query_mat: MatRef<T>,
    index: &mut NNDescentGpu<T, R>,
    k: usize,
    ef_search: Option<usize>,
    query_params: Option<CagraGpuSearchParams>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    use rayon::prelude::*;

    let n_queries = query_mat.nrows();
    let gpu_batch_threshold = 32;

    if n_queries >= gpu_batch_threshold && ef_search.is_none() {
        if verbose {
            println!("  GPU batch query: {} vectors, k={}...", n_queries, k);
        }

        let queries_flat: Vec<T> = (0..n_queries)
            .flat_map(|i| {
                let row = query_mat.row(i);
                if row.col_stride() == 1 {
                    unsafe { std::slice::from_raw_parts(row.as_ptr(), row.ncols()) }.to_vec()
                } else {
                    row.iter().cloned().collect()
                }
            })
            .collect();

        let (indices, distances) =
            index.query_batch_gpu(&queries_flat, n_queries, query_params, k, 42)?;

        if return_dist {
            Ok((indices, Some(distances)))
        } else {
            Ok((indices, None))
        }
    } else {
        if verbose {
            println!(
                "  CPU beam search: {} vectors (ef={:?})...",
                n_queries, ef_search
            );
        }

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let row = query_mat.row(i);
                index.query_row(row, k, ef_search)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }
}

#[cfg(feature = "gpu")]
/// Extract the internal kNN graph from an NNDescent GPU index.
///
/// No search is performed -- this simply reshapes the graph that
/// was already built during construction.
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `return_dist` - Return distances
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn extract_nndescent_knn_gpu<T, R>(
    index: &NNDescentGpu<T, R>,
    return_dist: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    index.extract_knn(return_dist)
}

#[cfg(feature = "gpu")]
/// Self-query the NNDescent GPU index via GPU beam search.
///
/// Searches the CAGRA navigational graph for every vector in the index,
/// producing a full kNN graph. Results differ from `extract_nndescent_knn_gpu`
/// which returns the raw NNDescent graph without beam search refinement.
///
/// ### Params
///
/// * `index` - Mutable reference to built index
/// * `k` - Number of neighbours
/// * `query_params` - Optional GPU beam search parameters
/// * `return_dist` - Return distances
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_nndescent_index_gpu_self<T, R>(
    index: &mut NNDescentGpu<T, R>,
    k: usize,
    query_params: Option<CagraGpuSearchParams>,
    return_dist: bool,
) -> KnnOptionResult<T>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let (indices, distances) = index.self_query_gpu(k, query_params, 42)?;

    if return_dist {
        Ok((indices, Some(distances)))
    } else {
        Ok((indices, None))
    }
}

#[cfg(feature = "gpu")]
/// Build an NSG index reusing a GPU NN-Descent index as the input kNN graph.
///
/// Skips the CPU NN-Descent build stage; the raw kNN graph is downloaded
/// once from the GPU NN-Descent index and handed to the standard CPU NSG
/// build. The GPU CAGRA-optimised graph is deliberately not used because
/// MRNG pruning wants a true kNN as input, not a navigational graph.
///
/// ### Params
///
/// * `mat` - Original data matrix
/// * `nnd_gpu` - Reference to a pre-built GPU NN-Descent index
/// * `r` - Maximum out-degree of the NSG graph
/// * `l_build` - Beam width for the per-node candidate search
/// * `c` - Cap on the candidate-set size before MRNG pruning
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress
///
/// ### Returns
///
/// The built [`NsgIndex`] on success.
#[allow(clippy::too_many_arguments)]
pub fn build_nsg_from_gpu_nndescent<T, R>(
    mat: MatRef<T>,
    nnd_gpu: &NNDescentGpu<T, R>,
    r: usize,
    l_build: usize,
    c: usize,
    seed: usize,
    verbose: bool,
) -> Result<NsgIndex<T>, AnnSearchErrors>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
    NsgIndex<T>: NsgState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let params = NsgBuildParams::new(r, l_build, c, nnd_gpu.k);
    NsgIndex::build_from_gpu_nndescent(mat, nnd_gpu, params, seed, verbose)
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
    dist_metric: &str,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
) -> Result<ExhaustiveIndexBinary<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        ExhaustiveIndexBinary::new_with_vector_store(mat, binary_init, n_bits, metric, seed, path)
    } else {
        ExhaustiveIndexBinary::new(mat, binary_init, n_bits, seed)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
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
            })?
        } else {
            // path where asymmetric queries are not sensible/possible
            let (indices, distances_u32) =
                query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                    index.query_row(query_mat.row(i), k)
                })?;
            let distances_t = distances_u32.map(|dists| {
                dists
                    .into_iter()
                    .map(|v| v.into_iter().map(|d| T::from_u32(d).unwrap()).collect())
                    .collect()
            });

            (indices, distances_t)
        };

        Ok((indices, dist))
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    let res = index.generate_knn(k, rerank_factor, return_dist, verbose)?;

    Ok(res)
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
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
    verbose: bool,
) -> Result<IvfIndexBinary<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        IvfIndexBinary::build_with_vector_store(
            mat,
            binarisation_init,
            n_bits,
            metric,
            nlist,
            k_means_params,
            seed,
            verbose,
            path,
        )
    } else {
        IvfIndexBinary::build(
            mat,
            binarisation_init,
            n_bits,
            metric,
            nlist,
            k_means_params,
            seed,
            verbose,
        )
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    if rerank {
        query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
            index.query_row_reranking(query_mat.row(i), k, nprobe, rerank_factor)
        })
    } else {
        let (indices, dist) = if index.use_asymmetric() {
            query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                index.query_row_asymmetric(query_mat.row(i), k, nprobe, rerank_factor)
            })?
        } else {
            let (indices, distances_u32) =
                query_parallel(query_mat.nrows(), return_dist, verbose, |i| {
                    index.query_row(query_mat.row(i), k, nprobe)
                })?;
            let distances_t = distances_u32.map(|dists| {
                dists
                    .into_iter()
                    .map(|v| v.into_iter().map(|d| T::from_u32(d).unwrap()).collect())
                    .collect()
            });
            (indices, distances_t)
        };
        Ok((indices, dist))
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
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
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
) -> Result<ExhaustiveIndexRaBitQ<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        ExhaustiveIndexRaBitQ::new_with_vector_store(mat, &metric, n_clust_rabitq, seed, path)
    } else {
        ExhaustiveIndexRaBitQ::new(mat, &metric, n_clust_rabitq, seed)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.generate_knn(k, n_probe, rerank_factor, return_dist, verbose)
}

////////////////
// IVF-RaBitQ //
////////////////

#[cfg(feature = "binary")]
/// Build an IVF-RaBitQ index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples x features
/// * `nlist` - Number of IVF cells (None for sqrt(n))
/// * `k_means_params` - Optional k-means trainings parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhatten" is
///   not supported.
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
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
    verbose: bool,
) -> Result<IvfIndexRaBitQ<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        IvfIndexRaBitQ::build_with_vector_store(
            mat,
            metric,
            nlist,
            k_means_params,
            seed,
            verbose,
            path,
        )
    } else {
        IvfIndexRaBitQ::build(mat, metric, nlist, k_means_params, seed, verbose)
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
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
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.generate_knn(k, nprobe, rerank_factor, return_dist, verbose)
}

///////////////////////////
// Exhaustive TurboQuant //
///////////////////////////

#[cfg(feature = "binary")]
/// Build an exhaustive TurboQuant index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples × features
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhattan" is
///   not supported.
/// * `bits` - Bits per coordinate (2, 3, or 4)
/// * `seed` - Random seed
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store is
///   true)
///
/// ### Returns
///
/// The initialised `TurboQuantExhaustive`
pub fn build_exhaustive_index_turboquant<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    bits: usize,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
) -> Result<TurboQuantExhaustive<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });
    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        TurboQuantExhaustive::new_with_vector_store(mat, &metric, bits, seed, path)
    } else {
        TurboQuantExhaustive::new(mat, &metric, bits, seed)
    }
}

#[cfg(feature = "binary")]
/// Helper function to query a given exhaustive TurboQuant index.
///
/// Uses the 4-query fused SIMD path (for 2/4-bit) rather than scoring each
/// query independently.
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The exhaustive TurboQuant index
/// * `k` - Number of neighbours to return
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank is true)
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
#[allow(clippy::too_many_arguments)]
pub fn query_exhaustive_index_turboquant<T>(
    query_mat: MatRef<T>,
    index: &TurboQuantExhaustive<T>,
    k: usize,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.query_batch(query_mat, k, rerank, rerank_factor, return_dist, verbose)
}

#[cfg(feature = "binary")]
/// Query an exhaustive TurboQuant index against itself
///
/// Generates a full kNN graph based on the internal data.
/// Requires vector store to be available (use save_store=true when building).
///
/// ### Params
///
/// * `index` - Reference to built index
/// * `k` - Number of neighbours
/// * `rerank_factor` - Multiplier for candidate set size
/// * `return_dist` - Return distances
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (indices, optional distances)
pub fn query_exhaustive_index_turboquant_self<T>(
    index: &TurboQuantExhaustive<T>,
    k: usize,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.generate_knn(k, rerank_factor, return_dist, verbose)
}

////////////////////
// IVF TurboQuant //
////////////////////

#[cfg(feature = "binary")]
/// Build an IVF-TurboQuant index
///
/// ### Params
///
/// * `mat` - The initial matrix with samples × features
/// * `nlist` - Number of IVF cells (None for sqrt(n))
/// * `k_means_params` - Optional k-means training parameters, see
///   [KMeansTrainingParams]. If not provided, will default to sensible
///   defaults.
/// * `dist_metric` - Distance metric: "euclidean" or "cosine". "manhattan" is
///   not supported.
/// * `bits` - Bits per coordinate (2, 3, or 4)
/// * `seed` - Random seed
/// * `save_store` - Whether to save vector store for reranking
/// * `save_path` - Path to save vector store files (required if save_store is
///   true)
/// * `verbose` - Print progress during build
///
/// ### Returns
///
/// The initialised `IvfTurboQuant`
#[allow(clippy::too_many_arguments)]
pub fn build_ivf_index_turboquant<T>(
    mat: MatRef<T>,
    nlist: Option<usize>,
    k_means_params: Option<KMeansTrainingParams>,
    dist_metric: &str,
    bits: usize,
    seed: usize,
    save_store: bool,
    save_path: Option<impl AsRef<Path>>,
    verbose: bool,
) -> Result<IvfTurboQuant<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + Pod,
{
    let metric = parse_ann_dist(dist_metric).unwrap_or_else(|| {
        println!("[WARNING] Weird string used for distance metric. Using default squared Euclidean distance");
        Dist::default()
    });

    if save_store {
        let path = save_path.expect("save_path required when save_store is true");
        IvfTurboQuant::build_with_vector_store(
            mat,
            metric,
            nlist,
            k_means_params,
            bits,
            seed,
            verbose,
            path,
        )
    } else {
        IvfTurboQuant::build(mat, metric, nlist, k_means_params, bits, seed, verbose)
    }
}

#[cfg(feature = "binary")]
/// Helper function to query a given IVF-TurboQuant index.
///
/// Uses the 4-query fused SIMD path (for 2/4-bit) rather than scoring each
/// query independently. Because batched queries union the probed cells of each
/// group of four, results may differ slightly from the single-query path.
///
/// ### Params
///
/// * `query_mat` - The query matrix containing the samples × features
/// * `index` - The IVF-TurboQuant index
/// * `k` - Number of neighbours to return
/// * `nprobe` - Number of IVF cells to probe (None for sqrt(nlist))
/// * `rerank` - Whether to use exact distance reranking (requires vector store)
/// * `rerank_factor` - Multiplier for candidate set size (only used if rerank
///   is true)
/// * `return_dist` - Shall the distances be returned
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
#[allow(clippy::too_many_arguments)]
pub fn query_ivf_index_turboquant<T>(
    query_mat: MatRef<T>,
    index: &IvfTurboQuant<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank: bool,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.query_batch(
        query_mat,
        k,
        nprobe,
        rerank,
        rerank_factor,
        return_dist,
        verbose,
    )
}

#[cfg(feature = "binary")]
/// Query an IVF-TurboQuant index against itself
///
/// Generates a full kNN graph based on the internal data. Requires vector store
/// to be available (use save_store = true when building).
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
pub fn query_ivf_index_turboquant_self<T>(
    index: &IvfTurboQuant<T>,
    k: usize,
    nprobe: Option<usize>,
    rerank_factor: Option<usize>,
    return_dist: bool,
    verbose: bool,
) -> KnnOptionResult<T>
where
    T: AnnSearchFloat + Pod,
{
    index.generate_knn(k, nprobe, rerank_factor, return_dist, verbose)
}
