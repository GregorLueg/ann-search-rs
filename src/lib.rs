#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod dist;
pub mod hnsw;
pub mod nndescent;
pub mod utils;
pub mod fanng;
pub mod exhaustive;
pub mod synthetic;

use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::default::Default;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::annoy::*;
use crate::hnsw::*;
use crate::nndescent::*;
use crate::utils::*;
use crate::fanng::*;
use crate::exhaustive::*;

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
pub fn build_annoy_index<T>(mat: MatRef<T>, n_trees: usize, seed: usize) -> AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    AnnoyIndex::new(mat, n_trees, seed)
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
    dist_metric: &str,
    search_budget: usize,
    return_dist: bool,
    verbose: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    let n_samples = query_mat.nrows();
    let ann_dist = parse_ann_dist(dist_metric).unwrap();
    let search_k = Some(k * search_budget);
    let counter = Arc::new(AtomicUsize::new(0));

    if return_dist {
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let (neighbors, dists) = index.query_row(query_mat.row(i), &ann_dist, k, search_k);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(" Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
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
                let (neighbors, _) = index.query_row(query_mat.row(i), &ann_dist, k, search_k);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(" Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
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
                        println!("  Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
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
                        println!("  Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
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

/// Get the kNN graph based on NN-Descent (with optional distance)
///
/// This function generates the kNN graph based via an approximate nearest
/// neighbour search based on the NN-Descent. The algorithm will use a
/// neighbours of neighbours logic to identify the approximate nearest
/// neighbours.
///
/// ### Params
///
/// * `mat` - Matrix in which rows represent the samples and columns the
///   respective embeddings for that sample
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `no_neighbours` - Number of neighbours for the KNN graph.
/// * `max_iter` - Maximum iterations for the algorithm.
/// * `delta` - Early stop criterium for the algorithm.
/// * `rho` - Sampling rate for the old neighbours. Will adaptively decrease
///   over time.
/// * `seed` - Seed for the NN Descent algorithm
/// * `verbose` - Controls verbosity of the algorithm
/// * `return_distances` - Shall the distances be returned.
///
/// ### Returns
///
/// The k-nearest neighbours based on the NN Desccent algorithm
///
/// ### Implementation details
///
/// In case of contrived synthetic data the algorithm sometimes does not
/// return enough neighbours. If that happens, the neighbours and distances will
/// be just padded.
#[allow(clippy::too_many_arguments)]
pub fn generate_knn_nndescent_with_dist<T>(
    mat: MatRef<T>,
    dist_metric: &str,
    no_neighbours: usize,
    max_iter: usize,
    delta: T,
    rho: T,
    seed: usize,
    verbose: bool,
    return_distances: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: Float + FromPrimitive + Send + Sync + Default,
    NNDescent<T>: UpdateNeighbours<T>,
{
    let graph: Vec<Vec<(usize, T)>> = NNDescent::build(
        mat,
        no_neighbours,
        dist_metric,
        max_iter,
        delta,
        rho,
        None,
        seed,
        verbose,
    );

    let mut indices = Vec::with_capacity(graph.len());
    let mut distances = if return_distances {
        Some(Vec::with_capacity(graph.len()))
    } else {
        None
    };

    for (i, neighbours) in graph.into_iter().enumerate() {
        let mut ids: Vec<usize> = Vec::with_capacity(no_neighbours);
        let mut dists: Vec<T> = Vec::with_capacity(no_neighbours);

        for (pid, dist) in neighbours {
            ids.push(pid);
            dists.push(dist);
        }

        if ids.len() < no_neighbours {
            let padding_needed = no_neighbours - ids.len();
            if ids.is_empty() {
                ids.resize(no_neighbours, i);
                dists.resize(no_neighbours, T::default());
            } else {
                for j in 0..padding_needed {
                    ids.push(ids[j % ids.len()]);
                    dists.push(dists[j % dists.len()]);
                }
            }
        }

        indices.push(ids);
        if let Some(ref mut d) = distances {
            d.push(dists);
        }
    }

    (indices, distances)
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
    verbose: bool
) -> Fanng<T> where
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
                        println!(" Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
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
                        println!(" Processed {} / {} samples.", count.separate_with_underscores(), n_samples.separate_with_underscores());
                    }
                }
                neighbors
            })
            .collect();
        (indices, None)
    }
}

//////////
// Test //
//////////

#[cfg(test)]
mod full_library_tests {
    use super::*;
    use faer::Mat;
    use std::collections::HashSet;

    fn create_clustered_data<T: Float + FromPrimitive>() -> Mat<T> {
        use rand::Rng;
        use rand::SeedableRng;

        let n_clusters = 3;
        let points_per_cluster = 100;
        let dims = 10;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut data = Vec::new();

        for cluster_id in 0..n_clusters {
            let offset = (cluster_id as f64) * 2.0;
            for _i in 0..points_per_cluster {
                for _d in 0..dims {
                    let noise = rng.random_range(-1.0..1.0);
                    data.push(T::from_f64(offset + noise).unwrap());
                }
            }
        }

        Mat::from_fn(n_clusters * points_per_cluster, dims, |i, j| {
            data[i * dims + j]
        })
    }

    fn compute_overlap(a: &[usize], b: &[usize]) -> f64 {
        let set_a: HashSet<_> = a.iter().collect();
        let overlap = b.iter().filter(|x| set_a.contains(x)).count();
        overlap as f64 / a.len() as f64
    }

    #[test]
    fn test_hnsw_finds_self() {
        let mat = create_clustered_data::<f64>();
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        let (hnsw_indices, hnsw_dists) = query_hnsw_index(mat.as_ref(), &hnsw_idx, 15, 400, true, false);
        
        let mut self_not_first = 0;
        let mut self_not_found = 0;
        
        for i in 0..mat.nrows() {
            if hnsw_indices[i][0] != i {
                self_not_first += 1;
                println!("HNSW: Point {} didn't find itself first. Found: {} with dist: {}", 
                        i, hnsw_indices[i][0], hnsw_dists.as_ref().unwrap()[i][0]);
            }
            if !hnsw_indices[i].contains(&i) {
                self_not_found += 1;
                println!("HNSW: Point {} didn't find itself at all! Neighbours: {:?}", i, &hnsw_indices[i]);
            }
        }
        
        assert_eq!(self_not_found, 0, "HNSW: {} points couldn't find themselves", self_not_found);
        assert!(self_not_first < 10, "HNSW: {} points didn't find themselves first", self_not_first);
    }


    #[test]
    fn test_nndescent_finds_self() {
        let mat = create_clustered_data::<f64>();
        let (nn_indices, nn_dists) = generate_knn_nndescent_with_dist(
            mat.as_ref(), "euclidean", 15, 30, 0.001, 0.5, 42, false, true
        );
        
        let mut self_not_found = 0;
        for i in 0..mat.nrows() {
            if !nn_indices[i].contains(&i) {
                self_not_found += 1;
            }
            assert_eq!(nn_indices[i][0], i, "NNDescent: Point {} should find itself first", i);
            assert!(nn_dists.as_ref().unwrap()[i][0] < 0.01, "NNDescent: Point {} distance to self should be ~0", i);
        }
        assert_eq!(self_not_found, 0);
    }



    #[test]
    fn test_annoy_finds_self() {
        let mat = create_clustered_data::<f64>();
        let annoy_idx = build_annoy_index(mat.as_ref(), 100, 42);
        let (annoy_indices, annoy_dists) = query_annoy_index(mat.as_ref(), &annoy_idx, 15, "euclidean" , 100, true, false);
        
        let mut self_not_found = 0;
        for i in 0..mat.nrows() {
            if !annoy_indices[i].contains(&i) {
                self_not_found += 1;
            }
            assert_eq!(annoy_indices[i][0], i, "Annoy: Point {} should find itself first", i);
            assert!(annoy_dists.as_ref().unwrap()[i][0] < 0.01, "Annoy: Point {} distance to self should be ~0", i);
        }
        assert_eq!(self_not_found, 0);
    }

  #[test]
    fn test_fanng_finds_self() {
        let mat = create_clustered_data::<f64>();
        let fanng_idx = build_fanng_index(mat.as_ref(), "euclidean", Some(FanngParams::fast()), 42, false);
        let (fanng_indices, fanng_dists) = query_fanng_index(mat.as_ref(), &fanng_idx, 15, 500, 25, true, false);
        
        let mut self_not_found = 0;
        for i in 0..mat.nrows() {
            if fanng_indices[i][0] != i {
                println!("FANNG: Point {} didn't find itself first. Found: {} with dist: {}", 
                        i, fanng_indices[i][0], fanng_dists.as_ref().unwrap()[i][0]);
            }
            if !fanng_indices[i].contains(&i) {
                self_not_found += 1;
                println!("FANNG: Point {} didn't find itself at all! Neighbours: {:?}", i, &fanng_indices[i]);
            }
        }
        
        assert_eq!(self_not_found, 0, "FANNG: {} points couldn't find themselves", self_not_found);
    }

    #[test]
    fn test_methods_find_cluster_neighbours() {
        // All of these should identify neighbours within their clusters
        let mat = create_clustered_data::<f64>();
        let k = 15;
        
        let annoy_idx = build_annoy_index(mat.as_ref(), 100, 42);
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        let fanng_idx = build_fanng_index(mat.as_ref(), "euclidean", Some(FanngParams::fast()), 42, false);
        let (nn_indices, _) = generate_knn_nndescent_with_dist(
            mat.as_ref(), "euclidean", k, 30, 0.001, 0.5, 42, false, false
        );
        
        let (annoy_indices, _) = query_annoy_index(mat.as_ref(), &annoy_idx, 15, "euclidean", 100, false, false);
        let (hnsw_indices, _) = query_hnsw_index(mat.as_ref(), &hnsw_idx, k, 400, false, false);
        let (fanng_indices, _) = query_fanng_index(mat.as_ref(), &fanng_idx, 15, 500, 25, false, false);
        
        // Check that neighbours are mostly from the same cluster
        for method_name in ["Annoy", "HNSW", "NNDescent", "FANNG"].iter() {
            let indices = match *method_name {
                "Annoy" => &annoy_indices,
                "HNSW" => &hnsw_indices,
                "NNDescent" => &nn_indices,
                "FANNG" => &fanng_indices,
                _ => unreachable!(),
            };
            
            for cluster_id in 0..3 {
                let cluster_start = cluster_id * 100;
                let cluster_end = (cluster_id + 1) * 100;
                
                // Sample a few points from this cluster
                for &test_point in &[cluster_start, cluster_start + 50, cluster_end - 1] {
                    let neighbours = &indices[test_point];
                    let same_cluster = neighbours.iter()
                        .filter(|&&n| n >= cluster_start && n < cluster_end)
                        .count();
                    
                    assert!(same_cluster >= k - 2, 
                        "{}: Point {} in cluster {} should find mostly cluster neighbours. Found {}/{} in cluster",
                        method_name, test_point, cluster_id, same_cluster, k);
                }
            }
        }
    }


    #[test]
    fn test_ann_methods_similarity() {
        let mat = create_clustered_data::<f64>();
        let k = 15;

        let annoy_idx = build_annoy_index(mat.as_ref(), 100, 42);
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        let fanng_idx = build_fanng_index(mat.as_ref(), "euclidean", Some(FanngParams::fast()), 42, false);
        let (nn_indices, _) = generate_knn_nndescent_with_dist(
            mat.as_ref(),
            "euclidean",
            k,
            30,
            0.001,
            0.5,
            42,
            false,
            false,
        );

        let (annoy_indices, _) =
            query_annoy_index(mat.as_ref(), &annoy_idx, k,"euclidean", 100, false, false);
        let (hnsw_indices, _) = query_hnsw_index(mat.as_ref(), &hnsw_idx, k, 400, false, false); // Increased ef_search
        let (fanng_indices, _) = query_fanng_index(mat.as_ref(), &fanng_idx, 15, 500, 25, false, false);


        let mut total_overlaps = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let sample_points: Vec<usize> = (0..300).step_by(30).collect();

        for &i in &sample_points {
            let ah = compute_overlap(&annoy_indices[i], &hnsw_indices[i]);
            let an = compute_overlap(&annoy_indices[i], &nn_indices[i]);
            let af = compute_overlap(&annoy_indices[i], &fanng_indices[i]);
            let hn = compute_overlap(&hnsw_indices[i], &nn_indices[i]);
            let hf = compute_overlap(&hnsw_indices[i], &fanng_indices[i]);
            let nf = compute_overlap(&nn_indices[i], &fanng_indices[i]);

            total_overlaps.0 += ah;
            total_overlaps.1 += an;
            total_overlaps.2 += af; 
            total_overlaps.3 += hn;
            total_overlaps.4 += hf;
            total_overlaps.5 += nf;
        }

        let n = sample_points.len() as f64;
        let avg_ah = total_overlaps.0 / n;
        let avg_an = total_overlaps.1 / n;
        let avg_af = total_overlaps.2 / n;
        let avg_hn = total_overlaps.3 / n;
        let avg_hf = total_overlaps.4 / n;
        let avg_nf = total_overlaps.5 / n;

        assert!(
            avg_ah > 0.75, 
            "Annoy/HNSW average overlap too low: {:.2}",
            avg_ah
        );
        assert!(
            avg_an > 0.75,
            "Annoy/NNDescent average overlap too low: {:.2}",
            avg_an
        );
        assert!(
            avg_af > 0.75,
            "Annoy/FANNG average overlap too low: {:.2}",
            avg_an
        );
        assert!(
            avg_hn > 0.75,
            "HNSW/NNDescent average overlap too low: {:.2}",
            avg_hn
        );
        assert!(
            avg_hf > 0.75,
            "HNSW/FANNG average overlap too low: {:.2}",
            avg_an
        );
         assert!(
            avg_nf > 0.75,
            "NNDescent/FANNG average overlap too low: {:.2}",
            avg_an
        );
    }
}
