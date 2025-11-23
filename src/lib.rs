#![allow(clippy::needless_range_loop)] // I want these loops!

pub mod annoy;
pub mod dist;
pub mod hnsw;
pub mod nndescent;
pub mod utils;

use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::default::Default;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::annoy::*;
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
/// * `dist_metric` - The distance metric to use. One of `"euclidean"` or
///   `"cosine"`.
/// * `k` - Number of neighbours to return
/// * `search_budget` - Search budget per tree
/// * `return_dist` - Shall the distances between the different points be
///   returned
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub fn query_annoy_index<T>(
    query_mat: MatRef<T>,
    index: &AnnoyIndex<T>,
    dist_metric: &str,
    k: usize,
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
                        println!(" Processed {} / {} cells.", count, n_samples);
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
                        println!(" Processed {} / {} cells.", count, n_samples);
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
                        println!("  Processed {} / {} cells.", count, n_samples);
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
                        println!("  Processed {} / {} cells.", count, n_samples);
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

#[cfg(test)]
mod integration_tests {
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
    fn test_hnsw_entry_point_reach() {
        let mat = create_clustered_data::<f64>();
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        
        // Check what entry point connects to at layer 0
        let entry = hnsw_idx.entry_point as usize;
        let offset = hnsw_idx.neighbour_offsets[entry];
        let max_neighbours = hnsw_idx.m * 2;
        
        let mut cluster_counts = [0; 3];
        let mut neighbours = Vec::new();
        
        for i in 0..max_neighbours {
            let neighbour = hnsw_idx.neighbours_flat[offset + i];
            if neighbour == u32::MAX {
                break;
            }
            let cluster = (neighbour as usize) / 100;
            cluster_counts[cluster] += 1;
            neighbours.push(neighbour as usize);
        }
        
        // Check if any cluster 0 nodes connect TO the entry point
        let mut reverse_connections = 0;
        for i in 0..100 {
            let offset = hnsw_idx.neighbour_offsets[i];
            let max_neighbours = hnsw_idx.m * 2;
            
            for j in 0..max_neighbours {
                let neighbour = hnsw_idx.neighbours_flat[offset + j];
                if neighbour == entry as u32 {
                    reverse_connections += 1;
                }
            }
        }
        
        assert!(cluster_counts[0] > 0 || reverse_connections > 0, 
                "Entry point cannot reach cluster 0!");
    }

    #[test]
    fn test_hnsw_cross_cluster_connectivity() {
        let mat = create_clustered_data::<f64>();
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        
        // Check if layer 0 has connections between clusters
        let mut cross_cluster_connections = 0;
        
        for i in 0..100 {  // Check cluster 0 nodes
            let offset = hnsw_idx.neighbour_offsets[i];
            let max_neighbours = hnsw_idx.m * 2;
            
            for j in 0..max_neighbours {
                let neighbour = hnsw_idx.neighbours_flat[offset + j];
                if neighbour == u32::MAX {
                    break;
                }
                let neighbour_cluster = (neighbour as usize) / 100;
                if neighbour_cluster != 0 {
                    cross_cluster_connections += 1;
                }
            }
        }
        
        // The entry point should ideally be able to reach all clusters
        assert!(cross_cluster_connections > 0, "No cross-cluster connections found!");
    }

    #[test]
    fn test_hnsw_graph_connectivity() {
        let mat = create_clustered_data::<f64>();
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        
        // Check how many nodes have connections
        let mut disconnected = 0;
        let mut connection_counts = Vec::new();
        
        for i in 0..hnsw_idx.n {
            let offset = hnsw_idx.neighbour_offsets[i];
            let max_neighbours = if hnsw_idx.layer_assignments[i] == 0 { 
                hnsw_idx.m * 2 
            } else { 
                hnsw_idx.m 
            };
            
            let mut count = 0;
            for j in 0..max_neighbours {
                if hnsw_idx.neighbours_flat[offset + j] != u32::MAX {
                    count += 1;
                }
            }
            
            if count == 0 {
                disconnected += 1;
            }
            connection_counts.push(count);
        }
        
        assert_eq!(disconnected, 0, "Found {} disconnected nodes", disconnected);
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
        let (annoy_indices, annoy_dists) = query_annoy_index(mat.as_ref(), &annoy_idx, "euclidean", 15, 100, true, false);
        
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
    fn test_methods_find_cluster_neighbours() {
        let mat = create_clustered_data::<f64>();
        let k = 15;
        
        let annoy_idx = build_annoy_index(mat.as_ref(), 100, 42);
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false);
        let (nn_indices, _) = generate_knn_nndescent_with_dist(
            mat.as_ref(), "euclidean", k, 30, 0.001, 0.5, 42, false, false
        );
        
        let (annoy_indices, _) = query_annoy_index(mat.as_ref(), &annoy_idx, "euclidean", k, 100, false, false);
        let (hnsw_indices, _) = query_hnsw_index(mat.as_ref(), &hnsw_idx, k, 400, false, false);
        
        // Check that neighbours are mostly from the same cluster
        for method_name in ["Annoy", "HNSW", "NNDescent"].iter() {
            let indices = match *method_name {
                "Annoy" => &annoy_indices,
                "HNSW" => &hnsw_indices,
                "NNDescent" => &nn_indices,
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
        let hnsw_idx = build_hnsw_index(mat.as_ref(), 16, 400, "euclidean", 42, false); // Reduced M, increased ef_construction
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
            query_annoy_index(mat.as_ref(), &annoy_idx, "euclidean", k, 100, false, false);
        let (hnsw_indices, _) = query_hnsw_index(mat.as_ref(), &hnsw_idx, k, 400, false, false); // Increased ef_search

        let mut total_overlaps = (0.0, 0.0, 0.0);
        let sample_points: Vec<usize> = (0..300).step_by(30).collect();

        for &i in &sample_points {
            let ah = compute_overlap(&annoy_indices[i], &hnsw_indices[i]);
            let an = compute_overlap(&annoy_indices[i], &nn_indices[i]);
            let hn = compute_overlap(&hnsw_indices[i], &nn_indices[i]);

            total_overlaps.0 += ah;
            total_overlaps.1 += an;
            total_overlaps.2 += hn;
        }

        let n = sample_points.len() as f64;
        let avg_ah = total_overlaps.0 / n;
        let avg_an = total_overlaps.1 / n;
        let avg_hn = total_overlaps.2 / n;

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
            avg_hn > 0.75,
            "HNSW/NNDescent average overlap too low: {:.2}",
            avg_hn
        );
    }
}
