use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::dist::*;
use crate::utils::*;

/// IVF (Inverted File) index for similarity search
///
/// Uses k-means clustering to partition vectors into nlist clusters. Each
/// cluster maintains an inverted list of vector indices assigned to it.
/// Queries search only the nprobe nearest clusters, trading perfect recall
/// for speed.
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `centroids` - Cluster centres (nlist * dim elements)
/// * `inverted_lists` - Vector indices for each cluster (nlist lists)
/// * `nlist` - Number of clusters in the index
pub struct IvfIndex<T> {
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    pub metric: Dist,
    pub centroids: Vec<T>,
    pub inverted_lists: Vec<Vec<usize>>,
    pub nlist: usize,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Return the flat vectors
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    /// Return the original dimensions
    fn dim(&self) -> usize {
        self.dim
    }

    /// Return the normalised values for the Cosine calculation
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

impl<T> IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Build an IVF index using k-means clustering
    ///
    /// Constructs an inverted file index by clustering vectors into nlist
    /// partitions using k-means. For large datasets (>500k), trains k-means
    /// on a random sample then assigns all vectors. Uses k-means|| for
    /// initialisation and Hamerly's algorithm for efficient Lloyd's iterations.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data (n * dim elements)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Number of clusters to create. Typically set to sqrt(n).
    /// * `max_iters` - Maximum Lloyd's iterations (defaults to 30 if None)
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress information
    ///
    /// ### Returns
    ///
    /// Constructed IVF index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors_flat: Vec<T>,
        dim: usize,
        n: usize,
        norms: Vec<T>,
        metric: Dist,
        nlist: usize,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let max_iters = max_iters.unwrap_or(30);

        if verbose {
            println!("Building IVF index with {} clusters", nlist);
        }

        // subsample for speed here
        let (training_data, training_indices) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for k-means training");
            }
            Self::sample_vectors(&vectors_flat, dim, n, 250_000, seed)
        } else {
            let indices: Vec<usize> = (0..n).collect();
            (vectors_flat.clone(), indices)
        };

        let n_train = training_indices.len();

        if verbose {
            println!("  Initialising centroids with k-means||");
        }
        let mut centroids =
            Self::kmeans_parallel_init(&training_data, dim, n_train, nlist, &metric, seed);

        if verbose {
            println!("  Running Lloyd's iterations with Hamerly pruning");
        }
        Self::hamerly_lloyd(
            &training_data,
            dim,
            n_train,
            &mut centroids,
            nlist,
            &metric,
            max_iters,
            verbose,
        );

        if verbose {
            println!("  Assigning all vectors to clusters");
        }
        let inverted_lists =
            Self::assign_to_clusters(&vectors_flat, dim, n, &centroids, nlist, &metric);

        if verbose {
            let list_sizes: Vec<usize> = inverted_lists.iter().map(|l| l.len()).collect();
            let avg_size = list_sizes.iter().sum::<usize>() as f64 / nlist as f64;
            let max_size = list_sizes.iter().max().unwrap_or(&0);
            let min_size = list_sizes.iter().min().unwrap_or(&0);
            println!(
                "  Cluster sizes - avg: {:.1}, min: {}, max: {}",
                avg_size, min_size, max_size
            );
        }

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            centroids,
            inverted_lists,
            nlist,
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs two-stage search: first finds nprobe nearest centroids to the
    /// query, then exhaustively searches all vectors in those clusters. Uses
    /// a max-heap to track top-k candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe.unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1));
        let k = k.min(self.n);

        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dist = match self.metric {
                    Dist::Euclidean => Self::euclidean_distance_static(query_vec, cent),
                    Dist::Cosine => Self::cosine_distance_static(query_vec, cent),
                };
                (dist, c)
            })
            .collect();

        cluster_dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            for &vec_idx in &self.inverted_lists[cluster_idx] {
                let dist = match self.metric {
                    Dist::Euclidean => self.euclidean_distance_to_query(vec_idx, query_vec),
                    Dist::Cosine => self.cosine_distance_to_query(vec_idx, query_vec, query_norm),
                };

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), vec_idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (dist, idx))
            .unzip();

        (indices, distances)
    }

    /// Sample random vectors from dataset
    ///
    /// Randomly shuffles indices and selects first n_sample vectors for
    /// k-means training. Used when dataset is large to reduce clustering time.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data
    /// * `dim` - Embedding dimensions
    /// * `n` - Total number of vectors
    /// * `n_sample` - Number of vectors to sample
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Tuple of (sampled vector data, sampled indices)
    fn sample_vectors(
        vectors_flat: &[T],
        dim: usize,
        n: usize,
        n_sample: usize,
        seed: usize,
    ) -> (Vec<T>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_sample);

        let mut sampled = Vec::with_capacity(n_sample * dim);
        for &idx in &indices {
            let start = idx * dim;
            sampled.extend_from_slice(&vectors_flat[start..start + dim]);
        }

        (sampled, indices)
    }

    /// k-means|| initialisation
    ///
    /// Parallel variant of k-means++ that oversamples centres in multiple
    /// rounds, then clusters them down to k using weighted k-means++. Much
    /// faster than sequential k-means++ with comparable quality.
    ///
    /// ### Algorithm
    ///
    /// 1. Pick first centroid uniformly at random
    /// 2. For log(k) rounds: sample k*2 new centres proportional to D²
    /// 3. Cluster oversampled candidates down to k using weighted k-means++
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `k` - Number of clusters to create
    /// * `metric` - Distance metric
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initial centroids (k * dim elements)
    fn kmeans_parallel_init(
        data: &[T],
        dim: usize,
        n: usize,
        k: usize,
        metric: &Dist,
        seed: usize,
    ) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let oversampling_factor = 2;
        let n_rounds = ((k as f64).ln() + 1.0) as usize;

        let first_idx = rng.random_range(0..n);
        let mut candidates = Vec::with_capacity(k * oversampling_factor * dim);
        candidates.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

        for _ in 0..n_rounds {
            let n_candidates = candidates.len() / dim;

            let distances: Vec<T> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    Self::min_distance_to_centroids(vec, &candidates, dim, n_candidates, metric)
                })
                .collect();

            let total_dist: f64 = distances.iter().map(|&d| d.to_f64().unwrap()).sum();

            for _ in 0..k * oversampling_factor {
                let threshold = rng.random::<f64>() * total_dist;
                let mut cumsum = 0.0;

                for (idx, &dist) in distances.iter().enumerate() {
                    cumsum += dist.to_f64().unwrap();
                    if cumsum >= threshold {
                        candidates.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
                        break;
                    }
                }
            }
        }

        Self::weighted_kmeans_plus_plus(&candidates, dim, k, metric, seed + 1)
    }

    /// Weighted k-means++ on oversampled candidates
    ///
    /// Final stage of k-means|| initialisation. Clusters the oversampled
    /// candidate centres down to exactly k centres using D² weighting.
    ///
    /// ### Params
    ///
    /// * `data` - Candidate centres (flattened)
    /// * `dim` - Embedding dimensions
    /// * `k` - Target number of clusters
    /// * `metric` - Distance metric
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Final k centroids (k * dim elements)
    fn weighted_kmeans_plus_plus(
        data: &[T],
        dim: usize,
        k: usize,
        metric: &Dist,
        seed: usize,
    ) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let n = data.len() / dim;

        if n <= k {
            return data.to_vec();
        }

        let mut centroids = Vec::with_capacity(k * dim);

        let first = rng.random_range(0..n);
        centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);

        for _ in 1..k {
            let n_cents = centroids.len() / dim;

            let distances: Vec<T> = (0..n)
                .map(|i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    Self::min_distance_to_centroids(vec, &centroids, dim, n_cents, metric)
                })
                .collect();

            let total: f64 = distances.iter().map(|&d| d.to_f64().unwrap()).sum();
            let threshold = rng.random::<f64>() * total;
            let mut cumsum = 0.0;

            for (idx, &dist) in distances.iter().enumerate() {
                cumsum += dist.to_f64().unwrap();
                if cumsum >= threshold {
                    centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
                    break;
                }
            }
        }

        centroids
    }

    /// Find minimum distance from vector to any centroid
    ///
    /// Computes distance to all centroids and returns the smallest. Used
    /// during k-means initialisation for D² weighting.
    ///
    /// ### Params
    ///
    /// * `vec` - Query vector
    /// * `centroids` - Current centroids (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n_centroids` - Number of centroids
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Minimum distance to any centroid
    fn min_distance_to_centroids(
        vec: &[T],
        centroids: &[T],
        dim: usize,
        n_centroids: usize,
        metric: &Dist,
    ) -> T {
        let mut min_dist = T::infinity();

        for c in 0..n_centroids {
            let cent = &centroids[c * dim..(c + 1) * dim];
            let dist = match metric {
                Dist::Euclidean => Self::euclidean_distance_static(vec, cent),
                Dist::Cosine => Self::cosine_distance_static(vec, cent),
            };
            if dist < min_dist {
                min_dist = dist;
            }
        }

        min_dist
    }

    /// Lloyd's algorithm with Hamerly pruning
    ///
    /// Refines k-means centroids using Lloyd's iterations with Hamerly's
    /// acceleration. For each point, tracks upper bound to assigned centroid
    /// and lower bound to nearest other centroid. Skips distance computations
    /// when bounds prove reassignment is impossible.
    ///
    /// ### Algorithm
    ///
    /// For each iteration:
    /// 1. Update bounds based on how far centroids moved
    /// 2. Skip points where upper < lower (definitely stay assigned)
    /// 3. Recompute exact distances only when necessary
    /// 4. Update centroids and check for convergence (<0.1% changed)
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `centroids` - Initial centroids (updated in-place)
    /// * `k` - Number of clusters
    /// * `metric` - Distance metric
    /// * `max_iters` - Maximum iterations
    /// * `verbose` - Print progress
    #[allow(clippy::too_many_arguments)]
    fn hamerly_lloyd(
        data: &[T],
        dim: usize,
        n: usize,
        centroids: &mut [T],
        k: usize,
        metric: &Dist,
        max_iters: usize,
        verbose: bool,
    ) {
        let mut assignments = vec![0usize; n];
        let mut upper_bounds = vec![T::infinity(); n];
        let mut lower_bounds = vec![T::zero(); n];

        for iter in 0..max_iters {
            let centroids_moved = if iter > 0 {
                Self::compute_centroid_distances(centroids, dim, k, metric)
            } else {
                vec![T::infinity(); k]
            };

            let mut changed = 0;

            for i in 0..n {
                let vec = &data[i * dim..(i + 1) * dim];
                let assigned = assignments[i];

                let max_move = centroids_moved[assigned];
                upper_bounds[i] = upper_bounds[i] + max_move;
                lower_bounds[i] = (lower_bounds[i] - max_move).max(T::zero());

                if upper_bounds[i] <= lower_bounds[i] {
                    continue;
                }

                let assigned_cent = &centroids[assigned * dim..(assigned + 1) * dim];
                let exact_dist = match metric {
                    Dist::Euclidean => Self::euclidean_distance_static(vec, assigned_cent),
                    Dist::Cosine => Self::cosine_distance_static(vec, assigned_cent),
                };
                upper_bounds[i] = exact_dist;

                if upper_bounds[i] <= lower_bounds[i] {
                    continue;
                }

                let mut best_cluster = assigned;
                let mut best_dist = exact_dist;

                for c in 0..k {
                    if c == assigned {
                        continue;
                    }

                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dist = match metric {
                        Dist::Euclidean => Self::euclidean_distance_static(vec, cent),
                        Dist::Cosine => Self::cosine_distance_static(vec, cent),
                    };

                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if best_cluster != assigned {
                    assignments[i] = best_cluster;
                    upper_bounds[i] = best_dist;
                    lower_bounds[i] = T::zero();
                    changed += 1;
                } else {
                    lower_bounds[i] = best_dist;
                }
            }

            Self::recompute_centroids(data, dim, n, centroids, k, &assignments);

            if verbose {
                println!("    Iteration {}: {} points changed", iter + 1, changed);
            }

            if changed < n / 1000 {
                if verbose {
                    println!("    Converged (< 0.1% changed)");
                }
                break;
            }
        }
    }

    /// Compute half-distances between centroids
    ///
    /// For Hamerly pruning, computes distance from each centroid to its
    /// nearest neighbour, then returns half that distance (maximum possible
    /// movement per Lloyd iteration).
    ///
    /// ### Params
    ///
    /// * `centroids` - Current centroids (flattened)
    /// * `dim` - Embedding dimensions
    /// * `k` - Number of clusters
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Vector of half-distances to nearest centroid for each cluster
    fn compute_centroid_distances(centroids: &[T], dim: usize, k: usize, metric: &Dist) -> Vec<T> {
        let mut max_moves = vec![T::zero(); k];

        for c in 0..k {
            let cent = &centroids[c * dim..(c + 1) * dim];
            let mut min_dist = T::infinity();

            for other in 0..k {
                if other == c {
                    continue;
                }
                let other_cent = &centroids[other * dim..(other + 1) * dim];
                let dist = match metric {
                    Dist::Euclidean => Self::euclidean_distance_static(cent, other_cent),
                    Dist::Cosine => Self::cosine_distance_static(cent, other_cent),
                };
                if dist < min_dist {
                    min_dist = dist;
                }
            }

            max_moves[c] = min_dist / T::from(2.0).unwrap();
        }

        max_moves
    }

    /// Recompute centroids from current assignments
    ///
    /// Updates each centroid to be the mean of all vectors assigned to it.
    /// Standard Lloyd's algorithm centroid update step.
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `centroids` - Centroids to update (in-place)
    /// * `k` - Number of clusters
    /// * `assignments` - Current cluster assignment for each vector
    fn recompute_centroids(
        data: &[T],
        dim: usize,
        n: usize,
        centroids: &mut [T],
        k: usize,
        assignments: &[usize],
    ) {
        let mut counts = vec![0usize; k];
        let mut sums = vec![T::zero(); k * dim];

        for i in 0..n {
            let cluster = assignments[i];
            counts[cluster] += 1;
            let vec = &data[i * dim..(i + 1) * dim];
            let sum_start = cluster * dim;
            for d in 0..dim {
                sums[sum_start + d] = sums[sum_start + d] + vec[d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let count = T::from(counts[c]).unwrap();
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] / count;
                }
            }
        }
    }

    /// Assign all vectors to their nearest centroid
    ///
    /// Final step of index building. Assigns each vector to its closest
    /// centroid in parallel, then collects assignments into inverted lists.
    ///
    /// ### Params
    ///
    /// * `data` - All vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `centroids` - Final centroids
    /// * `k` - Number of clusters
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Inverted lists (vector of k lists, each containing vector indices)
    fn assign_to_clusters(
        data: &[T],
        dim: usize,
        n: usize,
        centroids: &[T],
        k: usize,
        metric: &Dist,
    ) -> Vec<Vec<usize>> {
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                let mut best_cluster = 0;
                let mut best_dist = T::infinity();

                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dist = match metric {
                        Dist::Euclidean => Self::euclidean_distance_static(vec, cent),
                        Dist::Cosine => Self::cosine_distance_static(vec, cent),
                    };
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                best_cluster
            })
            .collect();

        let mut inverted_lists = vec![Vec::new(); k];
        for (idx, &cluster) in assignments.iter().enumerate() {
            inverted_lists[cluster].push(idx);
        }

        inverted_lists
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, None)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }
}
