use faer::RowRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::{collections::BinaryHeap, iter::Sum};

use crate::quantised::quantisers::*;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::k_means::*;

////////////////
// Main index //
////////////////

/// IVF index with product quantisation
///
/// ### Fields
///
/// * `quantised_codes` - Encoded vectors (M u8 codes per vector)
/// * `dim` - Original dimensions
/// * `n` - Number of samples in the index
/// * `metric` - Distance metric
/// * `centroids` - K-means cluster centroids
/// * `all_indices` - Vector indices for each cluster (CSR format)
/// * `offsets` - Offsets for each inverted list
/// * `codebook` - Product quantiser with M codebooks
/// * `nlist` - Number of k-means clusters
pub struct IvfPqIndex<T> {
    quantised_codes: Vec<u8>,
    dim: usize,
    n: usize,
    metric: Dist,
    centroids: Vec<T>,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    codebook: ProductQuantiser<T>,
    nlist: usize,
}

impl<T> IvfPqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Build an IVF index with product quantisation
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data (length = n * dim)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `nlist` - Number of IVF clusters
    /// * `m` - Number of subspaces for PQ (dim must be divisible by m)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `max_iters` - Maximum k-means iterations
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        mut vectors_flat: Vec<T>,
        dim: usize,
        n: usize,
        nlist: usize,
        m: usize,
        metric: Dist,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let max_iters = max_iters.unwrap_or(30);

        // Normalise for cosine distance
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising vectors for cosine distance");
            }
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 1. Subsample training data
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        // 2. Train IVF centroids
        let mut centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // Normalise centroids for cosine
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising centroids");
            }
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 3. Compute residuals for training data
        if verbose {
            println!("  Computing residuals for PQ training");
        }
        let training_assignments =
            assign_all_parallel(&training_data, dim, n_train, &centroids, nlist, &metric);

        let mut training_residuals = Vec::with_capacity(training_data.len());
        for (vec_idx, &cluster_id) in training_assignments.iter().enumerate() {
            let vec_start = vec_idx * dim;
            let vec = &training_data[vec_start..vec_start + dim];
            let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];

            for d in 0..dim {
                training_residuals.push(vec[d] - centroid[d]);
            }
        }

        // 4. Train PQ on residuals
        if verbose {
            println!("  Training product quantiser with m={}", m);
        }
        let codebook = ProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            &metric,
            max_iters,
            seed + 1000,
            verbose,
        );

        // 5. Assign all vectors to IVF clusters
        let assignments = assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);
        let (all_indices, offsets) = build_csr_layout(assignments.clone(), n, nlist);

        // 6. Encode all vectors with product quantiser
        if verbose {
            println!("  Encoding residuals");
        }
        let mut quantised_codes = vec![0u8; n * m];

        quantised_codes
            .par_chunks_mut(m)
            .zip(assignments.par_iter())
            .enumerate()
            .for_each(|(vec_idx, (chunk, &cluster_id))| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];

                // Compute residual
                let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
                let residual: Vec<T> = vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&v, &c)| v - c)
                    .collect();

                let codes = codebook.encode(&residual, &metric);
                chunk.copy_from_slice(&codes);
            });

        if verbose {
            println!("  Quantisation complete");
        }

        Self {
            quantised_codes,
            centroids,
            all_indices,
            offsets,
            codebook,
            dim,
            n,
            nlist,
            metric,
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Uses asymmetric distance computation (ADC): builds lookup tables
    /// from query subvectors to all centroids, then computes distances
    /// via table lookups.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to 15% of nlist)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let mut query_vec = query_vec.to_vec();

        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        let nprobe = nprobe.unwrap_or_else(|| (((self.nlist as f64) * 0.15) as usize).max(1));
        let k = k.min(self.n);

        // Find top nprobe centroids
        let mut cluster_scores: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dist = match self.metric {
                    Dist::Cosine => {
                        let ip: T = query_vec
                            .iter()
                            .zip(cent.iter())
                            .map(|(&q, &c)| q * c)
                            .sum();
                        T::one() - ip
                    }
                    Dist::Euclidean => query_vec
                        .iter()
                        .zip(cent.iter())
                        .map(|(&q, &c)| (q - c) * (q - c))
                        .sum(),
                };
                (dist, c)
            })
            .collect();

        cluster_scores.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            // Build lookup tables for this cluster's residual space
            let lookup_tables = self.build_lookup_tables(&query_vec, cluster_idx);

            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for &vec_idx in &self.all_indices[start..end] {
                let dist = self.compute_distance_adc(vec_idx, &lookup_tables);

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
        let (distances, indices) = results.into_iter().map(|(d, i)| (d.0, i)).unzip();
        (indices, distances)
    }

    /// Build ADC lookup tables for a specific cluster
    ///
    /// Computes distances from query residual (query - cluster_centroid) to all
    /// 256 centroids in each subspace's codebook.
    ///
    /// ### Returns
    ///
    /// Lookup tables: M x 256 distances
    fn build_lookup_tables(&self, query_vec: &[T], cluster_idx: usize) -> Vec<Vec<T>> {
        let m = self.codebook.m();
        let subvec_dim = self.codebook.subvec_dim();

        // Compute query residual
        let centroid = &self.centroids[cluster_idx * self.dim..(cluster_idx + 1) * self.dim];
        let query_residual: Vec<T> = query_vec
            .iter()
            .zip(centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        let mut tables = Vec::with_capacity(m);

        for subspace in 0..m {
            let query_sub = &query_residual[subspace * subvec_dim..(subspace + 1) * subvec_dim];
            let mut distances = Vec::with_capacity(256);

            for centroid_idx in 0..256 {
                let centroid_start = centroid_idx * subvec_dim;
                let centroid = &self.codebook.codebooks()[subspace]
                    [centroid_start..centroid_start + subvec_dim];

                let dist = match self.metric {
                    Dist::Euclidean => {
                        // Store squared distance for PQ
                        query_sub
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&q, &c)| {
                                let diff = q - c;
                                diff * diff
                            })
                            .sum()
                    }
                    Dist::Cosine => {
                        // For cosine: store negative inner product
                        let ip: T = query_sub
                            .iter()
                            .zip(centroid.iter())
                            .map(|(&q, &c)| q * c)
                            .sum();
                        -ip
                    }
                };
                distances.push(dist);
            }

            tables.push(distances);
        }

        tables
    }

    /// Compute distance using ADC lookup tables
    ///
    /// ### Params
    ///
    /// * `vec_idx` - Index of database vector
    /// * `lookup_tables` - Precomputed distance tables
    ///
    /// ### Returns
    ///
    /// Approximate distance
    #[inline(always)]
    fn compute_distance_adc(&self, vec_idx: usize, lookup_tables: &[Vec<T>]) -> T {
        let m = self.codebook.m();
        let codes_start = vec_idx * m;
        let codes = &self.quantised_codes[codes_start..codes_start + m];

        codes
            .iter()
            .enumerate()
            .map(|(subspace, &code)| lookup_tables[subspace][code as usize])
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Query using a matrix row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }
}
