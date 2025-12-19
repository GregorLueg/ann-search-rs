use faer::RowRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::dist::*;
use crate::quantised::quantisers::*;
use crate::utils::*;

pub struct IvfSq8Index<T> {
    quantised_vectors: Vec<u8>,
    centroids: Vec<T>,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    codebook: ScalarQuantiser<T>, // Single global codebook
    dim: usize,
    n: usize,
    nlist: usize,
}

impl<T> IvfSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors_flat: Vec<T>,
        dim: usize,
        n: usize,
        nlist: usize,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let metric = Dist::Euclidean;
        let max_iters = max_iters.unwrap_or(30);

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

        // 2. Train centroids
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // 3. Train global codebook
        if verbose {
            println!("  Training global codebook");
        }
        let codebook = ScalarQuantiser::train(&training_data, dim);

        // 4. Assign vectors to clusters
        let assignments = assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. Quantise all vectors with global codebook
        if verbose {
            println!("  Quantising vectors");
        }
        let mut quantised_vectors = vec![0u8; n * dim];

        quantised_vectors
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(vec_idx, chunk)| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];
                let quantised = codebook.encode(vec);
                chunk.copy_from_slice(&quantised);
            });

        if verbose {
            println!("  Quantisation complete");
        }

        Self {
            quantised_vectors,
            centroids,
            all_indices,
            offsets,
            codebook,
            dim,
            n,
            nlist,
        }
    }

    #[inline]
    pub fn query_symmetric(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe.unwrap_or_else(|| (((self.nlist as f64) * 0.2) as usize).max(1));
        let k = k.min(self.n);

        // 1. Find top nprobe centroids by INNER PRODUCT (highest is best)
        let mut cluster_scores: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let ip: T = query_vec
                    .iter()
                    .zip(cent.iter())
                    .map(|(&q, &c)| q * c)
                    .sum();
                (-ip, c) // Negate so we can use select_nth_unstable for top-k
            })
            .collect();

        cluster_scores.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());

        // 2. Quantise query once
        let query_u8 = self.codebook.encode(query_vec);

        // 3. Search clusters using inner product
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for &vec_idx in &self.all_indices[start..end] {
                let ip = self.inner_product_u8(vec_idx, &query_u8);
                let neg_ip = -ip;

                if heap.len() < k {
                    heap.push((OrderedFloat(neg_ip), vec_idx));
                } else if neg_ip < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(neg_ip), vec_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);
        let (distances, indices) = results.into_iter().map(|(d, i)| (-d.0, i)).unzip();
        (indices, distances)
    }

    #[inline]
    pub fn query_row_symmetric(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_symmetric(slice, k, nprobe);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_symmetric(&query_vec, k, nprobe)
    }

    /// Inner product on quantised codes
    #[inline(always)]
    fn inner_product_u8(&self, vec_idx: usize, query_u8: &[u8]) -> T {
        let start = vec_idx * self.dim;
        let db_vec = &self.quantised_vectors[start..start + self.dim];

        let sum: i32 = query_u8
            .iter()
            .zip(db_vec.iter())
            .map(|(&q, &d)| q as i32 * d as i32)
            .sum();

        T::from_i32(sum).unwrap()
    }
}
