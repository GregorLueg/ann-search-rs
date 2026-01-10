use cubecl::prelude::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::iter::Sum;

use crate::gpu::dist_gpu::*;
use crate::gpu::*;
use crate::utils::dist::*;
use crate::utils::*;

////////////////////////
// ExhaustiveIndexGpu //
////////////////////////

/// Exhaustive (brute-force) nearest neighbour index (on GPU)
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data for distance calculations. Flattened
///   for better cache locality
/// * `norms` - Normalised pre-calculated values per sample if distance is set to
///   Cosine
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
/// * `dist_metric` - The type of distance the index is designed for
/// * `device` - The cubecl runtime
pub struct ExhaustiveIndexGpu<T: Float, R: Runtime> {
    vectors_flat: Vec<T>,
    norms: Vec<T>,
    dim: usize,
    n: usize,
    metric: Dist,
    device: R::Device,
}

impl<T, R> ExhaustiveIndexGpu<T, R>
where
    R: Runtime,
    T: Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + num_traits::FromPrimitive
        + SimdDistance,
{
    /// Generate a new exhaustive index (on the GPU)
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for.
    /// * `device` - The runtime device for the cubecl
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index (on GPU)
    pub fn new(data: MatRef<T>, metric: Dist, device: R::Device) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // assertion for the manual unrolling
        assert!(
            dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            dim,
            LINE_SIZE
        );

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            vectors_flat,
            norms,
            dim,
            n,
            metric,
            device,
        }
    }

    /// Query the exhaustive index
    ///
    /// ### Params
    ///
    /// * `query_mat` - The samples x features matrix to query. n(features)
    ///   needs to be divisible by 4!
    /// * `k` - Number of neighbours to return
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// A tuple of `(Vec<indices>, Vec<distances>)`
    pub fn query_batch(
        &self,
        query_mat: MatRef<T>,
        k: usize,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        assert!(
            self.dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            self.dim,
            LINE_SIZE
        );

        let (vectors_query, n_query, dim_query) = matrix_to_flat(query_mat);

        assert!(
            self.dim == dim_query,
            "The query matrix has not the same dimensionality as the index"
        );

        let query_norms = if self.metric == Dist::Cosine {
            (0..n_query)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.dim;
                    vectors_query[start..start + self.dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let query_data = BatchData::new(&vectors_query, &query_norms, n_query);
        let db_data = BatchData::new(&self.vectors_flat, &self.norms, self.n);

        query_batch_gpu::<T, R>(
            k,
            &query_data,
            &db_data,
            self.dim,
            &self.metric,
            self.device.clone(),
            verbose,
        )
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let query_data = BatchData::new(&self.vectors_flat, &self.norms, self.n);
        let db_data = BatchData::new(&self.vectors_flat, &self.norms, self.n);

        let (indices, distances) = query_batch_gpu::<T, R>(
            k,
            &query_data,
            &db_data,
            self.dim,
            &self.metric,
            self.device.clone(),
            verbose,
        );

        if return_dist {
            (indices, Some(distances))
        } else {
            (indices, None)
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cpu::CpuDevice;
    use cubecl::cpu::CpuRuntime;
    use faer::Mat;

    #[test]
    fn test_exhaustive_index_query() {
        let device = CpuDevice;

        // 8 samples, 4 dimensions
        let data = Mat::from_fn(8, 4, |i, j| if i == j { 1.0_f32 } else { 0.0_f32 });

        let index =
            ExhaustiveIndexGpu::<f32, CpuRuntime>::new(data.as_ref(), Dist::Euclidean, device);

        let query = Mat::from_fn(2, 4, |i, j| if i == j { 1.0_f32 } else { 0.0_f32 });

        let (indices, distances) = index.query_batch(query.as_ref(), 3, false);

        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0].len(), 3);

        // First query [1,0,0,0] should match first db vector perfectly
        assert_eq!(indices[0][0], 0);
        assert!(distances[0][0] < 0.01);
    }

    #[test]
    fn test_exhaustive_index_cosine() {
        let device = CpuDevice;

        let data = Mat::from_fn(4, 4, |i, _j| i as f32 + 1.0);

        let index = ExhaustiveIndexGpu::<f32, CpuRuntime>::new(data.as_ref(), Dist::Cosine, device);

        let query = Mat::from_fn(1, 4, |_, _| 1.0_f32);
        let (indices, distances) = index.query_batch(query.as_ref(), 2, false);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].len(), 2);
        assert!(distances[0][0] >= 0.0 && distances[0][0] <= 2.0);
    }

    #[test]
    fn test_generate_knn() {
        let device = CpuDevice;

        let data = Mat::from_fn(6, 4, |i, j| if i == j { 1.0_f32 } else { 0.1_f32 });

        let index =
            ExhaustiveIndexGpu::<f32, CpuRuntime>::new(data.as_ref(), Dist::Euclidean, device);

        let (indices, distances) = index.generate_knn(3, true, false);

        assert_eq!(indices.len(), 6);
        assert!(distances.is_some());
        assert_eq!(distances.unwrap().len(), 6);
    }
}
