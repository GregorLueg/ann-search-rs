use cubecl::prelude::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::iter::Sum;

use crate::gpu::dist_gpu::*;
use crate::gpu::*;
use crate::utils::dist::Dist;
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
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + num_traits::FromPrimitive,
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
                    vectors_flat[start..start + dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
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
    ///
    /// ### Returns
    ///
    /// A tuple of `(Vec<indices>, Vec<distances>)`
    pub fn query_batch(&self, query_mat: MatRef<T>, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
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
        )
    }
}
