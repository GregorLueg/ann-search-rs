//! Synthetic dataset generators.
//!
//! Thin wrappers over `ann_search_rs::synthetic`. Same code and same seed as
//! the gridsearch examples, so a Python benchmark and a `cargo run --example`
//! run see identical points.
//!
//! The generators build a `faer::Mat`, which is column-major, so there is one
//! transpose-copy on the way into numpy. It happens once at generation time and
//! is dwarfed by the generation itself.

use ann_search_rs::prelude::matrix_to_flat;
use ann_search_rs::synthetic::{
    generate_cell_embeddings, generate_clustered_data, generate_clustered_data_high_dim,
    generate_low_rank_rotated_data, subsample_with_noise, DEFAULT_COR_STRENGTH,
};
use faer::Mat;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::convert::flat;

///////////
// Types //
///////////

/// What every generator hands back: an `(n, dim)` sample matrix and the `(n,)`
/// ground-truth cluster label per row.
type DatasetOut<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>);

/// Move a generator result into numpy.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `data` - Column-major sample matrix from the generator.
/// * `labels` - One cluster assignment per row.
///
/// ### Returns
///
/// `(X, labels)` as C-contiguous numpy arrays.
fn pack_dataset<'py>(
    py: Python<'py>,
    data: Mat<f32>,
    labels: Vec<usize>,
) -> PyResult<DatasetOut<'py>> {
    let (flat_data, n, dim) = matrix_to_flat(data.as_ref());
    let x = flat_data.into_pyarray(py).reshape([n, dim])?;
    let labels: Vec<i64> = labels.into_iter().map(|v| v as i64).collect();
    Ok((x, labels.into_pyarray(py)))
}

////////////////
// Generators //
////////////////

/// Separated Gaussian clusters joined by inter-cluster bridges.
///
/// ### Params
///
/// * `n_samples` - Rows to generate.
/// * `dim` - Features per row.
/// * `n_clusters` - Distinct clusters.
/// * `seed` - Fixes the whole draw.
///
/// ### Returns
///
/// `(X, labels)`.
#[pyfunction]
#[pyo3(signature = (n_samples, dim, n_clusters, seed))]
pub fn make_clustered(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> PyResult<DatasetOut<'_>> {
    let (data, labels) =
        py.detach(|| generate_clustered_data::<f32>(n_samples, dim, n_clusters, seed));
    pack_dataset(py, data, labels)
}

/// Clusters with local anisotropy plus a globally shared off-axis subspace.
///
/// The shared subspace is the structure OPQ's rotation exploits and PQ's
/// axis-aligned split cannot, which is what makes this the interesting case for
/// comparing the two.
///
/// ### Params
///
/// * `n_samples` - Rows to generate.
/// * `dim` - Features per row.
/// * `n_clusters` - Distinct clusters.
/// * `cor_strength` - Share of structured variance routed to the global
///   off-axis subspace, from 0.0 to 1.0. `None` uses the value behind the
///   published benchmark tables.
/// * `seed` - Fixes the whole draw.
///
/// ### Returns
///
/// `(X, labels)`.
#[pyfunction]
#[pyo3(signature = (n_samples, dim, n_clusters, seed, cor_strength = None))]
pub fn make_correlated(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
    cor_strength: Option<f64>,
) -> PyResult<DatasetOut<'_>> {
    let strength = cor_strength.unwrap_or(DEFAULT_COR_STRENGTH);
    let (data, labels) = py.detach(|| {
        generate_clustered_data_high_dim::<f32>(n_samples, dim, n_clusters, strength, seed)
    });
    pack_dataset(py, data, labels)
}

/// Data on a low-dimensional manifold inside a high-dimensional ambient space.
///
/// Cell types sit on the manifold with differentiation trajectories running
/// between them, so the intrinsic dimensionality is far below `dim`.
///
/// ### Params
///
/// * `n_samples` - Rows to generate.
/// * `dim` - Ambient features per row.
/// * `n_clusters` - Distinct cell types.
/// * `intrinsic_dim` - True dimensionality of the manifold.
/// * `seed` - Fixes the whole draw.
///
/// ### Returns
///
/// `(X, labels)`.
#[pyfunction]
#[pyo3(signature = (n_samples, dim, n_clusters, intrinsic_dim, seed))]
pub fn make_low_rank(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    intrinsic_dim: usize,
    seed: u64,
) -> PyResult<DatasetOut<'_>> {
    let (data, labels) = py.detach(|| {
        generate_low_rank_rotated_data::<f32>(n_samples, dim, intrinsic_dim, n_clusters, seed)
    });
    pack_dataset(py, data, labels)
}

/// Foundation-model cell embeddings, in the style of Geneformer or scGPT.
///
/// Heavy-tailed spectrum, a handful of high-variance rogue dimensions, and a
/// shared mean offset that puts every point inside an anisotropy cone. The
/// hardest of the four for a quantised index.
///
/// ### Params
///
/// * `n_samples` - Rows to generate.
/// * `dim` - Embedding width.
/// * `n_clusters` - Distinct cell types.
/// * `seed` - Fixes the whole draw.
///
/// ### Returns
///
/// `(X, labels)`.
#[pyfunction]
#[pyo3(signature = (n_samples, dim, n_clusters, seed))]
pub fn make_cell_embeddings(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> PyResult<DatasetOut<'_>> {
    let (data, labels) =
        py.detach(|| generate_cell_embeddings::<f32>(n_samples, dim, n_clusters, seed));
    pack_dataset(py, data, labels)
}

///////////////
// Utilities //
///////////////

/// Draw a query set from an existing dataset, with light Gaussian noise added.
///
/// Querying an index with rows it was built from is a flattering benchmark:
/// every query has an exact hit at distance zero. This perturbs the draw so the
/// queries sit near the data rather than on it.
///
/// ### Params
///
/// * `x` - The dataset to draw from, C-contiguous float32.
/// * `n_samples` - Rows to draw. Capped at the number available.
/// * `seed` - Fixes both the draw and the noise.
///
/// ### Returns
///
/// An `(min(n_samples, len(x)), dim)` array.
#[pyfunction]
#[pyo3(signature = (x, n_samples, seed))]
pub fn subsample_queries<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (data, n, dim) = flat(&x)?;
    let out = py.detach(|| {
        // `subsample_with_noise` wants a faer matrix, and faer is column-major,
        // so this cannot borrow the numpy buffer.
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let sub = subsample_with_noise(&mat, n_samples, seed);
        matrix_to_flat(sub.as_ref())
    });
    let (flat_out, rows, cols) = out;
    flat_out.into_pyarray(py).reshape([rows, cols])
}
