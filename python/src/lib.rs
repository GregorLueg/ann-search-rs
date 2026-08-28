//! Python bindings for `ann-search-rs`.
//!
//! Deliberately thin: one opaque handle per index with `build`, `query`,
//! `query_self`, `save`, `load` and pickle support, and nothing else. Defaults,
//! argument validation, metric aliasing, the squared-to-true Euclidean
//! transform and the scikit-learn estimator surface all live in the
//! hand-written `ann_search` Python package on top.
//!
//! Two invariants hold throughout, and both are load-bearing:
//!
//! - Everything expensive runs inside `Python::detach`, so the GIL is dropped
//!   for the whole rayon fan-out. See [`dispatch`] for the nesting rule.
//! - Nothing generic reaches the library's index constructors. The dispatch
//!   macros expand to concrete `f32` / `f64` code, which is what discharges the
//!   per-algorithm trait bounds without restating any of them.

#![warn(missing_docs)]

use pyo3::prelude::*;

//////////////////
// Shared plumbing //
//////////////////

mod convert;
mod dispatch;
mod error;
mod handle;
mod kmeans;
mod pool;
mod state;

/////////////
// Indices //
/////////////

mod annoy;
mod datasets;
mod exhaustive;
mod hnsw;
mod ivf;
mod kmknn;
mod nndescent;
mod nsg;
mod vamana;

////////////
// Module //
////////////

/// Assemble the extension module.
///
/// ### Params
///
/// * `m` - The module being initialised.
///
/// ### Returns
///
/// Nothing, or the first registration error.
#[pymodule]
fn _ann_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add("AnnSearchError", m.py().get_type::<error::AnnSearchError>())?;
    m.add("IndexIoError", m.py().get_type::<error::IndexIoError>())?;

    m.add_function(wrap_pyfunction!(pool::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pool::num_threads, m)?)?;

    m.add_function(wrap_pyfunction!(datasets::make_clustered, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::make_correlated, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::make_low_rank, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::make_cell_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::subsample_queries, m)?)?;

    m.add_class::<annoy::PyAnnoy>()?;
    m.add_class::<exhaustive::PyExhaustive>()?;
    m.add_class::<hnsw::PyHnsw>()?;
    m.add_class::<ivf::PyIvf>()?;
    m.add_class::<kmknn::PyKmknn>()?;
    m.add_class::<nndescent::PyNnDescent>()?;
    m.add_class::<nsg::PyNsg>()?;
    m.add_class::<vamana::PyVamana>()?;

    Ok(())
}
