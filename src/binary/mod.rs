//! This module contains all of the helpers, structures and methods related
//! to binary indices. This includes the index structures themselves and
//! distance calculations

pub mod binariser;
pub mod dist_binary;
pub mod exhaustive_binary;
pub mod exhaustive_rabitq;
pub mod exhaustive_tq;
pub mod ivf_binary;
pub mod ivf_rabitq;
pub mod ivf_tq;
pub mod qg;
pub mod rabitq;
#[cfg(feature = "quantised")]
pub mod rabitq_codec;
pub mod rabitq_ex;
pub mod rabitq_fastscan;
pub mod rotator;
pub mod turboquant;
pub mod vec_store;
