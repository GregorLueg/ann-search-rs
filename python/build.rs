//! Picks up whether `ann-search-rs` compiled its MLX backend.
//!
//! The `mlx` feature alone is not enough: the core crate leaves the MLX code
//! out when mlx-c is missing and says so through its `links` metadata, which
//! Cargo hands this script as `DEP_MLXC_AVAILABLE`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(mlx_available)");
    println!("cargo:rerun-if-env-changed=DEP_MLXC_AVAILABLE");
    if std::env::var_os("DEP_MLXC_AVAILABLE").is_some() {
        println!("cargo:rustc-cfg=mlx_available");
    }
}
