//! Links mlx-c for the experimental `mlx` feature.
//!
//! The MLX code compiles only when the feature is on AND `libmlxc.dylib` is
//! found: then this emits `cfg(mlx_available)` and the `links` metadata
//! `DEP_MLXC_AVAILABLE=1`, which dependants read in their own build scripts to
//! pick the MLX or the wgpu path. Without the library it warns and the crate
//! builds as if the feature were off.

use std::path::PathBuf;

/// Prefix searched when `MLX_C_PREFIX` is unset; where Homebrew puts the
/// `mlx-c` bottle on Apple Silicon.
const DEFAULT_MLX_C_PREFIX: &str = "/opt/homebrew";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    println!("cargo::rustc-check-cfg=cfg(mlx_available)");
    if std::env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }

    let prefix = PathBuf::from(
        std::env::var("MLX_C_PREFIX").unwrap_or_else(|_| DEFAULT_MLX_C_PREFIX.to_string()),
    );
    let lib_dir = prefix.join("lib");
    if !lib_dir.join("libmlxc.dylib").exists() {
        println!(
            "cargo:warning=The `mlx` feature is on but {} has no libmlxc.dylib, so the MLX \
             backend is NOT compiled and only the wgpu/cubecl path is available. Install it with \
             `brew install mlx-c`, or point MLX_C_PREFIX at an install prefix.",
            lib_dir.display()
        );
        return;
    }

    // Homebrew dylibs carry absolute install names, so no rpath is needed.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-cfg=mlx_available");
    println!("cargo:available=1");
}
