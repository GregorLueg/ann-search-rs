//! Links mlx-c for the experimental `mlx` feature. Does nothing otherwise.

use std::path::PathBuf;

/// Prefix searched when `MLX_C_PREFIX` is unset; where Homebrew puts the
/// `mlx-c` bottle on Apple Silicon.
const DEFAULT_MLX_C_PREFIX: &str = "/opt/homebrew";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    if std::env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }

    let prefix = PathBuf::from(
        std::env::var("MLX_C_PREFIX").unwrap_or_else(|_| DEFAULT_MLX_C_PREFIX.to_string()),
    );
    let lib_dir = prefix.join("lib");
    if !lib_dir.join("libmlxc.dylib").exists() {
        panic!(
            "The `mlx` feature needs mlx-c, but {} has no libmlxc.dylib. Install it with \
             `brew install mlx-c`, or point MLX_C_PREFIX at an install prefix.",
            lib_dir.display()
        );
    }

    // Homebrew dylibs carry absolute install names, so no rpath is needed.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mlxc");
}
