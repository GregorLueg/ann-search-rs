# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate

`ann-search-rs`: approximate nearest-neighbour / vector-search algorithms in Rust, focused on high in-memory performance for single-cell and computational-biology workloads. Library crate only, no binaries. Public API is a set of `build_*_index` / `query_*_index` / `query_*_self` free functions in `src/lib.rs` that thinly wrap the per-algorithm structs.

## Commands

Feature-gated modules matter. Most work requires enabling the right flag.

```bash
# Default (CPU-only) build & tests
cargo build --release
cargo test  --release

# Full test suite matching CI (macOS/Linux)
cargo test --release --features binary,quantised,gpu,serialise

# GPU integration tests (needs a working GPU + wgpu backend)
cargo test --release --features binary,gpu,gpu-tests,quantised,serialise

# Single test
cargo test --release --features quantised -- ivf_pq::tests::name_of_test --exact --nocapture

# Gridsearch examples (one per algorithm; feature-gated ones need the flag)
# CPU:       annoy, balltree, hnsw, ivf, kd_forest, kmknn, lsh, nndescent, nsg,
#            rnn_descent, vamana
# quantised: bf16, sq8, pq, opq
# binary:    binary, rabitq, tq
# gpu:       gpu, cagra, nsg_gpu (plus the knn_comparison_cagra example)
cargo run --example gridsearch_hnsw   --release
cargo run --example gridsearch_ivf    --release -- --n-samples 500000 --dim 128 --distance cosine
cargo run --example gridsearch_pq     --release --features quantised
cargo run --example gridsearch_gpu    --release --features gpu
cargo run --example gridsearch_rabitq --release --features binary

# GPU kernel microbenches (CubeCL Benchmark harness, not criterion)
cargo bench --bench gpu_ivf_kernels        --features gpu
cargo bench --bench gpu_exhaustive_kernels --features gpu

# Per-kernel GPU profile (no code changes needed; filter the WGSL dumps out)
CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout \
  cargo bench --bench gpu_exhaustive_kernels --features gpu 2>&1 | grep -v wgsl

# Docs (docs.rs config enables binary + quantised + mimalloc + serialise)
cargo doc --features binary,quantised,mimalloc,serialise --no-deps --open
```

The profiler emits one `| 15.675584ms | KernelName | WgpuRuntime |` line per launch;
aggregate by kernel name for an attribution breakdown. Other levels: `profile`,
`profile-full`, `debug`, `debug-full`. Note it inserts syncs, so it measures *isolated*
kernel cost, whereas deleting a kernel and re-timing measures a *pipelined* queue. The two
disagree, sometimes sharply. Use the profiler to attribute cost and ablation only to answer
"what is the ceiling if this disappears entirely". Profile before optimising: reasoning
about memory traffic on paper has a poor track record here.

Release profile in `Cargo.toml` sets `lto = true` and `codegen-units = 1`. Release builds are slow, but this is intentional. Do not weaken it for iteration without asking.

## Features

- `quantised`: BF16 / SQ8 / PQ / OPQ. Pulls in `half`.
- `binary`: bitwise / RaBitQ / TurboQuant, optional on-disk vector store for re-ranking. Pulls in `memmap2`, `bytemuck`, `statrs`. (`tempfile` is a dev-dependency only, used by tests and examples.)
- `gpu`: CubeCL + wgpu backend (agnostic to Metal / Vulkan / DX12 / CPU). Pulls in `cubecl` and `cubecl-utils-rs`.
- `gpu-tests`: enables tests that require a real GPU. CI-only, combined with `gpu`.
- `serialise`: save indices to disk and load them back (`IndexIo`, `save_index` / `load_index`). Pulls in `serde`, `bincode`, and `half?/serde`. Covers the CPU, quantised and binary indices; GPU indices are not covered.
- `mimalloc`: swaps in `mimalloc` as the global allocator.

Every code path under `src/quantised/`, `src/binary/`, `src/gpu/`, `src/serialise/` is behind its cfg. When editing, keep new items behind the correct `#[cfg(feature = "...")]`. The crate must still compile with default features off.

## Architecture

### Layout

```
src/
  lib.rs           # public wrapper fns; all `pub fn build_*` / `query_*` live here
  prelude.rs       # user-facing re-exports (Dist, KMeansTrainingParams, KnnResult, ...)
  errors.rs        # single AnnSearchErrors enum, thiserror-backed
  utils/           # SIMD dist, heaps, k-means, tree/graph/nndescent helpers,
                   # traits, striped locks (parallelism.rs), file staging
  cpu/             # CPU indices: annoy, ball_tree, exhaustive, hnsw, ivf,
                   #              kd_forest, kmknn, lsh, nndescent, nsg,
                   #              rnn_descent, vamana
  quantised/       # bf16/sq8/pq/opq × (exhaustive, ivf) + shared k_means & quantisers
  binary/          # binary/rabitq/tq × (exhaustive, ivf) + binariser, vec_store, turboquant/
  gpu/             # exhaustive_gpu, ivf_gpu, nndescent_gpu, cagra_gpu_search,
                   # forest_gpu, dist_gpu, topk_gpu
  serialise/       # IndexIo trait, save_index / load_index, bundle header
benches/           # GPU kernel microbenches (CubeCL Benchmark trait, not criterion)
examples/          # gridsearch_*.rs, one per algorithm, share examples/commons/mod.rs;
                   # fill_benchmarks.sh regenerates docs/ from docs/templates/
docs/              # benchmark result tables (markdown) + the templates they fill
```

Saving is a *directory*, not a file: `index.bin` plus any aux files (currently only the binary indices' mmap store). Everything is written under a temporary name and renamed at the end with `index.bin` last, so `index.bin` is the commit point. `utils/staging.rs` owns that dance, and it exists partly because truncating a mapped file is a `SIGBUS` waiting to happen. Do not replace it with a plain `File::create`.

### Public API shape

Every index follows the same pattern in `src/lib.rs`:

- `build_<name>_index(mat: MatRef<T>, ..., dist_metric: &str, ...) -> <Result<>>Index<T>`
- `query_<name>_index(query_mat, &index, k, ..., return_dist, verbose) -> KnnOptionResult<T>` for cross-set queries.
- `query_<name>_self(&index, k, ..., return_dist, verbose) -> KnnOptionResult<T>` for the full self-kNN graph. These use index-specific fast paths (IVF exploits Voronoi cells, HNSW walks the graph without re-entering).

NSG is the exception: it builds from an existing kNN graph, so alongside `build_nsg_index` there are `build_nsg_from_knn_index` and `build_nsg_from_gpu_knn`.

`KnnOptionResult<T> = Result<(Vec<Vec<usize>>, Option<Vec<Vec<T>>>), AnnSearchErrors>`. Distances are optional so callers can skip storing them when they only need indices.

`dist_metric` is parsed from a string (`"euclidean"|"l2"`, `"cosine"`, `"manhattan"|"l1"`) via `parse_ann_dist`. Unknown strings print a warning and fall back to squared Euclidean rather than erroring. Preserve that behaviour unless changing it deliberately.

### Data & numeric traits

- Vectors are `faer::MatRef<T>` (rows = samples, cols = features). Internally, indices flatten to `Vec<T>` via `utils::matrix_to_flat` for cache-friendly access.
- `AnnSearchFloat` (in `utils/traits.rs`) is the shared trait bound: `Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + ComplexField`, **plus `Serialize + DeserializeOwned` when `serialise` is on**. There are two `#[cfg]`-gated definitions of the trait; edit both. `f32` and `f64` are unaffected either way, and both are supported end-to-end.
- BF16-specific ops go through the `Bf16Compatible` bound (quantised feature only).
- GPU indices also require `CubeclFloat` (re-exported from `cubecl-utils-rs`).

### Parallelism & SIMD

- Queries fan out over samples using two shared helpers in `lib.rs`: `query_parallel` and `query_parallel_with_flags` (the latter tracks LSH miss rate). Both drive `rayon::into_par_iter` and centralise the verbose-progress counter. New algorithms should reuse these rather than rolling their own par-iters.
- Distance calculations in `utils/dist.rs` use the `wide` crate for portable SIMD (`f32x4/f32x8/f64x2/f64x4`). Runtime CPU-feature dispatch happens through a `OnceLock`-cached enum, so per-call cost is negligible.
- Prefetching helper `utils::prefetch_read` inlines `_mm_prefetch` on x86_64 and `prfm pldl1keep` inline-asm on aarch64. Anywhere you're chasing an indirection in a hot loop, keep the prefetch pattern.

### Validation harness

`utils::KnnValidation` is a trait for computing `Recall@k` against ground truth by running a small exhaustive search internally. Indices implement it, and the gridsearch examples rely on it. When adding a new index, wire it up too.

### GPU / CubeCL specifics (`src/gpu/`)

- Everything runs on CubeCL with the wgpu backend, so kernels are cross-platform (Metal/Vulkan/DX12) with a CPU fallback.
- **Tensors, device limits and dispatch geometry live in `cubecl-utils-rs`** (pinned to `0.1.0`), not here. Its prelude is the whole surface: `GpuTensor`, `GpuLimits`, `grid_2d`, `grid_2d_limited`, `checked_cube_count`, `fits_shared_memory`, `fits_binding`, `resolve_workgroup_size`, `resident_workgroups`, `plane_uniform`, `plane_partitions`, `pad_vectors`, `padded_dim`, `LINE_SIZE`, `CubeclFloat`, `CubeclUtilsErrors`. `bixverse-rs` and `manifolds-rs` consume the same crate. Changes to those primitives belong upstream there, not in a local copy here.
- What stays in `gpu/mod.rs` are the staging plans, which model *this crate's* kernel footprints: `pick_wg_y`, `tile_fits`, `mega_smem_bytes`, `plan_local_join_staging`, `plan_beam_search_staging`, `plan_db_chunk`, plus `QUERY_CHUNK_SIZE`, `DB_CHUNK_SIZE`, `WORKGROUP_SIZE_X`, `TILE_D`, `TILE_Q`.
- `gpu/topk_gpu.rs` carries its own footprint function, `radix_select_smem_bytes`, next to the kernels it sizes. The exhaustive and IVF paths dispatch to radix select via `radix_select_usable`, and the exhaustive path additionally gates on `RADIX_SELECT_MIN_K`; both fall back to the insertion-sort reducers in `dist_gpu.rs` (`extract_topk`, `reduce_ivf_topk`) otherwise.
- **Every device-limit decision is a pure function of `GpuLimits`.** Read the limits once per entry point with `GpuLimits::from_client(&client)` and pass them down; do not reach for `client.properties()` deeper in. That is what makes the smaller-device behaviour testable here, and every staging plan has host-only tests at synthetic budgets (16 KiB, 32 KiB, 48 KiB, 64 KiB against 4- and 8-byte elements).
- `pick_wg_y(dim_padded, elem_bytes, &limits)` starts from a table tuned for 32 KiB and `f32` and **shrinks** until the staging fits the device. It never grows: the Apple result is a regression-tested invariant (`test_pick_wg_y_apple_table_is_unchanged`). If you retune the table, that test is the thing that has to change deliberately.
- When touching kernels, be aware of the Metal/wgpu quirk fixed in `fb03735` and the IVF reducer variant in `3c657ee`. Divergence between Metal and other backends is real and needs cross-backend testing. The four distilled codegen rules (no `if` expressions for value selection, bit arithmetic over comparisons, `usize` counters plus `u32` sentinels in reducers, `SharedMemory::new` at kernel scope) are documented in the module header of `src/gpu/nndescent_gpu.rs`.
- Every launch in the crate is `launch_unchecked`. A dispatch that busts a device limit does no work, returns zeros and reports **no error** (the panic happens on a cubecl background thread). Benches must checksum their output before reporting a timing; an implausible speedup is the signature of a kernel that did nothing.
- Any new `SharedMemory::new` whose size depends on `k`, a graph degree or `dim` needs a plan function with a host-only test, not a constant. Any new grid dimension proportional to `n`, `nnz` or a cluster size goes through `grid_2d`, or through `checked_cube_count` when there is no free axis to flatten into.
- `WORKGROUP_SIZE_X` is 32, which matches Apple Silicon's plane size by coincidence rather than by design: **this crate uses no plane primitives**, every cross-thread reduction goes through `sync_cube()` plus shared memory. If you do add one, gate it on `plane_uniform(wg_size, &limits)` from `cubecl-utils-rs` and keep the existing kernel as the fallback arm, because a workgroup straddling two planes gives silently wrong answers.

### Errors

Single `AnnSearchErrors` enum in `src/errors.rs`, `thiserror`-derived and `#[non_exhaustive]` (variants come and go with the feature flags, so downstream matches need a `_` arm). Feature-gated variants carry `#[cfg(feature = "...")]`. `cubecl_utils_rs::CubeclUtilsErrors` and `cubecl::server::ServerError` both arrive through `#[from]` under `gpu`. When adding a new failure mode, add a variant with the appropriate `#[cfg]` rather than reaching for `anyhow` or `panic!`. The `0.4.2` bump was about replacing panics with typed errors, so keep that direction.

## Conventions & gotchas

- British English throughout (`quantised`, `binarised`, `neighbours`, `optimised`). Match it in new code, docs, and comments.
- `#![warn(missing_docs)]` is on at the crate root. Every `pub` item needs a doc comment. The existing style uses `### Params` / `### Returns` / `### Note` markdown sections.
- The `#![allow(clippy::needless_range_loop)]` at the crate root is deliberate. Indexed loops are what you want in numeric kernels here, so don't rewrite them to iterators to appease clippy.
- CubeCL version is pinned to `0.10.0`. There's a commented-out git-dep block in `Cargo.toml` for when the main branch is needed. `0.4.2` was the API break for CubeCL, keep an eye on that if the version moves again.
- Do not add new benchmarks under `benches/` without a `harness = false` entry in `Cargo.toml`. They use the CubeCL `Benchmark` trait, not criterion's default harness.
- Documentation-of-record for results lives in `docs/benchmarks_*.md`. `CHANGELOG.md` at the repo root tracks release-note-style changes.

## What's tracked outside this file

- Full user-facing docs & feature matrix: `README.md`.
- Change history: `CHANGELOG.md`.
- Benchmark tables per feature: `docs/benchmarks_standard.md`, `benchmarks_gpu.md`, `benchmarks_quantised.md`, `benchmarks_binary.md`.
