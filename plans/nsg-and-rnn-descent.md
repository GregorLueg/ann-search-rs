# NSG and Relative NN-Descent — status and follow-ups

Working branch: `feat-nsg-relative-nndescent`. Design plan for context:
[`docs/plans/enchanted-frolicking-gray.md`](../docs/plans/enchanted-frolicking-gray.md).

## Where we are

Phases 0, 1a, 1b, and 2 shipped. 626 tests pass under
`cargo test --release --lib --features binary,quantised,gpu` (up from 601
before). No regressions in existing indices.

Smoke-run recall on synthetic 5k × 32 clustered data:

- NSG: recall ≥ 0.99 across the full R × L × ef grid.
- RNN-Descent: recall ≥ 0.99 across the full S × R × T1 × ef grid.
- Build times: RNN-Descent ~10× faster than NSG at that scale (RNN skips
  the internal NN-Descent step), matching the paper's headline claim.

## What shipped

### Phase 0 — refactor

New file `src/utils/nndescent_utils.rs` holds the primitives that both
NN-Descent-family indices share:

- `Neighbour<T>` (bit-packed pid + is-new flag) + `SENTINEL_PID`
- `Update<T>` + `RadixKey` impl on `target`
- `UnsafeGraphPtr<T>`
- `ApplySortedUpdates<T>` trait
- `find_target_boundaries` helper

Rewired `src/cpu/nndescent.rs` and `src/gpu/nndescent_gpu.rs` to `use
crate::utils::nndescent_utils::*`. Added `NNDescent::graph()` and
`NNDescent::metric()` public accessors so NSG can consume them.

### Phase 1a — NSG (`src/cpu/nsg.rs`, ~950 lines)

- `NsgIndex<T>`, `NsgBuildParams`, `NsgState<T>` trait (thread-local
  `SearchState<T>` per numeric type).
- `NsgConstructionGraph` — parallel per-node fill via `UnsafeCell`, sequential
  DFS-fix via lock-free add-or-evict.
- Build: centroid → snap to nearest data point (navigating node), parallel
  MRNG prune per node, sequential DFS connectivity fix, flatten to `Vec<u32>`.
- Query: greedy beam from navigating node, reuses `SearchState<T>`.
- Three build entry points:
  - `NsgIndex::build(mat, metric, params, seed, verbose)` — internally builds NN-Descent.
  - `NsgIndex::build_from_nndescent(mat, &nnd, params, seed, verbose)` — reuse existing NN-Descent.
  - `NsgIndex::build_from_gpu_nndescent<R>(mat, &nnd_gpu, params, seed, verbose)` — GPU-fed (Phase 2).
- Impls: `VectorDistance`, `DimensionValidation`, `KnnValidation`.
- 13 unit tests, including `dfs_reachability_holds` and `recall_validation_high` (>0.9).

Public wrappers in `src/lib.rs`:

- `build_nsg_index<T>(mat, r, l_build, c, knn_k, dist_metric, seed, verbose)`
- `build_nsg_from_knn_index<T>(mat, &nndescent_idx, r, l_build, c, seed, verbose)`
- `build_nsg_from_gpu_nndescent<T, R>(...)` (feature-gated)
- `query_nsg_index`, `query_nsg_self`

Example: `examples/gridsearch_nsg.rs` (mirrors `gridsearch_vamana.rs`; R × L
grid over 32/48/64 × 50/100/150 with ef ∈ {50, auto, 150}).

### Phase 1b — Relative NN-Descent (`src/cpu/rnn_descent.rs`, ~1000 lines)

- `RnnDescentIndex<T>`, `RnnDescentBuildParams`, `RnnDescentState<T>` trait,
  `UpdateScratch<T>` (per-thread accepted + emitted buffers).
- Own `ApplySortedUpdates<T>` impl per f32/f64 via macro (mirrors NN-Descent's
  merge semantics: preserves is-new on existing, sets is-new on inserts,
  top-K by distance).
- Build: random seed graph → outer T1 rounds × inner T2 UpdateNeighbors
  passes; AddReverseEdges between rounds (skipped on the last).
- UpdateNeighbors uses the RNG rule from Alg. 4 (single condition
  `δ(u,v) ≥ δ(v,w)`, valid because candidates are visited in ascending
  distance from u). Pruned edges emit `Update(target=w, source=v, dist)`.
- Query: beam search over the final `Vec<(u32, T)>` graph via `SearchState`;
  entry point is nearest of a small random pool.
- Impls: `VectorDistance`, `DimensionValidation`, `KnnValidation`.
- 12 unit tests including `degree_within_r_cap`, `per_node_neighbours_sorted`,
  `recall_validation_high` (>0.85).

Public wrappers in `src/lib.rs`:

- `build_rnn_descent_index<T>(mat, s, r, t1, t2, dist_metric, seed, verbose)`
- `query_rnn_descent_index`, `query_rnn_descent_self`

Example: `examples/gridsearch_rnn_descent.rs` (S × R × T1 grid, T2 fixed
at 10 for pace).

### Phase 2 — GPU NN-Descent → NSG

- Two bound-free accessors added on `NNDescentGpu`: `metric()`, `knn_graph()`
  (moved out of the `NNDescentQuery`-bounded impl block).
- `NsgIndex::build_from_gpu_nndescent<R>` constructor and public wrapper
  `build_nsg_from_gpu_nndescent<T, R>` behind `#[cfg(feature = "gpu")]`.
- Uses the raw `knn_graph` field (before CAGRA rank-prune), which is the
  correct input for MRNG since MRNG expects a true kNN, not a navigational
  graph.
- No new CubeCL kernels needed.

## Still open

### Phase 3 — GPU MRNG prune kernel (stretch)

Not started. Only worth doing if profiling at n ≥ 10M shows CPU MRNG
dominating build time. Structure would mirror `cagra_rank_prune_shared`
in `src/gpu/nndescent_gpu.rs` (fixed-size loops over candidate set of
size C ≈ 500, shared-memory bitset for accept/reject). See design plan
§5, Q2 for the full assessment.

### Benchmark documentation

`docs/benchmarks_standard.md` needs new sections for NSG and RNN-Descent;
`docs/benchmarks_gpu.md` needs a row for GPU-fed NSG. Both require running
the gridsearches at 100k+ samples on the target hardware and piping the
output into `docs/templates/*.md.tmpl` via `examples/fill_benchmarks.sh`.

Command shape (matches the existing convention):

```bash
cargo run --release --example gridsearch_nsg -- --n-samples 100000 --dim 128 --distance euclidean
cargo run --release --example gridsearch_rnn_descent -- --n-samples 100000 --dim 128 --distance euclidean
```

The design plan's verification section (`docs/plans/enchanted-frolicking-gray.md`
§ Verification) lists the gate criteria (NSG within ±2pp of Vamana at
equal ef; RNN-Descent build competitive with NN-Descent + NSG).

### Public API surface not yet stitched into `prelude`

`NsgIndex`, `NsgBuildParams`, `NsgState`, `RnnDescentIndex`,
`RnnDescentBuildParams`, `RnnDescentState`, `UpdateScratch` are all
reachable via `ann_search_rs::*` (the examples import that way) but not
re-exported through `src/prelude.rs`. If we want them there too, add them
to the prelude; not strictly required.

## Getting oriented in a new session

Start here:

1. Read `docs/plans/enchanted-frolicking-gray.md` for the design and the
   rationale behind the phasing.
2. Read this file for the delta between plan and reality.
3. Run `cargo test --release --lib --features binary,quantised,gpu` — must
   see 626 passed (or higher if new tests added).
4. Smoke-run either gridsearch on 5k × 32 to confirm the pipeline before
   changing anything.

Files most likely to matter for follow-up work:

- Design: `docs/plans/enchanted-frolicking-gray.md`
- Shared primitives: `src/utils/nndescent_utils.rs`
- NSG: `src/cpu/nsg.rs`
- RNN-Descent: `src/cpu/rnn_descent.rs`
- Public wrappers: `src/lib.rs` (search for `// NSG //` and `// Relative NN-Descent //` banners)
- GPU accessors on NN-Descent: `src/gpu/nndescent_gpu.rs` (lightweight getters block)
- Docs to update: `docs/benchmarks_standard.md`, `docs/benchmarks_gpu.md`, `docs/templates/*.md.tmpl`, `examples/fill_benchmarks.sh`
