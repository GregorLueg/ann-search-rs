# NSG and Relative NN-Descent

## Context

The crate currently ships three graph-based ANN indices (HNSW, Vamana, NN-Descent) plus a GPU CAGRA-style NN-Descent. We want to add two more, both from recent-ish literature:

- **NSG** (Fu et al., 2017 / arXiv 1707.00143). A search graph built by pruning an approximate kNN graph via the MRNG rule, then patching connectivity with a DFS from a single navigating node. Well-regarded in the ANN benchmarks; slots naturally next to Vamana.
- **Relative NN-Descent** (Ono & Matsui, ACM MM 2023 / arXiv 2310.20419). Folds RNG-style pruning into the NN-Descent update loop so one pass produces a search-ready graph, no separate kNN → refinement stage. Reported ~2× faster than NN-Descent + NSG on GIST1M.

Both algorithms need an approximate kNN graph or NN-Descent-style refinement. That gives us a natural opportunity to consolidate primitives already living inside `src/cpu/nndescent.rs`. We already have GPU NN-Descent, which is the obvious kNN source for GPU-accelerated NSG.

Scope of this plan: two CPU indices in phase 1, then GPU-fed NSG in phase 2, and a stretch phase 3 for GPU MRNG pruning if the CPU version turns out to bottleneck. Not planning full GPU RNN-Descent (see §7).

## Approach summary

1. **Phase 0** — extract shared NN-Descent primitives (`Neighbour<T>`, `Update<T>`, `RadixKey`, `ApplySortedUpdates<T>`) to `src/utils/nndescent_utils.rs`. Behaviour-preserving refactor. Blocks everything.
2. **Phase 1** — NSG CPU and RNN-Descent CPU in parallel across two agents, disjoint files.
3. **Phase 2** — slot GPU NN-Descent into NSG as an alternative kNN source, update docs.
4. **Phase 3** (stretch) — GPU MRNG prune kernel if profiling justifies it.

## Shared primitives (Phase 0)

**Extract to new `src/utils/nndescent_utils.rs`:**
- `Neighbour<T>` + `SENTINEL_PID` + `IS_NEW_MASK` bit-packing (from `src/cpu/nndescent.rs:70-155`)
- `Update<T>` + its `RadixKey` impl (from `src/cpu/nndescent.rs:176-221`)
- `ApplySortedUpdates<T>` trait definition (from `src/cpu/nndescent.rs:228-252`)
- `UnsafeGraphPtr<T>` wrapper (from `src/cpu/nndescent.rs:166-170`)

Justification: RNN-Descent's flat `n * S` neighbour graph uses the same bit-packed layout with a new/old flag and sorted-by-distance slots, and its AddReverseEdges step is exactly the radix-sorted `(target, source, dist)` update pattern. Extracting these four items unifies two nearly-identical structures. Rewire `src/cpu/nndescent.rs` and `src/gpu/nndescent_gpu.rs` to `use crate::utils::nndescent_utils::*`.

**Do not extract:** `Forest<T>`, `build_candidates`, `mark_as_old`, `generate_updates_for_chunk`, `diversify_graph`, `compute_chunk_size`, `NNDescentQuery<T>`. These are NN-Descent-specific control flow; RNN-Descent replaces the entire candidate-selection loop with its own UpdateNeighbors kernel, and neither RNN-Descent nor NSG uses the per-metric `(pid, dist)` beam-search variants (both use flat-u32 graphs with `SearchState<T>`).

**Reuse from Vamana without extracting:**
- `compute_medoid` — free function at `src/cpu/vamana.rs:251` (already `pub`). NSG snaps centroid to nearest data point via one extra beam-search hop.
- `SearchState<T>` from `src/utils/graph_utils.rs` — the right abstraction for both new indices.
- `StripedLocks` from `src/utils/parallelism.rs` — reused directly.
- The `VamanaState<T>` trait pattern for f32/f64 thread-locals — mirror it for `NsgState<T>` and `RnnDescentState<T>`.

Do NOT reuse `VamanaConstructionGraph` verbatim — NSG doesn't want `initialise_random`, and its fill-once-then-DFS-patch lifecycle differs. Copy the ~80-line pattern into `src/cpu/nsg.rs` and drop the random init.

## NSG (Phase 1a)

**Files:**
- New: `src/cpu/nsg.rs` (target ~900 lines)
- New: `examples/gridsearch_nsg.rs` modelled on `examples/gridsearch_vamana.rs`
- Modified: `src/cpu/mod.rs` (append `pub mod nsg;`), `src/lib.rs` (append NSG wrappers)

**Public API in `src/lib.rs`:**
```rust
pub fn build_nsg_index<T>(mat, dist_metric, r, l_build, c, seed, verbose) -> Result<NsgIndex<T>, AnnSearchErrors>
pub fn build_nsg_from_knn_index<T>(mat, &nndescent_idx, r, l_build, c, seed) -> Result<NsgIndex<T>, AnnSearchErrors>
#[cfg(feature = "gpu")]
pub fn build_nsg_from_gpu_nndescent<T, R>(mat, &nndescent_gpu_idx, r, l_build, c, seed) -> ...
pub fn query_nsg_index<T>(query_mat, &index, k, ef_search, return_dist, verbose) -> KnnOptionResult<T>
pub fn query_nsg_self<T>(&index, k, ef_search, return_dist, verbose) -> KnnOptionResult<T>
```

Rationale for two build entry points (three with GPU): the internal-NN-Descent path is the default, single-call convenience. The `_from_knn_index` variant lets users share one kNN graph across NSG/RNN experiments without paying the build cost twice.

**Build steps in `NsgIndex::build_from_knn`:**
1. Compute centroid via `compute_medoid`-style pass, then snap centroid → nearest data point via one beam-search on `G_knn`. That is `n_p`.
2. Parallel over `v ∈ 0..n` (Rayon): beam-search `n_p → v` on `G_knn` with pool `L`, collect the entire visited set (not just the top-L), cap at `C ≈ 500`, union in `v`'s direct kNN neighbours from `G_knn`. Sort ascending by δ(·, v). Apply MRNG prune: for each candidate `p` in order, accept iff no already-selected `r` satisfies δ(p, r) < δ(p, v). Cap at `R`. Reverse edges are NOT added at this step.
3. Sequential DFS from `n_p`. For each unreachable `u`, run search-on-graph on the partial NSG to find nearest reachable `t`, add edge `t → u`. If `t` is at cap, evict its farthest neighbour. Terminates because each iteration shrinks the disconnected set by ≥1.
4. Flatten into `graph: Vec<u32>` of size `n * R` with `u32::MAX` sentinels.

Reuse `SearchState<T>` via a `NsgState<T>` trait mirroring `VamanaState<T>` (`src/cpu/vamana.rs:315-359`). Query path is a copy-adapt of Vamana's `query`, `query_row`, `generate_knn` — greedy beam from `n_p` with pool `L_search`.

**Errors:** no new variants. `DimensionMismatch` and `DistanceNotSupported` cover the failure modes. DFS is guaranteed to converge.

**MRNG vs Vamana `robust_prune`:** structurally similar loop but the acceptance rule differs. Vamana uses `alpha * δ(cand, selected) ≤ δ(cand, base)`; NSG uses `δ(p, r) < δ(p, v)` with no alpha. Do not try to share code — the semantics are close enough to invite bugs.

## Relative NN-Descent (Phase 1b)

**Files:**
- New: `src/cpu/rnn_descent.rs` (target ~1000 lines)
- New: `examples/gridsearch_rnn_descent.rs`
- Modified: `src/cpu/mod.rs`, `src/lib.rs`

**Public API:**
```rust
pub fn build_rnn_descent_index<T>(mat, dist_metric, s, r, t1, t2, seed, verbose) -> Result<RnnDescentIndex<T>, AnnSearchErrors>
pub fn query_rnn_descent_index<T>(query_mat, &index, k, ef_search, return_dist, verbose) -> KnnOptionResult<T>
pub fn query_rnn_descent_self<T>(&index, k, ef_search, return_dist, verbose) -> KnnOptionResult<T>
```

Paper defaults: `s=20`, `r=96`, `t1=4`, `t2=15`. Query beam default 100.

**Data layout:** during refinement, flat `Vec<Neighbour<T>>` of size `n * R` from `nndescent_utils`, sorted-by-distance per slot, tail-padded with sentinels. Final graph flattens to `Vec<(u32, T)>` for cache-friendly beam-search reads (matches NN-Descent's final layout).

**Build loop (Algorithm 6):**
1. Random init: each node picks `S=20` out-neighbours, all flagged new.
2. Outer `t1` in `1..T1`:
   - Inner `t2` in `1..T2`: run UpdateNeighbors (per-node, parallel).
   - If `t1 != T1`: AddReverseEdges — reuse `Update<T>` + `radix_sort_unstable` + `ApplySortedUpdates<T>` directly.
3. Return the current graph.

**UpdateNeighbors (per node `u`, parallel):** sort out-neighbours by distance; walk ascending; for each candidate `v`, scan already-accepted `w`s: skip check if both `(u,v)` and `(u,w)` are old-flagged; else compute δ(v, w) and test RNG rule `δ(u,v) < δ(v,w) AND δ(u,w) < δ(v,w)`; if violated, remove `(u,v)` and enqueue reinsert `(w, v, δ(w,v))` (target `w`, add `v` to its list). Surviving edges flip old at end of pass.

**AddReverseEdges:** symmetrise via `Update<T>` triples, radix sort by target, apply lock-free via `ApplySortedUpdates<T>`, then cap each per-node adjacency list at top-R by distance.

**Query:** fresh beam-search over the final `(u32, T)` graph using `SearchState<T>` — do NOT reuse `NNDescentQuery<T>`. Entry point: nearest of a random subset of R nodes (paper's approach). Simpler than NN-Descent's Forest-based entry.

**Errors:** no new variants.

## GPU integration (Phase 2 and 3)

**Phase 2 — GPU NN-Descent → CPU NSG.** Low effort. `NNDescentGpu<T, R>` already builds a complete graph on-device (see `extract_nndescent_knn_gpu` at `src/lib.rs:2438`). Add a `NsgIndex::build_from_gpu_nndescent` constructor that pulls the graph once via the existing extract path, then runs the pure-CPU NSG build. Interface change: 1 new `pub fn` in `src/lib.rs` behind `#[cfg(feature = "gpu")]`, 1 new constructor in `src/cpu/nsg.rs`. ~50 lines total. No new CubeCL kernels.

**Assessment for the user's four questions:**

- **GPU NN-Descent → NSG:** feasible in Phase 2 as described. ~1 day of work.
- **NSG MRNG prune on GPU:** feasible with caveats. Per-node work is O(C·R) with C≈500 and R≈50 — fixed-size loops, shared-memory-friendly, structurally similar to the existing `cagra_rank_prune_shared` kernel at `src/gpu/nndescent_gpu.rs:825`. Accept/reject decisions map to shared-memory bitsets. Worth it only if profiling shows CPU MRNG dominating build time on large n (likely at n > 10M). Ship in Phase 3.
- **NSG DFS connectivity fix on GPU:** don't try. Data-dependent traversal, and post-step-2 the disconnected set is typically <0.1% of n. Keep on CPU.
- **RNN-Descent refinement on GPU:** painful, low ROI in the first pass. The pruned-edge reinsertion `(u,v) → (w,v)` is a scatter with unpredictable target distribution; you'd need per-block reinsert queues plus a host round-trip for the radix sort. Skip for now, ship RNN CPU-only, revisit only if benchmarks show a bottleneck.

## Verification

**Per-module unit tests (bottom of each new file, matching `vamana.rs` pattern with ~15 tests):**
- NSG: entry-node validity, MRNG monotonicity, DFS reachability post-fix, R-cap enforcement, recall > 0.95 on 500×8 clustered fixture.
- RNN-Descent: convergence within T1·T2 iters, in/out degree ≤ R post-AddReverseEdges, per-adjacency distance-monotonic order, recall > 0.90 on same fixture.

**End-to-end gridsearches** on 100k × 128 synthetic clusters via `examples/commons/mod.rs`:
- `gridsearch_nsg`: (R ∈ {32, 48, 64}) × (L ∈ {50, 100, 150}) × ef_search ∈ {50, auto, 150}. One extra column for internal-kNN vs external-NNDescent. Compare against Vamana rows.
- `gridsearch_rnn_descent`: (S ∈ {20, 40}) × (R ∈ {64, 96, 128}) × (T1 ∈ {3, 4}) × ef_search ∈ {50, 100, 150}. Compare against NN-Descent at same k.

**Docs update:** append NSG and RNN-Descent sections to `docs/benchmarks_standard.md`; add GPU-fed NSG row to `docs/benchmarks_gpu.md` in Phase 2.

**Go/no-go gates per phase:**
- Phase 0: `cargo test --release --features binary,quantised,gpu` fully green; `gridsearch_nndescent` recall unchanged on fixed seed.
- Phase 1: both new `gridsearch_*` examples produce full recall tables. NSG's best row within ±2pp of Vamana's best at equal `ef`; RNN-Descent's build time competitive with NN-Descent + NSG.
- Phase 2: GPU-fed NSG matches CPU-fed NSG recall within 1pp on same seed.
- Phase 3 (if pursued): GPU-pruned NSG matches CPU-pruned NSG recall within 1pp, build time drops ≥3× on n ≥ 1M.

## Phasing and parallel agents

| Phase | Files touched | Agents | Parallel? |
|---|---|---|---|
| 0 (refactor) | new `src/utils/nndescent_utils.rs`, `src/utils/mod.rs`, `src/cpu/nndescent.rs`, `src/gpu/nndescent_gpu.rs` | 1 | No — blocks everything |
| 1a (NSG CPU) | `src/cpu/nsg.rs`, `src/cpu/mod.rs`, `src/lib.rs`, `examples/gridsearch_nsg.rs` | 1 | Yes — parallel with 1b |
| 1b (RNN CPU) | `src/cpu/rnn_descent.rs`, `src/cpu/mod.rs`, `src/lib.rs`, `examples/gridsearch_rnn_descent.rs` | 1 | Yes — parallel with 1a |
| 2 (GPU NND → NSG) | `src/lib.rs`, `src/cpu/nsg.rs`, `docs/benchmarks_standard.md`, `docs/benchmarks_gpu.md` | 1 | Sequential after 1a |
| 3 (GPU MRNG prune, optional) | new `src/gpu/nsg_gpu.rs`, `src/gpu/mod.rs`, `src/lib.rs` | 1 | Standalone, no dependencies |

Phase 1's only shared files are `src/cpu/mod.rs` (one-line module decl) and `src/lib.rs` (append-only wrapper blocks). Both agents append at end-of-file under clear `// NSG //` / `// RNN-Descent //` banners matching existing style. Merge conflict, if any, is trivial.

## Critical files

- `src/cpu/nndescent.rs` — source of primitives to extract
- `src/cpu/vamana.rs` — closest structural template for NSG
- `src/utils/graph_utils.rs` — `SearchState<T>` reused
- `src/utils/parallelism.rs` — `StripedLocks` reused
- `src/lib.rs` — public wrappers appended
- `src/gpu/nndescent_gpu.rs` — GPU NN-Descent extract path for Phase 2
- `examples/commons/mod.rs` — shared synthetic-data + benchmark plumbing
- `docs/benchmarks_standard.md` — user-facing results surface
