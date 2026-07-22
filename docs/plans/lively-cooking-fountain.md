# NSG build fix — reverse edges, DFS-patch, entry-point seeding

## Context

`examples/gridsearch_nsg.rs` on 150k × 32 clustered data reports:

- Only about 6.5k of 150k nodes reachable from the navigating node after the parallel MRNG step (`DFS: patching 143_469 unreachable nodes`).
- Query recall at `ef=50` collapses to 0.84 with a mean distance ratio of ~267x. Recall is only respectable when `ef` matches `l_build` (0.9996 at ef=150), which defeats the point of an NSG.
- The DFS patch loop then runs 143k iterations sequentially, dominating build time (~26s for 150k points).

The parallel MRNG code (`nsg.rs:678-708`), the connectivity fix (`nsg.rs:966-1013`) and the navigating-node picker (`nsg.rs:740-801`) all differ from the ZJULearning reference implementation in ways that add up to the observed behaviour. The root cause is a missing reverse-edge pass, compounded by two smaller defects.

## Root cause (verified against the paper and reference C++)

**1. `InterInsert` is missing entirely.** Reference NSG's `Link()` runs `get_neighbors` → `sync_prune` → `InterInsert` per node; `InterInsert` inserts each accepted edge `v → u` back into `u`'s adjacency (re-pruning under MRNG at capacity). Our `build_from_knn` jumps straight from the parallel MRNG at `nsg.rs:708` to `dfs_connectivity_fix` at `nsg.rs:711`. `grep -n 'reverse\|inter_insert\|reciprocal' src/cpu/nsg.rs` returns nothing. Vamana already implements the same idea at `vamana.rs:533-569` — it works there and NSG needs the same step. Without reverse edges the graph is a one-directional cone; forward DFS from the navigating node collapses onto a small subtree and beam search gets stuck unless the beam is very wide.

**2. `add_or_evict_neighbour` silently drops the patch.** `nsg.rs:264-289` returns without inserting when `new_dist >= worst_dist`, but the caller at `nsg.rs:1011` unconditionally sets `reachable[u] = true`. The final flat graph then contains a node marked reachable in the DFS bitmap but with no incoming edge from the reachable component. Reference `findroot` uses unconditional `push_back`.

**3. DFS-patch does not propagate reachability.** After adding `t → u`, we mark only `u` reachable (`nsg.rs:1011`) and iterate over a snapshotted `unreachable` list (`nsg.rs:985`). Reference `tree_grow` interleaves `DFS` and `findroot` in a `while (unlinked_cnt < nd_)` loop and re-runs DFS from each newly linked root, so entire subtrees flood-fill in one go. Ours doesn't, so 143k patches are executed even when a few thousand DFS restarts would cover the same set.

**4. Navigating node beam seeds a single random entry (`nsg.rs:749`).** Reference `Init_Center` seeds `search_with_L` with `L` random `init_ids` and takes `retset[0]`. Our single-entry beam gets trapped in whichever cluster the random entry lands in, so the pick is a cluster-interior point rather than a genuine medoid. Combined with (1), this is why DFS from `np` covers such a tiny fraction — `np` is not central in any global sense.

The MRNG rule itself (`nsg.rs:928-944`), the strict-inequality tie-break, the `C` cap, and the merge with `v`'s kNN neighbours are all correct against paper Alg. 2 and the reference.

## Plan

All four fixes go in `src/cpu/nsg.rs`. Test target is the gridsearch example on 150k × 32.

### 1. Add `InterInsert` after MRNG — the primary fix

Add per-node striped locks to `NsgConstructionGraph`. Use the same `StripedLocks` primitive Vamana already has (see `vamana.rs:34-70` and the `_guard = build_graph.locks.lock_guard(q)` pattern at `vamana.rs:536`). Wire it into `nsg.rs:188-213`.

Widen the per-node adjacency in `NsgConstructionGraph` from `Vec<u32>` to `Vec<(u32, T)>` — store neighbour id **with** distance-to-source. This is the fused-kernel move: InterInsert can then re-prune u's adjacency without recomputing u's distances to its existing R neighbours, which is where MRNG re-prune's cost lives. Flatten at the end drops the distance field for cache locality in the query path.

Extend the parallel loop at `nsg.rs:678-708`: after computing `neighbours` for source `v` and writing them via `set_neighbours` (now `Vec<(u32, T)>` with distances known from the MRNG selection), iterate each accepted `u ∈ neighbours` and, under `locks.lock_guard(u)`:

- If `u`'s degree < R: push `(v, dist(v, u))` — no re-prune.
- Else: read u's cached `Vec<(u32, T)>`, append `(v, dist(v, u))`, re-run MRNG selection on that pool, overwrite u's adjacency with the result (still `Vec<(u32, T)>` with cached distances).

Template: `vamana.rs:533-569`, but with the distance cache instead of recomputing u's distances on every call. The MRNG predicate used inside InterInsert must match the forward-prune predicate (strict `<`, no alpha).

Extract the MRNG selection out of `collect_and_prune_for_node` into a standalone helper (`fn mrng_prune(&self, base: usize, candidates: &mut [(OrderedFloat<T>, usize)], r: usize) -> Vec<u32>`) so both call sites use one implementation. Cost estimate: ~5-8s on 150k × 32 with the distance cache, vs ~15-25s without.

### 2. Rework `dfs_connectivity_fix` to match reference `tree_grow`

Three defects here, all in `nsg.rs:966-1013`:

- **Silent no-op patch**: `add_or_evict_neighbour` (`nsg.rs:264-289`) returns without inserting when `new_dist >= worst_dist`, but the caller unconditionally sets `reachable[u] = true` at `nsg.rs:1011`. Result: `u` is flagged reachable in the DFS bitmap but has no incoming edge in the final graph. This is a straight-up correctness bug.
- **No reachability propagation**: after patching `t -> u` we only flag `u` itself. `u`'s descendants remain on the unreachable list even though they're now trivially reachable via `u`. Wasted work at best, drives the ~15s DFS phase we're trying to eliminate.
- **Snapshotted unreachable set** at `nsg.rs:985`: even with propagation, iterating the stale list re-patches the same subtrees.

Rewrite the loop to match reference `tree_grow`:

```
loop:
    DFS from current root, marking flags persistently
    if all nodes flagged: break
    find first unflagged u
    find nearest flagged t via beam_search_on_partial_graph
    unconditionally push u into t's adjacency
    set root = u (next DFS iteration flood-fills u's subtree)
```

Introduce a `push_neighbour_unchecked` method on `NsgConstructionGraph` that always inserts and permits the node's degree to exceed R by a small amount (aligned with the Q1 choice of variable degree at fix step). The per-node vec is already `Vec<(u32, T)>` from fix (1), so it holds variable length naturally.

Update `NsgConstructionGraph::into_flat` (`nsg.rs:299-311`) to handle variable degree: compute `max_degree = max(nodes.iter().map(len)).max(R)` at flatten time, use that as the flat stride, pad shorter rows with `u32::MAX`. Add `max_degree: usize` to `NsgIndex`, use it in `get_neighbours_flat` (`nsg.rs:1111-1114`) instead of `self.r`.

After fix (1), the patch count should drop from 143k to a couple of hundred at most. Log the actual number in verbose mode to confirm.

### 3. Seed the navigating-node beam with `L` random entries

In `pick_navigating_node` (`nsg.rs:740-801`), draw `l_build` distinct random indices instead of one. Insert each into `state.candidates`, `state.working_sorted`, and mark visited before entering the pop loop. Rest of the beam search is unchanged.

Reference: `Init_Center` in `ZJULearning/nsg/blob/master/src/index_nsg.cpp`.

### 4. (Optional, low priority) Widen candidate-collection seed pool

`collect_and_prune_for_node` at `nsg.rs:846-852` seeds only `np`. Reference `get_neighbors` seeds with `final_graph_[ep_]` (np's L kNN neighbours) padded with random. Small quality improvement; only worth doing if fix 1-3 don't fully close the recall gap. Leave for a follow-up if the main fix lands cleanly.

## Files touched

- `src/cpu/nsg.rs` — all four fixes above.
- Nothing else. No public API changes. `NsgBuildParams` stays as is.

## Verification

Existing test coverage (13 tests including `dfs_reachability_holds` and `recall_validation_high`) plus:

```bash
cargo test --release --lib --features binary,quantised,gpu nsg
```

After the fix the DFS-patch verbose log line should read something like `DFS: patching 200 unreachable nodes` (order of magnitude, not 143k). Then re-run:

```bash
cargo run --release --example gridsearch_nsg
```

Gate criteria on 150k × 32 clustered data:

- `NSG-R32-L150-ef50 (query)` recall ≥ 0.95 (up from 0.836), mean distance ratio close to 1.
- `NSG-R32-L150-efauto (query)` recall ≥ 0.99 with mean distance ratio ≤ 1.1 (down from 15.76).
- Build time drops meaningfully because the DFS-patch loop shrinks by 3+ orders of magnitude.
- Self-query recall ≥ 0.99 with distance ratio ≤ 1.1.

Nice-to-have secondary check: `cargo run --release --example gridsearch_vamana` for a side-by-side sanity read against a working MRNG-based index at similar R/L.
