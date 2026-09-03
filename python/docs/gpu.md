# GPU

Three more estimators. They ship in the ordinary wheel: there's no separate
package and no extra to ask for.

| Class | What it is |
| --- | --- |
| `ExhaustiveGpuIndex` | Brute force on the device. Exact, and cheap ground truth. |
| `IvfGpuIndex` | k-means and vectors both resident. Bounded by device memory. |
| `CagraGpuIndex` | NN-Descent on device, pruned to a CAGRA graph, beam-searched. |

```python
import ann_search as ann

if ann.gpu_available():
    index = ann.CagraGpuIndex(n_neighbors=15).fit(X)
    distances, indices = index.kneighbors()
```

`gpu_available()` answers the only question worth asking: is there an adapter on
this machine. The backend is wgpu, so that means Metal on macOS and Vulkan or
DX12 elsewhere. There's no CUDA runtime to install and nothing extra to
`pip install`. On a box with no GPU it returns False and the CPU estimators are
unaffected.

## When CAGRA is worth the build

The graph costs 1.3 to 3.2 seconds to build on the benchmark shapes, and that
dominates until the dataset is large. For one query batch of 250k points at 128
dimensions `ExhaustiveGpuIndex` finishes the whole job in 2.4 s against CAGRA's
3.2 s, so the exact index wins outright. By 500k at 64 dimensions it reverses,
3.7 s against 5.4 s. Recall moves with size too: 0.95 to 0.97 at 150k in 32
dimensions, against 0.9999 at 250k in 64.

The graph is also not small. It holds 2 to 3x the memory of the raw vectors,
on top of the vectors themselves.

Build it when you're querying repeatedly, or when `n` is past a few hundred
thousand. For a single ground-truth pass, use `ExhaustiveGpuIndex`.

## Three differences, all forced by the backend

**float32 only.** WGSL has no float64, so `fit` narrows rather than failing
somewhere inside a kernel. Note it isn't the only narrowing in the package: a
float64 query against a float32 index is narrowed too, on every index. See the
[Guide](guide.md#dtypes).

**No persistence.** These hold device buffers and sit outside the crate's
`serialise` feature, so `save`, `load` and pickle raise `NotImplementedError`.
Rebuild instead.

**No Manhattan**, on any of the three.

## Thread safety

A fitted `CagraGpuIndex` is the one index here that isn't safe to query from two
threads at once. The beam search memoises its upload of the navigational graph
behind a mutable borrow, and the GIL is released while the kernel runs, so a
second thread entering `kneighbors` doesn't queue behind the first: it raises.
Give each thread its own handle.

`extract_knn` takes a shared borrow, so it never causes the conflict, though it
can still hit one raised by a concurrent query. The other two GPU indices are
fine.

## Getting ground truth cheaply

`ExhaustiveGpuIndex` is exact by construction and has no build-time knobs at
all: the data goes up, the norms get recorded, and that's the index. On a
dataset too large to score on the CPU it's the cheapest route to the truth you
need for a recall measurement.

Don't expect an order of magnitude out of it. On the M1 Max the benchmarks ran
on it comes out around 1.7x faster than the CPU exhaustive path (250k points at
64 dimensions: 1.44 s against 2.41 s). On Apple Silicon the GPU rarely blows
the CPU out of the water. A discrete card with real bandwidth will do better.

```python
truth = ann.ExhaustiveGpuIndex(n_neighbors=15).fit(X).kneighbors(return_distance=False)
```

## The CAGRA beam is all-or-nothing

`CagraGpuIndex` has four search-time knobs: `beam_width`, `max_beam_iters`,
`n_entry_points` and `expand_per_iter`. Leave **every one** of them at `None`
and the beam is sized from `k`:

- `beam_width = 2 * max(k, 16)`
- `max_beam_iters = 3 * beam_width`
- `n_entry_points = 8`, `expand_per_iter = 3`

Set **any one** of them and that scaling switches off for all four, and the
untouched ones fall back to flat constants: `beam_width = 16`,
`max_beam_iters = 48`, `n_entry_points = 8`, `expand_per_iter = 3`.

So asking for `n_entry_points=16` alone on a `k=50` query quietly narrows the
beam from 100 to 16, and recall drops for a reason nothing tells you about. If
you touch one, set `beam_width` too.

```python
# Fine: the library sizes everything from k.
index.kneighbors(Q)

# Fine: explicit about the knob that matters.
index.kneighbors(Q, beam_width=128, max_beam_iters=384)

# Trap: beam_width silently collapses to 16.
index.kneighbors(Q, n_entry_points=16)
```

## A CPU-only build

GPU support is compiled in by default and costs roughly 3 MB of wheel. That
seemed the better trade than a second distribution: wgpu has no driver runtime
to ship, and the alternative is a version pin between two packages that can
drift.

It's fixed at wheel-build time, so a Python extra can't switch it. Extras only
add Python requirements; they never change the compiled artefact. If the
megabytes matter, build from source without it:

```bash
maturin develop --release --no-default-features
```

`gpu_available()` then returns False on any machine, and
`import ann_search.gpu` raises with an explanation. Everything on the CPU side
is untouched.
