# News

Changes to the `ann-search` Python package. The Rust crate it wraps,
`ann-search-rs`, has its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.2.1

Requires `ann-search-rs` 0.8.2.

- Documentation updates where out-of-date things were claimed.
- Wiring in the faster RNN from the Rust parent package.

## 0.2.0

Requires `ann-search-rs` 0.8.1. First release on PyPI.

- The eleven quantised indices are now bound: `ExhaustiveBf16Index`,
  `IvfBf16Index`, `ExhaustiveSq8Index`, `IvfSq8Index`, `HnswSq8uIndex`,
  `ExhaustivePqIndex`, `IvfPqIndex`, `ExhaustiveOpqIndex`, `IvfOpqIndex`,
  `SoarPqIndex`, `SoarOpqIndex`. Same four-method surface as the rest, save /
  load / pickle included, float32 and float64 both supported. The binary
  indices are not bound yet.
- Docs corrected against the regenerated benchmark tables. HNSW is now the
  cheapest graph index to build rather than the most expensive, which reversed
  three separate claims in `choosing.md`.

## 0.1.0

Requires `ann-search-rs` 0.7.0. Never published; installable from the repo.

- Python bindings under `python/`, built with PyO3 and maturin. scikit-learn
  shaped estimators over the CPU indices, plus the synthetic generators.
