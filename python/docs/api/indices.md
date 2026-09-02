# Indices

The uncompressed CPU estimators. Every one takes its parameters in the
constructor, its data in `fit`, and hands results back from `kneighbors`.

Parameters described as search-time can be overridden per call, so
`index.kneighbors(X, ef_search=200)` leaves the fitted index alone. Everything
else is fixed at build time and changing it means refitting.

::: ann_search.indices
