# Quantised indices

The thirteen estimators over compressed vectors. See
[Quantised](../quantised.md) for what each codec does and when to reach for it.

Two things differ from the uncompressed estimators: the distances are the
codec's estimate rather than the distance (bar `QgIndex`, which keeps the
vectors), and none of them support Manhattan.

::: ann_search.quantised
