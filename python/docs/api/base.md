# Estimator base

Shared `fit` / `kneighbors` / `save` plumbing. You never instantiate these
directly, but every estimator inherits the methods documented here.

::: ann_search._base
    options:
      members:
        - BaseAnnIndex
        - ExtractKnnMixin
        - NotFittedError
