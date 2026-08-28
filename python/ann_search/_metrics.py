"""Metric names, in one place.

The Rust core takes a metric string and falls back to squared Euclidean when it
doesn't recognise one, warning via ``println!``. That warning is invisible in a
notebook: it goes to the process stdout, not ``sys.stdout``. So a typo would
silently change the answer. Nothing here ever sends the core a string it hasn't
already validated.

``"euclidean"`` maps to ``Dist::SquaredEuclidean`` in the core, so the true-L2
spellings carry a flag telling the caller to take the square root of what comes
back. ``"sqeuclidean"`` skips it, matching scipy's naming.
"""

from beartype import beartype

###########
# Globals #
###########

# name -> (string the Rust core wants, take the square root of the distances)
_METRICS: dict[str, tuple[str, bool]] = {
    "euclidean": ("euclidean", True),
    "l2": ("euclidean", True),
    "sqeuclidean": ("euclidean", False),
    "cosine": ("cosine", False),
    "manhattan": ("manhattan", False),
    "l1": ("manhattan", False),
}

#: Every metric the Python layer accepts.
ALL_METRICS: frozenset[str] = frozenset(_METRICS)

#: Metrics available to indices that cannot do Manhattan.
NO_MANHATTAN: frozenset[str] = ALL_METRICS - {"manhattan", "l1"}


@beartype
def resolve_metric(metric: str, supported: frozenset[str]) -> tuple[str, bool]:
    """Validate a metric name and translate it for the Rust core.

    Args:
        metric: Metric name as the user spelled it.
        supported: The subset this particular index accepts.

    Returns:
        ``(core_name, take_sqrt)``, where ``take_sqrt`` says whether the
        distances coming back need a square root to become true Euclidean.

    Raises:
        ValueError: If the metric is unknown or unsupported by this index.
    """
    key = metric.lower()
    if key not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(f"unsupported metric {metric!r}; expected one of: {allowed}")
    return _METRICS[key]
