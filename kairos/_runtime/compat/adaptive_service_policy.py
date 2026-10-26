"""Kairos adaptive service policy implementation."""

import math
import statistics


def _clean(values, name):
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def choose(*, current, residual, warm):
    """Return an auditable progressive-service rank.

    ``current`` is the predicted service of the next complete call and
    ``residual`` is the predicted service from the current boundary onward.
    A cold Program has no identifiable suffix, so only the current-call
    frontier is used.  Once at least one call has completed, the refreshed
    residual distribution becomes available.  The returned ``cost_s`` is a
    rank key only; it is not a reservation or a future-memory commitment.
    """
    current = _clean(current, "current")
    residual = _clean(residual, "residual")
    if type(warm) is not bool:
        raise TypeError("warm must be bool")
    source = residual if warm else current
    q25 = _quantile(source, 0.25)
    median = _quantile(source, 0.50)
    q75 = _quantile(source, 0.75)
    # Median is robust to the very wide cold-start suffix support.  The
    # interval is exposed for diagnostics and future confidence gating, but
    # no hand-tuned uncertainty penalty is injected into the rank.
    return {
        "mode": "warm-residual" if warm else "cold-probe",
        "cost_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "uncertainty_s": float(max(0.0, q75 - q25)),
        "current_median_s": float(_quantile(current, 0.50)),
        "residual_median_s": float(_quantile(residual, 0.50)),
        "samples": len(source),
        "warm": warm,
    }


def order_key(details):
    """Return a deterministic key for scheduler-side detail records."""
    return (float(details["cost_s"]), float(details.get("uncertainty_s", 0.0)))

