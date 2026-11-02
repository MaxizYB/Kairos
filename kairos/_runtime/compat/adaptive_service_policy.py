"""runtime.compat.adaptive service policy."""

import math


def _clean(values, name):
    values = tuple((float(value) for value in values))
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any((not math.isfinite(value) or value < 0 for value in values)):
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
    current = _clean(current, "current")
    residual = _clean(residual, "residual")
    if type(warm) is not bool:
        raise TypeError("warm must be bool")
    source = residual if warm else current
    q25 = _quantile(source, 0.25)
    median = _quantile(source, 0.5)
    q75 = _quantile(source, 0.75)
    return {
        "mode": "warm-residual" if warm else "cold-probe",
        "cost_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "uncertainty_s": float(max(0.0, q75 - q25)),
        "current_median_s": float(_quantile(current, 0.5)),
        "residual_median_s": float(_quantile(residual, 0.5)),
        "samples": len(source),
        "warm": warm,
    }


def order_key(details):
    return (float(details["cost_s"]), float(details.get("uncertainty_s", 0.0)))
