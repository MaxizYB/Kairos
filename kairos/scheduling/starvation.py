"""Queue-local bounded aging for the Kairos ready path."""

from __future__ import annotations

import math


def aged_priority(
    base_priority: int,
    waited_s: float,
    threshold_s: float = 4.0,
    quantum_s: float = 1.0,
) -> dict[str, float | int | bool]:
    """Return queue priority with bounded waiting-time credit."""
    if isinstance(base_priority, bool) or int(base_priority) != base_priority:
        raise ValueError("base_priority must be an integer")
    base = int(base_priority)
    if base < 0:
        raise ValueError("base_priority must be non-negative")
    if not all(math.isfinite(float(x)) for x in (waited_s, threshold_s, quantum_s)):
        raise ValueError("aging inputs must be finite")
    if threshold_s < 0 or quantum_s <= 0:
        raise ValueError("invalid aging parameters")
    waited = max(0.0, float(waited_s))
    if waited < threshold_s or base == 0:
        return dict(
            priority=base, base_priority=base, credit=0, waited_s=waited, aged=False
        )
    credit = min(base, 1 + int((waited - threshold_s) // quantum_s))
    return dict(
        priority=base - credit,
        base_priority=base,
        credit=credit,
        waited_s=waited,
        aged=credit > 0,
    )
