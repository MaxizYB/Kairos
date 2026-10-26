"""Kairos commitment order implementation."""
from bisect import bisect_right
import math


def empirical_dominates(left, right):
    """Whether left is first-order stochastically smaller than right.

    Unequal sample counts are supported. Cross-multiplied CDF counts avoid
    floating-point ambiguity at equal mass boundaries. Empty, negative and
    nonfinite quotes cannot justify a swap.
    """
    if not left or not right:
        return False
    if any(not math.isfinite(x) or x < 0 for x in (*left, *right)):
        return False
    a, b = sorted(left), sorted(right)
    strict = False
    for point in sorted(set(a + b)):
        difference = bisect_right(a, point) * len(b) - bisect_right(b, point) * len(a)
        if difference < 0:
            return False
        strict |= difference > 0
    return strict


def refine_order(incumbent, distributions, eligible):
    """Refine eligible slots; ineligible owners keep their original slots.

    Only actual returned calls compete. Running/idle owners do not form
    artificial barriers between ready calls, and an unseen cold suffix never
    authorizes a swap. Stable adjacent dominance swaps never invent a total
    order for overlapping empirical distributions.
    """
    result = list(incumbent)
    positions = [i for i, owner in enumerate(result) if owner in eligible]
    owners = [result[i] for i in positions]
    swaps = []
    for end in range(len(owners) - 1, 0, -1):
        changed = False
        for i in range(end):
            left, right = owners[i:i + 2]
            if empirical_dominates(distributions.get(right), distributions.get(left)):
                owners[i:i + 2] = [right, left]
                swaps.append((left, right))
                changed = True
        if not changed:
            break
    for position, owner in zip(positions, owners):
        result[position] = owner
    return result, swaps
