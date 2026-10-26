"""Kairos causal action q implementation."""

from __future__ import annotations

import math

from kairos._runtime.compat.call_commitment import Job, evaluate


def _residual_distribution(quote):
    current = float(quote.get("current_service_s", 0.0))
    values = quote.get("service_distribution_s")
    if not values:
        return None
    values = tuple(float(x) for x in values)
    if any(not math.isfinite(x) or x < 0.0 for x in values):
        return None
    residual = tuple(max(0.0, x - current) for x in values)
    weight = 1.0 / len(residual)
    return tuple((x, 0.0, weight) for x in residual)


def action_q_order(names, quotes):
    """Return a Q-minimizing order and auditable values.

    ``names`` must be the already-selected returning cohort.  Every owner is
    represented by its profiled current-call service and its causal posterior
    residual.  The external return is set to the current decision boundary
    (zero delay) because all candidates are already ready; no future tool
    timing is invented.  ``evaluate`` then compares each possible first
    commitment under common scenarios, so differences are due to the action,
    not independent Monte Carlo noise.

    The function returns ``(None, {}, 0)`` when quotes cannot form a comparison.
    """
    names = tuple(names)
    if len(names) < 2:
        return None, {}, 0
    jobs = []
    for name in names:
        quote = quotes.get(name, {})
        try:
            current = float(quote.get("current_service_s", 0.0))
        except (TypeError, ValueError):
            return None, {}, 0
        if not math.isfinite(current) or current < 0.0:
            return None, {}, 0
        try:
            pairs = _residual_distribution(quote)
        except (TypeError, ValueError):
            return None, {}, 0
        if not pairs:
            return None, {}, 0
        jobs.append(Job(name=name, current=current, pairs=pairs))
    values = {}
    scenario_count = max(len(job.pairs) for job in jobs)
    try:
        for candidate in names:
            values[candidate] = float(evaluate(
                tuple(jobs), (), candidate=candidate, scenarios=scenario_count))
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return None, {}, 0
    if any(not math.isfinite(v) for v in values.values()):
        return None, {}, 0
    order = sorted(names, key=lambda p: (values[p], names.index(p)))
    return order, values, scenario_count


def exact_action_q_order(names, quotes):
    """Integrate the all-ready aggregate-suffix model without scenarios.

    Program posteriors are independent. After the forced current call,
    clearing is by observable mean, with the same name tie-break as ``clear``.
    Only the forced candidate can disappear before clearing (zero suffix).
    Each pair therefore contributes either the earlier job's expected work,
    or that work times the candidate's survival probability. See the method
    document for the pairwise completion-area derivation and its limits.

    This is exact for that surrogate, not for batched self-suspending serving.
    Zero returned scenarios explicitly means analytic integration, not a
    missing distribution. Keep the sampled solver under its original tag.
    """
    names = tuple(names)
    if len(names) < 2 or len(set(names)) != len(names):
        return None, {}, 0
    current, means, survives, totals = {}, {}, {}, {}
    for name in names:
        quote = quotes.get(name, {})
        try:
            c = float(quote.get("current_service_s", 0.0))
            rows = _residual_distribution(quote)
        except (TypeError, ValueError):
            return None, {}, 0
        if not math.isfinite(c) or c < 0.0 or not rows:
            return None, {}, 0
        current[name] = c
        means[name] = math.fsum(r * p for r, _, p in rows)
        survives[name] = math.fsum(p for r, _, p in rows if r > 0.0)
        totals[name] = c + means[name]
    values = {}
    for p in names:
        others = sorted((q for q in names if q != p),
                        key=lambda q: (totals[q], q))
        # In the surrogate, the current call finishes at C_p. All n programs
        # therefore contribute that common offset. After it, candidate p is
        # either complete or contributes its residual job; other programs
        # contribute their current-plus-suffix aggregate jobs. For a fixed
        # mean order, expected sum completion time is the sum of expected
        # work plus each earlier job's work for every later completion.
        remaining = math.fsum((means[p], *(totals[q] for q in others)))
        later_work = math.fsum((len(others) - index - 1) * totals[q]
                               for index, q in enumerate(others))
        candidate_pairs = math.fsum(
            means[p] if (means[p], p) <= (totals[q], q)
            else survives[p] * totals[q]
            for q in others)
        values[p] = math.fsum((len(names) * current[p], remaining,
                                later_work, candidate_pairs))
    if any(not math.isfinite(v) for v in values.values()):
        return None, {}, 0
    position = {p: i for i, p in enumerate(names)}
    order = sorted(names, key=lambda p: (values[p], position[p]))
    return order, values, 0
