"""continuity.economics."""

from collections import Counter
from dataclasses import dataclass
import math
from kairos.scheduling.capacity import admission


@dataclass(frozen=True)
class ReturnLaw:
    durations: tuple
    horizon: float

    @classmethod
    def from_policy(cls, policy, tool):
        prior = tuple(policy.prior_history.get(tool) or policy.global_prior)
        live = tuple(policy.live_history.get(tool, ()))
        rows = prior + live
        if not rows:
            return cls((), 0.0)
        values = tuple((float(row[0]) for row in rows))
        if any((not math.isfinite(x) or x < 0 for x in values)):
            raise ValueError("Only completed finite tool durations are admissible")
        return cls(values, max((float(row[0]) for row in prior or live)))

    def choose(self, benefit, rate, *, age=0.0, deadline=None):
        if any((not math.isfinite(x) or x < 0 for x in (benefit, rate, age))):
            raise ValueError("Finite nonnegative causal lease quotes required")
        cap = self.horizon if deadline is None else min(self.horizon, deadline)
        if not math.isfinite(cap) or cap < 0:
            raise ValueError("Invalid absolute lease horizon")
        remaining = sorted(
            (d - age for d in self.durations if d > age or (age == 0.0 and d == 0.0))
        )
        upper = max(0.0, cap - age)
        candidates = sorted({0.0} | {d for d in remaining if d <= upper})
        best = dict(ttl_s=0.0, utility_s=0.0, probability=0.0, expected_idle_s=0.0)
        evidence = []
        n = len(remaining)
        prefix = 0.0
        index = 0
        for tau in candidates:
            while index < n and remaining[index] <= tau:
                prefix += remaining[index]
                index += 1
            probability = index / n if n else 0.0
            held = (prefix + (n - index) * tau) / n if n else 0.0
            row = dict(
                ttl_s=tau,
                utility_s=probability * benefit - rate * held,
                probability=probability,
                expected_idle_s=held,
            )
            evidence.append(row)
            if row["utility_s"] > best["utility_s"]:
                best = row
        boundary = next(
            (
                d
                for d in sorted(set(self.durations))
                if d > age and d - age == best["ttl_s"]
            ),
            age,
        )
        return dict(
            best,
            deadline_age_s=boundary,
            benefit_s=benefit,
            exposure_rate=rate,
            age_s=age,
            horizon_s=cap,
            surviving_samples=n,
            completed_samples=len(self.durations),
            tail_samples=sum((d > cap for d in self.durations)),
            candidates=evidence,
            source="causal-bounded-return-law",
            future_information=False,
        )


def exclusive_pages(blocks, retiring=()):
    own = Counter((b.block_id for b in blocks))
    native = Counter((b.block_id for b in retiring))
    refs = {b.block_id: b.ref_cnt for b in blocks}
    if any((getattr(b, "is_null", False) for b in blocks)):
        raise ValueError("Null blocks are not retention evidence")
    if retiring:
        if any((refs[bid] < native[bid] for bid in own)):
            raise ValueError("Inconsistent retiring reference evidence")
        return frozenset((bid for bid in own if refs[bid] == native[bid]))
    if any((refs[bid] < count for bid, count in own.items())):
        raise ValueError("Inconsistent pin reference evidence")
    return frozenset((bid for bid, count in own.items() if refs[bid] == count))


def scan_frontier(
    calls, reserved, occupied, optional, priorities, limit, slots, locked=()
):
    ledger = dict(reserved)
    active = set(occupied)
    options = set(optional)
    selected = []
    checks = []
    claim = None
    for owner, target in calls:
        credit = (
            min(max(0, limit - sum(ledger.values())), claim) if claim is not None else 0
        )
        plan = admission(
            owner,
            target,
            ledger,
            active,
            options,
            priorities,
            limit,
            slots,
            claim_pages=credit,
            claim_pending=claim is not None,
            locked=locked,
        )
        checks.append(
            dict(
                program=owner,
                funded=plan.funded,
                reason=plan.reason,
                victims=list(plan.victims),
                required_pages=plan.required_pages,
            )
        )
        if plan.funded:
            for victim in plan.victims:
                ledger.pop(victim, None)
                options.discard(victim)
            ledger[owner] = target
            active.add(owner)
            options.discard(owner)
            selected.append(owner)
        elif claim is None and plan.reason == "protected-capacity-wait":
            claim = plan.required_pages
    return dict(selected=selected, checks=checks, reserved_end=ledger)


def marginal_exposure(
    owner,
    unique_pages,
    calls,
    reserved,
    occupied,
    optional,
    priorities,
    limit,
    slots,
    locked=(),
):
    if type(unique_pages) is not int or not 0 <= unique_pages <= reserved.get(owner, 0):
        raise ValueError("Release credit must be actual unique optional pages")
    keep = scan_frontier(
        calls, reserved, occupied, optional, priorities, limit, slots, locked
    )
    released = dict(reserved)
    released[owner] = released.get(owner, 0) - unique_pages
    unpin = scan_frontier(
        calls,
        released,
        occupied,
        set(optional) - {owner},
        priorities,
        limit,
        slots,
        locked,
    )
    advanced = sorted(
        set(unpin["selected"]) - set(keep["selected"]), key=priorities.__getitem__
    )
    displaced = sorted(
        set(keep["selected"]) - set(unpin["selected"]), key=priorities.__getitem__
    )
    return dict(
        rate=float(len(advanced)),
        advanced=advanced,
        displaced=displaced,
        unique_release_pages=unique_pages,
        keep=keep,
        unpin=unpin,
        exact_JCT_value=False,
        future_information=False,
    )
