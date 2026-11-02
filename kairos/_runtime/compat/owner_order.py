"""runtime.compat.owner order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Admission:
    funded: bool
    victims: tuple
    reason: str
    required_pages: int
    free_pages: int


def preview(
    owner,
    target,
    reserved,
    occupied,
    optional,
    rank,
    limit,
    max_running,
    claim_pages=0,
    claim_pending=False,
):
    used = sum(reserved.values())
    extra = target - reserved.get(owner, 0)
    free = max(0, limit - used)
    if target > limit:
        return Admission(False, (), "oversize-contract", max(0, extra), free)
    if owner not in occupied and len(occupied) >= max_running:
        return Admission(False, (), "execution-slots", max(0, extra), free)
    protected = min(free, max(0, claim_pages))
    if used + extra <= limit - protected:
        return Admission(True, (), "free-capacity", max(0, extra), free)
    if protected or claim_pending:
        return Admission(
            False, (), "higher-priority-capacity-claim", max(0, extra), free
        )
    victims = []
    for victim in sorted(optional, key=lambda p: rank[p], reverse=True):
        if victim == owner or rank[victim] <= rank[owner]:
            continue
        victims.append(victim)
        used -= reserved.get(victim, 0)
        if used + extra <= limit:
            return Admission(
                True, tuple(victims), "lower-priority-optional", max(0, extra), free
            )
    return Admission(False, (), "protected-capacity-wait", max(0, extra), free)


def owner_order(incumbent, ready_scores, cold):
    cold = set(cold)
    mature_ready = set(ready_scores) - cold
    ranked = iter(sorted(mature_ready, key=lambda p: ready_scores[p]))
    owners = [
        next(ranked) if p in mature_ready else p for p in incumbent if p not in cold
    ]
    owners.extend(sorted(cold, key=lambda p: ready_scores[p]))
    return owners
