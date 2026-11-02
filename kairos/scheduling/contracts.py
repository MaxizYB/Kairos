"""scheduling.contracts."""

from dataclasses import dataclass

NATIVE_HEADROOM_PAGES = 32


@dataclass(frozen=True)
class CurrentCall:
    total_tokens: float
    remaining_tokens: float
    cap_tokens: int
    exact: bool
    source: str
    atoms: tuple
    mass: tuple


def capacity_pages(
    *,
    prompt,
    cap,
    generated=0,
    computed=0,
    pending=0,
    page=16,
    exact=False,
    max_len=16384,
):
    if (
        any(
            (
                type(x) is not int
                for x in (prompt, cap, generated, computed, pending, page, max_len)
            )
        )
        or prompt < 1
        or cap < 1
        or (page < 1)
        or (generated < 0)
        or (computed < 0)
        or (pending < 0)
        or (prompt + cap > max_len)
    ):
        raise ValueError("Invalid native capacity frontier")
    stop = prompt + cap
    end = stop if exact else min(stop, max(prompt + generated + pending, computed) + 1)
    return (end + page - 1) // page


@dataclass(frozen=True)
class ServiceOrder:
    ready: tuple
    priorities: dict
    boundaries: dict


@dataclass(frozen=True)
class Admission:
    funded: bool
    victims: tuple
    reason: str
    required_pages: int
    free_pages: int


def admission(
    owner,
    target,
    reserved,
    occupied,
    optional,
    priorities,
    limit,
    max_running,
    *,
    claim_pages=0,
    claim_pending=False,
    locked=(),
):
    if (
        target < 1
        or limit < 1
        or max_running < 1
        or (claim_pages < 0)
        or any((v < 0 for v in reserved.values()))
        or (owner not in priorities)
        or (not set(optional) <= priorities.keys())
    ):
        raise ValueError("Invalid admission snapshot")
    used = sum(reserved.values())
    extra = target - reserved.get(owner, 0)
    free = max(0, limit - used)
    required = max(0, extra)
    if target > limit:
        return Admission(False, (), "oversize-contract", required, free)
    if owner not in occupied and len(occupied) >= max_running:
        return Admission(False, (), "execution-slots", required, free)
    credit = min(free, claim_pages)
    if used + extra <= limit - credit:
        return Admission(True, (), "free-capacity", required, free)
    if credit or claim_pending:
        return Admission(False, (), "higher-priority-capacity-claim", required, free)
    victims = []
    eligible = set(optional) - set(occupied) - set(locked)
    for other in sorted(eligible, key=priorities.__getitem__, reverse=True):
        if other == owner or priorities[other] <= priorities[owner]:
            continue
        victims.append(other)
        used -= reserved.get(other, 0)
        if used + extra <= limit:
            return Admission(
                True, tuple(victims), "lower-priority-optional", required, free
            )
    return Admission(False, (), "protected-capacity-wait", required, free)
