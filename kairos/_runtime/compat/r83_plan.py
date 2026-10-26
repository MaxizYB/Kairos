"""Kairos r83 plan implementation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    reserved: tuple
    active: frozenset
    victims: tuple
    locked: frozenset = frozenset()


@dataclass(frozen=True)
class Call:
    program: str
    pages: int
    entered: bool


@dataclass(frozen=True)
class Grant:
    program: str
    pages: int
    victims: tuple
    after: State


def preview(state, call, limit, max_active, *, recovery_rank=None):
    """Dry-run exactly the incumbent call-commitment protocol legal reclaim branches, without side effects.

    A resident return may recover its observed growth. A nonresident return
    may use only uncommitted space. A never-entered call keeps the inherited
    optional-idle fallback. Victim ORDER, not a new victim objective, is input.
    """
    if call.pages <= 0 or limit < 0 or max_active <= 0:
        raise ValueError('Invalid complete-call capacity contract')
    p = call.program
    if p in state.locked:
        raise ValueError('Cannot readmit a committed call')
    reserved = dict(state.reserved)
    active = set(state.active)
    used = sum(reserved.values()) - reserved.get(p, 0) + call.pages
    def fits():
        return used <= limit and (p in active or len(active) < max_active)
    removed = []
    can_reclaim = ((p in active) if call.entered else (p not in active))
    if recovery_rank is not None:
        # The v11 frontier uses the same order for entering and relinquishing
        # optional support. Losing residency does not remove a ready call's
        # right to compete, but it cannot reclaim a preceding owner's lease.
        can_reclaim = True
    if not fits() and can_reclaim:
        for q in state.victims:
            if q == p or q in state.locked:
                continue
            if recovery_rank is not None and recovery_rank[q] <= recovery_rank[p]:
                continue
            used -= reserved.pop(q, 0)
            active.discard(q)
            removed.append(q)
            if fits():
                break
    if not fits():
        return None
    reserved[p] = call.pages
    active.add(p)
    remaining = tuple(q for q in state.victims if q not in removed)
    after = State(tuple(sorted(reserved.items())), frozenset(active), remaining,
                  state.locked | {p})
    return Grant(p, call.pages, tuple(removed), after)


def plan(state, calls, limit, max_active, window, *, recovery_rank=None):
    """Stable ranked feasible sweep; no failed action can finance a later one."""
    grants, blocked = [], []
    for call in calls:
        if len(grants) >= window:
            break
        action = preview(state, call, limit, max_active, recovery_rank=recovery_rank)
        if action is None:
            blocked.append(call.program)
        else:
            grants.append(action)
            state = action.after
    return tuple(grants), tuple(blocked), state
