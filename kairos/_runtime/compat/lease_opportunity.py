"""Kairos lease opportunity implementation."""
from kairos._runtime.compat.r83_plan import State, preview


def hold_opportunity(state, owner, calls, limit, max_active):
    """Return a one-program waiting-area rate and an auditable witness.

    Both states are AFTER owner's real cleanup; other DMA/current contracts
    stay locked. A retained lease keeps both pages and membership, matching
    incumbent call-commitment protocol. Existing reclaim rights participate in both feasibility checks.
    Multiple waiting calls are alternatives, not additive opportunity costs.
    No completion-time improvement or future-return prediction is asserted.
    """
    if owner in state.locked:
        raise ValueError('Finishing owner must be removed from current-call locks')
    released = State(tuple((p, n) for p, n in state.reserved if p != owner),
                     state.active - {owner},
                     tuple(p for p in state.victims if p != owner), state.locked)
    for call in calls:
        if call.program == owner or call.program in state.locked:
            continue
        held_action = preview(state, call, limit, max_active)
        released_action = preview(released, call, limit, max_active)
        if held_action is None and released_action is not None:
            return 1., dict(program=call.program, pages=call.pages,
                            entered=call.entered, victims=list(released_action.victims))
    return 0., None
