"""Kairos continuation demand implementation."""
import math

from kairos._runtime.compat.oracle_program_plan import Program, Turn


class OracleDemand:
    """Expose a suffix only after that identity has actually arrived.

    No oracle arrivals, fixed rank or precomputed admission calendar is used.
    Tool return time remains an external event, even in the diagnostic arms.
    """
    def __init__(self, plan):
        if not plan.get('future_information') or plan.get('deployable') is not False:
            raise ValueError('Demand truth must disclose nondeployable future information')
        self.rows = plan['programs']

    def forecast(self, name, inputs, outputs, *, after_completion=False,
                 memory_pages=0, page=16, **unused):
        if not inputs:
            raise ValueError('Cannot quote an unarrived identity')
        rows = self.rows[name]['turns']
        stage = len(inputs)-1
        if inputs != [r['prompt'] for r in rows[:len(inputs)]]:
            raise ValueError('Observed prompts differ from registered truth')
        if outputs != [r['output'] for r in rows[:len(outputs)]]:
            raise ValueError('Observed outputs differ from registered truth')
        start = stage+int(after_completion)
        turns = tuple(Turn(**r) for r in rows[start:])
        if not turns:
            raise ValueError('Finished Program must not be forecast as unfinished')
        # The first turn is represented by the actual running/pinned state.
        # A nonresident request cannot receive a free retained prefix.
        if not after_completion:
            t = turns[0]
            turns = (Turn(t.prompt, t.output, 0, t.gap),)+turns[1:]
        current = math.ceil((turns[0].prompt+turns[0].output)/page)
        peak = max(memory_pages, *(math.ceil((t.prompt+t.output)/page) for t in turns))
        return Program(name, 0., turns), dict(
            future_information=True, source_training_index=None, neighbors=0,
            stage_tail_fallback=False, peak_pages=peak, current_pages=current,
            predicted_turns=len(turns), predicted_output=turns[0].output)


def next_envelope(program, *, after_completion, page=16):
    """One concrete future call, never a whole-Program peak reservation."""
    index = 0 if after_completion else 1
    if index >= len(program.turns):
        return None
    t = program.turns[index]
    return math.ceil((t.prompt+t.output)/page)


def empirical_gittins(work):
    """Equal-weight service-budget index; no batching-optimality claim.

    Min E[min(R,b)]/P(R<=b). Between empirical support points the
    denominator is constant and the numerator nondecreasing, so only
    support endpoints matter. Tied endpoints are processed together.
    """
    xs=sorted(work)
    if not xs or xs[0]<0 or any(not math.isfinite(x) for x in xs):
        raise ValueError('Service samples must be finite, nonnegative and nonempty')
    total=0.;best=math.inf;n=len(xs)
    for i,x in enumerate(xs,1):
        total+=x
        if i<n and xs[i]==x:continue
        best=min(best,(total+(n-i)*x)/i)
    return best
