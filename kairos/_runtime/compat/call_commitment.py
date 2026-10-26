"""Kairos call commitment implementation."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Job:
    name: str
    current: float
    pairs: tuple  # (remaining solo service, next return interval, probability)
    idle: bool = False
    elapsed: float = 0.
    debt: float = 0.

    def conditional(self):
        rows = [(r, d-self.elapsed, p) for r, d, p in self.pairs
                if not self.idle or d > self.elapsed]
        total = math.fsum(p for _, _, p in rows)
        if total <= 0:
            raise ValueError('Return distribution has no conditional support')
        return tuple((r, max(0., d), p/total) for r, d, p in rows)


def sept(work):
    """Exact expected sum completion times for all-ready fixed SEPT order."""
    values = sorted(work)
    return math.fsum((len(values)-i)*w for i, w in enumerate(values))


def select_atom(rows, u):
    cumulative = 0.
    for r, d, p in rows:
        cumulative += p
        if cumulative >= u:
            return r, d
    return rows[-1][:2]


def clear(jobs, releases, actual, means, start=0., rate=1., trace=None):
    """Nonanticipative nonpreemptive SEPT with external release dates.

    `actual` is used only AFTER choosing a job. It must never enter the key.
    Conditional R means are updated from the observed return interval.
    """
    todo = set(jobs); t = start; total = 0.
    while todo:
        ready = [p for p in todo if releases[p] <= t]
        if not ready:
            t = min(releases[p] for p in todo)
            ready = [p for p in todo if releases[p] <= t]
        p = min(ready, key=lambda p: (means[p], p))
        if trace is not None:
            trace.append(p)
        t += actual[p]/rate
        total += t
        todo.remove(p)
    return total


def evaluate(jobs, committed, candidate=None, victims=(), *, wait=0.,
             rate=1., slowdown=1., scenarios=8):
    """Common-random-number expectation of complete commitment + clearing.

    Commitments run concurrently at the independently profiled batch slowdown.
    They cannot be interrupted by sampled returns. At their last completion,
    noncommitted current calls and all aggregate suffixes are cleared using
    observable posterior means, never the sampled unknown remaining work.
    """
    by = {j.name: j for j in jobs}
    forced = set(committed) | ({candidate} if candidate is not None else set())
    if any(by[p].idle for p in forced):
        raise ValueError('An idle Program cannot receive a current-call grant')
    victims = dict(victims)
    conditionals = {j.name: j.conditional() for j in jobs}
    result = 0.
    for k in range(scenarios):
        releases = {}; actual = {}; means = {}; completed = 0.
        ends = {p: (wait if p == candidate else 0.) +
                (by[p].current+by[p].debt)*slowdown for p in forced}
        end = max(ends.values(), default=wait)
        for index, j in enumerate(jobs):
            rows = conditionals[j.name]
            # Different fixed rotations preserve each marginal; the same
            # scenarios are reused by every action in this decision.
            u = ((k+.5)/scenarios + index*.6180339887498949) % 1.
            r, d = select_atom(rows, u)
            mean = math.fsum(x*p for x, _, p in rows)
            debt = j.debt+victims.get(j.name, 0.)
            if j.name in forced:
                finish = ends[j.name]
                if r == 0:
                    completed += finish
                    continue
                release = finish+d
                # D is observable on return. R itself is still unobserved.
                matched = [(x, p) for x, dd, p in rows if dd == d]
                conditional_mean = (math.fsum(x*p for x, p in matched)/
                                    math.fsum(p for _, p in matched))
                actual[j.name] = r
                means[j.name] = conditional_mean
            elif j.idle:
                release = d
                actual[j.name] = r+debt
                matched = [(x, p) for x, dd, p in rows if dd == d]
                means[j.name] = (math.fsum(x*p for x, p in matched)/
                                math.fsum(p for _, p in matched))+debt
            else:
                release = 0.
                actual[j.name] = j.current+r+debt
                means[j.name] = j.current+mean+debt
            releases[j.name] = release
        result += completed+clear(tuple(actual), releases, actual, means, end, rate)
    return result/scenarios


def wait_two_job(rows, si, sj, recovery):
    """Exact finite-support Eq.14, including zero and event-wait endpoint."""
    candidates = sorted({0., *(d for d, _ in rows)})
    def cost(tau):
        return math.fsum(p*(2*d+2*si+sj if d <= tau else
                           tau+sj+max(d, tau+sj)+si+recovery) for d, p in rows)
    return min(((cost(t), t) for t in candidates))


def freeable(owner_blocks, native_refs, revoked):
    """Shared physical union, subtract exactly the revoked optional refs."""
    removed = {}
    for p in revoked:
        for b in owner_blocks[p]:
            removed[b] = removed.get(b, 0)+1
    return {b for b, count in removed.items() if native_refs[b] == count}
