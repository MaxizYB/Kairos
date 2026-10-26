"""Kairos holding economics implementation."""
from bisect import bisect_right
import time

from kairos._runtime.compat.policy import TTLPolicy


class ToolHoldPolicy(TTLPolicy):
    def __init__(self, *args, replacement=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.replacement = replacement
        self.last_decision = None
        self.serial = 0

    def ttl(self, tool, benefit, oracle_gap=None):
        started = time.perf_counter()
        paper, source = super().ttl(tool, benefit, oracle_gap)
        self.serial += 1
        record = dict(serial=self.serial, paper_ttl=paper, hold_ttl=paper,
                      source=source, benefit=benefit, changed=False,
                      model='paper fallback outside mature empirical mode')
        if self.mode == 'paper' and source in ('per-tool-ecdf', 'global-ecdf'):
            rows = (self.history[tool] if source == 'per-tool-ecdf' else
                    [r for values in self.history.values() for r in values])
            xs = sorted(r[0] for r in rows)
            sums = [0.0]
            for value in xs:
                sums.append(sums[-1] + value)

            def terms(tau):
                count = bisect_right(xs, tau)
                return count/len(xs), (sums[count]+(len(xs)-count)*tau)/len(xs)

            selected, best = 0.0, 0.0
            for candidate in sorted(set([0.0]+xs)):
                probability, hold = terms(candidate)
                utility = probability*benefit-hold
                if utility > best:
                    selected, best = candidate, utility
            record.update(hold_ttl=selected, changed=abs(selected-paper)>1e-9,
                paper_expected_tool_hold=terms(paper)[1],
                selected_expected_tool_hold=terms(selected)[1],
                candidate_max=max(xs), samples=len(xs),
                model='F(tau)*benefit - mean(min(observed_tool_gap,tau))')
        record['economic_evaluation_s'] = time.perf_counter()-started
        self.last_decision = record
        if self.replacement and record['model'].startswith('F('):
            return record['hold_ttl'], 'expected-tool-hold/'+source
        return paper, source


class PriorityCertificates:
    """Original TTL eligibility with the native queued-return extension."""
    def __init__(self):
        self.deadlines = {}

    def grant(self, program, created, ttl):
        self.deadlines.pop(program, None)
        if ttl > 0:
            self.deadlines[program] = created+ttl

    def expire(self, now, queued):
        removed = []
        for program, deadline in list(self.deadlines.items()):
            if now >= deadline and program not in queued:
                self.deadlines.pop(program)
                removed.append(program)
        return removed

    def release(self, program):
        self.deadlines.pop(program, None)

    def has(self, program):
        return program in self.deadlines
