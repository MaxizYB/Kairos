"""Kairos cpu service lease implementation."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CPUServiceLease:
    keys: frozenset
    deadline: float
    bytes: int


class CPUServiceLedger:
    def __init__(self):
        self.leases = {}

    def expire(self, now):
        expired = [p for p, row in self.leases.items() if row.deadline <= now]
        for p in expired:
            del self.leases[p]
        return expired

    def revoke(self, program):
        return self.leases.pop(program, None)

    def protected(self, key, now):
        return any(row.deadline > now and key in row.keys for row in self.leases.values())

    def acquire(self, program, keys, ready, object_bytes, capacity, now, duration):
        if (not program or not keys or object_bytes <= 0 or capacity <= 0
                or not all(math.isfinite(x) for x in (now, duration))
                or not 0 < duration <= 5):
            raise ValueError('Invalid bounded CPU service commitment')
        self.expire(now)
        keys = frozenset(keys)
        previous=self.leases.get(program)
        if previous:
            # Repeated pressure can release another suffix of the same
            # program. Keep the earlier commitment until its finite expiry;
            # replacing it would silently expose the already-demoted KV.
            keys=keys|previous.keys
        if not keys <= ready:
            return dict(accepted=False, reason='incomplete-or-uncommitted-history')
        others = set().union(*(row.keys for p, row in self.leases.items()
                               if p != program)) if self.leases else set()
        # At least half the native finite pool stays outside this policy's
        # commitments. Shared keys are charged once, never allocated twice.
        union_bytes = len(others | keys)*object_bytes
        if union_bytes > capacity//2:
            return dict(accepted=False, reason='bounded-host-budget',
                        proposed_committed_bytes=union_bytes, budget_bytes=capacity//2)
        deadline=min(previous.deadline,now+duration) if previous else now+duration
        self.leases[program] = CPUServiceLease(keys, deadline, len(keys)*object_bytes)
        return dict(accepted=True, deadline=deadline, chunks=len(keys),
                    committed_bytes=union_bytes, budget_bytes=capacity//2)


def minimum_gpu_anchor(total_pages, required_pages, private_pages, cpu_chunks, chunks):
    """Least GPU prefix that joins a committed CPU suffix without a hole.

    This releases as much duplicated GPU tail as possible, in contrast to
    releasing the minimum number of pages and retaining a needlessly large
    GPU prefix. Last partial CPU chunks remain ordinary query work.
    """
    if min(total_pages, required_pages, private_pages, chunks) < 0 or not required_pages:
        return None
    present = set(cpu_chunks)
    missing = [i for i in range(chunks) if i not in present]
    start = max(missing, default=-1)+1
    retained = start*16
    release = total_pages-retained
    if start >= chunks or release < required_pages or release > private_pages:
        return None
    return dict(retained_pages=retained, release_pages=release,
                cpu_start_chunk=start, cpu_committed_chunks=chunks)
