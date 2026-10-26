"""Kairos executable recovery implementation."""
from collections import OrderedDict
from dataclasses import dataclass
import math

from kairos._runtime.layout.compact_recovery import mask_intervals


def page_masks(pages, end):
    masks = [0] * ((end + 255) // 256)
    for page in pages:
        if 0 <= page * 16 < end:
            masks[page // 16] |= 1 << (page % 16)
    return tuple(masks)


def mask_segments(begin, end, masks, loads=(), slots=2):
    """Same missing intervals, final-page anchor and merge ties as serving."""
    if not 0 <= begin < end or slots < 1:
        raise ValueError('Invalid executable recovery interval')
    loads = set(loads)
    anchor = max(begin, end - 16)
    segments = []
    def append(a, b):
        a, b = max(begin, a), min(end, b)
        if a >= b:
            return
        if segments and segments[-1][1] >= a:
            segments[-1] = (segments[-1][0], max(b, segments[-1][1]))
        else:
            segments.append((a, b))
    for chunk in range(begin // 256, (end + 255) // 256):
        mask = 65535 if chunk in loads else (masks[chunk] if chunk < len(masks) else 0)
        intervals = [(chunk * 256 + a, chunk * 256 + b) for a, b in mask_intervals(mask)]
        # All pages, including hits, recompute queries in the progress anchor.
        intervals.append((max(chunk * 256, anchor), (chunk + 1) * 256))
        for a, b in sorted(intervals):
            append(a, b)
    if len(segments) > slots:
        joined = set(sorted(range(len(segments) - 1), key=lambda i:
            (segments[i + 1][0] - segments[i][1], i))[:len(segments) - slots])
        merged = [segments[0]]
        for i, segment in enumerate(segments[1:]):
            if i in joined:
                merged[-1] = (merged[-1][0], segment[1])
            else:
                merged.append(segment)
        segments = merged
    return tuple(segments)


def selected_loads(costs, masks, eligible, count):
    """The existing two-state whole-chunk DP, using cached mask intervals."""
    states = {False: (0., ())}
    for chunk in range(count):
        mask = masks[chunk] if chunk < len(masks) else 0
        offset = chunk * 256
        segments = [(offset + a, offset + b) for a, b in mask_intervals(mask)]
        compute = costs.compute(segments)
        next_states = {}
        for previous, (value, choices) in states.items():
            delta = compute - (costs.segment if previous and segments
                               and segments[0][0] == offset else 0.)
            tail = bool(segments and segments[-1][1] == offset + 256)
            candidate = (value + delta, choices + (False,))
            if tail not in next_states or candidate[0] < next_states[tail][0]:
                next_states[tail] = candidate
            if mask != 65535 and chunk in eligible:
                candidate = (value + costs.load_token * 256 + costs.load_call,
                             choices + (True,))
                if False not in next_states or candidate[0] < next_states[False][0]:
                    next_states[False] = candidate
        states = next_states
    _, actions = min(states.values())
    return frozenset(i for i, load in enumerate(actions) if load)


@dataclass(frozen=True)
class RecoverySummary:
    route: str
    loads: frozenset
    segments: tuple
    prefix_frontier: int
    prefix_queries: int
    gap_queries: int
    gap_steps: int
    prefix_s: float
    gap_s: float
    dependency_s: float

    @property
    def cost(self):
        return self.prefix_s if self.route == 'prefix' else self.gap_s

    def diagnostic(self):
        return dict(route=self.route, native_queries=self.prefix_queries,
            segmented_queries=self.gap_queries, native_frontier=self.prefix_frontier,
            native_recovery_s=self.prefix_s, segmented_recovery_s=self.gap_s,
            dependency_s=self.dependency_s, planned_load_chunks=sorted(self.loads),
            planned_segments=self.segments, planned_segment_steps=self.gap_steps,
            cost_rejected=self.gap_queries < self.prefix_queries and self.route == 'prefix',
            executable_recovery=True)


class RecoveryPlans:
    """Bounded cache keyed by actual layout, never by an unlocked CPU epoch.

    Callers still take fresh CPU snapshots. Content keys invalidate quotes
    when either tier changes, including CPU eviction between allocations.
    """
    def __init__(self, costs, dependency_step=0., slots=2, query_budget=2048,
                 prefix_load_token=None, prefix_load_call=None, capacity=1024):
        self.costs, self.dependency_step = costs, dependency_step
        self.slots, self.query_budget = slots, query_budget
        self.prefix_load_token = costs.load_token if prefix_load_token is None else prefix_load_token
        self.prefix_load_call = costs.load_call if prefix_load_call is None else prefix_load_call
        self.capacity = capacity
        self.cache = OrderedDict()
        self.evaluations = self.hits = 0

    def get(self, masks, cpu, end, begin=None):
        if begin is not None and begin >= end:
            # A preempted decode can rematch generated full pages beyond the
            # prompt. It has no prompt-query recovery for this route gate.
            return RecoverySummary('prefix', frozenset(), (), begin, 0, 0, 0, 0., 0., 0.)
        masks, cpu = tuple(masks), frozenset(i for i in cpu if 0 <= i < (end - 1) // 256)
        if begin is None:
            begin = 0
            for mask in masks:
                if mask != 65535:
                    begin += ((~mask) & 65535 & -((~mask) & 65535)).bit_length() * 16 - 16
                    break
                begin += 256
            begin = min(begin, end - 1)
        key = (masks, cpu, end, begin)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.evaluations += 1
        frontier, native_loads = begin, 0
        while frontier < end - 1 and frontier // 256 in cpu:
            native_loads += 1
            frontier = min(end - 1, (frontier // 256 + 1) * 256)
        native_queries = end - frontier
        native_load_s = ((frontier - begin) * self.prefix_load_token + self.prefix_load_call
                         if native_loads else 0.)
        native = native_load_s + self.costs.compute([(frontier, end)])
        eligible = {i for i in cpu if i * 256 >= begin}
        loads = selected_loads(self.costs, masks, eligible, (end - 1) // 256)
        options = []
        for selected in dict.fromkeys((loads, frozenset())):
            segments = mask_segments(begin, end, masks, selected, self.slots)
            queries = sum(b - a for a, b in segments)
            steps = max(1, math.ceil(queries / self.query_budget))
            dependency = steps * self.dependency_step
            value = (len(selected) * (self.costs.load_token * 256 + self.costs.load_call)
                     + self.costs.compute(segments) + dependency)
            options.append((value, len(selected), selected, segments, queries, steps, dependency))
        gap, _, loads, segments, queries, steps, dependency = min(options, key=lambda x: x[:2])
        route = 'segmented' if queries < native_queries and gap < native else 'prefix'
        result = RecoverySummary(route, loads, segments, frontier, native_queries,
                                 queries, steps, native, gap, dependency)
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


class ExecutableLoss:
    def __init__(self, local, plans, lengths):
        self.local, self.plans, self.lengths = local, plans, lengths

    def gpu_loss(self, owners):
        changed = {}
        for name, page in owners:
            if name in self.local.masks:
                changed.setdefault(name, list(self.local.masks[name]))[page // 16] &= ~(1 << (page % 16))
        loss = 0.
        for name, masks in changed.items():
            end = self.lengths[name] * 16 + 1
            cpu = self.local.cpu.get(name, ())
            before = self.plans.get(self.local.masks[name], cpu, end)
            after = self.plans.get(masks, cpu, end)
            loss += self.local.weights.get(name, 1.) * max(0., after.cost - before.cost)
        return loss
