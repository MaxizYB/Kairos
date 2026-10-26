"""Kairos compact recovery implementation."""
from functools import lru_cache
import heapq

from kairos._runtime.layout.plan import query_segments


FULL = (1 << 16) - 1


@lru_cache(maxsize=65536)
def mask_intervals(mask):
    """Cache relative missing intervals once for each 16-bit layout."""
    answer = []
    for i in range(16):
        if not mask & (1 << i):
            if answer and answer[-1][1] == i * 16:
                answer[-1] = (answer[-1][0], (i + 1) * 16)
            else:
                answer.append((i * 16, (i + 1) * 16))
    return tuple(answer)


class MaskRecovery:
    def __init__(self, costs, masks, cpu, weights, cache=None, slots=2, lengths=None):
        self.costs = costs
        self.programs = masks
        self.masks = {p: list(row) for p, row in masks.items()}
        self.cpu = cpu
        self.weights = weights
        self.cache = cache if cache is not None else {}
        self.slots = slots
        self.lengths = lengths or {p: len(row) * 16 for p, row in masks.items()}
        self.plan_cache = {}
        self.plan_evaluations = 0

    def compute(self, chunk, mask):
        # Preserve the incumbent floating-point operation order, including
        # its native tie decisions; only interval construction is cached.
        offset = chunk * 256
        return self.costs.compute([(offset + a, offset + b)
                                   for a, b in mask_intervals(mask)])

    def value(self, name, chunk, remove=0):
        mask = self.masks[name][chunk] & ~remove
        host = chunk in self.cpu.get(name, ())
        key = (chunk, mask, host)
        if key not in self.cache:
            compute = self.compute(chunk, mask)
            load = self.costs.load_token * 256 + self.costs.load_call
            self.cache[key] = min(compute, load) if mask != FULL and host else compute
        return self.cache[key]

    def gpu_loss(self, owners):
        changed = {}
        for name, page in owners:
            if name in self.programs:
                key = (name, page // 16)
                changed[key] = changed.get(key, 0) | (1 << (page % 16))
        return sum(self.weights.get(name, 1.) * max(0.,
                   self.value(name, chunk, mask) - self.value(name, chunk))
                   for (name, chunk), mask in changed.items())

    def remove_gpu(self, owners):
        for name, page in owners:
            if name in self.masks:
                self.masks[name][page // 16] &= ~(1 << (page % 16))

    def _plan(self, name, masks):
        key = (name, tuple(masks))
        if key in self.plan_cache:
            return self.plan_cache[key]
        self.plan_evaluations += 1
        # Price retained complete history. Unknown appended tokens and future
        # runner contention are not predicted by this local recovery proxy.
        end = self.lengths[name] * 16
        count = end // 256
        gpu = {i * 16 + j for i, mask in enumerate(masks)
               for j in range(16) if mask & (1 << j)}
        begin = next((i * 16 for i in range(self.lengths[name]) if i not in gpu), end)
        cpu = self.cpu.get(name, ())
        frontier = begin
        native_loads = 0
        while frontier < end and frontier // 256 in cpu:
            native_loads += 1
            frontier = (frontier // 256 + 1) * 256
        load = self.costs.load_token * 256 + self.costs.load_call
        native = native_loads * load + self.costs.compute([(frontier, end)] if frontier < end else [])
        # Only two concrete load choices: the incumbent cost-selected chunks
        # and no loads. Both use the same bounded segment merger as serving.
        eligible = {i for i in cpu if i * 256 >= begin and i < count}
        _, loads = self.costs.estimate_pages(gpu, eligible, count)
        best = native
        for selected in (loads, set()):
            valid = gpu | {i * 16 + j for i in selected for j in range(16)}
            segments = query_segments(begin, end, valid, slots=self.slots) if begin < end else []
            best = min(best, len(selected) * load + self.costs.compute(segments))
        self.plan_cache[key] = best
        return best

    def plan_loss(self, owners):
        changed = {}
        for name, page in owners:
            if name in self.masks:
                changed.setdefault(name, list(self.masks[name]))[page // 16] &= ~(1 << (page % 16))
        return sum(self.weights.get(name, 1.) * max(0.,
                   self._plan(name, masks) - self._plan(name, self.masks[name]))
                   for name, masks in changed.items())


def choose_mask_bundles(free, owners, recovery, need, shortlist=0, executable=None):
    """Same native demand and chunk choices; optionally refine final choice.

    Scans use the maintained reverse index. A partial final bundle is priced
    at its actual size. Full-bundle scores remain valid for private histories;
    shared owners explicitly invalidate the remaining scores.
    """
    selected, groups, ranks = [], {}, {}
    for rank, page in enumerate(free):
        bid = page.block_id
        ranks[bid] = rank
        tracked = [o for o in owners.get(bid, ()) if o[0] in recovery.programs]
        if not tracked:
            selected.append(page)
            if len(selected) == need:
                return selected, 0
        else:
            name, position = tracked[0]
            groups.setdefault((name, position // 16), []).append(page)
    heap, evaluations = [], 0
    def pairs(group):
        return [o for page in group for o in owners.get(page.block_id, ())]
    for serial, group in enumerate(groups.values()):
        group.sort(key=lambda page: next(iter(owners[page.block_id]))[1])
        loss = recovery.gpu_loss(pairs(group))
        evaluations += 1
        heap.append((loss / len(group), min(ranks[p.block_id] for p in group), serial, group))
    heapq.heapify(heap)
    while len(selected) < need and heap:
        remaining = need - len(selected)
        final = remaining <= max(len(item[3]) for item in heap)
        if remaining < max(len(item[3]) for item in heap):
            options = []
            for _, rank, serial, group in heap:
                part = group[:remaining]
                evaluations += 1
                options.append((recovery.gpu_loss(pairs(part)) / len(part), rank, serial, part))
            answer = min(options, key=lambda row: row[:3])
        else:
            answer = heap[0]
            options = heap
        if executable is not None:
            # Compare only equal-capacity alternatives; execute precisely the
            # quoted partial bundle. Re-evaluate after each layout mutation.
            options = [row for row in options if len(row[3]) == len(answer[3])]
            candidates = heapq.nsmallest(3, options, key=lambda row: row[:3])
            answer = min(candidates, key=lambda row:
                (executable.gpu_loss(pairs(row[3])) / len(row[3]), row[:3]))
        if shortlist and final and len(answer[3]) == remaining:
            # Equal bytes are required: completing demand and a smaller
            # bundle are not comparable under a final-choice refinement.
            options = [row for row in options if len(row[3]) == remaining]
            candidates = heapq.nsmallest(shortlist, options, key=lambda row: row[:3])
            if len(candidates) > 1:
                answer = min(candidates, key=lambda row:
                             (recovery.plan_loss(pairs(row[3])), row[:3]))
        _, _, serial, chosen = answer
        selected.extend(chosen)
        recovery.remove_gpu(pairs(chosen))
        if final or executable is not None:
            heap = [item for item in heap if item[2] != serial]
            heapq.heapify(heap)
        else:
            heapq.heappop(heap)
        if any(len(owners.get(page.block_id, ())) > 1 for page in chosen):
            heap = [(recovery.gpu_loss(pairs(group)) / len(group), rank, seq, group)
                    for _, rank, seq, group in heap]
            evaluations += len(heap)
            heapq.heapify(heap)
    if len(selected) != need:
        raise RuntimeError('Mask bundles changed native allocation demand')
    return selected, evaluations
