"""Kairos recovery implementation."""
import heapq

from kairos._runtime.layout.plan import runs


class ChunkRecovery:
    def __init__(self, costs, programs, weights=None, cache=None):
        self.costs = costs
        self.programs = {p.name: p for p in programs}
        self.gpu = {p.name: set(p.gpu) for p in programs}
        self.cpu = {p.name: set(p.cpu) for p in programs}
        self.weights = weights or {p.name: p.probability for p in programs}
        self.cache = cache if cache is not None else {}

    def value(self, name, chunk, remove=(), cpu=None):
        pages = self.gpu[name]
        removed = set(remove)
        mask = sum(1 << j for j in range(16)
                   if chunk * 16 + j in pages and chunk * 16 + j not in removed)
        host = chunk in self.cpu[name] if cpu is None else cpu
        key = (chunk, mask, host)
        if key not in self.cache:
            missing = [chunk * 16 + j for j in range(16) if not mask & (1 << j)]
            segments = [(a * 16, b * 16) for a, b in runs(missing)]
            compute = self.costs.compute(segments)
            load = self.costs.load_token * 256 + self.costs.load_call
            self.cache[key] = min(compute, load) if missing and host else compute
        return self.cache[key]

    def gpu_loss(self, owners):
        changed = {}
        for name, page in owners:
            if name in self.programs:
                changed.setdefault((name, page // 16), set()).add(page)
        return sum(self.weights.get(name, 1.) * max(0.,
            self.value(name, chunk, remove=pages) - self.value(name, chunk))
            for (name, chunk), pages in changed.items())

    def remove_gpu(self, owners):
        for name, page in owners:
            self.gpu.get(name, set()).discard(page)

    def cpu_loss(self, name, chunk):
        if name not in self.programs:
            return 0.
        return self.weights.get(name, 1.) * max(0.,
            self.value(name, chunk, cpu=False) - self.value(name, chunk))


def choose_gpu_bundles(free, owners, recovery, need):
    """Choose exactly the allocator demand, with native tie order.

    Complete available chunk bundles are compared first. The last partial
    bundle is re-evaluated at its actual size. No referenced page enters
    this function: the caller owns the native legality checks.
    """
    selected = []
    groups = {}
    ranks = {}
    evaluations = 0
    for rank, page in enumerate(free):
        bid = page.block_id
        ranks[bid] = rank
        tracked = [(p, i) for p, i in owners.get(bid, ()) if p in recovery.programs]
        if not tracked:
            selected.append(page)
            if len(selected) == need:
                return selected, evaluations
        else:
            name, index = tracked[0]
            groups.setdefault((name, index // 16), []).append(page)
    heap = []
    for serial, group in enumerate(groups.values()):
        group.sort(key=lambda page: owners[page.block_id][0][1])
        loss = recovery.gpu_loss([owner for page in group for owner in owners[page.block_id]])
        evaluations += 1
        heapq.heappush(heap, (loss / len(group), min(ranks[p.block_id] for p in group),
                              serial, group))
    while len(selected) < need and heap:
        remaining = need - len(selected)
        # At a partial allocation boundary every candidate must be priced
        # at that exact size, rather than truncating a full-chunk score.
        if remaining < max(len(item[3]) for item in heap):
            options = []
            for _, rank, serial, group in heap:
                part = group[:remaining]
                loss = recovery.gpu_loss([o for page in part for o in owners[page.block_id]])
                evaluations += 1
                options.append((loss / len(part), rank, serial, part))
            _, _, serial, chosen = min(options, key=lambda item: item[:3])
            heap = [item for item in heap if item[2] != serial]
            heapq.heapify(heap)
        else:
            _, _, _, chosen = heapq.heappop(heap)
        selected.extend(chosen)
        recovery.remove_gpu([o for page in chosen for o in owners[page.block_id]])
        # Private serial program salts make chunks independent. If a physical
        # page has several owners, its deletion also changes their bundles.
        if any(len(owners[page.block_id]) > 1 for page in chosen):
            rebuilt = []
            for _, rank, serial, group in heap:
                loss = recovery.gpu_loss([o for page in group for o in owners[page.block_id]])
                evaluations += 1
                rebuilt.append((loss / len(group), rank, serial, group))
            heap = rebuilt
            heapq.heapify(heap)
    if len(selected) != need:
        raise RuntimeError('Recovery bundles changed native allocation demand')
    return selected, evaluations


def continuation_sensitivities(model, programs):
    """One-sided local delay slopes of the same continuation objective.

    Forward derivatives propagate an extra synchronous delay at first
    service to modeled completion events. This retains the schedule-derived
    meaning without one extra global rollout per program. Event-boundary
    branch changes remain outside this local approximation.
    """
    if len(programs) < 2:
        return {p.name: 1. for p in programs}, 0
    slopes = model.local_sensitivities(programs)
    weights = {p.name: slope if .99 <= slope <= len(programs) + .01 else 1.
               for p in programs for slope in (slopes[p.name],)}
    return weights, 1
