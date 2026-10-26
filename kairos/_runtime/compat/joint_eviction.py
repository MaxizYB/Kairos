"""Kairos joint eviction implementation."""
from __future__ import annotations

import heapq
import math


def restore_end(gpu_end, cpu_chunks, chunks):
    """Token boundary reachable by one GPU prefix + one CPU interval."""
    chunk = gpu_end // 16
    while chunk < chunks and chunk in cpu_chunks:
        chunk += 1
    return max(gpu_end, chunk * 16)


def native_zero_restore_loss(victims,layouts):
    """Certify the whole native victim set has zero joint prefix loss.

    With LRU ties this is precisely the existing greedy result. Check the
    collective frontier, including all shared owners; a per-page snapshot
    certificate would be insufficient. No future host survival is assumed.
    """
    victims=set(victims)
    for layout in layouts:
        if layout.get('state')=='finished':continue
        ids=layout['blocks']
        old=next((i for i,bid in enumerate(ids) if bid is None),len(ids))
        new=next((i for i,bid in enumerate(ids[:old]) if bid in victims),old)
        if new==old:continue
        cpu=set(layout['cpu_chunks'])
        if restore_end(new,cpu,layout['chunks'])<restore_end(old,cpu,layout['chunks']):
            return False
    return True


def choose_gpu_victims(free, layouts, count, mode="joint", prefer_gpu=False, weighted=False, costed=False):
    """Return exactly count pages; legacy mode breaks equal costs by LRU.

    Each layout contains ordered physical GPU IDs (None means absent), a
    committed CPU chunk set, and current program phase. Shared GPU pages are
    evaluated against every owner. Only supplied free IDs may be selected.
    Full-attention, 16-token GPU / 256-token CPU chunks are required.
    prefer_gpu uses GPU-prefix loss as a secondary objective: equal amounts
    of avoided recompute need not imply equal CPU-to-GPU restoration traffic.
    The single-owner fast path then uses descending position within a tied
    owner, and native rank between owners; it does not promise global LRU
    for already-unreachable pages when prefer_gpu is enabled.
    """
    if count < 0 or count > len(free):
        raise ValueError("Invalid free-page demand")
    if mode not in ("joint", "duplicate", "tail"):
        raise ValueError("Invalid GPU victim policy")
    if costed and mode != "joint":
        raise ValueError("Recovery cost requires the joint policy")
    if (mode=='joint' and not prefer_gpu and not weighted and not costed
            and native_zero_restore_loss(free[:count],layouts)):
        # Every prefix of this native set also has zero loss by monotonicity.
        # Greedy's LRU tie thus chooses it one page at a time. Avoid building
        # thousands of heap entries for an already optimal native choice.
        return list(free[:count])
    rank = {bid: i for i, bid in enumerate(free)}
    available = set(free)
    owners = {}
    states = []
    for layout in layouts:
        weight = float(layout.get("recovery_weight", 1.0)) if weighted else 1.0
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Recovery weight must be finite and positive")
        ids = list(layout["blocks"])
        end = next((i for i, bid in enumerate(ids) if bid is None), len(ids))
        state = dict(blocks=ids, gpu_end=end,
                     cpu=set() if mode == "tail" else set(layout["cpu_chunks"]), chunks=layout["chunks"],
                     finished=layout.get("state") == "finished", weight=weight)
        # CPU reachability only changes at a chunk boundary. Precompute it
        # once rather than scan the host interval for every candidate page.
        ends = [0] * (state["chunks"]+1)
        ends[-1] = state["chunks"] * 16
        for i in range(state["chunks"]-1, -1, -1):
            ends[i] = ends[i+1] if i in state["cpu"] else i*16
        state["cpu_ends"] = ends
        if costed:
            linear = float(layout.get("prefill_linear", 0))
            attention = float(layout.get("prefill_attention", 0))
            reload = float(layout.get("reload_per_token", 0))
            if any(not math.isfinite(x) or x < 0 for x in (linear, attention, reload)):
                raise ValueError("Recovery coefficients must be finite and nonnegative")
            # Marginal avoided recompute PLUS avoided H2D. With nonnegative
            # attention this saved-value curve is monotone, so the same
            # single-owner heap remains valid. Never apply it to legacy v19.
            benefits = []
            for gpu_end in range(len(ids)+1):
                reachable = max(gpu_end, ends[min(gpu_end//16, state["chunks"])])*16
                saved = max(0.0, (linear-reload)*reachable
                            + attention*reachable*(reachable+1)/2)
                benefits.append(saved+reload*gpu_end*16)
            state["benefits"] = benefits
        states.append(state)
        for i, bid in enumerate(ids):
            if bid is not None:
                owners.setdefault(bid, []).append((state, i))
    # A page with one live owner has loss max(0, R(frontier)-R(index)).
    # R(index) is immutable and nondecreasing. Keep a heap of zero-loss
    # pages in native order (or descending position), and positive pages by descending
    # R(index). As the frontier shrinks, pages only move positive -> zero.
    # This preserves the legacy greedy rule without rescanning every page
    # for each allocation. Shared pages use the general multi-owner rule.
    if all(len(rows) <= 1 for rows in owners.values()):
        heaps = []
        unknown = []
        for bid in free:
            rows = owners.get(bid, ())
            if not rows or rows[0][0]["finished"]:
                heapq.heappush(unknown, (rank[bid], bid))
        for state in states:
            if state["finished"]:
                continue
            positive, zero = [], []
            def reach(end):
                return (state["benefits"][end] if costed else
                        max(end, state["cpu_ends"][min(end//16, state["chunks"])]))
            boundary = reach(state["gpu_end"])
            for index, bid in enumerate(state["blocks"]):
                if bid not in available:
                    continue
                if mode == "duplicate":
                    value = int(index//16 not in state["cpu"])
                    entry = (value, 0, rank[bid], bid, index)
                    positive.append(entry)
                else:
                    end = reach(index)
                    if end >= boundary:
                        zero.append((-index if prefer_gpu else 0, rank[bid], bid, index))
                    else:
                        positive.append((-end, -index if prefer_gpu else 0, rank[bid], bid, index))
            heapq.heapify(positive)
            heapq.heapify(zero)
            heaps.append((state, positive, zero))
        selected = []
        while len(selected) < count:
            candidates = []
            if unknown:
                candidates.append((0, 0, unknown[0][0], unknown[0][1], -1))
            for n, (state, positive, zero) in enumerate(heaps):
                if mode == "duplicate":
                    if positive:
                        value, _, order, bid, _ = positive[0]
                        candidates.append((value, 0, order, bid, n))
                    continue
                end = state["gpu_end"]
                boundary = (state["benefits"][end] if costed else
                            max(end, state["cpu_ends"][min(end//16, state["chunks"])]))
                while positive and -positive[0][0] >= boundary:
                    _, position, order, bid, index = heapq.heappop(positive)
                    heapq.heappush(zero, (position, order, bid, index))
                if zero:
                    _, order, bid, index = zero[0]
                    gpu_loss = max(0, end-index) if prefer_gpu else 0
                    candidates.append((0, gpu_loss, order, bid, n))
                elif positive:
                    neg_end, _, order, bid, index = positive[0]
                    gpu_loss = max(0, end-index) if prefer_gpu else 0
                    candidates.append(((boundary+neg_end)*state["weight"], gpu_loss, order, bid, n))
            _, _, _, bid, owner = min(candidates)
            selected.append(bid)
            if owner < 0:
                heapq.heappop(unknown)
            else:
                state, positive, zero = heaps[owner]
                if mode == "duplicate" or not zero:
                    *_, index = heapq.heappop(positive)
                else:
                    _, _, _, index = heapq.heappop(zero)
                state["gpu_end"] = min(state["gpu_end"], index)
        return selected

    selected = []
    while len(selected) < count:
        def loss(bid):
            value = gpu_loss = 0
            for state, index in owners.get(bid, ()):
                if state["finished"]:
                    continue
                if mode == "duplicate":
                    value += index // 16 not in state["cpu"]
                else:
                    def reach(end):
                        return (state["benefits"][end] if costed else
                                max(end, state["cpu_ends"][min(end//16, state["chunks"])]))
                    before = reach(state["gpu_end"])
                    after = reach(min(index, state["gpu_end"]))
                    value += (before-after)*state["weight"]
                    gpu_loss += max(0, state["gpu_end"]-index)
            return value, gpu_loss if prefer_gpu else 0
        # Untracked warmup/finished-output pages have no known future program
        # restoration value; they stay in their original queue order.
        chosen = min(available, key=lambda bid: (loss(bid), rank[bid]))
        selected.append(chosen)
        available.remove(chosen)
        for state, index in owners.get(chosen, ()):
            state["gpu_end"] = min(state["gpu_end"], index)
    return selected
