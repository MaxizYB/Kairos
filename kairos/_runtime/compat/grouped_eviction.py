"""Kairos grouped eviction implementation."""
from __future__ import annotations

from kairos._runtime.compat.joint_eviction import choose_gpu_victims, restore_end


def restore_loss(layouts, victims):
    """Snapshot losses in 16-token pages, not a future latency guarantee."""
    victims = set(victims)
    joined = gpu = 0
    for row in layouts:
        if row.get("state") == "finished":
            continue
        ids = row["blocks"]
        before = next((i for i, bid in enumerate(ids) if bid is None), len(ids))
        after = next((i for i, bid in enumerate(ids)
                      if bid is None or bid in victims), len(ids))
        joined += (restore_end(before, row["cpu_chunks"], row["chunks"])
                   - restore_end(after, row["cpu_chunks"], row["chunks"]))
        gpu += before-after
    return joined, gpu


def choose_grouped_gpu_victims(free, layouts, count, *, max_evaluations=131072):
    """Optimize the whole allocation demand over private per-program pages.

    For each owner and each possible page count, taking the highest-position
    free pages minimizes its loss. A bounded DP then allocates the full demand
    across owners, minimizing joint loss and then GPU-prefix loss. Shared-page
    layouts, small demands, ties without strict restore improvement, and work
    above the comparison budget all preserve the legacy result exactly.
    """
    baseline = choose_gpu_victims(free, layouts, count)
    before_loss, before_gpu = restore_loss(layouts, baseline)
    facts = dict(baseline_restore_loss_tokens=before_loss*16,
                 selected_restore_loss_tokens=before_loss*16,
                 baseline_gpu_prefix_loss_tokens=before_gpu*16,
                 selected_gpu_prefix_loss_tokens=before_gpu*16,
                 accepted=False, evaluations=0, passes=0, reason="small-demand")
    if count < 16 or not before_loss:
        if not before_loss:
            facts["reason"] = "zero-restore-loss"
        return baseline, facts
    if len(set(free)) != len(free):
        raise ValueError("Duplicate physical free pages")
    owners = {}
    states = []
    for row in layouts:
        if row.get("state") == "finished":
            continue
        n = len(states)
        ids = row["blocks"]
        end = next((i for i, bid in enumerate(ids) if bid is None), len(ids))
        states.append((row, end))
        for i, bid in enumerate(ids):
            if bid is not None:
                owners.setdefault(bid, []).append((n, i))
    if any(len(owners.get(bid, ())) > 1 for bid in free):
        facts["reason"] = "shared-owner-fallback"
        return baseline, facts

    rank = {bid: i for i, bid in enumerate(free)}
    groups = [[] for _ in states]
    owner_group = {}
    unknown = []
    for bid in free:
        match = owners.get(bid)
        if match:
            n, position = match[0]
            groups[n].append((position, bid))
            owner_group[bid] = n
        else:
            unknown.append(bid)
            owner_group[bid] = len(states)
    groups.append([])
    groups[-1].extend((len(states), bid) for bid in unknown)

    baseline_counts = [0]*len(groups)
    for bid in baseline:
        baseline_counts[owner_group[bid]] += 1
    curves = []
    page_lists = []
    for (row, end), group in zip(states, groups[:-1], strict=True):
        group.sort(reverse=True)
        group = group[:count]
        page_lists.append([bid for _, bid in group])
        initial = restore_end(end, row["cpu_chunks"], row["chunks"])
        joined = [0]
        gpu = [0]
        for position, _ in group:
            after = min(end, position)
            joined.append(initial-restore_end(after, row["cpu_chunks"], row["chunks"]))
            gpu.append(end-after)
        curves.append((joined, gpu))
    unknown_rows = sorted(unknown, key=rank.__getitem__)[:count]
    page_lists.append(unknown_rows)
    curves.append(([0]*(len(unknown_rows)+1), [0]*(len(unknown_rows)+1)))

    # Each state is (joint loss, GPU-prefix loss, distance from legacy owner
    # counts, selected native ranks, allocation vector).
    states_dp = {0: (0, 0, 0, 0, ())}
    for group_index, ((joined, gpu), pages) in enumerate(zip(curves, page_lists, strict=True)):
        next_dp = {}
        max_take = min(count, len(pages))
        prefix_ranks = [0]
        for bid in pages:
            prefix_ranks.append(prefix_ranks[-1]+rank[bid])
        for used, prior in states_dp.items():
            for take in range(min(max_take, count-used)+1):
                if facts["evaluations"] >= max_evaluations:
                    facts["reason"] = "comparison-budget"
                    return baseline, facts
                facts["evaluations"] += 1
                candidate = (prior[0]+joined[take], prior[1]+gpu[take],
                             prior[2]+abs(take-baseline_counts[group_index]),
                             prior[3]+prefix_ranks[take],
                             prior[4]+(take,))
                total = used+take
                if total not in next_dp or candidate < next_dp[total]:
                    next_dp[total] = candidate
        states_dp = next_dp
    result = states_dp.get(count)
    if result is None:
        facts["reason"] = "infeasible"
        return baseline, facts
    if result[0] >= before_loss:
        facts["reason"] = "no-strict-improvement"
        return baseline, facts
    if result[1] > before_gpu:
        facts["reason"] = "transfer-tradeoff"
        return baseline, facts

    selected = sorted((bid for group, take in zip(page_lists, result[4], strict=True)
                       for bid in group[:take]), key=rank.__getitem__)
    selected_loss, selected_gpu = restore_loss(layouts, selected)
    if len(selected) != count or selected_loss != result[0] or selected_gpu != result[1]:
        raise RuntimeError("Invalid grouped capacity assignment")
    facts.update(accepted=True, reason="grouped-improvement", passes=1,
                 selected_restore_loss_tokens=selected_loss*16,
                 selected_gpu_prefix_loss_tokens=selected_gpu*16)
    return selected, facts
