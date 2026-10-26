"""Kairos retention implementation."""
from __future__ import annotations

import time
from kairos._runtime.compat.policy import evaluate_prefill
from kairos._runtime.compat.capacity_model import evaluate_capacity
from kairos._runtime.compat.event_model import evaluate as evaluate_residency


def candidate_plans(rows, deficit, initial, max_plans=16):
    plans = [dict(initial)]
    seen = {tuple(sorted(initial.items()))}

    def append(plan):
        plan = {key: value for key, value in plan.items() if value}
        key = tuple(sorted(plan.items()))
        if sum(plan.values()) == deficit and key not in seen and len(plans) < max_plans:
            seen.add(key)
            plans.append(plan)

    # Concave suffix restore curves favor concentration; also test a small
    # number of splits to cover prefix and chunk-boundary discontinuities.
    for row in rows:
        if row["limit"] >= deficit:
            append({row["id"]: deficit})
    for first in rows:
        for second in rows:
            if first is second:
                continue
            for count in sorted({min(first["limit"], deficit), deficit // 2,
                                 max(0, deficit-second["limit"])}):
                rest = deficit-count
                if 0 < count <= first["limit"] and 0 < rest <= second["limit"]:
                    append({first["id"]: count, second["id"]: rest})
            if len(plans) >= max_plans:
                return plans
    return plans


def scenario_requests(state, plan, scenario, block_size):
    requests = []
    for row in state:
        release = plan.get(row["id"], 0)
        kept = max(0, row["protected_blocks"]-release)
        arrival, added = row["scenarios"][scenario]
        prompt = row["prompt_tokens"] if row["queued"] else row["old_tokens"]+added
        hit = row["cached_tokens"]
        if release:
            hit = min(hit, kept*block_size)
        if not row["queued"] and arrival >= row["ttl_remaining"]:
            # No differential credit for protection that expires before return.
            hit = 0
        hit = min(hit, prompt-1)
        rank = 1 if kept and (row["queued"] or arrival < row["ttl_remaining"]) else 2
        if row.get("preempted"):
            rank = 0
        # A partially-computed active request has priority fixed at admission.
        if row.get("active_rank") is not None:
            rank = row["active_rank"]
        computed = max(hit, row.get("computed_tokens", 0))
        requests.append(dict(id=row["id"], arrival=arrival,
                             order=rank*1000000+row["order"],
                             intervals=[(computed+1, prompt)] if computed < prompt else []))
    return requests


def choose_retention(rows, deficit, initial, state, block_size, budget, beta,
                     budget_s=.002, epsilon=0.0, max_plans=16, objective="queue",
                     snapshot_s=0.0, extra_plans=(), reference=None):
    started = time.perf_counter()
    deadline = started+budget_s
    use_capacity = objective in ("capacity", "capacity-tail")
    candidates = [dict(initial)] if objective == "capacity-tail" else candidate_plans(rows, deficit, initial, max_plans)
    # Interleave amount and victim choices. Filling the bound with exact-D
    # plans first can accidentally exclude every concurrency-restoring budget.
    exact = candidates
    candidates = []
    seen = set()
    for index in range(max(len(exact), len(extra_plans))):
        for group in (exact, extra_plans):
            if index >= len(group) or len(candidates) >= max_plans:
                continue
            plan = group[index]
            key = tuple(sorted(plan.items()))
            if key not in seen:
                candidates.append(dict(plan))
                seen.add(key)
    evaluations = 0

    def cost(plan):
        total = 0.0
        for scenario in range(3):
            if use_capacity:
                snapshot = dict(capacity=state[0]["capacity"], block_size=block_size,
                                budget=budget, max_seqs=state[0]["max_seqs"], beta=beta,
                                decode_step_s=state[0]["decode_step_s"], rows=state,
                                decode_by_seqs=state[0].get("decode_by_seqs", {}))
                result = evaluate_capacity(snapshot, plan, scenario, deadline)
                if result is None:
                    return None
                total += result["J"]/3
            else:
                requests = scenario_requests(state, plan, scenario, block_size)
                groups = [requests] if objective == "queue" else [[request] for request in requests]
                for group in groups:
                    result = evaluate_prefill(group, budget, beta, deadline=deadline)
                    if result is None:
                        return None
                    total += result["J"]/3
        return total

    initial_cost = cost(initial if reference is None else reference)
    fallback = initial if reference is None else reference
    best, best_cost = fallback, initial_cost
    if initial_cost is not None:
        evaluations = 1
        for plan in candidates[1:] if reference is None else candidates:
            if time.perf_counter() >= deadline:
                break
            value = cost(plan)
            if value is None:
                if time.perf_counter() >= deadline:
                    break
                continue
            evaluations += 1
            if value < best_cost:
                best, best_cost = plan, value
    elapsed = time.perf_counter()-started
    # The sum-of-completion objective pays decision delay once per request.
    improvement = initial_cost-best_cost if initial_cost is not None else 0.0
    accepted = improvement > epsilon+len(state)*(elapsed+snapshot_s)
    if not accepted:
        best, best_cost = fallback, initial_cost
    return dict(best), dict(selection_mode="bounded-restore-queue-budget",
                            candidates=len(candidates), evaluations=evaluations,
                            initial_J=initial_cost, best_J=best_cost,
                            search_s=elapsed, search_accepted=accepted,
                            evaluation_complete=initial_cost is not None,
                            objective_requests=len(state),
                            objective=objective, snapshot_s=snapshot_s,
                            recovery_proxy="released-suffix-overwritten-capacity-completion-window"
                            if use_capacity else "released-suffix-overwritten-short-prefill-window")


def choose_residency(plans, snapshot, initial, reference=None, budget_s=.012,
                     snapshot_s=0.0, epsilon=0.0, scenarios=(0, 1, 2)):
    """Joint release amount/victim selection using actual cache residency.

    The fixed window ends after current requests and one return per idle
    program. It never assumes unpin is a miss and never recurses into search.
    A partially evaluated scenario/candidate is UNKNOWN, not a cheap plan.
    """
    start=time.perf_counter()
    deadline=start+budget_s
    snapshot={**snapshot,"objective":"completion"}
    fallback=initial if reference is None else reference
    values=[]
    # Scenario-independent queued states should not be replayed three times.
    # Preserve multiplicities so the empirical expectation is unchanged.
    known={r["program"] for r in snapshot["requests"]}
    grouped={}
    for scenario in scenarios:
        key=tuple(tuple(p["scenarios"][scenario]) for p in snapshot["pins"] if p["id"] not in known)
        key+=tuple(tuple(r["continuation"]["scenarios"][scenario])
                   for r in snapshot["requests"] if r.get("continuation"))
        if key not in grouped:
            grouped[key]=[scenario,0]
        grouped[key][1]+=1

    def cost(plan):
        total=0.0
        for scenario,multiplicity in grouped.values():
            result=evaluate_residency(snapshot,plan,scenario,deadline)
            if result is None:
                return None
            total+=multiplicity*result["J"]
        return total/len(scenarios)

    base=cost(fallback)
    best,best_cost=dict(fallback),base
    if base is not None:
        seen={tuple(sorted(fallback.items()))}
        for plan in plans:
            key=tuple(sorted(plan.items()))
            if key in seen:
                continue
            seen.add(key)
            value=cost(plan)
            if value is None:
                if time.perf_counter()>=deadline:
                    break
                continue
            values.append(dict(plan=plan,J=value))
            if value<best_cost:
                best,best_cost=dict(plan),value
    elapsed=time.perf_counter()-start
    count=len(snapshot["requests"])+sum(
        p["id"] not in {r["program"] for r in snapshot["requests"]}
        for p in snapshot["pins"])
    accepted=base is not None and base-best_cost>epsilon+count*(elapsed+snapshot_s)
    if not accepted:
        best,best_cost=dict(fallback),base
    return best,dict(selection_mode="residency-completion-budget",initial_J=base,best_J=best_cost,
        evaluations=len(values)+(base is not None),evaluation_complete=base is not None,
        search_accepted=accepted,search_s=elapsed,snapshot_s=snapshot_s,
        objective_requests=count,objective="completion",scenarios=list(scenarios),
        distinct_scenarios=len(grouped),candidate_evaluations=len(values),
        budget_exhausted=time.perf_counter()>=deadline,reference_plan=dict(fallback),
        candidate_costs=values,recovery_proxy="physical-residency-event-model-native-prefix")
