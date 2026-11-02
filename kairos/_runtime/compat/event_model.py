"""runtime.compat.event model."""

from __future__ import annotations
import math
import time
from collections import OrderedDict


def evaluate(snapshot, selection, scenario=0, deadline=None):
    state = snapshot
    block_size, budget = (state["block_size"], state["budget"])
    beta = state["beta"]
    refs = state["refs"].copy()
    valid = set(state["valid"])
    content = {bid: tuple(key) for bid, key in state.get("content", {}).items()}
    free = OrderedDict.fromkeys(state["free_queue"])
    pins = {p["id"]: {**p, "blocks": list(p["blocks"])} for p in state["pins"]}
    rows = [
        {**r, "allocated": list(r["allocated"]), "new_blocks": []}
        for r in state["requests"]
    ]
    for p in pins.values():
        if p["id"] not in {r["program"] for r in rows}:
            gap, added = p["scenarios"][scenario]
            rows.append(
                dict(
                    id="future:" + p["id"],
                    program=p["id"],
                    order=p["order"],
                    arrival=gap,
                    prompt=p["old_tokens"] + max(1, added),
                    computed=0,
                    remaining=state.get("future_output", 24),
                    allocated=[],
                    lookup=p["lookup"],
                    running=False,
                    preempted=False,
                    finish=None,
                )
            )
    for r in list(rows):
        continuation = r.get("continuation")
        if continuation:
            gap, added = continuation["scenarios"][scenario]
            successor = dict(
                id="next:" + r["id"],
                program=r["program"],
                order=r["order"],
                arrival=math.inf,
                prompt=r["prompt"] + max(1, added),
                computed=0,
                remaining=state.get("future_output", 24),
                allocated=[],
                lookup=[],
                running=False,
                preempted=False,
                finish=None,
                objective_start=r["arrival"],
                predicted_gap=max(0, gap),
            )
            r["successor"] = successor
            rows.append(successor)
    for r in rows:
        r.setdefault("finish", None)
        r.setdefault("preempted", False)
        r["ended"] = False
        r["sequence"] = r.get(
            "sequence",
            r["prompt"]
            + max(0, r.get("output_budget", r["remaining"]) - r["remaining"]),
        )
        r["objective"] = r["computed"] < r["prompt"]
        if not r["objective"]:
            r["finish"] = 0.0

    def release(blocks):
        for bid in blocks:
            refs[bid] -= 1
            if refs[bid] < 0:
                raise ValueError("negative simulated block reference")
            if refs[bid] == 0:
                free[bid] = None

    def unpin(p, count=None):
        n = len(p["blocks"]) if count is None else count
        if n:
            release(reversed(p["blocks"][-n:]))
            p["blocks"] = p["blocks"][:-n]

    for program, n in selection.items():
        if n < 0 or n > len(pins[program]["blocks"]):
            return None
        unpin(pins[program], n)
    for r in rows:
        for index, bid in enumerate(r["allocated"][: r["computed"] // block_size]):
            content[bid] = (r["program"], index)
            valid.add(bid)

    def touch(blocks):
        for bid in blocks:
            if refs[bid] == 0:
                del free[bid]
            refs[bid] += 1

    def allocate(r, target):
        n = max(0, math.ceil(target / block_size) - len(r["allocated"]))
        if n > len(free):
            return False
        for _ in range(n):
            bid, _ = free.popitem(last=False)
            valid.discard(bid)
            content[bid] = (r["program"], len(r["allocated"]))
            refs[bid] += 1
            r["allocated"].append(bid)
            r.setdefault("new_blocks", []).append(bid)
        return True

    def cached_prefix(r):
        result = []
        for index, bid in enumerate(r["lookup"][: (r["sequence"] - 1) // block_size]):
            if (
                bid is None
                or bid not in valid
                or content.get(bid) != (r["program"], index)
            ):
                break
            result.append(bid)
        return result

    def rank(r):
        pinned = (
            r.get("was_pinned", False)
            if r["running"]
            else bool(pins.get(r["program"], {}).get("blocks"))
        )
        return (0 if r["preempted"] else 1 if pinned else 2, r["order"])

    completion_objective = state.get("objective") == "completion"
    for r in rows:
        r["completion"] = None
    t = 0.0
    batches = []
    preemptions = 0
    for iteration in range(state.get("max_steps", 4096)):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        if (
            all((r["ended"] for r in rows))
            if completion_objective
            else all((r["finish"] is not None for r in rows))
        ):
            finishes = {
                r["id"]: r["completion"] if completion_objective else r["finish"]
                for r in rows
            }
            completion_J = (
                sum(
                    (
                        r["completion"] - r.get("objective_start", max(0, r["arrival"]))
                        for r in rows
                        if "successor" not in r
                    )
                )
                if completion_objective
                else None
            )
            return dict(
                J=completion_J
                if completion_objective
                else sum((r["finish"] - r["arrival"] for r in rows if r["objective"])),
                completion_J=completion_J,
                F=finishes,
                batches=batches,
                missing_tokens=sum((r.get("missing", 0) for r in rows)),
                preemptions=preemptions,
            )
        queued = {
            r["program"]
            for r in rows
            if not r["ended"] and (not r["running"]) and (r["arrival"] <= t)
        }
        for p in pins.values():
            if p["deadline"] <= t and p["id"] not in queued:
                unpin(p)
        running = sorted(
            [r for r in rows if r["running"] and (not r["ended"])], key=rank
        )
        waiting = sorted(
            [
                r
                for r in rows
                if not r["running"] and (not r["ended"]) and (r["arrival"] <= t)
            ],
            key=rank,
        )
        left = budget
        batch = []
        stalled = None
        preempted_this_step = False
        for r in running:
            if not r["running"] or not left:
                continue
            q = min(left, max(1, r["sequence"] - r["computed"]))
            while not allocate(r, r["computed"] + q):
                active = [x for x in running if x["running"]]
                if len(active) == 1:
                    victims = sorted(
                        [p for p in pins.values() if p["blocks"]],
                        key=lambda p: p["order"],
                        reverse=True,
                    )
                    if victims:
                        unpin(victims[0])
                        continue
                victim = max(active, key=rank)
                retained = []
                for entry in batch:
                    if entry[0] is victim:
                        left += entry[1]
                    else:
                        retained.append(entry)
                batch = retained
                release(reversed(victim["allocated"]))
                victim["allocated"] = []
                victim["computed"] = 0
                victim["running"] = False
                victim["preempted"] = True
                preemptions += 1
                preempted_this_step = True
                if victim is r:
                    stalled = r
                    break
            if stalled is not None:
                break
            batch.append((r, q, r["computed"] + 1))
            left -= q
            if left == 0:
                break
        if stalled is None and (not preempted_this_step):
            for r in waiting:
                if (
                    not left
                    or sum((x["running"] for x in running)) >= state["max_seqs"]
                ):
                    break
                hits = cached_prefix(r)
                need = math.ceil(r["sequence"] / block_size) - len(hits)
                reclaimable_hits = sum((refs[b] == 0 for b in hits))
                if need > len(free) - reclaimable_hits:
                    stalled = r
                    break
                touch(hits)
                r["allocated"] = hits
                r["computed"] = len(hits) * block_size
                r["missing"] = max(
                    0, min(len(r["lookup"]) * block_size, r["prompt"]) - r["computed"]
                )
                q = min(left, r["sequence"] - r["computed"])
                if not allocate(r, r["computed"] + q):
                    release(reversed(hits))
                    r["allocated"] = []
                    stalled = r
                    break
                if r["program"] in pins:
                    r["was_pinned"] = bool(pins[r["program"]]["blocks"])
                    unpin(pins[r["program"]])
                r["running"] = True
                running.append(r)
                batch.append((r, q, r["computed"] + 1))
                left -= q
        if not batch:
            if not running and (not waiting):
                events = [
                    r["arrival"]
                    for r in rows
                    if not r["ended"] and t < r["arrival"] < math.inf
                ]
                events += [
                    p["deadline"]
                    for p in pins.values()
                    if p["blocks"] and p["deadline"] > t
                ]
                if not events:
                    return None
                t = min(events)
                continue
            victims = sorted(
                [p for p in pins.values() if p["blocks"]],
                key=lambda p: p["order"],
                reverse=True,
            )
            if victims:
                unpin(victims[0])
                continue
            if running and stalled:
                victim = max(running, key=rank)
                release(reversed(victim["allocated"]))
                victim["allocated"] = []
                victim["computed"] = 0
                victim["running"] = False
                victim["preempted"] = True
                continue
            events = [
                r["arrival"]
                for r in rows
                if not r["ended"] and t < r["arrival"] < math.inf
            ]
            events += [
                p["deadline"]
                for p in pins.values()
                if p["blocks"] and p["deadline"] > t
            ]
            if not events:
                return None
            t = min(events)
            continue
        Q = sum((q for r, q, start in batch))
        A = sum((q * (2 * start + q - 1) // 2 for r, q, start in batch))
        N = len(batch)
        is_decode = all((start > r["prompt"] and q == 1 for r, q, start in batch))
        decode_steps = 1
        if is_decode and "decode_step_s" in state:
            duration = state.get("decode_by_seqs", {}).get(
                str(N), state["decode_step_s"]
            )
            if state.get("coalesce_decode", True):
                decode_steps = min(
                    (
                        min(
                            r["remaining"],
                            len(r["allocated"]) * block_size - r["computed"],
                        )
                        for r, q, start in batch
                    )
                )
                next_events = [
                    r["arrival"]
                    for r in rows
                    if not r["ended"] and t < r["arrival"] < math.inf
                ]
                next_events += [
                    p["deadline"]
                    for p in pins.values()
                    if p["blocks"] and p["deadline"] > t
                ]
                if next_events:
                    decode_steps = min(
                        decode_steps,
                        max(1, math.floor((min(next_events) - t) / duration)),
                    )
                decode_steps = max(1, decode_steps)
                if decode_steps > 1:
                    batch = [(r, decode_steps, start) for r, q, start in batch]
                    Q *= decode_steps
                    A = sum((q * (2 * start + q - 1) // 2 for r, q, start in batch))
                    duration *= decode_steps
        else:
            duration = max(1e-06, beta[0] + beta[1] * Q + beta[2] * A + beta[3] * N)
        end = t + duration
        batches.append(dict(start=t, end=end, Q=Q, A=A, N=N))
        for r, q, start in batch:
            previously_complete = r["computed"] // block_size
            r["computed"] += q
            if r["finish"] is None and r["computed"] >= r["prompt"]:
                r["finish"] = end
            for index in range(previously_complete, r["computed"] // block_size):
                bid = r["allocated"][index]
                content[bid] = (r["program"], index)
                valid.add(bid)
                if index < len(r["lookup"]):
                    if not r.get("lookup_copied"):
                        r["lookup"] = list(r["lookup"])
                        r["lookup_copied"] = True
                    r["lookup"][index] = bid
            if r["computed"] >= r["sequence"]:
                r["remaining"] -= decode_steps
                r["sequence"] += decode_steps
                if r["remaining"] <= 0:
                    successor = r.get("successor")
                    if successor is not None:
                        successor["arrival"] = end + successor["predicted_gap"]
                        successor["lookup"] = list(
                            r["allocated"][: r["prompt"] // block_size]
                        )
                        continuation = r["continuation"]
                        committed = list(r["allocated"][: r["computed"] // block_size])
                        if continuation["ttl"] > 0 and committed:
                            touch(committed)
                            pins[r["program"]] = dict(
                                id=r["program"],
                                order=r["order"],
                                blocks=committed,
                                deadline=end + continuation["ttl"],
                            )
                    release(reversed(r["allocated"]))
                    r["running"] = False
                    r["ended"] = True
                    r["completion"] = end
        t = end
    return None


def search_suffixes(rows, K, snapshot, budget_s=0.01, max_evals=32, epsilon=0.0):
    from kairos._runtime.compat.policy import select_suffixes

    start = time.perf_counter()
    initial, _ = select_suffixes(
        rows,
        K,
        snapshot["block_size"],
        snapshot["budget"],
        snapshot["beta"],
        optimize=False,
    )
    deadline = start + budget_s
    evaluations = 0

    def cost(plan):
        total = 0.0
        for s in range(3):
            value = evaluate(snapshot, plan, s, deadline)
            if value is None:
                return None
            total += value["J"]
        return total / 3

    base = cost(initial)
    best = base
    selected = initial
    if base is not None:
        while evaluations < max_evals and time.perf_counter() < deadline:
            candidate = None
            candidate_cost = best
            for g in (64, 16, 4, 1):
                for src in rows:
                    if selected.get(src["id"], 0) < g:
                        continue
                    for dst in rows:
                        if src is dst or selected.get(dst["id"], 0) + g > dst["limit"]:
                            continue
                        if evaluations >= max_evals or time.perf_counter() >= deadline:
                            break
                        trial = dict(selected)
                        trial[src["id"]] -= g
                        trial[dst["id"]] = trial.get(dst["id"], 0) + g
                        value = cost(trial)
                        evaluations += 1
                        if value is not None and value < candidate_cost:
                            candidate = trial
                            candidate_cost = value
            if candidate is None or best - candidate_cost <= epsilon:
                break
            selected, best = (candidate, candidate_cost)
    elapsed = time.perf_counter() - start
    objective_count = sum((r["computed"] < r["prompt"] for r in snapshot["requests"]))
    known = {r["program"] for r in snapshot["requests"]}
    objective_count += sum((p["id"] not in known for p in snapshot["pins"]))
    overhead_bound = elapsed * max(1, objective_count)
    accepted = (
        base is not None
        and best is not None
        and (base - best > epsilon + overhead_bound)
    )
    if not accepted:
        selected = initial
        best = base
    return (
        {p: n for p, n in selected.items() if n},
        dict(
            initial_J=base or 0,
            best_J=best or 0,
            evaluations=evaluations,
            search_s=elapsed,
            recovery_proxy="physical-residency-event-model-native-prefix",
            evaluation_complete=base is not None,
            search_accepted=accepted,
            search_overhead_objective_bound_s=overhead_bound,
        ),
    )
