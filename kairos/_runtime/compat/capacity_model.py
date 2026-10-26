"""Kairos capacity model implementation."""
from __future__ import annotations
import math
import time


def evaluate_capacity(snapshot, plan, scenario=0, deadline=None):
    size = snapshot["block_size"]
    pins = {p["id"]: max(0, p["protected_blocks"]-plan.get(p["id"], 0))
            for p in snapshot["rows"]}
    rows = []
    for source in snapshot["rows"]:
        row = dict(source)
        row["arrival"], added = row["scenarios"][scenario]
        row["prompt"] = row["prompt_tokens"] if row["queued"] else row["old_tokens"]+added
        row["computed"] = row.get("computed_tokens", 0)
        row["remaining"] = row["output_tokens"]
        row["ended"] = False
        row["finish"] = None
        row["running"] = bool(row.get("running"))
        row["was_pinned"] = row.get("active_rank") == 1
        row["footprint"] = math.ceil((max(row["prompt"], row["computed"])+max(0, row["remaining"]-1))/size)
        row["cached"] = row["cached_tokens"]
        if plan.get(row["id"]):
            row["cached"] = min(row["cached"], pins[row["id"]]*size)
        rows.append(row)
    capacity = snapshot["capacity"]
    t = 0.0
    batches = 0

    def rank(row):
        pinned = row["was_pinned"] if row["running"] else bool(pins[row["id"]])
        return (0 if row.get("preempted") else 1 if pinned else 2, row["order"])

    for _ in range(512):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        if all(row["ended"] for row in rows):
            return dict(J=sum(row["finish"]-row["arrival"] for row in rows), batches=batches,
                        F={row["id"]: row["finish"] for row in rows})
        for row in rows:
            if pins[row["id"]] and t >= row["ttl_remaining"] \
                    and not (row["queued"] or row["arrival"] <= t):
                pins[row["id"]] = 0
                row["cached"] = 0
        running = sorted([r for r in rows if r["running"] and not r["ended"]], key=rank)
        waiting = sorted([r for r in rows if not r["running"] and not r["ended"]
                          and r["arrival"] <= t], key=rank)
        used = sum(pins.values())+sum(r["footprint"] for r in running)
        # Full-footprint reservation is intentionally more conservative than
        # native chunk allocation; invalid snapshots return UNKNOWN.
        if used > capacity:
            return None
        for row in waiting:
            if len(running) >= snapshot["max_seqs"]:
                break
            need = row["footprint"]-pins[row["id"]]
            if used+need > capacity:
                break
            used += need
            row["was_pinned"] = bool(pins[row["id"]])
            pins[row["id"]] = 0
            row["computed"] = max(row["computed"], min(row["cached"], row["prompt"]-1))
            row["running"] = True
            running.append(row)
        if not running:
            if not waiting:
                events = [r["arrival"] for r in rows if not r["ended"] and r["arrival"] > t]
                events += [r["ttl_remaining"] for r in rows if pins[r["id"]]
                           and r["ttl_remaining"] > t]
                if not events:
                    return None
                t = min(events)
                continue
            victim = next((r for r in sorted(rows, key=lambda r:r["order"], reverse=True)
                           if pins[r["id"]]), None)
            if victim:
                pins[victim["id"]] = 0
                victim["cached"] = 0
                continue
            arrivals = [r["arrival"] for r in rows if not r["ended"] and r["arrival"] > t]
            if not arrivals:
                return None
            t = min(arrivals)
            continue
        running.sort(key=rank)
        left = snapshot["budget"]
        batch = []
        for row in running:
            if not left:
                break
            count = min(left, max(1, row["prompt"]-row["computed"]))
            batch.append((row, count, row["computed"]+1))
            left -= count
        decode = all(start > row["prompt"] for row,count,start in batch)
        steps = 1
        if decode:
            durations = snapshot.get("decode_by_seqs", {})
            duration = durations.get(str(len(batch)), snapshot["decode_step_s"])
            steps = min(row["remaining"] for row,count,start in batch)
            events = [r["arrival"] for r in rows if not r["ended"] and r["arrival"] > t]
            events += [r["ttl_remaining"] for r in rows if pins[r["id"]]
                       and r["ttl_remaining"] > t]
            if events:
                steps = min(steps, max(1, math.floor((min(events)-t)/duration)))
            duration *= steps
        else:
            Q = sum(count for row,count,start in batch)
            A = sum(count*(2*start+count-1)/2 for row,count,start in batch)
            beta = snapshot["beta"]
            duration = max(1e-6, beta[0]+beta[1]*Q+beta[2]*A+beta[3]*len(batch))
        t += duration
        batches += 1
        for row,count,start in batch:
            row["computed"] += steps if decode else count
            if row["computed"] >= row["prompt"]:
                row["remaining"] -= steps if decode else 1
                if row["remaining"] <= 0:
                    row["ended"] = True
                    row["running"] = False
                    row["finish"] = t
    return None
