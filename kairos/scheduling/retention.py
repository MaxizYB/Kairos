"""scheduling.retention."""

import hashlib
import json
from kairos.scheduling.capacity import admission


def hash_text(value):
    return value.hex() if isinstance(value, bytes) else repr(value)


def common_prefix(old, new):
    count = 0
    for a, b in zip(old, new):
        if a != b:
            break
        count += 1

    def digest(values):
        raw = json.dumps([hash_text(x) for x in values], separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    return (
        count,
        dict(
            old_pages=len(old),
            current_pages=len(new),
            common_pages=count,
            old_prefix_sha256=digest(old[:count]),
            current_prefix_sha256=digest(new[:count]),
            old_boundary=hash_text(old[count]) if count < len(old) else None,
            current_boundary=hash_text(new[count]) if count < len(new) else None,
        ),
    )


def reference_scan(
    calls, reserved, occupied, optional, priorities, limit, slots, *, cleaning=()
):
    ledger, occupied, optional = (dict(reserved), set(occupied), set(optional))
    checks, selected, victims, claim = ([], [], [], None)
    for call in calls:
        p, target = (call["program"], call["target_pages"])
        before = dict(ledger)
        locked = occupied | set(cleaning)
        eligible = optional - locked
        credit = (
            min(max(0, limit - sum(before.values())), claim["required_pages"])
            if claim
            else 0
        )
        decision = admission(
            p,
            target,
            before,
            occupied,
            eligible,
            priorities,
            limit,
            slots,
            claim_pages=credit,
            claim_pending=claim is not None,
            locked=locked,
        )
        row = dict(
            call,
            reserved_before=before,
            occupied=sorted(occupied),
            locked=sorted(locked),
            optional=sorted(eligible),
            limit=limit,
            max_running=slots,
            claim_before=dict(claim, credit_pages=credit) if claim else None,
            funded=decision.funded,
            victims=list(decision.victims),
            reason=decision.reason,
        )
        if decision.funded:
            for victim in decision.victims:
                ledger.pop(victim, None)
                optional.discard(victim)
                victims.append(victim)
            ledger[p] = target
            occupied.add(p)
            selected.append(p)
        elif claim is None and decision.reason == "protected-capacity-wait":
            claim = dict(
                program=p,
                request=call["request"],
                required_pages=decision.required_pages,
                target_pages=target,
            )
        row["reserved_after"] = dict(ledger)
        checks.append(row)
    return dict(
        checks=checks,
        selected=selected,
        victims=victims,
        reserved_at_end=ledger,
        claim=claim,
    )


def retention_plan(reference, initial, rows, priorities, limit, *, enabled):
    selected = set(reference["selected"])
    victims = reference["victims"]
    final_free = max(0, limit - sum(reference["reserved_at_end"].values()))
    claim = reference["claim"]
    credit = min(final_free, claim["required_pages"]) if claim else 0
    ledger = {p: n for p, n in initial.items() if p not in set(victims)}
    minimum_slack = max(0, limit - sum(ledger.values()))
    for row in reference["checks"]:
        if row["funded"]:
            ledger[row["program"]] = row["target_pages"]
            minimum_slack = min(minimum_slack, max(0, limit - sum(ledger.values())))
    if ledger != reference["reserved_at_end"]:
        raise ValueError("Reference commit path differs from its sequential plan")
    budget = min(max(0, final_free - credit), minimum_slack) if enabled else 0
    remaining, keep = (budget, {})
    for p in sorted(set(victims) - selected, key=priorities.__getitem__):
        row = rows[p]
        upper = min(initial[p], row["usable_pages"])
        candidate = min(remaining, upper)
        minimum_keep = row["total_pages"] - row["safe_suffix_pages"]
        if row["total_pages"] != initial[p] or candidate < minimum_keep:
            candidate = 0
        if candidate:
            keep[p] = candidate
            remaining -= candidate
    final = dict(reference["reserved_at_end"])
    final.update(keep)
    if sum(final.values()) > limit and (keep or reference["selected"]):
        raise ValueError("Retained prefix overcommits a funded plan")
    return dict(
        keep=keep,
        final_free_pages=final_free,
        claim_credit_pages=credit,
        minimum_commit_slack_pages=minimum_slack,
        available_pages=budget,
        retained_pages=sum(keep.values()),
        reserved_at_end=final,
    )


def private_suffix(pin, owners=None, *, maximum=None):
    count = 0
    seen = set()
    for block in reversed(pin.blocks):
        if (
            getattr(block, "is_null", False)
            or getattr(block, "block_hash", None) is None
            or block.ref_cnt != 1
            or (block.block_id in seen)
            or (owners is not None and owners[block.block_id] != 1)
        ):
            break
        seen.add(block.block_id)
        count += 1
        if maximum is not None and count >= maximum:
            break
    return count
