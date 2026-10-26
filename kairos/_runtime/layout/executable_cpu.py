"""Kairos executable cpu implementation."""
import math
import time

from kairos._runtime.compat.lmcache_replica import identity
from kairos._runtime.layout.executable_recovery import RecoveryPlans, page_masks
from kairos._runtime.layout.compact_recovery import MaskRecovery


def pressure_candidates(runtime, keys, layout_desc, mode):
    from lmcache.v1.distributed.error import L1Error
    if mode == 'update':
        return keys, []
    started = time.perf_counter()
    keys = list(dict.fromkeys(keys))
    size = sum(math.prod(shape) * dtype.itemsize for shape, dtype in
               zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
    with runtime.manager._lock:
        rows = list(runtime.manager._objects.items())
    present = {key for key, _ in rows}
    used, capacity = runtime.manager.get_memory_usage()
    free = max(0, (capacity - used) // size)
    absent = [key for key in keys if key not in present]
    if len(absent) <= free:
        return keys, []
    histories = runtime.registry.history.programs
    known = {}
    for name, row in histories.items():
        for i, key in enumerate(row.keys):
            known.setdefault(key.base(), []).append((name, i))
    cpu = {name: set() for name in histories}
    for key, entry in rows:
        if not entry.write_lock.is_locked() and not entry.is_temporary:
            for name, i in known.get(identity(key).base(), ()):
                cpu[name].add(i)
    masks = {name: page_masks(runtime.gapped_gpu_pages.get(name, ()), len(row.keys) * 256 + 1)
             for name, row in histories.items()}
    plans = runtime.executable_plans
    local = MaskRecovery(runtime.gapped_costs, masks, cpu, {name: 1. for name in masks})
    def local_value(key, adding=False):
        total = 0.
        for name, i in known.get(identity(key).base(), ()):
            if histories[name].state == 'finished':
                continue
            mask = masks[name][i]
            compute = local.compute(i, mask)
            backed = min(compute, runtime.gapped_costs.load_token * 256 +
                         runtime.gapped_costs.load_call)
            total += max(0., compute - backed) if (i not in cpu[name] if adding else i in cpu[name]) else 0.
        return total
    def value(key, adding=False):
        total = 0.
        for name, i in known.get(identity(key).base(), ()):
            if histories[name].state == 'finished':
                continue
            end = len(histories[name].keys) * 256 + 1
            before = plans.get(masks[name], cpu[name], end).cost
            changed = cpu[name] | {i} if adding else cpu[name] - {i}
            after = plans.get(masks[name], changed, end).cost
            total += max(0., before - after if adding else after - before)
        return total
    with runtime.eviction_policy._lock:
        ranks = {key: i for i, key in enumerate(runtime.eviction_policy._order)}
    pending = {key for group in runtime.repairs.values() for key in group}
    eligible = {key for key, _ in rows if key not in keys
        and identity(key) not in pending and identity(key).base() not in runtime.hicache_holds
        and not runtime.cpu_service_ledger.protected(identity(key).base(), time.monotonic())
        and runtime.manager.is_key_evictable(key)}
    kept = {key for key in keys if key in present}
    remaining = set(absent)
    decisions = []
    evaluations = plans.evaluations
    while remaining:
        # Whole-plan evaluation stays off the all-object hot scan. The same
        # cheap conditional chunk scores screen three concrete alternatives.
        incoming_choices = sorted(remaining,
            key=lambda key: (-local_value(key, True), keys.index(key)))[:3]
        incoming = max(incoming_choices, key=lambda key: (value(key, True), -keys.index(key)))
        gain = value(incoming, True)
        victim_choices = sorted(eligible,
            key=lambda key: (local_value(key), ranks.get(key, 0)))[:3] if not free else []
        victim = min(victim_choices, key=lambda key: (value(key), ranks.get(key, 0))) if victim_choices else None
        loss = value(victim) if victim is not None else float('inf')
        remaining.remove(incoming)
        if not free and (victim is None or gain <= loss):
            decisions.append(dict(accepted=False, incoming_value_s=gain, victim_value_s=loss))
            continue
        if not free:
            eligible.remove(victim)
            if runtime.manager.delete([victim], force=False).get(victim) != L1Error.SUCCESS:
                decisions.append(dict(accepted=False, incoming_value_s=gain,
                                      victim_value_s=loss, delete_refused=True))
                continue
            runtime.registry.record_deleted([identity(victim)])
            runtime.publish_admission_telemetry(deleted=1)
            for name, i in known.get(identity(victim).base(), ()):
                cpu[name].discard(i)
            free += 1
        kept.add(incoming)
        for name, i in known.get(identity(incoming).base(), ()):
            cpu[name].add(i)
        free -= 1
        decisions.append(dict(accepted=True, incoming_value_s=gain,
                              victim_value_s=0. if victim is None else loss))
    skipped = [key for key in keys if key not in kept]
    runtime.record('executable_cpu_admission', skipped_chunks=len(skipped),
        admitted_chunks=len(kept - present), decisions=decisions,
        plan_evaluations=plans.evaluations - evaluations,
        controller_s=time.perf_counter() - started,
        scope='observed history/current GPU advice, not future survival; native reservation authoritative')
    return [key for key in keys if key in kept], skipped


def configure(runtime, message):
    config = message.get('executable_config')
    runtime.executable_plans = RecoveryPlans(runtime.gapped_costs, **config) if config else None
