"""Kairos recovery runtime implementation."""
import math
import time

from kairos._runtime.layout.pcv import Program
from kairos._runtime.layout.recovery import ChunkRecovery


def recycle_cpu(runtime, rows, wanted, deficit, chunk_bytes, trigger):
    from lmcache.v1.distributed.error import L1Error
    from kairos._runtime.compat.lmcache_replica import identity
    started = time.perf_counter()
    with runtime.eviction_policy._lock:
        ranks = {k: i for i, k in enumerate(runtime.eviction_policy._order)}
    known = {k.base(): (p, i) for p, keys in runtime.hicache_manifests.items()
             for i, k in enumerate(keys)}
    cpu = {p: set() for p in runtime.hicache_manifests}
    for key, entry in rows:
        owner = known.get(identity(key).base())
        if owner and not entry.write_lock.is_locked() and not entry.is_temporary:
            cpu[owner[0]].add(owner[1])
    programs = [Program(**state, gpu=frozenset(runtime.gapped_gpu_pages.get(p, ())),
                        cpu=frozenset(cpu.get(p, ()))) for p, state in runtime.pcv_state.items()]
    by_name = {p.name: p for p in programs}
    slopes = getattr(runtime, 'pcv_slopes', {}) if runtime.gapped_policy == 'gapped-event' else {}
    weights = {p.name: p.probability * slopes.get(p.name, 1.) for p in programs}
    recovery = ChunkRecovery(runtime.gapped_costs, programs, weights)
    pending = {key for group in runtime.repairs.values() for key in group}
    eligible = [key for key, _ in rows if key not in wanted and identity(key) not in wanted
                and identity(key) not in pending
                and identity(key).base() not in runtime.hicache_holds
                and runtime.manager.is_key_evictable(key)]
    def score(key):
        owner = known.get(identity(key).base())
        if not owner or owner[0] not in by_name:
            return 0., ranks.get(key, 0)
        p, index = owner
        if runtime.gapped_policy == 'gapped-fast':
            duplicate = all(j in recovery.gpu[p] for j in range(index * 16, (index + 1) * 16))
            loss = 0. if duplicate else max(0., runtime.gapped_costs.chunk_compute(index) -
                runtime.gapped_costs.load_token * 256 - runtime.gapped_costs.load_call)
            return by_name[p].probability * loss, ranks.get(key, 0)
        return recovery.cpu_loss(p, index), ranks.get(key, 0)
    # Chunk-local plans are independent in this bounded proxy. A successful
    # removal updates coverage; another chunk's score is unchanged. This is
    # not a claim of exact global segment/queue optimality.
    ranked = sorted(eligible, key=score)
    deleted, decisions = [], []
    required = math.ceil(deficit / chunk_bytes)
    for key in ranked:
        if len(deleted) >= required:
            break
        value = score(key)[0]
        if runtime.manager.delete([key], force=False).get(key) == L1Error.SUCCESS:
            deleted.append(key)
            owner = known.get(identity(key).base())
            if owner:
                recovery.cpu[owner[0]].discard(owner[1])
            decisions.append(dict(program=owner[0] if owner else None,
                chunk=owner[1] if owner else None, next_round_marginal_s=value))
    runtime.registry.record_deleted([identity(key) for key in deleted])
    runtime.record('pcv_cpu_reclaim', policy=runtime.gapped_policy, trigger=trigger,
        deficit_bytes=deficit, deleted_bytes=len(deleted) * chunk_bytes,
        decisions=decisions, evaluations=len(eligible), budget_fallback_chunks=0,
        controller_s=time.perf_counter() - started,
        prediction='conditional whole-chunk recovery; no per-victim rollout')
    return deleted
