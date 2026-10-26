"""Kairos runtime implementation."""
import math
import os
import time

from kairos._runtime.compat.lmcache_replica import ReplicaRuntime, identity
from kairos._runtime.layout.plan import Costs


class GappedRuntime(ReplicaRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gapped_policy = ''
        self.gapped_costs = Costs()
        self.gapped_gpu_pages = {}

    def dispatch(self, message):
        with self.registry.lock:
            if message['op'] == 'experiment':
                self.gapped_policy = message.get('gapped_policy', '')
                self.gapped_gpu_pages.clear()
                self.gapped_costs = Costs(**message.get('gapped_costs', {}))
            if message['op'] == 'gapped_gpu_pages':
                self.gapped_gpu_pages = {p: set(pages) for p, pages in message['programs'].items()}
                return {}
            if message['op'] == 'gapped_gpu_withdraw':
                for p, pages in message['programs'].items():
                    self.gapped_gpu_pages.get(p, set()).difference_update(pages)
                return {}
            if message['op'] == 'gapped_fixture_cpu':
                if os.environ.get('CONTINUUM_GAPPED_FIXTURE') != '1':
                    raise RuntimeError('Controlled layouts are disabled in benchmark runs')
                from lmcache.v1.distributed.error import L1Error
                self.dispatch(dict(op='hicache_register', program=message['program'],
                    model=message['model'], salt=message['salt'], tokens=message['tokens']))
                manifest = self.hicache_manifests[message['program']]
                keep = set(message['keep'])
                with self.manager._lock:
                    objects = list(self.manager._objects)
                known = {key.base(): i for i, key in enumerate(manifest)}
                victims = [key for key in objects if identity(key).base() in known
                           and known[identity(key).base()] not in keep]
                if any(not self.manager.is_key_evictable(key) for key in victims):
                    raise RuntimeError('Controlled layout requires drained, unlocked CPU data')
                result = self.manager.delete(victims, force=False)
                if any(result.get(key) != L1Error.SUCCESS for key in victims):
                    raise RuntimeError('CPU fixture deletion failed')
                self.registry.record_deleted([identity(key) for key in victims])
                with self.manager._lock:
                    remaining = set(self.manager._objects)
                present = sorted({known[identity(key).base()] for key in remaining
                                  if identity(key).base() in known})
                if set(present) != keep:
                    raise RuntimeError('CPU fixture does not contain every requested retained chunk')
                self.record('gapped_controlled_cpu_layout', keep=sorted(keep),
                            deleted_chunks=len(victims), synthetic=True)
                return dict(deleted=len(victims), cpu_present_chunks=present)
            return super().dispatch(message)

    def hicache_make_room(self, rows, wanted, deficit, chunk_bytes, trigger):
        if self.gapped_policy not in ('gapped-tier', 'gapped-joint'):
            return super().hicache_make_room(rows, wanted, deficit, chunk_bytes, trigger)
        from lmcache.v1.distributed.error import L1Error
        started = time.perf_counter()
        costs = self.gapped_costs
        with self.eviction_policy._lock:
            recency = {key: i for i, key in enumerate(self.eviction_policy._order)}
        known = {k.base(): (p, i) for p, keys in self.hicache_manifests.items()
                 for i, k in enumerate(keys)}
        cpu = {p: set() for p in self.hicache_manifests}
        for key, entry in rows:
            match = known.get(identity(key).base())
            if match and not entry.write_lock.is_locked() and not entry.is_temporary:
                cpu[match[0]].add(match[1])
        eligible = {key for key, _ in rows if key not in wanted
                    and identity(key).base() not in self.hicache_holds
                    and self.manager.is_key_evictable(key)}
        deleted, decisions = [], []
        for _ in range(math.ceil(deficit / chunk_bytes)):
            if not eligible:
                break
            before = {p: costs.estimate_pages(self.gapped_gpu_pages.get(p, set()),
                       chunks, len(self.hicache_manifests[p]))[0] for p, chunks in cpu.items()}
            def score(key):
                match = known.get(identity(key).base())
                if match is None:
                    return (0., recency.get(key, 0))
                p, i = match
                gpu = self.gapped_gpu_pages.get(p, set())
                if self.gapped_policy == 'gapped-joint':
                    value = costs.estimate_pages(gpu, cpu[p] - {i},
                                                 len(self.hicache_manifests[p]))[0] - before[p]
                else:
                    duplicate = all(j in gpu for j in range(i * 16, (i + 1) * 16))
                    value = 0. if duplicate else max(0., costs.chunk_compute(i) -
                        (costs.load_token * costs.chunk + costs.load_call))
                return (max(0., value), recency.get(key, 0))
            key = min(eligible, key=score)
            loss = score(key)[0]
            eligible.remove(key)
            if self.manager.delete([key], force=False).get(key) == L1Error.SUCCESS:
                deleted.append(key)
                match = known.get(identity(key).base())
                if match:
                    cpu[match[0]].discard(match[1])
                decisions.append(dict(program=match[0] if match else None,
                    chunk=match[1] if match else None, marginal_recovery_s=loss))
        self.registry.record_deleted([identity(key) for key in deleted])
        self.record('gapped_cpu_reclaim', policy=self.gapped_policy,
            deficit_bytes=deficit, deleted_bytes=len(deleted) * chunk_bytes,
            decisions=decisions, controller_s=time.perf_counter() - started)
        return deleted
