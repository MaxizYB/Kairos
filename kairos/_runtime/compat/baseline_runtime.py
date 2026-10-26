"""Kairos baseline runtime implementation."""
from kairos._runtime.compat.lmcache_replica import ReplicaRuntime, identity
from kairos._runtime.compat.replica import ReplicaKey


class BaselineRuntime(ReplicaRuntime):
    def dispatch(self, message):
        if message["op"] != "exclusive_drop":
            return super().dispatch(message)
        from lmcache.v1.distributed.error import L1Error
        with self.registry.lock:
            wanted = {ReplicaKey(message["model"], message["salt"], h.hex(),
                rank=message["kv_rank"]).base()
                for h in self.hasher.compute_chunk_hashes(message["tokens"])}
            with self.manager._lock:
                rows = [key for key in self.manager._objects if identity(key).base() in wanted]
            result = self.manager.delete(rows, force=False)
            deleted = [key for key, status in result.items() if status == L1Error.SUCCESS]
            locked = len(rows)-len(deleted)
            self.record("baseline_exclusive_drop", deleted=deleted, locked=locked,
                        gpu_proof="completed_forward_with_active_request_references")
            return dict(deleted=len(deleted), locked=locked)

    def repair_room(self, message):
        if not (self.experiment or "").startswith(("cache-writeback", "cache-exclusive")):
            return super().repair_room(message)
        from lmcache.v1.distributed.error import L1Error
        indices = message["indices"]
        if not indices or len(indices) > 4 or indices != list(range(indices[0], indices[-1]+1)):
            raise ValueError("Unbounded writeback batch")
        hashes = self.hasher.compute_chunk_hashes(message["tokens"])
        wanted = [ReplicaKey(message["model"], message["salt"], hashes[i].hex(),
                            rank=message["kv_rank"], chunk_index=i) for i in indices]
        with self.manager._lock:
            native = {identity(k): k for k in self.manager._objects}
        size = 256*147456
        needed = sum(key not in native for key in wanted)*size
        used, capacity = self.manager.get_memory_usage()
        deficit = max(0, needed-(capacity-used))
        with self.eviction_policy._lock:
            ordered = list(self.eviction_policy._order)
        pending = {k for keys in self.repairs.values() for k in keys}
        victims = [k for k in ordered if identity(k) not in set(wanted) | pending
                   and self.manager.is_key_evictable(k)]
        count = (deficit+size-1)//size
        results = self.manager.delete(victims[:count], force=False)
        deleted = [k for k, status in results.items() if status == L1Error.SUCCESS]
        used, capacity = self.manager.get_memory_usage()
        allowed = needed <= capacity-used
        if allowed:
            self.repairs[message["repair"]] = wanted
        self.record("replica_repair_room", repair=message["repair"],
            requested_bytes=needed, deficit_bytes=deficit, deleted=deleted,
            deleted_bytes=len(deleted)*size, cpu_used_bytes=used,
            cpu_capacity_bytes=capacity, allowed=allowed, baseline_lru=True)
        return dict(allowed=allowed, needed_bytes=needed)
