"""Kairos recovery probe implementation."""
from __future__ import annotations

import os


def aligned_release(total_pages, demand_pages, private_pages):
    if min(total_pages, demand_pages, private_pages) <= 0:
        raise ValueError("Positive page counts required")
    keep = (total_pages-demand_pages)//16*16
    release = total_pages-keep
    if keep <= 0 or release > private_pages:
        raise ValueError("No bounded private suffix can cover this demand")
    return dict(retained_pages=keep, release_pages=release,
                first_cpu_chunk=keep//16, full_chunks=total_pages//16)


def runtime_class():
    # This import is confined to the server; local helpers remain lightweight.
    from kairos._runtime.compat.lmcache_replica import ReplicaRuntime, identity

    class RecoveryProbeRuntime(ReplicaRuntime):
        def dispatch(self, message):
            if message["op"] != "probe_layout":
                return super().dispatch(message)
            if os.environ.get("CONTINUUM_RECOVERY_PROBE") != "1":
                raise RuntimeError("Layout fixture disabled")
            from lmcache.v1.distributed.error import L1Error
            with self.registry.lock:
                program = message["program"]
                if program != "probe-A" or not self.experiment:
                    raise ValueError("Fixture only owns probe-A")
                grant = self.registry.grants.get(program)
                if grant is None:
                    raise RuntimeError("No live GPU reference certificate")
                start, end = int(message["start"]), int(message["end"])
                if not 0 <= start < end <= len(grant.keys) <= 32:
                    raise ValueError("Unexpected bounded fixture layout")
                with self.manager._lock:
                    native = {identity(key).base(): key for key, entry in self.manager._objects.items()
                              if not entry.write_lock.is_locked()}
                wanted = grant.keys[start:end]
                if any(key.base() not in native for key in wanted):
                    return dict(ready=False)
                if message["layout"] not in ("complete", "gap"):
                    raise ValueError("Unknown fixture layout")
                deleted = []
                if message["layout"] == "gap":
                    key = native[wanted[0].base()]
                    result = self.manager.delete([key], force=False)
                    if result.get(key) != L1Error.SUCCESS:
                        return dict(ready=False)
                    deleted = [start]
                self.record("probe_cpu_layout", program=program, start=start, end=end,
                            layout=message["layout"], deleted_indices=deleted,
                            native_force=False, live_gpu_certificate=True)
                return dict(ready=True, deleted_indices=deleted,
                            ready_suffix_indices=[i for i in range(start, end) if i not in deleted])

    return RecoveryProbeRuntime
