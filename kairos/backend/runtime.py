"""backend.runtime."""

import math
import time
from kairos._runtime.compat.lmcache_replica import identity
from kairos._runtime.layout.runtime import GappedRuntime
from kairos.backend.coverage import CoverageHistory


class CoverageRuntime(GappedRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry.history = CoverageHistory()
        self.record(
            "forecast_residency_registry",
            objective="joint restorable prefix endpoint",
            cpu_reclaim="actual STORE pressure; zero loss for every live physical-key owner",
            admission="unchanged complementary history and boundary backup",
            gpu_allocation="native ref_cnt=0 pages only",
            timing_in_zero_loss_selection=False,
        )

    def history_resident_keys(self, rows):
        return [
            identity(key)
            for key, entry in rows
            if not entry.write_lock.is_locked() and (not entry.is_temporary)
        ]

    def pressure_write_candidates(self, keys, layout_desc, mode):
        if mode == "update" or not self.pressure_write_bridge:
            return (keys, [])
        size = sum(
            (
                math.prod(shape) * dtype.itemsize
                for shape, dtype in zip(
                    layout_desc.shapes, layout_desc.dtypes, strict=True
                )
            )
        )
        if size <= 0:
            raise ValueError("Invalid STORE payload size")
        with self.manager._lock:
            present = set(self.manager._objects)
        used, capacity = self.manager.get_memory_usage()
        free = max(0, (capacity - used) // size)
        kept, skipped, audit = self.registry.history.write_budget(
            [identity(k) for k in keys],
            [identity(k) for k in present],
            free,
            self.registry.grants,
            time.monotonic(),
        )
        selected = {k.base() for k in kept}
        if skipped:
            self.record(
                "forecast_pressure_write_budget", skipped_chunks=len(skipped), **audit
            )
        return (
            [k for k in keys if identity(k).base() in selected],
            [k for k in keys if identity(k).base() not in selected],
        )
