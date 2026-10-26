"""Kairos lmcache replica implementation."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path

from kairos._runtime.compat.replica import ControlServer, LeaseRegistry, MODES, ReplicaKey, replica_model_supported, replica_bytes_per_token
from kairos._runtime.compat.cpu_service_lease import CPUServiceLedger


def serializable(value):
    if isinstance(value, bytes):
        return value.hex()
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if hasattr(value, "value"):
        return serializable(value.value)
    return value


def identity(key):
    return ReplicaKey(key.model_name, key.cache_salt, key.chunk_hash.hex(),
                      key.kv_rank, key.object_group_id)


class ReplicaRuntime:
    def __init__(self, manager, path, log_path, hasher, mode="observe"):
        self.manager = manager
        self.registry = LeaseRegistry(mode)
        self.registry.prefix_floor = int(os.environ.get(
            "CONTINUUM_REPLICA_PREFIX_FLOOR_CHUNKS", "4"))
        self.registry.repair_chunk_limit = int(os.environ.get("CONTINUUM_REPAIR_MAX_CHUNKS", "8"))
        self.hasher = hasher
        self.log_path = Path(log_path)
        self.log_lock = threading.Lock()
        self.experiment = None
        self.repairs = {}
        self.allocation_reclaim = False
        self.partial_admission = False
        self.duplicate_admission = False
        self.complementary_admission = False
        self.demand_zero_reclaim = False
        self.shelter_arrived_tail = False
        self.pressure_write_bridge = False
        self.reference_first_reclaim = False
        self.demand_lru_reclaim = False
        self.hicache_reclaim = False
        self.hicache_manifests = {}
        self.hicache_gpu = {}
        # A demotion transaction can create a host copy before the native
        # allocator reuses the source GPU pages.  Keep that copy out of the
        # host duplicate walk for the lifetime of the transaction; otherwise
        # the next allocation can delete it and enqueue the same D2H again.
        # Values are manifest base keys, not physical allocator keys.
        self.hicache_holds = set()
        self.official_admission_telemetry = False
        self.admission_evictions = deque()
        self.admission_last_publication = 0.
        self.eviction_policy = None
        self._cpu_service_ledger = CPUServiceLedger()
        self.control = ControlServer(path, self.dispatch)
        self.record("replica_init", mode=mode, chunk_size=hasher.chunk_size,
                    hash_algorithm=hasher.hash_algorithm_name,
                    implementation="reclamation-with-bounded-repair-v2")

    @property
    def cpu_service_ledger(self):
        # Some dependency-free tests construct only the native-operation
        # adapter with __new__. The ledger is empty in baseline modes.
        if not hasattr(self, '_cpu_service_ledger'):
            self._cpu_service_ledger = CPUServiceLedger()
        return self._cpu_service_ledger

    def record(self, event, **data):
        with self.log_lock:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(serializable(dict(
                    event=event, time=time.time(), experiment=self.experiment,
                    **data))) + "\n")

    def publish_admission_telemetry(self, deleted=0, force=False):
        if not getattr(self, "official_admission_telemetry", False):
            return
        now = time.monotonic()
        if deleted:
            self.admission_evictions.append((now, deleted))
        while self.admission_evictions and self.admission_evictions[0][0] < now-60.:
            self.admission_evictions.popleft()
        if not force and now-self.admission_last_publication < 1.:
            return
        self.admission_last_publication = now
        with self.manager._lock:
            registered = len(self.manager._objects)
        _, capacity = self.manager.get_memory_usage()
        row = dict(t=time.time(), evicted_in_window=sum(n for _, n in self.admission_evictions),
                   registered=registered, capacity=capacity//(256*replica_bytes_per_token()))
        path = Path(str(self.log_path) + ".admission.json")
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(row))
        pending.replace(path)

    def dispatch(self, message):
        with self.registry.lock:
            self.publish_admission_telemetry()
            op = message["op"]
            if op == "hicache_register":
                keys = [ReplicaKey(message["model"], message["salt"], h.hex(),
                    16777472, 0, i) for i, h in enumerate(
                    self.hasher.compute_chunk_hashes(message["tokens"]))]
                self.hicache_manifests[message["program"]] = keys
                return {}
            if op == "hicache_hold":
                manifest = self.hicache_manifests.get(message["program"], [])
                for index in message.get("indices", []):
                    if 0 <= index < len(manifest):
                        self.hicache_holds.add(manifest[index].base())
                self.record("hicache_demotion_hold", program=message["program"],
                            chunks=list(message.get("indices", [])))
                return dict(held=len(self.hicache_holds))
            if op == "hicache_hold_release":
                count = len(self.hicache_holds)
                self.hicache_holds.clear()
                self.record("hicache_demotion_hold_release", chunks=count,
                            reason=message["reason"])
                return dict(released=count)
            if op == "hicache_gpu_layout":
                self.hicache_gpu = {}
                for program, row in message["programs"].items():
                    manifest = self.hicache_manifests.get(program, [])
                    referenced = set(row["referenced"])
                    for index in row["covered"]:
                        if index < len(manifest):
                            self.hicache_gpu[manifest[index].base()] = (
                                True, index in referenced, program, index)
                return {}
            if op == "hicache_gpu_withdraw":
                for program, indices in message["programs"].items():
                    manifest = self.hicache_manifests.get(program, [])
                    for index in indices:
                        if index < len(manifest):
                            self.hicache_gpu.pop(manifest[index].base(), None)
                            # The source pages are now being reused (or the
                            # allocator cancelled their reuse), so a host
                            # copy is no longer a duplicate that must be
                            # protected by the demotion transaction.
                            self.hicache_holds.discard(manifest[index].base())
                self.record("hicache_demotion_release",
                            chunks=sum(len(indices) for indices in message["programs"].values()))
                return {}
            for program in self.cpu_service_ledger.expire(time.monotonic()):
                self.record('cpu_service_lease_expired', program=program)
            if op == 'cpu_service_release':
                row = self.cpu_service_ledger.revoke(message['program'])
                if row:
                    self.record('cpu_service_lease_released', program=message['program'],
                                reason=message.get('reason','explicit'), chunks=len(row.keys))
                return dict(released=bool(row))
            if op == 'cpu_service_acquire':
                grant = self.registry.grants.get(message['program'])
                chunks = int(message['chunks'])
                start = int(message.get('start_chunk',0))
                if grant is None or not 0 <= start < chunks <= len(grant.keys):
                    return dict(accepted=False, reason='no-exact-live-GPU-certificate')
                with self.manager._lock:
                    ready = {identity(key).base() for key, entry in self.manager._objects.items()
                             if not entry.write_lock.is_locked() and not entry.is_temporary}
                _, capacity = self.manager.get_memory_usage()
                answer = self.cpu_service_ledger.acquire(message['program'],
                    [key.base() for key in grant.keys[start:chunks]], ready,
                    256*replica_bytes_per_token(), capacity, time.monotonic(), float(message['duration_s']))
                self.record('cpu_service_lease_acquire', program=message['program'],
                            start_chunk=start, end_chunk=chunks, **answer)
                return answer
            if op == "hello":
                return dict(chunk_size=self.hasher.chunk_size,
                            hash_algorithm=self.hasher.hash_algorithm_name)
            if op == "experiment":
                if self.cpu_service_ledger.leases:
                    raise RuntimeError('Cannot switch with live CPU service commitments')
                if self.registry.grants or self.registry.borrowed or getattr(self, "repairs", {}):
                    raise RuntimeError("Cannot switch with live GPU grants")
                mode = message["mode"]
                if mode not in MODES:
                    raise ValueError("Unsupported replica mode")
                self.registry.mode = mode
                self.registry.history.programs.clear()
                self.registry.history.configure(message.get("tool_history", {}),
                    message.get("beta", [0, 1e-5, 0, 0]), message.get("reload_per_token", 0))
                self.allocation_reclaim = bool(message.get("allocation_reclaim", False))
                self.partial_admission = bool(message.get("partial_admission", False))
                self.complementary_admission = bool(message.get("complementary_admission", False))
                self.demand_zero_reclaim = bool(message.get("demand_zero_reclaim", False))
                self.shelter_arrived_tail = bool(message.get("shelter_arrived_tail", False))
                self.pressure_write_bridge = bool(message.get("pressure_write_bridge", False))
                self.reference_first_reclaim = bool(message.get("reference_first_reclaim", False))
                self.demand_lru_reclaim = bool(message.get("demand_lru_reclaim", False))
                self.hicache_reclaim = bool(message.get("hicache_reclaim", False))
                self.hicache_manifests = {}
                self.hicache_gpu = {}
                self.hicache_holds = set()
                self.hicache_host_port = None
                self.official_admission_telemetry = bool(message.get("official_admission_telemetry", False))
                self.admission_evictions.clear()
                self.publish_admission_telemetry(force=True)
                self.duplicate_admission = (bool(message.get("duplicate_admission", False))
                                            or self.partial_admission and mode in ("ledger", "span"))
                self.experiment = message["name"]
                self.record("replica_experiment", mode=mode, allocation_reclaim=self.allocation_reclaim,
                            partial_admission=self.partial_admission,
                            complementary_admission=self.complementary_admission,
                            demand_zero_reclaim=self.demand_zero_reclaim,
                            shelter_arrived_tail=self.shelter_arrived_tail,
                            pressure_write_bridge=self.pressure_write_bridge,
                            reference_first_reclaim=self.reference_first_reclaim,
                            demand_lru_reclaim=self.demand_lru_reclaim,
                            hicache_reclaim=self.hicache_reclaim,
                            official_admission_telemetry=self.official_admission_telemetry,
                            duplicate_admission=self.duplicate_admission)
                return {}
            if op == "history":
                if message['state'] == 'finished':
                    held = self.cpu_service_ledger.revoke(message['program'])
                    if held:
                        self.record('cpu_service_lease_released', program=message['program'],
                                    reason='program-finished', chunks=len(held.keys))
                keys = None
                if "tokens" in message:
                    keys = [ReplicaKey(message["model"], message["salt"], h.hex(),
                                int(message.get("kv_rank", 16777472)), 0, index)
                            for index, h in enumerate(self.hasher.compute_chunk_hashes(message["tokens"]))]
                self.registry.history.update(message["program"], message["state"],
                    keys, message.get("tool", ""), message.get("observed_gap"))
                self.record("replica_history", program=message["program"], state=message["state"],
                            chunks=len(keys) if keys is not None else None)
                return {}
            if op == "gpu_history":
                if message.get('event_tracked'):
                    self.registry.history.observe_gpu(message['prefixes'],event_tracked=True)
                else:
                    # Frozen v21 histories use the original advisory API.
                    self.registry.history.observe_gpu(message['prefixes'])
                self.record("replica_gpu_history", prefixes=message["prefixes"],
                            advisory_only=True, source=message.get("source", "periodic_observation"))
                return {}
            if op == "cpu_residency":
                if message.get("withdraw_gpu_advice"):
                    # Allocator reuse can invalidate an unpinned APC hint
                    # within its usual 50ms refresh interval. Stop treating
                    # that hint as coverage while the victim plan is built.
                    # Actual live-reference grants remain unchanged.
                    self.registry.history.observe_gpu({
                        program: 0 for program in self.registry.history.programs})
                with self.manager._lock:
                    ready = {identity(key).base() for key, entry in self.manager._objects.items()
                             if not entry.write_lock.is_locked()}
                    writing = ({identity(key).base() for key, entry in self.manager._objects.items()
                                if entry.write_lock.is_locked()}
                               if message.get('include_pending') else set())
                programs = {program: dict(state=row.state, chunks=len(row.keys),
                    cpu_chunks=[i for i, key in enumerate(row.keys) if key.base() in ready])
                    for program, row in self.registry.history.programs.items()}
                if message.get('include_pending'):
                    for program,layout in programs.items():
                        layout['writing_chunks']=[i for i,key in enumerate(
                            self.registry.history.programs[program].keys) if key.base() in writing]
                if message.get("recovery_weights") or message.get("recovery_costs"):
                    now = time.monotonic()
                    for program, layout in programs.items():
                        row = self.registry.history.programs[program]
                        grant = self.registry.grants.get(program)
                        lease = max(0.0, grant.deadline-now) if grant else 0.0
                        _, residual = self.registry.history.timing(row, lease, now)
                        layout.update(recovery_weight=1/(0.05+residual),
                                      estimated_residual_s=residual)
                        if message.get("recovery_costs"):
                            layout.update(prefill_linear=max(0., self.registry.history.beta[1]),
                                          prefill_attention=max(0., self.registry.history.beta[2]),
                                          reload_per_token=self.registry.history.reload_per_token)
                return dict(programs=programs)
            if op == "prepare_release":
                grant = self.registry.grants.get(message["program"])
                if grant is None:
                    return dict(missing=[])
                limit = int(message["chunks"])
                with self.manager._lock:
                    present = {identity(key) for key, entry in self.manager._objects.items()
                               if not entry.write_lock.is_locked()}
                missing = [index for index, key in enumerate(grant.keys[:limit])
                           if key not in present]
                if self.registry.mode in ("ledger", "span"):
                    self.registry.settle_committed(message["program"], present)
                borrowed = self.registry.borrowed_missing(message["program"], present)
                own_missing = [index for index in missing if grant.keys[index] in borrowed]
                repair = (own_missing if self.registry.mode in ("loan", "ledger", "span") else missing)
                if self.registry.mode == "history-active":
                    # Copy only what can extend the leading restore frontier;
                    # do not spend transfers beyond the finite host retention
                    # budget. Existing committed chunks cost no new D2H.
                    row = self.registry.history.programs.get(message["program"])
                    used, capacity = self.manager.get_memory_usage()
                    chunk_bytes = self.hasher.chunk_size * replica_bytes_per_token()
                    budget = min(40, int(.8*capacity)//chunk_bytes)
                    now = time.monotonic()
                    others = sum(h.retained_chunks for p, h in self.registry.history.programs.items()
                                 if p != message["program"] and h.state == "tool" and h.retain_until > now)
                    maximum = min(limit, max(0, budget-others))
                    repair = [i for i in missing if i < maximum][:self.registry.repair_chunk_limit]
                    if not repair:
                        kept = self.registry.history.retain_prefix(message["program"], present,
                                                                   budget=budget)
                        self.record("replica_cpu_retention", program=message["program"],
                                    retained_chunks=kept, budget_chunks=budget, timeout_s=5.0)
                if self.registry.mode == "span" and own_missing:
                    # Repair the leading gaps needed to reach our borrowed
                    # tail. Copying only that tail behind native holes cannot
                    # extend a CPU-only restore after GPU protection ends.
                    repair = [index for index in missing if index <= max(own_missing)][
                        :max(0, self.registry.repair_chunk_limit)]
                self.record("replica_backup_holes", program=message["program"],
                            reason=message["reason"], missing=missing,
                            borrowed_missing=[index for index in missing
                                              if grant.keys[index] in borrowed],
                            selected_repair_indices=repair,
                            closure_native_chunks=len(set(repair)-set(own_missing)),
                            certified_chunks=len(grant.keys))
                return dict(missing=repair)
            if op == "repair_room":
                return self.repair_room(message)
            if op == "coverage":
                hashes = self.hasher.compute_chunk_hashes(message["tokens"])
                wanted = [ReplicaKey(message["model"], message["salt"], h.hex(),
                                    rank=int(message["kv_rank"])) for h in hashes]
                with self.manager._lock:
                    ready = {identity(key) for key, entry in self.manager._objects.items()
                             if not entry.write_lock.is_locked()}
                def contiguous(start):
                    end = start
                    while end < len(wanted) and wanted[end] in ready:
                        end += 1
                    return end
                local = int(message["local_tokens"])
                start = local//self.hasher.chunk_size
                prefix_end = contiguous(0)*self.hasher.chunk_size
                range_end = contiguous(start)*self.hasher.chunk_size
                return dict(local_tokens=local, cpu_prefix_tokens=prefix_end,
                            cpu_range_end_tokens=range_end,
                            cpu_present_chunk_indices=[i for i, key in enumerate(wanted) if key in ready],
                            cpu_tokens_after_first_range_gap=sum(key in ready for key in wanted[contiguous(start):])*self.hasher.chunk_size,
                            extra_suffix_opportunity_tokens=max(0, range_end-max(local, prefix_end)))
            if op == "repair_done":
                keys = getattr(self, "repairs", {}).get(message["repair"], [])
                with self.manager._lock:
                    ready = {identity(key) for key, entry in self.manager._objects.items()
                             if not entry.write_lock.is_locked()}
                    writing = {identity(key) for key, entry in self.manager._objects.items()
                               if entry.write_lock.is_locked()}
                missing = [key for key in keys if key not in ready]
                # The native CUDA completion event precedes the asynchronous
                # host finish_write callback. Keep the CPU eviction exclusion
                # until that callback publishes the committed objects.
                pending = any(key in writing for key in missing)
                if pending:
                    return dict(stored_complete=False, pending=True)
                getattr(self, "repairs", {}).pop(message["repair"], None)
                if self.registry.mode == "history-active" and keys and not missing:
                    programs = {p for p, row in self.registry.history.programs.items()
                                if any(k.base() in {x.base() for x in keys} for k in row.keys)}
                    for program in programs:
                        _, capacity = self.manager.get_memory_usage()
                        budget = min(40, int(.8*capacity)//(self.hasher.chunk_size*replica_bytes_per_token()))
                        kept = self.registry.history.retain_prefix(program, ready, budget=budget)
                        self.record("replica_cpu_retention", program=program,
                                    retained_chunks=kept, budget_chunks=budget, timeout_s=5.0)
                self.record("replica_repair_complete", repair=message["repair"],
                            chunks=len(keys), keys=keys, missing=missing,
                            stored_complete=bool(keys) and not missing)
                return dict(stored_complete=bool(keys) and not missing, pending=False)
            if op == "grant":
                tokens = message["tokens"]
                if not tokens or len(tokens) % self.hasher.chunk_size:
                    raise ValueError("Only complete, contiguous GPU chunks may be granted")
                if len(tokens) > 131072 or any(type(t) is not int or t < 0 for t in tokens):
                    raise ValueError("Invalid token sequence")
                salt, model = message["salt"], message["model"]
                if not salt or not model:
                    raise ValueError("The prototype requires salted, single-model requests")
                rank = int(message.get("kv_rank", 0))
                if rank < 0:
                    raise ValueError("Invalid KV rank")
                keys = [ReplicaKey(model, salt, h.hex(), rank=rank,
                                   chunk_index=index)
                        for index, h in enumerate(self.hasher.compute_chunk_hashes(tokens))]
                self.registry.grant(message["program"], keys, float(message["deadline"]))
                if self.registry.mode in ("ledger", "span"):
                    with self.manager._lock:
                        ready = [identity(key) for key, entry in self.manager._objects.items()
                                 if not entry.write_lock.is_locked()]
                    self.registry.settle_committed(message["program"], ready)
                self.record("replica_grant", program=message["program"],
                            deadline=message["deadline"], keys=keys)
                self.record("replica_snapshot", **self.snapshot())
                return dict(chunks=len(keys))
            if op == "revoke":
                preserve = (self.registry.mode in ("ledger", "span")
                            and bool(message.get("preserve_borrowed", False)))
                self.registry.revoke(message["program"], preserve_borrowed=preserve)
                self.record("replica_revoke", program=message["program"],
                            preserve_borrowed=preserve,
                            outstanding_recorded_chunks=len(self.registry.borrowed.get(message["program"], ())))
                return {}
            if op == "snapshot":
                return self.snapshot()
            raise ValueError("Unknown replica operation")

    def repair_room(self, message):
        """Make finite L1 room; never force-delete locked transfer objects.

        A snapshot is not an allocation reservation.  The worker verifies
        actual completed objects after copying and falls back to normal
        recomputation if another native store wins the allocation race.
        """
        from lmcache.v1.distributed.error import L1Error
        indices = message["indices"]
        if (not indices or len(indices) > 4
                or any(type(index) is not int or index < 0 for index in indices)
                or indices != list(range(indices[0], indices[-1]+1))):
            raise ValueError("Repair batches are bounded to four chunks")
        hashes = self.hasher.compute_chunk_hashes(message["tokens"])
        if not replica_model_supported(message["model"]) or max(indices) >= len(hashes):
            raise ValueError("Unexpected repair layout or token range")
        rank = int(message["kv_rank"])
        wanted = [ReplicaKey(message["model"], message["salt"],
                             hashes[index].hex(), rank=rank,
                             chunk_index=index) for index in indices]
        wanted_set = set(wanted)
        # Known Qwen3-8B BF16 layout: validate exact bytes rather than use a
        # caller-supplied unbounded allocation size.
        chunk_bytes = self.hasher.chunk_size * replica_bytes_per_token()
        with self.manager._lock:
            rows = list(self.manager._objects.items())
            present = {identity(key) for key, _ in rows}
        needed = sum(key not in present for key in wanted) * chunk_bytes
        used, capacity = self.manager.get_memory_usage()
        deficit = max(0, needed-(capacity-used))
        deleted = []
        protected_pending = {key for group in getattr(self, "repairs", {}).values()
                             for key in group}
        candidates = []
        for key, entry in rows:
            ikey = identity(key)
            if (ikey not in wanted_set and ikey not in protected_pending
                    and self.manager.is_key_evictable(key)
                    and (self.registry.mode not in ("coordinated", "frontier", "loan", "ledger", "span", "history-active")
                         or self.registry.protected(ikey))):
                candidates.append((key, entry.memory_obj.get_size()))
        # Whole-program tail first preserves native prefix recoverability.
        positions = {key: index for grant in self.registry.grants.values()
                     for index, key in enumerate(grant.keys)}
        candidates.sort(key=lambda row: (
            -max(0, (self.registry.protected_deadline(identity(row[0])) or 0)
                 - time.monotonic()), -positions.get(identity(row[0]), -1)))
        recovered = 0
        if getattr(self, "hicache_reclaim", False):
            deleted = self.hicache_make_room(rows, wanted_set, deficit, chunk_bytes,
                                             trigger="demotion_writeback")
            recovered = len(deleted) * chunk_bytes
        elif sum(size for _, size in candidates) >= deficit:
            count = (deficit+chunk_bytes-1)//chunk_bytes
            native = {identity(key): key for key, _ in candidates}
            victims = self.registry.select(
                list(native), count,
                eligible=lambda candidate: candidate in native,
                now=time.monotonic(), resident_keys=[identity(key) for key, _ in rows])
            for candidate in victims:
                key = native[candidate]
                size = next(size for item, size in candidates if item == key)
                if self.manager.delete([key], force=False).get(key) == L1Error.SUCCESS:
                    deleted.append(key)
                    recovered += size
        self.registry.record_deleted([identity(key) for key in deleted])
        used, capacity = self.manager.get_memory_usage()
        available = capacity-used >= needed
        if available:
            if not hasattr(self, "repairs"):
                self.repairs = {}
            self.repairs[message["repair"]] = wanted
        self.record("replica_repair_room", repair=message["repair"],
                    requested_bytes=needed, deficit_bytes=deficit,
                    deleted=deleted, deleted_bytes=recovered,
                    cpu_used_bytes=used, cpu_capacity_bytes=capacity,
                    allowed=available)
        return dict(allowed=available, needed_bytes=needed)

    def snapshot(self):
        deadlines = self.registry.deadlines()
        with self.manager._lock:
            rows = [(key, entry.memory_obj.get_size(),
                     entry.read_lock.is_locked() or entry.write_lock.is_locked())
                    for key, entry in self.manager._objects.items()]
        resident = {identity(key) for key, _, _ in rows}
        holes = {program: sum(key not in resident for key in grant.keys)
                 for program, grant in self.registry.grants.items()}
        return dict(cpu_bytes=sum(size for _, size, _ in rows),
                    protected_duplicate_bytes=sum(size for key, size, _ in rows
                                                  if self.registry.protected(identity(key))),
                    unlocked_protected_duplicate_bytes=sum(size for key, size, locked in rows
                                                           if not locked and self.registry.protected(identity(key))),
                    objects=len(rows), grants=len(self.registry.grants),
                    protected_backup_missing_chunks=holes,
                    mode=self.registry.mode)

    def hicache_make_room(self, rows, wanted, deficit, chunk_bytes, trigger):
        from lmcache.v1.distributed.error import L1Error
        from kairos._runtime.compat.hicache_policy import HostRulePort, SOURCE_SHA, CORE_BLOB, COMPONENT_BLOB
        if not deficit:
            return []
        if self.hicache_host_port is None:
            self.hicache_host_port = HostRulePort()
        with self.eviction_policy._lock:
            rank = {key: i for i, key in enumerate(self.eviction_policy._order)}
        known = {key.base(): (p, i) for p, keys in self.hicache_manifests.items()
                 for i, key in enumerate(keys)}
        pending = {key for group in self.repairs.values() for key in group}
        adapted = []
        for key, entry in rows:
            ikey = identity(key)
            program, index = known.get(ikey.base(), (None, -1))
            eligible = (key not in wanted and ikey not in wanted and ikey not in pending
                        and ikey.base() not in self.hicache_holds
                        and self.manager.is_key_evictable(key))
            adapted.append((key, ikey.base(), program, index, eligible,
                            rank.get(key, len(rank))))
        result = self.hicache_host_port.reclaim(adapted, self.hicache_gpu,
            math.ceil(deficit/chunk_bytes),
            lambda key: self.manager.delete([key], force=False).get(key) == L1Error.SUCCESS,
            chunk_tokens=self.hasher.chunk_size)
        deleted = result["deleted"]
        self.registry.record_deleted([identity(key) for key in deleted])
        self.record("hicache_host_reclaim", trigger=trigger, source_sha=SOURCE_SHA,
            core_blob=CORE_BLOB, component_blob=COMPONENT_BLOB,
            deficit_bytes=deficit, deleted_bytes=len(deleted)*chunk_bytes,
            gpu_full_chunks=len(self.hicache_gpu), **result)
        return deleted

    def prepare_native_write(self, keys, layout_desc, mode):
        """Make exactly the missing allocation space before native STORE.

        Caller holds registry.lock; native deletion/reservation retain their
        own locks. Concurrent readers can refuse deletion, and allocations
        can still lose a race, so the native reservation remains authoritative.
        Ordinary demand-LRU uses the SAME hook as the replica variants.
        """
        from lmcache.v1.distributed.error import L1Error
        started = time.perf_counter()
        if mode == "update":
            return None
        object_bytes = sum(math.prod(shape)*dtype.itemsize
                           for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
        if object_bytes != 256*replica_bytes_per_token() or any(not replica_model_supported(key.model_name) for key in keys):
            raise ValueError("Unexpected native allocation layout in replica prototype")
        with self.manager._lock:
            rows = list(self.manager._objects.items())
        native_present = {key for key, _ in rows}
        needed = sum(key not in native_present for key in keys)*object_bytes
        used, capacity = self.manager.get_memory_usage()
        deficit = max(0, needed-(capacity-used))
        if not deficit:
            return None
        pending = {key for group in self.repairs.values() for key in group}
        wanted = set(keys)
        protected_bytes = sum(entry.memory_obj.get_size() for key, entry in rows
                              if self.registry.protected(identity(key)))
        deleted = []
        duplicate_only = bool(getattr(self, "duplicate_admission", False))
        demand_zero = bool(getattr(self, "demand_zero_reclaim", False))
        demand_lru = bool(getattr(self, "demand_lru_reclaim", False))
        demand_reclaim_candidates = 0
        demand_reclaim_selected = 0
        if getattr(self, "hicache_reclaim", False):
            deleted = self.hicache_make_room(rows, wanted, deficit, object_bytes, trigger="store")
        elif (self.allocation_reclaim or duplicate_only or demand_zero or demand_lru) and self.eviction_policy is not None:
            with self.eviction_policy._lock:
                ordered = list(self.eviction_policy._order)
            native = {identity(key): key for key in ordered}
            eligible = lambda candidate: native[candidate] not in wanted \
                and candidate not in pending and self.manager.is_key_evictable(native[candidate]) \
                and not self.cpu_service_ledger.protected(candidate.base(), time.monotonic()) \
                and (not duplicate_only or self.registry.protected(candidate))
            if duplicate_only:
                demand_reclaim_candidates = sum(eligible(candidate) for candidate in native)
            if demand_zero:
                candidates = self.registry.history.select(
                    [candidate for candidate in native if eligible(candidate)],
                    math.ceil(deficit/object_bytes), [identity(key) for key, _ in rows],
                    self.registry.grants, "history-cover", zero_loss_only=True,
                    shelter_arrived_tail=getattr(self, "shelter_arrived_tail", False),
                    reference_first=getattr(self, "reference_first_reclaim", False))
            elif demand_lru:
                candidates = [candidate for candidate in native if eligible(candidate)][:math.ceil(deficit/object_bytes)]
                self.registry.history.last_decisions = []
            else:
                candidates = self.registry.select(
                    list(native), math.ceil(deficit/object_bytes),
                    eligible=eligible,
                    resident_keys=[identity(key) for key, _ in rows])
            demand_reclaim_selected = len(candidates) if duplicate_only else 0
            # Do not discard useful cache if this cannot cover the whole
            # batch. The native allocator otherwise rolls back partial alloc.
            if duplicate_only or demand_zero or demand_lru or len(candidates)*object_bytes >= deficit:
                result = self.manager.delete([native[key] for key in candidates], force=False)
                deleted = [key for key, status in result.items() if status == L1Error.SUCCESS]
                self.registry.record_deleted([identity(key) for key in deleted])
        deleted_protected_chunks = sum(self.registry.protected(identity(key)) for key in deleted)
        self.publish_admission_telemetry(deleted=len(deleted))
        if self.registry.mode in ("history-tail", "history-value", "history-cover", "history-cover-wide", "history-horizon", "history-active"):
            self.record("replica_history_decision", trigger="allocation",
                        decisions=self.registry.history.last_decisions if deleted else [],
                        successful_deleted_chunks=len(deleted))
        return dict(requested_new_bytes=needed, deficit_bytes=deficit,
                    cpu_used_before_bytes=used, cpu_capacity_bytes=capacity,
                    protected_duplicate_bytes=protected_bytes,
                    deleted_bytes=len(deleted)*object_bytes,
                    protected_deleted_bytes=deleted_protected_chunks*object_bytes,
                    demand_duplicate_reclaim=duplicate_only,
                    demand_reclaim_deficit_chunks=math.ceil(deficit/object_bytes) if duplicate_only else 0,
                    demand_reclaim_candidates=demand_reclaim_candidates,
                    demand_reclaim_selected=demand_reclaim_selected,
                    demand_reclaim_deleted_chunks=deleted_protected_chunks if duplicate_only else 0,
                    demand_zero_reclaim=demand_zero,
                    demand_lru_reclaim=demand_lru,
                    allocation_reclaim=self.allocation_reclaim,
                    duplicate_admission=duplicate_only,
                    controller_s=time.perf_counter()-started)

    def pressure_write_candidates(self, keys, layout_desc, mode):
        """Under CPU pressure, fund useful intervals before redundant STORE.

        Keep native existing objects, unknown objects, uncovered chunks, and
        one missing duplicate nearest each GPU/CPU boundary. Extra missing
        GPU duplicates may use pre-existing free slots, but may not evict a
        different program's backup just to re-create a second copy. Native
        reservation/copy fences remain authoritative. This is a separate
        admission experiment; baseline v20 is unchanged.
        """
        if mode == "update" or not getattr(self, "pressure_write_bridge", False):
            return keys, []
        size = sum(math.prod(shape)*dtype.itemsize
                   for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
        with self.manager._lock:
            present = set(self.manager._objects)
        used, capacity = self.manager.get_memory_usage()
        absent = [key for key in keys if key not in present]
        free_slots = max(0, (capacity-used)//size)
        if len(absent) <= free_slots:
            return keys, []
        now = time.monotonic()
        history = self.registry.history
        known = {key.base(): (program, i, row)
                 for program, row in history.programs.items()
                 for i, key in enumerate(row.keys)}
        duplicates = {}
        essential = set()
        for key in absent:
            match = known.get(identity(key).base())
            if match is None:
                essential.add(key)
                continue
            program, index, row = match
            grant = self.registry.grants.get(program)
            certified = min(len(row.keys), len(grant.keys)) if grant else 0
            if grant:
                certified = next((i for i in range(certified)
                    if grant.keys[i].base() != row.keys[i].base()), certified)
            observed = row.gpu_chunks if 0 <= now-row.gpu_observed_at <= .15 else 0
            if index >= max(certified, observed):
                essential.add(key)
            else:
                duplicates.setdefault(program, []).append((index, key))
        extras = []
        for group in duplicates.values():
            group.sort(reverse=True, key=lambda item: item[0])
            essential.add(group[0][1])
            extras.extend(key for _, key in group[1:])
        # Available space is consumed by essential writes first. Only the
        # pre-existing surplus may fund extra duplicates; no projected freed
        # slots are used to justify them.
        essential.update(extras[:max(0, free_slots-len(essential))])
        kept = [key for key in keys if key in present or key in essential]
        skipped = [key for key in keys if key not in present and key not in essential]
        if skipped:
            self.record("replica_pressure_write_budget", skipped_chunks=len(skipped),
                        free_slots=free_slots, essential_chunks=len(essential),
                        missing_chunks=len(absent), bridge_chunks_per_program=1)
        return kept, skipped

    def reserve_partial(self, original, keys, is_temporary, layout_desc, mode):
        """Admit a bounded subset through native allocation and CUDA STORE.

        rc4 rolls back the whole NEW allocation batch when it cannot fit.
        STORE already accepts missing destinations and copies only actual
        reservations. Keep earliest absent chunks; bypass later cache writes,
        never model input or an existing read/write lock. Native allocation is
        still authoritative if free-byte estimates lose a race or fragment.
        """
        from lmcache.v1.distributed.error import L1Error
        if mode == "update" or not getattr(self, "partial_admission", False):
            return original(keys, is_temporary, layout_desc, mode), 0
        size = sum(math.prod(shape)*dtype.itemsize
                   for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
        if size != 256*replica_bytes_per_token():
            raise ValueError("Unexpected partial admission object size")
        with self.manager._lock:
            present = set(self.manager._objects)
        used, capacity = self.manager.get_memory_usage()
        allowance = max(0, (capacity-used)//size)
        absent = [key for key in keys if key not in present]
        ranked = (self.registry.history.admission_order(
            [identity(key) for key in absent], self.registry.grants)
            if getattr(self, "complementary_admission", False) else
            [identity(key) for key in absent])
        selected = set(ranked[:allowance])
        admitted, temporary, skipped = [], [], []
        for key, is_temp in zip(keys, is_temporary, strict=True):
            if key in present:
                admitted.append(key)
                temporary.append(is_temp)
            elif identity(key) in selected:
                admitted.append(key)
                temporary.append(is_temp)
            else:
                skipped.append(key)
        result = original(admitted, temporary, layout_desc, mode) if admitted else {}
        if skipped:
            now = time.monotonic()
            known = {k.base(): (program, i, row)
                     for program, row in self.registry.history.programs.items()
                     for i, k in enumerate(row.keys)}
            details = []
            for key in keys:
                if key in present:
                    continue
                match = known.get(identity(key).base())
                if match is None:
                    details.append(dict(program=None, admitted=key in admitted))
                    continue
                program, index, row = match
                grant = self.registry.grants.get(program)
                certified = len(grant.keys) if grant else 0
                observed = row.gpu_chunks if 0 <= now-row.gpu_observed_at <= .15 else 0
                details.append(dict(program=program, chunk=index, state=row.state,
                    gpu_chunks=max(certified, observed), certified_gpu_chunks=certified,
                    admitted=key in admitted))
            self.record("replica_partial_admission", candidates=details,
                        complementary_admission=getattr(self, "complementary_admission", False),
                        admitted_new_chunks=sum(d["admitted"] for d in details),
                        skipped_new_chunks=len(skipped))
        # This is a deliberate cache-admission bypass, not an allocator OOM
        # and not a claim that the skipped data has been stored.
        result.update({key: (L1Error.KEY_NOT_WRITABLE, None) for key in skipped})
        return result, len(skipped)

    def execute(self, controller, action):
        from lmcache.v1.distributed.error import L1Error

        # The same lock serializes a GPU revoke with the final native deletion,
        # not merely with a previously sampled victim list.
        with self.registry.lock:
            if (getattr(self, "demand_zero_reclaim", False) or getattr(self, "demand_lru_reclaim", False)
                    or getattr(self, "hicache_reclaim", False)):
                self.record("replica_periodic_skipped", candidate_chunks=len(action.keys),
                            reason="capacity-is-reclaimed-only-by-store-demand")
                return
            start = time.perf_counter()
            policy = controller._eviction_policy
            with policy._lock:
                ordered = list(policy._order)
            deadlines = self.registry.deadlines()
            native = {identity(key): key for key in ordered}
            with self.manager._lock:
                sizes = {key: entry.memory_obj.get_size()
                         for key, entry in self.manager._objects.items()}
            selected = self.registry.select(
                list(native), len(action.keys),
                eligible=lambda key: self.manager.is_key_evictable(native[key])
                and not self.cpu_service_ledger.protected(key.base(), time.monotonic())
                and key not in {item for group in getattr(self, "repairs", {}).values()
                                for item in group},
                resident_keys=[identity(key) for key in sizes])
            before = self.snapshot()
            original = list(action.keys)
            # Observe mode executes precisely the native action, including
            # the original eligibility snapshot. Failed deletions are logged.
            keys = original if self.registry.mode == "observe" else [native[key] for key in selected]
            results = self.manager.delete(keys, force=False)
            deleted = [key for key, result in results.items() if result == L1Error.SUCCESS]
            protected = [key for key in deleted if self.registry.protected(identity(key))]
            self.registry.record_deleted([identity(key) for key in deleted])
            self.record("replica_eviction", **before,
                        native_candidates=original, selected=keys, deleted=deleted,
                        deleted_bytes=sum(sizes.get(key, 0) for key in deleted),
                        protected_deleted_bytes=sum(sizes.get(key, 0) for key in protected),
                        unprotected_deleted_bytes=sum(sizes.get(key, 0) for key in deleted
                                                      if identity(key) not in deadlines),
                        native_unprotected_candidates=sum(not self.registry.protected(identity(key)) for key in original),
                        refused=len(results)-len(deleted),
                        controller_s=time.perf_counter()-start,
                        unique_copy_verified=False)
            if self.registry.mode in ("history-tail", "history-value", "history-cover", "history-cover-wide", "history-horizon", "history-active"):
                self.record("replica_history_decision", trigger="periodic", decisions=self.registry.history.last_decisions)


def install():
    """Install wrappers in our helper process; leave site-packages untouched."""
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.storage_controllers.eviction_controller import L1EvictionController
    from lmcache.v1.multiprocess.token_hasher import TokenHasher
    from lmcache.v1.multiprocess.modules.lookup import LookupModule
    from kairos._runtime.compat.range_helper import install_range_lookup

    install_range_lookup(LookupModule)

    original_init = L1Manager.__init__
    original_execute = L1EvictionController.execute_eviction_action
    original_controller_init = L1EvictionController.__init__
    original_reserve = L1Manager.reserve_write

    def initialize(manager, *args, **kwargs):
        original_init(manager, *args, **kwargs)
        manager._continuum_replica = ReplicaRuntime(
            manager, os.environ["CONTINUUM_REPLICA_SOCKET"],
            os.environ["CONTINUUM_REPLICA_LOG"], TokenHasher(chunk_size=256,
                hash_algorithm="blake3"), os.environ.get("CONTINUUM_REPLICA_MODE", "observe"))

    def execute(controller, action):
        runtime = getattr(controller._l1_manager, "_continuum_replica", None)
        if runtime is None:
            return original_execute(controller, action)
        runtime.execute(controller, action)

    def controller_init(controller, *args, **kwargs):
        original_controller_init(controller, *args, **kwargs)
        runtime = getattr(controller._l1_manager, "_continuum_replica", None)
        if runtime is not None:
            runtime.eviction_policy = controller._eviction_policy

    def reserve(manager, keys, is_temporary, layout_desc, mode="all"):
        from lmcache.v1.distributed.error import L1Error
        runtime = getattr(manager, "_continuum_replica", None)
        if runtime is None:
            return original_reserve(manager, keys, is_temporary, layout_desc, mode)
        with runtime.registry.lock:
            budgeted, pressure_skipped = runtime.pressure_write_candidates(keys, layout_desc, mode)
            temporary_by_key = dict(zip(keys, is_temporary, strict=True))
            budgeted_temporary = [temporary_by_key[key] for key in budgeted]
            observation = runtime.prepare_native_write(budgeted, layout_desc, mode)
            result, skipped = runtime.reserve_partial(
                lambda ks, ts, layout, write_mode: original_reserve(manager, ks, ts, layout, write_mode),
                budgeted, budgeted_temporary, layout_desc, mode)
            result.update({key: (L1Error.KEY_NOT_WRITABLE, None) for key in pressure_skipped})
            if observation is not None:
                runtime.record("replica_native_allocation", **observation,
                    policy_skipped_new_chunks=skipped+len(pressure_skipped),
                    failed_new_chunks=sum(status == L1Error.OUT_OF_MEMORY for status, _ in result.values()),
                    successful_chunks=sum(status == L1Error.SUCCESS for status, _ in result.values()))
            runtime.publish_admission_telemetry()
            return result

    original_close = L1Manager.close

    def close(manager):
        runtime = getattr(manager, "_continuum_replica", None)
        if runtime is not None:
            runtime.control.close()
        return original_close(manager)

    L1Manager.__init__ = initialize
    L1Manager.close = close
    L1Manager.reserve_write = reserve
    L1EvictionController.__init__ = controller_init
    L1EvictionController.execute_eviction_action = execute
