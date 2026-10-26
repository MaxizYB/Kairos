"""Kairos replica connector implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

from kairos._runtime.compat.replica import replica_model_supported, rpc


@dataclass
class Repair:
    ident: str
    tokens: list[int]
    block_ids: list[int]
    indices: list[int]
    salt: str
    model: str
    program: str


@dataclass
class RepairWorkerMetadata(LMCacheMPWorkerMetadata):
    repaired: dict[str, bool]
    repair_stats: dict[str, dict] = field(default_factory=dict)

    def aggregate(self, other):
        if not isinstance(other, RepairWorkerMetadata):
            raise TypeError("Unexpected replica worker metadata")
        counts = dict(self.completed_store_requests)
        for key, value in other.completed_store_requests.items():
            counts[key] = counts.get(key, 0)+value
        repaired = dict(self.repaired)
        for key, value in other.repaired.items():
            repaired[key] = repaired.get(key, True) and value
        stats = dict(self.repair_stats)
        stats.update(other.repair_stats)
        return RepairWorkerMetadata(counts, repaired, stats)


@dataclass
class RepairRequestMetadata(LMCacheMPRequestMetadata):
    prefix_stop_on_failure: bool = False


@dataclass
class PendingRepair:
    items: list
    active: object | None = None
    submitted: float = 0
    commit_started: float | None = None
    copied: bool = False
    complete: bool = True
    worker_control_s: float = 0
    started: float = field(default_factory=time.monotonic)


class ReplicaConnector(LMCacheMPConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.repair_enabled = False
        self.bounded_repair = False
        self.repair_on_ttl_expiry = True
        self.replica_mode = "observe"
        self.active_certificates = False
        self.async_repair = False
        self._repair_queue = []
        self._repair_refs = {}
        self._repair_completed = {}
        self._repair_stats = {}
        self._worker_repairs = {}
        self.range_lookup = False
        self._coverage_logged = set()
        self._repair_serial = 0
        self.replica_emit = lambda *args, **kwargs: None
        self._replica_socket = os.environ.get("CONTINUUM_REPLICA_SOCKET")

    def set_replica_mode(self, canonical):
        if self._repair_refs or self._repair_queue or self._worker_repairs:
            raise RuntimeError("Cannot switch with pending GPU backup repairs")
        self.repair_enabled = canonical in (
            "replica-coord", "replica-repair", "replica-bounded", "replica-demand",
            "replica-async", "replica-repair-async", "replica-range", "replica-frontier", "replica-loan",
            "replica-ledger", "replica-span", "replica-history-repair", "replica-active-commit")
        self.bounded_repair = canonical in (
            "replica-bounded", "replica-demand", "replica-async", "replica-repair-async",
            "replica-range", "replica-frontier", "replica-loan", "replica-ledger", "replica-span", "replica-history-repair", "replica-active-commit")
        self.async_repair = canonical in (
            "replica-async", "replica-repair-async", "replica-range", "replica-frontier", "replica-loan",
            "replica-ledger", "replica-span", "replica-history-repair", "replica-active-commit")
        self.range_lookup = canonical in ("vllm-range", "zero-range", "continuum-range", "replica-range", "replica-frontier", "replica-loan",
                                         "replica-ledger", "replica-span", "replica-history-tail",
                                         "replica-history-value", "replica-history-repair", "replica-history-cover",
                                         "replica-active", "replica-active-commit",
                                         "replica-gpu-duplicate", "replica-joint", "replica-joint-tail",
                                         "replica-gpu-tail", "replica-joint-boundary", "replica-joint-fast",
                                         "replica-joint-incremental")
        self.active_certificates = canonical in ("replica-active", "replica-active-commit")
        self._coverage_logged.clear()
        self.repair_on_ttl_expiry = canonical != "replica-demand"
        self.replica_mode = canonical

    def on_new_request(self, request):
        if self.range_lookup:
            # Wait for the scheduler's actual APC match before selecting the
            # CPU range. Eager lookup from zero cannot see a cached suffix.
            return
        return super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestState
        from vllm.v1.request import RequestStatus
        from kairos._runtime.compat.range_lookup import RangeLookupAdapter
        adapter = self.scheduler_adapter
        if not isinstance(adapter, RangeLookupAdapter):
            from lmcache.integration.vllm.vllm_multi_process_adapter import send_lmcache_request
            from lmcache.v1.multiprocess.protocol import RequestType
            adapter = self.scheduler_adapter = RangeLookupAdapter(
                adapter, send_lmcache_request, RequestType.LOOKUP)
        tracker = self._get_or_create_request_tracker(request)
        if self.range_lookup and request.status != RequestStatus.PREEMPTED:
            start = num_computed_tokens//adapter.lmcache_tokens_per_chunk*adapter.lmcache_tokens_per_chunk
            previous = adapter.starts.get(request.request_id, 0)
            if previous > start:
                # APC pages were recycled while the request awaited lookup.
                # Release the old suffix locks and repeat from the new match;
                # never advertise a CPU suffix across a missing GPU prefix.
                old_end = adapter.check_lookup_result(request.request_id)
                if old_end is None:
                    return None, True
                if old_end:
                    adapter.free_lookup_locks(tracker.get_token_ids(), previous, old_end,
                        request.request_id, cache_salt=tracker.cache_salt,
                        request_configs=tracker.request_configs)
                adapter.cleanup_lookup_result(request.request_id)
                adapter.starts.pop(request.request_id, None)
            if request.request_id not in adapter._pending_lookups:
                adapter.starts[request.request_id] = start
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if (result[0] == 0 and request.status != RequestStatus.PREEMPTED
                and tracker.state == LMCacheMPRequestState.PREFETCHING):
            # Native rc4 returns early on CPU miss before recording an APC
            # hit. Its later STORE accounting then omits the already-computed
            # history and may skip new suffix KV. Fix this for EVERY method,
            # including the ordinary baseline; it is a compatibility fix.
            tracker.num_vllm_hit_tokens = num_computed_tokens//self._hit_alignment_tokens*self._hit_alignment_tokens
        if request.request_id not in self._coverage_logged and result[0] is not None:
            self._coverage_logged.add(request.request_id)
            coverage = rpc(self._replica_socket, dict(
                op="coverage", tokens=tracker.get_token_ids(), model=adapter.model_name,
                salt=tracker.cache_salt, kv_rank=16777472,
                local_tokens=num_computed_tokens), timeout=5)
            # Admission ports need the host-only prefix, not the joint
            # GPU-prefix/CPU-suffix endpoint in num_lmcache_hit_tokens.
            if "cpu_prefix_tokens" in coverage:
                tracker._continuum_cpu_prefix_tokens = coverage["cpu_prefix_tokens"]
            self.replica_emit("replica_coverage", request=request.request_id,
                              **coverage)
        return result

    def queue_backup_repair(self, pin, indices, block_size):
        if not indices:
            return False
        if block_size != 16 or not replica_model_supported(self.scheduler_adapter.model_name):
            raise ValueError("Repair prototype is restricted to Qwen3-8B/16-token blocks")
        # A repair is a bounded opportunity, not a second GPU reservation.
        # Holding every missing chunk of a long program can consume the same
        # blocks that the pressure fallback is trying to free.  Copy the
        # earliest missing contiguous prefix first; later chunks remain a
        # normal recomputation fallback.  Also cap aggregate in-flight source
        # references so several simultaneous fallbacks cannot deadlock the
        # allocator.
        max_chunks = (max(1, int(os.environ.get("CONTINUUM_REPAIR_MAX_CHUNKS", "8")))
                      if self.bounded_repair else len(indices))
        held_limit = (max(16, int(os.environ.get("CONTINUUM_REPAIR_HELD_BLOCKS", "256")))
                      if self.bounded_repair else None)
        if len(indices) > max_chunks:
            indices = list(indices[:max_chunks])
            self.replica_emit("backup_repair_capped", program=pin.program,
                              chunks=len(indices), max_chunks=max_chunks)
        self._repair_serial += 1
        ident = f"replica-repair-{self._repair_serial}"
        # Hold only pages whose exact original contents will be copied.
        ids = [block.block_id for index in indices
               for block in pin.blocks[index*16:(index+1)*16]]
        if len(ids) != len(indices)*16 or len(set(ids)) != len(ids):
            raise RuntimeError("Repair source does not cover distinct complete chunks")
        held_now = sum(len(source_ids) for source_ids in self._repair_refs.values())
        if held_limit is not None and held_now + len(ids) > held_limit:
            self.replica_emit("backup_repair_skipped", program=pin.program,
                              chunks=len(indices), held_blocks=held_now,
                              held_limit=held_limit, reason="pending_budget")
            return False
        self._gpu_block_pool.touch([self._gpu_block_pool.blocks[bid] for bid in ids])
        self._repair_refs[ident] = ids
        self._repair_queue.append(Repair(ident, list(pin.token_ids),
                                        [block.block_id for block in pin.blocks],
                                        list(indices), pin.cache_salt, pin.model,
                                        pin.program))
        self.replica_emit("backup_repair_queued", repair=ident,
                          program=pin.program, chunks=len(indices),
                          held_gpu_blocks=len(ids), max_chunks=max_chunks)
        return True

    def has_pending_repairs(self):
        return bool(self._repair_queue or self._repair_refs)

    def has_pending_push_work(self):
        return super().has_pending_push_work() or self.has_pending_repairs()

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        repairs, self._repair_queue = self._repair_queue, []
        for repair in repairs:
            for offset in range(0, len(repair.indices), 4):
                indices = repair.indices[offset:offset+4]
                groups = []
                for index in indices:
                    if not groups or index != groups[-1][-1]+1:
                        groups.append([index])
                    else:
                        groups[-1].append(index)
                for group in groups:
                    start, end = group[0]*256, (group[-1]+1)*256
                    ident = f"{repair.ident}-{group[0]}"
                    op = LoadStoreOp(
                        token_ids=repair.tokens,
                        block_ids=[repair.block_ids[start//16:end//16]],
                        start=start, end=end)
                    metadata.add_request_metadata(RepairRequestMetadata(
                        request_id=ident, direction="STORE", op=op,
                        cache_salt=repair.salt,
                        prefix_stop_on_failure=self.bounded_repair))
        # Scheduler and worker connectors are separate processes. Send the
        # mode with each iteration rather than relying on a worker env var.
        metadata.replica_async_repair = self.async_repair
        return metadata

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        repairs = [item for item in metadata.requests
                   if item.direction == "STORE"
                   and item.request_id.startswith("replica-repair-")]
        # The normal connector submits every STORE in metadata.  Remove our
        # private repair records before entering that path, then submit them
        # below with the same worker adapter/fence exactly once.
        if repairs:
            metadata.requests = [item for item in metadata.requests
                                 if item not in repairs]
        super().start_load_kv(forward_context, **kwargs)
        if getattr(metadata, "replica_async_repair", False):
            for item in repairs:
                root = item.request_id.rsplit("-", 1)[0]
                self._worker_repairs.setdefault(root, PendingRepair([])).items.append(item)
            self._progress_async_repairs()
            return
        complete_by_root = {}
        sync_control = {}
        for item in repairs:
            ident = item.request_id
            root = ident.rsplit("-", 1)[0]
            if (item.prefix_stop_on_failure
                    and complete_by_root.get(root) is False):
                # Later suffixes cannot extend the native recoverable prefix
                # across an earlier failed chunk. Do not spend transfer time
                # or CPU space copying them in the bounded variant.
                continue
            complete = False
            control_start = time.monotonic()
            indices = list(range(item.op.start//256, item.op.end//256))
            payload = dict(repair=ident, tokens=item.op.token_ids,
                           indices=indices, salt=item.cache_salt,
                           model=self.worker_adapter.model_name,
                           kv_rank=16777472)
            room = rpc(self._replica_socket,
                       dict(payload, op="repair_room"), timeout=5)
            if room["allowed"]:
                event = self.worker_adapter.create_recorded_event()
                self.worker_adapter.submit_store_request(
                    ident, item.op, event, cache_salt=item.cache_salt,
                    request_configs=item.request_configs)
                future = self.worker_adapter.store_futures.get(ident)
                if future is None:
                    raise RuntimeError("Backup repair did not submit a transfer")
                # A timeout must stop this run. Never acknowledge and release
                # source pages while a transfer might still access them.
                copied = bool(future.result(timeout=30))
                self.worker_adapter.store_futures.pop(ident, None)
                self.worker_adapter.store_events.pop(ident, None)
                commit_deadline = time.monotonic() + 2
                while True:
                    checked = rpc(self._replica_socket,
                                  dict(op="repair_done", repair=ident), timeout=5)
                    if not checked.get("pending", False):
                        break
                    if time.monotonic() >= commit_deadline:
                        raise TimeoutError("Backup repair CPU commit is still pending")
                    time.sleep(.002)
                complete = copied and checked["stored_complete"]
            complete_by_root[root] = complete_by_root.get(root, True) and complete
            sync_control[root] = sync_control.get(root, 0) + time.monotonic() - control_start
        self._repair_completed.update(complete_by_root)
        self._repair_stats.update({root: dict(async_repair=False, worker_control_s=seconds)
                                   for root, seconds in sync_control.items()})

    def _progress_async_repairs(self):
        """Poll finite transfers; never wait for DMA or CPU publication.

        Private futures are removed from the native adapter's completion
        namespace so they cannot be mistaken for finished model requests.
        The scheduler holds source refs until our own host-commit ack arrives.
        """
        for root, state in list(self._worker_repairs.items()):
            control_start = time.monotonic()
            if state.active is not None:
                item, future, event = state.active
                if state.commit_started is None:
                    if not self.worker_adapter.is_healthy:
                        raise RuntimeError("LMCache unavailable with live repair source references")
                    if not future.query():
                        if time.monotonic() - state.submitted > 30:
                            raise TimeoutError("Async backup repair copy is still pending")
                        continue
                    state.copied = bool(future.result(timeout=0))
                    state.commit_started = time.monotonic()
                checked = rpc(self._replica_socket,
                              dict(op="repair_done", repair=item.request_id), timeout=5)
                if checked.get("pending", False):
                    if time.monotonic() - state.commit_started > 2:
                        raise TimeoutError("Async backup repair CPU commit is still pending")
                    state.worker_control_s += time.monotonic() - control_start
                    continue
                complete = state.copied and checked["stored_complete"]
                state.complete = state.complete and complete
                state.active = None
                state.commit_started = None
                if item.prefix_stop_on_failure and not complete:
                    state.items.clear()
            if state.items:
                item = state.items.pop(0)
                payload = dict(repair=item.request_id, tokens=item.op.token_ids,
                               indices=list(range(item.op.start//256, item.op.end//256)),
                               salt=item.cache_salt, model=self.worker_adapter.model_name,
                               kv_rank=16777472)
                room = rpc(self._replica_socket, dict(payload, op="repair_room"), timeout=5)
                if room["allowed"]:
                    event = self.worker_adapter.create_recorded_event()
                    self.worker_adapter.submit_store_request(
                        item.request_id, item.op, event, cache_salt=item.cache_salt,
                        request_configs=item.request_configs)
                    future = self.worker_adapter.store_futures.pop(item.request_id, None)
                    self.worker_adapter.store_events.pop(item.request_id, None)
                    if future is None:
                        raise RuntimeError("Async backup repair did not submit a transfer")
                    # Retain both future and IPC event until host publication.
                    state.active = (item, future, event)
                    state.submitted = time.monotonic()
                else:
                    state.complete = False
                    if item.prefix_stop_on_failure:
                        state.items.clear()
            state.worker_control_s += time.monotonic() - control_start
            if state.active is None and not state.items:
                self._repair_completed[root] = state.complete
                self._repair_stats[root] = dict(
                    async_repair=True, worker_control_s=state.worker_control_s,
                    lifetime_s=time.monotonic() - state.started)
                del self._worker_repairs[root]

    def get_finished(self, finished_req_ids):
        self._progress_async_repairs()
        return super().get_finished(finished_req_ids)

    def build_connector_worker_meta(self):
        normal = super().build_connector_worker_meta()
        repaired = self._repair_completed
        self._repair_completed = {}
        stats, self._repair_stats = self._repair_stats, {}
        if not repaired and normal is None:
            return None
        return RepairWorkerMetadata(
            dict(normal.completed_store_requests) if normal else {}, repaired, stats)

    def update_connector_output(self, output):
        super().update_connector_output(output)
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, RepairWorkerMetadata):
            return
        for ident, complete in metadata.repaired.items():
            root = ident
            ids = self._repair_refs.pop(root, None)
            if ids is None:
                raise RuntimeError("Duplicate or unowned repair completion")
            self._gpu_block_pool.free_blocks(
                [self._gpu_block_pool.blocks[bid] for bid in ids])
            self.replica_emit("backup_repair_released", repair=ident,
                              stored_complete=complete,
                              released_gpu_blocks=len(ids),
                              **metadata.repair_stats.get(ident, {}))
