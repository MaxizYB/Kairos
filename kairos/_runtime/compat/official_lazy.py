"""Kairos official lazy implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import re
import sys

from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata, LMCacheMPRequestState, LMCacheMPRequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

from kairos._runtime.compat.replica_connector import ReplicaConnector, RepairWorkerMetadata
from kairos._runtime.compat.scheduler import PressureControlledScheduler

UPSTREAM_SHA = "68a9c6b5667ff6d9493530984bd2e40fb3941451"
_manager_class = None


def load_official_manager():
    """Load exact upstream modules only inside this private Python process."""
    global _manager_class
    if _manager_class is not None:
        return _manager_class
    root = Path(__file__).resolve().parents[1] / "vendor/lmcache-lazy-68a9c6b"
    prefix = "lmcache.integration.vllm."
    files = [
        ("lazy_offload_policy.base", "lazy_offload_policy/base.py"),
        ("lazy_offload_policy.eviction_aware", "lazy_offload_policy/eviction_aware.py"),
        ("lazy_offload_policy.fifo", "lazy_offload_policy/fifo.py"),
        ("lazy_offload_state", "lazy_offload_state.py"),
        ("lazy_offload_policy", "lazy_offload_policy/__init__.py"),
        ("lazy_offload_manager", "lazy_offload_manager.py"),
    ]
    for name, file in files:
        spec = importlib.util.spec_from_file_location(prefix + name, root / file,
            submodule_search_locations=[str(root / "lazy_offload_policy")]
            if file.endswith("__init__.py") else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    _manager_class = sys.modules[prefix + "lazy_offload_manager"].LazyOffloadManager
    return _manager_class


@dataclass
class OfficialWorkerMetadata(RepairWorkerMetadata):
    failed_store_requests: set[str] = field(default_factory=set)

    def aggregate(self, other):
        normal = super().aggregate(other)
        return OfficialWorkerMetadata(normal.completed_store_requests, normal.repaired,
            normal.repair_stats, self.failed_store_requests | other.failed_store_requests)


class FailureReceiptFuture:
    """Observe native result once it is read, preserving its future API."""
    def __init__(self, future, request, failed):
        self.future, self.request, self.failed = future, request, failed

    def __getattr__(self, name):
        return getattr(self.future, name)

    def result(self, *args, **kwargs):
        result = self.future.result(*args, **kwargs)
        if not result:
            self.failed.add(self.request)
        return result


class PendingStoreBridge:
    """Old tracker emission routed to upstream add_store_candidate."""
    def __init__(self, connector):
        self.connector = connector

    def bind_gpu_block_pool(self, pool):
        self.connector._official_manager.bind_block_pool(pool)

    def add(self, meta):
        if meta.request_id in getattr(self.connector, "_admission_skipped", ()):
            return
        self.connector._official_manager.add_store_candidate(meta)


class OfficialLazyConnector(ReplicaConnector):
    def __init__(self, config, role, kv_cache_config=None):
        super().__init__(config, role, kv_cache_config)
        self._failed_stores = set()
        self._official_admission = None
        self._admission_skipped = set()
        if role == KVConnectorRole.SCHEDULER:
            self._official_configs = dict(config.kv_transfer_config.kv_connector_extra_config)
            self._official_configs["lmcache.mp.lazy_offload_policy"] = "EVICTION_AWARE"
            cls = load_official_manager()
            self._official_manager = cls(self._official_configs,
                self._group_tokens_per_block, self.scheduler_adapter)
            self._pending_store = PendingStoreBridge(self)

    def reset_official_policy(self, horizon, enabled=True):
        old = self._official_manager
        if old.has_inflight_store_work():
            raise RuntimeError("Cannot reset upstream lazy policy with outstanding copies")
        old.log_final_stats()
        # Buffered metadata owns no references. End its finished sessions
        # before resetting APC/helper state for the next independent replay.
        for request in list(old._requests._slots):
            self.scheduler_adapter.end_session(request)
        self._official_configs["lmcache.mp.lazy_offload_horizon_steps"] = horizon
        self._official_manager = load_official_manager()(self._official_configs,
            self._group_tokens_per_block, self.scheduler_adapter)
        self._official_manager.bind_block_pool(self._gpu_block_pool)
        self.lazy_offload = enabled

    def _process_lazy_offload_store_requests(self, metadata):
        # Old rc4 calls this twice per step. The upstream manager drains once,
        # after BOTH new and cached request candidates have been offered.
        return

    def _apply_actions(self, actions, metadata=None):
        if metadata is not None:
            for item in actions.stores_to_submit:
                metadata.add_request_metadata(item)
        for request in actions.sessions_to_end:
            self.scheduler_adapter.end_session(request)

    def build_connector_meta(self, output):
        metadata = super().build_connector_meta(output)
        if self._official_admission is not None:
            # rc4 MP does not consume lmcache.skip_save. Filter STORE only;
            # lookup, RETRIEVE, compute, decode and scheduling are unchanged.
            metadata.requests[:] = [item for item in metadata.requests
                if item.direction != "STORE" or item.request_id not in self._admission_skipped]
        if self.lazy_offload:
            actions = self._official_manager.on_scheduler_step(output)
            self._apply_actions(actions, metadata)
        metadata.official_lazy_enabled = self.lazy_offload
        return metadata

    def set_official_admission(self, mode, threshold=8):
        from kairos._runtime.compat.official_admission import OfficialAdmission, SOURCE_SHA, SOURCE_BLOB
        import os
        from kairos._runtime.compat.replica import replica_bytes_per_token
        host_gib = float(os.environ.get('CONTINUUM_LMCACHE_CPU_GB', '2'))
        token_bytes = replica_bytes_per_token()
        self._admission_skipped.clear()
        self._official_admission = (OfficialAdmission(mode, host_gib, 256, token_bytes,
            os.environ["CONTINUUM_REPLICA_LOG"] + ".admission.json", threshold) if mode else None)
        self.replica_emit("official_admission_policy", mode=mode,
            source_sha=SOURCE_SHA, source_blob=SOURCE_BLOB, threshold=threshold,
            chunk_tokens=256, bytes_per_token=token_bytes, host_gib=host_gib,
            transport="MP-policy-port", telemetry_s=1., telemetry_max_age_s=5.)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        admission = self._official_admission
        if (admission is not None and result[0] is not None
                and request.request_id not in admission.rule._ea_decided):
            tracker = self.request_trackers[request.request_id]
            extra = request.sampling_params.extra_args or {}
            skip, rows = admission.resolve(request, num_computed_tokens, result,
                tracker._continuum_cpu_prefix_tokens,
                str(extra.get("cont_program_id") or request.cache_salt or request.request_id))
            if skip:
                self._admission_skipped.add(request.request_id)
            for row in rows:
                self.replica_emit("official_admission_decision", request=request.request_id,
                    upstream_event=row.pop("event"), **row)
        return result

    def start_load_kv(self, forward_context, **kwargs):
        enabled = self._get_connector_metadata().official_lazy_enabled
        if self.lazy_offload != enabled:
            if self.worker_adapter.store_futures or self.worker_adapter.retrieve_futures:
                raise RuntimeError("Cannot switch transfer mode with outstanding futures")
            self.lazy_offload = self.worker_adapter.lazy_offload = enabled
        return super().start_load_kv(forward_context, **kwargs)

    def _get_or_create_request_tracker(self, request):
        if not self.lazy_offload:
            return super()._get_or_create_request_tracker(request)
        ident = request.request_id
        if request.status == RequestStatus.PREEMPTED and ident in self.request_trackers:
            if self.request_trackers[ident].state != LMCacheMPRequestState.PREFETCHING:
                self.request_trackers.pop(ident)
                self._official_manager.on_request_reset(ident)
        if ident not in self.request_trackers:
            self._apply_actions(self._official_manager.on_request_arrived(ident))
            self.request_trackers[ident] = LMCacheMPRequestTracker(request)
        return self.request_trackers[ident]

    def request_finished(self, request, block_ids):
        if not self.lazy_offload:
            return super().request_finished(request, block_ids)
        # Upstream lifecycle: generation completion does not end a deferred
        # STORE session. Pins end on receipts; queued operations remain unpinned.
        self._cleanup_request_tracker(request.request_id)
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)
        self._apply_actions(self._official_manager.on_request_finished(request.request_id))
        return False, None

    def update_connector_output(self, output):
        if not self.lazy_offload:
            return super().update_connector_output(output)
        meta = output.kv_connector_worker_meta
        if isinstance(meta, OfficialWorkerMetadata):
            self._apply_actions(self._official_manager.on_store_results(
                meta.failed_store_requests, meta.completed_store_requests))

    def has_pending_push_work(self):
        return (self.lazy_offload and self.role == KVConnectorRole.SCHEDULER
                and self._official_manager.has_inflight_store_work())

    def get_finished(self, finished_req_ids):
        if not self.lazy_offload:
            return super().get_finished(finished_req_ids)
        adapter = self.worker_adapter
        if not adapter.is_healthy:
            self._failed_stores.update(adapter.store_futures)
        for ident, future in list(adapter.store_futures.items()):
            if not isinstance(future, FailureReceiptFuture):
                adapter.store_futures[ident] = FailureReceiptFuture(future, ident, self._failed_stores)
        return super().get_finished(finished_req_ids)

    def build_connector_worker_meta(self):
        if not self.lazy_offload:
            return super().build_connector_worker_meta()
        normal = super().build_connector_worker_meta()
        failed, self._failed_stores = self._failed_stores, set()
        if normal is None and not failed:
            return None
        return OfficialWorkerMetadata(normal.completed_store_requests if normal else {},
            normal.repaired if normal else {}, normal.repair_stats if normal else {}, failed)


class OfficialLazyScheduler(PressureControlledScheduler):
    # Engineering controls, not claims to reproduce a second official system.
    # They give the same native allocator a demand/partial path without any
    # layout scoring, or remove exactly one tier of v21's decisions.
    _FAIR_METHODS = {
        "continuum-hostlru-partial": ("continuum-index", "host-lru"),
        "ea-conditioned-partial": ("continuum-index", "ea-conditioned"),
        "ea-fixed-partial": ("continuum-index", "ea-fixed"),
        "ea-conditioned-k32-partial": ("continuum-index", "ea-conditioned-k32"),
        "v21-cpu-only": ("replica-joint-demand-bridge-admit-partial", "cpu-only"),
        "v21-gpu-only": ("replica-joint-demand-bridge-admit-partial", "gpu-only"),
        # The parent expands public version names before this hook.
        "replica-joint-demand-bridge-admit-partial-cpu-only": (
            "replica-joint-demand-bridge-admit-partial", "cpu-only"),
        "replica-joint-demand-bridge-admit-partial-gpu-only": (
            "replica-joint-demand-bridge-admit-partial", "gpu-only"),
    }

    def normalize_baseline_method(self, value):
        match = re.search(r"-lazy-(default|h5|h10)$", value)
        eager = value.endswith("-eager") or value.startswith("vllm-")
        if not match and not eager:
            raise ValueError("Every method in this matrix must specify an official lazy policy")
        horizon = {"default": 2.5, "h5": 5.0, "h10": 10.0}[match[1] if match else "default"]
        self.connector.reset_official_policy(horizon, enabled=not eager)
        self._emit("official_lazy_policy", source_sha=UPSTREAM_SHA, horizon=horizon,
            max_drain_per_step=64, max_deferral_seconds=0.0, enabled=not eager)
        base = value[:match.start()] if match else value.removesuffix("-eager")
        admission = re.fullmatch(r"ea-(conditioned|fixed)-k([1-9][0-9]*)-partial", base)
        if admission:
            base, self._fair_ablation = "continuum-index", (
                "ea-" + admission[1] + "-k" + admission[2])
        else:
            base, self._fair_ablation = self._FAIR_METHODS.get(base, (base, None))
        return base

    def _replica_call(self, payload):
        mode = getattr(self, "_fair_ablation", None)
        ea = bool(mode and mode.startswith("ea-"))
        if payload.get("op") == "experiment" and (mode in (
                "host-lru", "gpu-only") or ea):
            # Same per-STORE deficit, partial native reservation and copy
            # fences as v21; native LRU order, no complementary ordering,
            # zero-loss filter, boundary duplicate quota, or extra pin.
            payload = dict(payload, mode="observe", allocation_reclaim=False,
                partial_admission=True, complementary_admission=False,
                duplicate_admission=False, demand_zero_reclaim=False,
                pressure_write_bridge=False, demand_lru_reclaim=True,
                shelter_arrived_tail=False, reference_first_reclaim=False)
        if payload.get("op") == "experiment":
            mode = getattr(self, "_fair_ablation", None)
            payload = dict(payload)
            payload.setdefault("official_admission_telemetry", bool(mode and mode.startswith("ea-")))
        return super()._replica_call(payload)

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        mode = getattr(self, "_fair_ablation", None)
        ea = re.fullmatch(r"ea-(conditioned|fixed)(?:-k(\d+))?", mode or "")
        self.connector.set_official_admission(ea[1] if ea else None,
            int(ea[2]) if ea and ea[2] else 8)
        if mode in ("host-lru", "cpu-only") or ea:
            self.gpu_victim_mode = None
            self.publish_gpu_reuse = False
        if getattr(self, "_fair_ablation", None):
            self._emit("official_lazy_ablation", ablation=self._fair_ablation,
                gpu_victim=self.gpu_victim_mode,
                official_policy_unchanged=True)

    def has_unfinished_requests(self):
        return super().has_unfinished_requests() or self.connector.has_pending_push_work()
