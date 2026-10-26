"""Kairos scheduler implementation."""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus
from kairos._runtime.compat.policy import TTLPolicy, select_suffixes
from kairos._runtime.compat.event_model import search_suffixes
from kairos._runtime.compat.retention import choose_retention, candidate_plans, choose_residency
from kairos._runtime.compat.versions import expand_version

from kairos._runtime.compat.replica import rpc as replica_rpc


@dataclass
class Pin:
    program: str
    deadline: float
    started: float
    tool: str
    tokens: int
    blocks: list[Any] = field(default_factory=list)
    lookup_hashes: list[Any] = field(default_factory=list)
    token_ids: tuple[int, ...] = ()
    model: str = ""
    cache_salt: str = ""


class ContinuumScheduler(Scheduler):
    """Continuum port with explicit tool metadata and safe reference handover."""
    controller = "whole"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("Continuum requires one full-attention KV group")
        if (self.scheduler_config.async_scheduling and not
                getattr(self, 'native_async_prefix_execution', False)) or self.use_pp:
            raise ValueError("Continuum requires synchronous TP=1/PP=1")
        if not self.cache_config.enable_prefix_caching:
            raise ValueError("Continuum requires prefix caching")
        self._validate_speculative_config(self.vllm_config.speculative_config)
        self.pins: dict[str, Pin] = {}
        self.continuity_leases: dict[str, float] = {}
        self.priority_lease_mode = False
        self.growth_guard = False
        # A cohort interval is an execution-orchestration ablation.  It does
        # not alter TTL, physical KV references, or victim selection: while a
        # decode-only batch is making progress, newly waiting prefills can be
        # held for a bounded number of scheduler steps and then admitted as a
        # wave.  The default interval=1 is the Continuum behavior.
        self.cohort_interval = 1
        self.cohort_ticks = 0
        self.cohort_adaptive = False
        self.order: dict[str, int] = {}
        self.departures: dict[str, tuple[float, str, int]] = {}
        self.waiting_samples = deque(maxlen=256)
        self.logged_admissions: set[str] = set()
        self.logged_inputs: set[str] = set()
        self.experiment = None
        self.beta = [0.001, 0.00001, 1e-10, 0.0001]
        self.profile = {}
        profile_path = os.environ.get("CONTINUUM_PROFILE")
        if profile_path:
            self.profile = json.loads(Path(profile_path).read_text())
            self.beta = self.profile.get("beta", self.beta)
        self.policy_model = TTLPolicy(
            mode=os.environ.get("CONTINUUM_TTL_MODE", "paper"),
            fixed=float(os.environ.get("CONTINUUM_TTL_SECONDS", "2")),
            k=int(os.environ.get("CONTINUUM_HISTORY_K", "100")),
            history=self.profile.get("tool_history", {}),
        )
        # The paper's memoryfulness factor is workload-dependent.  A measured
        # profile may provide it; the explicit environment override is useful
        # for ablations.  Keep the paper's fully-memoryful default only when
        # no profile is available.
        self.eta = float(os.environ.get(
            "CONTINUUM_ETA", str(self.profile.get("eta", "1"))))
        self.hysteresis = int(os.environ.get("CONTINUUM_HYSTERESIS_BLOCKS", "32"))
        self.logfile = None
        self.log_timing = os.environ.get("CONTINUUM_LOG_TIMING") == "1"
        self._previous_log_write = None
        log_path = os.environ.get("CONTINUUM_LOG_PATH")
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.logfile = open(log_path, "a", buffering=1)
        self.replica_socket = os.environ.get("CONTINUUM_REPLICA_SOCKET")
        self.replica_model = os.environ.get("CONTINUUM_REPLICA_MODEL", "")
        self.history_gpu_hashes = {}
        self.active_replicas = set()
        self.gpu_victim_mode = None
        self.grouped_gpu_recovery = False
        self.incremental_gpu = False
        self.gpu_index = None
        self.paper_engine_compat = False
        self.strict_pin_fallback = False
        self._next_gpu_history = 0.0
        if self.replica_socket and hasattr(self.connector, "replica_emit"):
            self.connector.replica_emit = self._emit
        self.original_allocate = self.kv_cache_manager.allocate_slots
        self.kv_cache_manager.allocate_slots = self._allocate
        self.original_get_new_blocks = self.kv_cache_manager.block_pool.get_new_blocks
        self.kv_cache_manager.block_pool.get_new_blocks = self._get_new_blocks
        # Observe the native hash map's actual insert/pop operations. These
        # wrappers keep all native behavior, including duplicate hash aliases.
        self._attach_gpu_index_hooks()
        self._emit("init", controller=self.controller, mode=self.policy_model.mode,
                   capacity=self._capacity(), beta=self.beta, version="0.28-port-v2")

    def _validate_speculative_config(self, config):
        if config is not None:
            raise ValueError("Speculative decoding is unsupported by the frozen scheduler")

    def _attach_gpu_index_hooks(self):
        # Native reset_prefix_cache replaces the entire map object. Reattach
        # after EVERY reset before constructing an index for the next method.
        hash_map = self.kv_cache_manager.block_pool.cached_block_hash_to_block
        if getattr(hash_map, "_continuum_index_hooks", False):
            return
        for operation in ("insert", "pop"):
            original = getattr(hash_map, operation)
            def tracked(key, *call_args, _original=original, **call_kw):
                result = _original(key, *call_args, **call_kw)
                if self.incremental_gpu and self.gpu_index is not None:
                    self.gpu_index.changed(key)
                return result
            setattr(hash_map, operation, tracked)
        hash_map._continuum_index_hooks = True

    @staticmethod
    def metadata(request):
        params = getattr(request, "sampling_params", None)
        return getattr(params, "extra_args", None) or {}

    def program(self, request):
        return str(self.metadata(request).get("cont_program_id") or
                   getattr(request, "session_id", None) or request.request_id)

    def _replica_history_extension(self, program, state, tool=None):
        """Optional controller state piggybacked on the existing history RPC.

        The replica manager already receives a synchronous lifecycle message
        whenever a request starts, enters its tool gap, or finishes.  A
        scheduler with an additional victim model can attach a compact state
        snapshot here without introducing another round trip.  The base
        Continuum path deliberately sends nothing.
        """
        return {}

    def _emit(self, event, **kw):
        if self.logfile:
            row = {"event": event, "time": time.time(), "experiment": self.experiment, **kw}
            if getattr(self, "log_timing", False):
                row["previous_log_write"] = getattr(self, "_previous_log_write", None)
                # Serialize outside the timed write: distinguish filesystem
                # stalls from controller work / JSON serialization costs.
                line = json.dumps(row) + "\n"
                started = time.perf_counter()
                self.logfile.write(line)
                self._previous_log_write = dict(event=event, experiment=self.experiment,
                                                seconds=time.perf_counter()-started)
            else:
                self.logfile.write(json.dumps(row) + "\n")

    def _replica_call(self, payload):
        path = getattr(self, "replica_socket", None)
        if not path:
            return None
        try:
            return self._replica_transport(path, payload)
        except Exception as exc:
            self._emit("replica_control_error", op=payload.get("op"), error=str(exc))
            # In particular, never release GPU references after an unacked
            # revoke: that would leave a stale protected-copy claim on CPU.
            raise RuntimeError("Replica coordination failed; stop this run") from exc

    def _replica_transport(self, path, payload):
        return replica_rpc(path, payload, timeout=2.0)

    def _replica_revoke(self, program, preserve_borrowed=False):
        if getattr(self, "replica_socket", None):
            self._replica_call({"op": "revoke", "program": str(program),
                                "preserve_borrowed": preserve_borrowed})

    def _replica_grant(self, pin):
        if not getattr(self, "replica_socket", None) or not pin.token_ids or not pin.cache_salt:
            return
        usable = len(pin.blocks) * self.block_size
        usable -= usable % 256
        if usable <= 0:
            return
        if len(pin.token_ids) < usable:
            raise RuntimeError("GPU pin is missing its exact token sequence")
        # LMCache ObjectKey encodes TP/PP/DCP worker identity as a bitmap,
        # while the scheduler's local rank is zero in this prototype.
        from lmcache.v1.distributed.api import ObjectKey
        pc = self.vllm_config.parallel_config
        if (int(pc.world_size) != 1 or int(pc.rank) != 0
                or int(pc.decode_context_parallel_size) != 1):
            raise ValueError("Replica certificates require TP=PP=DCP=1")
        kv_rank = ObjectKey.ComputeKVRank(
            world_size=1, global_rank=0, local_world_size=1, local_rank=0)
        self._replica_call({"op": "grant", "program": pin.program,
                            "model": pin.model or self.replica_model,
                            "salt": pin.cache_salt,
                            "deadline": pin.deadline,
                            "tokens": list(pin.token_ids[:usable]),
                            "kv_rank": kv_rank})

    def _switch_experiment(self, name):
        if name==self.experiment:return
        if self.get_num_unfinished_requests():
            raise RuntimeError("Cannot switch experiment while requests are executing")
        for p in list(self.pins):self._release(p,"experiment_reset")
        if not self.kv_cache_manager.reset_prefix_cache():
            raise RuntimeError("Method isolation failed: KV references remain at reset")
        if hasattr(self.kv_cache_manager.block_pool, "cached_block_hash_to_block"):
            self._attach_gpu_index_hooks()
        self.order.clear();self.departures.clear();self.waiting_samples.clear();self.logged_admissions.clear()
        self.continuity_leases = {}
        self.logged_inputs=set()
        self.history_gpu_hashes = {}
        self.active_replicas = set()
        self._next_gpu_history = 0.0
        self.experiment=name
        reload_path=os.environ.get('CONTINUUM_RELOAD_PROFILE')
        if self.connector and reload_path and Path(reload_path).exists():
            reload_profile=json.loads(Path(reload_path).read_text())
            for key in ('reload_seconds_per_token','reload_fixed_seconds'):
                self.profile[key]=float(reload_profile[key])
            self._emit('reload_profile',**{key:self.profile[key] for key in
                ('reload_seconds_per_token','reload_fixed_seconds')})
        self._event_snapshot_logged=False
        self._capacity_last_signature=None
        self._decision_cache={}
        canonical=expand_version(name.split("~",1)[0]).removesuffix("-repeat")
        canonical = re.sub(r"-(?:a|b|c)$", "", canonical)
        # Optional isolated upstream-baseline adapter; ordinary v21 is unchanged.
        canonical = getattr(self, "normalize_baseline_method", lambda value: value)(canonical)
        decode_match = re.search(r'-decode-(guard|joint|safe|tail|oracle|commit|span)$',canonical)
        self.decode_admission_mode = decode_match.group(1) if decode_match else 'off'
        if decode_match:
            canonical = canonical[:decode_match.start()]
        self.batch_opportunity_mode = canonical.endswith("-batchgain")
        canonical = canonical.removesuffix("-batchgain")
        fit_eta = canonical.endswith("-fit")
        canonical = canonical.removesuffix("-fit")
        self.strict_pin_fallback = canonical.endswith("-strict")
        canonical = canonical.removesuffix("-strict")
        self.paper_engine_compat = canonical.endswith("-paper")
        canonical = canonical.removesuffix("-paper")
        self.chunk_engine_compat = self.paper_engine_compat or canonical.endswith("-chunk")
        canonical = canonical.removesuffix("-chunk")
        self.eta = float(self.profile.get("eta_audit" if fit_eta else "eta", 1))
        partial_admission = canonical.endswith("-partial")
        canonical = canonical.removesuffix("-partial")
        allocation_reclaim = canonical.endswith("-alloc")
        canonical = canonical.removesuffix("-alloc")
        improvement = canonical
        # CPU-line ablations: preserve v21 allocation, TTL and GPU victims.
        # Active certificates use the request's own existing references.
        active_cpu_proof = improvement in (
            "replica-joint-demand-bridge-active-admit",
            "replica-joint-demand-bridge-reference-admit")
        reference_first_reclaim = improvement == "replica-joint-demand-bridge-reference-admit"
        demand_duplicate_reclaim = improvement == "replica-joint-demand-admit"
        shelter_arrived_tail = improvement == "replica-joint-demand-tail-admit"
        pressure_write_bridge = improvement == "replica-joint-demand-bridge-admit" or active_cpu_proof
        demand_zero_reclaim = improvement == "replica-joint-demand-zero-admit" or shelter_arrived_tail or pressure_write_bridge
        demand_lru_reclaim = improvement == "replica-joint-demand-lru-admit"
        complementary_admission = improvement in ("replica-joint-admit", "replica-joint-cost-admit",
                                                   "replica-joint-group-admit",
                                                   "replica-joint-demand-admit",
                                                   "replica-joint-demand-zero-admit",
                                                   "replica-joint-demand-tail-admit",
                                                   "replica-joint-demand-bridge-admit",
                                                   "replica-joint-demand-bridge-active-admit",
                                                   "replica-joint-demand-bridge-reference-admit",
                                                   "replica-joint-demand-lru-admit")
        self.grouped_gpu_recovery = improvement in ("replica-joint-group", "replica-joint-group-admit")
        if improvement == "replica-joint-cost-admit":
            improvement = "replica-joint-cost-flat"
        elif (improvement == "replica-joint-admit" or self.grouped_gpu_recovery
              or demand_duplicate_reclaim or demand_zero_reclaim or demand_lru_reclaim):
            canonical = "replica-joint-incremental"
        wide_cpu = improvement == "replica-joint-wide"
        self.costed_gpu_recovery = improvement in ("replica-joint-cost", "replica-joint-cost-flat", "replica-joint-cost-horizon")
        self.weighted_gpu_recovery = (self.costed_gpu_recovery and improvement != "replica-joint-cost-flat") or improvement in ("replica-joint-urgency", "replica-joint-horizon")
        horizon_cpu = improvement in ("replica-joint-cpu-horizon", "replica-joint-horizon", "replica-joint-cost-horizon")
        if self.weighted_gpu_recovery or self.costed_gpu_recovery or horizon_cpu:
            canonical = "replica-joint-incremental"
        if wide_cpu:
            canonical = "replica-joint-boundary"
        self.gpu_victim_mode = ("duplicate" if canonical == "replica-gpu-duplicate" else
                               "tail" if canonical == "replica-gpu-tail" else
                               "joint" if canonical in ("replica-joint", "replica-joint-tail",
                                                        "replica-joint-boundary", "replica-joint-fast",
                                                        "replica-joint-incremental") else None)
        self.publish_gpu_reuse = canonical in ("replica-joint-boundary", "replica-joint-fast", "replica-joint-incremental")
        self.prefer_gpu_hit = canonical == "replica-joint-fast"
        self.incremental_gpu = canonical in ("replica-joint-incremental", "continuum-index", "vllm-index")
        self.gpu_index = None
        if self.incremental_gpu:
            from kairos._runtime.compat.gpu_index import GPUHistoryIndex
            mapping = self.kv_cache_manager.block_pool.cached_block_hash_to_block
            self.gpu_index = GPUHistoryIndex(lambda key: (
                block.block_id if (block := mapping.get_one_block(key)) is not None else None))
        if getattr(self, "replica_socket", None):
            replica_mode = {"replica-lease": "lease",
                            "replica-lru": "duplicate-lru",
                            "replica-prefix": "prefix-lease",
                            "replica-coord": "coordinated",
                            "replica-bounded": "coordinated",
                            "replica-async": "coordinated",
                            "replica-range": "coordinated",
                            "replica-frontier": "frontier",
                            "replica-loan": "loan",
                            "replica-ledger": "ledger",
                            "replica-span": "span",
                            "replica-history-tail": "history-tail",
                            "replica-history-value": "history-value",
                            "replica-history-cover": "history-cover",
                            "replica-history-repair": "history-value",
                            "replica-active": "history-active",
                            "replica-active-commit": "history-active",
                            "replica-joint": "history-cover",
                            "replica-joint-boundary": "history-cover",
                            "replica-joint-fast": "history-cover",
                            "replica-joint-incremental": "history-cover",
                            "replica-joint-tail": "history-tail",
                            "replica-gpu-tail": "history-tail",
                            "replica-demand": "coordinated"}.get(canonical, "observe")
            if horizon_cpu:
                replica_mode = "history-horizon"
            if wide_cpu:
                replica_mode = "history-cover-wide"
            self._replica_call(dict(op="experiment", name=name, mode=replica_mode,
                                   allocation_reclaim=allocation_reclaim or demand_duplicate_reclaim,
                                   partial_admission=partial_admission,
                                   complementary_admission=complementary_admission,
                                   duplicate_admission=demand_duplicate_reclaim,
                                   demand_zero_reclaim=demand_zero_reclaim,
                                   shelter_arrived_tail=shelter_arrived_tail,
                                   pressure_write_bridge=pressure_write_bridge,
                                   reference_first_reclaim=reference_first_reclaim,
                                   demand_lru_reclaim=demand_lru_reclaim,
                                   tool_history=self.profile.get("tool_history", {}), beta=self.beta,
                                   reload_per_token=self.profile.get("reload_seconds_per_token", 0)))
            if hasattr(self.connector, "set_replica_mode"):
                self.connector.set_replica_mode({"continuum-index": "continuum-range",
                                                  "vllm-index": "vllm-range"}.get(canonical, canonical))
                if active_cpu_proof:
                    self.connector.active_certificates = True
        if canonical == "continuum-index":
            # Fair instrumentation ablation: native CPU/GPU victims, identical
            # Continuum TTL and priorities, same event-maintained GPU observer.
            canonical = "continuum-range"
        if canonical in ("replica-lease", "replica-lru", "replica-prefix",
                         "replica-coord", "replica-repair", "replica-bounded", "replica-demand",
                         "replica-async", "replica-repair-async", "replica-range",
                         "continuum-range", "replica-frontier", "replica-loan",
                         "replica-ledger", "replica-span", "replica-history-tail",
                         "replica-history-value", "replica-history-repair", "replica-history-cover",
                         "replica-active", "replica-active-commit", "replica-gpu-duplicate",
                         "replica-joint", "replica-joint-tail", "replica-gpu-tail",
                         "replica-joint-boundary", "replica-joint-fast", "replica-joint-incremental"):
            canonical = "continuum"
        calibration=os.environ.get("CONTINUUM_DECODE_PROFILE_PATH")
        if calibration and Path(calibration).exists():
            data=json.loads(Path(calibration).read_text())
            self.profile["decode_step_by_seqs"]=data["decode_step_by_seqs"]
            self.profile["small_prefill_wall_s"]=data.get("small_prefill_wall_s")
            self.profile["small_prefill_samples"]=data.get("small_prefill_samples", 0)
            if 'pipeline_service' in data:
                self.profile['pipeline_service']=data['pipeline_service']
            self._emit("decode_calibration",**data)
        policy_canonical=canonical
        canonical=canonical.removesuffix("-fixed")
        cohort_match = re.fullmatch(r"cohort-i([1-9][0-9]*)", canonical)
        self.cohort_interval = int(cohort_match.group(1)) if cohort_match else 1
        self.cohort_adaptive = canonical == "cohort-adaptive"
        if self.cohort_adaptive:
            self.cohort_interval = 4
        self.cohort_ticks = 0
        # Lightweight variants deliberately do not enter the online search or
        # change the ordinary waiting-request allocation path.
        self.priority_lease_mode = canonical in ("lease-whole", "lease-tail", "lease-growth")
        self.growth_guard = canonical in ("growth-guard", "lease-growth")
        if canonical == "lease-tail":
            canonical = "tail-d"
        self.controller=canonical if canonical in ("tail-d", "tail-admit", "tail-reserve", "capacity", "capacity-tail", "residency", "residency-next", "residency-joint", "residency-progress") else (
            canonical if canonical in ("retention", "retention-local") else (
            "partial-j-demand" if canonical=="partial-demand" else (
            "partial-static-demand" if canonical=="partial-demand-static" else (
            "partial-j" if canonical in ("partial","partial-fixed") else (
            "partial-static" if canonical in ("partial-static","partial-static-fixed") else "whole")))))
        self.policy=SchedulingPolicy.FCFS if name.startswith("vllm-") else SchedulingPolicy.PRIORITY
        # FCFS prepends preempted requests; a leftover priority heap would
        # silently change this baseline behavior even with equal priorities.
        self.waiting=create_request_queue(self.policy)
        self.skipped_waiting=create_request_queue(self.policy)
        mode="zero" if canonical in ("zero", "zero-range", "profile") or name.startswith("vllm-") else (
            "fixed" if "fixed" in policy_canonical else ("oracle" if canonical=="oracle" else "paper"))
        self.policy_model=TTLPolicy(mode=mode,fixed=float(os.environ.get("CONTINUUM_TTL_SECONDS","2")),
            k=100,history=self.profile.get("tool_history",{}))
        self._emit("experiment_start",method=name, eta=self.eta,
                   paper_engine_compat=self.paper_engine_compat,
                   chunk_engine_compat=self.chunk_engine_compat,
                   strict_pin_fallback=self.strict_pin_fallback,
                   decode_admission_mode=self.decode_admission_mode,
                   incremental_gpu=self.incremental_gpu)

    def _cohort_gate(self):
        """Return the waiting queues to hide for one decode-only step.

        The parked requests remain in their native waiting queues and retain
        their Continuum KV references.  This is intentionally a short-window
        ablation: it changes only when a ready prefill is admitted, not which
        KV blocks are retained or which program has priority.
        """
        interval = getattr(self, "cohort_interval", 1)
        if interval <= 1:
            return None
        if not self.running or not (self.waiting or self.skipped_waiting):
            return None
        # Do not hold a partially computed prefill.  The first version only
        # tests interference from a completed decode cohort.
        if any(getattr(request, "is_prefill_chunk", False) for request in self.running):
            return None
        if getattr(self, "cohort_adaptive", False):
            # Admit cold/non-pinned work immediately.  The estimate only uses
            # an already-protected append-only prefix, never future tool gaps.
            q = a = 0
            max_wait = 0.0
            for request in list(self.waiting) + list(self.skipped_waiting):
                pin = self.pins.get(self.program(request))
                if pin is None or request.status == RequestStatus.PREEMPTED:
                    return None
                hit = min(pin.tokens, len(pin.blocks)*self.block_size,
                          request.num_prompt_tokens-1)
                count = request.num_prompt_tokens-hit
                if count > 512:
                    return None
                q += count
                a += count*(2*hit+count+1)//2
                max_wait = max(max_wait, time.time()-request.arrival_time)
            variable_cost = self.beta[1]*q + self.beta[2]*a
            # A wave is sufficiently amortized when fixed prefill overhead is
            # <= 10% of its calibrated total cost.  An 80-ms deadline and at
            # most four decode steps bound added request waiting.  These are
            # explicit prototype parameters, not an optimal-JCT guarantee.
            if variable_cost >= 9*self.beta[0] or max_wait >= .08 or self.cohort_ticks >= 4:
                return None
        elif self.cohort_ticks % interval == 0:
            return None
        waiting, skipped = self.waiting, self.skipped_waiting
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        return waiting, skipped

    def _has_continuity_priority(self, program):
        return program in self.pins or (
            getattr(self, "priority_lease_mode", False)
            and program in self.continuity_leases)

    def _clear_continuity_lease(self, program, reason):
        deadline = getattr(self, "continuity_leases", {}).pop(program, None)
        if deadline is not None:
            self._emit("priority_lease", program=program, action="revoke",
                       reason=reason, deadline=deadline)

    def _capacity(self):
        return self.kv_cache_manager.block_pool.num_gpu_blocks - 1

    def _idle_ids(self):
        return {b.block_id for p in self.pins.values() for b in p.blocks}

    def _release(self, program, reason, count=None):
        pin = self.pins.get(program)
        if not pin:
            if reason in ("program_finished", "abort_or_no_tool", "experiment_reset", "shutdown"):
                self._replica_revoke(program)
            return 0
        n = len(pin.blocks) if count is None else min(count, len(pin.blocks))
        if not n:
            return 0
        if (getattr(self.connector, "repair_enabled", False)
                and reason in ("ttl_expired", "deadlock_fallback", "partial_pressure")
                and (reason != "ttl_expired" or
                     getattr(self.connector, "repair_on_ttl_expiry", True))):
            chunks = len(pin.blocks)*self.block_size//256
            plan = self._replica_call(dict(op="prepare_release", program=program,
                                          reason=reason, chunks=chunks))
            # Partial releases repair only the chunks whose GPU protection is
            # actually about to end. Full Continuum releases include them all.
            cutoff = max(0, (len(pin.blocks)-n)*self.block_size//256)
            indices = [index for index in plan["missing"] if index >= cutoff]
            self.connector.queue_backup_repair(pin, indices, self.block_size)
        self._replica_revoke(program, preserve_borrowed=reason in (
            "ttl_expired", "deadlock_fallback", "partial_pressure", "r6_reclaim", "r7_reclaim",
            "handover_after_allocation", "replace"))
        released = pin.blocks[-n:]
        before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self.kv_cache_manager.block_pool.free_blocks(reversed(released))
        pin.blocks = pin.blocks[:-n]
        after = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self._emit("unpin", program=program, reason=reason, references=n,
                   physical_freed=after-before, retained=len(pin.blocks),
                   logical_start=len(pin.blocks), block_ids=[b.block_id for b in released])
        if not pin.blocks:
            del self.pins[program]
        else:
            # The revoked lease is reissued for the still-resident prefix only.
            # This makes partial pressure reclaim visible to CPU eviction while
            # preserving the remaining complete chunks.
            self._replica_grant(pin)
        # Capacity reclaim can cancel physical protection without cancelling
        # the original, bounded scheduling lease. Expiry/finalization/handover
        # must still remove it. A lease never extends the original deadline.
        if reason not in ("partial_pressure", "deadlock_fallback"):
            self._clear_continuity_lease(program, reason)
        elif program in getattr(self, "continuity_leases", {}):
            self._emit("priority_lease", program=program, action="retain-after-unpin",
                       reason=reason, deadline=self.continuity_leases[program])
        return after-before

    def add_request(self, request):
        experiment=self.metadata(request).get("cont_experiment")
        changed = experiment and str(experiment) != self.experiment
        if experiment:self._switch_experiment(str(experiment))
        if changed and "cont_bfcl_reset_history" in self.metadata(request):
            self.policy_model = TTLPolicy(mode=self.policy_model.mode, fixed=self.policy_model.fixed,
                k=self.policy_model.k, history=json.loads(self.metadata(request)["cont_bfcl_reset_history"]))
            self._emit('tool_history_reset',samples=sum(map(len,self.policy_model.history.values())),
                       tools=sorted(self.policy_model.history))
        p = self.program(request)
        self.order.setdefault(p, len(self.order))
        previous = self.departures.pop(p, None)
        request._continuum_returning = previous is not None
        if previous:
            end, tool, old_len = previous
            gap = max(0, time.monotonic()-end)
            new_input = max(0, request.num_prompt_tokens-old_len)
            self.policy_model.observe(tool, gap, new_input)
            self._emit("tool_observed", program=p, tool=tool, duration=gap, added_tokens=new_input)
        if getattr(self, "replica_socket", None):
            self.history_gpu_hashes[p] = tuple(request.block_hashes[:request.num_prompt_tokens//self.block_size])
            if self.incremental_gpu:
                from vllm.v1.core.block_pool import make_block_hash_with_group_id
                self.gpu_index.replace(p, [make_block_hash_with_group_id(h, 0)
                                          for h in self.history_gpu_hashes[p]])
            payload = dict(op="history", program=p, state="running",
                tokens=list(request.prompt_token_ids), model=self.connector.scheduler_adapter.model_name,
                salt=str(getattr(request, "cache_salt", "") or ""),
                observed_gap=gap if previous else None)
            payload.update(self._replica_history_extension(p, "running"))
            self._replica_call(payload)
        self._emit("arrival", program=p, request=request.request_id,
                   prompt_tokens=request.num_prompt_tokens, meta=self.metadata(request))
        super().add_request(request)

    def _free_request_blocks(self, request):
        if not getattr(request, '_continuum_logical_finish_recorded', False):
            self._continuum_complete_policy(request)
        super()._free_request_blocks(request)

    def _continuum_complete_policy(self, request):
        """Record the Program transition; page ownership is freed separately.

        The legacy path invokes this at block cleanup. An opt-in adapter can
        invoke it at logical completion before asynchronous connector cleanup,
        while native request/DMA references remain authoritative.
        """
        meta = self.metadata(request)
        # Autonomous clients cannot know before generation whether the last
        # user turn will require another tool step. Detect the actual Qwen
        # delimiter after generation, without altering the retention policy.
        if "cont_bfcl_tool_marker" in meta:
            meta = dict(meta)
            output = request.output_token_ids
            if "cont_bfcl_think_end_token" in meta and meta["cont_bfcl_think_end_token"] in output:
                cut=max(i for i,token in enumerate(output) if token==meta["cont_bfcl_think_end_token"])
                output=output[cut+1:]
            has_call = meta["cont_bfcl_tool_marker"] in output
            meta["cont_final"] = bool(meta.get("cont_bfcl_last_turn") and not has_call)
            meta["cont_tool"] = "bfcl-tool" if has_call else "bfcl-next-user"
        p = self.program(request)
        self._revoke_active_replica(p)
        success = request.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
        has_tool = bool(meta.get("cont_tool")) and not meta.get("cont_final", False)
        if success and has_tool:
            length = min(request.num_computed_tokens, request.num_tokens)
            if getattr(self, 'native_async_prefix_execution', False):
                # Pending output placeholders have advanced the scheduler's
                # progress but are not a committed cross-turn cache proof.
                length = min(length, request.num_computed_tokens-request.num_output_placeholders)
            wait = sum(self.waiting_samples)/len(self.waiting_samples) if self.waiting_samples else 0
            reload = self.beta[0]+self.beta[1]*length+self.beta[2]*length*(length+1)/2
            if self.connector:
                reload = (self.profile.get("reload_fixed_seconds",0)+
                          self.profile.get("reload_seconds_per_token", self.beta[1])*length)
            reload = self._continuation_restore_estimate(request,length,reload)
            benefit = wait*self.eta+reload
            r5_rate = None
            r5_continuity = None
            if getattr(self.policy_model, 'mode', '') == 'r5-conditional-hold':
                # eta is already a global continuity fraction in the
                # incumbent model. Replace it once with the local downstream
                # fraction; do not multiply the same waiting benefit by a
                # second count of Programs.
                r5_continuity = getattr(self, '_r5_continuity_fraction', lambda _r: self.eta)(request)
                benefit = wait*r5_continuity + reload
                r5_rate = getattr(self, '_r5_opportunity_rate', lambda _r, _n: 1.0)(request, length)
            ttl, source = self.policy_model.ttl(str(meta["cont_tool"]), benefit,
                                                 meta.get("cont_oracle_gap"),
                                                 **({'opportunity_rate': r5_rate} if r5_rate is not None else {}))
            self._release(p, "replace")
            self._clear_continuity_lease(p, "return")
            # Acquire independent TTL references BEFORE normal cleanup. Only
            # full, committed blocks are eligible for reuse, never output KV
            # whose final token has not yet passed through the model.
            blocks = list(self.kv_cache_manager.get_blocks(request.request_id).blocks[0])
            blocks = blocks[:length//self.block_size]
            # A chunk certificate must describe a contiguous prefix. Do not
            # collapse holes and accidentally certify unrelated later blocks.
            contiguous = []
            for block in blocks:
                if block.is_null or block.block_hash is None:
                    break
                contiguous.append(block)
            blocks = contiguous
            # Optional two-variable controller; default engine semantics are
            # byte-for-byte the same action. Choose BEFORE acquiring optional
            # references or publishing replica certificates (no grant/revoke
            # round-trip for a transient full-length pin).
            choose = getattr(self, '_choose_time_length', None)
            if choose is not None:
                ttl, blocks, source = choose(request, ttl, blocks, source, length)
            if getattr(self, "replica_socket", None):
                decision_fields = {}
                if r5_rate is not None:
                    decision_fields.update(
                        continuity_fraction=r5_continuity,
                        opportunity_rate=r5_rate,
                        hold_objective='expected-recovery-area minus state opportunity cost')
                self._emit("pin_decision", program=p, tokens=length,
                           committed_prefix_blocks=len(blocks), ttl=ttl,
                           source=source, wait_mean=wait, reload_estimate=reload,
                           protection_created=bool(ttl > 0 and blocks), **decision_fields)
            if ttl > 0 and blocks:
                self.kv_cache_manager.block_pool.touch(blocks)
                now = time.monotonic()
                token_ids = tuple(request.all_token_ids[:length]) if getattr(self, "replica_socket", None) else ()
                model = self.connector.scheduler_adapter.model_name if token_ids else ""
                self.pins[p] = Pin(p, now+ttl, now, str(meta["cont_tool"]), request.num_prompt_tokens, blocks,
                                  list(request.block_hashes), token_ids,
                                  model,
                                  str(getattr(request, "cache_salt", "") or ""))
                self._replica_grant(self.pins[p])
                if self.priority_lease_mode:
                    self.continuity_leases[p] = now+ttl
                    self._emit("priority_lease", program=p, action="grant",
                               deadline=self.continuity_leases[p], ttl=ttl)
                self._emit("pin", program=p, references=len(blocks), ttl=ttl,
                           source=source, wait_mean=wait, reload_estimate=reload)
            self.departures[p] = (time.monotonic(), str(meta["cont_tool"]), request.num_prompt_tokens)
            if getattr(self, "replica_socket", None):
                payload = dict(op="history", program=p, state="tool", tool=str(meta["cont_tool"]))
                payload.update(self._replica_history_extension(p, "tool", str(meta["cont_tool"])))
                self._replica_call(payload)
        else:
            self._release(p, "program_finished" if meta.get("cont_final") else "abort_or_no_tool")
            self._clear_continuity_lease(p, "program-finished")
            if meta.get("cont_final"):
                self.departures.pop(p, None)
            if getattr(self, "replica_socket", None):
                payload = dict(op="history", program=p, state="finished")
                payload.update(self._replica_history_extension(p, "finished"))
                self._replica_call(payload)
                self.history_gpu_hashes.pop(p, None)
                if self.incremental_gpu:
                    self.gpu_index.remove(p)
    def _continuation_restore_estimate(self,request,length,default):
        """Original Continuum restoration quote; isolated variants override."""
        return default

    def _preempt_request(self, request, *args, **kwargs):
        self._revoke_active_replica(self.program(request))
        self._emit("preempt",program=self.program(request),request=request.request_id,
                   computed_tokens=request.num_computed_tokens)
        return super()._preempt_request(request,*args,**kwargs)

    def _refresh_priority(self):
        if self.experiment and self.experiment.startswith("vllm-"):return
        # Rebuild heaps after updating mutable priority keys.
        stride = 1000000
        for name in ("waiting", "skipped_waiting"):
            old = getattr(self, name)
            queue = create_request_queue(self.policy)
            for request in list(old):
                p = self.program(request)
                # An already allocated asynchronous restore can hold most
                # of the pool. The native waiting loop stops at allocation
                # failure, so an unallocated higher-priority request can
                # otherwise hide a completed restore forever. Drain ready
                # restores first; native promotion/locks remain unchanged.
                restored = ((request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                             and request.request_id in self.finished_recving_kv_req_ids)
                            or (request.status == RequestStatus.WAITING
                                and getattr(request, "num_computed_tokens", 0) > 0))
                rank = -1 if restored else (0 if request.status == RequestStatus.PREEMPTED
                         else (1 if self._has_continuity_priority(p) else 2))
                request.priority = rank*stride+self.order.get(p, stride-1)
                queue.add_request(request)
            setattr(self, name, queue)
        for r in self.running:
            if getattr(self, "paper_engine_compat", False):
                # Paper TTL priority applies to Q (waiting), not a repeated
                # reordering of already-running chunks. Use program arrival
                # for native contention victims; preserve admission order.
                r.priority = self.order.get(self.program(r), stride-1)
            else:
                r.priority=(1 if getattr(r,"_continuum_was_pinned",False) else 2)*stride+self.order.get(self.program(r),stride-1)
        if not getattr(self, "paper_engine_compat", False):
            self.running.sort(key=lambda r: (r.priority, self.order.get(self.program(r), 0)))

    def schedule(self, throttle_prefills=False):
        queued = {self.program(r) for q in (self.waiting, self.skipped_waiting) for r in q}
        now = time.monotonic()
        self._expire_idle_protection(now, queued)
        self._refresh_priority()
        self._observe_gpu_history()
        ready_decode_ids = [request.request_id for request in self.running
                            if not request.is_prefill_chunk]
        start = time.perf_counter()
        parked = self._cohort_gate()
        try:
            output = self._run_native_schedule(throttle_prefills=throttle_prefills)
        finally:
            if parked is not None:
                # Native allocation can preempt a running request even during
                # a held step.  Preserve those newly queued requests as well.
                parked[0].prepend_requests(list(self.waiting))
                parked[1].prepend_requests(list(self.skipped_waiting))
                self.waiting, self.skipped_waiting = parked
        native_schedule_s = time.perf_counter()-start
        features = [0, 0, 0]
        prefill_queries = 0
        members = []
        for rid, count in output.num_scheduled_tokens.items():
            r = self.requests.get(rid)
            if not r:
                continue
            end = r.num_computed_tokens
            begin = end-count+1
            prefill_queries += max(0, min(end,r.num_prompt_tokens)-begin+1)
            features[0] += count
            features[1] += (begin+end)*count//2
            features[2] += 1
            members.append(dict(request=rid, program=self.program(r), begin=begin, end=end,
                                queries=count, prompt=r.num_prompt_tokens,
                                prefill_queries=max(0, min(end,r.num_prompt_tokens)-begin+1),
                                returning=getattr(r, "_continuum_returning", False),
                                output_tokens=r.num_output_tokens))
            if rid not in self.logged_admissions:
                self.logged_admissions.add(rid)
                wait = max(0, time.time()-r.arrival_time)
                if getattr(r,"_continuum_returning",False) and not getattr(r,"_continuum_was_pinned",False):
                    self.waiting_samples.append(wait)
                stats = r.prefill_stats
                self._emit("admit", program=self.program(r), request=rid, waiting_s=wait,
                           local_cached_tokens=getattr(stats,"num_local_cached_tokens", 0),
                           external_cached_tokens=getattr(stats,"num_external_cached_tokens", 0),
                           prompt_tokens=r.num_prompt_tokens)
        # SchedulerOutput currently is a dataclass without slots.
        output._continuum_features = features
        output._continuum_start = time.perf_counter()
        self._continuum_step_serial = getattr(self, '_continuum_step_serial', 0)+1
        output._continuum_step_serial = self._continuum_step_serial
        if features[0]:
            idle_started = time.perf_counter()
            idle_blocks = len(self._idle_ids())
            idle_scan_s = time.perf_counter()-idle_started
            self._emit("batch_schedule", Q=features[0], A=features[1], N=features[2],
                       step_id=output._continuum_step_serial,
                       prefill_queries=prefill_queries, decode_queries=features[0]-prefill_queries,
                       decode_only=prefill_queries == 0,
                       running=len(self.running), waiting=len(self.waiting)+len(self.skipped_waiting),
                       idle_blocks=idle_blocks, free=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                       ready_decode_ids=ready_decode_ids,
                       unscheduled_decode_ids=[rid for rid in ready_decode_ids
                                               if rid not in output.num_scheduled_tokens],
                       schedule_s=time.perf_counter()-start, members=members,
                       native_schedule_s=native_schedule_s, idle_scan_s=idle_scan_s)
        elif self.replica_socket and time.monotonic() >= getattr(self, "_next_idle_diagnostic", 0):
            self._next_idle_diagnostic = time.monotonic()+1
            self._emit("transfer_only_state",
                       free=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                       pins=len(self.pins),
                       deferred_free_blocks=sum(len(blocks) for _, blocks in
                                                getattr(self, "deferred_frees", ())),
                       waiting=[dict(request=r.request_id, status=r.status.name,
                                     computed=r.num_computed_tokens,
                                     prompt=r.num_prompt_tokens)
                                for q in (self.waiting, self.skipped_waiting) for r in q],
                       finished_receives=sorted(self.finished_recving_kv_req_ids),
                       repair_refs=sum(len(ids) for ids in
                                       getattr(self.connector, "_repair_refs", {}).values()))
        if getattr(self, "cohort_interval", 1) > 1:
            # A prefill wave resets the interval.  If no prefill was admitted,
            # a decode-only step advances the bounded hold window.
            if prefill_queries:
                self.cohort_ticks = 1
            elif features[0]:
                self.cohort_ticks += 1
            self._emit("cohort_gate", interval=self.cohort_interval,
                       tick=self.cohort_ticks, deferred=parked is not None,
                       queued=len(queued), prefill_queries=prefill_queries,
                       decode_queries=features[0]-prefill_queries)
        return output

    def _run_native_schedule(self, throttle_prefills=False):
        return super().schedule(throttle_prefills=throttle_prefills)

    def _observe_gpu_history(self):
        """Small, common read-only APC sampler; it changes no references.

        A hash match is advisory for CPU eviction only. Ordinary GPU/CPU
        request matching remains the sole authority for actual restoration.
        Run identically for every replica-control baseline and candidate.
        """
        now = time.monotonic()
        if not self.replica_socket or now < self._next_gpu_history:
            return
        self._next_gpu_history = now+.05
        started = time.perf_counter()
        pool = self.kv_cache_manager.block_pool
        prefixes = {}
        if getattr(self, "incremental_gpu", False):
            prefixes = self.gpu_index.prefixes()
        else:
            for program, hashes in self.history_gpu_hashes.items():
                count = 0
                for block_hash in hashes:
                    if pool.get_cached_block(block_hash, [0]) is None:
                        break
                    count += 1
                prefixes[program] = count*self.block_size//256
        self._replica_call(dict(op="gpu_history", prefixes=prefixes))
        self._emit("gpu_history_observed", prefixes=prefixes,
                   controller_s=time.perf_counter()-started, advisory_only=True)

    def _revoke_active_replica(self, program):
        if program in getattr(self, "active_replicas", set()):
            # Revoke before the native request releases or reuses its pages.
            self._replica_revoke(program)
            self.active_replicas.remove(program)
            self._emit("active_replica_revoke", program=program)

    def _get_new_blocks(self, count):
        """Reorder only native free pages immediately before their reuse.

        Allocation demand, page count, refcounts and cached hash removal stay
        native. CPU observations affect victim preference only; a host copy
        can disappear after sampling and is never advertised as a cache hit.
        """
        if not self.gpu_victim_mode or not count:
            return self.original_get_new_blocks(count)
        from kairos._runtime.compat.joint_eviction import choose_gpu_victims
        started = time.perf_counter()
        pool = self.kv_cache_manager.block_pool
        queue = pool.free_block_queue
        if getattr(self, "incremental_gpu", False):
            if count > queue.num_free_blocks:
                return self.original_get_new_blocks(count)
            head = queue.fake_free_list_head.next_free_block
            before_blocks = []
            for _ in range(count):
                before_blocks.append(head)
                head = head.next_free_block
            free = None
        else:
            free = queue.get_all_free_blocks()
            before_blocks = free[:count]
        if ((free is not None and count > len(free))
                or not any(b.block_hash is not None for b in before_blocks)):
            return self.original_get_new_blocks(count)
        if (getattr(self, "incremental_gpu", False)
                and not self.gpu_index.shrinks([b.block_id for b in before_blocks])):
            # Native victims all have zero GPU-prefix loss for EVERY owner.
            # Therefore their joint restore loss is zero irrespective of CPU
            # evictions, and legacy v19's LRU tie picks precisely these pages.
            # No prefix hint is invalidated, so no withdrawal RPC is needed.
            self._emit("joint_gpu_eviction", mode=self.gpu_victim_mode,
                       pages=count, changed_pages=0, prefix_changes={},
                       controller_s=time.perf_counter()-started,
                       fast_path="native-zero-prefix-loss", boundary_published=False,
                       extra_gpu_references=0, extra_copy_bytes=0)
            return self.original_get_new_blocks(count)
        if free is None:
            free = queue.get_all_free_blocks()
        publish = getattr(self, "publish_gpu_reuse", False)
        cpu = self._replica_call(dict(op="cpu_residency", withdraw_gpu_advice=publish,
                                     recovery_weights=getattr(self, "weighted_gpu_recovery", False),
                                     recovery_costs=getattr(self, "costed_gpu_recovery", False)))["programs"]
        if (getattr(self, "fast_joint_frontier", False)
                and self.incremental_gpu and self.gpu_victim_mode == "joint"
                and not (self.prefer_gpu_hit or self.weighted_gpu_recovery
                         or self.costed_gpu_recovery or self.grouped_gpu_recovery)):
            chosen = self.gpu_index.zero_restore_victims(
                [b.block_id for b in free[:max(count, 256)]], cpu, count)
            if chosen is not None:
                changes = self.gpu_index.prefix_changes(chosen, cpu)
                if publish:
                    self._replica_call(dict(op="gpu_history", prefixes={
                        p: row["after_pages"]//16 for p, row in changes.items()},
                        source="before_gpu_page_reuse"))
                before = [b.block_id for b in before_blocks]
                if chosen != before:
                    blocks = [pool.blocks[bid] for bid in chosen]
                    for block in blocks:
                        if block.is_null or block.ref_cnt != 0:
                            raise RuntimeError("Frontier fast path selected a referenced GPU page")
                        queue.remove(block)
                    queue.prepend_n(blocks)
                    if pool.get_num_free_blocks() != len(free):
                        raise RuntimeError("Frontier fast path changed free-page capacity")
                self._emit("joint_gpu_eviction", mode=self.gpu_victim_mode,
                    pages=count, changed_pages=len(set(chosen)-set(before)),
                    prefix_changes=changes, controller_s=time.perf_counter()-started,
                    fast_path="sparse-zero-restore-loss", boundary_published=publish,
                    extra_gpu_references=0, extra_copy_bytes=0)
                return self.original_get_new_blocks(count)
        if getattr(self, "incremental_gpu", False):
            layouts = self.gpu_index.layouts(cpu)
        else:
            layouts = []
            for program, hashes in self.history_gpu_hashes.items():
                row = cpu.get(program)
                if row is None:
                    continue
                ids = []
                for block_hash in hashes:
                    found = pool.get_cached_block(block_hash, [0])
                    ids.append(found[0].block_id if found is not None else None)
                layouts.append(dict(row, program=program, blocks=ids))
        group_facts = {}
        if getattr(self, "grouped_gpu_recovery", False):
            from kairos._runtime.compat.grouped_eviction import choose_grouped_gpu_victims
            chosen, group_facts = choose_grouped_gpu_victims(
                [b.block_id for b in free], layouts, count)
        else:
            chosen = choose_gpu_victims([b.block_id for b in free], layouts,
                                       count, self.gpu_victim_mode,
                                       prefer_gpu=getattr(self, "prefer_gpu_hit", False),
                                       weighted=getattr(self, "weighted_gpu_recovery", False),
                                       costed=getattr(self, "costed_gpu_recovery", False))
        before = [b.block_id for b in free[:count]]
        victims = set(chosen)
        prefix_changes = {}
        for layout in layouts:
            ids = layout["blocks"]
            old_end = next((i for i, bid in enumerate(ids) if bid is None), len(ids))
            new_end = next((i for i, bid in enumerate(ids) if bid is None or bid in victims), len(ids))
            prefix_changes[layout["program"]] = dict(before_pages=old_end, after_pages=new_end)
        if publish:
            # Withdraw the old APC hint before choosing victims, then publish
            # a conservative post-reuse boundary before native hash removal.
            # CPU eviction can still discard its own tails under pressure;
            # this is metadata coordination, not a CPU read lock or a pin.
            self._replica_call(dict(op="gpu_history", prefixes={
                program: row["after_pages"]//16 for program, row in prefix_changes.items()},
                source="before_gpu_page_reuse"))
        if chosen != before:
            blocks = [pool.blocks[bid] for bid in chosen]
            for block in blocks:
                if block.is_null or block.ref_cnt != 0:
                    raise RuntimeError("Joint eviction attempted to reuse a referenced GPU page")
                pool.free_block_queue.remove(block)
            pool.free_block_queue.prepend_n(blocks)
            if pool.get_num_free_blocks() != len(free):
                raise RuntimeError("Joint eviction changed free-page capacity")
        self._emit("joint_gpu_eviction", mode=self.gpu_victim_mode,
                   pages=count, changed_pages=len(set(chosen)-set(before)),
                   controller_s=time.perf_counter()-started,
                   prefix_changes=prefix_changes, boundary_published=publish,
                   prefer_gpu_hit=getattr(self, "prefer_gpu_hit", False),
                   weighted_recovery=getattr(self, "weighted_gpu_recovery", False),
                   costed_recovery=getattr(self, "costed_gpu_recovery", False),
                   grouped_recovery=getattr(self, "grouped_gpu_recovery", False), group_selection=group_facts,
                   recovery_weights={row["program"]: row.get("recovery_weight", 1)
                                     for row in layouts} if getattr(self, "weighted_gpu_recovery", False) else {},
                   extra_gpu_references=0, extra_copy_bytes=0)
        return self.original_get_new_blocks(count)

    def _certify_active_replica(self, request):
        if not getattr(self.connector, "active_certificates", False):
            return
        program = self.program(request)
        if program in self.active_replicas:
            return
        length = min(request.num_computed_tokens, request.num_prompt_tokens)
        if getattr(self, 'native_async_prefix_execution', False):
            length = min(length, request.num_computed_tokens-request.num_output_placeholders)
        usable = length-length % 256
        blocks = list(self.kv_cache_manager.get_blocks(request.request_id).blocks[0])[:usable//self.block_size]
        # Called only after model execution. Cached hashes alone are not
        # enough: these pages must have the native request's live references.
        if not usable or len(blocks)*self.block_size != usable or any(
                b.is_null or b.block_hash is None or b.ref_cnt <= 0 for b in blocks):
            return
        pin = Pin(program, time.monotonic()+30, time.monotonic(), "active",
                  usable, blocks, list(request.block_hashes),
                  tuple(request.all_token_ids[:usable]),
                  self.connector.scheduler_adapter.model_name,
                  str(getattr(request, "cache_salt", "") or ""))
        self._replica_grant(pin)
        self.active_replicas.add(program)
        self._emit("active_replica_grant", program=program, chunks=usable//256,
                   extra_gpu_references=0)

    def _expire_idle_protection(self, now, queued):
        for p, pin in list(self.pins.items()):
            if now >= pin.deadline and p not in queued:
                self._release(p, "ttl_expired")
        for p, deadline in list(self.continuity_leases.items()):
            if now >= deadline and p not in queued:
                self._clear_continuity_lease(p, "expired")

    def has_unfinished_requests(self):
        return (super().has_unfinished_requests()
                or bool(getattr(self.connector, "has_pending_repairs", lambda: False)()))

    def update_from_output(self, scheduler_output, model_runner_output):
        kv_output = model_runner_output.kv_connector_output
        if self.replica_socket and kv_output and (
                kv_output.finished_recving or kv_output.finished_sending):
            self._emit("transfer_finished",
                       receiving=sorted(kv_output.finished_recving or ()),
                       sending=sorted(kv_output.finished_sending or ()))
        features = getattr(scheduler_output, "_continuum_features", None)
        if features and features[0]:
            self._emit("batch_finish", Q=features[0], A=features[1], N=features[2],
                       step_id=getattr(scheduler_output, '_continuum_step_serial', None),
                       engine_step_wall_s=time.perf_counter()-scheduler_output._continuum_start)
            for rid in scheduler_output.num_scheduled_tokens:
                r=self.requests.get(rid)
                if r and r.num_computed_tokens>=r.num_prompt_tokens and rid not in self.logged_inputs:
                    self._certify_active_replica(r)
                    self.logged_inputs.add(rid)
                    self._emit("input_complete",request=rid,program=self.program(r),
                               prompt_tokens=r.num_prompt_tokens)
        return super().update_from_output(scheduler_output, model_runner_output)

    def _allocate(self, request, num_new_tokens, *args, **kw):
        if getattr(self, "chunk_engine_compat", False):
            # v0.10.2 allocates the current chunk plus matched KV, rather than
            # adding v0.28's separate full-prompt admission gate. External KV
            # still needs its real slots; the native capacity/refcount checks
            # remain authoritative. No installed vLLM code is edited.
            kw["full_sequence_must_fit"] = False
        if request.num_computed_tokens == 0:
            request._continuum_was_pinned = self._has_continuity_priority(self.program(request))
        computed=kw.get("new_computed_blocks")
        excluded={b.block_id for group in computed.blocks for b in group} if computed else set()
        running = getattr(self,"running",[])
        stalled = not running or (len(running)==1 and running[0] is request)
        paper_admission_failure = (getattr(self, "strict_pin_fallback", False)
                                   and request not in running)
        fallback_allowed = stalled or paper_admission_failure
        exact_controller = self.controller in ("tail-d", "tail-admit", "retention", "retention-local",
                                                "tail-reserve", "capacity", "capacity-tail", "residency", "residency-next", "residency-joint", "residency-progress")
        if not exact_controller and self.controller != "whole" and self.pins and (not self.controller.endswith("-demand") or stalled):
            # Check both the reserve-based high-water mark and true allocation
            # deficit at the same synchronous allocator boundary.
            self._partial_reclaim(request,num_new_tokens,excluded,kw)
        result = self.original_allocate(request, num_new_tokens, *args, **kw)
        p = self.program(request)
        if result is None and os.environ.get("CONTINUUM_BATCH_PRESSURE_AUDIT") == "1":
            # Diagnostic only: native allocation and pin decisions are unchanged.
            # A capacity counterfactual is not proof that a request is ready to
            # decode, or that releasing a pin would improve end-to-end latency.
            pressure = self._required_capacity(request, num_new_tokens, kw)
            safe = self._safe_suffix_rows(excluded)
            self._emit("pin_capacity_audit", program=p, request=request.request_id,
                       status=request.status.name, **pressure,
                       running_programs=[self.program(r) for r in running],
                       running_decode=sum(not r.is_prefill_chunk for r in running),
                       safe_private_pin_blocks=sum(row["limit"] for row in safe),
                       pin_release_covers_deficit=bool(pressure["D"] and
                           sum(row["limit"] for row in safe) >= pressure["D"]),
                       pins=[dict(program=victim, blocks=len(pin.blocks),
                                  remaining_ttl_s=max(0., pin.deadline-time.monotonic()),
                                  return_queued=victim in {
                                      self.program(r) for q in (self.waiting, self.skipped_waiting)
                                      for r in q}) for victim, pin in self.pins.items()])
        if (result is None and getattr(self, 'batch_opportunity_mode', False)
                and request not in running and len(running) == 1
                and not running[0].is_prefill_chunk and self.pins):
            result = self._batch_opportunity_reclaim(request, num_new_tokens, excluded, args, kw)
        if self.logfile and result is None and time.monotonic() >= getattr(self, "_next_allocation_diagnostic", 0):
            self._next_allocation_diagnostic = time.monotonic()+1
            pressure = self._required_capacity(request, num_new_tokens, kw)
            self._emit("allocation_wait", program=p, request=request.request_id, **pressure,
                       pinned_programs=list(self.pins),
                       active=[dict(program=self.program(r), request=r.request_id,
                           computed=r.num_computed_tokens, prompt=r.num_prompt_tokens,
                           priority=r.priority,
                           blocks=len(self.kv_cache_manager.get_blocks(r.request_id).blocks[0]))
                               for r in running])
        if (result is None and getattr(self, "growth_guard", False)
                and self.pins and request in running and len(running) > 1
                and request.num_computed_tokens >= request.num_prompt_tokens):
            result = self._guard_decode_growth(request, num_new_tokens, excluded, args, kw)
        # Ordinary contention should wait for running work or use vLLM's
        # preemption.  Reclaim whole pins only when there is no other running
        # request that can make progress (paper deadlock fallback).
        if result is None and self.pins:
            if exact_controller and (stalled or self.controller in ("tail-admit", "tail-reserve", "capacity", "capacity-tail", "residency", "residency-next", "residency-joint", "residency-progress")):
                result = self._exact_reclaim(request, num_new_tokens, excluded, args, kw)
            if result is None and fallback_allowed:
                pressure = self._required_capacity(request, num_new_tokens, kw)
                self._emit("allocation_failed", program=p, **pressure,
                           pin_blocks=len(self._idle_ids()))
            if result is None and fallback_allowed:
                if paper_admission_failure and not stalled:
                    self._emit("paper_pin_fallback", program=p,
                               running_programs=[self.program(r) for r in running])
                # Paper fallback: latest program arrival first. Hit blocks about
                # to be adopted stay referenced throughout every retry.
                for victim in sorted(list(self.pins), key=lambda x:self.order[x], reverse=True):
                    if any(b.block_id in excluded for b in self.pins[victim].blocks):
                        continue
                    self._release(victim, "deadlock_fallback")
                    result = self.original_allocate(request, num_new_tokens, *args, **kw)
                    if result is not None:
                        break
        if result is not None and p in self.pins:
            self._release(p, "handover_after_allocation")
        if result is not None:
            self._clear_continuity_lease(p, "handover")
        return result

    def _batch_opportunity_reclaim(self, request, num_new_tokens, excluded, args, kw):
        """CPU-backed private suffix only, after real waiting-admission failure.

        CPU residence is advisory; future native lookups/locks are authoritative.
        No new reference, DMA, TTL extension or installed-engine patch.
        Unknown shapes and insufficient opportunity retain the original policy.
        """
        from kairos._runtime.compat.batch_opportunity import backed_suffix, local_gain
        signature = self._decision_signature(request)
        if self._decision_cache.get(request.request_id) == signature:
            return None
        self._decision_cache[request.request_id] = signature
        table = self.profile.get('decode_step_by_seqs', {})
        if '1' not in table or '2' not in table:
            return None
        start = time.perf_counter()
        pressure = self._required_capacity(request, num_new_tokens, kw)
        rows = self._safe_suffix_rows(excluded)
        demand = max(pressure['D'], self._completion_release_budget(request, pressure))
        if demand <= 0 or not any(row['limit'] >= demand for row in rows):
            return None
        cpu = self._replica_call(dict(op='cpu_residency'))['programs']
        active = self.running[0]
        local = request.num_computed_tokens+kw.get('num_new_computed_tokens', 0)
        computed = local+kw.get('num_external_computed_tokens', 0)
        missing = max(0, request.num_prompt_tokens-computed)
        attention = missing*(2*computed+missing+1)/2
        delay = (self.beta[0]+self.beta[1]*missing+self.beta[2]*attention
                 if missing else 0.)
        if kw.get('num_external_computed_tokens', 0):
            delay += self.profile.get('reload_fixed_seconds', 0.) + (
                kw['num_external_computed_tokens']*self.profile.get('reload_seconds_per_token', 0.))
        candidates = []
        now = time.monotonic()
        for row in rows:
            p = row['id']
            pin = self.pins[p]
            layout = cpu.get(p)
            if not layout:
                continue
            suffix = backed_suffix(len(pin.blocks), demand, row['limit'],
                                   set(layout['cpu_chunks']), layout['chunks'])
            if suffix is None:
                continue
            # Includes the incomplete last chunk and all known committed tail.
            tail = max(0, len(pin.token_ids)-len(pin.blocks)//16*256)
            beginning = len(pin.blocks)//16*256
            cost = self.profile.get('reload_fixed_seconds', 0.) + (
                suffix['cpu_restore_tokens']*self.profile.get('reload_seconds_per_token', 0.))
            if tail:
                cost += self.beta[0]+self.beta[1]*tail+self.beta[2]*tail*(2*beginning+tail+1)/2
            gain = local_gain(float(table['1']), float(table['2']),
                              max(0., pin.deadline-now),
                              max(0, active.max_tokens-active.num_output_tokens),
                              request.max_tokens, delay, cost)
            if gain:
                candidates.append(dict(victim=p, **suffix, **gain))
        accepted = [row for row in candidates if row['accept']]
        selected = max(accepted, key=lambda row: row['saved_gpu_work_upper_bound_s']-
                       row['estimated_return_cost_s']) if accepted else None
        self._emit('batch_opportunity', program=self.program(request), demand_pages=demand,
                   candidates=candidates, selected=selected,
                   incoming_delay_estimate_s=delay, controller_s=time.perf_counter()-start,
                   assumption='output caps upper-bound work; snapshot CPU residence is advisory')
        if selected is None:
            return None
        self._release(selected['victim'], 'cpu_backed_batch_opportunity', selected['release_pages'])
        return self.original_allocate(request, num_new_tokens, *args, **kw)

    def _guard_decode_growth(self, request, num_new_tokens, excluded, args, kw):
        """Isolated active-decode rescue; ordinary admission is unchanged.

        Only current allocator failure can trigger it. The reservation covers
        known remaining output caps of running requests, not future turns.
        All victims must be safe private idle suffixes. No event simulation,
        prediction or per-token background scan is performed.
        """
        start = time.perf_counter()
        pressure = self._required_capacity(request, num_new_tokens, kw)
        rows = self._safe_suffix_rows(excluded)
        available = sum(row["limit"] for row in rows)
        if not pressure["D"] or available < pressure["D"]:
            return None
        amount = min(available, self._completion_release_budget(request, pressure))
        plan = self._latest_suffix_plan(rows, max(pressure["D"], amount))
        before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for victim, count in plan.items():
            self._release(victim, "partial_pressure", count)
        freed = self.kv_cache_manager.block_pool.get_num_free_blocks()-before
        result = self.original_allocate(request, num_new_tokens, *args, **kw)
        self._emit("exact_controller", program=self.program(request), **pressure,
                   selection=plan,
                   physical_freed=freed,
                   admitted=result is not None, controller_s=time.perf_counter()-start,
                   selection_mode="decode-growth-only", search_s=0.0,
                   search_accepted=False, evaluation_complete=False, evaluations=0)
        return result

    def _required_capacity(self, request, num_new_tokens, kw):
        """Mirror both native admission and incremental-allocation gates."""
        manager = self.kv_cache_manager
        local = request.num_computed_tokens + kw.get("num_new_computed_tokens", 0)
        ext = kw.get("num_external_computed_tokens", 0)
        total = min(local + ext, self.max_model_len)
        hits = kw.get("new_computed_blocks") or manager.empty_kv_cache_blocks
        watermark = manager.watermark_blocks if (
            kw.get("has_scheduled_reqs", True)
            and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)) else 0

        def needed(tokens, admission=False, main_tokens=None):
            return manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id, num_tokens=tokens,
                new_computed_blocks=hits.blocks,
                num_encoder_tokens=kw.get("num_encoder_tokens", 0),
                total_computed_tokens=total if admission else local+ext,
                num_local_computed_tokens=local,
                num_tokens_main_model=tokens if main_tokens is None else main_tokens,
                apply_admission_cap=admission)

        incremental_tokens = min(total + num_new_tokens + kw.get("num_lookahead_tokens", 0),
                                 self.max_model_len)
        incremental = needed(incremental_tokens, main_tokens=total+num_new_tokens) + watermark + kw.get("reserved_blocks", 0)
        admission = needed(min(request.num_tokens, self.max_model_len), True) + watermark \
            if kw.get("full_sequence_must_fit") else 0
        required = max(admission, incremental)
        free = manager.block_pool.get_num_free_blocks()
        return dict(G=required, D=max(0, required-free), free=free,
                    admission_required=admission, incremental_required=incremental,
                    watermark=watermark, reserved=kw.get("reserved_blocks", 0),
                    full_sequence_must_fit=bool(kw.get("full_sequence_must_fit")))

    def _safe_suffix_rows(self, excluded):
        pool = self.kv_cache_manager.block_pool
        owners = {}
        for pin in self.pins.values():
            for block in pin.blocks:
                owners[block.block_id] = owners.get(block.block_id, 0) + 1
        rows = []
        for program in sorted(self.pins, key=lambda p: self.order[p], reverse=True):
            pin = self.pins[program]
            # Keep precisely the original whole-fallback victim eligibility.
            # The incoming request's own hit program is not a tail-D victim.
            if any(block.block_id in excluded for block in pin.blocks):
                continue
            limit = 0
            for block in reversed(pin.blocks):
                if block.block_id in excluded or owners[block.block_id] != 1 or block.ref_cnt != 1:
                    break
                limit += 1
            if limit:
                rows.append(dict(id=program, order=self.order[program], total=len(pin.blocks), limit=limit))
        return rows

    @staticmethod
    def _latest_suffix_plan(rows, deficit):
        plan = {}
        for row in rows:
            count = min(deficit, row["limit"])
            if count:
                plan[row["id"]] = count
                deficit -= count
            if not deficit:
                break
        return plan if not deficit else {}

    def _exact_reclaim(self, request, num_new_tokens, excluded, args, kw):
        start = time.perf_counter()
        pressure = self._required_capacity(request, num_new_tokens, kw)
        if self.controller.startswith(("capacity", "residency")):
            signature = self._decision_signature(request)
            cache = getattr(self, "_decision_cache", {})
            if cache.get(request.request_id) == signature:
                return None
            # Multiple blocked requests can alternate in one schedule call;
            # one global last signature made them invalidate each other.
            cache[request.request_id] = signature
            self._decision_cache = cache
            active = {r.request_id for r in self.running}
            active.update(r.request_id for q in (self.waiting, self.skipped_waiting) for r in q)
            for rid in list(cache):
                if rid != request.request_id and rid not in active:
                    del cache[rid]
        rows = self._safe_suffix_rows(excluded)
        if not pressure["D"] or sum(row["limit"] for row in rows) < pressure["D"]:
            self._emit("exact_infeasible", program=self.program(request), **pressure,
                       safe_capacity=sum(row["limit"] for row in rows))
            return None
        plan = self._latest_suffix_plan(rows, pressure["D"])
        detail = dict(selection_mode="latest-program-exact-deficit")
        if self.controller == "tail-reserve" or (
                self.controller in ("residency-joint","residency-progress") and request in self.running):
            # A None returned for an active request causes immediate native
            # preemption, not a harmless wait. Protect current progress first.
            reserve = self._completion_release_budget(request, pressure)
            amount = min(sum(row["limit"] for row in rows), reserve)
            plan = self._latest_suffix_plan(rows, max(pressure["D"], amount))
            detail = dict(selection_mode="active-completion-reserve",
                          completion_release_budget=reserve, evaluations=0,
                          search_s=0.0, search_accepted=False,
                          evaluation_complete=False)
        elif self.controller == "residency-progress":
            if any(r is not request for r in self.running):
                plan={}
            else:
                amount=0
                for row in rows:
                    amount+=row["limit"]
                    if amount>=pressure["D"]:
                        break
                plan=self._latest_suffix_plan(rows,amount)
            detail=dict(selection_mode="continuum-reference-no-search",
                        search_s=0.0,search_accepted=False,evaluations=0,
                        evaluation_complete=False)
        elif self.controller.startswith("residency"):
            plan, detail = self._residency_plan(request, rows, pressure, plan)
        elif self.controller.startswith(("retention", "capacity")):
            plan, detail = self._retention_plan(request, rows, pressure, plan, kw)
        if not plan:
            self._emit("exact_controller", program=self.program(request), **pressure,
                       selection={}, physical_freed=0, admitted=False,
                       controller_s=time.perf_counter()-start, decision="wait", **detail)
            return None
        before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for victim, count in plan.items():
            self._release(victim, "partial_pressure", count)
        freed = self.kv_cache_manager.block_pool.get_num_free_blocks() - before
        result = self.original_allocate(request, num_new_tokens, *args, **kw)
        self._emit("exact_controller", program=self.program(request), **pressure,
                   selection=plan, physical_freed=freed, admitted=result is not None,
                   controller_s=time.perf_counter()-start, **detail)
        return result

    def _completion_release_budget(self, request, pressure):
        """Headroom for known current input and bounded remaining output.

        Uses the configured output cap, never unobserved future program turns.
        This is a separate reserve ablation, not the joint objective itself.
        """
        requests = list(self.running)
        if request not in requests:
            requests.append(request)
        extra = 0
        for r in requests:
            blocks = self.kv_cache_manager.get_blocks(r.request_id).blocks[0]
            owned = sum(not b.is_null for b in blocks)
            if not owned:
                _, hit, _ = self.kv_cache_manager.get_computed_blocks(r)
                owned = hit//self.block_size
            target = min(self.max_model_len,
                         r.num_prompt_tokens + max(0,r.max_tokens-1))
            extra += max(0,(target+self.block_size-1)//self.block_size-owned)
        # G already includes the current allocation gate. Add only growth
        # beyond that gate, then subtract today's actual free capacity.
        return max(pressure["D"], extra-pressure["free"])

    def _decision_signature(self, request):
        """Revisit a wait decision on structural changes, not each decode tick.

        Native allocation is still attempted every tick. A suppressed search
        never suppresses TTL expiry, handover, admission or deadlock fallback.
        """
        def stage(r):
            decode = r.num_computed_tokens >= r.num_prompt_tokens
            # Revisit physical allocation boundaries, and approaching output
            # completion, without paying for every single decode token.
            remaining = max(0, getattr(r, "max_tokens", 0)-getattr(r, "num_output_tokens", 0))
            return (r.request_id, r.num_computed_tokens//self.block_size,
                    decode, remaining//8 if decode else None)
        return (stage(request),tuple((p,len(pin.blocks)) for p,pin in self.pins.items()),
                tuple(stage(r) for r in self.running),
                tuple(r.request_id for q in (self.waiting,self.skipped_waiting) for r in q),
                self.kv_cache_manager.block_pool.get_num_free_blocks())

    def _residency_plan(self, request, rows, pressure, initial):
        start=time.perf_counter()
        snapshot=self._event_snapshot(rows)
        if self.controller in ("residency-next", "residency-joint"):
            by_id = {r.request_id:r for r in list(self.running)+list(self.waiting)+list(self.skipped_waiting)}
            wait = sum(self.waiting_samples)/len(self.waiting_samples) if self.waiting_samples else 0
            for row in snapshot["requests"]:
                request_row = by_id[row["id"]]
                meta = self.metadata(request_row)
                if meta.get("cont_tool") and not meta.get("cont_final"):
                    length = row["prompt"]+max(0,row["remaining"]-1)
                    reload = self.beta[0]+self.beta[1]*length+self.beta[2]*length*(length+1)/2
                    ttl,_ = self.policy_model.ttl(str(meta["cont_tool"]),wait*self.eta+reload,
                        meta.get("cont_oracle_gap") if self.policy_model.mode == "oracle" else None)
                    row["continuation"] = dict(ttl=ttl,
                        scenarios=self.policy_model.conditional(str(meta["cont_tool"]),0))
        snapshot["decode_by_seqs"]=self.profile.get("decode_step_by_seqs",{})
        plans=[initial]
        total_safe=sum(row["limit"] for row in rows)
        # Budget breakpoints are actual owner boundaries, not a constant
        # reserve. Include both original order and cheapest recovery victim.
        cumulative=0
        for row in rows[:3]:
            cumulative+=row["limit"]
            plans.append(self._latest_suffix_plan(rows,min(total_safe,max(pressure["D"],cumulative))))
        for row in rows:
            if row["limit"]>=pressure["D"]:
                plans.append({row["id"]:pressure["D"]})
                plans.append({row["id"]:row["limit"]})
        reference={} if any(r is not request for r in self.running) else None
        if self.controller == "residency-joint":
            # Keep Continuum's wait/whole-program reference if search is
            # unknown. Exact-D was previously a harmful implicit fallback.
            if reference is None:
                amount=0
                for row in rows:
                    amount+=row["limit"]
                    if amount>=pressure["D"]:
                        break
                reference=self._latest_suffix_plan(rows,amount)
            # Evaluate a wider reservation before filling the search budget
            # with many exact-D victim alternatives. Full owner release is
            # an important reference because FIFO often preserves its prefix.
            reserve=min(total_safe,self._completion_release_budget(request,pressure))
            prioritized=[self._latest_suffix_plan(rows,reserve)]
            prioritized += [{row["id"]:row["limit"]} for row in rows
                            if row["limit"]>=pressure["D"]]
            prioritized += [initial]
            plans=list({tuple(sorted(p.items())):p for p in prioritized}.values())[:4]
        plan,detail=choose_residency(plans,snapshot,initial,reference=reference,
            budget_s=float(os.environ.get("CONTINUUM_SEARCH_BUDGET_MS","12"))/1000,
            snapshot_s=time.perf_counter()-start,
            epsilon=self.profile.get("decision_error_s",0.0))
        # Full physical snapshots are diagnostic artifacts, not needed by the
        # serving path. Bound JSON serialization on the scheduler critical path.
        if self.controller != "residency-joint" or not getattr(self,"_event_snapshot_logged",False):
            self._emit("residency_snapshot",rows=rows,state=snapshot,D=pressure["D"],
                       initial=initial,selection=plan)
            self._event_snapshot_logged=True
        return plan,detail

    def _retention_plan(self, request, rows, pressure, initial, kw):
        start = time.perf_counter()
        now = time.monotonic()
        queued = {self.program(r): r for queue in (self.waiting, self.skipped_waiting)
                  for r in queue}
        capacity_mode = self.controller.startswith("capacity")
        if capacity_mode:
            queued.update({self.program(r): r for r in self.running})
        queued[self.program(request)] = request
        state = []
        for program in set(self.pins) | set(queued):
            pin = self.pins.get(program)
            r = queued.get(program)
            old_tokens = pin.tokens if pin else (r.num_prompt_tokens if r else 0)
            if r:
                if r is request:
                    cached = r.num_computed_tokens+kw.get("num_new_computed_tokens", 0)
                else:
                    # get_computed_blocks returns the total APC prefix, not
                    # an increment to the request's computed position.
                    _, cached, _ = self.kv_cache_manager.get_computed_blocks(r)
                    cached = max(cached, r.num_computed_tokens)
                scenarios = [(0.0, 0)]*3
            else:
                cached = 0
                for block_hash in pin.lookup_hashes[:old_tokens//self.block_size]:
                    if self.kv_cache_manager.block_pool.get_cached_block(block_hash, [0]) is None:
                        break
                    cached += self.block_size
                scenarios = self.policy_model.conditional(pin.tool, now-pin.started)
            active_rank = None
            if r is request and r.num_computed_tokens:
                active_rank = 1 if getattr(r, "_continuum_was_pinned", False) else 2
            state.append(dict(id=program, order=self.order[program], queued=r is not None,
                              old_tokens=old_tokens, cached_tokens=cached,
                              prompt_tokens=r.num_prompt_tokens if r else 0,
                              computed_tokens=r.num_computed_tokens if r else 0,
                              protected_blocks=len(pin.blocks) if pin else 0,
                              ttl_remaining=max(0, pin.deadline-now) if pin else 0,
                              preempted=r.status == RequestStatus.PREEMPTED if r else False,
                              active_rank=active_rank, scenarios=scenarios))
        state.sort(key=lambda row: row["order"])
        # The capacity-aware proxy includes decode occupancy; calibration is
        # read-only profile data and remains separate from measured timings.
        decode_by_seqs = self.profile.get("decode_step_by_seqs", {})
        if capacity_mode:
            for row in state:
                r = queued.get(row["id"])
                row.update(running=bool(r and r in self.running),
                       capacity=self._capacity(), max_seqs=self.max_num_running_reqs,
                       decode_step_s=self.profile.get("decode_step_s", .020),
                       decode_by_seqs=decode_by_seqs,
                       output_tokens=max(1, r.max_tokens-r.num_output_tokens) if r else self.profile.get("output_tokens", 24))
                if r and r in self.running:
                    row["active_rank"] = 1 if getattr(r,"_continuum_was_pinned",False) else 2
        extra = []
        if capacity_mode:
            total_safe = sum(row["limit"] for row in rows)
            # Extra capacity budgets correspond to actual program release
            # boundaries or the demand of the next waiting head, never a
            # fixed arbitrary block buffer.
            budgets = {pressure["D"]}
            cumulative = 0
            for row in rows[:3]:
                cumulative += row["limit"]
                budgets.add(max(pressure["D"], cumulative))
            waiting = sorted([r for r in queued.values() if r not in self.running and r is not request],
                             key=lambda r:(r.priority,self.order[self.program(r)]))
            if waiting:
                next_r = waiting[0]
                _, hits, _ = self.kv_cache_manager.get_computed_blocks(next_r)
                own_pin = self.pins.get(self.program(next_r))
                own = len(own_pin.blocks) if own_pin else 0
                budgets.add(pressure["D"]+max(0,(next_r.num_prompt_tokens+15)//16-own))
            for row in rows:
                budgets.add(max(pressure["D"],row["limit"]))
            amount_plans = []
            for count in sorted(budgets):
                count = min(count, total_safe)
                latest = self._latest_suffix_plan(rows, count)
                if self.controller == "capacity-tail":
                    extra.append(latest)
                else:
                    amount_plans.append(candidate_plans(rows,count,latest,max_plans=6))
            # Cover all release budgets before adding alternative victims.
            for index in range(6):
                extra.extend(plans[index] for plans in amount_plans if index < len(plans))
            # Reclaim during ordinary contention only when a complete model
            # predicts it beats waiting for the other running requests.
            reference = {} if any(r is not request for r in self.running) else None
        else:
            reference = None
        plan, detail = choose_retention(rows, pressure["D"], initial, state, self.block_size,
                                       self.max_num_scheduled_tokens, self.beta,
                                       budget_s=.003 if capacity_mode else .002,
                                       epsilon=self.profile.get("decision_error_s", 0.0),
                                       snapshot_s=time.perf_counter()-start,
                                       objective=self.controller if capacity_mode else (
                                           "local" if self.controller == "retention-local" else "queue"),
                                       extra_plans=extra, reference=reference,
                                       max_plans=24 if capacity_mode else 16)
        self._emit("retention_snapshot", rows=rows, state=state, D=pressure["D"],
                   initial=initial, selection=plan)
        return plan, detail

    def _partial_reclaim(self, request, num_new_tokens, excluded, kw):
        pool = self.kv_cache_manager.block_pool
        C = self._capacity()
        idle = self._idle_ids()
        owner_counts = {}
        for pin in self.pins.values():
            for b in pin.blocks:
                owner_counts[b.block_id] = owner_counts.get(b.block_id, 0)+1
        safe = {bid for bid in idle if pool.blocks[bid].ref_cnt == owner_counts[bid] and bid not in excluded}
        E = C-pool.get_num_free_blocks()-len(safe)
        local = request.num_computed_tokens+kw.get("num_new_computed_tokens",0)
        ext = kw.get("num_external_computed_tokens",0)
        target = min(request.num_tokens if kw.get("full_sequence_must_fit") else
                     local+ext+num_new_tokens+kw.get("num_lookahead_tokens",0), self.max_model_len)
        hit = kw.get("new_computed_blocks",self.kv_cache_manager.empty_kv_cache_blocks)
        G = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=target, new_computed_blocks=hit.blocks,
            num_encoder_tokens=0, total_computed_tokens=local+ext,
            num_local_computed_tokens=local, num_tokens_main_model=target,
            apply_admission_cap=bool(kw.get("full_sequence_must_fit")))
        watermark = self.kv_cache_manager.watermark_blocks if (
            kw.get("has_scheduled_reqs",True) and request.status in
            (RequestStatus.WAITING,RequestStatus.PREEMPTED)) else 0
        G += kw.get("reserved_blocks",0)+watermark
        D = max(0,G-pool.get_num_free_blocks())
        h = self.hysteresis
        hi = max(0,C-E-G)
        lo = max(0,hi-h)
        K = max(0,len(safe)-lo,D)
        if not K or (len(safe)<=hi and D==0):
            return
        snapshots = []
        for p,pin in self.pins.items():
            # Prefix-only backend: legal suffix release, private physical
            # blocks only. Shared/active/hit blocks are conservatively excluded.
            limit = 0
            for b in reversed(pin.blocks):
                if b.block_id not in safe or owner_counts[b.block_id] != 1:
                    break
                limit += 1
            if not limit:
                continue
            gaps = self.policy_model.conditional(pin.tool,time.monotonic()-pin.started)
            snapshots.append(dict(id=p, order=self.order[p], total=len(pin.blocks), limit=limit,
                                  old_tokens=pin.tokens, scenarios=gaps))
        # Do not sacrifice protected pages if the safe candidate set cannot
        # satisfy admission. Existing running work must first release capacity.
        if sum(r["limit"] for r in snapshots) < D:
            self._emit("controller_infeasible",C=C,E=E,G=G,D=D,K=K)
            return
        if not snapshots:
            return
        start = time.perf_counter()
        model_valid=self.profile.get("cost_model_valid",False)
        if self.controller.startswith("partial-j") and model_valid and not self.connector and len(snapshots)>1:
            state=self._event_snapshot(snapshots)
            if self.logfile and not getattr(self,"_event_snapshot_logged",False):
                self._emit("event_snapshot",state=state,rows=snapshots,K=K)
                self._event_snapshot_logged=True
            selection,detail=search_suffixes(snapshots,K,state,
                budget_s=float(os.environ.get("CONTINUUM_SEARCH_BUDGET_MS","10"))/1000,
                max_evals=32,epsilon=self.profile.get("decision_error_s",0.0))
        else:
            selection, detail = select_suffixes(snapshots,K,self.block_size,
                self.max_num_scheduled_tokens,self.beta,optimize=False)
        for victim,n in selection.items():
            self._release(victim,"partial_pressure",n)
        self._emit("controller", C=C,E=E,G=G,I=len(safe),D=D,h=h,K=K,
                   selection=selection, controller_s=time.perf_counter()-start,
                   model_valid=model_valid,**detail)

    def _event_snapshot(self, scenarios):
        """Read native physical metadata without changing hits or references."""
        pool=self.kv_cache_manager.block_pool
        now=time.monotonic()
        scenario_by_id={s["id"]:s for s in scenarios}
        def lookup(hashes):
            return [(hit[0].block_id if (hit:=pool.get_cached_block(h,[0])) else None)
                    for h in hashes]
        pins=[]
        for p,pin in self.pins.items():
            s=scenario_by_id.get(p)
            gaps=s["scenarios"] if s and "scenarios" in s else self.policy_model.conditional(pin.tool,now-pin.started)
            pins.append(dict(id=p,order=self.order[p],blocks=[b.block_id for b in pin.blocks],
                old_tokens=pin.tokens,deadline=max(0,pin.deadline-now),scenarios=gaps,
                lookup=lookup(pin.lookup_hashes[:pin.tokens//self.block_size])))
        requests=[]
        content={}
        for pin in pins:
            content.update({bid:(pin["id"],index) for index,bid in enumerate(pin["lookup"]) if bid is not None})
        active_ids={r.request_id for r in self.running}
        for r in list(self.running)+list(self.waiting)+list(self.skipped_waiting):
            blocks=self.kv_cache_manager.get_blocks(r.request_id).blocks[0]
            request_lookup=lookup(r.block_hashes)
            content.update({bid:(self.program(r),index) for index,bid in enumerate(request_lookup) if bid is not None})
            requests.append(dict(id=r.request_id,program=self.program(r),order=self.order[self.program(r)],
                arrival=min(0,r.arrival_time-time.time()),prompt=r.num_prompt_tokens,
                computed=r.num_computed_tokens,remaining=max(1,r.max_tokens-r.num_output_tokens),
                output_budget=r.max_tokens,
                allocated=[b.block_id for b in blocks if not b.is_null],lookup=request_lookup,
                running=r.request_id in active_ids,preempted=r.status==RequestStatus.PREEMPTED,
                was_pinned=getattr(r,"_continuum_was_pinned",False)))
        return dict(block_size=self.block_size,budget=self.max_num_scheduled_tokens,beta=self.beta,
            refs={b.block_id:b.ref_cnt for b in pool.blocks if not b.is_null},
            content=content,
            valid=[b.block_id for b in pool.blocks if b.block_hash is not None and not b.is_null],
            free_queue=[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
            pins=pins,requests=requests,max_seqs=self.max_num_running_reqs,
            future_output=requests[0]["output_budget"] if requests else self.profile.get("output_tokens",24),
            decode_step_s=self.profile.get("decode_step_s",.020))

    def shutdown(self):
        for p in list(self.pins):
            self._release(p,"shutdown")
        self._emit("shutdown", remaining_pins=len(self.pins))
        self.continuity_leases.clear()
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        super().shutdown()


class PressureControlledScheduler(ContinuumScheduler):
    """Prefix-compatible partial unpin with bounded completion-time search."""
    controller = "partial-j"


class StaticPartialScheduler(ContinuumScheduler):
    """Same pressure trigger and suffix backend, static selection ablation."""
    controller = "partial-static"


class TailDeficitScheduler(ContinuumScheduler):
    """Original victim order; reclaim only the native allocation deficit."""
    controller = "tail-d"


class RetentionScheduler(ContinuumScheduler):
    """Exact-deficit release with a bounded restoration/queue cost choice."""
    controller = "retention"


class LocalRetentionScheduler(RetentionScheduler):
    """Same recovery candidates, but without shared queue externality."""
    controller = "retention-local"


class AdmissionDeficitScheduler(TailDeficitScheduler):
    controller = "tail-admit"


class CapacityScheduler(RetentionScheduler):
    controller = "capacity"


class CapacityTailScheduler(CapacityScheduler):
    controller = "capacity-tail"


class ResidencyScheduler(CapacityScheduler):
    controller = "residency"


class ClosedLoopResidencyScheduler(ResidencyScheduler):
    controller = "residency-next"


class JointResidencyScheduler(ClosedLoopResidencyScheduler):
    """Conservative completion objective with a native-progress fast path."""
    controller = "residency-joint"


class CompletionReserveScheduler(AdmissionDeficitScheduler):
    """Reserve-only ablation; no restoration or completion-time search."""
    controller = "tail-reserve"


class ProgressResidencyScheduler(JointResidencyScheduler):
    """No-search ablation of progress rescue plus Continuum's default policy."""
    controller = "residency-progress"
