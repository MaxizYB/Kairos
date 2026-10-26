"""Kairos cache baselines implementation."""
from __future__ import annotations

import re
from types import SimpleNamespace

from kairos._runtime.compat.scheduler import ContinuumScheduler
from kairos._runtime.compat.replica_connector import ReplicaConnector
from kairos._runtime.compat.writeback_plan import plan_writebacks


def canonical(name):
    name = name.split("~", 1)[0].removesuffix("-repeat")
    return re.sub(r"-(?:a|b|c)$", "", name).removesuffix("-partial")


class BaselineConnector(ReplicaConnector):
    def set_replica_mode(self, name):
        super().set_replica_mode(name)
        self.ordinary_writeback = name in ("cache-writeback", "cache-exclusive")
        if self.ordinary_writeback:
            self.range_lookup = True
            # Safety cap is applied by the scheduler, not by prefix repair.
            self.bounded_repair = False
            self.async_repair = False

    def build_connector_meta(self, output):
        metadata = super().build_connector_meta(output)
        if getattr(self, "ordinary_writeback", False):
            suppressed = [m for m in metadata.requests if m.direction == "STORE"
                          and not m.request_id.startswith("replica-repair-")]
            metadata.requests = [m for m in metadata.requests if m not in suppressed]
            if suppressed:
                self.replica_emit("baseline_store_deferred", chunks=sum(
                    (m.op.end-m.op.start)//256 for m in suppressed))
        return metadata


class BaselineScheduler(ContinuumScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_baseline_allocate = self.original_allocate
        self.original_allocate = self._baseline_allocate
        self.baseline_kind = None
        self.baseline_requests = {}
        self.writeback_attempted = set()
        self.exclusive_attempted = set()

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        self.baseline_kind = canonical(name)
        self.baseline_requests = {}
        self.writeback_attempted = set()
        self.exclusive_attempted = set()
        if self.baseline_kind in ("cache-writeback", "cache-exclusive"):
            # Ordinary LRU makes room for demand stores, not a TTL-aware
            # CPU replacement rule. Same partial-admission fallback as v19.
            self._replica_call(dict(op="experiment", name=name, mode="observe",
                allocation_reclaim=True, partial_admission=True,
                tool_history=self.profile.get("tool_history", {}), beta=self.beta,
                reload_per_token=self.profile.get("reload_seconds_per_token", 0)))

    def add_request(self, request):
        super().add_request(request)
        if self.baseline_kind in ("cache-writeback", "cache-exclusive"):
            self.baseline_requests[self.program(request)] = SimpleNamespace(
                program=self.program(request), token_ids=tuple(request.prompt_token_ids),
                lookup_hashes=tuple(request.block_hashes),
                cache_salt=str(request.cache_salt or ""),
                model=self.connector.scheduler_adapter.model_name)

    def _baseline_allocate(self, request, num_new_tokens, *args, **kwargs):
        if self.baseline_kind in ("cache-writeback", "cache-exclusive"):
            if self._prepare_imminent_writeback(request, num_new_tokens, kwargs):
                return None
        return self._native_baseline_allocate(request, num_new_tokens, *args, **kwargs)

    def _allocate(self, request, num_new_tokens, *args, **kw):
        if self.baseline_kind not in ("cache-writeback", "cache-exclusive"):
            return super()._allocate(request, num_new_tokens, *args, **kw)
        if request.num_computed_tokens == 0:
            request._continuum_was_pinned = self._has_continuity_priority(self.program(request))
        computed = kw.get("new_computed_blocks")
        excluded = {b.block_id for group in computed.blocks for b in group} if computed else set()
        stalled = not self.running or (len(self.running) == 1 and self.running[0] is request)
        # Copy waiting is NOT a physical-capacity failure. Do not cancel more
        # Continuum pins, nor let native allocation substitute uncopied pages.
        if self._prepare_imminent_writeback(request, num_new_tokens, kw):
            self._emit("baseline_writeback_wait", request=request.request_id,
                       reason="source_copy_or_host_commit")
            return None
        result = self._native_baseline_allocate(request, num_new_tokens, *args, **kw)
        if result is None and self.pins and stalled:
            pressure = self._required_capacity(request, num_new_tokens, kw)
            self._emit("allocation_failed", program=self.program(request), **pressure,
                       pin_blocks=len(self._idle_ids()))
            for victim in sorted(list(self.pins), key=lambda x: self.order[x], reverse=True):
                if any(b.block_id in excluded for b in self.pins[victim].blocks):
                    continue
                self._release(victim, "deadlock_fallback")
                if self._prepare_imminent_writeback(request, num_new_tokens, kw):
                    self._emit("baseline_writeback_wait", request=request.request_id,
                               reason="source_copy_after_native_capacity_fallback")
                    return None
                result = self._native_baseline_allocate(request, num_new_tokens, *args, **kw)
                if result is not None:
                    break
        program = self.program(request)
        if result is not None and program in self.pins:
            self._release(program, "handover_after_allocation")
        if result is not None:
            self._clear_continuity_lease(program, "handover")
        return result

    def _prepare_imminent_writeback(self, request, num_new_tokens, kw):
        if self.connector.has_pending_repairs():
            return True
        pool = self.kv_cache_manager.block_pool
        computed = kw.get("new_computed_blocks")
        groups = computed.blocks if computed else self.kv_cache_manager.empty_kv_cache_blocks.blocks
        local = request.num_computed_tokens + kw.get("num_new_computed_tokens", 0)
        external = kw.get("num_external_computed_tokens", 0)
        main = min(local+external, self.max_model_len)+num_new_tokens
        needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(main+kw.get("num_lookahead_tokens", 0), self.max_model_len),
            new_computed_blocks=groups, num_encoder_tokens=kw.get("num_encoder_tokens", 0),
            total_computed_tokens=local+external, num_local_computed_tokens=local,
            num_tokens_main_model=main)
        if needed > pool.get_num_free_blocks():
            return  # No speculative write when native admission already fails.
        adopt = {b.block_id for group in groups for b in group if b.ref_cnt == 0}
        count = max(0, needed-len(adopt))
        free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()
                if b.block_id not in adopt]
        if not count or not any(pool.blocks[bid].block_hash is not None for bid in free[:count]):
            return
        held = sum(len(ids) for ids in self.connector._repair_refs.values())
        budget = min(8, max(0, (256-held)//16))
        if not budget:
            return
        cpu = self._replica_call(dict(op="cpu_residency"))["programs"]
        layouts = []
        for program, saved in self.baseline_requests.items():
            ids = []
            for h in saved.lookup_hashes[:len(saved.token_ids)//16]:
                block = pool.get_cached_block(h, [0])
                ids.append(block[0].block_id if block is not None else None)
            row = cpu.get(program, {})
            layouts.append(dict(program=program, salt=saved.cache_salt, saved=saved,
                state=row.get("state"), cpu_chunks=row.get("cpu_chunks", []),
                chunks=[ids[i:i+16] for i in range(0, len(ids)//16*16, 16)]))
        queued = False
        for row, indices in plan_writebacks(free, layouts, count, self.writeback_attempted, budget):
            saved = row["saved"]
            ids = [bid for group in row["chunks"] for bid in group]
            # Sparse block_ids outside selected chunks are never accessed by
            # queue_backup_repair or the worker; keep absolute positions.
            saved.blocks = [pool.blocks[bid] if bid is not None else SimpleNamespace(block_id=-1)
                            for bid in ids]
            if self.connector.queue_backup_repair(saved, indices, 16):
                queued = True
                for i in indices:
                    self.writeback_attempted.add((saved.cache_salt, i, tuple(row["chunks"][i])))
                self._emit("baseline_writeback", program=row["program"], chunks=indices,
                    trigger="imminent_native_reuse", held_until_host_commit=True)
        return queued

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        if self.baseline_kind == "cache-exclusive":
            for request in list(self.running):
                blocks = list(self.kv_cache_manager.get_blocks(request.request_id).blocks[0])
                end = 0
                for block in blocks[:min(request.num_computed_tokens, request.num_prompt_tokens)//16]:
                    if block.is_null or block.block_hash is None or block.ref_cnt <= 0:
                        break
                    end += 16
                end -= end % 256
                if not end:
                    continue
                key = (request.request_id, end)
                if key in self.exclusive_attempted:
                    continue
                deleted = self._replica_call(dict(op="exclusive_drop",
                    tokens=list(request.prompt_token_ids[:end]),
                    salt=str(request.cache_salt or ""),
                    model=self.connector.scheduler_adapter.model_name, kv_rank=16777472))
                if not deleted["locked"]:
                    self.exclusive_attempted.add(key)
        return result
