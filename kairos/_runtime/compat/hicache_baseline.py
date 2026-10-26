"""Kairos hicache baseline implementation."""
from __future__ import annotations

from types import SimpleNamespace

from kairos._runtime.compat.cache_baselines import BaselineScheduler
from kairos._runtime.compat.hicache_policy import DeviceRulePort, SOURCE_SHA, CORE_BLOB, COMPONENT_BLOB
from kairos._runtime.compat.official_lazy import OfficialLazyConnector, OfficialLazyScheduler


class HiCacheMechanismConnector(OfficialLazyConnector):
    def set_hicache_mode(self, mode):
        self.hicache_writeback = mode == "hicache-writeback"
        if self.hicache_writeback:
            self.range_lookup = True
            self.bounded_repair = False
            self.async_repair = False

    def build_connector_meta(self, output):
        metadata = super().build_connector_meta(output)
        if getattr(self, "hicache_writeback", False):
            suppressed = [item for item in metadata.requests if item.direction == "STORE"
                          and not item.request_id.startswith("replica-repair-")]
            metadata.requests[:] = [item for item in metadata.requests if item not in suppressed]
            if suppressed:
                self.replica_emit("hicache_store_deferred", chunks=sum(
                    (item.op.end-item.op.start)//256 for item in suppressed))
        return metadata

    def has_pending_push_work(self):
        return super().has_pending_push_work() or self.has_pending_repairs()


class HiCacheMechanismScheduler(OfficialLazyScheduler):
    _FAIR_METHODS = dict(OfficialLazyScheduler._FAIR_METHODS, **{
        "hicache-duplicate-partial": ("continuum-index", "hicache-duplicate"),
        "hicache-writeback-partial": ("continuum-index", "hicache-writeback"),
    })

    def __init__(self, *args, **kwargs):
        from vllm.v1.core.sched.scheduler import Scheduler
        from kairos._runtime.compat.copy_wait_schedule import install_copy_wait_guard
        self.copy_wait_provenance = install_copy_wait_guard(Scheduler)
        self._hicache_copy_wait = False
        super().__init__(*args, **kwargs)
        self._native_baseline_allocate = self.original_allocate
        self.original_allocate = self._hicache_allocate
        self.baseline_kind = None
        self.baseline_requests = {}
        self.writeback_attempted = set()
        self.hicache_generations = {}
        self.hicache_selection = None
        self.hicache_layout_signature = None
        self.hicache_device = DeviceRulePort()

    def _replica_call(self, payload):
        if payload.get("op") == "experiment" and (getattr(self, "_fair_ablation", None) or "").startswith("hicache-"):
            payload = dict(payload, mode="observe", allocation_reclaim=False,
                partial_admission=True, complementary_admission=False,
                duplicate_admission=False, demand_zero_reclaim=False,
                pressure_write_bridge=False, demand_lru_reclaim=False,
                hicache_reclaim=True)
        return super()._replica_call(payload)

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        mode = getattr(self, "_fair_ablation", None)
        self.baseline_requests = {}
        self.writeback_attempted = set()
        self.hicache_generations = {}
        self.hicache_selection = None
        self.hicache_layout_signature = None
        self.baseline_kind = "cache-writeback" if mode == "hicache-writeback" else None
        self.connector.set_hicache_mode(mode)
        if mode and mode.startswith("hicache-"):
            if self.baseline_kind and self.connector.lazy_offload:
                raise ValueError("True write-back cannot also enable the lazy STORE manager")
            self.gpu_victim_mode = None
            self.publish_gpu_reuse = False
            self._emit("hicache_mechanism_policy", source_sha=SOURCE_SHA,
                core_blob=CORE_BLOB, component_blob=COMPONENT_BLOB,
                mode=mode, duplicate_two_pass=True, host_leaf_fallback=True,
                device_leaf_writeback=bool(self.baseline_kind), source_fence="native_refs_and_host_commit",
                backend="vLLM-LMCache-MP", gpu_page_tokens=16, cpu_chunk_tokens=256)
            self._emit("hicache_copy_wait_policy", **self.copy_wait_provenance)

    def add_request(self, request):
        super().add_request(request)
        mode = getattr(self, "_fair_ablation", None)
        if mode and mode.startswith("hicache-"):
            program = self.program(request)
            saved = SimpleNamespace(program=program, token_ids=tuple(request.prompt_token_ids),
                lookup_hashes=tuple(request.block_hashes), cache_salt=str(request.cache_salt or ""),
                model=self.connector.scheduler_adapter.model_name)
            self.baseline_requests[program] = saved
            self._replica_call(dict(op="hicache_register", program=program,
                tokens=list(saved.token_ids), salt=saved.cache_salt, model=saved.model))
            self.hicache_layout_signature = None

    def _layouts(self, cpu=None, programs=None):
        pool = self.kv_cache_manager.block_pool
        layouts = []
        for program, saved in self.baseline_requests.items():
            if programs is not None and program not in programs:
                continue
            indexed = self.gpu_index.rows.get(program) if self.gpu_index is not None else None
            if indexed is not None:
                # The common observer follows actual native hash insert/pop.
                # Copy the row: victim simulation must never mutate the index.
                blocks = list(indexed["blocks"])
            else:
                blocks = []
                for h in saved.lookup_hashes[:len(saved.token_ids)//16]:
                    found = pool.get_cached_block(h, [0])
                    blocks.append(found[0].block_id if found is not None else None)
            row = dict((cpu or {}).get(program, {}), program=program,
                       saved=saved, blocks=blocks)
            row["chunks"] = [blocks[i:i+16] for i in range(0, len(blocks)//16*16, 16)]
            layouts.append(row)
        return layouts

    def _publish_layout(self, layouts):
        pool = self.kv_cache_manager.block_pool
        programs = {}
        for row in layouts:
            covered, referenced = [], []
            for i, group in enumerate(row["chunks"]):
                if None not in group:
                    covered.append(i)
                    if any(pool.blocks[bid].ref_cnt > 0 for bid in group):
                        referenced.append(i)
            programs[row["program"]] = dict(covered=covered, referenced=referenced)
        signature = tuple((program, tuple(row["covered"]), tuple(row["referenced"]))
                          for program, row in programs.items())
        if signature == self.hicache_layout_signature:
            return
        self._replica_call(dict(op="hicache_gpu_layout", programs=programs))
        self.hicache_layout_signature = signature

    def _allocate(self, request, num_new_tokens, *args, **kw):
        self._hicache_copy_wait = False
        if self.baseline_kind:
            # This branch runs the already-tested capacity/deadlock handling;
            # copy waiting is not allowed to cancel unrelated Continuum pins.
            result = BaselineScheduler._allocate(self, request, num_new_tokens, *args, **kw)
            self._hicache_copy_wait = result is None and self.connector.has_pending_repairs()
            return result
        return super()._allocate(request, num_new_tokens, *args, **kw)

    def _hicache_allocate(self, request, num_new_tokens, *args, **kw):
        mode = getattr(self, "_fair_ablation", None)
        if mode and mode.startswith("hicache-"):
            if self.baseline_kind:
                if self._prepare_imminent_writeback(request, num_new_tokens, kw):
                    return None
            else:
                self._publish_layout(self._layouts())
        return self._native_baseline_allocate(request, num_new_tokens, *args, **kw)

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
        layouts = self._layouts()
        self._publish_layout(layouts)
        if needed > pool.get_num_free_blocks():
            self._end_hicache_transaction("native_capacity_unavailable")
            return False
        adopt = {b.block_id for group in groups for b in group if b.ref_cnt == 0}
        count = max(0, needed-len(adopt))
        if not count:
            self._end_hicache_transaction("no_reuse_needed")
            return False
        free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()
                if b.block_id not in adopt]
        selection_key = (request.request_id, request.num_computed_tokens,
                         num_new_tokens, tuple(sorted(adopt)), count)
        pending = self.hicache_selection
        if (pending and pending[0] == selection_key and set(pending[1]).issubset(free)
                and pending[2] == tuple(self.hicache_generations.get(bid, 0) for bid in pending[1])):
            selected = pending[1]
        else:
            self._end_hicache_transaction("selection_changed")
            selected = self.hicache_device.choose(free, layouts, count)
            self.hicache_selection = (selection_key, selected,
                tuple(self.hicache_generations.get(bid, 0) for bid in selected))
        chosen = set(selected)
        if not any(pool.blocks[bid].block_hash is not None for bid in chosen):
            return False
        cpu = self._replica_call(dict(op="cpu_residency"))["programs"]
        budget = 8
        queued = False
        for row in layouts:
            saved = row["saved"]
            host = set(cpu.get(row["program"], {}).get("cpu_chunks", []))
            indices = []
            for i, group in enumerate(row["chunks"]):
                # Hashes, not physical IDs, identify immutable content across
                # allocator generations. A failed backup is attempted once.
                key = (saved.cache_salt, i, tuple(saved.lookup_hashes[i*16:(i+1)*16]),
                       tuple((bid, self.hicache_generations.get(bid, 0)) for bid in group))
                if i in host:
                    self.writeback_attempted.discard(key)
                if (budget and None not in group and chosen.intersection(group)
                        and i not in host and key not in self.writeback_attempted):
                    indices.append(i)
                    self.writeback_attempted.add(key)
                    budget -= 1
            if not indices:
                continue
            saved.blocks = [pool.blocks[bid] if bid is not None else SimpleNamespace(block_id=-1)
                            for bid in row["blocks"]]
            if self.connector.queue_backup_repair(saved, indices, 16):
                # Protect the host destination for the complete demotion
                # transaction.  Without this, an earlier committed batch can
                # be selected by the next duplicate-first pass while the GPU
                # source is still resident, causing an unbounded D2H loop.
                self._replica_call(dict(op="hicache_hold",
                    program=row["program"], indices=indices))
                queued = True
                self._emit("hicache_demotion_writeback", program=row["program"],
                    chunks=indices, trigger="device_leaf_native_reuse", held_until_host_commit=True)
        return queued

    def _end_hicache_transaction(self, reason):
        if self.hicache_selection is not None:
            self._replica_call(dict(op="hicache_hold_release", reason=reason))
            self.hicache_selection = None

    def _get_new_blocks(self, count):
        mode = getattr(self, "_fair_ablation", None)
        if not (mode and mode.startswith("hicache-")) or not count:
            return super()._get_new_blocks(count)
        pool = self.kv_cache_manager.block_pool
        free = pool.free_block_queue.get_all_free_blocks()
        if count > len(free):
            return self.original_get_new_blocks(count)
        layouts = self._layouts()
        before = [b.block_id for b in free[:count]]
        pending = self.hicache_selection
        if (self.baseline_kind and pending and len(pending[1]) == count
                and set(pending[1]).issubset(b.block_id for b in free)):
            chosen = pending[1]
        else:
            chosen = (self.hicache_device.choose([b.block_id for b in free], layouts, count)
                      if self.baseline_kind else before)
        victims = set(chosen)
        withdraw = {row["program"]: [i for i, group in enumerate(row["chunks"])
                    if victims.intersection(group)] for row in layouts}
        # Synchronous withdrawal precedes every native hash removal: a stale
        # GPU snapshot cannot certify duplicates after the pages are reused.
        withdraw = {p: indices for p, indices in withdraw.items() if indices}
        if withdraw:
            self._replica_call(dict(op="hicache_gpu_withdraw", programs=withdraw))
            self.hicache_layout_signature = None
        if chosen != before:
            blocks = [pool.blocks[bid] for bid in chosen]
            for block in blocks:
                if block.ref_cnt or block.is_null:
                    raise RuntimeError("HiCache device walk selected a referenced/null native page")
                pool.free_block_queue.remove(block)
            pool.free_block_queue.prepend_n(blocks)
        self._emit("hicache_gpu_reuse", pages=count, changed_pages=len(victims-set(before)),
            device_leaf_walk=bool(self.baseline_kind), withdrawn_chunks=sum(map(len, withdraw.values())))
        for bid in chosen:
            self.hicache_generations[bid] = self.hicache_generations.get(bid, 0)+1
        result = self.original_get_new_blocks(count)
        self._end_hicache_transaction("native_reuse_complete")
        return result
