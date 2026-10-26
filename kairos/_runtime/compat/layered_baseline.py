"""Kairos layered baseline implementation."""
from types import SimpleNamespace
import time

from kairos._runtime.compat.hicache_baseline import HiCacheMechanismScheduler
from kairos._runtime.compat.hicache_policy import SOURCE_SHA, COMPONENT_BLOB, DeviceRulePort
from kairos._runtime.compat.layered_policy import LAYERS, cpu_flags, withdraw_chunks


class LayeredScheduler(HiCacheMechanismScheduler):
    _FAIR_METHODS = dict(HiCacheMechanismScheduler._FAIR_METHODS, **{
        name + "-partial": (
            "replica-joint-demand-bridge-admit-partial" if layer.gpu else "continuum-index", name)
        for name, layer in LAYERS.items()
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_native_get_new_blocks = self.original_get_new_blocks
        self.original_get_new_blocks = self._layer_native_reuse
        self._cached_device_leaf = None

    def _replica_call(self, payload):
        layer = LAYERS.get(getattr(self, "_fair_ablation", None))
        if payload.get("op") == "experiment" and layer:
            payload = dict(payload, **cpu_flags(layer))
            if layer.host != "joint":
                payload["mode"] = "observe"
        return super()._replica_call(payload)

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        layer = LAYERS.get(getattr(self, "_fair_ablation", None))
        if layer:
            self.connector.set_official_admission(layer.admission, threshold=8)
            self.gpu_victim_mode = "joint" if layer.gpu and not layer.recovery_gpu else None
            self.publish_gpu_reuse = layer.gpu and not layer.recovery_gpu
            self._emit("layered_policy", mode=self._fair_ablation,
                official_ea=layer.admission, official_ea_threshold=8,
                official_hicache_host=layer.host == "hicache",
                joint_gpu=layer.gpu and not layer.recovery_gpu,
                segmented_recovery_gpu=layer.recovery_gpu, chunk_admission=layer.chunk_admission,
                recovery_weight=layer.recovery_weight if layer.recovery_gpu else None,
                official_device_leaf=layer.device_leaf,
                leaf_metadata_reused=layer.leaf_reuse_nodes,
                ttl_changed=False, additional_d2h=False, **cpu_flags(layer))

    def add_request(self, request):
        super().add_request(request)
        layer = LAYERS.get(getattr(self, "_fair_ablation", None))
        if layer and layer.host == "hicache":
            program = self.program(request)
            saved = SimpleNamespace(program=program, token_ids=tuple(request.prompt_token_ids),
                lookup_hashes=tuple(request.block_hashes), cache_salt=str(request.cache_salt or ""),
                model=self.connector.scheduler_adapter.model_name)
            self.baseline_requests[program] = saved
            self._replica_call(dict(op="hicache_register", program=program,
                tokens=list(saved.token_ids), salt=saved.cache_salt, model=saved.model))
            self.hicache_layout_signature = None

    def _hicache_allocate(self, request, num_new_tokens, *args, **kwargs):
        layer = LAYERS.get(getattr(self, "_fair_ablation", None))
        if layer and layer.host == "hicache":
            self._publish_layout(self._layouts())
            return self._native_baseline_allocate(request, num_new_tokens, *args, **kwargs)
        return super()._hicache_allocate(request, num_new_tokens, *args, **kwargs)

    def _layer_native_reuse(self, count):
        layer = LAYERS.get(getattr(self, "_fair_ablation", None))
        if layer and layer.host == "hicache" and count:
            # This callback runs AFTER any GPU preference has reordered the
            # queue. Withdraw those actual victims, never the original head.
            queue = self.kv_cache_manager.block_pool.free_block_queue
            if count <= queue.num_free_blocks:
                if layer.device_leaf:
                    started = time.perf_counter()
                    pool = self.kv_cache_manager.block_pool
                    free = queue.get_all_free_blocks()
                    before = [b.block_id for b in free[:count]]
                    fast_path = bool(layer.leaf_reuse_nodes
                                     and all(b.block_hash is None for b in free[:count]))
                    if fast_path:
                        # Every native head page is uncached. The official
                        # adapter consumes these unknown pages in queue order
                        # before any device leaf, so its result is unchanged.
                        chosen = before
                    else:
                        if layer.leaf_reuse_nodes and self._cached_device_leaf is None:
                            self._cached_device_leaf = DeviceRulePort(reuse_nodes=True)
                        walk = self._cached_device_leaf if layer.leaf_reuse_nodes else self.hicache_device
                        chosen = walk.choose([b.block_id for b in free], self._layouts(), count)
                    if len(chosen) != count or len(set(chosen)) != count:
                        raise RuntimeError("Official device walk changed allocation demand")
                    if chosen != before:
                        for bid in chosen:
                            page = pool.blocks[bid]
                            if page.ref_cnt or page.is_null:
                                raise RuntimeError("Official device walk selected a referenced/null page")
                            queue.remove(page)
                        queue.prepend_n([pool.blocks[bid] for bid in chosen])
                    if queue.num_free_blocks != len(free):
                        raise RuntimeError("Official device walk changed free capacity")
                    self._emit("layered_device_leaf", source_sha=SOURCE_SHA,
                        component_blob=COMPONENT_BLOB, pages=count,
                        changed_pages=len(set(chosen)-set(before)),
                        controller_s=time.perf_counter()-started,
                        extra_gpu_references=0, extra_copy_bytes=0,
                        writeback=False, metadata_reused=layer.leaf_reuse_nodes,
                        fast_path="uncached-native-head" if fast_path else None)
                block = queue.fake_free_list_head.next_free_block
                victims = []
                for _ in range(count):
                    victims.append(block.block_id)
                    block = block.next_free_block
                withdraw = withdraw_chunks(self.gpu_index, victims)
                ended = set(self.baseline_requests) - set(self.gpu_index.rows)
                if ended:
                    chosen = set(victims)
                    for row in self._layouts():
                        if row["program"] in ended:
                            indices = [i for i, group in enumerate(row["chunks"])
                                       if chosen.intersection(group)]
                            if indices:
                                withdraw[row["program"]] = indices
                if withdraw:
                    self._replica_call(dict(op="hicache_gpu_withdraw", programs=withdraw))
                    self.hicache_layout_signature = None
        return self._layer_native_get_new_blocks(count)
