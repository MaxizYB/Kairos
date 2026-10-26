"""Kairos connector implementation."""
from dataclasses import dataclass
import json
import math
import os
import time

from kairos._runtime.compat.hicache_baseline import HiCacheMechanismConnector
from kairos._runtime.compat.range_lookup import RangeLookupAdapter
from kairos._runtime.compat.replica import rpc
from kairos._runtime.layout.plan import runs
from kairos._runtime.layout.execution_route import execution_route
from kairos._runtime.layout.executable_recovery import page_masks


@dataclass
class ChunkLookup:
    ident: str
    index: int
    hit: bool | None = None
    submitted: bool = False
    end_index: int | None = None
    matched_end: int | None = None

    @property
    def stop(self):
        return self.end_index if self.end_index is not None else self.index + 1

    def chunks(self):
        return range(self.index, self.matched_end if self.matched_end is not None else self.stop)


def lookup_ranges(indices, maximum=2):
    """Join consecutive selected chunks, bounded by logical 512-token tiles."""
    result=[]
    for begin,end in runs(indices):
        while begin<end:
            stop=min(end,((begin//maximum)+1)*maximum)
            result.append((begin,stop));begin=stop
    return result


def restore_limit(end, allocated_pages, coalesced):
    return min(math.ceil(end/16)*16,allocated_pages*16) if coalesced else end


def loadable_items(items, end, allocated_pages, coalesced):
    """The indivisible LOOKUP/RETRIEVE ranges executable by this step.

    A hit in the first chunk of a two-chunk session is not a usable hit
    until the entire matched session fits the allocated destination table.
    Budgeting and metadata construction must use the same boundary.
    """
    limit = restore_limit(end, allocated_pages, coalesced)
    return [item for item in items if item.hit
            and (item.matched_end or item.stop) * 256 <= limit]


def is_gapped_internal_id(ident):
    """IDs owned by segmented execution, never native scheduler requests."""
    _, separator, suffix = ident.rpartition('::')
    kind, colon, index = suffix.partition(':')
    return bool(separator and colon and kind in ('gquery', 'gload')
                and index.isdecimal())


class GappedConnector(HiCacheMechanismConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gapped_enabled = False
        self.gapped_policy = 'all'
        self.gapped_lookups = {}
        self.gapped_loaded = {}
        self.gapped_step_loads = []
        self.gapped_inflight = []
        self.gapped_gpu_lookup = None
        self.gapped_costs = None
        self.gapped_plan = None
        self.gapped_use_observe = None
        self.gapped_coalesce = False
        self.gapped_route_policy = 'segmented'
        self.gapped_routes = {}
        self.gapped_route_slots = 2
        self.gapped_route_cost = False
        self.gapped_dependency_step = 0.
        self.gapped_recovery_plans = None

    def set_gapped(self, enabled, policy, costs):
        if self.gapped_lookups or self.gapped_inflight:
            raise RuntimeError("Cannot reset with owned CPU lookup sessions")
        self.gapped_enabled, self.gapped_policy = enabled, policy
        self.gapped_costs = costs
        self.gapped_loaded.clear()
        self.gapped_routes.clear()
        self.gapped_coalesce = os.environ.get('CONTINUUM_GAPPED_COALESCE','0') == '1'

    def uses_gapped(self, rid):
        return self.gapped_enabled and self.gapped_routes.get(rid) != 'prefix'

    def on_new_request(self, request):
        if not self.gapped_enabled:
            return super().on_new_request(request)
        self._get_or_create_request_tracker(request)

    def _adapter(self):
        if not isinstance(self.scheduler_adapter, RangeLookupAdapter):
            from lmcache.integration.vllm.vllm_multi_process_adapter import send_lmcache_request
            from lmcache.v1.multiprocess.protocol import RequestType
            self.scheduler_adapter = RangeLookupAdapter(self.scheduler_adapter,
                send_lmcache_request, RequestType.LOOKUP)
        return self.scheduler_adapter

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self.uses_gapped(request.request_id):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        from vllm.v1.request import RequestStatus
        tracker = self._get_or_create_request_tracker(request)
        # Scheduler _free_request_blocks already closes the previous layout.
        # Do not cancel new asynchronous LOOKUPs on every poll while the
        # native request remains PREEMPTED waiting for its new allocation.
        adapter = self._adapter()
        rid = request.request_id
        if rid not in self.gapped_lookups:
            gpu = self.gapped_gpu_lookup(request)
            count = (request.num_prompt_tokens - 1) // 256
            gpu_chunks = {i for i in range(count)
                          if all(j in gpu for j in range(i * 16, (i + 1) * 16))}
            payload = dict(op='coverage',
                tokens=tracker.get_token_ids(), model=adapter.model_name,
                salt=tracker.cache_salt, kv_rank=16777472,
                local_tokens=num_computed_tokens)
            if getattr(self,'gapped_cpu_continuation',False):
                extra=request.sampling_params.extra_args or {}
                payload.update(pcv_continuation_program=str(extra.get('cont_program_id') or rid),
                               pcv_continuation_gpu_chunks=sorted(gpu_chunks))
            coverage = rpc(self._replica_socket,payload,timeout=5)
            cpu = set(coverage['cpu_present_chunk_indices']) & set(range(count))
            if self.gapped_use_observe is not None:self.gapped_use_observe(request,gpu,cpu)
            # Whole-chunk restore must not overwrite a native prefix page
            # already shared with other requests. Recompute/adopt the tail
            # of the partially hit chunk instead of loading over that prefix.
            eligible = {i for i in cpu if i * 256 >= num_computed_tokens}
            loads = eligible - gpu_chunks
            if self.gapped_policy == 'joint' and getattr(self, 'gapped_recovery_plans', None) is None:
                _, loads = self.gapped_costs.estimate_pages(gpu, eligible, count)
            if self.gapped_plan is not None:
                loads=self.gapped_plan(request,gpu,eligible,count,loads)
            if not set(loads)<=eligible:
                raise RuntimeError('Recovery planner requested an unavailable CPU chunk')
            if self.gapped_route_policy == 'layout' and rid not in self.gapped_routes:
                if getattr(self, 'gapped_recovery_plans', None) is not None:
                    planned_at = time.perf_counter()
                    summary = self.gapped_recovery_plans.get(page_masks(gpu, request.num_prompt_tokens),
                        cpu, request.num_prompt_tokens, num_computed_tokens)
                    choice = summary.diagnostic()
                    choice['controller_s'] = time.perf_counter() - planned_at
                    loads = summary.loads
                    if not loads <= eligible:
                        raise RuntimeError('Executable recovery chose an unavailable/overlapping CPU chunk')
                else:
                    choice = execution_route(num_computed_tokens, request.num_prompt_tokens,
                                         gpu, cpu, loads, slots=self.gapped_route_slots,
                                         costs=self.gapped_costs if self.gapped_route_cost else None,
                                         dependency_step=self.gapped_dependency_step)
                self.gapped_routes[rid] = choice['route']
                self.replica_emit('recovery_execution_route', request=rid,
                    **choice, source='current hash/committed-CPU snapshot; not consumption proof')
                if choice['route'] == 'prefix':
                    # Defer to native LOOKUP locks, APC revalidation and
                    # range RETRIEVE metadata. Publish the diagnostic snapshot
                    # once; neither this RPC nor the route promises a hit.
                    self._coverage_logged.add(rid)
                    tracker._continuum_cpu_prefix_tokens = coverage['cpu_prefix_tokens']
                    self.replica_emit('replica_coverage', request=rid, **coverage)
                    return super().get_num_new_matched_tokens(request, num_computed_tokens)
            ranges=(lookup_ranges(loads) if self.gapped_coalesce else [(i,i+1) for i in sorted(loads)])
            self.gapped_lookups[rid] = [ChunkLookup(f'{rid}::gload:{i}', i,end_index=end)
                                        for i,end in ranges]
            tracker._continuum_cpu_prefix_tokens = coverage['cpu_prefix_tokens']
            self.replica_emit('gapped_lookup', request=rid, gpu_pages=len(gpu),
                gpu_page_indices=sorted(gpu) if os.environ.get('CONTINUUM_PCV_USE_AUDIT','0')=='1' else None,
                cpu_chunks=sorted(cpu), planned_load_chunks=sorted(loads),
                lookup_calls=len(ranges),coalesced=self.gapped_coalesce,
                contiguous_local_tokens=num_computed_tokens, policy=self.gapped_policy)
        pending = False
        for item in self.gapped_lookups[rid]:
            if item.hit is not None:
                continue
            if not item.submitted:
                adapter.starts[item.ident] = item.index * 256
                # Limit LOOKUP to one chunk: native reader locks and session
                # accounting then have identical lookup/retrieve ranges.
                adapter.maybe_submit_lookup_request(item.ident,
                    tracker.get_token_ids()[:item.stop * 256],
                    tracker.cache_salt, tracker.request_configs)
                item.submitted = True
            result = adapter.check_lookup_result(item.ident)
            if result is None:
                pending = True
            else:
                item.matched_end=max(item.index,min(item.stop,result//256))
                item.hit = item.matched_end > item.index
                if not item.hit:
                    adapter.end_session(item.ident)
        if pending:
            return None, True
        tracker.num_vllm_hit_tokens = num_computed_tokens
        tracker.num_lmcache_hit_tokens = 0
        if self._official_admission is not None and rid not in self._official_admission.rule._ea_decided:
            extra = request.sampling_params.extra_args or {}
            skip, rows = self._official_admission.resolve(request, num_computed_tokens,
                (0, False), tracker._continuum_cpu_prefix_tokens,
                str(extra.get('cont_program_id') or request.cache_salt or rid))
            if skip:
                self._admission_skipped.add(rid)
            for row in rows:
                self.replica_emit('official_admission_decision', request=rid,
                    upstream_event=row.pop('event'), **row)
        # Gapped tokens are NEVER advertised as a continuous prefix.
        return 0, False

    def load_chunks(self, rid):
        return {i for item in self.gapped_lookups.get(rid, ()) if item.hit for i in item.chunks()}

    def valid_cpu_pages(self, rid):
        return {j for i in self.gapped_loaded.get(rid, set())
                for j in range(i * 16, (i + 1) * 16)}

    def budget_cpu_pages(self, rid, end):
        # Admission will allocate every destination page through `end`.
        # Do not credit a partial coalesced session as a completed transfer.
        items = loadable_items(self.gapped_lookups.get(rid, ()), end,
                               math.ceil(end / 16), self.gapped_coalesce)
        return self.valid_cpu_pages(rid) | {j for item in items
            for i in item.chunks() for j in range(i * 16, (i + 1) * 16)}

    def build_connector_meta(self, output):
        metadata = super().build_connector_meta(output)
        if not self.gapped_enabled:
            return metadata
        from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
        from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
        self.gapped_step_loads = []
        for rid, count in output.num_scheduled_tokens.items():
            if not self.uses_gapped(rid):
                continue
            tracker = self.request_trackers[rid]
            # Metadata is built before scheduler _update_after_schedule.
            start = next((item.num_computed_tokens for item in output.scheduled_new_reqs
                          if item.req_id == rid), None)
            if start is None:
                i = output.scheduled_cached_reqs.req_ids.index(rid)
                start = output.scheduled_cached_reqs.num_computed_tokens[i]
            end = start + count
            # A decode token can leave the logical tile one token short of
            # a chunk boundary. Native allocation already owns its final
            # physical page; loading that page's same-history KV is safe
            # before causal attention, even for the last <=15 future tokens.
            # Do not load any page beyond this request's actual block table.
            for item in loadable_items(self.gapped_lookups.get(rid, ()), end,
                    len(tracker.allocated_block_ids[0]), self.gapped_coalesce):
                if item.index not in self.gapped_loaded.get(rid, set()):
                    a, b = item.index * 256, (item.matched_end or item.stop) * 256
                    metadata.add_request_metadata(LMCacheMPRequestMetadata(item.ident,
                        'RETRIEVE', LoadStoreOp(token_ids=tracker.get_token_ids(),
                            block_ids=[tracker.allocated_block_ids[0][a // 16:b // 16]],
                            start=a, end=b), tracker.cache_salt, tracker.request_configs))
                    self.gapped_loaded.setdefault(rid, set()).update(item.chunks())
                    self.gapped_step_loads.extend((rid,item.ident,i) for i in item.chunks())
        metadata.gapped_load_ids = [ident for _, ident, _ in self.gapped_step_loads]
        metadata.gapped_restore_fence = getattr(self, 'gapped_restore_fence', 'before-forward')
        # SchedulerOutput is an immutable ownership record for this transfer
        # step. A later submission must never be drained by an earlier result.
        output._gapped_load_session_ids = tuple(dict.fromkeys(metadata.gapped_load_ids))
        self.gapped_inflight.extend(self.gapped_step_loads)
        return metadata

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        identities = set(getattr(metadata, 'gapped_load_ids', ()))
        custom = [item for item in metadata.requests if item.request_id in identities]
        metadata.requests[:] = [item for item in metadata.requests if item not in custom]
        super().start_load_kv(forward_context, **kwargs)
        if not custom:
            self._record_use_ready()
            return
        began = time.perf_counter()
        # CUDA's native LMCache-driven transport requires an IPC event even
        # for a synchronous restore. It orders writes after the engine's page
        # zeroing/CoW; the device-aware future then waits on the server's
        # completion event before attention consumes the restored pages.
        event = self.worker_adapter.create_recorded_event()
        for item in custom:
            self.worker_adapter.submit_retrieve_request(item.request_id, item.op,
                event, cache_salt=item.cache_salt, request_configs=item.request_configs)
        mode = getattr(metadata, 'gapped_restore_fence', 'before-forward')
        if mode == 'first-write' and self._defer_until_first_writer(forward_context, custom, began):
            return
        self._finish_gapped_load(custom, began, mode='before-forward')

    def _defer_until_first_writer(self, context, custom, began):
        """Overlap only eager work which cannot read/write a restored page.

        wait_for_layer_load is TOO LATE on this vLLM version: the unified
        KV-update op precedes that callback. Restored chunks can contain
        queries recomputed by segment merging or the logits anchor. Fence
        at entry to the first Attention.forward, before either op. No hook
        is added to ordinary decode, and CUDA graph replay falls back.
        """
        if context is None:
            return False
        from vllm.config import CUDAGraphMode
        if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
            return False
        layers = context.no_compile_layers
        first_name = next(iter(self.worker_adapter.kv_caches), None)
        first = layers.get(first_name)
        if first is None or not hasattr(first, 'register_forward_pre_hook'):
            raise RuntimeError('First-write restore requires a known first attention writer')
        if getattr(self, '_gapped_pending_load', None) is not None:
            raise RuntimeError('An earlier gapped restore still owns its consumption fence')
        self._gapped_pending_load = (custom, began)

        def before_write(module, args):
            # Remove before any subsequent layer/step; only this real eager
            # forward owns the pending CPU sessions and destination refs.
            self._gapped_writer_hook.remove()
            self._gapped_writer_hook = None
            pending, submitted = self._gapped_pending_load
            self._finish_gapped_load(pending, submitted, mode='first-write')
            self._gapped_pending_load = None

        self._gapped_writer_hook = first.register_forward_pre_hook(before_write, prepend=True)
        return True

    def _finish_gapped_load(self, custom, began, mode):
        fence_began = time.perf_counter()
        for item in custom:
            value = self.worker_adapter.retrieve_futures.pop(item.request_id, None)
            if value is None or not value[0].result(timeout=30):
                # Fail this private run before attention rather than consume
                # partially loaded KV or silently report a hit.
                raise RuntimeError('Gapped load failed before model attention')
            self.worker_adapter.retrieve_events.pop(item.request_id, None)
        import torch
        torch.cuda.current_stream().synchronize()
        wait = time.perf_counter() - began
        blocking = time.perf_counter() - fence_began
        chunk_count=sum((item.op.end-item.op.start)//256 for item in custom)
        self.replica_emit('gapped_load_fence', chunks=chunk_count, calls=len(custom),worker_wait_s=wait,
            blocking_wait_s=blocking, pre_fence_host_s=fence_began-began, fence_mode=mode)
        # Worker connectors do not own the scheduler's bound emit callback.
        # Record device-fenced waits separately, attributable by request IDs.
        path = os.environ.get('CONTINUUM_GAPPED_WORKER_LOG')
        if path:
            line = json.dumps(dict(event='gapped_load_fence', time=time.time(),
                requests=[item.request_id for item in custom], chunks=chunk_count,calls=len(custom),
                worker_wait_s=wait, blocking_wait_s=blocking,
                pre_fence_host_s=fence_began-began, fence_mode=mode,
                consumption_fenced=True)) + '\n'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)
        self._record_use_ready()

    def _record_use_ready(self):
        # Host interval boundary immediately after load fences and before
        # attention launches. No extra CUDA synchronization for instrumentation.
        if os.environ.get('CONTINUUM_PCV_USE_AUDIT','0')!='1':return
        from kairos._runtime.layout.worker import CURRENT_REQUESTS
        seen=getattr(self,'_use_seen',set())
        new=[rid for rid in CURRENT_REQUESTS if rid not in seen]
        if not new:return
        self._use_seen=seen|set(new)
        path=os.environ.get('CONTINUUM_GAPPED_WORKER_LOG')
        if path:
            line=json.dumps(dict(event='pcv_kv_ready_before_attention',time=time.time(),
                requests=new,consumption_timestamp='host boundary, not CUDA kernel start'))+'\n'
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            try:os.write(fd,line.encode())
            finally:os.close(fd)

    def get_finished(self, finished_req_ids):
        if getattr(self, '_gapped_pending_load', None) is not None:
            raise RuntimeError('Gapped restore did not reach its first-writer fence')
        # The model runner needs virtual query IDs in finished_req_ids to
        # reclaim its slots. They own no STORE or native scheduler request:
        # forwarding them makes eager LMCache report an immediate completed
        # STORE, which would free an unknown request in the scheduler.
        # Keep worker-slot cleanup intact while giving the transport only
        # real request IDs. Real async STORE lifetimes remain native.
        real_ids = {ident for ident in finished_req_ids
                    if not is_gapped_internal_id(ident)}
        sending, receiving = super().get_finished(real_ids)
        # Private RETRIEVEs are consumed/fenced in start_load_kv. Also drain
        # any adapter completion for their internal IDs here rather than
        # exposing a gload ID to the native async-receive state machine.
        return tuple(None if ids is None else
                     {ident for ident in ids if not is_gapped_internal_id(ident)}
                     for ids in (sending, receiving))

    def complete_step(self, identities=None):
        adapter = self._adapter()
        owned = ({ident for _, ident, _ in self.gapped_inflight}
                 if identities is None else set(identities))
        ended=set()
        for _, ident, _ in self.gapped_inflight:
            if ident not in owned:continue
            if ident in ended:continue
            ended.add(ident)
            adapter.end_session(ident)
            for items in self.gapped_lookups.values():
                for item in items:
                    if item.ident == ident:
                        item.submitted = False
        self.gapped_inflight[:] = [row for row in self.gapped_inflight if row[1] not in owned]

    def close_gapped(self, rid):
        if any(request == rid for request, _, _ in self.gapped_inflight):
            raise RuntimeError('Cannot release CPU session during an active restore')
        if self.gapped_lookups.get(rid):
            adapter = self._adapter()
            for item in self.gapped_lookups[rid]:
                if item.submitted and item.hit is not False:
                    adapter.end_session(item.ident)
        self.gapped_lookups.pop(rid, None)
        self.gapped_loaded.pop(rid, None)
        self.gapped_routes.pop(rid, None)
