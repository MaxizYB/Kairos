"""Kairos scheduler implementation."""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

from kairos._runtime.compat.layered_baseline import LayeredScheduler
from kairos._runtime.compat.layered_policy import LAYERS, SEGMENTED_LAYERS, CONTINUATION_LAYERS
from kairos._runtime.layout.plan import Costs, query_segments, prefill_query_budget
from kairos._runtime.layout.query_budget import QueryBudgetMixin
from kairos._runtime.layout.execution_capacity import registered_caps, experiment_cap


class GappedScheduler(QueryBudgetMixin, LayeredScheduler):
    _FAIR_METHODS = dict(LayeredScheduler._FAIR_METHODS, **{
        'gapped-v21-partial': ('replica-joint-demand-bridge-admit-partial', 'gapped-v21'),
        'gapped-strong-partial': ('continuum-index', 'gapped-strong'),
        'gapped-tier-partial': ('continuum-index', 'gapped-tier'),
        'gapped-joint-partial': ('continuum-index', 'gapped-joint'),
    })

    def __init__(self, *args, **kwargs):
        self.gapped_enabled = False
        self.gapped_gpu_pages = {}
        self.gapped_virtual_finished = set()
        self.gapped_serial = 0
        super().__init__(*args, **kwargs)
        self.gapped_runner_capacity = self.max_num_running_reqs
        self.gapped_segment_capacity = False
        self.gapped_max_segments = 2
        self.execution_request_caps = registered_caps(
            os.environ.get('CONTINUUM_EXECUTION_REQUEST_CAPS'), self.gapped_runner_capacity)
        if self.execution_request_caps and os.environ.get('CONTINUUM_SEGMENT_RUNNER_CAPACITY'):
            raise ValueError('Real concurrency experiment cannot use the historical descriptor-only override')
        if os.environ.get('CONTINUUM_SEGMENT_RUNNER_CAPACITY') == '16':
            if self.max_num_running_reqs != 16:
                raise ValueError('Segment descriptors require the registered 16-slot runner')
            # All arms still admit at most eight real requests. The extra
            # worker descriptors never reserve extra KV or admit programs.
            self.max_num_running_reqs = 8
        from kairos._runtime.layout.gpu import install
        install(self)
        self.connector.gapped_gpu_lookup = self._gapped_snapshot

    def _gapped_snapshot(self, request):
        pool = self.kv_cache_manager.block_pool
        return {i for i, h in enumerate(request.block_hashes[:(request.num_prompt_tokens - 1) // 16])
                if pool.get_cached_block(h, [0]) is not None}

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        if self.execution_request_caps:
            self.max_num_running_reqs = experiment_cap(
                name, self.execution_request_caps, self.gapped_runner_capacity)
            self._next_execution_capacity_audit = 0.0
            self._emit('execution_request_capacity', real_request_cap=self.max_num_running_reqs,
                runner_descriptors=self.gapped_runner_capacity,
                registered_caps=self.execution_request_caps,
                program_order_and_ttl_unchanged=True, dynamic_policy=False)
        mode = getattr(self, '_fair_ablation', '') or ''
        self.gapped_enabled = mode.startswith('gapped-')
        self.gapped_gpu_pages.clear()
        if self.gapped_enabled and self.profile.get('cost_model_valid') is not True:
            raise RuntimeError('Gapped cost comparisons require a validated independent model profile')
        reload_path = os.environ.get('CONTINUUM_RELOAD_PROFILE')
        if reload_path and Path(reload_path).exists():
            calibration = json.loads(Path(reload_path).read_text())
            if 'gapped_load_chunk_seconds' in calibration:
                self.profile['gapped_load_chunk_seconds'] = calibration['gapped_load_chunk_seconds']
        beta = self.profile.get('beta', [0., .00003, 5e-9, .001])
        chunk_load = self.profile.get('gapped_load_chunk_seconds')
        self.gapped_costs = Costs(query=max(0., beta[1]),
            attention_pair=max(0., beta[2]), segment=max(0., beta[3]),
            load_token=(chunk_load / 256 if chunk_load is not None else
                        max(0., self.profile.get('reload_seconds_per_token', .00001))),
            load_call=(0. if chunk_load is not None else
                       max(0., self.profile.get('reload_fixed_seconds', .001))))
        self.connector.set_gapped(self.gapped_enabled,
                                  'joint' if mode in ('gapped-joint','gapped-pcv',
                                      'gapped-pcv-cpu','gapped-pcv-gpu','gapped-return','gapped-return-clean',
                                      'gapped-asym-tier','gapped-fast','gapped-bundle',
                                      'gapped-event','gapped-use','gapped-survival',
                                      'gapped-use-survival','gapped-enst',
                                      'gapped-use-discount','gapped-use-discount-survival',
                                      'gapped-cpu-use','gapped-cpu-insurance',
                                      'gapped-cpu-continuation',
                                      'gapped-layered-ea-joint',
                                      'gapped-layered-ea-continuation') + SEGMENTED_LAYERS else 'all',
                                  self.gapped_costs)
        self.connector.gapped_route_policy = (LAYERS[mode].recovery_route
                                              if mode in LAYERS else 'segmented')
        self.connector.gapped_route_cost = bool(mode in LAYERS and LAYERS[mode].cost_route)
        self.connector.gapped_restore_fence = (LAYERS[mode].restore_fence
                                               if mode in LAYERS else 'before-forward')
        self.connector.gapped_dependency_step = max(0., float(getattr(self, 'beta', beta)[0]))
        self.connector.gapped_recovery_plans = None
        self.gapped_segment_capacity = bool(mode in LAYERS and LAYERS[mode].segment_capacity)
        if self.gapped_segment_capacity and self.gapped_runner_capacity < 16:
            raise RuntimeError('Segment capacity requires the common 16-descriptor engine')
        self.gapped_max_segments = 8 if self.gapped_segment_capacity else 2
        self.connector.gapped_route_slots = self.gapped_max_segments
        if mode in LAYERS and LAYERS[mode].executable_route:
            from kairos._runtime.layout.executable_recovery import RecoveryPlans
            # Matched-unit ablation: reuse the incumbent's chunk-fence proxy.
            # Actual prefix uses a different protocol; this is deliberately
            # a shared proxy, not a claim that both paths have equal latency.
            self.connector.gapped_recovery_plans = RecoveryPlans(self.gapped_costs,
                self.connector.gapped_dependency_step, self.gapped_max_segments,
                prefix_load_token=self.gapped_costs.load_token,
                prefix_load_call=self.gapped_costs.load_call)
        if self.gapped_enabled and self.connector.gapped_route_policy == 'layout':
            self.connector.range_lookup = True
        self.connector.gapped_cpu_continuation=mode in ('gapped-cpu-continuation',) + tuple(
            name for name in CONTINUATION_LAYERS if LAYERS[name].host!='continuation-current')
        if self.gapped_enabled and mode not in LAYERS:
            self.gpu_victim_mode = 'joint' if mode == 'gapped-v21' else None
            self.publish_gpu_reuse = mode == 'gapped-v21'
            self.connector.set_official_admission('conditioned' if mode != 'gapped-v21' else None)
        self._emit('gapped_policy', enabled=self.gapped_enabled, mode=mode,
                   costs=self.gapped_costs.__dict__,
                   ttl_changed=name.startswith(('pcvu-m2-oracle-', 'pcvu-m2-zero-')),
                   load_calibration='actual-gapped-chunks' if chunk_load is not None else 'contiguous-bootstrap',
                   execution='shared-page virtual segments, FA2 causal attention',
                   adaptive_decode_threshold=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_THRESHOLD','0')),
                   adaptive_token_budget=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_TOKENS','512')),
                   fixed_prefill_budget=(LAYERS[mode].prefill_budget if mode in LAYERS else None),
                   recovery_route=self.connector.gapped_route_policy,
                   recovery_cost_route=self.connector.gapped_route_cost,
                   restore_fence=self.connector.gapped_restore_fence,
                   executable_recovery=bool(self.connector.gapped_recovery_plans),
                   real_request_capacity=self.max_num_running_reqs,
                   runner_descriptor_capacity=self.gapped_runner_capacity,
                   independent_segments=self.gapped_segment_capacity,
                   maximum_query_segments=self.gapped_max_segments,
                   adopt_load_hits=os.environ.get('CONTINUUM_GAPPED_ADOPT_LOAD_HITS','0') == '1',
                   defer_publication=os.environ.get('CONTINUUM_PCV_DEFER_PUBLICATION','0') == '1',
                   query_budget=os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET','0') == '1')

    def _replica_call(self, payload):
        mode = getattr(self, '_fair_ablation', '') or ''
        if payload.get('op') == 'experiment' and mode.startswith('gapped-'):
            if mode != 'gapped-v21' and mode not in LAYERS:
                payload = dict(payload, mode='observe', partial_admission=True,
                    hicache_reclaim=True, allocation_reclaim=False,
                    complementary_admission=False, demand_zero_reclaim=False,
                    demand_lru_reclaim=False, pressure_write_bridge=False,
                    duplicate_admission=False, official_admission_telemetry=True)
            payload['gapped_policy'] = mode
            reload_path = os.environ.get('CONTINUUM_RELOAD_PROFILE')
            calibration = (json.loads(Path(reload_path).read_text())
                           if reload_path and Path(reload_path).exists() else {})
            chunk_load = calibration.get('gapped_load_chunk_seconds')
            beta = self.profile.get('beta', [0., .00003, 5e-9, .001])
            payload['gapped_costs'] = dict(query=max(0., beta[1]),
                attention_pair=max(0., beta[2]), segment=max(0., beta[3]),
                load_token=(chunk_load / 256 if chunk_load is not None else
                            max(0., self.profile.get('reload_seconds_per_token', .00001))),
                load_call=(0. if chunk_load is not None else
                           max(0., self.profile.get('reload_fixed_seconds', .001))))
            if mode in LAYERS and LAYERS[mode].executable_cpu:
                payload['executable_config'] = dict(dependency_step=max(0., beta[0]),
                    slots=2, query_budget=2048,
                    prefix_load_token=calibration.get('reload_seconds_per_token', .00001),
                    prefix_load_call=calibration.get('reload_fixed_seconds', .001))
        return super()._replica_call(payload)

    def add_request(self, request):
        extra = request.sampling_params.extra_args or {}
        fixture = extra.get('cont_gapped_fixture')
        if fixture:
            if os.environ.get('CONTINUUM_GAPPED_FIXTURE') != '1':
                raise RuntimeError('Controlled layouts are disabled in benchmark runs')
            if isinstance(fixture,str):fixture=json.loads(fixture)
            self._switch_experiment(extra['cont_experiment'])
            pool = self.kv_cache_manager.block_pool
            keep = set(fixture['gpu_keep_chunks'])
            keep_pages = set(fixture.get('gpu_keep_pages', ()))
            page_fixture = 'gpu_keep_pages' in fixture
            removed = []
            for i, h in enumerate(request.block_hashes):
                found = pool.get_cached_block(h, [0])
                retained = i in keep_pages if page_fixture else i // 16 in keep
                if found is not None and not retained:
                    block = found[0]
                    if block.ref_cnt:
                        raise RuntimeError('Controlled layout requires unreferenced GPU pages')
                    pool._remove_cached_block_hashes(block)
                    removed.append(i)
            result = self._replica_call(dict(op='gapped_fixture_cpu',
                program=self.program(request), model=self.connector.scheduler_adapter.model_name,
                salt=str(request.cache_salt or ''), tokens=list(request.prompt_token_ids),
                keep=fixture['cpu_keep_chunks']))
            present = sorted({i // 16 for i, h in enumerate(request.block_hashes)
                              if pool.get_cached_block(h, [0]) is not None})
            present_pages = {i for i, h in enumerate(request.block_hashes)
                             if pool.get_cached_block(h, [0]) is not None}
            if (present_pages != keep_pages if page_fixture else set(present) != keep):
                raise RuntimeError('GPU fixture does not contain every requested retained chunk')
            self._emit('gapped_controlled_layout', gpu_keep_chunks=sorted(keep),
                cpu_keep_chunks=fixture['cpu_keep_chunks'], gpu_removed_pages=removed,
                gpu_present_chunks=present, cpu_present_chunks=result['cpu_present_chunks'],
                gpu_keep_pages=sorted(keep_pages) if page_fixture else None,
                gpu_present_pages=sorted(present_pages) if page_fixture else None,
                cpu_deleted_chunks=result['deleted'], synthetic=True)
        super().add_request(request)
        if self.gapped_enabled:
            # The segmented backend can merge intervals and rewrite adopted
            # full pages. This prototype validates one outstanding request per
            # isolated Program cache identity; concurrent shared-prefix writes
            # need an additional full-page CoW path, not an implicit alias.
            if not request.cache_salt or any(
                    other.request_id != request.request_id
                    and not other.is_finished()
                    and other.cache_salt == request.cache_salt
                    for other in self.requests.values()):
                raise RuntimeError('Gapped backend requires isolated, serial Program cache identities')
            from types import SimpleNamespace
            program = self.program(request)
            self.baseline_requests[program] = SimpleNamespace(program=program,
                token_ids=tuple(request.prompt_token_ids), lookup_hashes=tuple(request.block_hashes),
                cache_salt=str(request.cache_salt or ''), model=self.connector.scheduler_adapter.model_name)
            self._replica_call(dict(op='hicache_register', program=program,
                tokens=list(request.prompt_token_ids), salt=str(request.cache_salt or ''),
                model=self.connector.scheduler_adapter.model_name))

    def _hicache_allocate(self, request, num_new_tokens, *args, **kwargs):
        if self.gapped_enabled:
            # Most decode steps fit pages already owned by this request.
            # They change neither recoverable historical pages nor CPU-store
            # demand. Avoid a synchronous full-layout publication and PCV
            # state RPC when the native allocator cannot reclaim anything.
            blocks=self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks.get(
                request.request_id,())
            endpoint=(request.num_computed_tokens+num_new_tokens+
                      kwargs.get('num_lookahead_tokens',0))
            if (request.num_computed_tokens>=request.num_prompt_tokens
                    and not args and not kwargs.get('num_external_computed_tokens',0)
                    and not kwargs.get('num_new_computed_tokens',0)
                    and not kwargs.get('num_encoder_tokens',0)
                    and not kwargs.get('new_computed_blocks')
                    and math.ceil(endpoint/16)<=len(blocks)):
                return super()._hicache_allocate(request,num_new_tokens,*args,**kwargs)
            layouts = self._layouts()
            self._publish_layout(layouts)
            self._replica_call(dict(op='gapped_gpu_pages', programs={row['program']:
                [i for i, bid in enumerate(row['blocks']) if bid is not None] for row in layouts},
                **self._gapped_state_payload(layouts)))
        return super()._hicache_allocate(request, num_new_tokens, *args, **kwargs)

    def _gapped_state_payload(self, layouts):
        return {}

    def _layer_native_reuse(self, count):
        if not self.gapped_enabled or not count:
            return super()._layer_native_reuse(count)
        mode = getattr(self, '_fair_ablation', '')
        pool = self.kv_cache_manager.block_pool
        queue = pool.free_block_queue
        free = queue.get_all_free_blocks()
        if count > len(free):
            return self._layer_native_get_new_blocks(count)
        if all(page.block_hash is None for page in free[:count]):
            # Every such native head has zero recovery loss and no proof to
            # withdraw. All compared cost policies prefer it at the native
            # recency tie break; avoid scanning layouts and RPC on cold starts.
            return self._layer_native_get_new_blocks(count)
        layouts = self._layouts()
        started = time.perf_counter()
        if mode in ('gapped-strong', 'gapped-tier', 'gapped-joint'):
            cpu = self._replica_call(dict(op='cpu_residency'))['programs']
            ownership = {}
            gpu = {}
            for row in layouts:
                p = row['program']
                gpu[p] = {i for i, bid in enumerate(row['blocks']) if bid is not None}
                for i, bid in enumerate(row['blocks']):
                    if bid is not None:
                        ownership.setdefault(bid, []).append((p, i))
            page_cost = replace(self.gapped_costs, chunk=16)
            def scalar(page):
                value = 0.
                for p, i in ownership.get(page.block_id, ()):
                    compute = page_cost.chunk_compute(i)
                    backed = i // 16 in cpu.get(p, {}).get('cpu_chunks', ())
                    value += min(compute, self.gapped_costs.load_token * 16 +
                        self.gapped_costs.load_call / 16) if backed and mode != 'gapped-strong' else compute
                return value
            remaining = sorted(enumerate(free), key=lambda pair: (scalar(pair[1]), pair[0]))
            chosen = []
            for _ in range(count):
                if mode != 'gapped-joint' or not remaining or remaining[0][1].block_id not in ownership:
                    rank, page = remaining.pop(0)
                else:
                    # A bounded shortlist limits online controller work. Re-score
                    # after every action against the changed layout, not just
                    # one initial independent priority vector.
                    shortlist = remaining[:8]
                    def marginal(pair):
                        _, candidate = pair
                        loss = 0.
                        for p, i in ownership.get(candidate.block_id, ()):
                            chunks = set(cpu.get(p, {}).get('cpu_chunks', ()))
                            n = len(layouts[[r['program'] for r in layouts].index(p)]['chunks'])
                            before = self.gapped_costs.estimate_pages(gpu[p], chunks, n)[0]
                            after = self.gapped_costs.estimate_pages(gpu[p] - {i}, chunks, n)[0]
                            loss += max(0., after - before)
                        return loss, pair[0]
                    rank, page = min(shortlist, key=marginal)
                    remaining.remove((rank, page))
                chosen.append(page)
                for p, i in ownership.get(page.block_id, ()):
                    gpu[p].discard(i)
            if any(b.ref_cnt or b.is_null for b in chosen):
                raise RuntimeError('Recovery-cost policy selected a referenced GPU page')
            for b in chosen:
                queue.remove(b)
            queue.prepend_n(chosen)
            self._emit('gapped_gpu_reclaim', policy=mode, pages=count,
                changed_pages=len({b.block_id for b in chosen} - {b.block_id for b in free[:count]}),
                controller_s=time.perf_counter() - started)
        victims = [b.block_id for b in queue.get_all_free_blocks()[:count]]
        chosen_ids = set(victims)
        withdraw = {row['program']: [i for i, bid in enumerate(row['blocks']) if bid in chosen_ids]
                    for row in layouts}
        withdraw = {p: indices for p, indices in withdraw.items() if indices}
        if withdraw:
            self._replica_call(dict(op='gapped_gpu_withdraw', programs=withdraw))
            self._replica_call(dict(op='hicache_gpu_withdraw', programs={p:
                sorted({i // 16 for i in indices}) for p, indices in withdraw.items()}))
            self.hicache_layout_signature = None
        return self._layer_native_get_new_blocks(count)

    def schedule(self, throttle_prefills=False):
        # Worker request slots from prior virtual segments are reclaimed before
        # this step. These ids have no scheduler request or connector tracker.
        self.finished_req_ids.update(self.gapped_virtual_finished)
        self.gapped_virtual_finished.clear()
        normal_budget=self.max_num_scheduled_tokens
        threshold=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_THRESHOLD','0'))
        decode_count=sum(r.num_computed_tokens>=r.num_prompt_tokens for r in self.running)
        if self.gapped_enabled:
            layer=LAYERS.get(getattr(self,'_fair_ablation',''))
            self.max_num_scheduled_tokens=prefill_query_budget(normal_budget,decode_count,threshold,
                max(self.max_num_running_reqs,int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_TOKENS','512'))),
                layer.prefill_budget if layer else None)
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        finally:
            self.max_num_scheduled_tokens=normal_budget
        if not self.gapped_enabled or not output.total_num_scheduled_tokens:
            return output
        # Decode already uses the complete causal KV table and has no query
        # skipping. Keep native metadata and lifecycle intact rather than
        # rebuilding valid-page sets and virtual segments for every token.
        if all(self.requests[rid].num_computed_tokens-count>=
               self.requests[rid].num_prompt_tokens
               for rid,count in output.num_scheduled_tokens.items()):
            features=output._continuum_features
            self._emit('gapped_batch',Q=features[0],A=features[1],N=features[2],
                logical_Q=features[0],prefill_queries=0,decode_queries=features[0],
                controller_s=0.,fast_path='native-decode',
                query_budget_prefill_calls=getattr(self,'_gapped_prefill_plan_calls',None),
                query_budget_prefill_s=getattr(self,'_gapped_prefill_plan_s',None),
                impossible_plan_skips=getattr(self,'_gapped_impossible_plan_skips',None))
            return output
        from vllm.v1.core.sched.output import NewRequestData
        began = time.perf_counter()
        logical = dict(output.num_scheduled_tokens)
        runner_limit = (self.gapped_runner_capacity if self.gapped_segment_capacity
                        else self.max_num_running_reqs)
        extra_slots = runner_limit - len(logical)
        new = {item.req_id: item for item in output.scheduled_new_reqs}
        virtuals = []
        features = [0, 0, 0]
        prefill_queries = 0
        for rid, count in logical.items():
            request = self.requests[rid]
            end = request.num_computed_tokens
            begin = end - count
            valid = self.gapped_gpu_pages.get(rid, set()) | self.connector.valid_cpu_pages(rid)
            slots = extra_slots + 1
            if os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET','0') == '1':
                plan = getattr(self, '_gapped_step_plans', {}).get(rid)
                if plan is not None:
                    slots = min(slots, plan[3])
            segmented = self.connector.uses_gapped(rid)
            segments = (query_segments(begin, end, valid, slots=slots)
                        if segmented and begin < request.num_prompt_tokens else [(begin, end)])
            last = segments[-1]
            output.num_scheduled_tokens[rid] = last[1] - last[0]
            if rid in new:
                new[rid].num_computed_tokens = last[0]
            else:
                j = output.scheduled_cached_reqs.req_ids.index(rid)
                output.scheduled_cached_reqs.num_computed_tokens[j] = last[0]
            for a, b in segments[:-1]:
                self.gapped_serial += 1
                ident = f'{rid}::gquery:{self.gapped_serial}'
                data = NewRequestData.from_request(request,
                    self.kv_cache_manager.get_blocks(rid).get_block_ids())
                virtuals.append(replace(data, req_id=ident,
                    prompt_token_ids=request.prompt_token_ids[:b], num_computed_tokens=a))
                output.num_scheduled_tokens[ident] = b - a
                self.gapped_virtual_finished.add(ident)
            extra_slots -= len(segments) - 1
            for a, b in segments:
                features[0] += b - a
                features[1] += (a + b + 1) * (b - a) // 2
                features[2] += 1
                prefill_queries += max(0, min(b, request.num_prompt_tokens) - a)
            self._emit('gapped_segments', request=rid, program=self.program(request),
                step_id=output._continuum_step_serial,
                begin=begin, end=end, segments=segments, logical_tokens=count,
                actual_queries=sum(b - a for a, b in segments),
                prompt_tokens=request.num_prompt_tokens,
                returning=getattr(request, '_continuum_returning', False),
                gpu_valid_pages=len(self.gapped_gpu_pages.get(rid, set())),
                cpu_valid_pages=len(self.connector.valid_cpu_pages(rid)),
                segment_slots=slots,
                execution_route='segmented' if segmented else 'prefix',
                unbounded_query_count=(sum(b-a for a,b in query_segments(
                    begin,end,valid,slots=max(1,len(valid)+2)))
                    if segmented and begin<request.num_prompt_tokens and
                    os.environ.get('CONTINUUM_PCV_USE_AUDIT','0')=='1' else None))
        output.scheduled_new_reqs.extend(virtuals)
        # Native cascade metadata was computed before virtual sequences were
        # added. Its shared-prefix length may cross a shorter segment's causal
        # endpoint. Standard paged FA2 remains exact without cascade attention.
        output.num_common_prefix_blocks = [0] * len(output.num_common_prefix_blocks)
        output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
        if (output.total_num_scheduled_tokens > self.max_num_scheduled_tokens
                or extra_slots < 0 or len(logical) > self.max_num_running_reqs):
            raise RuntimeError('Segmented prefill exceeded native token/sequence budgets')
        output._gapped_logical_counts = logical
        output._continuum_features = features
        self._emit('gapped_batch', Q=features[0], A=features[1], N=features[2],
                   step_id=output._continuum_step_serial,
                   logical_Q=sum(logical.values()), prefill_queries=prefill_queries,
                   decode_queries=features[0] - prefill_queries,
                   query_budget_calls=getattr(self, '_gapped_budget_calls', None),
                   query_budget_available=getattr(self, '_gapped_budget_available', None),
                   query_budget_charge=sum(getattr(self, '_gapped_step_charges', {}).values()),
                   query_budget_prefill_calls=getattr(self,'_gapped_prefill_plan_calls',None),
                   query_budget_prefill_s=getattr(self,'_gapped_prefill_plan_s',None),
                   impossible_plan_skips=getattr(self,'_gapped_impossible_plan_skips',None),
                   controller_s=time.perf_counter() - began)
        return output

    def update_from_output(self, output, runner_output):
        # Native scheduler updates ONLY real request ids. The runner's sampled
        # array still has its original indexing, so no tensor slicing is needed.
        logical = getattr(output, '_gapped_logical_counts', None)
        if logical is not None:
            output.num_scheduled_tokens = logical
            self.connector.complete_step(getattr(output, '_gapped_load_session_ids', None))
        for rid in list(getattr(self, '_gapped_pending_close', ())):
            if not any(request == rid for request, _, _ in self.connector.gapped_inflight):
                self.connector.close_gapped(rid)
                self._gapped_pending_close.remove(rid)
        return super().update_from_output(output, runner_output)

    def _free_request_blocks(self, request):
        if self.gapped_enabled:
            rid = request.request_id
            if (getattr(self, 'native_async_prefix_execution', False) and any(
                    ident == rid for ident, _, _ in self.connector.gapped_inflight)):
                # Aborts are processed before their execution result. Keep
                # the CPU lease until that step has consumed and fenced it.
                # Native deferred-free fences independently retain GPU pages.
                if not hasattr(self, '_gapped_pending_close'):
                    self._gapped_pending_close = set()
                self._gapped_pending_close.add(rid)
            else:
                self.connector.close_gapped(rid)
            self.gapped_gpu_pages.pop(request.request_id, None)
        return super()._free_request_blocks(request)
