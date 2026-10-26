"""Kairos recovery scheduler implementation."""
from dataclasses import replace
import math
import os
import time

from kairos._runtime.layout.pcv import Program
from kairos._runtime.layout.recovery import ChunkRecovery, choose_gpu_bundles, continuation_sensitivities
from kairos._runtime.layout.next_use import USE_MODES
from kairos._runtime.compat.layered_policy import SEGMENTED_LAYERS, LAYERS
from kairos._runtime.layout.compact_recovery import MaskRecovery, choose_mask_bundles


RECOVERY_MODES = ('gapped-fast', 'gapped-bundle', 'gapped-event')
DEFER_MODES = RECOVERY_MODES + ('gapped-return', 'gapped-return-clean', 'gapped-asym', 'gapped-asym-tier', 'gapped-cpu-continuation',
                              ) + SEGMENTED_LAYERS + USE_MODES


class RecoverySchedulerMixin:
    def __init__(self, *args, **kwargs):
        self._recovery_context = None
        self._recovery_cost_cache = {}
        self._sensitivity_signature = None
        self._sensitivity_weights = {}
        self._recovery_publication_signature = None
        self._event_history_signature = None
        self._control_client = None
        self._control_stream_calls = 0
        self._control_stream_s = 0.
        super().__init__(*args, **kwargs)

    def _switch_experiment(self, name):
        if name != self.experiment:
            self._close_control_stream()
            self._recovery_context = None
            self._recovery_cost_cache = {}
            self._sensitivity_signature = None
            self._sensitivity_weights = {}
            self._recovery_publication_signature = None
            self._event_history_signature = None
        return super()._switch_experiment(name)

    def _close_control_stream(self):
        client = getattr(self, '_control_client', None)
        if client is not None:
            client.close()
            self._control_client = None
            self._emit('replica_control_stream', calls=self._control_stream_calls,
                       controller_s=self._control_stream_s,
                       mutation_retry=False, acknowledgement='one per operation')
            self._control_stream_calls = 0
            self._control_stream_s = 0.

    def _replica_transport(self, path, payload):
        layer = LAYERS.get(getattr(self, '_fair_ablation', ''))
        use_stream = (layer is not None and layer.control_stream) or (
            layer is not None and layer.control_readonly_stream and
            payload.get('op') == 'gpu_history')
        if not use_stream:
            self._close_control_stream()
            return super()._replica_transport(path, payload)
        from kairos._runtime.compat.replica import ControlClient
        client = getattr(self, '_control_client', None)
        if client is None:
            client = self._control_client = ControlClient(path)
        elif client.path != str(path):
            raise RuntimeError('Control helper changed within a method')
        started = time.perf_counter()
        response = client.call(payload)
        self._control_stream_calls += 1
        self._control_stream_s += time.perf_counter() - started
        return response

    def _observe_gpu_history(self):
        layer = LAYERS.get(getattr(self, '_fair_ablation', ''))
        if layer is None or not layer.event_gpu_history:
            return super()._observe_gpu_history()
        started = time.perf_counter()
        prefixes = self.gpu_index.prefixes()
        signature = tuple((p, n, self.gpu_index.rows[p]['manifest_revision'])
                          for p, n in prefixes.items())
        if signature == self._event_history_signature:
            return
        # Native hash insertion/removal maintains this index. Every tracked
        # page chosen by this compact arm is synchronously withdrawn below
        # before actual reuse; retirement makes the CPU history finished.
        self._replica_call(dict(op='gpu_history', prefixes=prefixes, event_tracked=True))
        self._event_history_signature = signature
        self._emit('gpu_history_observed', prefixes=prefixes,
            controller_s=time.perf_counter()-started, advisory_only=True,
            publication='native-prefix-change; synchronous pre-reuse withdrawal')

    def _hicache_allocate(self, request, num_new_tokens, *args, **kwargs):
        mode = getattr(self, '_fair_ablation', '')
        if (self.gapped_enabled and mode in DEFER_MODES
                and os.environ.get('CONTINUUM_PCV_DEFER_PUBLICATION', '0') == '1'):
            # The native allocator checks admission capacity before invoking
            # get_new_blocks. A failed attempt, or an already-owned extension,
            # cannot overwrite any cached page. Publish at that callback,
            # after its exact demand is known and before any reuse occurs.
            return self._native_baseline_allocate(request, num_new_tokens, *args, **kwargs)
        if not self.gapped_enabled or mode not in RECOVERY_MODES:
            return super()._hicache_allocate(request, num_new_tokens, *args, **kwargs)
        blocks = self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks.get(
            request.request_id, ())
        endpoint = request.num_computed_tokens + num_new_tokens + kwargs.get('num_lookahead_tokens', 0)
        owns = (request.num_computed_tokens >= request.num_prompt_tokens and not args
                and not kwargs.get('num_external_computed_tokens', 0)
                and not kwargs.get('num_new_computed_tokens', 0)
                and not kwargs.get('num_encoder_tokens', 0)
                and not kwargs.get('new_computed_blocks')
                and math.ceil(endpoint / 16) <= len(blocks))
        if owns:
            return self._native_baseline_allocate(request, num_new_tokens, *args, **kwargs)
        self._recovery_context = self._recovery_snapshot()
        try:
            return self._native_baseline_allocate(request, num_new_tokens, *args, **kwargs)
        finally:
            self._recovery_context = None

    def _recovery_snapshot(self, return_cpu=True):
        # A pure-decode extension often allocates a new physical page while
        # changing no historical coverage or Program lifecycle. Native hash
        # publication updates this index; only watched-history changes need
        # a fresh GPU proof. The host already subtracts elapsed time from the
        # published forecast and TTL. Do not reuse a CPU residency response.
        mode = getattr(self, '_fair_ablation', '')
        index = getattr(self, 'gpu_index', None)
        signature = None
        if (not return_cpu and index is not None and
                mode in SEGMENTED_LAYERS):
            signature = (self.experiment, index.revision,
                tuple((p, id(saved)) for p, saved in self.baseline_requests.items()),
                tuple((r.request_id, str(r.status), r.num_prompt_tokens)
                      for r in self.requests.values() if not r.is_finished()),
                tuple((p, row[0], row[1]) for p, row in self.departures.items()),
                tuple((p, pin.deadline, tuple(b.block_id for b in pin.blocks))
                      for p, pin in self.pins.items()))
            if signature == self._recovery_publication_signature:
                return [], []
        started = time.perf_counter()
        # Finished private Programs have no next continuation. Their old
        # manifests stay available to attribute CPU objects, but scanning
        # every retired GPU hash cannot change an active recovery plan.
        layouts = self._continuation_layouts() if mode in SEGMENTED_LAYERS else self._layouts()
        state = self._program_state(layouts)
        pool = self.kv_cache_manager.block_pool
        host_layout = {}
        for row in layouts:
            covered, referenced = [], []
            for i, group in enumerate(row['chunks']):
                if None not in group:
                    covered.append(i)
                    if any(pool.blocks[bid].ref_cnt for bid in group):
                        referenced.append(i)
            host_layout[row['program']] = dict(covered=covered, referenced=referenced)
        config = dict(capacity_pages=self._capacity(), token_budget=self.max_num_scheduled_tokens,
            sequence_budget=self.max_num_running_reqs,
            decode_step=float(self.profile.get('decode_step_s', .0174)),
            fixed_step=float(self.beta[0]),
            adaptive_decode_threshold=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_THRESHOLD','0')),
            adaptive_token_budget=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_TOKENS','512')),
            query_budget=os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET','0') == '1')
        # One transaction publishes both duplicate hints and page coverage,
        # and returns a CPU snapshot. Native LOOKUP still owns actual locks.
        response = self._replica_call(dict(op='gapped_gpu_pages',
            programs={row['program']: [i for i, bid in enumerate(row['blocks']) if bid is not None]
                      for row in layouts}, pcv_programs=state, pcv_config=config,
            hicache_programs=host_layout, return_cpu=return_cpu,
            pcv_slopes=self._sensitivity_weights))
        self._recovery_publication_signature = signature
        self._emit('pcv_recovery_publication', programs=len(layouts),
                   controller_s=time.perf_counter()-started, return_cpu=return_cpu)
        programs = [Program(**state[row['program']],
            gpu=frozenset(i for i, bid in enumerate(row['blocks']) if bid is not None),
            cpu=frozenset(response.get('cpu_programs', {}).get(row['program'], {}).get('cpu_chunks', ())))
            for row in layouts if row['program'] in state]
        return layouts, programs

    def _event_weights(self, programs):
        # Progress and page changes update local recovery plans. Arrival,
        # running membership and pin transitions invalidate queue slopes.
        signature = tuple((p.name, p.phase, p.order, p.output > 0,
                           p.protected_s > 0, self.departures.get(p.name, (None,))[0])
                          for p in programs)
        if signature == self._sensitivity_signature:
            return {p.name: p.probability * self._sensitivity_weights.get(p.name, 1.)
                    for p in programs}
        started = time.perf_counter()
        table = tuple((int(k), float(v)) for k, v in
                      self.profile.get('decode_step_by_seqs', {}).items())
        model = replace(self._model(), decode_by_width=table, forecast_idle_overwrites=False)
        slopes, evaluations = continuation_sensitivities(model, programs)
        self._sensitivity_signature = signature
        self._sensitivity_weights = slopes
        self._emit('pcv_sensitivity', programs=len(programs), evaluations=evaluations,
            slopes=slopes, controller_s=time.perf_counter() - started,
            forecast='fixed snapshot topology; forward synchronous-delay derivative')
        return {p.name: p.probability * slopes.get(p.name, 1.) for p in programs}

    def _continuation_layouts(self):
        index = getattr(self, 'gpu_index', None)
        # Native add_request registers ready histories too, before serving;
        # only an actual finished lifecycle removes the index entry.
        return self._layouts(programs=set(index.rows)) if index is not None else self._layouts()

    def _recovery_native_reuse(self, count, mode):
        layer = LAYERS.get(getattr(self, '_fair_ablation', ''))
        if layer is not None and layer.compact_recovery:
            return self._compact_native_reuse(count, mode, layer)
        pool = self.kv_cache_manager.block_pool
        queue = pool.free_block_queue
        free = queue.get_all_free_blocks()
        if count > len(free) or all(b.block_hash is None for b in free[:count]):
            return self._layer_native_get_new_blocks(count)
        started = time.perf_counter()
        index=getattr(self,'gpu_index',None)
        if mode in ('gapped-next-recovery','gapped-static-recovery') and index is not None:
            # choose_gpu_bundles always consumes untracked objects in native
            # order before any tracked bundle. If they cover exact demand,
            # its answer needs neither a CPU snapshot nor a value model.
            untracked=[page for page in free if not index.by_block.get(page.block_id)]
            if len(untracked)>=count:
                chosen=untracked[:count]
                if any(page.ref_cnt or page.is_null for page in chosen):
                    raise RuntimeError('Untracked fast path selected a referenced/null page')
                for page in chosen:queue.remove(page)
                queue.prepend_n(chosen)
                self._emit('pcv_gpu_reclaim',policy=mode,pages=count,evaluations=0,
                    budget_fallback_pages=0,decisions=[],
                    changed_pages=len({p.block_id for p in chosen}-{p.block_id for p in free[:count]}),
                    controller_s=time.perf_counter()-started,
                    prediction='identical untracked-first bundle choice; no CPU snapshot',
                    fast_path='untracked-demand-covered')
                return self._layer_native_get_new_blocks(count)
        if self._recovery_context is not None:
            layouts, programs = self._recovery_context
        elif mode in ('gapped-next-recovery','gapped-static-recovery'):
            layouts, programs = self._recovery_snapshot(return_cpu=True)
        else:
            layouts = self._layouts()
            state = self._program_state(layouts)
            cpu = self._replica_call(dict(op='cpu_residency'))['programs']
            programs = [Program(**state[row['program']],
                gpu=frozenset(i for i, bid in enumerate(row['blocks']) if bid is not None),
                cpu=frozenset(cpu.get(row['program'], {}).get('cpu_chunks', ())))
                for row in layouts if row['program'] in state]
        owners = {}
        for row in layouts:
            for i, bid in enumerate(row['blocks']):
                if bid is not None:
                    owners.setdefault(bid, []).append((row['program'], i))
        by_name = {p.name: p for p in programs}
        weights = (self._event_weights(programs) if mode == 'gapped-event' else
                   {p.name: 1. for p in programs} if mode=='gapped-static-recovery' else
                   {p.name: sum(1./(1.+t) for t in p.returns)/len(p.returns)
                    for p in programs} if mode == 'gapped-next-recovery' else
                   {p.name: p.probability for p in programs})
        evaluations = 0
        if mode == 'gapped-fast':
            page_cost = replace(self.gapped_costs, chunk=16)
            def score(page):
                value = 0.
                for name, index in owners.get(page.block_id, ()):
                    if name not in by_name:
                        continue
                    p = by_name[name]
                    compute = page_cost.chunk_compute(index)
                    marginal = (min(compute, self.gapped_costs.load_token * 16 +
                                    self.gapped_costs.load_call / 16)
                                if index // 16 in p.cpu else compute)
                    value += p.probability * marginal
                return value
            chosen = [page for _, page in sorted(enumerate(free),
                      key=lambda pair: (score(pair[1]), pair[0]))[:count]]
        else:
            recovery = ChunkRecovery(self.gapped_costs, programs, weights, self._recovery_cost_cache)
            chosen, evaluations = choose_gpu_bundles(free, owners, recovery, count)
        if len(chosen) != count or len({b.block_id for b in chosen}) != count:
            raise RuntimeError('Recovery policy changed native allocation demand')
        if any(b.ref_cnt or b.is_null for b in chosen):
            raise RuntimeError('Recovery policy selected referenced/null page')
        chosen_ids = {b.block_id for b in chosen}
        withdraw = {row['program']: [i for i, bid in enumerate(row['blocks']) if bid in chosen_ids]
                    for row in layouts}
        withdraw = {p: indices for p, indices in withdraw.items() if indices}
        if withdraw:
            # The page and whole-chunk proofs are withdrawn in one RPC
            # before any allocator overwrite, even on a stale policy plan.
            self._replica_call(dict(op='gapped_gpu_withdraw', programs=withdraw,
                                   withdraw_hicache=True, pcv_slopes=self._sensitivity_weights))
            self.hicache_layout_signature = None
        for page in chosen:
            queue.remove(page)
        queue.prepend_n(chosen)
        self._emit('pcv_gpu_reclaim', policy=mode, pages=count,
            evaluations=evaluations, budget_fallback_pages=0, decisions=[],
            changed_pages=len(chosen_ids - {p.block_id for p in free[:count]}),
            controller_s=time.perf_counter() - started,
            prediction='conditional whole-chunk recovery; bounded heap; no per-victim rollout')
        return self._layer_native_get_new_blocks(count)

    def _compact_native_reuse(self, count, mode, layer):
        started = time.perf_counter()
        pool = self.kv_cache_manager.block_pool
        queue = pool.free_block_queue
        # Decode usually needs one page. Inspect the native linked head
        # before materializing the entire free queue.
        head = queue.fake_free_list_head.next_free_block
        native = []
        for _ in range(min(count, queue.num_free_blocks)):
            native.append(head)
            head = head.next_free_block
        if count > queue.num_free_blocks or all(p.block_hash is None for p in native):
            return self._layer_native_get_new_blocks(count)
        index = self.gpu_index
        free = queue.get_all_free_blocks()
        untracked = []
        for page in free:
            if not index.by_block.get(page.block_id):
                untracked.append(page)
                if len(untracked) == count:
                    break
        publication_s = 0.
        evaluations = 0
        plans = 0
        plan_hits = 0
        if len(untracked) == count:
            chosen = untracked
            fast_path = 'untracked-demand-covered'
        else:
            began = time.perf_counter()
            # The joint CPU rule needs concrete coverage and reference hints,
            # not a queue rollout or all Program timing fields. Publish the
            # maintained masks and obtain a fresh committed CPU snapshot in
            # one existing transaction. Never cache CPU residency responses.
            masks = {p: row['masks'] for p, row in index.rows.items()}
            pages = {p: [i for i, bid in enumerate(row['blocks']) if bid is not None]
                     for p, row in index.rows.items()}
            host = {}
            for p, row in index.rows.items():
                covered = [i for i, mask in enumerate(row['masks']) if mask == 65535]
                referenced = [i for i in covered if any(pool.blocks[bid].ref_cnt
                    for bid in row['blocks'][i * 16:(i + 1) * 16])]
                host[p] = dict(covered=covered, referenced=referenced)
            response = self._replica_call(dict(op='gapped_gpu_pages', programs=pages,
                                               hicache_programs=host, return_cpu=True))
            publication_s = time.perf_counter() - began
            cpu = {p: frozenset(r.get('cpu_chunks', ()))
                   for p, r in response.get('cpu_programs', {}).items()}
            weights = {p: 1. for p in masks}
            if mode == 'gapped-next-recovery':
                from kairos._runtime.layout.pcv import conditional_returns
                now = time.monotonic()
                for p, (entered, tool, _) in self.departures.items():
                    if p in masks:
                        samples = conditional_returns(self.policy_model.history, tool,
                                                      max(0., now - entered))
                        weights[p] = sum(1. / (1. + t) for t in samples) / len(samples)
            recovery = MaskRecovery(self.gapped_costs, masks, cpu, weights,
                self._recovery_cost_cache, slots=self.gapped_max_segments,
                lengths={p: len(row['blocks']) for p, row in index.rows.items()})
            executable = None
            plan_evaluations = plan_hits = 0
            if layer.executable_gpu:
                from kairos._runtime.layout.executable_recovery import ExecutableLoss
                executable = ExecutableLoss(recovery, self.connector.gapped_recovery_plans,
                    recovery.lengths)
                plan_evaluations = executable.plans.evaluations
                plan_hits = executable.plans.hits
            chosen, evaluations = choose_mask_bundles(free, index.by_block, recovery,
                count, layer.plan_refinement, executable=executable)
            plans = recovery.plan_evaluations
            if executable:
                plans = executable.plans.evaluations - plan_evaluations
                plan_hits = executable.plans.hits - plan_hits
            fast_path = ('executable-bundles' if executable else
                         'compact-plan' if layer.plan_refinement else 'compact-chunks')
        if (len(chosen) != count or len({p.block_id for p in chosen}) != count
                or any(p.ref_cnt or p.is_null for p in chosen)):
            raise RuntimeError('Compact recovery changed native demand or legality')
        withdraw = {}
        for page in chosen:
            for p, position in index.by_block.get(page.block_id, ()):
                withdraw.setdefault(p, []).append(position)
        if withdraw:
            self._replica_call(dict(op='gapped_gpu_withdraw', programs=withdraw,
                                   withdraw_hicache=True))
            self.hicache_layout_signature = None
        for page in chosen:
            queue.remove(page)
        queue.prepend_n(chosen)
        self._emit('pcv_gpu_reclaim', policy=mode, pages=count,
            evaluations=evaluations, plan_evaluations=plans, publication_s=publication_s,
            plan_cache_hits=plan_hits,
            budget_fallback_pages=0, decisions=[],
            changed_pages=len({p.block_id for p in chosen} - {p.block_id for p in native}),
            controller_s=time.perf_counter() - started,
            prediction='indexed chunk masks; at most three final plan candidates',
            fast_path=fast_path)
        return self._layer_native_get_new_blocks(count)
