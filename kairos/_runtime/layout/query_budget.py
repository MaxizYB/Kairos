"""Kairos query budget implementation."""
import ast
import inspect
import math
import os
import textwrap
import time

from kairos._runtime.layout.plan import query_segments
from kairos._runtime.layout.segment_capacity import query_slots


def adapt_schedule(native):
    source = getattr(native, '_copy_wait_source', None)
    if source is None:
        if hasattr(native, '_copy_wait_provenance'):
            raise RuntimeError('Copy-wait adapter must expose its transformed function source')
        source = textwrap.dedent(inspect.getsource(native))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('Expected native schedule function source, not a class/module')
    changes = dict(running=0, waiting=0, debit=0, refund=0, constraint=0, slots=0)

    def expression(source):
        return ast.parse(source, mode='eval').body

    class Adapter(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'num_new_tokens'):
                value = ast.unparse(node.value)
                if value == 'min(num_new_tokens, token_budget, input_budget - draft_slots)':
                    changes['running'] += 1
                    node.value = expression('self._gapped_budget_tokens(request, '
                        'request.num_computed_tokens, num_new_tokens, '
                        'min(token_budget, input_budget - draft_slots))')
                elif value == 'min(num_new_tokens, request_token_budget)':
                    changes['waiting'] += 1
                    node.value = expression('self._gapped_budget_tokens(request, '
                        'num_computed_tokens, num_new_tokens, request_token_budget)')
            return node

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            target, value = ast.unparse(node.target), ast.unparse(node.value)
            if target in ('token_budget', 'input_budget'):
                if isinstance(node.op, ast.Sub) and value in (
                        'num_new_tokens', 'num_new_tokens + draft_slots'):
                    changes['debit'] += 1
                    node.value = expression('self._gapped_query_charge(request, num_new_tokens)'
                        + (' + draft_slots' if target == 'input_budget' else ''))
                elif isinstance(node.op, ast.Add) and value in (
                        'restored', 'restored + draft_slots'):
                    changes['refund'] += 1
                    node.value = expression('self._gapped_step_charges.get(preempted_req_id, restored)'
                        + (' + draft_slots' if target == 'input_budget' else ''))
            return node

        def visit_Assert(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == 'total_num_scheduled_tokens <= self.max_num_scheduled_tokens':
                changes['constraint'] += 1
                node.test = expression('sum(self._gapped_step_charges.get(rid, count) '
                    'for rid, count in num_scheduled_tokens.items()) <= self.max_num_scheduled_tokens')
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            if ast.unparse(node) == 'num_running >= self.max_num_running_reqs':
                changes['slots'] += 1
                return expression('self._gapped_admission_slots_full(num_running)')
            return node

    tree = Adapter().visit(tree)
    expected = dict(running=1, waiting=1, debit=4, refund=2, constraint=1, slots=1)
    if changes != expected:
        raise RuntimeError(f'Unsupported native query-budget schedule: {changes}, expected {expected}')
    tree.body[0].decorator_list = []
    tree.body.insert(0, ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0))
    namespace = dict(native.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<private-gapped-query-budget>', 'exec'), namespace)
    adapted = namespace[native.__name__]
    required = {'_gapped_budget_tokens', '_gapped_query_charge', '_gapped_admission_slots_full'}
    if not required <= set(adapted.__code__.co_names):
        raise RuntimeError('Query-budget hooks are missing from the compiled schedule')
    if hasattr(native, '_copy_wait_provenance') and '_hicache_copy_wait' not in adapted.__code__.co_consts:
        raise RuntimeError('Query-budget composition lost the native copy-wait guard')
    return adapted


class QueryBudgetMixin:
    def _run_native_schedule(self, throttle_prefills=False):
        if not (self.gapped_enabled
                and os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET', '0') == '1'):
            return super()._run_native_schedule(throttle_prefills=throttle_prefills)
        self._gapped_step_plans = {}
        self._gapped_step_charges = {}
        self._gapped_slot_reservations = {}
        self._gapped_budget_calls = 0
        self._gapped_budget_available = 0
        self._gapped_prefill_plan_calls = 0
        self._gapped_prefill_plan_s = 0.
        self._gapped_impossible_plan_skips = 0
        if getattr(self, '_query_budget_schedule', None) is None:
            from vllm.v1.core.sched.scheduler import Scheduler
            self._query_budget_schedule = adapt_schedule(Scheduler.schedule)
        # Optional, bounded diagnostic sampling. Disabled in confirmation
        # runs; measurements collected under it are not headline timings.
        samples=int(os.environ.get('CONTINUUM_PROFILE_DECODE_STEPS','0'))
        if not samples:
            return self._query_budget_schedule(self, throttle_prefills=throttle_prefills)
        if getattr(self,'_query_profile_experiment',None)!=self.experiment:
            self._query_profile_experiment=self.experiment
            self._query_profile_calls=0
            self._query_profile=None
        self._query_profile_calls+=1
        first=int(os.environ.get('CONTINUUM_PROFILE_DECODE_BEGIN','3000'))
        active=first<=self._query_profile_calls<first+samples
        if active:
            if self._query_profile is None:
                import cProfile
                self._query_profile=cProfile.Profile()
            self._query_profile.enable()
        try:
            return self._query_budget_schedule(self, throttle_prefills=throttle_prefills)
        finally:
            if active:
                self._query_profile.disable()
                if self._query_profile_calls==first+samples-1:
                    import pstats
                    stats=pstats.Stats(self._query_profile).stats
                    top=sorted(stats.items(),key=lambda row:-row[1][3])[:25]
                    self._emit('gapped_schedule_profile',samples=samples,top=[
                        dict(function=f'{key[0]}:{key[1]}:{key[2]}',calls=value[1],
                             self_s=value[2],cumulative_s=value[3]) for key,value in top])

    def _gapped_reserved_slots(self):
        return sum(self._gapped_slot_reservations.values())

    def _gapped_admission_slots_full(self, num_running):
        runner = (self.gapped_runner_capacity if getattr(self, 'gapped_segment_capacity', False)
                  else self.max_num_running_reqs)
        if (num_running >= self.max_num_running_reqs
                and getattr(self, 'execution_request_caps', ())):
            from kairos._runtime.layout.execution_capacity import audit_waiting_capacity
            audit_waiting_capacity(self, num_running)
        return (num_running >= self.max_num_running_reqs
                or num_running + self._gapped_reserved_slots() >= runner)

    def _gapped_budget_tokens(self, request, begin, available, budget):
        self._gapped_budget_calls += 1
        self._gapped_budget_available = max(self._gapped_budget_available, available)
        native = min(available, budget)
        if available <= 0 or begin >= request.num_prompt_tokens:
            return native
        if (getattr(self.connector, 'uses_gapped', None) is not None
                and not self.connector.uses_gapped(request.request_id)):
            return native
        started = time.perf_counter()
        self._gapped_prefill_plan_calls = getattr(self, '_gapped_prefill_plan_calls', 0) + 1
        try:
            return self._gapped_prefill_budget_tokens(request, begin, available, budget)
        finally:
            self._gapped_prefill_plan_s = getattr(self, '_gapped_prefill_plan_s', 0.) + time.perf_counter() - started

    def _gapped_prefill_budget_tokens(self, request, begin, available, budget):
        native = min(available, budget)
        pool = self.kv_cache_manager.block_pool
        owned = self.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks.get(
            request.request_id, ())
        if (os.environ.get('CONTINUUM_GAPPED_SKIP_IMPOSSIBLE_PLAN', '0') == '1'
                and getattr(self, 'scheduler_reserve_full_isl', False)
                and not getattr(self, 'chunk_engine_compat', False)
                and not getattr(self, 'pins', {})
                and getattr(getattr(request, 'status', None), 'name', None) == 'WAITING'
                and request.num_computed_tokens == 0):
            # Optimistic lower bound on the native FULL-SEQUENCE gate. Even
            # granting every matched-prefix and already-owned page for free
            # cannot fit the request. With no pin to revoke, the allocator
            # will refuse regardless of this step's query plan. Still call
            # the native scheduler/allocator; no admission or order changes.
            full = min(request.num_tokens, self.max_model_len)
            lower = max(0, math.ceil(full / 16) - begin // 16 - len(owned))
            if lower > pool.get_num_free_blocks():
                self._gapped_impossible_plan_skips = getattr(self, '_gapped_impossible_plan_skips', 0) + 1
                return native
        owned_ids = {b.block_id for b in owned}
        loads = self.connector.load_chunks(request.request_id)
        gpu = set(self.gapped_gpu_pages.get(request.request_id, ()))
        allow_load_hits = os.environ.get('CONTINUUM_GAPPED_ADOPT_LOAD_HITS', '0') == '1'
        for i in range(begin // 16, (request.num_prompt_tokens - 1) // 16):
            if i // 16 in loads and not allow_load_hits:
                continue
            found = pool.get_cached_block(request.block_hashes[i], [0])
            if found and (found[0].ref_cnt == 0 or found[0].block_id in owned_ids):
                gpu.add(i)
        loaded = self.connector.valid_cpu_pages(request.request_id)
        running = getattr(self, 'running', ())
        runner = (self.gapped_runner_capacity if getattr(self, 'gapped_segment_capacity', False)
                  else self.max_num_running_reqs)
        # One extra query segment is enough for an early gap plus a logits
        # anchor. Reserve its physical runner slot before admitting followers.
        slots = query_slots(self.max_num_running_reqs, runner, len(running),
            self._gapped_reserved_slots(), request not in running,
            getattr(self, 'gapped_max_segments', 2))
        self._gapped_step_plans[request.request_id] = (begin, gpu | loaded, loads, slots)

        def charge(count):
            end = begin + count
            valid = gpu | self.connector.budget_cpu_pages(request.request_id, end)
            return sum(b - a for a, b in query_segments(begin, end, valid, slots=slots))

        if charge(available) <= budget:
            return available
        low, high = native, available
        while low < high:
            middle = (low + high + 1) // 2
            if charge(middle) <= budget:
                low = middle
            else:
                high = middle - 1
        return low

    def _gapped_query_charge(self, request, count):
        plan = self._gapped_step_plans.get(request.request_id)
        if plan is None or not count:
            value = count
        else:
            begin, gpu, loads, slots = plan
            end = begin + count
            valid = gpu | self.connector.budget_cpu_pages(request.request_id, end)
            segments = query_segments(begin, end, valid, slots=slots)
            value = sum(b - a for a, b in segments)
            self._gapped_slot_reservations[request.request_id] = len(segments) - 1
        self._gapped_step_charges[request.request_id] = value
        return value
