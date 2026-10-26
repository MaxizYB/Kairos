"""Kairos pcvu control implementation."""
import re
import time
import math

from kairos._runtime.layout.pcv import conditional_returns


def controls(name):
    match = re.match(r'pcvu-([01]{3})(-oracle)?(?:-(?:nofinal|finalonly))?-partial', name)
    return tuple(c == '1' for c in match[1]) + (bool(match[2]),) if match else (False,) * 4


def completion_headroom(requests, owned, max_len, page=16):
    """Configured output caps bound capacity; they do not predict EOS.

    Requests are (identity, prompt_tokens, output_cap). Already-owned native
    blocks are subtracted, so incremental decode growth is not charged twice.
    No cache hit is assumed for a new follower.
    """
    return sum(max(0, math.ceil(min(max_len, prompt+max(0,cap-1))/page)
                   -owned.get(rid,0)) for rid,prompt,cap in requests)


def continuation_cpu_loss(costs, index, gpu, protected, cpu, times, continues=True):
    """Conditional loss of the whole CPU chunk at its next consumption.

    Native refs protect *current* execution, not the next independent
    recovery. Only an unexpired policy lease supplies future coverage.
    """
    if not continues:
        return 0.
    pages = set(range(index * 16, (index + 1) * 16))
    missing = pages - set(gpu)
    from kairos._runtime.layout.plan import runs
    compute = costs.compute([(a * 16, b * 16) for a, b in runs(missing)])
    load = costs.load_token * 256 + costs.load_call
    immediate = max(0., compute - min(compute, load))
    insurance = max(0., costs.chunk_compute(index) - load)
    survives = sum(t <= protected for t in times) / max(1, len(times))
    # A partially absent GPU chunk already needs the CPU at return. For a
    # duplicate, future insurance is evaluated at the same return scenarios.
    return immediate if missing else (1. - survives) * insurance


class PCVUControlMixin:
    def __init__(self, *args, **kwargs):
        self.pcvu_flags = (False,) * 4
        self.pcvu_lengths = []
        self.pcvu_leased = set()
        self.pcvu_preferred = None
        self.pcvu_cpu_deadlines = {}
        self.pcvu_blocked = None
        self.pcvu_bypass_count = {}
        super().__init__(*args, **kwargs)

    def _switch_experiment(self, name):
        changed = name != self.experiment
        if changed:
            for p in self.pcvu_leased:
                self._replica_call(dict(op='cpu_service_release', program=p, reason='experiment-reset'))
            self.pcvu_flags = controls(name)
            self.pcvu_leased = set()
            self.pcvu_lengths = []
            self.pcvu_preferred = None
            self.pcvu_truth = {}
            self.pcvu_cpu_deadlines = {}
            self.pcvu_blocked = None
            self.pcvu_bypass_count = {}
        super()._switch_experiment(name)
        if changed and any(self.pcvu_flags):
            self._emit('pcvu_action_registry', m1=self.pcvu_flags[0], m2=self.pcvu_flags[1],
                m3=self.pcvu_flags[2], oracle=self.pcvu_flags[3],
                scheduling='bounded-feasible-program-bypass', lease='CPU-next-service',
                placement='conditional-whole-chunk')

    def _replica_call(self, payload):
        if payload.get('op') == 'experiment':
            payload = dict(payload, pcvu_m2=self.pcvu_flags[1], pcvu_m3=self.pcvu_flags[2],
                pcvu_finalonly='-finalonly-' in (self.experiment or ''))
        if payload.get('op') == 'hicache_register' and self.pcvu_flags[2]:
            payload = dict(payload, **self._replica_history_extension(payload['program'], 'running'))
        return super()._replica_call(payload)

    def _has_continuity_priority(self, program):
        # CPU-backed continuation keeps service priority while the original
        # finite window remains valid. It makes no GPU-residency claim.
        return super()._has_continuity_priority(program) or (
            self.pcvu_flags[1] and self.pcvu_cpu_deadlines.get(program, 0.) > time.monotonic())

    def _refresh_priority(self):
        super()._refresh_priority()
        if not self.pcvu_flags[0] or not self.running:
            return
        self._pcvu_fit_bypass()

    def _run_native_schedule(self, throttle_prefills=False):
        result = super()._run_native_schedule(throttle_prefills=throttle_prefills)
        if self.pcvu_flags[1]:
            for rid in result.num_scheduled_tokens:
                r = self.requests.get(rid)
                p = self.program(r) if r is not None else None
                if p in self.pcvu_leased:
                    # LOOKUP owns native read locks by this point. End the
                    # policy lease at actual service, not HTTP arrival.
                    self._replica_call(dict(op='cpu_service_release',program=p,reason='first-service'))
                    self.pcvu_leased.discard(p)
                    self.pcvu_cpu_deadlines.pop(p,None)
        if self.pcvu_preferred in result.num_scheduled_tokens:
            self.pcvu_preferred = None
        return result

    def _pcvu_fit_bypass(self):
        """Bounded work conservation without interrupting active Programs.

        Only a head proven unable to allocate at the native boundary can be
        bypassed. Choose the earliest ordinary ready follower whose *full*
        prompt fits already free pages. Two bypass admissions per blocked
        head bound unfairness. Cache hits are not assumed for feasibility.
        """
        from vllm.v1.request import RequestStatus
        blocked = self.pcvu_blocked
        if not blocked or self.pcvu_bypass_count.get(blocked,0) >= 2:
            return
        from vllm.v1.core.sched.request_queue import create_request_queue
        rows=list(self.waiting)
        if not rows or rows[0].request_id != blocked:
            return
        free=self.kv_cache_manager.block_pool.get_num_free_blocks()
        watermark=self.kv_cache_manager.watermark_blocks
        owned={r.request_id:sum(not b.is_null for b in
            self.kv_cache_manager.get_blocks(r.request_id).blocks[0]) for r in self.running}
        active=[(r.request_id,r.num_prompt_tokens,r.max_tokens) for r in self.running]
        growth=completion_headroom(active,owned,self.max_model_len)
        candidate=next((r for r in rows[1:] if r.status==RequestStatus.WAITING
            and not r.num_computed_tokens
            and r.priority//1000000==rows[0].priority//1000000
            and completion_headroom([(r.request_id,r.num_prompt_tokens,r.max_tokens)],
                {},self.max_model_len)+growth+watermark <= free),None)
        if candidate is None:
            return
        candidate.priority=rows[0].priority-1
        q=create_request_queue(self.policy)
        for r in rows:q.add_request(r)
        self.waiting=q
        self.pcvu_preferred=candidate.request_id
        self._emit('pcvu_m1_fit_bypass',blocked=blocked,candidate=candidate.request_id,
            free_pages=free,required_upper_pages=completion_headroom(
                [(candidate.request_id,candidate.num_prompt_tokens,candidate.max_tokens)],
                {},self.max_model_len)+growth+watermark,
            running_growth_pages=growth,capacity_bound='configured-output-caps',
            continuity_class_preserved=True,
            bypass_count=self.pcvu_bypass_count.get(blocked,0),physical_preemption=False)

    def _hicache_allocate(self, request, num_new_tokens, *args, **kw):
        result = super()._hicache_allocate(request, num_new_tokens, *args, **kw)
        if self.pcvu_flags[0] and request not in self.running:
            if result is None:
                self.pcvu_blocked=request.request_id
            elif request.request_id==self.pcvu_preferred and self.pcvu_blocked:
                self.pcvu_bypass_count[self.pcvu_blocked]=self.pcvu_bypass_count.get(self.pcvu_blocked,0)+1
                self._emit('pcvu_m1_bypass_admitted',request=request.request_id,blocked=self.pcvu_blocked)
            elif request.request_id==self.pcvu_blocked:
                self.pcvu_blocked=None
        if result is not None or not self.pcvu_flags[1] or not self.pins:
            return result
        pressure = self._required_capacity(request, num_new_tokens, kw)
        if pressure['D'] <= 0:
            return result
        # An admission-sized release can succeed now and preempt the very
        # same cohort a few decode steps later. Fund its complete configured
        # capacity envelope before changing protected residency.
        demand=max(pressure['D'],self._completion_release_budget(request,pressure))
        computed = kw.get('new_computed_blocks')
        excluded = {b.block_id for group in computed.blocks for b in group} if computed else set()
        rows = self._safe_suffix_rows(excluded)
        queued={self.program(r) for q in (self.waiting,self.skipped_waiting) for r in q}
        # Evaluate only releases sufficient for this actual allocation. A
        # live CPU commitment must cover every released complete chunk.
        for row in rows:
            p = row['id']
            pin = self.pins[p]
            if p == self.program(request) or p in queued or row['limit'] < demand:
                continue
            end = min(len(pin.blocks)//16,pin.tokens//256)
            begin = min((len(pin.blocks)-demand)//16,end-1)
            count = len(pin.blocks)-begin*16
            if not 0 <= begin < end or count > row['limit']:
                continue
            step = float(self.profile.get('decode_step_s',.0174))
            remaining = [max(1, r.max_tokens-r.num_output_tokens) if self.pcvu_flags[3]
                         else max(32, int(sum(self.pcvu_lengths)/len(self.pcvu_lengths))-
                             r.num_output_tokens) if self.pcvu_lengths else 146
                         for r in self.running]
            service_delay = min(remaining,default=0)*step
            duration = min(5., max(.1, pin.deadline-time.monotonic(),service_delay+.1))
            answer = self._replica_call(dict(op='cpu_service_acquire',program=p,
                start_chunk=begin,chunks=end,duration_s=duration))
            if not answer.get('accepted'):
                continue
            self.pcvu_leased.add(p)
            self.pcvu_cpu_deadlines[p] = pin.deadline
            released = self._release(p, 'pcvu_cpu_handover', count)
            result = super()._hicache_allocate(request, num_new_tokens, *args, **kw)
            self._emit('pcvu_m2_handover', program=p, incoming=self.program(request),
                released_pages=released, deficit_pages=pressure['D'], completion_demand_pages=demand,
                cpu_start=begin,cpu_end=end,
                admitted=result is not None, proactive_d2h_bytes=0,
                uninsured_tail_pages=count-(end-begin)*16)
            return result
        return result

    def _pcvu_extend_state(self, state):
        if not self.pcvu_flags[2]:
            return state
        active = {self.program(r): r for r in self.requests.values() if not r.is_finished()}
        for p, row in state.items():
            r = active.get(p)
            # Replay exposes final-round status before generation finishes.
            # nofinal removes that advance knowledge from placement policy.
            continues = ('-nofinal-' in (self.experiment or '') or row['phase'] == 'tool'
                         or not self.metadata(r).get('cont_final', False))
            if row['phase'] == 'tool':
                times = row['returns']
                if self.pcvu_flags[3]:
                    began, _, _ = self.departures[p]
                    gap = getattr(self, 'pcvu_truth', {}).get(p)
                    if gap is not None:
                        times = (max(.001, began + gap-time.monotonic()),) * 3
            else:
                tool = str(self.metadata(r).get('cont_tool', ''))
                duration = row['output'] * float(self.profile.get('decode_step_s', .0174))
                times = tuple(duration + t for t in conditional_returns(self.policy_model.history, tool, 0.))
            row.update(cpu_next_times=times, cpu_continues=continues)
        return state

    def _replica_history_extension(self, program, state, tool=None):
        if not self.pcvu_flags[2]:
            return super()._replica_history_extension(program, state, tool)
        if state == 'finished':
            return dict(pcv_remove_program=str(program))
        row = self._return_program_state(self._layouts()).get(program)
        return dict(pcv_lifecycle=dict(program=program, row=row)) if row else {}

    def add_request(self, request):
        super().add_request(request)
        p = self.program(request)
        if not hasattr(self, 'pcvu_truth'):
            self.pcvu_truth = {}
        if self.pcvu_flags[3]:
            self.pcvu_truth[p] = float(self.metadata(request)['cont_oracle_gap'])

    def _free_request_blocks(self, request):
        success = getattr(request.status, 'name', '') in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED')
        if success:
            self.pcvu_lengths = (self.pcvu_lengths + [request.num_output_tokens])[-128:]
        p = self.program(request)
        if self.metadata(request).get('cont_final') and p in self.pcvu_leased:
            self._replica_call(dict(op='cpu_service_release',program=p,reason='program-finished'))
            self.pcvu_leased.discard(p)
        super()._free_request_blocks(request)
