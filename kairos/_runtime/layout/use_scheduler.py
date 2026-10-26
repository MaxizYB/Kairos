"""Kairos use scheduler implementation."""
import time

from kairos._runtime.layout.next_use import AUDIT_MODES, USE_MODES, SurvivalEstimator, UseInfo, InsuranceEstimator, service_windows, gpu_value, retention_weight
from kairos._runtime.layout.pcv import Program


class UseSchedulerMixin:
    def __init__(self, *args, **kwargs):
        self._reset_use()
        super().__init__(*args, **kwargs)

    def _reset_use(self):
        self.use_gpu = SurvivalEstimator()
        self.use_cpu = SurvivalEstimator()
        self.use_windows = {}
        self.use_ready = {}
        self.use_selected = set()
        self.use_consumed = set()
        self.use_waits = []
        self.use_signature = None
        self.use_gpu_insurance=InsuranceEstimator()
        self.use_cpu_insurance=InsuranceEstimator()

    def _switch_experiment(self, name):
        if name != self.experiment:
            self._reset_use()
        return super()._switch_experiment(name)

    def add_request(self, request):
        super().add_request(request)
        if getattr(self, '_fair_ablation', '') in AUDIT_MODES:
            self.use_ready[request.request_id] = time.monotonic()
            self._emit('pcv_ready', request=request.request_id, program=self.program(request),
                       returning=getattr(request, '_continuum_returning', False))

    def _program_state(self, layouts):
        started = time.perf_counter()
        state = self._return_program_state(layouts)
        if getattr(self, '_fair_ablation', '') not in AUDIT_MODES or not state:
            return state
        programs = [Program(**row, gpu=frozenset(), cpu=frozenset()) for row in state.values()]
        # Three scenarios over the active Program set, only at physical
        # allocation. Repeated ready sorting makes this a small-P heuristic,
        # not a linear-time queue forecast.
        windows = service_windows(programs, self._capacity(), self.max_num_running_reqs,
            float(self.profile.get('decode_step_s', .0174)),
            max(.001, float(self.beta[0]) + self.gapped_costs.compute([(0, 128)])))
        for p in programs:
            if p.phase == 'tool':
                _, tool, _ = self.departures[p.name]
                samples = [float(v[0]) for v in self.policy_model.history.get(tool, ())]
                if len(samples) < 8:
                    samples = [float(v[0]) for rows in self.policy_model.history.values() for v in rows]
                residuals = [v - p.age for v in samples if v > p.age]
                # Conditional wait per arrival scenario; preserves dependence
                # between A and W rather than adding their marginal means.
                waits = [max(0., u - a) for u, a in zip(windows[p.name], p.returns)]
                if residuals:
                    def delay(a):
                        j = min(range(3), key=lambda j: abs(a - p.returns[j]))
                        return waits[j]
                    q = sum(a + delay(a) <= 1. for a in residuals) / len(residuals)
                else:
                    q = p.probability * sum(u <= 1. for u in windows[p.name]) / 3
            else:
                q = sum(u <= 1. for u in windows[p.name]) / 3
            state[p.name].update(use_times=windows[p.name], use_probability=q)
        self.use_windows = windows
        signature = tuple((p.name, p.phase, p.order, tuple(round(x, 2) for x in windows[p.name]),
                           round(state[p.name]['use_probability'], 2)) for p in programs)
        if signature != self.use_signature:
            self.use_signature = signature
            self._emit('pcv_use_forecast', programs={p: dict(times=row['use_times'],
                q=row['use_probability'], return_q=row['probability'], phase=row['phase'])
                for p, row in state.items()}, controller_s=time.perf_counter() - started,
                model='three full-context service scenarios; base order, no pin/coldness priority')
        return state

    def _replica_call(self, payload):
        audit = getattr(self, '_fair_ablation', '') in AUDIT_MODES
        if audit and payload.get('op') == 'gapped_gpu_pages':
            state = payload.get('pcv_programs', {})
            objects = {(p, i): p for p, pages in payload['programs'].items() for i in pages}
            self.use_gpu.observe(objects, time.monotonic(), set(state))
            payload = dict(payload, pcv_use_gpu=self.use_gpu.summary())
        if audit and payload.get('op') == 'gapped_gpu_withdraw':
            self.use_gpu.remove([(p, i) for p, pages in payload['programs'].items() for i in pages],
                                time.monotonic())
            payload = dict(payload, pcv_use_gpu=self.use_gpu.summary())
            if getattr(self,'_use_victim_probes',None):
                payload['pcv_use_probes']=self._use_victim_probes
                self._use_victim_probes=[]
        response = super()._replica_call(payload)
        if audit and response and 'pcv_use_cpu' in response:
            self.use_cpu = SurvivalEstimator.from_summary(response['pcv_use_cpu'])
        if audit and response and 'pcv_cpu_insurance' in response:
            self.use_cpu_insurance=InsuranceEstimator.from_summary(response['pcv_cpu_insurance'])
            self.use_gpu_insurance=InsuranceEstimator.from_summary(response['pcv_gpu_insurance'])
        return response

    def _use_gpu_score(self, p, compute, load, cpu_present):
        mode = getattr(self, '_fair_ablation', '')
        q = self._use_weight(p)
        times = p.returns if mode == 'gapped-survival' else p.use_times
        info = UseInfo(times, q, p.probability, p.protected_s,
                       all_times=mode.startswith('gapped-use-discount'))
        return gpu_value(info, compute, load, cpu_present, self.use_cpu_insurance,
                         mode in ('gapped-survival', 'gapped-use-survival','gapped-use-discount-survival'))

    def _use_weight(self,p):
        return retention_weight(p,getattr(self,'_fair_ablation',''))

    def _use_lookup_observation(self,request,gpu,cpu):
        if getattr(self,'_fair_ablation','') not in AUDIT_MODES:return
        p=self.program(request)
        chunks={i for i in range((request.num_prompt_tokens-1)//256)
                if all(j in gpu for j in range(i*16,(i+1)*16))}
        self._replica_call(dict(op='pcv_use_observe',program=p,gpu_chunks=sorted(chunks),
                                cpu_chunks=sorted(cpu)))

    def _use_track_gpu_victims(self,withdraw,by_name):
        if not withdraw:return
        rows=[]
        for p,pages in withdraw.items():
            if p not in by_name:continue
            r=by_name[p]
            times=r.returns if self._fair_ablation=='gapped-survival' else r.use_times
            delay=sum(times)/len(times)
            rows += [dict(program=p,chunk=i,delay=delay) for i in {j//16 for j in pages} if i in r.cpu]
        self._use_victim_probes=rows

    def schedule(self, throttle_prefills=False):
        output = super().schedule(throttle_prefills=throttle_prefills)
        if getattr(self, '_fair_ablation', '') in AUDIT_MODES:
            for rid in output.num_scheduled_tokens:
                if rid not in self.requests or rid in self.use_selected:
                    continue
                self.use_selected.add(rid)
                now = time.monotonic()
                wait = now - self.use_ready.get(rid, now)
                self.use_waits = (self.use_waits + [wait])[-128:]
                self._emit('pcv_first_service', request=rid, program=self.program(self.requests[rid]),
                    ready_wait_s=wait, future_window=self.use_windows.get(self.program(self.requests[rid])))
        return output

    def update_from_output(self, output, runner_output):
        if getattr(self, '_fair_ablation', '') in AUDIT_MODES:
            for rid in getattr(output, '_gapped_logical_counts', output.num_scheduled_tokens):
                if rid not in self.requests or rid in self.use_consumed:
                    continue
                self.use_consumed.add(rid)
                p = self.program(self.requests[rid])
                self._emit('pcv_first_execution_complete', request=rid, program=p,
                    ready_elapsed_s=time.monotonic() - self.use_ready.get(rid, time.monotonic()),
                    gpu_survival=self.use_gpu.summary(), cpu_survival=self.use_cpu.summary())
        return super().update_from_output(output, runner_output)
