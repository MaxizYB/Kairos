"""Kairos worker implementation."""
import faulthandler
import json
import os
from pathlib import Path
import time

from vllm.v1.worker.gpu_worker import Worker

CURRENT_REQUESTS = ()


class DiagnosticWorker(Worker):
    def __init__(self, *args, **kwargs):
        path = Path(os.environ['CONTINUUM_GAPPED_WORKER_LOG']).with_name(
            f'worker-stacks-{os.getpid()}.log')
        self._gapped_stack_file = path.open('a')
        faulthandler.enable(file=self._gapped_stack_file, all_threads=True)
        super().__init__(*args, **kwargs)

    def execute_model(self, scheduler_output):
        global CURRENT_REQUESTS
        if os.environ.get('CONTINUITY_DENSE_DECODE_CAPTURE')=='1':
            if not hasattr(scheduler_output,'_continuum_exact_decode_graph'):
                raise RuntimeError('Missing registered per-arm graph policy in worker IPC')
            exact=bool(getattr(scheduler_output,'_continuum_exact_decode_graph',False))
            experiment=getattr(scheduler_output,'_continuum_graph_experiment',None)
            if exact!=getattr(self,'_continuum_graph_exact',None):
                from kairos._runtime.compat.decode_graph_policy import set_decode_graph_policy
                dispatcher=self.model_runner.cudagraph_dispatcher
                policy=set_decode_graph_policy(dispatcher,exact)
                self._continuum_graph_exact=exact
                self._continuum_graph_policy=policy
            if experiment!=getattr(self,'_continuum_graph_experiment',None):
                self._continuum_graph_experiment=experiment
                row=dict(event='decode_graph_shape_policy',time=time.time(),
                    experiment=experiment,**self._continuum_graph_policy)
                path=Path(os.environ['CONTINUUM_GAPPED_WORKER_LOG'])
                with path.open('a') as log:log.write(json.dumps(row)+'\n')
        CURRENT_REQUESTS = tuple(r for r in scheduler_output.num_scheduled_tokens if '::gquery:' not in r)
        probe = None
        # This opt-in run is diagnostic only. It deliberately synchronizes
        # for one step per width and must never supply headline JCT timings.
        if os.environ.get('CONTINUUM_KERNEL_DIAGNOSTIC') == '1':
            self._kernel_calls = getattr(self, '_kernel_calls', 0) + 1
            q = scheduler_output.num_scheduled_tokens
            begins = {r.req_id: r.num_computed_tokens
                      for r in scheduler_output.scheduled_new_reqs}
            cached = scheduler_output.scheduled_cached_reqs
            begins.update(zip(cached.req_ids, cached.num_computed_tokens))
            width = len(q)
            seen = getattr(self, '_kernel_widths', set())
            active_shape = (getattr(scheduler_output, '_kernel_experiment', '')
                            .startswith('continuum-index-partial-'))
            if (active_shape and self._kernel_calls > 500 and 1 <= width <= 8 and width not in seen
                    and all(n == 1 for n in q.values())):
                import torch
                torch.cuda.synchronize()
                probe = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA])
                probe.__enter__()
                self._kernel_widths = seen | {width}
                kernel_shape = dict(width=width, contexts=[begins.get(r, 0)+1 for r in q],
                    call=self._kernel_calls, experiment=scheduler_output._kernel_experiment,
                    included_in_performance_comparison=False)
        faulthandler.dump_traceback_later(45, repeat=True,
            file=self._gapped_stack_file)
        try:
            return super().execute_model(scheduler_output)
        finally:
            if probe is not None:
                import torch
                torch.cuda.synchronize()
                probe.__exit__(None, None, None)
                path = Path(os.environ['CONTINUUM_GAPPED_WORKER_LOG']).with_name('kernel-diagnostic.jsonl')
                operators = [dict(name=e.key, calls=e.count,
                    self_device_us=e.self_device_time_total, self_cpu_us=e.self_cpu_time_total,
                    device_type=str(e.device_type)) for e in probe.key_averages()]
                with path.open('a') as log:
                    log.write(json.dumps(dict(**kernel_shape, operators=operators))+'\n')
                probe.export_chrome_trace(str(path.with_name(f'kernel-width-{width}.trace.json')))
            CURRENT_REQUESTS = ()
            faulthandler.cancel_dump_traceback_later()
