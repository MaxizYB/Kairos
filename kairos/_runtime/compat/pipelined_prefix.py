"""Kairos pipelined prefix implementation."""
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from kairos._runtime.compat.continuity_bridge import ContinuityBridgeScheduler
from kairos._runtime.compat.component_ablation import native_lru_payload


class PipelinedPrefixScheduler(ContinuityBridgeScheduler, AsyncScheduler):
    native_async_prefix_execution = True
    allow_segmented_pipeline = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.scheduler_config.async_scheduling or not self.defer_block_free:
            raise ValueError('Pipelined prefix requires native async deferred-free fences')
        if self.vllm_config.speculative_config is not None:
            raise ValueError('No speculative decoding in this execution experiment')
        if not hasattr(self, '_spec_token_placeholders'):
            raise RuntimeError('Native AsyncScheduler initialization is absent')
        self._pipeline_inflight = {}
        self.pipeline_capacity_fence = os.environ.get('CONTINUUM_ASYNC_CAPACITY_FENCE') == '1'
        self._pipeline_wait_for_fence = False

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        tags = set(name.split('~')[1:])
        if 'tierlru' in tags:
            self.connector.range_lookup = True
            if 'r5callcommitment' not in tags:
                self._emit('tier_ablation_registry',
                    gpu='native LRU among legal free pages',
                    cpu='native LRU, same per-STORE deficit and partial reservation',
                    cpu_range_lookup=True,
                    changed='remove v21 complementary cross-tier layout choices',
                    native_reference_and_dma_fences=True)
        if (self.gapped_enabled and not self.allow_segmented_pipeline) or any(self.bridge_flags):
            raise ValueError('Pilot supports original Continuum/v21 prefix controls only')
        self._emit('pipeline_execution_policy', native_async=True,
            max_concurrent_batches=self.vllm_config.max_concurrent_batches,
            algorithm='original Program order/TTL; registered execution dependencies', native_placeholders=True,
            deferred_frees=True, speculative_decoding=False,
            trace_future_information=False, prefix_only=not self.allow_segmented_pipeline,
            segmented_boundary_drain=self.allow_segmented_pipeline,
            segmented_dependency_scope='submitted exact segment; queued native work may pipeline',
            capacity_retry_after_prior_result=self.pipeline_capacity_fence)

    def _replica_call(self, payload):
        tags = set((self.experiment or '').split('~')[1:])
        if 'tierlru' in tags and 'r5callcommitment' not in tags:
            payload = native_lru_payload(payload)
        return super()._replica_call(payload)

    def schedule(self, throttle_prefills=False):
        # A failed running allocation while a prior batch still owns pages
        # must first settle that batch. Native scheduling performs a normal
        # empty/partial submission; its metadata and worker-slot state remain
        # intact. Persistent capacity failure after settling may preempt in
        # the ordinary native path. Never weaken its deferred-free fence.
        self._pipeline_wait_for_fence = bool(self.pipeline_capacity_fence and self._pipeline_inflight)
        output = super().schedule(throttle_prefills=throttle_prefills)
        if output.total_num_scheduled_tokens:
            self._pipeline_inflight[output._continuum_step_serial] = output._continuum_start
            self._emit('pipeline_submission', step_id=output._continuum_step_serial,
                pending_batches=len(self._pipeline_inflight),
                deferred_pages=sum(len(blocks) for _, blocks in self.deferred_frees))
        return output

    def update_from_output(self, output, runner_output):
        result = super().update_from_output(output, runner_output)
        self._pipeline_inflight.pop(getattr(output, '_continuum_step_serial', None), None)
        return result

    def _allocate(self, request, num_new_tokens, *args, **kwargs):
        result=super()._allocate(request,num_new_tokens,*args,**kwargs)
        if result is None and getattr(self,'_pipeline_wait_for_fence',False):
            self._emit('pipeline_capacity_wait',request=request.request_id,
                pending_batches=len(self._pipeline_inflight),
                deferred_pages=sum(len(blocks) for _,blocks in self.deferred_frees),
                free_pages=self.kv_cache_manager.block_pool.get_num_free_blocks())
        return result
