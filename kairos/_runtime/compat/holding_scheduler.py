"""Kairos holding scheduler implementation."""
import time

from kairos._runtime.compat.pipelined_prefix import PipelinedPrefixScheduler
from kairos._runtime.compat.holding_economics import PriorityCertificates, ToolHoldPolicy


class HoldingEconomicsScheduler(PipelinedPrefixScheduler):
    def __init__(self, *args, **kwargs):
        self.holding_variant = 'paper'
        self.logical_policy_completion = False
        self.holding_priorities = PriorityCertificates()
        super().__init__(*args, **kwargs)

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        tags = set(name.split('~')[1:])
        variants = tags & {'holdlinked', 'holdsplit'}
        if len(variants) > 1:
            raise ValueError('Conflicting holding-economics experiment tags')
        self.holding_variant = next(iter(variants), 'paper')
        self.logical_policy_completion = 'finishlogical' in tags
        self.holding_priorities = PriorityCertificates()
        if variants:
            if name.split('~', 1)[0] != 'v21-eager-a':
                raise ValueError('Holding experiment requires the common v21 placement/execution')
        original = self.policy_model
        self.policy_model = ToolHoldPolicy(mode=original.mode, fixed=original.fixed,
            k=original.k, history=original.history, replacement=bool(variants))
        self._emit('holding_economics_registry', variant=self.holding_variant,
            physical_objective=('empirical expected tool hold' if variants else 'original paper'),
            priority=('original paper deadline, with queued-return extension'
                      if self.holding_variant == 'holdsplit' else 'coupled to physical pin'),
            tool_predictions='same causal history; unchanged cold-start fallback',
            future_information=False, admission='unchanged', preemption='unchanged',
            placement='unchanged v21 joint-prefix',
            lifecycle=('logical request finish; native DMA/free fences unchanged'
                       if self.logical_policy_completion else 'legacy deferred block cleanup'),
            model_limit='Expected hold excludes return-queue time and forced revocation; measured separately')

    def _continuum_complete_policy(self, request):
        policy = self.policy_model
        before = getattr(policy, 'serial', 0)
        started = time.perf_counter()
        super()._continuum_complete_policy(request)
        program = self.program(request)
        if getattr(policy, 'serial', 0) == before:
            return
        decision = policy.last_decision
        pin = self.pins.get(program)
        if self.holding_variant == 'holdsplit' and pin is not None:
            self.holding_priorities.grant(program, pin.started, decision['paper_ttl'])
        self._emit('holding_economics_decision', program=program,
            request=request.request_id, variant=self.holding_variant, **decision,
            actual_physical_ttl=(pin.deadline-pin.started if pin is not None else 0.),
            actual_protection=pin is not None,
            original_priority_deadline=(pin.started+decision['paper_ttl'] if pin is not None else None),
            scope='Same-state alternative TTL, not a counterfactual JCT estimate',
            departure_handler_s=time.perf_counter()-started)

    def _free_request(self, request, *args, **kwargs):
        if request.is_finished():
            request._holding_logical_finish_at = time.monotonic()
            self._emit('holding_logical_finish', request=request.request_id,
                program=self.program(request), lifecycle_repaired=self.logical_policy_completion)
        if (self.logical_policy_completion and request.is_finished()
                and not getattr(request, '_continuum_logical_finish_recorded', False)):
            self._continuum_complete_policy(request)
            request._continuum_logical_finish_recorded = True
        return super()._free_request(request, *args, **kwargs)

    def _free_request_blocks(self, request):
        finished_at = getattr(request, '_holding_logical_finish_at', None)
        if finished_at is not None:
            self._emit('holding_physical_cleanup', request=request.request_id,
                program=self.program(request), dma_cleanup_delay_s=max(0.,time.monotonic()-finished_at),
                policy_already_recorded=getattr(request, '_continuum_logical_finish_recorded', False))
        return super()._free_request_blocks(request)

    def _has_continuity_priority(self, program):
        if self.holding_variant == 'holdsplit':
            return program in self.pins and self.holding_priorities.has(program)
        return super()._has_continuity_priority(program)

    def _expire_idle_protection(self, now, queued):
        super()._expire_idle_protection(now, queued)
        if self.holding_variant == 'holdsplit':
            for program in self.holding_priorities.expire(now, queued):
                self._emit('holding_priority_expired', program=program,
                    physical_protection_retained=program in self.pins)

    def _release(self, program, *args, **kwargs):
        pin = self.pins.get(program)
        if pin is not None:
            self._emit('holding_release_state', program=program,
                age_s=max(0., time.monotonic()-pin.started),
                remaining_ttl_s=max(0., pin.deadline-time.monotonic()),
                priority_eligible=self._has_continuity_priority(program),
                reason=args[0] if args else kwargs.get('reason'),
                references_before=len(pin.blocks))
        result = super()._release(program, *args, **kwargs)
        if program not in self.pins:
            self.holding_priorities.release(program)
        return result

    def _allocate(self, request, num_new_tokens, *args, **kwargs):
        # Native waiting statistics use the continuity class. Record physical
        # protection separately: a split-priority return may still be pinned.
        if request.num_computed_tokens == 0:
            program = self.program(request)
            request._holding_physical_pin = program in self.pins
            request._holding_priority = self._has_continuity_priority(program)
        return super()._allocate(request, num_new_tokens, *args, **kwargs)
