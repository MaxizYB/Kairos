"""Kairos commitment scheduler implementation."""
import math
import time

from kairos._runtime.compat.probation_scheduler import ProbationScheduler
from kairos._runtime.compat.online_path_scheduler import FinitePathProtection
from kairos.protection import CommitmentHoldPolicy
from kairos._runtime.compat.commitment_order import refine_order
from kairos._runtime.compat.causal_action_q import action_q_order, exact_action_q_order


class CommitmentScheduler(ProbationScheduler):
    def _r5_dispatch_refinement(self, by, original_order, quotes, entered, scores, order):
        """Dispatch-only extension point; the historical policy is identity.

        Capacity, legal reclaim and grants remain below this boundary. A
        refinement must preserve the complete victim/continuity owner rank.
        """
        return scores, order

    def _r5_continuity_fraction(self, request):
        """Local replacement for the global memoryfulness coefficient.

        Continuum's eta is already a system-level continuity fraction.  The
        first R5 implementation multiplied that same benefit by the number
        of visible Programs, double-counting the completion-area effect and
        making every long wait purchase an extreme lease.  Here the fraction
        is derived from the current committed order: among visible ready or
        running peers, only those downstream of this continuation can be
        delayed by its recovery.  Tool-idle Programs are excluded because
        they cannot be delayed until they become ready.
        """
        queued = [r for q in (getattr(self, 'waiting', ()),
                              getattr(self, 'skipped_waiting', ())) for r in q
                  if getattr(r, 'request_id', None) not in getattr(self, 'online_granted', set())]
        running = list(getattr(self, 'running', ()))
        owner = getattr(self, 'program', lambda r: getattr(r, 'request_id', ''))(request)
        names = {owner}
        names.update(self.program(r) for r in queued)
        names.update(self.program(r) for r in running)
        # A lost prefix delays the owner and only the calls that follow it in
        # the currently committed service order.  Counting every visible
        # Program was an upper bound, not a marginal completion-area quote,
        # and made the hold policy protect pages for unrelated/preceding
        # calls.  The order is observable and causal; unknown names retain
        # their arrival order, so no future rank is introduced.
        # Dispatch Q is a one-call intervention.  It must not silently
        # change the completion-continuity value used by the later pin/TTL
        # decision.  Keep the incumbent causal (victim/continuity) rank for
        # that second mechanism whenever action Q is enabled.
        continuity_rank = (getattr(self, 'online_victim_rank', None)
                           if getattr(self, 'r5_hold_rank', False) else None)
        continuity_rank = continuity_rank or self.online_rank
        def rank(p):
            return (continuity_rank.get(p, self.order.get(p, 0)),
                    self.order.get(p, 0))
        owner_rank = rank(owner)
        peers = max(1, len(names) - 1)
        downstream = sum(rank(p) > owner_rank for p in names if p != owner)
        return float(downstream / peers)

    def _r5_opportunity_rate(self, request, length):
        """Continuous marginal cost of holding optional pages in this state.

        A pinned page consumes no completion area while all currently ready
        calls fit in the post-completion free capacity.  Once the frontier is
        short, the rate grows with the unresolved fraction of the ready
        capacity demand.  This is a physical deficit ratio, rather than a
        hand-set TTL threshold.
        """
        if getattr(self, 'r5_feasible_hold', False):
            return self._r5_feasible_hold_rate(request)
        queues = [r for q in (getattr(self, 'waiting', ()),
                              getattr(self, 'skipped_waiting', ())) for r in q
                  if getattr(r, 'request_id', None) not in getattr(self, 'online_granted', set())]
        if not queues:
            return 0.0
        pool = getattr(getattr(self, 'kv_cache_manager', None), 'block_pool', None)
        if pool is None:
            return 1.0
        free = int(pool.get_num_free_blocks())
        # The completed call is about to release its committed prefix. Include
        # it in the available post-handover capacity, but never inspect any
        # unobserved future turn.
        released = max(0, (int(length) + self.block_size - 1) // self.block_size)
        available = free + released
        demand = 0
        for row in queues:
            prompt = int(getattr(row, 'num_prompt_tokens', 0))
            cap = int(getattr(row, 'max_tokens', 0))
            demand += max(0, (prompt + cap + self.block_size - 1) // self.block_size - available)
        if demand <= 0:
            return 0.0
        # 32 pages is the inherited native allocator safety headroom used by
        # the base scheduler as well.  It is an execution-safety reserve, not
        # an R5 reward/priority threshold; keep it explicit in the event log
        # and in the method documentation so the two meanings are not mixed.
        native_headroom = 32
        capacity = max(1, int(self._capacity()) - native_headroom)
        exposed = min(1.0, len(queues) / max(1, int(self.max_num_running_reqs)))
        return min(1.0, demand / capacity) * exposed

    def _r5_feasible_hold_rate(self, request):
        from kairos._runtime.compat.lease_opportunity import hold_opportunity
        from kairos._runtime.compat.r83_plan import Call, State
        owner = self.program(request)
        rows = {self.program(r): r for q in (self.waiting, self.skipped_waiting)
                for r in q if r.request_id not in self.online_granted
                and r.request_id != request.request_id}
        returns = {p: r for p, r in rows.items() if self.online_outputs.get(p)}
        rows = returns or rows
        locked = {self.program(r) for r in self.running}
        locked.update(p for p, r in self.online_requests.items()
                      if r.request_id in self.online_granted)
        locked.discard(owner)
        victims = [p for p in self.pins if p != owner and p not in locked
                   and p not in self.online_cleaning and self._online_idle_reclaimable(p)]
        rank = getattr(self, 'online_victim_rank', None) or self.online_rank
        victims.sort(key=lambda p: rank.get(p, self.order.get(p, 0)), reverse=True)
        # The completed owner is reclaimable after its DMA cleanup, exactly
        # like its other optional leases. No physical reference is freed here.
        victims.append(owner)
        state = State(tuple(sorted(self.online_reserved.items())),
                      frozenset(self.online_active), tuple(victims), frozenset(locked))
        calls = [Call(p, self._online_target(r), p in self.online_entered)
                 for p, r in rows.items()]
        limit = self._capacity() - 32
        rate, witness = hold_opportunity(state, owner, calls, limit, self.max_num_running_reqs)
        self._emit('r5_hold_opportunity', program=owner, request=request.request_id,
                   rate=rate, witness=witness, limit=limit,
                   max_active=self.max_num_running_reqs,
                   reserved=dict(state.reserved), active=sorted(state.active),
                   locked=sorted(state.locked), victims=list(state.victims),
                   calls=[dict(program=c.program, pages=c.pages, entered=c.entered) for c in calls],
                   scope='post-owner-cleanup admission feasibility; no future JCT estimate')
        return rate

    def _switch_experiment(self,name):
        # Request metadata repeats the experiment name on every arrival.  The
        # base scheduler treats an unchanged name as a no-op; preserve that
        # contract here or the R5 prior/hold policy is rebuilt per request,
        # discarding causal live observations and emitting duplicate registry
        # records.
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        if 'r5callcommitment' not in name.split('~'):
            return
        self.r5_commitment=True
        self.progressive_compete=True
        self.progressive_window_compete=False
        self.progressive_debt_compete=False
        # Both versions expose empirical service quotes; C2 also uses the
        # mean for ordering. SEPT is justified in the all-ready serial
        # special case, not a general optimality theorem for batched agents.
        tags=set(name.split('~'))
        self.r5_lean = 'r5lean' in tags
        self.r5_feasible_hold = 'feasiblehold' in tags
        self.r5_joint_service = not self.r5_lean or 'jointmeanrank' in tags
        self.r5_fifo = 'r5fifo' in tags
        self.r5_use_distribution_order = 'progressivewindow' in tags
        # incumbent call-commitment protocol kept the incumbent snapshot ordinal for reproducibility.  The
        # C2 variant makes the intended statistic executable: after the first
        # observed call, a continuation is ranked by its conditional expected
        # total solo service.  Cold calls still use only their current call;
        # no population suffix is allowed to decide an unobserved program.
        self.r5_mean_rank = 'jointmeanrank' in tags
        # The service-order statistic and the cache-victim order have
        # different objectives.  `splitrank` is an explicit ablation of that
        # distinction: conditional mean service may order a returning call,
        # while optional support is still reclaimed in the incumbent causal
        # continuity order.  Without this tag C2 keeps its coupled rank for a
        # matched comparison.
        self.r5_split_rank = 'splitrank' in tags
        self.r5_dominance_rank = 'dominancerank' in tags
        # R5-D evaluates the next full-call action over the returning cohort
        # only.  It is intentionally separate from victim ranking: a fast
        # service action need not be the right optional KV owner to revoke.
        self.r5_exact_action_q = 'exactactionq' in tags
        self.r5_action_q = 'actionq' in tags or self.r5_exact_action_q
        self.r5_hold_rank = 'holdrank' in tags
        if self.r5_hold_rank and not self.r5_action_q:
            raise ValueError('holdrank requires an explicit action-Q intervention')
        if self.r5_action_q and (self.r5_mean_rank or self.r5_fifo or self.r5_dominance_rank):
            raise ValueError('actionq is an exclusive dispatch intervention')
        # Retain the existing v21 protection lifecycle, but replace the
        # legacy max-support lease with the same completion-area objective.
        # The legacy COST history uses a different tool taxonomy and a
        # different gap collection (for example `get_file`), so mixing it
        # into the Program prior silently creates invalid TTL support.  R5's
        # protection and R,D posterior must share one provenance: the frozen
        # CommitmentPrior tool observations plus causal live updates.
        history={}
        training=getattr(self.forecaster,'tool_samples',{})
        self.policy_model=FinitePathProtection(CommitmentHoldPolicy(history,training,
            allow_live_tail='livetoolcdf' in tags),self.forecaster,
                                               live=False,zero='leasezero' in tags)
        self.policy_model.support_max = 'supportmax' in tags
        self._emit('r5_commitment_registry',
            feasible_hold_opportunity=self.r5_feasible_hold,
            causal_live_tool_tail='livetoolcdf' in tags,
            controller='call-boundary nonpreemptive capacity-feasible scheduler',
            predictor='stage-conditioned empirical service population and tool-gap samples',
            protection='conditional return event boundaries, utility uses expected held time',
            priority='returned-first; current service for cold; posterior order for returns',
            returning_rank=('conditional empirical mean service'
                            if self.r5_mean_rank else 'incumbent snapshot ordinal'),
            victim_rank=('incumbent causal continuity order (service-only mean rank)'
                         if self.r5_mean_rank and self.r5_split_rank else
                         'conditional empirical mean service' if self.r5_mean_rank else
                         'incumbent snapshot ordinal'),
            dispatch_rank=('empirical first-order-dominance swaps'
                           if self.r5_dominance_rank else
                           'conditional empirical mean service'
                           if self.r5_mean_rank else 'incumbent snapshot ordinal'),
            future_information=False, pathforecast=False,
            unused_joint_scenarios_omitted=self.r5_lean,
            protection_ablation=('zero' if 'leasezero' in tags else
                                 'support maximum' if 'supportmax' in tags else 'economic'),
            ordering_ablation='arrival' if self.r5_fifo else None,
            native_headroom_pages=32, global_action_q=False,
            action_q=('causal returning-cohort completion-area Q' if self.r5_action_q else None),
            action_q_solver=('analytic-independent-marginals' if self.r5_exact_action_q
                             else 'common-scenario-quadrature' if self.r5_action_q else None),
            hold_continuity_rank=('incumbent causal order' if self.r5_hold_rank
                                  else 'dispatch order'),
            execution='v21 prefix/CPU recovery, native refs and DMA fences unchanged')

    def _progressive_compete(self,queued,snapshot,limit):
        """Evaluate every ready call in one stable completion-time order.

        The physical commit remains native and nonpreemptive. A call is
        granted only after its full current capacity package fits; optional
        support is revoked only by the existing legal reclaim transaction.
        """
        started=time.perf_counter()
        if snapshot is None:
            self.online_dirty=False; return False
        programs,original_order,initial,quotes,minimum,cleanup=snapshot
        # Lifetime admission is not evidence of a completed observation. A
        # program is mature only after its first full call has been recorded
        # in online_outputs; this keeps an unusual duplicate/early request
        # on the cold policy instead of trusting its unobserved suffix.
        observed_at_decision = {p for p, outputs in self.online_outputs.items()
                                if outputs}
        # A granted request can remain in vLLM's waiting deque until the
        # native scheduler consumes it.  It is already a committed action and
        # must not be reintroduced into this boundary competition.  The R4
        # progressive controller had this guard; the first R5 implementation
        # omitted it, producing duplicate grant events and double-counting
        # reservations under dense returns.
        by={self.program(r):r for r in queued
            if r.request_id not in self.online_granted}
        scores={}
        entered={p for p in by if p in observed_at_decision}
        posterior_rank = {p:i for i,p in enumerate(original_order)}
        mean_rank={}
        if self.r5_mean_rank:
            # Rebuild the complete owner order from quotes, rather than
            # inheriting the lower progressive frontier's ordinal.  Idle
            # owners participate in victim selection but never become ready
            # candidates; cold owners use only their observed current call.
            for p in original_order:
                q=quotes.get(p,{})
                value=(q.get('expected_total_service_s', q.get('total_service_s', 0.))
                       if self.online_outputs.get(p) else
                       q.get('current_service_s', 0.))
                mean_rank[p]=(float(value), self.order.get(p,0))
        for p,r in by.items():
            q=quotes.get(p,{})
            # Current-call service is observed/profiled; R is the population
            # posterior only for the clearing value. No future path rank is
            # copied into the request's priority.
            # Quotes are produced by OnlinePathScheduler in profiled seconds;
            # never reconstruct service from prompt length here.  The old
            # implementation omitted decode work and silently compared a
            # seconds term with a token-count term.
            current=float(q.get('current_service_s',0.))
            residual=float(q.get('residual_service_s',0.))
            # A first call has no identifying history.  Its population
            # residual is therefore a prior, not evidence about this
            # particular program; using it to displace another cold call is
            # precisely the unidentifiable cold-start failure.  The first
            # complete call is the information quantum.  Once a program has
            # paid that quantum, its posterior residual can participate in
            # the completion-area order.
            expected_total=float(q.get('expected_total_service_s', current + residual))
            # C2 uses the empirical posterior expectation produced by the
            # same causal scenarios that built the snapshot.  The old incumbent call-commitment protocol
            # ordinal path remains selectable for a matched historical
            # comparison.  Missing scenarios fall back to the profiled
            # envelope, never to future truth.
            if p in entered and self.r5_mean_rank:
                scores[p]=mean_rank.get(p,(expected_total,self.order.get(p,0)))
            elif p in entered and self.r5_use_distribution_order and p in posterior_rank:
                scores[p]=(float(posterior_rank[p]), self.order.get(p,0))
            else:
                scores[p]=((expected_total) if p in entered else current,
                           self.order.get(p,0))
        order=sorted(by,key=lambda p:scores[p])
        if self.r5_fifo:
            original_order=sorted(original_order,key=lambda p:self.order[p])
            posterior_rank={p:i for i,p in enumerate(original_order)}
            order=sorted(by,key=lambda p:self.order[p])
        service_order = None
        dominance_swaps = []
        action_q = {}
        action_q_ordered = None
        action_q_samples = 0
        action_q_seconds = 0.0
        if getattr(self, 'r5_action_q', False) and len(entered) >= 2:
            # A Q comparison is valid only for the same mature returning
            # cohort.  Cold calls have no individual residual distribution.
            solver = (exact_action_q_order if getattr(self, 'r5_exact_action_q', False)
                      else action_q_order)
            q_started = time.perf_counter()
            action_q_ordered, action_q, action_q_samples = solver(
                tuple(p for p in original_order if p in entered), quotes)
            action_q_seconds = time.perf_counter() - q_started
            if action_q_ordered is not None:
                order = action_q_ordered + [p for p in order if p not in entered]
        if self.r5_mean_rank and self.r5_dominance_rank:
            # Mean order is only trusted when the empirical posterior gives a
            # stronger statement: a returning owner is no slower at every
            # observed service boundary.  Adjacent swaps on the incumbent
            # causal order keep the policy stable when distributions overlap.
            service_order, dominance_swaps = refine_order(
                getattr(self,'r5_incumbent_order',original_order),
                {p:q.get('service_distribution_s',()) for p,q in quotes.items()}, entered)
            # Keep the original cold current-call rule. Only mature ready
            # calls use the uncertainty-aware refinement.
            order=([p for p in service_order if p in entered] +
                   [p for p in order if p not in entered])
        # The same mature posterior order must reach both native waiting
        # priority and v21 victim selection.  Cold calls may still be
        # selected by their observed current-call price below, but their
        # unobserved future must not overwrite the continuation rank used by
        # cache reclaim.
        # The snapshot includes idle owners and running calls as well as the
        # ready set, so keep its complete order for recovery, not a ready-only
        # rank that drops the protected owners.
        if self.r5_mean_rank:
            dispatch_order = (service_order if service_order is not None else
                              sorted(original_order, key=lambda p: mean_rank[p]))
            self.online_rank = {p: i for i, p in enumerate(dispatch_order)}
            # C3 restores the pre-mean incumbent order only for optional
            # cache victims. C2 deliberately couples both objectives.
            victim_order = (getattr(self, 'r5_incumbent_order', original_order)
                            if self.r5_split_rank else dispatch_order)
            self.online_victim_rank = {p: i for i, p in enumerate(victim_order)}
        else:
            self.online_rank = (dict(posterior_rank)
                                if self.r5_use_distribution_order
                                else {p: i for i, p in enumerate(order)})
            self.online_victim_rank = dict(self.online_rank)
            # R5-D must not silently transfer a service-order intervention to
            # physical victims.  Preserve the causal incumbent order for
            # optional support whenever Q is active.
            if getattr(self, 'r5_action_q', False):
                # Preserve every owner, including idle and running owners.
                # Replace eligible returning slots only; a ready-only map
                # changes native priorities and drops cache owners.
                if action_q_ordered is not None:
                    replacement = iter(action_q_ordered)
                    dispatch_order = [next(replacement) if p in entered else p
                                      for p in original_order]
                    self.online_rank = {p: i for i, p in enumerate(dispatch_order)}
                self.online_victim_rank = dict(posterior_rank)
        scores, order = self._r5_dispatch_refinement(
            by, original_order, quotes, entered, scores, order)
        selected=[];blocked=[]
        # Continuations are one committed cohort.  A fresh call can acquire
        # its first-call information only when no returned call is waiting;
        # otherwise a cold population prior could create a false shortcut
        # around an already observed continuation.  This is the same causal
        # boundary used by the R4 incumbent, expressed in the R5 evaluator.
        returns=[p for p in order if p in entered]
        candidates=returns if returns else [p for p in order if p not in entered]
        window=(max(1,self.max_num_running_reqs-len(self.online_active))
                if not returns else len(candidates))
        for p in candidates:
            if len(selected)>=window:
                break
            r=by[p]
            if p in self.online_entered:
                funded=self._online_reclaim(r,limit) if p in self.online_active else self._online_restore_fits(r,limit)
            else:
                funded=self._online_restore_fits(r,limit)
                if not funded and p not in self.online_active:
                    # A cold call may exchange only optional support through
                    # the same transaction; no return-first privilege.
                    funded=self._online_reclaim(r,limit)
            if funded:
                kind='return' if p in entered else 'cold'
                self._online_grant(r,minimum.get(p,self._online_target(r)),'r5-call-commitment')
                selected.append(dict(program=p,score=scores[p],kind=kind))
            else: blocked.append(dict(program=p,score=scores[p]))
        self._emit('r5_commitment_decision',order=order,selected=selected,blocked=blocked,
            dispatch_owner_order=sorted(self.online_rank,key=self.online_rank.get),
            victim_owner_order=sorted(self.online_victim_rank,key=self.online_victim_rank.get),
            dominance_swaps=dominance_swaps,
            candidate_count=len(order),same_action_evaluator=False,
            global_action_q=False,
            one_step_action_q=action_q_ordered is not None,
            action_q_scenarios=action_q_samples,
            action_q_seconds=action_q_seconds,
            action_q_solver=('analytic-independent-marginals'
                             if getattr(self, 'r5_exact_action_q', False) else
                             'common-scenario-quadrature'
                             if getattr(self, 'r5_action_q', False) else None),
            incumbent_owner_order=original_order,
            action_q_values=action_q,
            ranking=('conditional mean service after first observed call; current-call only for cold'
                     if self.r5_mean_rank else 'incumbent snapshot ordinal; current-call only for cold'),
            posterior_order=('causal joint empirical mean' if self.r5_mean_rank else
                             'causal suffix/ordinal incumbent order' if self.r5_use_distribution_order
                             else 'causal joint empirical mean fallback'),
            returning_rank=('causal completion-area Q' if action_q_ordered is not None else
                            'conditional empirical mean service' if self.r5_mean_rank
                            else 'incumbent snapshot ordinal'),
            victim_rank=('incumbent causal continuity order (dispatch-only mean)'
                         if self.r5_mean_rank and self.r5_split_rank else
                         'conditional empirical mean service' if self.r5_mean_rank else
                         'incumbent snapshot ordinal'),
            dispatch_rank=('causal completion-area action Q' if action_q_ordered is not None else
                           'empirical first-order-dominance swaps'
                           if self.r5_dominance_rank else
                           'conditional empirical mean service'
                           if self.r5_mean_rank else 'incumbent snapshot ordinal'),
            service_quotes=({p:dict(current_s=quotes[p].get('current_service_s'),
                                   expected_total_s=quotes[p].get('expected_total_service_s'),
                                   source=quotes[p].get('ranking_source'),
                                   entered=p in observed_at_decision)
                             for p in original_order} if self.r5_mean_rank else {}),
            cold_policy='current-call information quantum; residual withheld until first completion',
            returned_first=bool(returns),
            seconds=time.perf_counter()-started)
        self.online_dirty=False
        return bool(selected)
