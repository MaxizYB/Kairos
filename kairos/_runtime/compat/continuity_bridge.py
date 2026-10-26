"""Kairos continuity bridge implementation."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time

from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus
from kairos._runtime.layout.pcv_scheduler import PCVScheduler
from kairos._runtime.compat.continuity_bridge_model import decode_samples, conditional_samples, prefix_restore, Release, pressure_choice, completion_reserve, native_completion_reserve, decode_population, residual_strata, oracle_program_service, program_service_population, program_service_quote, choose_fitting_follower, continuation_target, continuation_growth_debt, common_history_pages, continuation_queue_rank


class ContinuityBridgeScheduler(PCVScheduler):
    _FAIR_METHODS=dict(PCVScheduler._FAIR_METHODS,**{
        f'continuity-bridge-{bits}-partial':
          ('replica-joint-demand-bridge-admit-partial' if bits[2]=='1' else 'continuum-index',
           'continuity-bridge') for bits in ('011','101','110','111')},**{
        'continuity-bridge-011-oracle-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-capacityonly-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-programoracle-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-program-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-fit-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-fitoracle-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-layoutttl-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-nocputtl-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-growth-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-growthoracle-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-frontier-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-resident-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-101-returnpriority-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge'),
        'continuity-bridge-011-exactgraph-partial':
            ('replica-joint-demand-bridge-admit-partial','continuity-bridge')})

    def __init__(self,*args,**kwargs):
        self.bridge_flags=(False,)*3;self.bridge_stage={};self.bridge_samples=[]
        self.bridge_dirty=True;self.bridge_last_signature=None;self.bridge_host={};self.bridge_priorities={}
        self.bridge_completion=os.environ.get('CONTINUITY_BRIDGE_COMPLETION','0')=='1'
        self.bridge_oracle=False
        self.bridge_program_oracle=False;self.bridge_oracle_trace=None
        self.bridge_program_online=False;self.bridge_service_populations={}
        self.bridge_capacity_only=False;self.bridge_populations={}
        self.bridge_fit=False;self.bridge_fit_blocked=None
        self.bridge_layoutttl=False;self.bridge_nocputtl=False
        self.bridge_growth=False;self.bridge_growth_oracle=False;self.bridge_targets={}
        self.bridge_resident_priority=False;self.bridge_all_return_priority=False
        self.bridge_fit_counts={};self.bridge_fit_pending={};self.bridge_fit_signature=None
        prior=os.environ.get('CONTINUITY_PHASE_PRIOR')
        self.bridge_prior=json.loads(Path(prior).read_text()) if prior else {}
        if set(self.bridge_prior.get('heldout_ids',())) & set(self.bridge_prior.get('evaluation_ids',())):
            raise ValueError('Phase-prior training/evaluation overlap')
        super().__init__(*args,**kwargs)
        self.bridge_native_allocate=self.original_allocate
        self.original_allocate=self._bridge_allocate

    def _switch_experiment(self,name):
        if name==self.experiment:return
        match=re.match(r'continuity-bridge-([01]{3})(-oracle|-capacityonly|-programoracle|-program|-fit|-fitoracle|-layoutttl|-nocputtl|-growth|-growthoracle|-frontier|-resident|-returnpriority|-exactgraph)?-partial',name)
        self.fast_joint_frontier=bool(match and match[2]=='-frontier')
        self.bridge_resident_priority=bool(match and match[2] in ('-resident','-returnpriority'))
        self.bridge_all_return_priority=bool(match and match[2]=='-returnpriority')
        self.bridge_resident_signature=None
        self.bridge_flags=tuple(c=='1' for c in match[1]) if match else (False,)*3
        self.bridge_oracle=bool(match and match[2] in ('-oracle','-fitoracle'))
        self.bridge_fit=bool(match and match[2] in ('-fit','-fitoracle'))
        self.bridge_layoutttl=bool(match and match[2] in ('-layoutttl','-nocputtl'))
        self.bridge_nocputtl=bool(match and match[2]=='-nocputtl')
        self.bridge_growth=bool(match and match[2] in ('-growth','-growthoracle'))
        self.bridge_growth_oracle=bool(match and match[2]=='-growthoracle')
        self.bridge_targets={}
        self.bridge_fit_blocked=None;self.bridge_fit_counts={};self.bridge_fit_pending={}
        self.bridge_fit_signature=None
        self.bridge_program_oracle=bool(match and match[2]=='-programoracle')
        self.bridge_program_online=bool(match and match[2]=='-program')
        self.bridge_service_populations={}
        if (self.bridge_program_oracle or self.bridge_growth_oracle) and self.bridge_oracle_trace is None:
            fixture=os.environ.get('CONTINUITY_BRIDGE_ORACLE_TRACE')
            if not fixture:raise ValueError('Program oracle requires explicit audit fixture')
            self.bridge_oracle_trace={p['id']:p['turns'] for p in
                                      json.loads(Path(fixture).read_text())['programs']}
        self.bridge_capacity_only=bool(match and match[2] in (
            '-capacityonly','-layoutttl','-nocputtl','-growth','-growthoracle','-frontier','-exactgraph'))
        self.bridge_stage={};self.bridge_samples=[];self.bridge_dirty=True
        self.bridge_last_signature=None;self.bridge_host={};self.bridge_priorities={}
        self.bridge_populations={}
        super()._switch_experiment(name)
        if match:
            self._emit('bridge_registry',m1=self.bridge_flags[0],m2=self.bridge_flags[1],
                m3=self.bridge_flags[2],m1_action=(
                    'all-return continuation class; simple no-layout control'
                    if self.bridge_all_return_priority else
                    'unprotected return joins continuation class only with complete exact shared GPU history'
                    if self.bridge_resident_priority else
                    'AUDIT: within-class remaining whole-Program service oracle'
                    if self.bridge_program_oracle else 'within-class conditional whole-Program service'
                    if self.bridge_program_online else 'within-Continuum-class stage-cost admission'),
                m2_action=('service-period headroom; no additional pressure unpin'
                    if self.bridge_capacity_only else 'private suffix release with service-period headroom'
                    if self.bridge_completion else 'pressure-only private suffix release'),
                m3_action='frozen-v21 joint-prefix CPU/GPU placement' if self.bridge_flags[2]
                    else 'original-Continuum LRU placement',
                ttl='original paper objective; no renewal',policy_preemption=False,
                segmented_execution=False,future_trace_information=(
                    self.bridge_program_oracle or self.bridge_growth_oracle),
                prior_sha=self.bridge_prior.get('training_sha256'),
                completion_capacity=self.bridge_completion,decode_oracle=self.bridge_oracle,
                capacity_only=self.bridge_capacity_only,
                layout_ttl=self.bridge_layoutttl,no_cpu_ttl=self.bridge_nocputtl,
                continuation_growth_reserve=self.bridge_growth,
                continuation_growth_oracle=self.bridge_growth_oracle,
                decision_equivalent_frontier=self.fast_joint_frontier,
                residency_continuation_priority=self.bridge_resident_priority,
                all_return_priority_control=self.bridge_all_return_priority,
                exact_decode_graph=bool(match and match[2]=='-exactgraph'),
                growth_action='Soft unallocated next-input envelope for older pinned Programs; '
                    'native running/returning service has precedence; no reservation when no running work',
                fit_bypass=self.bridge_fit,fit_bypass_limit=2,
                fit_action='oldest capacity-funded follower after native head allocation failure',
                objective='bounded next-stage delay proxy; no global JCT optimality claim')

    def _continuation_restore_estimate(self,request,length,default):
        if self.bridge_growth:
            program=self.program(request)
            if self.bridge_growth_oracle:
                turns=self.bridge_oracle_trace[program]
                stage=self.bridge_stage[program]+1
                if stage>=len(turns):raise ValueError('Growth oracle requested after final call')
                target=math.ceil(min(self.max_model_len,turns[stage]['prompt_tokens'])/self.block_size)
                source='AUDIT: exact future next-prompt length'
            else:
                tool=str(self.metadata(request)['cont_tool'])
                rows=self.policy_model.history.get(tool,())
                if len(rows)<8:
                    rows=[r for rs in self.policy_model.history.values() for r in rs]
                target=continuation_target(request.num_prompt_tokens,
                    [r[1] for r in rows],self.max_model_len,self.block_size)
                source='observed tool-conditioned mean input growth; no future labels'
            self.bridge_targets[program]=target
            self._emit('bridge_continuation_envelope',program=program,request=request.request_id,
                prompt_tokens=request.num_prompt_tokens,computed_tokens=length,target_pages=target,
                source=source,oracle=self.bridge_growth_oracle)
        if not self.bridge_layoutttl:return default
        started=time.perf_counter()
        coverage=self._replica_call(dict(op='coverage',tokens=list(request.all_token_ids[:length]),
            model=self.connector.scheduler_adapter.model_name,
            salt=str(request.cache_salt or ''),kv_rank=16777472,local_tokens=0))
        cpu=set() if self.bridge_nocputtl else set(coverage['cpu_present_chunk_indices'])
        # Evaluate loss of GPU protection against the concrete native CPU
        # prefix path. This is a current snapshot, not guaranteed future
        # insurance or a claim that unpin immediately overwrites every page.
        quote=prefix_restore(length,0,cpu,self.beta,
            self.profile.get('reload_fixed_seconds',0.),
            self.profile.get('reload_seconds_per_token',.00001))
        reload=quote['seconds']
        self._emit('bridge_layout_ttl_quote',program=self.program(request),
            request=request.request_id,length=length,original_reload_s=default,
            quote=quote,committed_cpu_prefix=coverage['cpu_prefix_tokens'],
            committed_cpu_chunks=len(coverage['cpu_present_chunk_indices']),
            no_cpu_assumption=self.bridge_nocputtl,controller_s=time.perf_counter()-started)
        return reload

    def add_request(self,request):
        old_history=self.history_gpu_hashes.get(self.program(request),())
        super().add_request(request)
        if self.bridge_resident_priority and getattr(request,'_continuum_returning',False):
            request._bridge_shared_input_pages=common_history_pages(old_history,request.block_hashes)
            self._emit('bridge_shared_history',program=self.program(request),
                request=request.request_id,shared_input_pages=request._bridge_shared_input_pages,
                source='observed exact hash prefix of previous and current input; no future labels')
        if not any(self.bridge_flags):return
        p=self.program(request);self.bridge_stage[p]=self.bridge_stage.get(p,-1)+1
        self.bridge_dirty=True

    def _bridge_rows(self,request):
        if self.bridge_oracle:
            return ((max(1,request.max_tokens-request.num_output_tokens),)*8,
                    dict(source='explicit-fixed-replay-cap-oracle',rows=1,excluded_censored=0))
        if request.request_id not in self.bridge_populations:
            rows=self.bridge_prior.get('rows',[])+self.bridge_samples
            self.bridge_populations[request.request_id]=decode_population(rows,
                self.bridge_stage.get(self.program(request),0),request.num_prompt_tokens)
        xs,info=self.bridge_populations[request.request_id]
        samples,source=residual_strata(xs,request.num_output_tokens)
        return samples,dict(info,source=source)

    def _bridge_step(self):
        n=max(1,min(self.max_num_running_reqs,len(self.running)+1))
        return float(self.profile.get('decode_step_by_seqs',{}).get(str(n),
                     self.profile.get('decode_step_s',.0205)))

    def _bridge_quote(self,request):
        pool=self.kv_cache_manager.block_pool;gpu=0
        if request.status==RequestStatus.RUNNING:
            gpu=request.num_computed_tokens
        else:
            for h in request.block_hashes[:(request.num_prompt_tokens-1)//self.block_size]:
                if pool.get_cached_block(h,[0]) is None:break
                gpu+=self.block_size
        cpu=set(self.bridge_host.get(self.program(request),{}).get('cpu_chunks',()))
        return prefix_restore(request.num_prompt_tokens,gpu,cpu,self.beta,
            self.profile.get('reload_fixed_seconds',0.),
            self.profile.get('reload_seconds_per_token',.00001))

    def _refresh_priority(self):
        super()._refresh_priority()
        if not self.bridge_flags[0]:return
        if self.bridge_resident_priority:
            self._bridge_resident_queue()
            return
        if self.bridge_fit:
            self._bridge_fit_priority()
            return
        rows=list(self.waiting)+list(self.skipped_waiting)
        signature=tuple(sorted((r.request_id,r.status.name,r.num_computed_tokens,
            self._has_continuity_priority(self.program(r))) for r in rows))
        # No scan/RPC on ordinary decode steps. Changes in free capacity are
        # handled by the native allocator, not a speculative admission mask.
        if not self.bridge_dirty and signature==self.bridge_last_signature:
            for r in rows:
                if r.request_id in self.bridge_priorities:r.priority=self.bridge_priorities[r.request_id]
            self._bridge_rebuild_queues()
            return
        self.bridge_last_signature=signature;self.bridge_dirty=False
        if len(rows)<2:return
        started=time.perf_counter()
        self.bridge_host=self._replica_call(dict(op='cpu_residency'))['programs']
        scores={};now=time.time();step=self._bridge_step()
        for r in rows:
            p=self.program(r)
            restored=((r.status==RequestStatus.WAITING_FOR_REMOTE_KVS and
                r.request_id in self.finished_recving_kv_req_ids) or
                (r.status==RequestStatus.WAITING and r.num_computed_tokens>0))
            rank=-1 if restored else (0 if r.status==RequestStatus.PREEMPTED else
                                      1 if self._has_continuity_priority(p) else 2)
            samples,forecast=self._bridge_rows(r);q=self._bridge_quote(r)
            cost=q['seconds']+sum(samples)/len(samples)*step
            if self.bridge_program_oracle:
                forecast=oracle_program_service(self.bridge_oracle_trace[p],
                    self.bridge_stage[p],self.beta,step,q['seconds'],r.num_output_tokens)
                cost=forecast['seconds']
            elif self.bridge_program_online:
                if r.request_id not in self.bridge_service_populations:
                    self.bridge_service_populations[r.request_id]=program_service_population(
                        self.bridge_prior['rows'],self.bridge_stage[p],r.num_prompt_tokens,self.beta)
                population,info=self.bridge_service_populations[r.request_id]
                forecast=dict(info,**program_service_quote(population,r.num_output_tokens,step,q['seconds']))
                cost=forecast['seconds']
            wait=max(0.,now-r.arrival_time)
            # Within the existing continuation class, short work goes first
            # unless it has waited longer than three predicted stage costs.
            aged=wait>max(10.,3*cost)
            key=-min(wait,1000.) if aged else min(cost,1000.)
            r.priority=rank*10**12+int(key*10**6)*100+self.order.get(p,0)
            self.bridge_priorities[r.request_id]=r.priority
            scores[p]=dict(rank=rank,cost_s=cost,restore=q,forecast=forecast,
                           waited_s=wait,aged=aged)
        self._bridge_rebuild_queues()
        self._emit('bridge_admission_order',scores=scores,controller_s=time.perf_counter()-started)

    def _bridge_resident_queue(self):
        """Preserve original Program FCFS inside a restoration-qualified class.

        This reads the incrementally validated native hash index, takes no
        references and changes no running request, pin, lease, or DMA. Loss
        of the shared-history frontier immediately removes the queue hint.
        Native lookup/allocation remain the later cache-hit authority.
        """
        started=time.perf_counter();facts=[];changed=False
        for request in list(self.waiting)+list(self.skipped_waiting):
            p=self.program(request)
            restored=((request.status==RequestStatus.WAITING_FOR_REMOTE_KVS
                and request.request_id in self.finished_recving_kv_req_ids)
                or (request.status==RequestStatus.WAITING and request.num_computed_tokens>0))
            row=self.gpu_index.rows.get(p,{}) if self.incremental_gpu else {}
            required=getattr(request,'_bridge_shared_input_pages',0)
            resident=row.get('frontier',0)
            pinned=self._has_continuity_priority(p)
            rank=continuation_queue_rank(restored,request.status==RequestStatus.PREEMPTED,
                pinned,getattr(request,'_continuum_returning',False),required,resident,
                self.bridge_all_return_priority)
            priority=rank*1000000+self.order.get(p,999999)
            changed |= priority!=request.priority
            request.priority=priority
            facts.append(dict(request=request.request_id,program=p,rank=rank,
                protected=pinned,returning=getattr(request,'_continuum_returning',False),
                common_pages=required,resident_pages=resident,
                promoted=rank==1 and not pinned))
        if changed:self._bridge_rebuild_queues()
        signature=tuple((f['request'],f['rank'],f['promoted']) for f in facts)
        if signature!=self.bridge_resident_signature:
            self.bridge_resident_signature=signature
            self._emit('bridge_resident_queue',candidates=facts,
                all_return_control=self.bridge_all_return_priority,
                changed_priority=changed,controller_s=time.perf_counter()-started,
                extra_gpu_references=0,extra_copy_bytes=0)

    def _bridge_fit_priority(self):
        """Change queue authority, while preserving physical Continuum pin.

        The previously failed head remains protected and queued. A follower
        only borrows an actually empty slot with a funded capacity envelope.
        Ready asynchronous restores and native preempted requests retain
        precedence. No CPU lookup/RPC is added to this decision.
        """
        blocked=self.bridge_fit_blocked
        if not blocked or len(self.running)>=self.max_num_running_reqs:return
        rows=list(self.waiting)+list(self.skipped_waiting)
        head=next((r for r in rows if r.request_id==blocked),None)
        if head is None:return
        highest=min(rows,key=lambda r:(r.priority,r.arrival_time))
        if highest is not head or head.status!=RequestStatus.WAITING:return
        if head.num_computed_tokens:return
        pool=self.kv_cache_manager.block_pool;free=pool.get_num_free_blocks()
        # The previous failure is only a trigger, not a capacity certificate.
        # A request may complete between batches and fund the original head.
        # Refresh native local-hit/allocation metadata before any inversion.
        hits,hit_tokens,_=self.kv_cache_manager.get_computed_blocks(head)
        head_pressure=self._required_capacity(head,
            min(self.max_num_scheduled_tokens,max(1,head.num_tokens-hit_tokens)),
            dict(new_computed_blocks=hits,num_new_computed_tokens=hit_tokens,
                full_sequence_must_fit=True,has_scheduled_reqs=bool(self.running)))
        if head_pressure['D']<=0:
            self.bridge_fit_blocked=None;self.bridge_fit_signature=None
            self._emit('bridge_fit_head_recovered',request=blocked,
                program=self.program(head),pressure=head_pressure)
            return
        active=list(self.running)+[r for r in rows if r.status==RequestStatus.WAITING_FOR_REMOTE_KVS]
        owned={r.request_id:sum(not b.is_null for b in
            self.kv_cache_manager.get_blocks(r.request_id).blocks[0]) for r in active}
        signature=(blocked,free,tuple((r.request_id,owned[r.request_id],r.num_output_tokens//16)
                   for r in active),tuple(r.request_id for r in rows),
                   self.bridge_fit_counts.get(blocked,0))
        # Recheck age eligibility without fitting populations every token.
        now=time.time()
        if signature==self.bridge_fit_signature and now<getattr(self,'bridge_fit_recheck',0.):return
        self.bridge_fit_signature=signature;self.bridge_fit_recheck=now+.05
        started=time.perf_counter();live=[]
        for r in active:
            samples,_=self._bridge_rows(r)
            live.append((r.request_id,r.num_prompt_tokens,r.num_output_tokens,
                         owned[r.request_id],samples))
        growth=sum(completion_reserve(live,self.block_size).values())
        candidates=[]
        for r in rows:
            if r is head or r.status!=RequestStatus.WAITING or r.num_computed_tokens:continue
            if r.priority<=head.priority:continue
            # A conservative physical quote: no APC or CPU hits are assumed.
            samples,_=self._bridge_rows(r)
            tail=sorted(samples)[-1]
            target=math.ceil((r.num_prompt_tokens+r.num_output_tokens+tail-1)/self.block_size)
            compute=max(0.,self.beta[0]+self.beta[1]*r.num_prompt_tokens+
                self.beta[2]*r.num_prompt_tokens*(r.num_prompt_tokens+1)/2+self.beta[3])
            service=compute+sum(samples)/len(samples)*self._bridge_step()
            candidates.append(dict(request=r.request_id,program=self.program(r),
                order=self.order[self.program(r)],target_pages=target,
                wait_s=max(0.,now-r.arrival_time),service_s=service))
        chosen=choose_fitting_follower(candidates,free,growth,len(self.running),
            self.max_num_running_reqs,self.bridge_fit_counts.get(blocked,0),
            head_deficit_pages=head_pressure['D'])
        if chosen:
            candidate=next(r for r in rows if r.request_id==chosen['request'])
            candidate.priority=head.priority-1
            self.bridge_fit_pending[candidate.request_id]=blocked
            self._bridge_rebuild_queues()
        self._emit('bridge_fit_decision',blocked=blocked,head_program=self.program(head),
            free_pages=free,live_growth_pages=growth,running=len(self.running),
            candidates=candidates,chosen=chosen,bypasses=self.bridge_fit_counts.get(blocked,0),
            controller_s=time.perf_counter()-started,decode_oracle=self.bridge_oracle,
            physical_unpin=False,proactive_d2h=False,head_pressure=head_pressure)

    def _run_native_schedule(self,throttle_prefills=False):
        output=super()._run_native_schedule(throttle_prefills=throttle_prefills)
        if self.bridge_fit:
            for rid in output.num_scheduled_tokens:
                blocked=self.bridge_fit_pending.pop(rid,None)
                if blocked is not None:
                    self.bridge_fit_counts[blocked]=self.bridge_fit_counts.get(blocked,0)+1
                    self.bridge_fit_signature=None
                    self._emit('bridge_fit_admitted',request=rid,blocked=blocked,
                        bypass_count=self.bridge_fit_counts[blocked])
        return output

    def _bridge_rebuild_queues(self):
        for name in ('waiting','skipped_waiting'):
            queue=create_request_queue(self.policy)
            for r in getattr(self,name):queue.add_request(r)
            setattr(self,name,queue)

    def _allocate(self,request,num_new_tokens,*args,**kw):
        if (self.bridge_growth and request not in self.running and self.running
                and request.status==RequestStatus.WAITING
                and request.num_computed_tokens==0
                and self.pins):
            ready={self.program(r):r.num_prompt_tokens
                   for q in (self.waiting,self.skipped_waiting) for r in q}
            # Existing Continuum FCFS can admit a later protected return
            # while an earlier Program is still in its tool phase. Its
            # growth can consume that earlier continuation's free extension
            # just as a cold admission can. Only older promises constrain
            # this admission; younger quotas never invert Program order.
            older={p:len(pin.blocks) for p,pin in self.pins.items()
                   if self.order[p]<self.order[self.program(request)]}
            debt=continuation_growth_debt(older,
                self.bridge_targets,ready,self.program(request),self.max_model_len,self.block_size)
            if debt:
                pressure=self._required_capacity(request,num_new_tokens,kw)
                reserve=native_completion_reserve(pressure['admission_required'],
                    pressure['incremental_required'],kw.get('reserved_blocks',0),sum(debt.values()))
                if max(pressure['admission_required'],
                       pressure['incremental_required']-kw.get('reserved_blocks',0)+reserve)>pressure['free'] \
                        and pressure['D']==0:
                    # A soft quota must not enter native deadlock fallback:
                    # no physical pin is withdrawn for an advisory shortfall.
                    self._emit('bridge_continuation_guard',request=request.request_id,
                        program=self.program(request),debt_pages=debt,pressure=pressure,
                        reserved_blocks=reserve,action='defer-later-program-admission',
                        oracle=self.bridge_growth_oracle)
                    return None
                if pressure['D']==0:
                    kw=dict(kw,reserved_blocks=reserve)
        if (self.bridge_completion and self.bridge_flags[1] and request not in self.running):
            # A successful current chunk must not consume the pages that
            # already-running calls need to reach their next release. Quote
            # each live history once; max_tokens is never used online.
            owned={r.request_id:len(self.kv_cache_manager.get_blocks(r.request_id).blocks[0])
                   for r in self.running}
            active=[]
            for r in self.running:
                samples,_=self._bridge_rows(r)
                active.append((r.request_id,r.num_prompt_tokens,r.num_output_tokens,
                               owned[r.request_id],samples))
            reserve=completion_reserve(active,self.block_size)
            samples,_=self._bridge_rows(request)
            future=max(0,math.ceil((request.num_prompt_tokens+
                request.num_output_tokens+sorted(samples)[-1]-1)/self.block_size)
                -math.ceil(request.num_tokens/self.block_size))
            pressure=self._required_capacity(request,num_new_tokens,kw)
            # Native reserved_blocks is charged to incremental allocation.
            # Bridge the gap to full-prompt admission so chunked allocation
            # cannot advertise the same unallocated history as headroom.
            extra=native_completion_reserve(pressure['admission_required'],
                pressure['incremental_required'],kw.get('reserved_blocks',0),
                sum(reserve.values())+future)
            kw=dict(kw,reserved_blocks=extra)
            self._emit('bridge_completion_capacity',request=request.request_id,
                program=self.program(request),running_reserve=reserve,
                incoming_growth=future,extra_reserved=extra,decode_oracle=self.bridge_oracle)
        return super()._allocate(request,num_new_tokens,*args,**kw)

    def _bridge_allocate(self,request,num_new_tokens,*args,**kw):
        result=self.bridge_native_allocate(request,num_new_tokens,*args,**kw)
        if self.bridge_fit and request not in self.running:
            if result is None and self._required_capacity(request,num_new_tokens,kw)['D']>0:
                self.bridge_fit_blocked=request.request_id
            elif result is not None and request.request_id==self.bridge_fit_blocked:
                self.bridge_fit_blocked=None;self.bridge_fit_signature=None
        if (result is not None or not self.bridge_flags[1] or self.bridge_capacity_only
                or not self.pins):return result
        # First actual failed allocation is authoritative. Never confuse
        # waiting for a D2H source/commit with lack of capacity.
        if self.connector.has_pending_repairs():return result
        computed=kw.get('new_computed_blocks')
        excluded={b.block_id for group in computed.blocks for b in group} if computed else set()
        pressure=self._required_capacity(request,num_new_tokens,kw)
        if pressure['D']<=0:return result
        ready={self.program(r) for q in (self.waiting,self.skipped_waiting) for r in q}
        rows=[row for row in self._safe_suffix_rows(excluded) if row['id'] not in ready]
        if not rows:return result
        if sum(row['limit'] for row in rows)<pressure['D']:
            self._emit('bridge_pressure_insufficient',request=request.request_id,
                deficit=pressure['D'],private_idle_pages=sum(row['limit'] for row in rows))
            return result
        started=time.perf_counter();now=time.monotonic()
        self.bridge_host=self._replica_call(dict(op='cpu_residency'))['programs']
        releases=[];pool=self.kv_cache_manager.block_pool
        for r in self.running:
            if r is request:continue
            pages=sum(b.ref_cnt==1 for b in self.kv_cache_manager.get_blocks(r.request_id).blocks[0])
            samples,_=self._bridge_rows(r);q=self._bridge_quote(r)
            times=tuple(q['seconds']+s*self._bridge_step() for s in samples)
            if pages:releases.append((pages,times))
        candidates=[];details={}
        for row in rows:
            p=row['id'];pin=self.pins[p]
            # Release at most the true deficit; candidates are suffix bundles.
            count=min(row['limit'],pressure['D'])
            length=len(pin.blocks)*self.block_size+1
            cpu=set(self.bridge_host.get(p,{}).get('cpu_chunks',()))
            before=prefix_restore(length,length-1,cpu,self.beta,
                self.profile.get('reload_fixed_seconds',0.),
                self.profile.get('reload_seconds_per_token',.00001))
            after=prefix_restore(length,(len(pin.blocks)-count)*self.block_size,cpu,self.beta,
                self.profile.get('reload_fixed_seconds',0.),
                self.profile.get('reload_seconds_per_token',.00001))
            age=max(0.,now-pin.started)
            vals=[v[0] for v in self.policy_model.history.get(pin.tool,())]
            if len(vals)<8:vals=[v[0] for rs in self.policy_model.history.values() for v in rs]
            tools,source=conditional_samples(vals,age)
            expires=max(0.,pin.deadline-now)
            # Layout penalty is conditioned on return before the ORIGINAL
            # lease ends. After expiry baseline protection is also gone.
            prob=sum(t<=expires for t in tools)/len(tools)
            penalty=max(0.,after['seconds']-before['seconds'])*prob
            candidates.append(Release(p,count,penalty,expires))
            details[p]=dict(before=before,after=after,conditional_return_probability=prob,
                            tool_forecast=source,tool_age_s=age,remaining_ttl_s=expires)
        decision=pressure_choice(candidates,pressure['D'],releases,
                                 len(self.waiting)+len(self.skipped_waiting))
        self._emit('bridge_pressure_decision',request=request.request_id,program=self.program(request),
            pressure=pressure,decision={k:[asdict(c) for c in v] if k=='selected' else v
                                       for k,v in decision.items()},quotes=details,
            controller_s=time.perf_counter()-started)
        if decision['action']=='release':
            for victim in decision['selected']:
                self._release(victim.program,'bridge_pressure',victim.pages)
            self.bridge_dirty=True
            return self.bridge_native_allocate(request,num_new_tokens,*args,**kw)
        return result

    def _free_request_blocks(self,request):
        if any(self.bridge_flags) and request.is_finished():
            sample=dict(
                stage=self.bridge_stage.get(self.program(request),0),prompt_tokens=request.num_prompt_tokens,
                output_tokens=request.num_output_tokens,
                censored=request.status==RequestStatus.FINISHED_LENGTH_CAPPED)
            self.bridge_samples=(self.bridge_samples+[sample])[-256:]
            # Only genuine new EOS labels change the fitted population.
            # Fixed teacher-cap completions are excluded by the estimator.
            if not sample['censored']:self.bridge_populations.clear()
            else:self.bridge_populations.pop(request.request_id,None)
            self.bridge_dirty=True
            self.bridge_service_populations.pop(request.request_id,None)
        return super()._free_request_blocks(request)

    def schedule(self,throttle_prefills=False):
        started=time.perf_counter();output=super().schedule(throttle_prefills=throttle_prefills)
        if os.environ.get('CONTINUUM_KERNEL_DIAGNOSTIC')=='1':
            output._kernel_experiment=str(self.experiment)
        if os.environ.get('CONTINUITY_DENSE_DECODE_CAPTURE')=='1':
            output._continuum_exact_decode_graph='-exactgraph-' in str(self.experiment)
            output._continuum_graph_experiment=self.experiment
        if (any(self.bridge_flags) or os.environ.get('CONTINUITY_BRIDGE_PROFILE_CONTROLS')=='1') \
                and output.num_scheduled_tokens:
            self._emit('bridge_schedule_wall',seconds=time.perf_counter()-started)
        return output
