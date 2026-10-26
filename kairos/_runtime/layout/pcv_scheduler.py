"""Kairos pcv scheduler implementation."""
from dataclasses import replace
import os
import time
from statistics import median

from kairos._runtime.layout.scheduler import GappedScheduler
from kairos._runtime.layout.pcv import Program, Continuation, conditional_returns, return_probability, asym_frequency
from kairos._runtime.layout.recovery_scheduler import RecoverySchedulerMixin, RECOVERY_MODES, DEFER_MODES
from kairos._runtime.layout.use_scheduler import UseSchedulerMixin
from kairos._runtime.layout.next_use import USE_MODES, AUDIT_MODES
from kairos._runtime.layout.pcvu_control import PCVUControlMixin
from kairos._runtime.compat.layered_baseline import LayeredScheduler
from kairos._runtime.compat.layered_policy import SEGMENTED_LAYERS, CONTINUATION_LAYERS, LAYERS

LAYERED_SEGMENT_MODES = SEGMENTED_LAYERS
CPU_CONTINUATION_MODES = ('gapped-cpu-continuation',) + CONTINUATION_LAYERS


class PCVScheduler(PCVUControlMixin, UseSchedulerMixin, RecoverySchedulerMixin, GappedScheduler):
    _FAIR_METHODS = dict(GappedScheduler._FAIR_METHODS, **{
        'pcv-partial': ('continuum-index', 'gapped-pcv'),
        'pcv-cpu-partial': ('continuum-index', 'gapped-pcv-cpu'),
        'pcv-gpu-partial': ('continuum-index', 'gapped-pcv-gpu'),
        'pcv-return-partial': ('continuum-index', 'gapped-return'),
        'pcv-return-clean-partial': ('continuum-index', 'gapped-return-clean'),
        'asym-paper-partial': ('continuum-index', 'gapped-asym'),
        'asym-tier-partial': ('continuum-index', 'gapped-asym-tier'),
        'pcv-fast-partial': ('continuum-index', 'gapped-fast'),
        'pcv-bundle-partial': ('continuum-index', 'gapped-bundle'),
        'pcv-event-partial': ('continuum-index', 'gapped-event'),
        'pcv-use-partial': ('continuum-index', 'gapped-use'),
        'pcv-survival-partial': ('continuum-index', 'gapped-survival'),
        'pcv-use-survival-partial': ('continuum-index', 'gapped-use-survival'),
        'pcv-enst-partial': ('continuum-index', 'gapped-enst'),
        'pcv-use-discount-partial': ('continuum-index', 'gapped-use-discount'),
        'pcv-use-discount-survival-partial': ('continuum-index', 'gapped-use-discount-survival'),
        'pcv-cpu-use-partial': ('continuum-index', 'gapped-cpu-use'),
        'pcv-cpu-insurance-partial': ('continuum-index', 'gapped-cpu-insurance'),
        'pcv-cpu-continuation-partial': ('continuum-index', 'gapped-cpu-continuation'),
        'pcvu-m2-oracle-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-m2-zero-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-000-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-100-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-010-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-001-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-001-nofinal-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-001-finalonly-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-011-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-011-nofinal-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-111-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-111-oracle-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-100-oracle-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-010-oracle-partial': ('continuum-index', 'gapped-return-clean'),
        'pcvu-001-oracle-partial': ('continuum-index', 'gapped-return-clean'),
    })

    def __init__(self,*args,**kwargs):
        self.pcv_programs={};self.pcv_output_history=[]
        super().__init__(*args,**kwargs)
        self.connector.gapped_plan=self._pcv_recovery_plan
        self.connector.gapped_use_observe=self._use_lookup_observation

    def _model(self):
        return Continuation(self.gapped_costs,self._capacity(),
            self.max_num_scheduled_tokens,self.max_num_running_reqs,
            float(self.profile.get('decode_step_s',.0174)),max(0.,self.beta[0]),
            int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_THRESHOLD','0')),
            int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_TOKENS','512')),
            query_budget=os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET','0') == '1')

    def _pcv_recovery_plan(self,request,gpu,cpu,count,default):
        if getattr(self,'_fair_ablation','')!='gapped-pcv':return default
        start=time.perf_counter();layouts=self._layouts();state=self._program_state(layouts)
        current=self.program(request)
        host=self._replica_call(dict(op='cpu_residency'))['programs']
        programs=[]
        for row in layouts:
            p=row['program']
            if p not in state:continue
            values=state[p]
            if p==current:
                values=dict(values,chunks=count,computed=request.num_computed_tokens,
                    prompt_tokens=request.num_prompt_tokens)
            programs.append(Program(**values,
                gpu=frozenset(gpu if p==current else
                    (i for i,b in enumerate(row['blocks']) if b is not None)),
                cpu=frozenset(cpu if p==current else host.get(p,{}).get('cpu_chunks',()))))
        if not any(p.name==current for p in programs):return default
        model=self._model();options=[];plan_cache={}
        candidates=list(dict.fromkeys((frozenset(default),frozenset(cpu),frozenset())))
        for loads in candidates:
            changed=[replace(p,forced_loads=loads) if p.name==current else p for p in programs]
            options.append((model.objective(changed,cache=plan_cache),len(loads),loads))
        objective,_,chosen=min(options,key=lambda x:x[:2])
        self._emit('pcv_recovery_plan',program=current,request=request.request_id,
            candidates=len(options),default_chunks=sorted(default),load_chunks=sorted(chosen),
            predicted_completion_sum_s=objective,controller_s=time.perf_counter()-start)
        return set(chosen)

    def _switch_experiment(self,name):
        changed = name != self.experiment
        if name!=self.experiment:
            self.pcv_programs={};self.pcv_output_history=[]
        super()._switch_experiment(name)
        if changed:
            override = ('oracle-economic' if name.startswith('pcvu-m2-oracle-') else
                        'zero' if name.startswith('pcvu-m2-zero-') else None)
            if override is not None:
                self.policy_model.mode = override
            self._emit('pcvu_mechanisms', m1='feasible-bypass' if self.pcvu_flags[0] else 'continuum',
                       m2='bounded-cpu-handover' if self.pcvu_flags[1] else self.policy_model.mode,
                       m3=('termination-hint-only' if '-finalonly-' in name else 'next-event-cpu')
                            if self.pcvu_flags[2] else getattr(self, '_fair_ablation', ''),
                       advance_termination_hint=bool(self.pcvu_flags[2] and '-nofinal-' not in name),
                       future_information=override == 'oracle-economic' or self.pcvu_flags[3],
                       cpu_upcoming_tool_type_known=(getattr(self,'_fair_ablation','')
                           in CPU_CONTINUATION_MODES and getattr(self,'_fair_ablation','')
                           not in LAYERED_SEGMENT_MODES),
                       policy_preemption=False, new_tier_lease=self.pcvu_flags[1])

    def _return_program_state(self,layouts):
        now=time.monotonic(); history=self.policy_model.history
        active={self.program(r):r for r in self.requests.values() if not r.is_finished()}
        result={}
        for row in layouts:
            p=row['program']; n=len(row['chunks'])
            if p in self.departures:
                began,tool,old=self.departures[p];age=max(0.,now-began)
                arrivals=conditional_returns(history,tool,age)
                values=[float(v[0]) for v in history.get(tool,())]
                if not values:values=[float(v[0]) for vs in history.values() for v in vs]
                observed_lifespan=sum(values[-128:])/len(values[-128:]) if values else None
                values.sort(); lifespan=values[int(.99*(len(values)-1))] if values else 1.
                additions=[int(v[1]) for v in history.get(tool,()) if v[0]>age]
                append=int(median(additions)) if additions else 128
                phase='tool';computed=0;prompt_tokens=n*256+append
            elif p in active:
                r=active[p]
                age=0.;arrivals=(0.,0.,0.);lifespan=1.;append=max(16,r.num_prompt_tokens-n*256)
                phase=('running' if str(r.status).endswith('RUNNING')
                       or getattr(r.status,'name','')=='RUNNING' else 'ready')
                computed=r.num_computed_tokens;prompt_tokens=r.num_prompt_tokens
                values=[];observed_lifespan=None
            else:
                continue
            pin=self.pins.get(p)
            expected=int(median(self.pcv_output_history)) if self.pcv_output_history else 24
            output=max(1,expected-getattr(active.get(p),'num_output_tokens',0))
            pinned={b.block_id for b in pin.blocks} if pin else set()
            result[p]=dict(name=p,chunks=n,returns=arrivals,
                order=self.order.get(p,1000000),output=output,append=max(16,append),
                phase=phase,protected_s=max(0.,pin.deadline-now) if pin else 0.,
                probability=return_probability(history,tool,age,1.) if phase=='tool' else 1.,
                age=age,lifespan=lifespan,computed=computed,prompt_tokens=prompt_tokens,
                protected_pages=[i for i,b in enumerate(row['blocks']) if b in pinned],
                observed_lifespan=observed_lifespan)
            if getattr(self, '_fair_ablation', '') in CPU_CONTINUATION_MODES:
                # CPU data consumed by the current request is insurance for
                # its *next* continuation, not for an attention already using
                # the live GPU reference. No future trace values are read.
                request=active.get(p)
                meta=self.metadata(request) if request is not None else {}
                # New matched arms use no advance final-round hint. Only a
                # completed lifecycle event removes a program from the model.
                continues=(True if self._fair_ablation in LAYERED_SEGMENT_MODES else
                           phase == 'tool' or not bool(meta.get('cont_final', False)))
                if phase == 'tool':
                    next_times=arrivals
                else:
                    # In a SWE agent, the next command is generated by this
                    # request. Replay annotations must not reveal it early.
                    # Use pooled completed-tool history until tool entry.
                    next_tool=('' if self._fair_ablation in LAYERED_SEGMENT_MODES
                               else str(meta.get('cont_tool', '')))
                    remaining=output*float(self.profile.get('decode_step_s', .0174))
                    if computed < prompt_tokens:
                        remaining+=max(0., self.beta[0])+self.gapped_costs.compute(
                            [(computed,prompt_tokens)])
                    next_times=tuple(remaining+v for v in
                        conditional_returns(history,next_tool,0.))
                result[p].update(cpu_next_times=next_times,cpu_continues=continues)
        self.pcv_programs=result
        return self._pcvu_extend_state(result)

    def _replica_history_extension(self, program, state, tool=None):
        """Publish PCV lifecycle state on the already-required history RPC.

        CPU eviction runs in the LMCache process between scheduler batches.  A
        snapshot sent only when a GPU allocation occurs can therefore still
        say that a request is running after it has entered a tool gap.  That
        makes a duplicate CPU chunk look risk-free exactly when its GPU copy
        is about to age out.  Piggyback the post-transition state after the
        base scheduler records ``departures[p]``; no extra socket round trip is
        introduced.
        """
        if getattr(self, '_fair_ablation', '') not in CPU_CONTINUATION_MODES:
            return super()._replica_history_extension(program, state, tool)
        if state == 'finished':
            return {'pcv_remove_program': str(program)}
        layouts = (self._continuation_layouts() if self._fair_ablation in LAYERED_SEGMENT_MODES
                   else self._layouts())
        snapshot = self._return_program_state(layouts)
        row = snapshot.get(str(program))
        if row is None:
            return {'pcv_lifecycle': dict(program=str(program), phase=state)}
        return {
            'pcv_lifecycle': dict(program=str(program), row=row),
        }

    def _replica_call(self,payload):
        if (payload.get('op')=='hicache_register' and
                getattr(self,'_fair_ablation','') in CPU_CONTINUATION_MODES):
            payload=dict(payload,**self._replica_history_extension(payload['program'],'running'))
        return super()._replica_call(payload)

    def _gapped_state_payload(self,layouts):
        if (getattr(self,'_fair_ablation','') or '') in (
                'gapped-pcv','gapped-pcv-gpu','gapped-pcv-cpu','gapped-return','gapped-return-clean',
                'gapped-asym','gapped-asym-tier','gapped-cpu-continuation') + LAYERED_SEGMENT_MODES + RECOVERY_MODES + USE_MODES:
            state=self._program_state(layouts)
            return dict(pcv_programs=state,pcv_config=dict(
                capacity_pages=self._capacity(),token_budget=self.max_num_scheduled_tokens,
                sequence_budget=self.max_num_running_reqs,
                decode_step=float(self.profile.get('decode_step_s',.0174)),
                fixed_step=float(self.beta[0]),
                adaptive_decode_threshold=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_THRESHOLD','0')),
                adaptive_token_budget=int(os.environ.get('CONTINUUM_GAPPED_ADAPTIVE_TOKENS','512')),
                query_budget=os.environ.get('CONTINUUM_GAPPED_QUERY_BUDGET','0') == '1'))
        return {}

    def _free_request_blocks(self,request):
        output=getattr(request,'num_output_tokens',0)
        if output:self.pcv_output_history=(self.pcv_output_history+[output])[-128:]
        return super()._free_request_blocks(request)

    def _layer_native_reuse(self,count):
        mode=getattr(self,'_fair_ablation','')
        if mode in LAYERS and LAYERS[mode].recovery_gpu:
            if not count:
                return self._layer_native_get_new_blocks(count)
            # The outer native prefix chooser is disabled for this arm.
            # Price the concrete same-history page/chunk masks instead;
            # withdraw actual victims before native allocation overwrites.
            weight=LAYERS[mode].recovery_weight
            if weight not in ('time','uniform'):
                raise ValueError('Unsupported recovery weight: '+weight)
            return self._recovery_native_reuse(count,
                'gapped-next-recovery' if weight=='time' else 'gapped-static-recovery')
        if mode in LAYERED_SEGMENT_MODES:
            # These arms share the existing strongest GPU chooser. Do not
            # run the old per-page Return chooser on top of it. CPU coverage
            # must be withdrawn for its ACTUAL chosen pages before reuse.
            result=LayeredScheduler._layer_native_reuse(self,count)
            if count:
                self._recovery_snapshot(return_cpu=False)
            return result
        if (self.gapped_enabled and count and mode in DEFER_MODES
                and os.environ.get('CONTINUUM_PCV_DEFER_PUBLICATION','0') == '1'):
            queue=self.kv_cache_manager.block_pool.free_block_queue
            if count <= queue.num_free_blocks:
                head=queue.fake_free_list_head.next_free_block
                cached=False
                for _ in range(count):
                    cached |= head.block_hash is not None
                    head=head.next_free_block
                self._recovery_context=self._recovery_snapshot(return_cpu=cached)
                try:
                    return self._pcv_native_reuse(count)
                finally:
                    self._recovery_context=None
        return self._pcv_native_reuse(count)

    def _pcv_native_reuse(self,count):
        mode=getattr(self,'_fair_ablation','')
        if self.gapped_enabled and mode in RECOVERY_MODES and count:
            return self._recovery_native_reuse(count, mode)
        if not self.gapped_enabled or mode not in ('gapped-pcv','gapped-pcv-gpu',
                'gapped-pcv-cpu','gapped-return','gapped-return-clean','gapped-asym','gapped-asym-tier',
                'gapped-cpu-continuation') + USE_MODES or not count:
            return super()._layer_native_reuse(count)
        pool=self.kv_cache_manager.block_pool;queue=pool.free_block_queue
        free=queue.get_all_free_blocks()
        if count>len(free) or all(b.block_hash is None for b in free[:count]):
            return super()._layer_native_reuse(count)
        started=time.perf_counter()
        if self._recovery_context is not None:
            layouts,programs=self._recovery_context
        else:
            layouts=self._layouts();state=self._program_state(layouts)
            cpu=self._replica_call(dict(op='cpu_residency'))['programs']
            programs=[Program(**state[row['program']],
                gpu=frozenset(i for i,b in enumerate(row['blocks']) if b is not None),
                cpu=frozenset(cpu.get(row['program'],{}).get('cpu_chunks',())))
                for row in layouts if row['program'] in state]
        owners={};known={p.name for p in programs}
        for row in layouts:
            p=row['program']
            if p not in known:continue
            for i,b in enumerate(row['blocks']):
                if b is not None:owners.setdefault(b,[]).append((p,i))
        by_name={p.name:p for p in programs}; page_cost=replace(self.gapped_costs,chunk=16)
        use_factors={p.name:self._use_gpu_score(p,1.,0.,True)
                     for p in programs} if mode in USE_MODES else {}
        page_compute={}
        def compute_at(i):
            if i not in page_compute:page_compute[i]=page_cost.chunk_compute(i)
            return page_compute[i]
        def simple(b):
            score=0.
            for p,i in owners.get(b.block_id,()):
                r=by_name[p];compute=compute_at(i)
                value=min(compute,self.gapped_costs.load_token*16+
                          self.gapped_costs.load_call/16) if i//16 in r.cpu else compute
                if mode in USE_MODES:
                    q=self._use_weight(r)
                    s=use_factors[p][1] if i//16 in r.cpu else 0.
                    value=q*(s*min(compute,self.gapped_costs.load_token*16+
                        self.gapped_costs.load_call/16)+(1.-s)*compute)
                    score+=value
                    continue
                score+=(asym_frequency(r.age,r.lifespan,
                    observed_lifespan=r.observed_lifespan)*compute if mode.startswith('gapped-asym')
                        else r.probability*value)
            return score
        ranked=sorted(enumerate(free),key=lambda x:(simple(x[1]),x[0]))
        chosen=[];evaluations=0;budget_fallback=0;decisions=[]
        if mode in ('gapped-pcv','gapped-pcv-gpu') and programs:
            model=self._model();plan_cache={}
            budget_s=float(os.environ.get('CONTINUUM_PCV_BUDGET_S','.040'))
            while len(chosen)<count:
                available=[(rank,b) for rank,b in ranked if b not in chosen]
                if not available:break
                if time.perf_counter()-started>budget_s:
                    tail=[b for _,b in available[:count-len(chosen)]]
                    chosen.extend(tail);budget_fallback+=len(tail);break
                unknown=[b for _,b in available if b.block_id not in owners]
                if unknown:
                    chosen.extend(unknown[:count-len(chosen)]);continue
                remaining=count-len(chosen);groups=[]
                # Include at least one option per program, rather than let a
                # position-only shortlist hide an urgent program completely.
                heads=[];seen=set()
                for pair in available:
                    p=owners[pair[1].block_id][0][0]
                    if p not in seen:heads.append(pair);seen.add(p)
                shortlist=(heads+available[:8])[:12]
                for _,b in shortlist:
                    p,i=owners[b.block_id][0]
                    group=[x for _,x in available if any(op==p and oi//16==i//16
                            for op,oi in owners.get(x.block_id,()))][:remaining]
                    # Native allocation requests a bundle. Evaluate whole
                    # chunk groups first; individual candidates are useful
                    # only at a partially filled chunk or a small deficit.
                    for candidate in (group, [b] if remaining<16 or len(group)<16 else []):
                        if candidate and candidate not in groups:groups.append(candidate)
                baseline=model.objective(programs,cache=plan_cache);options=[]
                for group in groups[:12]:
                    if len(options)>=2 and time.perf_counter()-started>=budget_s:break
                    changed=list(programs)
                    for b in group:
                        for p,i in owners[b.block_id]:
                            changed=[replace(r,gpu=r.gpu-{i}) if r.name==p else r for r in changed]
                    loss=model.objective(changed,cache=plan_cache)-baseline;evaluations+=1
                    options.append((loss/len(group),min(b.block_id for b in group),group,changed))
                score,_,group,programs=min(options,key=lambda x:x[:2]);chosen.extend(group)
                decisions.append(dict(pages=len(group),next_round_marginal_s=score*len(group),
                    programs=sorted({p for b in group for p,_ in owners[b.block_id]})))
        else:
            chosen=[b for _,b in ranked[:count]]
        if len(chosen)!=count or len({b.block_id for b in chosen})!=count:
            raise RuntimeError('PCV changed native allocation demand')
        if mode in AUDIT_MODES:
            chosen_ids={b.block_id for b in chosen}
            withdraw={row['program']:[i for i,b in enumerate(row['blocks']) if b in chosen_ids]
                      for row in layouts}
            withdraw={p:indices for p,indices in withdraw.items() if indices}
            if withdraw:
                self._use_track_gpu_victims(withdraw,by_name)
                self._replica_call(dict(op='gapped_gpu_withdraw',programs=withdraw,
                                        withdraw_hicache=True))
                self.hicache_layout_signature=None
            return_ranked=sorted(enumerate(free),key=lambda pair:(sum(
                by_name[p].probability*(min(compute_at(i),
                    self.gapped_costs.load_token*16+self.gapped_costs.load_call/16)
                    if i//16 in by_name[p].cpu else compute_at(i))
                for p,i in owners.get(pair[1].block_id,())),pair[0]))
            disagreement=len(chosen_ids-{b.block_id for _,b in return_ranked[:count]})
            self._emit('pcv_use_gpu_decision',policy=mode,count=count,
                disagreement_pages=disagreement,gpu_estimator=self.use_gpu.summary(),
                cpu_estimator=self.use_cpu.summary(),
                candidates=[dict(rank=rank,block=b.block_id,owners=owners.get(b.block_id,()),
                    score=simple(b),selected=b.block_id in chosen_ids)
                    for rank,b in enumerate(free) if b.block_id in owners]
                    if os.environ.get('CONTINUUM_PCV_USE_DETAIL','0')=='1' else [],
                victims={p:[dict(page=i,cpu_present=i//16 in by_name[p].cpu)
                    for i in indices] for p,indices in withdraw.items() if p in by_name})
        for b in chosen:
            if b.ref_cnt or b.is_null:raise RuntimeError('PCV selected protected page')
            queue.remove(b)
        queue.prepend_n(chosen)
        self._emit('pcv_gpu_reclaim',policy=mode,pages=count,evaluations=evaluations,
            changed_pages=len({b.block_id for b in chosen}-{b.block_id for b in free[:count]}),
            controller_s=time.perf_counter()-started,
            budget_fallback_pages=budget_fallback,
            budget_s=float(os.environ.get('CONTINUUM_PCV_BUDGET_S','.040')),
            decisions=decisions,
            prediction='next-round fixed-order continuous-batch proxy; no future trace')
        return super()._layer_native_reuse(count)
