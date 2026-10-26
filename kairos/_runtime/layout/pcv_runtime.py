"""Kairos pcv runtime implementation."""
from dataclasses import replace
from collections import Counter
import math
import os
import time

from kairos._runtime.compat.lmcache_replica import identity
from kairos._runtime.layout.runtime import GappedRuntime
from kairos._runtime.layout.pcv import Program, Continuation, asym_frequency
from kairos._runtime.layout.next_use import USE_MODES, AUDIT_MODES, CPU_TIMED_MODES, SurvivalEstimator, InsuranceEstimator, UseInfo, cpu_value, retention_weight
from kairos._runtime.compat.layered_policy import CONTINUATION_LAYERS, LAYERS


class PCVRuntime(GappedRuntime):
    def __init__(self,*args,**kwargs):
        self.pcv_state={};self.pcv_config={}
        self.pcv_slopes={}
        self.use_cpu=SurvivalEstimator();self.use_gpu=SurvivalEstimator()
        self.use_state_time=0.
        self.pcv_state_times={}
        self.gpu_insurance=InsuranceEstimator();self.cpu_insurance=InsuranceEstimator()
        super().__init__(*args,**kwargs)

    def dispatch(self,message):
        if message['op']=='experiment':
            self.executable_plans = None
            self.pcvu_m2 = bool(message.get('pcvu_m2', False))
            self.pcvu_m3 = bool(message.get('pcvu_m3', False))
            self.pcvu_finalonly = bool(message.get('pcvu_finalonly', False))
            self.pcv_state={};self.pcv_slopes={}
            self.use_cpu=SurvivalEstimator();self.use_gpu=SurvivalEstimator()
            self.gpu_insurance=InsuranceEstimator();self.cpu_insurance=InsuranceEstimator()
            self.pcv_state_times={}
        # Lifecycle messages are already part of the mandatory replica
        # protocol.  PCV uses the optional piggybacked snapshot to keep the
        # eviction process aligned with the scheduler after a request enters
        # a tool gap; without this, the next CPU eviction can see a stale
        # running/ready state until the next GPU allocation publication.
        if message['op'] in ('history','hicache_register'):
            with self.registry.lock:
                lifecycle=message.get('pcv_lifecycle') or {}
                removed=message.get('pcv_remove_program')
                if removed:
                    self.pcv_state.pop(str(removed), None)
                    self.pcv_state_times.pop(str(removed),None)
                    self.gpu_insurance.pending={k:v for k,v in self.gpu_insurance.pending.items()
                                                if k[0] != str(removed)}
                    self.use_cpu.previous={k:v for k,v in self.use_cpu.previous.items()
                                           if v != str(removed)}
                    self.use_gpu.previous={k:v for k,v in self.use_gpu.previous.items()
                                           if v != str(removed)}
                elif lifecycle.get('row') is not None:
                    self.pcv_state[lifecycle['program']]=lifecycle['row']
                    self.pcv_state_times[lifecycle['program']]=time.monotonic()
                    self.record('pcv_cpu_lifecycle',program=lifecycle['program'],
                        phase=lifecycle['row']['phase'],
                        next_times=lifecycle['row'].get('cpu_next_times'),
                        continues=lifecycle['row'].get('cpu_continues'))
        if message['op']=='coverage' and 'pcv_continuation_program' in message:
            with self.registry.lock:
                response=super().dispatch(message)
                p=message['pcv_continuation_program']
                labels=self.gpu_insurance.resolve(p,set(message['pcv_continuation_gpu_chunks']))
                self.record('pcv_cpu_continuation_labels',program=p,labels=labels)
                return response
        if message['op'] in ('pcv_use_observe','pcv_use_gpu_victims'):
            with self.registry.lock:
                if message['op']=='pcv_use_observe':
                    p=message['program']
                    g=self.gpu_insurance.resolve(p,set(message['gpu_chunks']))
                    c=self.cpu_insurance.resolve(p,set(message['cpu_chunks']))
                    self.record('pcv_insurance_labels',program=p,gpu=g,cpu=c)
                else:
                    for r in message['rows']:self.cpu_insurance.track(r['program'],r['chunk'],r['delay'])
                return dict(pcv_cpu_insurance=self.cpu_insurance.summary(),
                            pcv_gpu_insurance=self.gpu_insurance.summary())
        if message['op']=='gapped_gpu_pages' and 'pcv_programs' in message:
            with self.registry.lock:
                self.pcv_state=message['pcv_programs']
                self.pcv_config=message['pcv_config']
                self.use_state_time=time.monotonic()
                self.pcv_state_times={p:self.use_state_time for p in self.pcv_state}
        if message['op'] in ('gapped_gpu_pages','gapped_gpu_withdraw'):
            with self.registry.lock:
                if message['op'] == 'gapped_gpu_withdraw':
                    self.registry.history.withdraw_gpu_pages(message['programs'])
                for r in message.get('pcv_use_probes',()):
                    self.cpu_insurance.track(r['program'],r['chunk'],r['delay'])
                if 'pcv_use_gpu' in message:
                    self.use_gpu=SurvivalEstimator.from_summary(message['pcv_use_gpu'])
                if 'pcv_slopes' in message:
                    self.pcv_slopes=message['pcv_slopes']
                if 'hicache_programs' in message:
                    super().dispatch(dict(op='hicache_gpu_layout', programs=message['hicache_programs']))
                if message.get('withdraw_hicache'):
                    super().dispatch(dict(op='hicache_gpu_withdraw', programs={p:
                        sorted({i//16 for i in pages}) for p,pages in message['programs'].items()}))
                if message.get('return_cpu'):
                    super().dispatch(message)
                    response=dict(cpu_programs=super().dispatch(dict(op='cpu_residency'))['programs'])
                    if self.gapped_policy in AUDIT_MODES:
                        objects={(p,i):p for p,row in response['cpu_programs'].items()
                                 for i in row['cpu_chunks']}
                        self.use_cpu.observe(objects,time.monotonic(),set(self.pcv_state))
                        response['pcv_use_cpu']=self.use_cpu.summary()
                        response.update(pcv_cpu_insurance=self.cpu_insurance.summary(),
                                        pcv_gpu_insurance=self.gpu_insurance.summary())
                    return response
        if message['op']=='pcv_state':
            with self.registry.lock:
                self.pcv_state=message['programs']
                self.pcv_config={k:message[k] for k in ('capacity_pages','token_budget',
                    'sequence_budget','decode_step','fixed_step',
                    'adaptive_decode_threshold','adaptive_token_budget')}
            return {}
        response = super().dispatch(message)
        if message['op'] == 'experiment' and message.get('executable_config'):
            from kairos._runtime.layout.executable_cpu import configure
            configure(self, message)
        return response

    def pressure_write_candidates(self,keys,layout_desc,mode):
        if getattr(self, 'executable_plans', None) is not None:
            from kairos._runtime.layout.executable_cpu import pressure_candidates
            return pressure_candidates(self, keys, layout_desc, mode)
        if not getattr(self,'pcvu_m3',False):
            return super().pressure_write_candidates(keys,layout_desc,mode)
        keys=list(dict.fromkeys(keys))
        from kairos._runtime.layout.pcvu_control import continuation_cpu_loss
        started=time.perf_counter()
        size=sum(math.prod(shape)*dtype.itemsize for shape,dtype in
                 zip(layout_desc.shapes,layout_desc.dtypes,strict=True))
        with self.manager._lock:rows=list(self.manager._objects.items())
        present={k for k,_ in rows}
        used,capacity=self.manager.get_memory_usage()
        free=max(0,(capacity-used)//size)
        absent=[k for k in keys if k not in present]
        if len(absent)<=free:return keys,[]
        known={k.base():(p,i) for p,manifest in self.hicache_manifests.items()
               for i,k in enumerate(manifest)}
        now=time.monotonic()
        def value(key):
            owner=known.get(identity(key).base())
            if not owner or owner[0] not in self.pcv_state:return None
            p,i=owner;r=self.pcv_state[p]
            if getattr(self,'pcvu_finalonly',False):
                if not r.get('cpu_continues',True):return 0.
                duplicate=all(j in self.gapped_gpu_pages.get(p,())
                              for j in range(i*16,(i+1)*16))
                return 0. if duplicate else r.get('probability',1.)*max(0.,
                    self.gapped_costs.chunk_compute(i)-self.gapped_costs.load_token*256-
                    self.gapped_costs.load_call)
            elapsed=max(0.,now-self.pcv_state_times.get(p,now))
            times=tuple(max(.001,t-elapsed) for t in (r.get('cpu_next_times') or r['returns']))
            protected=max(0.,r['protected_s']-elapsed) if all(
                j in r.get('protected_pages',()) for j in range(i*16,(i+1)*16)) else 0.
            return continuation_cpu_loss(self.gapped_costs,i,self.gapped_gpu_pages.get(p,()),
                protected,set(),times,r.get('cpu_continues',True))
        pending={k for group in self.repairs.values() for k in group}
        # Keep the concrete key with each score.  The previous implementation
        # retained only scores, so an accepted incoming chunk was passed to
        # the native writer while the scored victim was never evicted.  That
        # made the policy a write filter, not a replacement policy; the
        # subsequent native LRU silently made the actual victim choice.
        victims=[]
        for k,entry in rows:
            if (k in keys or identity(k) in pending or identity(k).base() in self.hicache_holds
                    or self.cpu_service_ledger.protected(identity(k).base(),now)
                    or entry.write_lock.is_locked() or entry.is_temporary
                    or not self.manager.is_key_evictable(k)):
                continue
            v=value(k)
            if v is not None:victims.append((v,k))
        victims.sort()
        kept=set(present);skipped=[];decisions=[]
        ranked=sorted(enumerate(absent),key=lambda pair:(-(value(pair[1]) or 0.),pair[0]))
        for _,k in ranked:
            v=value(k)
            if free:
                free-=1;kept.add(k)
            elif v is None:
                kept.add(k)  # unknown state is an explicit baseline fallback
            elif victims and v>victims[0][0]+1e-9:
                victim_value,victim_key=victims.pop(0)
                # Perform the scored replacement now, while the same
                # registry transaction still protects the native reservation.
                # A failed delete is treated as a refusal and leaves the
                # incoming write unadmitted; no force-delete is permitted.
                from lmcache.v1.distributed.error import L1Error
                deleted=self.manager.delete([victim_key], force=False).get(victim_key)
                if deleted == L1Error.SUCCESS:
                    registry=getattr(self, 'registry', None)
                    if registry is not None and hasattr(registry, 'record_deleted'):
                        registry.record_deleted([identity(victim_key)])
                    kept.add(k)
                    decisions.append(dict(incoming_value_s=v,
                        victim_value_s=victim_value,accepted=True,
                        victim_deleted=True,victim_key=identity(victim_key).base()))
                else:
                    victims.insert(0,(victim_value,victim_key))
                    skipped.append(k)
                    decisions.append(dict(incoming_value_s=v,
                        victim_value_s=victim_value,accepted=False,
                        victim_deleted=False,delete_status=str(deleted)))
            else:
                skipped.append(k)
                decisions.append(dict(incoming_value_s=v,accepted=False))
        self.record('pcvu_m3_admission',skipped_chunks=len(skipped),
            admitted_chunks=sum(k not in present and k in kept for k in keys),
            decisions=decisions,controller_s=time.perf_counter()-started,
            termination_hint_only=getattr(self,'pcvu_finalonly',False))
        return [k for k in keys if k in kept],skipped

    def hicache_make_room(self,rows,wanted,deficit,chunk_bytes,trigger):
        mode=self.gapped_policy
        if mode in CONTINUATION_LAYERS:
            return self._continuation_make_room(rows,wanted,deficit,chunk_bytes,trigger)
        if mode in ('gapped-fast','gapped-bundle','gapped-event'):
            from kairos._runtime.layout.recovery_runtime import recycle_cpu
            return recycle_cpu(self,rows,wanted,deficit,chunk_bytes,trigger)
        if mode not in ('gapped-pcv','gapped-pcv-cpu','gapped-pcv-gpu','gapped-return','gapped-return-clean',
                        'gapped-asym','gapped-asym-tier','gapped-cpu-continuation') + USE_MODES:
            return super().hicache_make_room(rows,wanted,deficit,chunk_bytes,trigger)
        if mode == 'gapped-asym':
            return super().hicache_make_room(rows,wanted,deficit,chunk_bytes,trigger)
        from lmcache.v1.distributed.error import L1Error
        start=time.perf_counter();plan_cache={}
        budget_s=float(os.environ.get('CONTINUUM_PCV_BUDGET_S','.040'))
        with self.eviction_policy._lock:
            ranks={k:i for i,k in enumerate(self.eviction_policy._order)}
        known={k.base():(p,i) for p,keys in self.hicache_manifests.items()
               for i,k in enumerate(keys)}
        cpu={p:set() for p in self.hicache_manifests}
        for k,entry in rows:
            owner=known.get(identity(k).base())
            if owner and not entry.write_lock.is_locked() and not entry.is_temporary:
                cpu[owner[0]].add(owner[1])
        programs=[Program(**state,gpu=frozenset(self.gapped_gpu_pages.get(p,())),
                    cpu=frozenset(cpu.get(p,()))) for p,state in self.pcv_state.items()]
        model=Continuation(self.gapped_costs,**self.pcv_config) if self.pcv_config else None
        pending={k for group in self.repairs.values() for k in group}
        eligible={k for k,_ in rows if k not in wanted and identity(k) not in wanted
            and identity(k) not in pending and identity(k).base() not in self.hicache_holds
            and self.manager.is_key_evictable(k)
            and not self.cpu_service_ledger.protected(identity(k).base(), time.monotonic())}
        audit=mode in AUDIT_MODES
        if audit:
            if not hasattr(self, 'use_cpu'): self.use_cpu=SurvivalEstimator()
            if not hasattr(self, 'use_gpu'): self.use_gpu=SurvivalEstimator()
            objects={(p,i):p for p,indices in cpu.items() for i in indices}
            self.use_cpu.observe(objects,time.monotonic(),set(self.pcv_state))
        disagreements=0;candidate_log=[]
        deleted=[];decisions=[];evaluations=0;budget_fallback=0
        # Timing and availability depend on the Program and transaction, not
        # on which of its equal-sized CPU chunks is scored. Cache these terms
        # once; page position supplies only the recovery loss below.
        continuation_factors={}
        if mode=='gapped-cpu-continuation':
            now=time.monotonic()
            for r in programs:
                elapsed=max(0.,now-self.pcv_state_times.get(r.name,self.use_state_time))
                times=tuple(max(.001,t-elapsed) for t in (r.cpu_next_times or r.returns))
                q=sum(1./(1.+t) for t in times)/len(times) if r.cpu_continues else 0.
                info=UseInfo(times,q,all_times=True)
                s,s_fallback=info.survival(self.gpu_insurance,True)
                protected=max(0.,r.protected_s-elapsed)
                sp,sp_fallback=info.survival(self.gpu_insurance,True,protected)
                continuation_factors[r.name]=(q,s,sp,times,s_fallback,sp_fallback)
        chunk_compute={i:self.gapped_costs.chunk_compute(i)
                       for _,i in known.values()} if mode=='gapped-cpu-continuation' else {}
        for _ in range(math.ceil(deficit/chunk_bytes)):
            if not eligible:break
            by_name={p.name:p for p in programs}
            def simple(k):
                owner=known.get(identity(k).base())
                if not owner or owner[0] not in by_name:return 0.,ranks.get(k,0)
                p,i=owner;r=by_name[p]
                # The continuation arm is itself the M3 implementation.  It
                # must reach the branch below, where the next-use insurance
                # guard is evaluated.  Treating it as the legacy generic M3
                # arm silently disabled the mechanism we intended to test.
                if (getattr(self, 'pcvu_m3', False)
                        and mode != 'gapped-cpu-continuation'
                        and not getattr(self,'pcvu_finalonly',False)):
                    from kairos._runtime.layout.pcvu_control import continuation_cpu_loss
                    elapsed=max(0.,time.monotonic()-self.pcv_state_times.get(p,self.use_state_time))
                    times=tuple(max(.001,t-elapsed) for t in (r.cpu_next_times or r.returns))
                    protected=max(0.,r.protected_s-elapsed) if all(
                        j in r.protected_pages for j in range(i*16,(i+1)*16)) else 0.
                    loss=continuation_cpu_loss(self.gapped_costs,i,r.gpu,protected,r.cpu,
                                               times,r.cpu_continues)
                    return loss,ranks.get(k,0)
                duplicate=all(j in r.gpu for j in range(i*16,(i+1)*16))
                loss=0. if duplicate else max(0.,self.gapped_costs.chunk_compute(i)-
                    self.gapped_costs.load_token*256-self.gapped_costs.load_call)
                weight=(asym_frequency(r.age,r.lifespan,
                    observed_lifespan=r.observed_lifespan) if mode=='gapped-asym-tier'
                    else r.probability)
                if getattr(self,'pcvu_finalonly',False) and not r.cpu_continues:
                    return 0.,ranks.get(k,0)
                if mode == 'gapped-cpu-continuation':
                    q,s,sp,_,s_fallback,sp_fallback=continuation_factors[p]
                    protected_now=all(j in r.protected_pages for j in range(i*16,(i+1)*16))
                    if protected_now:
                        s=sp;s_fallback=sp_fallback
                    if not duplicate:s=0.
                    potential=q*max(0.,chunk_compute[i]-self.gapped_costs.load_token*256-
                                    self.gapped_costs.load_call)
                    # A cold insurance estimator is uncertainty, not proof
                    # that a live GPU duplicate will survive until the next
                    # service opportunity.  Earlier versions treated the
                    # fallback s=1 as certainty; a zero-valued tie then
                    # deleted CPU insurance and produced later double misses.
                    # Keep an unknown duplicate behind a known non-duplicate
                    # victim.  Once the GPU is explicitly protected through
                    # the predicted use, the zero-loss duplicate path remains
                    # available.
                    unknown_duplicate=bool(duplicate and s_fallback and not protected_now)
                    value=potential if unknown_duplicate else (1.-s)*potential
                    # With no failure evidence, duplicate primary losses are
                    # all zero. Break that tie by the next-continuation loss
                    # at risk, without inventing a nonzero failure probability
                    # or displacing a positive-cost sole copy for an unknown
                    # insurance benefit.
                    return value,1 if unknown_duplicate else 0,potential,ranks.get(k,0)
                if mode in USE_MODES:
                    q=retention_weight(r,mode,tier='cpu')
                    times=r.returns if mode=='gapped-survival' else r.use_times
                    protected=(max(0.,r.protected_s-(time.monotonic()-self.use_state_time))
                        if all(j in r.protected_pages for j in range(i*16,(i+1)*16)) else 0.)
                    value,_,_=cpu_value(UseInfo(times,q,r.probability,r.protected_s,
                        all_times=mode.startswith('gapped-use-discount') or mode in CPU_TIMED_MODES),
                        self.gapped_costs.chunk_compute(i),
                        self.gapped_costs.load_token*256+self.gapped_costs.load_call,
                        duplicate,self.gpu_insurance,mode in ('gapped-survival','gapped-use-survival',
                            'gapped-use-discount-survival','gapped-cpu-insurance'),protected)
                    return value,ranks.get(k,0)
                return weight*loss,ranks.get(k,0)
            ranked=sorted(eligible,key=simple);heads=[];seen=set()
            for k in ranked:
                owner=known.get(identity(k).base())
                p=owner[0] if owner else None
                if p not in seen:heads.append(k);seen.add(p)
            candidates=list(dict.fromkeys(heads+ranked[:8]))[:12];value=0.
            if (mode in ('gapped-pcv','gapped-pcv-cpu') and model and programs
                    and time.perf_counter()-start<budget_s):
                base=model.objective(programs,cache=plan_cache);options=[]
                for k in candidates:
                    if len(options)>=2 and time.perf_counter()-start>=budget_s:break
                    owner=known.get(identity(k).base())
                    changed=[replace(p,cpu=p.cpu-{owner[1]})
                        if owner and p.name==owner[0] else p for p in programs]
                    loss=model.objective(changed,cache=plan_cache)-base;evaluations+=1
                    options.append((loss,ranks.get(k,0),k,changed))
                value,_,key,changed=min(options,key=lambda x:x[:2])
            else:
                if mode in ('gapped-pcv','gapped-pcv-cpu'):budget_fallback+=1
                key=ranked[0];value=simple(key)[0]
                owner=known.get(identity(key).base())
                changed=[replace(p,cpu=p.cpu-{owner[1]})
                    if owner and p.name==owner[0] else p for p in programs]
            if audit:
                def return_score(k):
                    owner=known.get(identity(k).base())
                    if not owner or owner[0] not in by_name:return 0.,ranks.get(k,0)
                    p,i=owner;r=by_name[p]
                    duplicate=all(j in r.gpu for j in range(i*16,(i+1)*16))
                    return (r.probability*(0. if duplicate else max(0.,
                        self.gapped_costs.chunk_compute(i)-self.gapped_costs.load_token*256-
                        self.gapped_costs.load_call)),ranks.get(k,0))
                disagreements+=key!=min(eligible,key=return_score)
                if os.environ.get('CONTINUUM_PCV_USE_DETAIL','0')=='1':candidate_log.append([dict(program=known.get(identity(k).base(),(None,None))[0],
                    chunk=known.get(identity(k).base(),(None,None))[1],
                    return_score=return_score(k)[0],score=simple(k)[0],selected=k==key)
                    for k in ranked])
            eligible.remove(key)
            if self.manager.delete([key],force=False).get(key)==L1Error.SUCCESS:
                programs=changed;deleted.append(key)
                owner=known.get(identity(key).base())
                if owner:cpu[owner[0]].discard(owner[1])
                decisions.append(dict(program=owner[0] if owner else None,
                    chunk=owner[1] if owner else None,next_round_marginal_s=value,
                    gpu_pages=sorted(j for j in range(owner[1]*16,(owner[1]+1)*16)
                        if j in by_name[owner[0]].gpu) if owner and owner[0] in by_name else []))
                if mode=='gapped-cpu-continuation' and owner and owner[0] in by_name:
                    r=by_name[owner[0]];i=owner[1]
                    elapsed=max(0.,time.monotonic()-self.pcv_state_times.get(r.name,self.use_state_time))
                    times=tuple(max(.001,t-elapsed) for t in (r.cpu_next_times or r.returns))
                    duplicate=all(j in r.gpu for j in range(i*16,(i+1)*16))
                    if duplicate and r.cpu_continues:
                        self.gpu_insurance.track(r.name,i,sum(times)/len(times))
                    decisions[-1].update(phase=r.phase,cpu_next_times=times,
                        cpu_continues=r.cpu_continues,backup_availability=
                        self.gpu_insurance.probability(sum(times)/len(times))[0],
                        backup_availability_fallback=self.gpu_insurance.probability(
                            sum(times)/len(times))[1],
                        # This flag distinguishes an evidence-backed zero
                        # loss from a conservative guard used while the
                        # survival estimator is still sparse.
                        unknown_duplicate=bool(
                            duplicate and self.gpu_insurance.probability(
                                sum(times)/len(times))[1]),
                        uncertainty_guard=bool(
                            duplicate and self.gpu_insurance.probability(
                                sum(times)/len(times))[1]))
                if audit and owner:self.use_cpu.remove([owner],time.monotonic())
                if audit and owner and owner[0] in by_name:
                    r=by_name[owner[0]];i=owner[1]
                    if all(j in r.gpu for j in range(i*16,(i+1)*16)):
                        if not hasattr(self,'gpu_insurance'):self.gpu_insurance=InsuranceEstimator()
                        times=r.returns if mode=='gapped-survival' else r.use_times
                        protected=(r.protected_s if all(j in r.protected_pages for j in range(i*16,(i+1)*16)) else 0.)
                        self.gpu_insurance.track(r.name,i,sum(times)/len(times),protected)
        if audit:
            self.record('pcv_use_cpu_decision',policy=mode,disagreements=disagreements,
                candidates=candidate_log,gpu_estimator=self.use_gpu.summary(),
                cpu_estimator=self.use_cpu.summary())
        self.registry.record_deleted([identity(k) for k in deleted])
        self.record('pcv_cpu_reclaim',policy=mode,trigger=trigger,deficit_bytes=deficit,
            deleted_bytes=len(deleted)*chunk_bytes,decisions=decisions,
            evaluations=evaluations,controller_s=time.perf_counter()-start,
            budget_fallback_chunks=budget_fallback,budget_s=budget_s,
            pcvu_m3=getattr(self,'pcvu_m3',False),pcvu_m2=getattr(self,'pcvu_m2',False))
        return deleted

    def _continuation_make_room(self, rows, wanted, deficit, chunk_bytes, trigger):
        """Bounded whole-chunk next-continuation loss, on the EA foundation.

        Unlike the old controller, equal-size independent chunks are sorted
        once per pressure event. No scheduler rollout, extra DMA or hard pin.
        This controls victims only; the common foundation budgets new writes.
        Sparse availability is an explicit conservative cost bound, not an
        estimated failure probability. Native deletion may always refuse.
        """
        from lmcache.v1.distributed.error import L1Error
        from kairos._runtime.layout.plan import runs
        started=time.perf_counter();now=time.monotonic()
        with self.eviction_policy._lock:
            ranks={key:i for i,key in enumerate(self.eviction_policy._order)}
        owners={key.base():(p,i) for p,keys in self.hicache_manifests.items()
                for i,key in enumerate(keys)}
        # Keep the key-identity join observable.  CPU objects are LMCache
        # keys while manifests are control-plane ReplicaKeys; a silent join
        # miss turns the continuation policy into an unlabelled LRU fallback.
        # This counter is diagnostic only and does not affect victim choice.
        row_bases={identity(key).base() for key, _ in rows}
        matched_bases=row_bases.intersection(owners)
        self.record('pcv_continuation_join', rows=len(row_bases),
                    manifest_keys=len(owners), matched=len(matched_bases),
                    programs=len(self.hicache_manifests),
                    state_programs=len(self.pcv_state),
                    active_matched=sum(owners[b][0] in self.pcv_state for b in matched_bases),
                    sample_row=next(iter(row_bases), None),
                    sample_manifest=next(iter(owners), None))
        # A missing join means this runtime cannot attribute an object to a
        # Program.  The continuation score has no evidence in that case;
        # preserve the strong host policy instead of interpreting an
        # unlabelled object as zero-loss and evicting it first.
        if not matched_bases:
            return super().hicache_make_room(rows, wanted, deficit, chunk_bytes, trigger)
        pending={identity(k).base() for group in self.repairs.values() for k in group}
        wanted_bases={identity(k).base() for k in wanted}
        factors={};scores=[];eligible_counts=Counter();blocked_counts=Counter()
        layer=LAYERS[self.gapped_policy]
        for key,entry in rows:
            base=identity(key).base()
            owner=owners.get(base)
            attribution=('active' if owner and owner[0] in self.pcv_state else
                         'inactive-owner' if owner else 'unknown-object')
            reason=('incoming-store' if base in wanted_bases else
                    'pending-transfer' if base in pending else
                    'held' if base in self.hicache_holds else
                    'write-locked' if entry.write_lock.is_locked() else
                    'temporary' if entry.is_temporary else
                    'service-lease' if self.cpu_service_ledger.protected(base,now) else
                    'native-non-evictable' if not self.manager.is_key_evictable(key) else None)
            if reason:
                blocked_counts[attribution+':'+reason]+=1
                continue
            eligible_counts[attribution]+=1
            detail=dict(program=owner[0] if owner else None,
                chunk=owner[1] if owner else None,gpu_pages=[],
                attribution=attribution)
            value=0.
            if owner and owner[0] in self.pcv_state:
                p,i=owner;r=self.pcv_state[p]
                if p not in factors:
                    elapsed=max(0.,now-self.pcv_state_times.get(p,now))
                    times=tuple(max(.001,t-elapsed) for t in
                                (r.get('cpu_next_times') or r['returns']))
                    q=(1. if layer.host=='continuation-current' and
                        layer.recovery_weight=='uniform' else
                        sum(1./(1.+t) for t in times)/len(times))
                    protected=max(0.,r.get('protected_s',0.)-elapsed)
                    if layer.host=='continuation-current':
                        availability,fallback=None,False
                    else:
                        availability,fallback=UseInfo(times,q,all_times=True).survival(
                            self.gpu_insurance,True)
                    factors[p]=(times,q,protected,availability,fallback)
                times,q,protected,availability,fallback=factors[p]
                pages=set(range(i*16,(i+1)*16))
                gpu=pages & self.gapped_gpu_pages.get(p,set())
                missing=pages-gpu
                load=self.gapped_costs.load_token*256+self.gapped_costs.load_call
                immediate=self.gapped_costs.compute(
                    [(a*16,b*16) for a,b in runs(missing)])
                insurance=max(0.,self.gapped_costs.chunk_compute(i)-load)
                safe=pages<=set(r.get('protected_pages',()))
                guaranteed=sum(t<=protected for t in times)/len(times) if safe else 0.
                if layer.host == 'continuation-current':
                    # Intuitive strong control: the SAME concrete chunk
                    # recovery margin, optionally times next-use discount,
                    # without predicted GPU survival or backup insurance.
                    value=q*max(0.,min(immediate,self.gapped_costs.chunk_compute(i))-load)
                    guard=False
                elif layer.host == 'continuation-repair':
                    # A partial GPU copy does not remove the future risk of
                    # losing its remaining pages. Evaluate both recovery
                    # layouts, using the SAME availability approximation as
                    # complete duplicates. Sparse data remains a cost bound.
                    guaranteed=(sum(t<=protected for t in times)/len(times)
                        if gpu and gpu <= set(r.get('protected_pages',())) else 0.)
                    guard=bool(gpu and fallback and guaranteed<1.)
                    survives=guaranteed if fallback else max(guaranteed,availability)
                    current_loss=max(0.,min(immediate,self.gapped_costs.chunk_compute(i))-load)
                    value=q*(survives*current_loss+(1.-survives)*insurance)
                elif missing:
                    value=q*max(0.,immediate-load)
                    guard=False
                else:
                    guard=bool(fallback and guaranteed<1.)
                    survives=guaranteed if fallback else max(guaranteed,availability)
                    value=q*(1.-survives)*insurance
                detail=dict(program=p,chunk=i,gpu_pages=sorted(gpu),
                    attribution='active',
                    cpu_next_times=times,phase=r['phase'],uncertainty_guard=guard,
                    backup_availability_fallback=fallback,guaranteed_coverage=guaranteed,
                    backup_availability=availability,
                    next_round_marginal_s=value)
            scores.append((value,ranks.get(key,0),key,detail))
        scores.sort(key=lambda item:item[:2])
        deleted=[];decisions=[]
        for value,_,key,detail in scores:
            if len(deleted)*chunk_bytes>=deficit:break
            if self.manager.delete([key],force=False).get(key)!=L1Error.SUCCESS:continue
            deleted.append(key);decisions.append(detail)
            if (layer.host!='continuation-current' and
                    len(detail['gpu_pages'])==16 and detail.get('program')):
                self.gpu_insurance.track(detail['program'],detail['chunk'],
                    sum(detail['cpu_next_times'])/len(detail['cpu_next_times']))
        self.registry.record_deleted([identity(k) for k in deleted])
        self.record('pcv_cpu_reclaim',policy=self.gapped_policy,trigger=trigger,
            deficit_bytes=deficit,deleted_bytes=len(deleted)*chunk_bytes,
            decisions=decisions,evaluations=len(scores),
            controller_s=time.perf_counter()-started,
            action='next-continuation-cpu-victims',prediction='online-no-final-hint',
            gpu_survival_model=layer.host!='continuation-current',
            continuation_weight=layer.recovery_weight if layer.host=='continuation-current' else 'time',
            eligible_chunks=dict(eligible_counts),blocked_chunks=dict(blocked_counts),
            pcvu_m3=False,pcvu_m2=False,budget_fallback_chunks=0)
        return deleted
