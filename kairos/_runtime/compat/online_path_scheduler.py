"""Kairos online path scheduler implementation."""
import hashlib
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from vllm.v1.core.sched.request_queue import create_request_queue
from kairos._runtime.compat.holding_scheduler import HoldingEconomicsScheduler
from kairos._runtime.compat.online_path import PathForecast, choose_continuation, known_request_output
from kairos._runtime.compat.continuation_demand import OracleDemand, next_envelope, empirical_gittins
from kairos._runtime.compat.rank_contract import selected_policy, phase_policy, rank_score, ScalarTruth, StageWorkPrior
from kairos._runtime.compat.component_ablation import native_lru_payload

CALIBRATED_MODES={'onlinefitcalibrated','onlinefitcalgrowth'}
MEAN_MODES={'onlinefitmean','onlinefitmeangrowth','onlinefitgittins'}|CALIBRATED_MODES
STATE_MODES={'onlinefitstate','onlinefitstatejoint','onlinefitstatetruth'}
WORK_MODES={'onlinework','onlineworktruth','onlineunified','onlineunifiedtruth'}
RESIDUAL_MODES={'onlinefitresidual','onlinefittruth','onlinefittruthgrowth','onlinefitgrowth'}|MEAN_MODES|STATE_MODES|WORK_MODES
GROWTH_MODES={'onlinefittruthgrowth','onlinefitgrowth','onlinefitmeangrowth','onlinefitcalgrowth'}
TRUTH_MODES={'onlinefittruth','onlinefittruthgrowth','onlinefitstatetruth'}
FIT_MODES={'onlinefit','onlinefitnative','onlinefitrank','onlinefitfresh'}|RESIDUAL_MODES

# A single runtime registry prevents a policy tag from reaching suffix
# observation/completion paths without loading the matching predictor.
SUFFIX_CDF_POLICIES=frozenset({
    'cdfmean','cdfindex','cdfwarm','progressive','progressivedebt',
    'progressivewindow',
})
PROGRESSIVE_POLICIES=frozenset({'progressive','progressivedebt','progressivewindow'})


def _current_turn_view(program):
    """Return a one-turn view for service pricing.

    Production paths use the frozen Program dataclass. Lightweight scheduler
    contract fixtures intentionally use SimpleNamespace; keeping this adapter
    here makes tests exercise the same pricing semantics without requiring a
    future-shaped object to be a dataclass.
    """
    try:
        return replace(program, turns=program.turns[:1])
    except TypeError:
        view=SimpleNamespace(**vars(program))
        view.turns=program.turns[:1]
        return view


class FinitePathProtection:
    def __init__(self, original, forecast, live=False, zero=False):
        self.original=original; self.forecast=forecast
        self.live=live; self.observed={}
        self.zero=zero
        for key in ('history','mode','k','fixed'): setattr(self,key,getattr(original,key))
    def observe(self,tool,duration,added_tokens):
        if self.live:
            if not math.isfinite(duration) or duration<0:
                raise ValueError('Live tool support needs a finite completed return interval')
            samples=self.observed.setdefault(tool,[])
            samples.append(duration)
            del samples[:-512]
        return self.original.observe(tool,duration,added_tokens)
    def conditional(self,*a,**kw): return self.original.conditional(*a,**kw)
    def ttl(self,tool,benefit,oracle_gap=None,**kwargs):
        if self.zero:return 0.,'ablation/no-optional-tool-lease'
        if getattr(self,'support_max',False):
            return self.forecast.idle_bound(tool),'ablation/training-tool-support-maximum'
        if getattr(self.original, 'mode', '') == 'r5-conditional-hold':
            return self.original.ttl(tool, benefit, oracle_gap, **kwargs)
        # Only the completed call's tool type is consulted here.
        bound=self.forecast.idle_bound(tool)
        if self.live:
            bound=max(bound,max(self.observed.get(tool,()),default=0.))
            return bound,'online-path/causal-return-support'
        return bound,'online-path/training-tool-support'


class OnlinePathScheduler(HoldingEconomicsScheduler):
    def _replica_call(self,payload):
        if 'tierlru' in (self.experiment or '').split('~'):
            payload=native_lru_payload(payload)
        return super()._replica_call(payload)

    def __init__(self,*args,**kwargs):
        self.online_mode='off'
        data=Path(os.environ['CONTINUUM_ONLINE_PATH_PRIOR']).read_bytes()
        self.online_prior=json.loads(data); self.online_sha=hashlib.sha256(data).hexdigest()
        if os.environ.get('CONTINUUM_R5') == '1':
            from kairos.protection import CommitmentPrior
            self.forecaster=CommitmentPrior(self.online_prior)
        else:
            self.forecaster=PathForecast(self.online_prior)
        if not self.forecaster.protection_support:
            raise ValueError('A work-only prior cannot supply r12 protection leases; verified tool timings required')
        self.online_evaluator=None
        # The compact learned arm is deliberately lazy.  The ordinary r12
        # controller and all its unit-test fixtures must not depend on a
        # tokenizer, predictor artifact, or online-only bookkeeping.
        self.compact_predictor=None
        self.learned_tokenizer=None
        self.learned_service_total={}
        self.learned_predictions={}
        if os.environ.get('CONTINUUM_ONLINE_PATH_NATIVE')=='1':
            from kairos._runtime.compat.online_path_native import warmup, simulate
            warmup();self.online_evaluator=simulate
        super().__init__(*args,**kwargs)

    def _switch_experiment(self,name):
        if name==self.experiment:return
        self.online_mode='off'; super()._switch_experiment(name)
        tags=set(name.split('~')[1:]); modes=tags&({'onlinepeak','onlinepath'}|FIT_MODES)
        if 'tierlru' in tags:
            self.gpu_victim_mode=None;self.publish_gpu_reuse=False
            self.connector.range_lookup=True
            self._emit('tier_ablation_registry',gpu='native LRU among legal free pages',
                cpu='native LRU, same per-STORE deficit and partial reservation',
                cpu_range_lookup=True,
                changed='remove v21 complementary cross-tier layout choices',
                native_reference_and_dma_fences=True)
        # Explicit version tag: frozen r12 comparisons retain their original
        # prediction semantics. Future exact-length replays can omit an
        # unnecessary current-output prediction without treating max_tokens
        # as a known EOS length for ordinary online requests.
        self.online_known_output = 'knownoutput' in tags
        cold_tags=tags & {'coldadmissiontruth','coldadmissionlinear','coldadmissionprompt'}
        if len(cold_tags)>1:raise ValueError('Choose one cold admission intervention')
        self.cold_admission_policy=next(iter(cold_tags),None)
        if self.cold_admission_policy and modes!={'onlinefitresidual'}:
            raise ValueError('Cold admission requires the identical residual controller')
        self.scalar_rank_policy = selected_policy(tags)
        # Progressive service is a scheduler contract, not merely another
        # scalar rank. It uses the same causal predictor, but the grant loop
        # limits each round and refreshes the posterior at call boundaries.
        self.adaptive_compete = self.scalar_rank_policy == 'adaptive'
        self.progressive_compete = self.scalar_rank_policy in PROGRESSIVE_POLICIES
        # r1 recorded attained-service debt but native vLLM rebuilt its queue
        # from posterior ranks. r2 publishes the debt order as queue priority.
        self.progressive_debt_compete = self.scalar_rank_policy == 'progressivedebt'
        self.progressive_window_compete = self.scalar_rank_policy == 'progressivewindow'
        if self.cold_admission_policy and self.scalar_rank_policy not in (None,'predicted'):
            raise ValueError('Cold admission must retain incumbent global ranking')
        self.scalar_rank_truth = None
        self.scalar_stage_prior = None
        self.scalar_verified_prior = None
        if self.scalar_rank_policy:
            if modes != {'onlinefitresidual'}:
                raise ValueError('Scalar rank ablation requires the identical r12 residual controller')
            if self.scalar_rank_policy in ('truth','truthcold','truthwarm'):
                truth_bytes = Path(os.environ['CONTINUUM_SCALAR_RANK_TRUTH']).read_bytes()
                self.scalar_rank_truth = ScalarTruth(json.loads(truth_bytes),
                    os.environ['CONTINUUM_SCALAR_TRACE_SHA'])
            stage_sha=None
            verified_sha=None
            if self.scalar_rank_policy == 'verified':
                verified_bytes=Path(os.environ['CONTINUUM_SCALAR_VERIFIED_PRIOR']).read_bytes()
                self.scalar_verified_prior=json.loads(verified_bytes)
                if (self.scalar_verified_prior.get('schema_version')!=2 or
                    set(self.scalar_verified_prior['heldout_ids']) & set(self.online_prior['evaluation_ids'])):
                    raise ValueError('Verified prior must audit independent complete endpoints')
                verified_sha=hashlib.sha256(verified_bytes).hexdigest()
            if self.scalar_rank_policy == 'stage':
                stage_bytes=Path(os.environ['CONTINUUM_SCALAR_STAGE_PRIOR']).read_bytes()
                stage_prior=json.loads(stage_bytes)
                if set(stage_prior['heldout_ids']) & set(self.online_prior['evaluation_ids']):
                    raise ValueError('Stage prior includes development identities')
                self.scalar_stage_prior=StageWorkPrior(stage_prior)
                stage_sha=hashlib.sha256(stage_bytes).hexdigest()
            self._emit('scalar_rank_registry', policy=self.scalar_rank_policy,
                future_information=self.scalar_rank_policy in ('truth','truthcold','truthwarm'),
                deployable=self.scalar_rank_policy not in ('truth','truthcold','truthwarm'),
                same_control=True, same_lease='leasezero' not in tags,
                same_placement='tierlru' not in tags,same_current_pricing=True,
                truth_sha256=(hashlib.sha256(truth_bytes).hexdigest()
                    if self.scalar_rank_policy in ('truth','truthcold','truthwarm') else None),
                stage_prior_sha256=stage_sha,
                verified_prior_sha256=verified_sha,
                interventions=('progressive service rounds: current-call probe for cold states, '
                    'posterior residual after each logical completion; native waiting priority'
                    if self.scalar_rank_policy == 'adaptive' else
                    'PIS-KV-r4: cold-safe return-first rounds; bounded cold window; '
                    'suffix residual only after one observed completion'
                    if self.scalar_rank_policy == 'progressivewindow' else
                    'PIS-KV progressive service rounds: current-call cold probe, '
                    'posterior residual after each logical completion'
                    if self.scalar_rank_policy in PROGRESSIVE_POLICIES else
                    'rank only: fresh and return order, optional idle victim order, native waiting priority'),
                oracle_phase=('zero completed calls' if self.scalar_rank_policy=='truthcold' else
                    'at least one completed call' if self.scalar_rank_policy=='truthwarm' else
                    'all' if self.scalar_rank_policy=='truth' else 'none'),
                idle_current_work='zero: no currently arrived LLM call')
        self.online_grant_allocation = 'grantalloc' in tags
        self.online_live_lease = 'livelease' in tags
        self.online_zero_lease = 'leasezero' in tags
        if self.online_zero_lease and self.online_live_lease:
            raise ValueError('Zero-lease ablation cannot also enable live leases')
        if len(modes)>1: raise ValueError('Conflicting online controls')
        self.online_mode=next(iter(modes),'off')
        self.online_inputs={};self.online_outputs={};self.online_requests={}
        self.online_reserved={};self.online_granted=set();self.online_rank={}
        # Only R5 rank-separation experiments publish a distinct victim map.
        # None preserves the exact native/R4 online_rank path.
        self.online_victim_rank=None
        self.online_active=set();self.online_dirty=True;self.online_ended=set()
        # Lifetime service identity and current resident memory are distinct.
        # Expiring a cache lease never turns a continuation into a fresh job.
        self.online_entered=set()
        self.online_cleaning={}
        self.online_next={};self.online_insurance={}
        self.online_service_cache={}
        self.cold_admission_scores={}
        self.cold_admission_truth=None
        self.cold_admission_ranker=None
        if self.cold_admission_policy=='coldadmissiontruth':
            self.cold_admission_truth=ScalarTruth(json.loads(
                Path(os.environ['CONTINUUM_SCALAR_RANK_TRUTH']).read_bytes()),
                os.environ['CONTINUUM_SCALAR_TRACE_SHA'])
        if self.cold_admission_policy=='coldadmissionlinear':
            from kairos._runtime.compat.cold_admission import ColdAdmissionRanker
            from transformers import AutoTokenizer
            self.cold_admission_ranker=ColdAdmissionRanker(
                os.environ['CONTINUUM_COLD_ADMISSION_MODEL'],self.online_prior['evaluation_ids'])
            self.cold_admission_tokenizer=AutoTokenizer.from_pretrained(
                os.environ['CONTINUUM_MODEL'],local_files_only=True)
        if self.cold_admission_policy:
            self._emit('cold_admission_registry',policy=self.cold_admission_policy,
                future_information=self.cold_admission_truth is not None,
                deployable=self.cold_admission_truth is None,
                model_sha256=(self.cold_admission_ranker.sha256 if self.cold_admission_ranker else None),
                authority='first admission candidate order only',
                global_rank_lease_victim_capacity_native_priority='incumbent unchanged',
                output_contract='ordinal-only-lower-first')
        self.learned_service_total={}
        self.learned_predictions={};self.learned_previous_hashes={};self.learned_current_keep={}
        self.suffix_predictor=None;self.suffix_tokenizer=None;self.suffix_predictions={}
        self.suffix_service_total={};self.suffix_previous_hashes={};self.suffix_current_keep={}
        self.progressive_observed_s={}
        if self.online_mode=='off':return
        if getattr(self,'scalar_rank_policy',None) in ('learned','learnedgated'):
            path=os.environ.get('CONTINUUM_COMPACT_PREDICTOR')
            if not path:
                raise ValueError('Learned rank requires CONTINUUM_COMPACT_PREDICTOR')
            from kairos._runtime.unsupported import CompactPredictor
            from transformers import AutoTokenizer
            self.compact_predictor=CompactPredictor(path)
            audit_path=os.environ.get('CONTINUUM_COMPACT_PREDICTOR_AUDIT')
            if not audit_path:
                raise ValueError('Learned rank requires predictor training-identity audit')
            self.compact_predictor_audit=json.loads(Path(audit_path).read_bytes())
            if (self.compact_predictor_audit.get('selected')!=self.compact_predictor.model.get('config') or
                set(self.compact_predictor_audit.get('train_ids',())) &
                    set(self.online_prior.get('evaluation_ids',()))):
                raise ValueError('Predictor artifact/config or training/evaluation identity mismatch')
            model=os.environ['CONTINUUM_MODEL']
            self.learned_tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
            self._emit('compact_predictor_registry',
                model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                audit_sha256=hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
                training_programs=len(self.compact_predictor_audit['train_ids']),
                training_evaluation_disjoint=True,
                target='post-current residual standard service seconds',
                features='causal visible messages, prompt/output history, completed service history',
                future_information=False,capacity_control_unchanged=True,
                fallback='PathForecast when visible message metadata is unavailable')
        if getattr(self,'scalar_rank_policy',None) in SUFFIX_CDF_POLICIES:
            path=os.environ.get('CONTINUUM_SUFFIX_CDF')
            if not path:
                raise ValueError('Conditional CDF rank requires CONTINUUM_SUFFIX_CDF')
            from kairos._runtime.unsupported import SuffixPredictor
            from transformers import AutoTokenizer
            self.suffix_predictor=SuffixPredictor(path)
            self.suffix_tokenizer=AutoTokenizer.from_pretrained(
                os.environ['CONTINUUM_MODEL'],local_files_only=True)
            if self.suffix_predictor is None or self.suffix_tokenizer is None:
                raise RuntimeError('Suffix CDF policy initialized without predictor/tokenizer')
            if self.suffix_predictor.training_ids & set(self.online_prior.get('evaluation_ids',())):
                raise ValueError('Suffix CDF predictor includes evaluation identities')
            self._emit('suffix_cdf_registry',policy=self.scalar_rank_policy,
                model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                training_programs=len(self.suffix_predictor.training_ids),
                training_evaluation_disjoint=True,
                target='conditional distribution of post-current stored suffix solo service',
                features='causal prefix: accumulated prompt/output and visible tool progress',
                fixed_support=len(self.suffix_predictor.model['atoms_s']),
                controller='rank only; Continuum lease, v21 tier policy, capacity and native fences unchanged')
        self.online_oracle_rank={}
        self.online_truth=None;truth_sha=None
        if self.online_mode in TRUTH_MODES:
            truth_data=Path(os.environ['CONTINUUM_ONLINE_PATH_TRUTH']).read_bytes()
            self.online_truth=OracleDemand(json.loads(truth_data))
            truth_sha=hashlib.sha256(truth_data).hexdigest()
        rank_sha=None
        if self.online_mode=='onlinefitrank':
            rank_data=Path(os.environ['CONTINUUM_ONLINE_PATH_ORACLE_RANK']).read_bytes()
            rank_plan=json.loads(rank_data)
            if not rank_plan.get('future_information') or rank_plan.get('deployable') is not False:
                raise ValueError('Oracle rank must explicitly disclose future information')
            order=rank_plan['order']
            if len(set(order))!=len(order):raise ValueError('Duplicate oracle rank identity')
            self.online_oracle_rank={p:i for i,p in enumerate(order)}
            rank_sha=hashlib.sha256(rank_data).hexdigest()
        calibration_sha=None
        r5 = 'r5callcommitment' in tags or os.environ.get('CONTINUUM_R5') == '1'
        if self.online_mode in MEAN_MODES:
            neighbors=8
            if self.online_mode in CALIBRATED_MODES:
                raw=Path(os.environ['CONTINUUM_ONLINE_DEMAND_CALIBRATION']).read_bytes()
                calibration=json.loads(raw)
                if (calibration.get('evaluation_outcomes_used') is not False or
                    calibration.get('future_information') is not False or
                    calibration['prior_sha256']!=self.online_sha or
                    calibration['training_ids']!=self.online_prior['heldout_ids']):
                    raise ValueError('Demand calibration must use the same disjoint training Programs only')
                neighbors=calibration['neighbors'];calibration_sha=hashlib.sha256(raw).hexdigest()
            if not r5:self.forecaster=PathForecast(self.online_prior,neighbors=neighbors,fast=True)
        elif not r5:self.forecaster=PathForecast(self.online_prior)
        if self.scalar_verified_prior is not None:
            self.forecaster=PathForecast(self.scalar_verified_prior)
            old=PathForecast(self.online_prior)
            if any(old.idle_bound(k)!=self.forecaster.idle_bound(k) for k in
                   set(old.tool_samples)|set(self.forecaster.tool_samples)|{'unknown-tool-fallback'}):
                raise ValueError('Verified work correction changed protection support')
        if name.split('~')[0]!='v21-eager-a' or not self.logical_policy_completion:
            raise ValueError('Online path requires common v21 logical-finish pipeline')
        self.policy_model=FinitePathProtection(self.policy_model,self.forecaster,
                                                live=self.online_live_lease,zero=self.online_zero_lease)
        self._emit('online_path_registry',mode=self.online_mode,prior_sha256=self.online_sha,
            future_information=bool(self.online_oracle_rank) or self.online_mode in TRUTH_MODES,
            deployable=not (bool(self.online_oracle_rank) or self.online_mode in TRUTH_MODES),
            truth_sha256=truth_sha,
            oracle_rank_sha256=rank_sha,arrived_identities_only=True,neighbors=self.forecaster.neighbors,
            demand_calibration_sha256=calibration_sha,
            objective='predicted complete residual Program sum JCT',
            execution='event-only plan; actual full-call memory and native DMA fences',
            predictor=('equal-weight neighboring complete paths, conditional on observed history'
                       if self.online_mode in MEAN_MODES else
                       'arrived-only diagnostic true suffix' if self.online_mode in TRUTH_MODES else
                       'conditional suffix CDF with cold probe/warm residual separation'
                       if self.scalar_rank_policy in ('progressive','progressivedebt','progressivewindow') else
                       'median disjoint complete path conditional on observed input growth and completed outputs'),
            rank_rule=('progressive service frontier plus bounded return competition'
                       if self.scalar_rank_policy == 'adaptive' else
                       'PIS-KV-r4: returned continuity first, cold-safe bounded probe window, '
                       'warm suffix posterior order'
                       if self.scalar_rank_policy == 'progressivewindow' else
                       'PIS-KV: returned continuity first, attained-service debt among returns, '
                       'cold probe only when no return is fundable'
                       if self.scalar_rank_policy in ('progressive','progressivedebt') else
                       'empirical Gittins service-budget index' if self.online_mode=='onlinefitgittins' else
                       'conditional mean residual service' if self.online_mode in MEAN_MODES else
                       'single-path residual service' if self.online_mode in RESIDUAL_MODES else 'legacy path rank'),
            fast_forecast=self.online_mode in MEAN_MODES,
            public_output_contract=self.online_known_output,
            grant_allocation=self.online_grant_allocation,
            protection_support=('zero optional protection; running/granted/DMA safety unchanged'
                if self.online_zero_lease else 'independent training plus completed live return intervals; last 512 per tool'
                if self.online_live_lease else 'independent training only'),
            controls='onlinepeak: predicted whole-path peak; onlinepath: stage admission with complete-continuation check; '
                     'onlinefit: same rank/leases/returns, fresh entry checks actual current capacity only; '
                     'onlinefitnative: same capacity ledger, execution slots enforced by native running limit only; '
                     'onlinefitrank: onlinefit execution with previously optimized future-informed fixed Program rank only; '
                     'onlinefitfresh: onlinefit with rank refreshed before every event decision, including returns without fresh candidates; '
                     'onlinefitresidual: fresh ranks deduct actual computed KV and generated tokens from current residual service',
            safety='idle finite lease; resident slots separate from lifetime service identity; native refs unchanged',
            service_model=self.profile.get('pipeline_service',{}),
            evaluator='native-reference-equivalent' if self.online_evaluator else 'Python reference',
            growth_arrangement='one-next-call logical funding before fresh admissions; optional refs only on mismatch'
                if self.online_mode in GROWTH_MODES else 'current call only',
            optimality='bounded admission search only for onlinepath; fit modes use ordered capacity scan; no global guarantee')

    def add_request(self,request):
        super().add_request(request)
        if self.online_mode=='off':return
        p=self.program(request)
        # A task ID is checked for train/eval overlap, never a ranking feature.
        if (p in self.online_prior['heldout_ids'] or
                (getattr(self,'scalar_stage_prior',None) is not None and
                 p in self.scalar_stage_prior.training_ids) or
                p in getattr(self,'compact_predictor_audit',{}).get('train_ids',()) or
                p in getattr(getattr(self,'suffix_predictor',None),'training_ids',set())):
            raise ValueError('Evaluation/training overlap')
        self.online_inputs.setdefault(p,[]).append(request.num_prompt_tokens)
        self.online_outputs.setdefault(p,[])
        self.online_requests[p]=request;self.online_dirty=True
        if (getattr(self,'cold_admission_ranker',None) is not None and
                p not in self.online_entered):
            if p in self.cold_admission_ranker.training_ids:
                raise ValueError('Cold admission evaluation/training overlap')
            started=time.perf_counter()
            try:
                from kairos._runtime.compat.cold_admission import first_features
                messages=self.metadata(request).get('cont_visible_messages')
                if isinstance(messages,str):messages=json.loads(messages)
                if len(self.online_inputs[p])!=1 or self.online_outputs[p]:
                    raise ValueError('Expected first arrived call')
                text=self.cold_admission_tokenizer.decode(request.prompt_token_ids,skip_special_tokens=False)
                score=self.cold_admission_ranker.score(first_features(
                    text,messages,request.num_prompt_tokens))
                self.cold_admission_scores[p]=score
                self._emit('cold_admission_prediction',program=p,ordinal_score=score,
                    seconds=time.perf_counter()-started)
            except (AttributeError,KeyError,TypeError,ValueError,IndexError) as exc:
                self.cold_admission_scores.pop(p,None)
                self._emit('cold_admission_fallback',program=p,reason=str(exc),
                    seconds=time.perf_counter()-started)
        if getattr(self,'scalar_rank_policy',None) in ('learned','learnedgated'):
            previous=self.learned_previous_hashes.get(p,())
            common=0
            for left,right in zip(previous,getattr(request,'block_hashes',())):
                if left!=right:break
                common+=1
            self.learned_current_keep[p]=min(common*self.block_size,
                (request.num_prompt_tokens-1)//self.block_size*self.block_size)
            self.learned_previous_hashes.pop(p,None)
            self._learned_observe_request(request,after=False)
        if getattr(self,'scalar_rank_policy',None) in SUFFIX_CDF_POLICIES:
            previous=self.suffix_previous_hashes.get(p,());common=0
            for left,right in zip(previous,getattr(request,'block_hashes',())):
                if left!=right:break
                common+=1
            self.suffix_current_keep[p]=min(common*self.block_size,
                (request.num_prompt_tokens-1)//self.block_size*self.block_size)
            self.suffix_previous_hashes.pop(p,None)
            self._suffix_observe_request(request)
        if self.online_mode=='onlinefitrank':
            if p not in self.online_oracle_rank:raise ValueError('Unregistered oracle rank identity')
            self.online_rank[p]=self.online_oracle_rank[p]

    def _learned_observe_request(self, request, *, after):
        """Refresh one causal learned suffix prediction, if this arm is active.

        The harness supplies the messages visible at this request boundary in
        ``cont_visible_messages``.  We never reconstruct a future assistant
        response here.  Missing metadata is a valid cold/deployment fallback
        and is recorded rather than turning a ranking diagnostic into an
        execution failure.
        """
        if getattr(self,'scalar_rank_policy',None) not in ('learned','learnedgated'):
            return
        if after:
            # R+ excludes this entire call. In a fixed teacher replay, actual
            # generated text is not the source of the next teacher input.
            # Reuse the pre-call estimate until an authoritative prefix arrives.
            return
        started=time.perf_counter()
        p=self.program(request)
        raw=self.metadata(request).get('cont_visible_messages')
        if raw is None:
            self.learned_predictions.pop(p,None)
            self._emit('compact_prediction_fallback',program=p,reason='missing-causal-visible-messages',
                after=after,seconds=time.perf_counter()-started)
            return
        try:
            messages=json.loads(raw) if isinstance(raw,str) else raw
            if not isinstance(messages,list):raise ValueError('visible messages must be a list')
            if not messages or messages[-1].get('role')=='assistant':
                raise ValueError('Expected a pre-call visible-message boundary')
            text=self.learned_tokenizer.decode(request.prompt_token_ids,skip_special_tokens=False)
            prompts=self.online_inputs[p]
            outputs=self.online_outputs[p]
            from kairos.features import features as old_features
            from kairos.causal import causal_features
            values,_=causal_features(messages,prompts,outputs,
                (self.learned_service_total.get(p,0.),),after)
            result=self.compact_predictor.predict(old_features(text,prompts,outputs),
                list(values.values()),prompts,outputs,after)
            self.learned_predictions[p]=result
            self._emit('compact_prediction',program=p,after=after,point_s=result['point_s'],
                median_s=result['median_s'],quantiles_s=result['quantiles_s'],
                cold_start=result.get('cold_start',False),fallback=False,
                feature_seconds=time.perf_counter()-started)
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            self.learned_predictions.pop(p,None)
            self._emit('compact_prediction_fallback',program=p,reason=type(exc).__name__,
                after=after,seconds=time.perf_counter()-started)

    def _learned_future(self, p, fallback):
        result=self.learned_predictions.get(p)
        if result is None:return fallback,None
        value=float(result.get('point_s',fallback))
        if not math.isfinite(value) or value<0:return fallback,None
        qs=result.get('quantiles_s') or []
        interval=None
        if len(qs)>=5 and all(math.isfinite(float(x)) and float(x)>=0 for x in qs):
            interval=dict(low_s=float(qs[2]),high_s=float(qs[4]),
                min_s=float(qs[0]),max_s=float(qs[-1]),samples=1,
                source='compact-forest-oof-calibration')
        return value,interval

    def _suffix_observe_request(self, request):
        """Cache one causal suffix distribution at the arrived prefix boundary."""
        started=time.perf_counter();p=self.program(request)
        if self.scalar_rank_policy in SUFFIX_CDF_POLICIES and (
                self.suffix_predictor is None or self.suffix_tokenizer is None):
            # A missing causal artifact is a deployment/configuration error,
            # not a prediction miss.  Falling back here would silently turn a
            # claimed progressive experiment into the incumbent controller.
            raise RuntimeError('Suffix CDF runtime is not initialized for '+self.scalar_rank_policy)
        if self.scalar_rank_policy=='cdfwarm' and not self.online_outputs[p]:
            self.suffix_predictions.pop(p,None)
            self._emit('suffix_cdf_skipped',program=p,reason='cold-incumbent-domain',
                seconds=time.perf_counter()-started)
            return
        raw=self.metadata(request).get('cont_visible_messages')
        if raw is None:
            self.suffix_predictions.pop(p,None)
            self._emit('suffix_cdf_fallback',program=p,reason='missing-causal-visible-messages',
                seconds=time.perf_counter()-started)
            return
        try:
            from kairos.features import features as old_features
            from kairos.causal import causal_features
            messages=json.loads(raw) if isinstance(raw,str) else raw
            if not isinstance(messages,list) or not messages or messages[-1].get('role')=='assistant':
                raise ValueError('Expected a pre-call visible-message boundary')
            text=self.suffix_tokenizer.decode(request.prompt_token_ids,skip_special_tokens=False)
            prompts=self.online_inputs[p];outputs=self.online_outputs[p]
            values,_=causal_features(messages,prompts,outputs,
                (self.suffix_service_total.get(p,0.),),False)
            result=self.suffix_predictor.predict(old_features(text,prompts,outputs),
                list(values.values()),prompts,outputs,program=p)
            self.suffix_predictions[p]=result
            self._emit('suffix_cdf_prediction',program=p,mean_s=result['point_s'],
                median_s=result['median_s'],quantiles_s=result['quantiles_s'],
                cold_start=result['cold_start'],seconds=time.perf_counter()-started)
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            self.suffix_predictions.pop(p,None)
            self._emit('suffix_cdf_fallback',program=p,reason=type(exc).__name__,
                seconds=time.perf_counter()-started)

    @staticmethod
    def _interval_is_residual(policy, interval, current):
        if interval is None:return None
        result=dict(interval)
        if policy!='learnedgated':
            result['low_s']=max(0.,result['low_s']-current)
            result['high_s']=max(0.,result['high_s']-current)
        return result

    def _expire_idle_protection(self,now,queued):
        if self.online_mode=='off':return super()._expire_idle_protection(now,queued)
        for p,pin in list(self.pins.items()):
            if now>=pin.deadline and p not in queued:
                self._release(p,'online_idle_support_expired')
                self.online_insurance.pop(p,None)
                self.online_active.discard(p)
                if p not in self.online_cleaning:self.online_reserved.pop(p,None)
                self.online_dirty=True

    def _refresh_priority(self):
        if self.online_mode=='off':return super()._refresh_priority()
        for name in ('waiting','skipped_waiting'):
            queue=create_request_queue(self.policy)
            for r in getattr(self,name):
                restored=r.request_id in self.finished_recving_kv_req_ids or r.num_computed_tokens>0
                r.priority=(0 if restored else 1)*1000000+self.online_rank.get(self.program(r),self.order[self.program(r)])
                queue.add_request(r)
            setattr(self,name,queue)
        for r in self.running:r.priority=self.online_rank.get(self.program(r),self.order[self.program(r)])

    def _online_target(self,r):
        return math.ceil((r.num_prompt_tokens+r.max_tokens)/self.block_size)

    def _allocate(self, request, num_new_tokens, *args, **kwargs):
        # Materialize an ALREADY FUNDED fixed-output call at its first decode
        # page boundary. This changes allocation granularity, never scheduled
        # tokens, committed hashes, attention length, or available capacity.
        # In particular, do not preallocate a partial prefill or async restore.
        if (not getattr(self, 'online_grant_allocation', False) or
                self.online_mode == 'off' or args or
                request.request_id not in self.online_granted or
                request.num_computed_tokens < request.num_prompt_tokens or
                known_request_output(request) is None or
                any(kwargs.get(k, 0) for k in ('num_new_computed_tokens',
                    'num_external_computed_tokens', 'num_encoder_tokens')) or
                kwargs.get('new_computed_blocks')):
            return super()._allocate(request, num_new_tokens, *args, **kwargs)
        manager = self.kv_cache_manager
        p = self.program(request)
        endpoint = min(self.max_model_len, request.num_computed_tokens + num_new_tokens)
        target = min(self.max_model_len, request.num_prompt_tokens + request.max_tokens)
        blocks = manager.get_blocks(request.request_id).blocks[0]
        # Stay on the ordinary constant-time path until a page is needed.
        if (len(blocks)*self.block_size >= endpoint or target <= endpoint or
                self.online_reserved.get(p, 0)*self.block_size < target):
            return super()._allocate(request, num_new_tokens, *args, **kwargs)
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=target,
            new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0, total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=endpoint)
        free = manager.block_pool.get_num_free_blocks()
        if needed > free - kwargs.get('reserved_blocks', 0):
            # No failed large allocation is sent into pressure-unpin logic.
            # Native one-step allocation may still make ordinary progress.
            self._emit('grant_allocation_deferred', program=p, request=request.request_id,
                required_pages=needed, free_pages=free)
            return super()._allocate(request, num_new_tokens, *args, **kwargs)
        extended = dict(kwargs, num_lookahead_tokens=max(
            kwargs.get('num_lookahead_tokens', 0), target-endpoint))
        started = time.perf_counter()
        result = super()._allocate(request, num_new_tokens, **extended)
        if result is None:
            # The synchronous native preflight above must agree for this
            # full-attention decode path. Fail visibly rather than silently
            # reclaiming state in a second, different allocation transaction.
            raise RuntimeError('Funded decode allocation disagreed with native preflight')
        self._emit('grant_allocation', program=p, request=request.request_id,
            target_tokens=target, scheduled_tokens=num_new_tokens,
            allocated_pages=needed, extra_slots=target-endpoint,
            seconds=time.perf_counter()-started,
            reservation_pages=self.online_reserved[p])
        return result

    def _online_work(self,program,initial,step):
        work=0.
        for index,turn in enumerate(program.turns):
            state=initial.get(program.name,{}) if index==0 else {}
            computed=min(turn.prompt,max(turn.keep,int(state.get('computed',0))))
            generated=min(turn.output,int(state.get('generated',0)))
            queries=turn.prompt-computed
            work+=(turn.output-generated)*step+max(0.,self.beta[1]*queries+
                self.beta[2]*queries*(turn.prompt+computed+1)/2)
        return work

    def _distribution_work(self,name,samples,initial,step):
        """Only current-call service changes between observed-history events."""
        key=(name,tuple(self.online_inputs[name]),tuple(self.online_outputs[name]))
        if key not in self.online_service_cache:
            tails=[]
            for sample in samples:
                tails.append(math.fsum(t.output*step+max(0.,self.beta[1]*(t.prompt-t.keep)+
                    self.beta[2]*(t.prompt-t.keep)*(t.prompt+t.keep+1)/2) for t in sample.turns[1:]))
            # One history per active Program; retired histories cannot grow
            # this cache throughout the lifetime of the serving process.
            for old in [k for k in self.online_service_cache if k[0]==name]:del self.online_service_cache[old]
            self.online_service_cache[key]=tails
        state=initial.get(name,{})
        computed=int(state.get('computed',0));generated=int(state.get('generated',0))
        values=[]
        for sample,tail in zip(samples,self.online_service_cache[key]):
            t=sample.turns[0];keep=min(t.prompt,max(t.keep,computed));q=t.prompt-keep
            values.append(tail+(t.output-min(t.output,generated))*step+
                max(0.,self.beta[1]*q+self.beta[2]*q*(t.prompt+keep+1)/2))
        return values

    def _rank_work_interval(self, provider, name, inputs, outputs, initial, step,
                            request, *, after_completion):
        """Return an empirical middle interval for the causal rank forecast.

        The interval is formed from the same disjoint training paths used by the
        point forecast.  It is advisory: it never authorizes memory or changes
        a native request.  A missing scenario interface deliberately disables
        the gate instead of inventing confidence.
        """
        if not hasattr(provider, 'scenarios'):
            return None
        kwargs=dict(attained=len(request.output_token_ids) if request is not None else 0,
                    current_cap=request.max_tokens if request is not None else None,
                    after_completion=after_completion,page=self.block_size,
                    max_len=self.max_model_len)
        if (request is not None and getattr(self, 'online_known_output', False)):
            kwargs['known_current_output']=known_request_output(request)
        samples=provider.scenarios(name, inputs, outputs, **kwargs)
        values=sorted(self._online_work(sample, initial, step) for sample in samples)
        if not values:
            return None
        # A central empirical interval is intentionally used instead of a
        # hand-tuned seconds margin.  With one neighbor it collapses to a point.
        def quantile(q):
            position=(len(values)-1)*q
            left=int(math.floor(position));right=int(math.ceil(position))
            if left==right:return values[left]
            weight=position-left
            return values[left]*(1.-weight)+values[right]*weight
        lo=quantile(.25)
        hi=quantile(.75)
        return dict(low_s=lo,high_s=hi,samples=len(values),
                    min_s=values[0],max_s=values[-1])

    def _online_snapshot(self,queued):
        programs=[];initial={};quotes={};minimum={}
        ready={self.program(r):r for r in queued}
        running={self.program(r):r for r in self.running}
        now=time.monotonic();names=set(ready)|set(running)|set(self.pins)
        # Physically cleaning-up calls still occupy logical capacity; no
        # imaginary early release is allowed in either rollout or execution.
        cleaning=set(self.online_reserved)-names
        provider=self.online_truth if self.online_mode in TRUTH_MODES else self.forecaster
        self.online_next={}
        for p in sorted(names,key=self.order.get):
            r=running.get(p) or ready.get(p)
            if r is not None:
                attained=len(r.output_token_ids)
                knowledge = ({'known_current_output': known_request_output(r)}
                             if getattr(self, 'online_known_output', False) else {})
                program,quote=provider.forecast(p,self.online_inputs[p],self.online_outputs[p],
                    attained=attained,current_cap=r.max_tokens,memory_pages=self.online_reserved.get(p,0),
                    page=self.block_size,max_len=self.max_model_len,**knowledge)
                minimum[p]=self._online_target(r)
                if p in self.online_active or p in running:
                    computed=min(r.num_computed_tokens,r.num_tokens)
                    if p not in running and not computed:
                        pin=self.pins.get(p);common=0
                        if pin:
                            for a,b in zip(r.block_hashes,pin.lookup_hashes):
                                if a!=b or common>=len(pin.blocks):break
                                common+=1
                        computed=min(common*self.block_size,r.num_prompt_tokens-1)
                    initial[p]=dict(ready=0.,computed=computed,generated=attained,
                        pages=self.online_reserved.get(p,minimum[p]),started=p in running)
            else:
                program,quote=provider.forecast(p,self.online_inputs[p],self.online_outputs[p],
                    after_completion=True,memory_pages=self.online_reserved.get(p,0),
                    page=self.block_size,max_len=self.max_model_len)
                pin=self.pins[p];age=max(0.,now-pin.started)
                # Unknown tool duration is estimated from disjoint support;
                # actual tool return is an external event, never synthesized.
                initial[p]=dict(ready=self.forecaster.idle_remaining(pin.tool,age),computed=min(
                    self.online_inputs[p][-1]//self.block_size*self.block_size,program.turns[0].prompt-1),
                    generated=0,pages=self.online_reserved.get(p,len(pin.blocks)),started=False)
            future=next_envelope(program,after_completion=r is None,page=self.block_size)
            if future is not None:self.online_next[p]=future
            programs.append(program);quotes[p]=quote
        service=self.profile.get('pipeline_service',{})
        step=float(service.get('decode_step_by_seqs',{}).get('1',.02))
        # Publish the same service units used by the incumbent controller to
        # every R5 action evaluator.  The prior's output-token posterior is an
        # information artifact; it is not a wall-clock service quote.  Keeping
        # this conversion at the scheduler boundary prevents token counts and
        # seconds from being silently added to one another.
        for p in programs:
            state=initial.get(p.name,{})
            total=self._online_work(p,initial,step)
            current=(self._online_work(_current_turn_view(p),initial,step)
                     if p.name in ready or p.name in running else 0.)
            quotes[p.name]['current_service_s']=max(0.,current)
            quotes[p.name]['total_service_s']=max(0.,total)
            quotes[p.name]['residual_service_s']=max(0.,total-current)
            quotes[p.name]['service_unit']='profiled_seconds'
        # The historical incumbent call-commitment protocol computes these diagnostic means but dispatches
        # by its representative envelope order. C2 explicitly makes the
        # mean executable; the lean arm omits means unused by the incumbent.
        # A mean is sufficient for SEPT only in the all-ready serial special
        # case, not for batched self-suspending agents with cache constraints.
        if getattr(self, 'r5_joint_service', False):
            for p in programs:
                r = running.get(p.name) or ready.get(p.name)
                if not self.online_outputs.get(p.name):
                    continue
                # C2 prices the same posterior for idle victims and ready
                # continuations. incumbent call-commitment protocol preserves its historical ready-only
                # calculation and ordinal action order.
                if r is None and not getattr(self, 'r5_mean_rank', False):
                    continue
                try:
                    knowledge = ({'known_current_output': known_request_output(r)}
                                 if r is not None and getattr(self, 'online_known_output', False) else {})
                    samples = self.forecaster.scenarios(
                        p.name, self.online_inputs[p.name], self.online_outputs[p.name],
                        attained=len(r.output_token_ids) if r is not None else 0,
                        current_cap=r.max_tokens if r is not None else None,
                        after_completion=r is None, page=self.block_size,
                        max_len=self.max_model_len, **knowledge)
                    totals = [max(0., self._online_work(sample, initial, step))
                              for sample in samples]
                    if totals:
                        # Equal-weight empirical scenarios are the posterior
                        # mass emitted by CommitmentPrior.  Keep the values
                        # in the quote for auditability and avoid a hidden
                        # scheduler-side re-prediction.
                        quotes[p.name]['service_distribution_s'] = sorted(totals)
                        quotes[p.name]['expected_total_service_s'] = math.fsum(totals) / len(totals)
                        quotes[p.name]['expected_residual_service_s'] = max(
                            0., quotes[p.name]['expected_total_service_s'] -
                            quotes[p.name]['current_service_s'])
                        quotes[p.name]['ranking_source'] = 'causal-joint-empirical-mean'
                except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                    # A missing scenario interface is a deployment fallback,
                    # not permission to use future truth.  The quoted median
                    # path remains a conservative executable price.
                    quotes[p.name]['ranking_source'] = 'profiled-envelope-fallback'
        def work(p):
            if self.online_mode in MEAN_MODES:
                r=running.get(p.name) or ready.get(p.name)
                samples=self.forecaster.scenarios(p.name,self.online_inputs[p.name],self.online_outputs[p.name],
                    attained=len(r.output_token_ids) if r is not None else 0,
                    current_cap=r.max_tokens if r is not None else None,
                    after_completion=r is None,page=self.block_size,max_len=self.max_model_len,
                    **({'known_current_output': known_request_output(r)} if r is not None and
                       getattr(self, 'online_known_output', False) else {}))
                values=(self._distribution_work(p.name,samples,initial,step)
                        if self.online_mode in CALIBRATED_MODES else
                        [self._online_work(sample,initial,step) for sample in samples])
                value=(empirical_gittins(values) if self.online_mode=='onlinefitgittins'
                       else math.fsum(values)/len(values))
                quotes[p.name]['work_s']=value;quotes[p.name]['work_samples']=len(samples)
                quotes[p.name]['work_rule']='gittins-service-budget' if self.online_mode=='onlinefitgittins' else 'conditional-mean'
                return value
            if self.online_mode in RESIDUAL_MODES:return self._online_work(p,initial,step)
            return sum(t.output*step+max(0.,self.beta[1]*(t.prompt-t.keep)+
                self.beta[2]*(t.prompt-t.keep)*(t.prompt+t.keep+1)/2) for t in p.turns)
        scores={p.name:work(p) for p in programs}
        if getattr(self, 'r5_mean_rank', False):
            # Retain the incumbent completion-envelope order separately.
            # Service order and cache-victim order are distinct decisions:
            # the C3 split-rank arm uses the conditional mean for dispatch,
            # but needs this pre-intervention order for optional-pin victims.
            incumbent_order=sorted(programs,
                key=lambda p:(scores[p.name],self.order[p.name]))
            self.r5_incumbent_order=[p.name for p in incumbent_order]
            # Publish one order for dispatch, continuity pricing and optional
            # support reclaim. Known current work ranks cold calls; the
            # empirical expected total ranks every entered ready/idle owner.
            scores={p.name: quotes[p.name].get('expected_total_service_s',
                                             quotes[p.name]['total_service_s'])
                    if self.online_outputs.get(p.name) else
                    quotes[p.name]['current_service_s'] for p in programs}
        if getattr(self,'scalar_rank_policy',None):
            details={}
            rank_intervals={}
            for p in programs:
                r=running.get(p.name) or ready.get(p.name)
                current=(self._online_work(_current_turn_view(p),initial,step)
                         if r is not None else 0.)
                future=max(0.,scores[p.name]-current)
                learned_interval=None
                path_future=future
                learned_available=p.name in getattr(self,'learned_predictions',{})
                if self.scalar_rank_policy in ('learned','learnedgated'):
                    future,learned_interval=self._learned_future(p.name,future)
                    scores[p.name]=current+future
                # This model was fitted at pre-call landmarks. A tool-idle
                # state is a different conditioning event; retain the original
                # idle victim ordering rather than reuse a terminal mass there.
                # Cold-start suffixes are not identifiable from a causal
                # prefix.  progressivewindow deliberately keeps the mature
                # incumbent prior for the first call and only trusts the
                # distribution after one observed completion.  This is a
                # policy choice, not a predictor fallback: the prediction is
                # still logged, but cannot reorder the first-call cohort.
                suffix_distribution=(getattr(self,'suffix_predictions',{}).get(p.name)
                    if r is not None and self.scalar_rank_policy in SUFFIX_CDF_POLICIES and
                       (self.scalar_rank_policy in ('cdfmean','cdfindex','progressive','progressivedebt') or
                        (self.scalar_rank_policy in ('cdfwarm','progressivewindow') and
                         len(self.online_outputs[p.name])>0)) else None)
                if suffix_distribution is not None:
                    future=float(suffix_distribution['point_s'])
                if self.scalar_rank_policy in ('gated','learnedgated') and len(self.online_outputs[p.name]) > 0:
                    interval=self._rank_work_interval(provider,p.name,self.online_inputs[p.name],
                        self.online_outputs[p.name],initial,step,r,after_completion=r is None)
                    if self.scalar_rank_policy == 'learnedgated':
                        interval=learned_interval
                    if interval is not None:
                        # The path interval quotes the complete current state;
                        # compact predictor quantiles already target R+ only.
                        interval=self._interval_is_residual(
                            self.scalar_rank_policy,interval,current)
                        rank_intervals[p.name]=interval
                stage_future=(self.scalar_stage_prior.future_work(len(self.online_outputs[p.name]),
                    idle=r is None,beta=self.beta,step=step) if getattr(self,'scalar_stage_prior',None) is not None else None)
                truth_future=(self.scalar_rank_truth.after_current(p.name,
                    len(self.online_outputs[p.name]),idle=r is None,beta=self.beta,step=step)
                    if self.scalar_rank_truth is not None else None)
                effective_policy=('predicted' if self.scalar_rank_policy in ('warm','gated','learnedgated') and
                                  len(self.online_outputs[p.name]) > 0 else
                                  'current' if self.scalar_rank_policy in ('warm','gated','learnedgated') else
                                  self.scalar_rank_policy)
                effective_policy=phase_policy(effective_policy,len(self.online_outputs[p.name]))
                adaptive_frontier = None
                progressive_frontier = None
                if effective_policy == 'adaptive':
                    # Progressive revelation: cold states use only the next
                    # call support; warm states use the freshly conditioned
                    # residual support. The actual return-competition gate is
                    # applied in _online_control below.
                    try:
                        from kairos._runtime.compat.adaptive_service_policy import choose
                        samples = provider.scenarios(
                            p.name, self.online_inputs[p.name], self.online_outputs[p.name],
                            attained=len(r.output_token_ids) if r is not None else 0,
                            current_cap=r.max_tokens if r is not None else None,
                            after_completion=r is None, page=self.block_size,
                            max_len=self.max_model_len,
                            **({'known_current_output': known_request_output(r)}
                               if r is not None and getattr(self, 'online_known_output', False)
                               else {}))
                        current_values = [self._online_work(
                            replace(sample, turns=sample.turns[:1]), initial, step)
                            for sample in samples]
                        residual_values = [self._online_work(sample, initial, step)
                                           for sample in samples]
                        adaptive = choose(current=current_values, residual=residual_values,
                                          warm=bool(self.online_outputs[p.name]))
                        score = adaptive['cost_s']
                        adaptive_frontier = adaptive
                    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                        # Missing posterior data never blocks a real request.
                        score = current
                        adaptive_frontier = dict(
                            mode='probe-fallback', cost_s=current)
                elif effective_policy in PROGRESSIVE_POLICIES:
                    # The progressive controller does not use a cold suffix
                    # point forecast to displace a returned continuation.
                    # It exposes the current-call price for a cold probe and
                    # a posterior residual only for diagnostics/warm ranking.
                    try:
                        from kairos._runtime.compat.progressive_service_policy import choose
                        samples = provider.scenarios(
                            p.name, self.online_inputs[p.name], self.online_outputs[p.name],
                            attained=len(r.output_token_ids) if r is not None else 0,
                            current_cap=r.max_tokens if r is not None else None,
                            after_completion=r is None, page=self.block_size,
                            max_len=self.max_model_len,
                            **({'known_current_output': known_request_output(r)}
                               if r is not None and getattr(self, 'online_known_output', False)
                               else {}))
                        current_values=[self._online_work(
                            replace(sample, turns=sample.turns[:1]), initial, step)
                            for sample in samples]
                        # A conditional suffix model can have no compatible
                        # complete path at a terminal/overlong stage.  The
                        # control boundary must remain executable: the
                        # already observed current-call quote is the only
                        # causal atom available, not an empty distribution.
                        if not current_values:
                            current_values=[max(0., current)]
                        if suffix_distribution is not None:
                            current_median=statistics.median(current_values)
                            atoms=[current_median+x for x in suffix_distribution['atoms_s']]
                            mass=suffix_distribution['mass']
                        else:
                            atoms=sorted(self._online_work(sample, initial, step)
                                         for sample in samples)
                            if not atoms:
                                atoms=[max(0., current)]
                            mass=[1.0/len(atoms)]*len(atoms)
                        progressive_frontier=choose(current=current_values,
                            residual_atoms=atoms,residual_mass=mass,
                            warm=bool(self.online_outputs[p.name]),
                            observed_s=self.progressive_observed_s.get(p.name,0.))
                        score=progressive_frontier['score_s']
                    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
                        score = current if not self.online_outputs[p.name] else current + future
                        progressive_frontier = dict(mode='fallback', score_s=score,
                            predictor='path-fallback')
                    progressive_gate = ('cold-probe' if not self.online_outputs[p.name]
                        else 'warm-residual-diagnostic')
                elif effective_policy=='cdfindex' and suffix_distribution is not None:
                    from kairos._runtime.unsupported import completion_cost
                    score=completion_cost(current,suffix_distribution['atoms_s'],suffix_distribution['mass'])
                elif effective_policy in ('cdfmean','cdfwarm') and suffix_distribution is None:
                    # Cold start is outside the CDF model's reliable domain;
                    # retain incumbent causal path ranking until one call is observed.
                    effective_policy='predicted' if effective_policy=='cdfwarm' else effective_policy
                    score=rank_score(effective_policy,current=current,
                        future=stage_future if stage_future is not None else future,
                        arrival_order=self.order[p.name],truth_future=truth_future)
                else:
                    score=rank_score(effective_policy,current=current,
                        future=stage_future if stage_future is not None else future,
                        arrival_order=self.order[p.name],truth_future=truth_future)
                details[p.name]=dict(score=score,current_s=current,predicted_future_s=future,
                    path_forecast_future_s=path_future,
                    learned_forecast_future_s=(future if learned_available else None),
                    learned_forecast_available=learned_available,
                    stage_future_s=stage_future,
                    truth_future_s=truth_future,arrived_call=r is not None,
                    rank_policy_used=effective_policy,
                    completed_calls=len(self.online_outputs[p.name]),
                    state=('running' if p.name in running else 'ready' if p.name in ready else 'tool'),
                    entered=p.name in self.online_entered,
                    granted=bool(r is not None and r.request_id in self.online_granted),
                    minimum_pages=minimum.get(p.name,0),
                    reserved_pages=self.online_reserved.get(p.name,0),
                    suffix_prediction_available=suffix_distribution is not None,
                    suffix_prediction_used=suffix_distribution is not None,
                    completion_index_s=(score if effective_policy=='cdfindex' and suffix_distribution else None),
                    adaptive_frontier=adaptive_frontier,
                    progressive_gate=progressive_gate if effective_policy in PROGRESSIVE_POLICIES else None,
                    progressive_frontier=progressive_frontier)
                scores[p.name]=score
            if self.scalar_rank_policy in ('gated','learnedgated'):
                # Do not let an uncertain point forecast reorder a pair.  A
                # pair is trusted only when its predicted central intervals do
                # not overlap; all cold-start programs conservatively retain
                # current-work ordering.
                use={p.name: len(self.online_outputs[p.name]) > 0 and
                     p.name in rank_intervals for p in programs}
                totals={p.name: details[p.name]['current_s']+
                        details[p.name]['predicted_future_s'] for p in programs}
                current_scores={p.name: details[p.name]['current_s'] for p in programs}
                names=[p.name for p in programs]
                for i,left in enumerate(names):
                    for right in names[i+1:]:
                        predicted_order=(totals[left],self.order[left]) < (totals[right],self.order[right])
                        current_order=(current_scores[left],self.order[left]) < (current_scores[right],self.order[right])
                        if not (use[left] and use[right]) or predicted_order == current_order:
                            continue
                        a=rank_intervals[left];b=rank_intervals[right]
                        a_lo=details[left]['current_s']+a['low_s'];a_hi=details[left]['current_s']+a['high_s']
                        b_lo=details[right]['current_s']+b['low_s'];b_hi=details[right]['current_s']+b['high_s']
                        if max(a_lo,b_lo) <= min(a_hi,b_hi):
                            use[left]=use[right]=False
                for p in programs:
                    row=details[p.name];interval=rank_intervals.get(p.name)
                    row['rank_policy_used']='predicted' if use[p.name] else 'current'
                    row['gated_prediction_used']=use[p.name]
                    row['learned_prediction_used']=use[p.name] if self.scalar_rank_policy=='learnedgated' else None
                    row['gated_interval_s']=(dict(interval) if interval is not None else None)
                    if not use[p.name]:
                        scores[p.name]=row['current_s']
            self._emit('scalar_rank_decision',policy=self.scalar_rank_policy,rows=details,
                pricing_beta=self.beta,pricing_decode_step=step,
                capacity_pages=self._capacity()-32,
                reserved_pages=sum(self.online_reserved.values()),
                active_count=len(self.online_active),resident_limit=self.max_num_running_reqs)
        order=[p.name for p in sorted(programs,key=lambda p:(scores[p.name],self.order[p.name]))]
        if self.online_mode=='onlinefitrank':
            order.sort(key=self.online_oracle_rank.__getitem__)
            self.online_rank={p:self.online_oracle_rank[p] for p in order}
        else:self.online_rank={p:i for i,p in enumerate(order)}
        return programs,order,initial,quotes,minimum,sum(self.online_reserved[p] for p in cleaning)

    def _online_fund_growth(self,order,limit):
        """Execute concrete next-call funding before spending slack on entry.

        Running/current grants and DMA cleanup are never released. Insurance
        is retained until consumed, the idle lease expires, or the existing
        legal growth-recovery path revokes an optional idle reference.
        This deliberately small policy is a simple-reservation control, not
        an optimal calendar or a guarantee against prediction failure.
        """
        used=sum(self.online_reserved.values())
        for p in order:
            if p not in self.online_active or p in self.online_ended:continue
            target=self.online_next.get(p,0)
            old=self.online_reserved.get(p,0)
            if target<=old or used+target-old>limit:continue
            self.online_reserved[p]=target;used+=target-old
            self.online_insurance[p]=target
            self._emit('online_path_growth_funding',program=p,pages=target,
                added_pages=target-old,reserved_pages=used,
                completed_calls=len(self.online_outputs[p]),
                capacity_pages=limit+32,information='truth' if self.online_mode in TRUTH_MODES else 'prediction')

    def _online_has_slot(self,p):
        # Cache residence consumes HBM, not a native execution descriptor.
        # In the attribution arm, native vLLM alone enforces max_num_seqs;
        # granted waiting calls still reserve their complete real-call KV.
        return (self.online_mode=='onlinefitnative' or p in self.online_active or
                len(self.online_active)<self.max_num_running_reqs)

    def _online_reclaim(self,request,limit):
        """Observed forecast failure: recover progress using optional refs only."""
        p=self.program(request);target=self._online_target(request)
        used=sum(self.online_reserved.values())
        def fits():
            return (used-self.online_reserved.get(p,0)+target<=limit and self._online_has_slot(p))
        if fits():return True
        running={self.program(r) for r in self.running}
        victims=[q for q in self.pins if q!=p and q not in running and q not in self.online_cleaning
            and self._online_idle_reclaimable(q) and not (
            q in self.online_requests and
            self.online_requests[q].request_id in self.online_granted)]
        victim_rank = getattr(self, 'online_victim_rank', None) or self.online_rank
        victims.sort(key=lambda q:victim_rank.get(q,self.order[q]),reverse=True)
        for q in victims:
            self._release(q,'online_observed_growth_deficit')
            self.online_insurance.pop(q,None)
            self.online_active.discard(q);used-=self.online_reserved.pop(q,0)
            self._emit('online_path_recovery',program=p,victim=q,target_pages=target,reserved_pages=used)
            if fits():return True
        return False

    def _online_idle_reclaimable(self, program):
        return True

    def _online_restore_fits(self,request,limit):
        """Lifetime service identity does not grant another cohort's memory.

        Restoring a nonresident continuation uses genuinely uncommitted
        capacity. Only growth of an already resident call can revoke an idle
        commitment as forecast-error recovery. This prevents reciprocal
        restore/revoke cycles while preserving the continuation queue.
        """
        p=self.program(request)
        return (sum(self.online_reserved.values())-self.online_reserved.get(p,0)+
                self._online_target(request)<=limit and self._online_has_slot(p))

    def _cold_admission_order(self,programs,order,initial,candidates):
        policy=getattr(self,'cold_admission_policy',None)
        if policy is None:return order
        from kairos._runtime.compat.cold_admission import reorder_fresh
        started=time.perf_counter()
        # Only never-entered first calls are eligible. A cache-evicted return,
        # an already granted call and a preempted native request cannot enter.
        fresh={p for p,r in candidates.items() if p not in self.online_entered and
            not self.online_outputs[p] and len(self.online_inputs[p])==1 and
            r.request_id not in self.online_granted and not r.num_computed_tokens}
        scores={p:self.cold_admission_scores[p] for p in fresh
                if p in self.cold_admission_scores}
        if policy=='coldadmissionprompt':
            scores={p:candidates[p].num_prompt_tokens for p in fresh}
        if policy=='coldadmissiontruth':
            service=self.profile['pipeline_service']
            step=float(service.get('decode_step_by_seqs',{}).get('1',.02))
            for p in programs:
                if p.name in fresh:
                    current=self._online_work(_current_turn_view(p),initial,step)
                    scores[p.name]=current+self.cold_admission_truth.after_current(
                        p.name,0,beta=self.beta,step=step)
        result=reorder_fresh(order,fresh,scores)
        self._emit('cold_admission_decision',policy=policy,
            base_order=[p for p in order if p in fresh],
            order=[p for p in result if p in fresh],scores=scores,
            changed=result!=order,fallback=any(p not in scores for p in fresh),
            seconds=time.perf_counter()-started)
        return result

    def _adaptive_compete(self, queued, snapshot, limit):
        """Let a returned call and a fresh call compete in one service round.

        Continuum's normal contract gives every returned continuation first
        claim.  That is appropriate for cache continuity, but it hides the
        information-acquisition problem: a long unknown Program can keep
        re-entering before a cold Program is ever observed.  This bounded
        policy keeps all hard capacity/DMA rules and only changes the order of
        *funded, arrived* calls.  Each call still runs to its native boundary;
        the next completion causes a fresh posterior/rank update.
        """
        if snapshot is None:
            return False
        _, order, _, quotes, minimum, _ = snapshot
        by_program = {self.program(r): r for r in queued}
        selected = []
        blocked = []
        # Do not walk the whole queue and grant every candidate.  A grant is
        # a service opportunity; the next logical completion must make the
        # controller dirty so that the posterior can be refreshed first.
        free_slots = max(0, self.max_num_running_reqs - len(self.online_active))
        budget = max(1, free_slots)
        for p in order:
            if len(selected) >= budget:
                break
            r = by_program.get(p)
            if r is None or r.request_id in self.online_granted:
                continue
            if p in self.online_entered:
                funded = (self._online_reclaim(r, limit) if p in self.online_active
                          else self._online_restore_fits(r, limit))
                if funded:
                    self._online_grant(r, self._online_target(r), 'adaptive-return-competition')
                    selected.append(dict(program=p, kind='return', request=r.request_id,
                                         score=self.online_rank.get(p)))
                else:
                    blocked.append(p)
                continue
            if p not in minimum or p not in by_program:
                continue
            pages = minimum[p]
            if not self._online_has_slot(p) or sum(self.online_reserved.values()) + pages > limit:
                blocked.append(p)
                continue
            self._online_grant(r, pages, 'adaptive-probe-competition')
            selected.append(dict(program=p, kind='probe', request=r.request_id,
                                 score=self.online_rank.get(p)))
        self._emit('adaptive_service_decision', order=order, selected=selected,
            blocked=blocked, cold=sum(x['kind']=='probe' for x in selected),
            warm=sum(x['kind']=='return' for x in selected),
            returned_first_contract=False, quantum='one native call boundary',
            posterior_refresh='after each logical completion')
        self.online_dirty = False
        return True

    def _progressive_compete(self, queued, snapshot, limit):
        """PIS-KV control: continuity-preserving returns, bounded cold probe.

        Returned continuations remain the first class of work, but are ordered
        by measured solo service already paid by that Program.  A cold request
        is admitted only when no return can currently be funded; it therefore
        obtains one real call as an information quantum without displacing a
        resident continuation or its KV lease.
        """
        started=time.perf_counter()
        if snapshot is None:
            return False
        _, order, _, _, minimum, _ = snapshot
        by_program = {self.program(r): r for r in queued}
        returns = [p for p, r in by_program.items()
                   if p in self.online_entered and r.request_id not in self.online_granted]
        if getattr(self, 'progressive_debt_compete', False):
            returns.sort(key=lambda p: (self.progressive_observed_s.get(p, 0.0),
                                        self.online_rank.get(p, self.order.get(p, 0))))
        else:
            # r3 leaves the incumbent posterior order intact for returns;
            # only the independent cold-probe window is widened.
            returns.sort(key=lambda p: self.online_rank.get(p, self.order.get(p, 0)))
        debt_order = list(returns)
        if getattr(self, 'progressive_debt_compete', False):
            # r1 used the debt order only while funding calls. Native vLLM
            # then rebuilt its waiting queue from ``online_rank``. Publish the
            # same order here so the observed information reaches execution.
            debt_set = set(debt_order)
            rest = [p for p in order if p not in debt_set]
            self.online_rank = {p: i for i, p in enumerate(debt_order + rest)}
        selected=[]; blocked=[]
        initial_cold_window=(max(1,self.max_num_running_reqs-len(self.online_active))
                             if getattr(self,'progressive_window_compete',False) else 1)
        for p in returns:
            r=by_program[p]
            funded=(self._online_reclaim(r, limit) if p in self.online_active
                    else self._online_restore_fits(r, limit))
            if funded:
                self._online_grant(r, self._online_target(r), 'progressive-return-debt')
                selected.append(dict(program=p, kind='return',
                                     attained_service_s=self.progressive_observed_s.get(p,0.0)))
            else:
                blocked.append(p)
        # Do not use a cold probe while any return was granted or is blocked;
        # this is the KV-continuity guard that the first adaptive prototype
        # lacked.
        if not selected and not returns:
            fresh=[p for p in order if p in by_program and p not in self.online_entered]
            window=(max(1,self.max_num_running_reqs-len(self.online_active))
                    if getattr(self,'progressive_window_compete',False) else 1)
            for p in fresh:
                if len(selected) >= window:
                    break
                r=by_program[p]; pages=minimum.get(p, self._online_target(r))
                if self._online_has_slot(p) and sum(self.online_reserved.values())+pages<=limit:
                    self._online_grant(r,pages,'progressive-cold-probe')
                    selected.append(dict(program=p,kind='cold-probe'))
                    continue
                blocked.append(p)
        self._emit('progressive_service_decision',order=order,selected=selected,
            blocked=blocked,returned_first=bool(returns),cold_probe=any(
                x.get('kind')=='cold-probe' for x in selected),
            execution_order=(debt_order + [p for p in order if p not in set(debt_order)]
                             if getattr(self, 'progressive_debt_compete', False) else order),
            rank_source=('observed-service-debt' if getattr(self, 'progressive_debt_compete', False)
                         else 'posterior-score'),
            cold_probe_window=initial_cold_window,
            information_quantum='one complete native call',
            posterior_refresh='after each logical completion',
            observed_service=dict(self.progressive_observed_s),
            seconds=time.perf_counter()-started)
        self.online_dirty=False
        return True

    def _online_control(self,queued):
        if not self.online_dirty:return
        started=time.perf_counter();limit=self._capacity()-32
        snapshot=None;forecast_seconds=0.
        if self.online_mode=='onlinefitfresh' or self.online_mode in RESIDUAL_MODES:
            quoted_at=time.perf_counter()
            snapshot=self._online_snapshot(queued)
            forecast_seconds=time.perf_counter()-quoted_at
        if getattr(self, 'adaptive_compete', False):
            # Do not run the incumbent returned-first loop below.  Returned
            # continuations and fresh calls must enter the same bounded
            # service-round competition.
            self._adaptive_compete(queued, snapshot, limit)
            return
        if getattr(self, 'progressive_compete', False):
            self._progressive_compete(queued, snapshot, limit)
            return
        # An already-entered actual return takes precedence over fresh entry,
        # even if its optional cache protection has expired. It still funds
        # its full real call and competes for resident slots before granting.
        # If every call waits on growth, optional idle references are revoked;
        # no running/DMA reference is ever reclaimed by this controller.
        for r in sorted(queued,key=lambda r:self.online_rank.get(self.program(r),self.order[self.program(r)])):
            p=self.program(r)
            if p in self.online_entered and r.request_id not in self.online_granted:
                funded=(self._online_reclaim(r,limit) if p in self.online_active
                        else self._online_restore_fits(r,limit))
                if funded:self._online_grant(r,self._online_target(r),'actual-return')
        if self.online_mode in GROWTH_MODES:
            self._online_fund_growth(snapshot[1],limit)
        blocked_returns=[r for r in queued if self.program(r) in self.online_entered
                         and r.request_id not in self.online_granted]
        candidates={self.program(r):r for r in queued if self.program(r) not in self.online_entered}
        if not candidates or (blocked_returns and self.online_mode not in FIT_MODES):
            self.online_dirty=False
            self._emit('online_path_control',mode=self.online_mode,seconds=time.perf_counter()-started,
                order=snapshot[1] if snapshot else [],quotes=snapshot[3] if snapshot else {},
                forecast_seconds=forecast_seconds,
                selected='continuation-capacity-wait' if blocked_returns else 'no-candidate',
                evaluations=[],initial=[],admit=[],
                forecast_cache_hits=self.forecaster.cache_hits,forecast_cache_misses=self.forecaster.cache_misses,
                reserved_pages=sum(self.online_reserved.values()),capacity_pages=self._capacity(),
                withheld=sum(r.request_id not in self.online_granted for r in queued))
            return
        if snapshot is None:
            quoted_at=time.perf_counter()
            snapshot=self._online_snapshot(queued)
            forecast_seconds=time.perf_counter()-quoted_at
        programs,order,initial,quotes,minimum,cleanup=snapshot
        order=self._cold_admission_order(programs,order,initial,candidates)
        # The peak arm is an attribution control with the SAME causal predictor
        # and rank. It differs only in future capacity commitment and gating.
        result=dict(admit=[],evaluations=[],selected='predicted-peak')
        if self.online_mode in FIT_MODES:
            # Causal admission ablation: forecast/rank/leases and return
            # handling are unchanged. Fresh entry reserves its actual call,
            # without a speculative complete-future feasibility gate and
            # without revoking somebody else's idle commitment. A blocked
            # larger return does not conceal genuinely uncommitted space.
            result=dict(admit=[],evaluations=[],selected='current-capacity')
            used=sum(self.online_reserved.values())
            slots=len(self.online_active)
            for p in order:
                if p not in candidates:continue
                if self.online_mode!='onlinefitnative' and slots>=self.max_num_running_reqs:break
                if used+minimum[p]>limit:continue
                result['admit'].append(p);used+=minimum[p];slots+=1
        elif self.online_mode=='onlinepeak':
            used=sum(self.online_reserved.values())
            for p in order:
                if p not in candidates:continue
                target=quotes[p]['peak_pages']
                if used+target>limit or len(self.online_active)+len(result['admit'])>=self.max_num_running_reqs:break
                result['admit'].append(p);used+=target
        elif candidates:
            service=self.profile.get('pipeline_service')
            if not service:raise RuntimeError('Online planning requires independent pipeline service calibration')
            settings=dict(capacity_pages=self._capacity()-cleanup,headroom_pages=32,
                page_size=self.block_size,max_seqs=self.max_num_running_reqs,
                query_budget=self.max_num_scheduled_tokens,
                mixed_floor=0.,host_step=0.)
            result=choose_continuation(programs,order,initial,service['beta'],
                service['decode_step_by_seqs'],settings=settings,minimum_pages=minimum,
                evaluator=self.online_evaluator)
        for p in result['admit']:
            if p not in candidates:
                raise RuntimeError('Residual planner attempted to re-admit mandatory work')
            r=candidates[p];target=(quotes[p]['peak_pages'] if self.online_mode=='onlinepeak' else minimum[p])
            if sum(self.online_reserved.values())+target>limit:continue
            if not self._online_has_slot(p):break
            self._online_grant(r,target,'new-admission')
            if self.online_mode in GROWTH_MODES:
                # This new resident must fund its own predicted growth before
                # another fresh entry can consume that same logical space.
                self._online_fund_growth([p],limit)
        # If the model rejects every continuation, actual finite running work
        # and tool events normally unblock it. With none available, let one
        # real request proceed, reclaiming only optional idle state as needed.
        progress=bool(self.running or getattr(self,'_pipeline_inflight',{}) or
                      getattr(self,'deferred_frees',[]) or self.pins or self.online_granted)
        if not progress and candidates:
            p=next(p for p in order if p in candidates);r=candidates[p]
            if self._online_reclaim(r,limit):self._online_grant(r,minimum[p],'no-progress-fallback')
        self.online_dirty=False
        self._emit('online_path_control',mode=self.online_mode,seconds=time.perf_counter()-started,
            order=order,quotes=quotes,selected=result['selected'],evaluations=result['evaluations'],
            forecast_seconds=forecast_seconds,planning_seconds=result.get('seconds',0.),
            forecast_cache_hits=self.forecaster.cache_hits,forecast_cache_misses=self.forecaster.cache_misses,
            initial=list(initial),admit=result['admit'],reserved_pages=sum(self.online_reserved.values()),
            capacity_pages=self._capacity(),withheld=sum(r.request_id not in self.online_granted for r in queued))

    def _online_grant(self,r,pages,reason):
        p=self.program(r)
        if self.online_mode=='onlinepeak': pages=max(pages,self.online_reserved.get(p,0))
        if self.online_mode in GROWTH_MODES:
            insured=self.online_insurance.pop(p,0)
            if insured:
                self._emit('online_path_growth_consumed',program=p,request=r.request_id,
                    funded_pages=insured,actual_pages=pages,covered=insured>=pages)
            pages=max(pages,self.online_reserved.get(p,0))
        self.online_entered.add(p);self.online_active.add(p);self.online_reserved[p]=pages
        self.online_granted.add(r.request_id)
        self._emit('online_path_grant',program=p,request=r.request_id,pages=pages,
            reserved_pages=sum(self.online_reserved.values()),reason=reason)

    def _run_native_schedule(self,throttle_prefills=False):
        if self.online_mode=='off':return super()._run_native_schedule(throttle_prefills)
        rank_changed=self.online_dirty and (self.online_mode=='onlinefitfresh' or self.online_mode in RESIDUAL_MODES)
        self._online_control(list(self.waiting)+list(self.skipped_waiting))
        if rank_changed:self._refresh_priority()
        saved=(self.waiting,self.skipped_waiting)
        try:
            for name in ('waiting','skipped_waiting'):
                queue=create_request_queue(self.policy)
                for r in getattr(self,name):
                    if r.request_id in self.online_granted:queue.add_request(r)
                setattr(self,name,queue)
            return super()._run_native_schedule(throttle_prefills)
        finally:
            for name,original in zip(('waiting','skipped_waiting'),saved):
                queue=create_request_queue(self.policy);seen=set()
                for r in [r for r in original if r.request_id not in self.online_granted]+list(getattr(self,name)):
                    if r.request_id not in seen:queue.add_request(r);seen.add(r.request_id)
                setattr(self,name,queue)

    def _continuum_complete_policy(self,request):
        super()._continuum_complete_policy(request)
        if self.online_mode=='off':return
        p=self.program(request);self.online_outputs[p].append(len(request.output_token_ids))
        progressive_keep=getattr(self,'suffix_current_keep',{}).get(p,0)
        if getattr(self,'scalar_rank_policy',None) in ('learned','learnedgated'):
            from kairos.causal import service
            keep=self.learned_current_keep.get(p,0)
            self.learned_service_total[p]=self.learned_service_total.get(p,0.)+service(
                request.num_prompt_tokens,len(request.output_token_ids),keep,
                self.compact_predictor.model['profile'])
            self.learned_previous_hashes[p]=tuple(getattr(request,'block_hashes',()))
            self.learned_current_keep.pop(p,None)
        if getattr(self,'scalar_rank_policy',None) in SUFFIX_CDF_POLICIES:
            from kairos.causal import service
            keep=progressive_keep
            self.suffix_service_total[p]=self.suffix_service_total.get(p,0.)+service(
                request.num_prompt_tokens,len(request.output_token_ids),keep,
                self.suffix_predictor.model.get('profile',{}))
            self.suffix_previous_hashes[p]=tuple(getattr(request,'block_hashes',()))
            self.suffix_current_keep.pop(p,None)
        if getattr(self,'scalar_rank_policy',None) in PROGRESSIVE_POLICIES:
            from kairos.causal import service
            # The call has completed, so its realized service is now public
            # scheduler state. It is used as debt, never as future truth.
            realized=service(request.num_prompt_tokens,len(request.output_token_ids),progressive_keep,
                self.suffix_predictor.model.get('profile',{}) if self.suffix_predictor else self.profile)
            self.progressive_observed_s[p]=self.progressive_observed_s.get(p,0.)+realized
        self.online_cleaning.setdefault(p,set()).add(request.request_id)
        self.online_requests.pop(p,None);self.online_granted.discard(request.request_id)
        if self.metadata(request).get('cont_final'):self.online_ended.add(p)
        self.online_dirty=True

    def _free_request_blocks(self,request):
        result=super()._free_request_blocks(request)
        if self.online_mode!='off' and request.is_finished():
            p=self.program(request)
            pending=self.online_cleaning.get(p,set());pending.discard(request.request_id)
            if not pending:self.online_cleaning.pop(p,None)
            # An older response can finish DMA after a newer call arrived,
            # adopted the optional pin and started running. Its cleanup may
            # NEVER erase the newer call's reservation or membership.
            if p not in self.online_requests and (p in self.online_ended or p not in self.pins):
                self.online_reserved.pop(p,None);self.online_active.discard(p)
                self.online_insurance.pop(p,None)
            # Logical completion already revised the residual path. A later
            # cleanup with unchanged capacity and no new candidate adds no
            # decision. Actual capacity release remains an immediate event.
            self.online_dirty=self.online_dirty or p not in self.online_reserved
            self._emit('online_path_cleanup',program=p,ended=p in self.online_ended,
                reserved_pages=sum(self.online_reserved.values()))
        return result
