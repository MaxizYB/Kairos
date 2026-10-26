"""Kairos dispatch implementation."""
from concurrent.futures import ThreadPoolExecutor
import math
import os
import time

from kairos._runtime.compat.commitment_scheduler import CommitmentScheduler
from kairos.resources import PredictorResources
from kairos.arrival import ArrivalPredictor
from kairos._runtime.compat.online_path import known_request_output


class ForecastDispatchScheduler(CommitmentScheduler):
    VERSION = 'kairos'
    SCORE = 'standard current service s + conditional forest mean E[R+]'
    RANKING = 'mature ready slots: standard current + forest conditional mean'
    # Reuse static resources and causal input serialization only. In particular,
    # do NOT inherit R82's _online_grant or _refresh_priority overrides.
    _suffix_prepare_resources = PredictorResources._suffix_prepare_resources
    _suffix_observation = PredictorResources._suffix_observation
    _suffix_service = PredictorResources._suffix_service

    def _emit(self, event, **kw):
        if event == 'r5_commitment_decision' and getattr(self, 'forecast_enabled', False):
            kw.update(forecast_version=self.VERSION,
                ranking=self.RANKING,
                returning_rank=self.SCORE+'; unready incumbent slots',
                dispatch_rank='original call-commitment protocol queue with mature estimator refinement',
                victim_rank='unchanged call-commitment protocol incumbent owner order',
                posterior_order=self.SCORE+' only in prediction-ready mature slots')
        return super()._emit(event, **kw)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('CONTINUUM_R83_PRELOAD') == '1':
            self._suffix_prepare_resources(phase='service-startup')

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        executor = getattr(self, 'forecast_executor', None)
        if executor is not None:
            executor.shutdown(wait=True)
        super()._switch_experiment(name)
        self.forecast_enabled = self.VERSION in name.split('~')
        if not self.forecast_enabled:
            return
        if (not self.r5_commitment or not self.r5_use_distribution_order
                or not self.online_known_output or self.r5_mean_rank
                or self.r5_fifo or self.r5_action_q or self.scalar_rank_policy is not None):
            raise ValueError('Kairos requires the original call-commitment protocol lifecycle and rank only')
        self.suffix_reference, self.suffix_forest, profile_sha = self._suffix_prepare_resources()
        if self.suffix_forest.training_ids & set(self.online_prior.get('evaluation_ids', ())):
            raise ValueError('Forest training/evaluation overlap')
        self.forecast_worker = ArrivalPredictor(self.suffix_forest, self._suffix_service, self.block_size)
        self.forecast_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='causal-forecast')
        self.forecast_pending = []
        self.forecast_arrivals, self.forecast_predictions = {}, {}
        self._emit('forecast_registry', version=self.VERSION,
            score=self.SCORE,
            predictor='honest_cdf157; post-current standard solo service distribution',
            forest_sha256=self.suffix_forest.sha256, profile_sha256=profile_sha,
            training_evaluation_disjoint=True, future_information=False,
            static_resources_phase=self.suffix_resource_phase,
            lifecycle='original CommitmentScheduler._progressive_compete',
            ttl='original call-commitment protocol; incumbent continuity rank',
            victims='original call-commitment protocol owner order and legal reclaim',
            running='original call-commitment protocol/native; no grant-sequence override',
            forecast='FIFO frozen-arrival CPU worker; never gates admission',
            missing_prediction='retain incumbent slots; no cohort-wide reset',
            backend='unchanged v21')

    def add_request(self, request):
        super().add_request(request)
        if not getattr(self, 'forecast_enabled', False):
            return
        observation = self._suffix_observation(request)
        p, rid = observation['p'], observation['request_id']
        self.forecast_arrivals[p] = rid
        self.forecast_predictions.pop(p, None)
        self.forecast_pending.append(self.forecast_executor.submit(self.forecast_worker.observe, observation))
        self._emit('r83_observation', program=p, request=rid,
            stage=len(observation['outputs']), frozen_at_arrival=True, future_information=False)

    def _forecast_collect(self):
        while self.forecast_pending and self.forecast_pending[0].done():
            value = self.forecast_pending.pop(0).result()
            obs = value['observation']
            p, rid = obs['p'], obs['request_id']
            current = self.forecast_arrivals.get(p) == rid
            post = value['prediction']
            self._emit('forecast_features', program=p, request=rid, current=current,
                seconds=value['seconds'], inference_s=value['inference_s'])
            if current and post is not None:
                self.forecast_predictions[p] = dict(request=rid, keep=value['keep'], post=post)
                self._emit('suffix_prediction', program=p, request=rid,
                    stage=len(obs['outputs']), atoms_s=post['atoms'], mass=post['mass'],
                    pred_s=post['pred'], seconds=value['inference_s'],
                    feature_keep=value['keep'], service_seen_s=value['services'],
                    asynchronous=True, future_information=False)

    def _forecast_quotes(self, by, eligible):
        raise NotImplementedError("Concrete scheduler supplies the physical quote")

    def _r5_dispatch_refinement(self, by, original_order, quotes, entered, scores, order):
        if not getattr(self, 'forecast_enabled', False):
            return super()._r5_dispatch_refinement(by, original_order, quotes, entered, scores, order)
        started = time.perf_counter()
        self._forecast_collect()
        eligible = [p for p in order if p in entered and
                    self.forecast_predictions.get(p, {}).get('request') == by[p].request_id]
        details = {}
        if len(eligible) >= 2:
            details = self._forecast_quotes(by, eligible)
            for p in eligible:
                scores[p] = (details[p]['score'], self.order[p])
            refined = sorted(eligible, key=lambda p: scores[p])
            slots = iter(refined)
            eligible_set = set(eligible)
            order = [next(slots) if p in eligible_set else p for p in order]
            slots = iter(refined)
            owners = [next(slots) if p in eligible_set else p for p in original_order]
            self.online_rank = {p: n for n, p in enumerate(owners)}
        self._emit('forecast_dispatch', before=eligible,
            after=[p for p in order if p in eligible], quotes=details,
            request_ids={p: by[p].request_id for p in entered},
            changed=eligible != [p for p in order if p in eligible],
            candidate_order=list(order), victim_rank=dict(self.online_victim_rank),
            seconds=time.perf_counter()-started, future_information=False)
        return scores, order

    def _r5_continuity_fraction(self, request):
        if not getattr(self, 'forecast_enabled', False):
            return super()._r5_continuity_fraction(request)
        saved = self.online_rank
        try:
            self.online_rank = self.online_victim_rank or saved
            return super()._r5_continuity_fraction(request)
        finally:
            self.online_rank = saved
