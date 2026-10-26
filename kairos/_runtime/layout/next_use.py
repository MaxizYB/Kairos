"""Kairos next use implementation."""
from dataclasses import dataclass
import math


USE_MODES = ('gapped-use', 'gapped-survival', 'gapped-use-survival', 'gapped-enst',
             'gapped-use-discount','gapped-use-discount-survival',
             'gapped-cpu-use','gapped-cpu-insurance')
AUDIT_MODES = ('gapped-return',) + USE_MODES
CPU_TIMED_MODES = ('gapped-cpu-use','gapped-cpu-insurance')


def retention_weight(program, mode, tier='gpu'):
    """Optional asymmetric horizons: GPU Return, CPU all-time discount.

    Discount weights are not probabilities. The CPU-only pair isolates the
    observed long-tool insurance opportunity from changes in GPU ranking.
    """
    if mode.startswith('gapped-use-discount') or tier=='cpu' and mode in CPU_TIMED_MODES:
        return sum(1./(1.+t) for t in program.use_times)/len(program.use_times)
    if mode=='gapped-enst':return 1./(1.+sum(program.use_times)/len(program.use_times))
    return program.use_probability if mode in ('gapped-use','gapped-use-survival') else program.probability


class InsuranceEstimator:
    """Action-conditioned, delay-bucketed next-lookup availability.

One chunk probe per tier/program resolves before the next restore. Labels
describe this policy's trajectory, including reinsertions; not counterfactual
causal probabilities. Sparse buckets use explicit Return fallback.
"""
    def __init__(self, minimum=8, conservative=True):
        self.minimum=minimum
        self.conservative=conservative
        self.pending={}
        self.buckets={}

    @staticmethod
    def bucket(delay):
        return next((i for i,edge in enumerate((.1,.3,1.,3.)) if delay<=edge),4)

    def track(self,program,chunk,delay,protected=0.):
        # Predicted use inside a pin window is not a censoring condition:
        # actual return may be later, after protection and GPU coverage end.
        # Keep the real next-lookup label. Scoring inside a still-valid
        # protection window remains one, and training/query delay features
        # retain the same definition.
        self.pending.setdefault((program,chunk),self.bucket(delay))

    def resolve(self,program,present_chunks):
        labels=[]
        for (p,i),bucket in list(self.pending.items()):
            if p!=program:continue
            del self.pending[(p,i)]
            count,survived=self.buckets.get(bucket,(0,0))
            hit=i in present_chunks
            self.buckets[bucket]=(count+1,survived+hit)
            labels.append(dict(chunk=i,bucket=bucket,survived=hit))
        return labels

    def probability(self,delay,protected=0.):
        if delay<=protected:return 1.,False
        count,survived=self.buckets.get(self.bucket(delay),(0,0))
        if count<self.minimum:return 1.,True
        if self.conservative:
            # Use a Wilson upper score for availability: a stable tier with
            # no observed failures should not acquire fictitious risk solely
            # from Laplace pseudocounts. Correlated chunks mean this is an
            # uncertainty heuristic, not a statistical coverage guarantee.
            z=1.645;p=survived/count
            upper=(p+z*z/(2*count)+z*math.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count)
            return min(1.,upper),False
        return (survived+1)/(count+2),False

    def summary(self):
        return dict(kind='action-conditioned next-lookup availability',
                    buckets={str(b):dict(count=n,survived=s) for b,(n,s) in self.buckets.items()},
                    pending=len(self.pending),minimum=self.minimum,conservative=self.conservative)

    @classmethod
    def from_summary(cls,summary):
        result=cls(summary.get('minimum',8),summary.get('conservative',True))
        result.buckets={int(b):(v['count'],v['survived']) for b,v in summary.get('buckets',{}).items()}
        return result


def service_windows(programs, capacity_pages, sequence_budget, decode_step, query_step):
    """Three small, cache-independent admission scenarios; no JCT rollout.

Base program order resolves ready contenders. Running work retains its
service slot. Future requests reserve full-context memory, not query bytes.
Uniform prefill quantum avoids using cache coldness to postpone eligibility.
"""
    result = {p.name: [] for p in programs}
    for scenario in range(3):
        pending = list(programs)
        live = []
        now = 0.
        while pending:
            live = [(end, pages) for end, pages in live if end > now + 1e-9]
            ready = sorted((p for p in pending if p.returns[scenario] <= now + 1e-9),
                           key=lambda p: (p.phase != 'running', p.order, p.name))
            admitted = False
            for p in ready:
                pages = math.ceil(max(p.prompt_tokens, p.chunks * 256 + p.append) / 16)
                if pages > capacity_pages:
                    # Impossible context is not made feasible by the forecast.
                    result[p.name].append(max(now, 1e6))
                    pending.remove(p)
                    admitted = True
                    continue
                if len(live) >= sequence_budget or sum(n for _, n in live) + pages > capacity_pages:
                    continue
                result[p.name].append(now)
                duration = max(.001, p.output * decode_step +
                               (query_step if p.computed < p.prompt_tokens else 0.))
                live.append((now + duration, pages))
                pending.remove(p)
                admitted = True
            if pending:
                events = [end for end, _ in live if end > now + 1e-9]
                events += [p.returns[scenario] for p in pending if p.returns[scenario] > now + 1e-9]
                if not events:
                    raise RuntimeError('Next-service forecast made no progress')
                now = min(events)
    return {name: tuple(times) for name, times in result.items()}


class SurvivalEstimator:
    """Object-time hazard measured under this arm, reset at every replay.

Loss events are correlated; require several distinct pressure events, not
thousands of pages from one event. Finished owners are censored. The model
is an exponential approximation, not a guarantee or causal counterfactual.
"""
    def __init__(self, minimum_events=4):
        self.minimum_events = minimum_events
        self.previous = {}
        self.time = None
        self.exposure = 0.
        self.losses = 0
        self.events = 0

    def observe(self, objects, now, active_owners):
        objects = dict(objects)
        previous = {k: owner for k, owner in self.previous.items() if owner in active_owners}
        if self.time is not None:
            self.exposure += max(0., now - self.time) * len(previous)
            lost = set(previous) - set(objects)
            self.losses += len(lost)
            self.events += bool(lost)
        self.previous = objects
        self.time = now

    def remove(self, keys, now):
        removed=set(keys)
        current = {k: owner for k, owner in self.previous.items() if k not in removed}
        self.observe(current, now, set(self.previous.values()))

    def probability(self, delay, protected=0.):
        if delay <= protected or delay <= 0.:
            return 1., False
        if self.events < self.minimum_events or self.exposure <= 0.:
            # Exact Return fallback; this is NOT evidence of future survival.
            return 1., True
        return math.exp(-min(40., self.losses / self.exposure * (delay - protected))), False

    def summary(self):
        return dict(events=self.events, losses=self.losses, exposure_s=self.exposure,
                    hazard=self.losses / self.exposure if self.exposure else None,
                    fallback=self.events < self.minimum_events)

    @classmethod
    def from_summary(cls, summary):
        result = cls()
        result.events = summary.get('events', 0)
        result.losses = summary.get('losses', 0)
        result.exposure = summary.get('exposure_s', 0.)
        return result


@dataclass(frozen=True)
class UseInfo:
    times: tuple = (0., 0., 0.)
    probability: float = 1.
    return_probability: float = 1.
    protected_s: float = 0.
    all_times: bool = False

    def survival(self, estimator, present, protected=0.):
        if not present:
            return 0., False
        # Average only the predicted uses inside the same one-second window.
        times = list(self.times) if self.all_times else [t for t in self.times if t <= 1.]
        if not times:
            return 1., True
        if isinstance(estimator, InsuranceEstimator):
            # Probes are bucketed by the predicted distribution's mean.
            # Query that same feature, not each scenario's unrelated bucket.
            estimate=estimator.probability(sum(self.times)/len(self.times))
            values=[(1.,False) if t<=protected else estimate for t in times]
        else:
            values = [estimator.probability(t, protected) for t in times]
        weights=[1./(1.+t) if self.all_times else 1. for t in times]
        return sum(w*v for w,(v,_) in zip(weights,values))/sum(weights), any(f for _, f in values)


def gpu_value(info, compute, load, cpu_present, cpu_estimator, survival):
    s, fallback = (info.survival(cpu_estimator, cpu_present) if survival
                   else (float(cpu_present), False))
    return info.probability * (s * min(load, compute) + (1. - s) * compute), s, fallback


def cpu_value(info, compute, load, duplicate, gpu_estimator, survival, protected=0.):
    s, fallback = (info.survival(gpu_estimator, duplicate, protected) if survival
                   else (float(duplicate), False))
    return info.probability * (1. - s) * max(0., compute - load), s, fallback
