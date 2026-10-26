"""Kairos pcv implementation."""
from dataclasses import dataclass, replace
import math

from kairos._runtime.layout.plan import Costs, query_segments


def conditional_returns(history, tool, age, fallback=1.):
    samples = [float(row[0]) for row in history.get(tool, ()) if row[0] > age]
    if len(samples) < 8:
        samples = [float(row[0]) for rows in history.values()
                   for row in rows if row[0] > age]
    if not samples:
        # No observed survivor is not proof that the program will never return.
        return (max(.01, fallback),) * 3
    samples.sort()
    return tuple(max(.001, samples[int(q * (len(samples)-1))] - age)
                 for q in (.2, .5, .8))


def return_probability(history, tool, age, horizon):
    samples = [float(row[0]) for row in history.get(tool, ())]
    if len(samples) < 8:
        samples = [float(row[0]) for rows in history.values() for row in rows]
    survivors = [x for x in samples if x > age]
    return sum(x <= age + horizon for x in survivors) / len(survivors) if survivors else .5


def asym_frequency(age, lifespan, slope_ratio=40., turning_value=.99,
                   observed_lifespan=None):
    """AsymCache Eq. 9: turning point at P99, steeper second slope.

    The published tool correction's numerical coefficient is unspecified.
    Serial live tool programs get correction=1 (all certainly continue).
    """
    alpha = max(lifespan, .001) / -math.log(turning_value)
    beta = alpha / slope_ratio
    tau0 = max(lifespan, .001) * (1 - beta / alpha)
    # Eq. 10 applies lambda to the second tree, moving the crossing to
    # the observed sliding-window lifespan without changing either slope.
    log_lambda = (0. if observed_lifespan is None else
                  (observed_lifespan-tau0)/beta-observed_lifespan/alpha)
    return math.exp(max(-700., min(-age / alpha,
                                 log_lambda-(age-tau0)/beta)))


@dataclass(frozen=True)
class Program:
    name: str
    chunks: int
    gpu: frozenset
    cpu: frozenset
    returns: tuple = (0., 0., 0.)
    order: int = 0
    output: int = 24
    append: int = 128
    phase: str = 'tool'
    protected_s: float = 0.
    probability: float = 1.
    age: float = 0.
    lifespan: float = 1.
    computed: int = 0
    predicted_pressure_pages: int = 0
    prompt_tokens: int = 0
    protected_pages: frozenset = frozenset()
    observed_lifespan: float | None = None
    forced_loads: frozenset | None = None
    recovery_delay: float = 0.
    use_times: tuple = (0., 0., 0.)
    use_probability: float = 1.
    cpu_next_times: tuple | None = None
    cpu_continues: bool = True


@dataclass(frozen=True)
class Continuation:
    costs: Costs
    capacity_pages: int
    token_budget: int = 2048
    sequence_budget: int = 8
    decode_step: float = .0174
    fixed_step: float = .0035
    adaptive_decode_threshold: int = 0
    adaptive_token_budget: int = 512
    decode_bulk: bool = True
    decode_by_width: tuple = ()
    forecast_idle_overwrites: bool = True
    query_budget: bool = False
    # Unprotected GPU pages remain valid in the snapshot. Their later loss is
    # simulated only if allocation demand exceeds the available capacity.

    def objective(self, programs, cache=None):
        if not programs:
            return 0.
        if cache is None:
            cache={}
        return sum(self._scenario(programs,s,cache) for s in range(3))/3

    def local_sensitivities(self, programs):
        """Forward delay derivatives with the snapshot event topology fixed.

        A synchronous delay injected at a program's first service shifts
        subsequent completion events, until an idle gap absorbs it. Branch
        changes at arrivals/admission boundaries are outside this derivative.
        """
        result = {p.name: 0. for p in programs}
        cache = {}
        for scenario in range(3):
            slopes = {p.name: 0. for p in programs}
            self._scenario(programs, scenario, cache, slopes)
            for name, value in slopes.items():
                result[name] += value / 3
        return result

    def _scenario(self, programs, scenario, cache, sensitivities=None):
        rows = []
        for p in programs:
            end = p.prompt_tokens or p.chunks*256+p.append
            pages=math.ceil((end+p.output)/16)
            if pages > self.capacity_pages:
                raise ValueError('A program does not fit full-attention capacity')
            rows.append(dict(p=p, arrival=p.returns[scenario], gpu=set(p.gpu),
                valid=set(p.gpu), loads=set(), loaded=set(), end=end,
                cursor=min(p.computed,end), decode=max(1,p.output),
                admitted=p.phase=='running', finish=None, pages=pages,
                pending_delay=p.recovery_delay,
                pending_sensitivity=sensitivities is not None))
        def plan(r):
            eligible={i for i in r['p'].cpu if i*256 >= r['cursor']}
            if r['p'].forced_loads is not None:
                r['loads']=set(r['p'].forced_loads)&eligible
                r['valid']=set(r['gpu'])
                return
            key=('load',frozenset(r['gpu']),frozenset(eligible),r['p'].chunks)
            if key not in cache:
                cache[key]=self.costs.estimate_pages(r['gpu'],eligible,r['p'].chunks)[1]
            r['loads']=cache[key]
            r['valid']=set(r['gpu'])
        for r in rows:
            if r['admitted']:plan(r)
        t=0.; total=0.; iterations=0
        time_slopes = {p.name: 0. for p in programs} if sensitivities is not None else None

        def finished():
            if sensitivities is not None:
                for name, value in time_slopes.items():
                    sensitivities[name] += value
        while any(r['finish'] is None for r in rows):
            iterations += 1
            if iterations > 12000:
                raise RuntimeError('Continuation simulation exceeded its bound')
            running=[r for r in rows if r['admitted'] and r['finish'] is None]
            used=sum(r['pages'] for r in running)
            ready=sorted((r for r in rows if not r['admitted'] and r['arrival'] <= t),
                         key=lambda r:(r['p'].order, r['p'].name))
            for r in ready:
                if len(running) >= self.sequence_budget:
                    break
                idle=[x for x in rows if x is not r and not x['admitted']
                      and x['finish'] is None]
                pinned=sum(len(x['gpu'] & set(x['p'].protected_pages))
                    for x in idle if x['p'].protected_s > t)
                if used+r['pages']+pinned > self.capacity_pages:
                    continue
                # Only actual allocation pressure overwrites idle pages.
                # Expiry by itself leaves them present. LRU age is a bounded
                # forecast, not an assertion about the future native victim.
                deficit=max(0,used+r['pages']+sum(len(x['gpu']) for x in idle)
                            -self.capacity_pages)
                for x in (sorted(idle,key=lambda x:(-x['p'].age,x['p'].order))
                          if self.forecast_idle_overwrites else ()):
                    protected=(set(x['p'].protected_pages)
                               if x['p'].protected_s > t else set())
                    victims=sorted(x['gpu']-protected)[:deficit]
                    x['gpu'].difference_update(victims);deficit-=len(victims)
                    if not deficit:break
                r['admitted']=True;running.append(r);used+=r['pages'];plan(r)
            if not running:
                future=[r['arrival'] for r in rows if r['finish'] is None and r['arrival'] > t]
                future += [r['p'].protected_s for r in rows
                           if not r['admitted'] and r['p'].protected_s > t]
                if not future:
                    raise RuntimeError('Continuation capacity deadlock')
                t=min(future)
                if time_slopes is not None:
                    time_slopes = {p.name: 0. for p in programs}
                continue
            decode_count=sum(r['cursor']==r['end'] for r in running)
            budget=(min(self.token_budget,self.adaptive_token_budget)
                    if self.adaptive_decode_threshold and
                    decode_count>self.adaptive_decode_threshold else self.token_budget)
            if (self.decode_bulk and decode_count==len(running) and budget>=len(running)
                    and not any(r['pending_delay'] for r in running)):
                # No layout or batch change until a completion, arrival or
                # protection expiry. Advance identical decode iterations in
                # one operation, preserving the step-boundary admission rule.
                step=max(dict(self.decode_by_width).get(len(running),self.decode_step),1e-6)
                steps=min(r['decode'] for r in running)
                future=[r['arrival'] for r in rows if not r['admitted']
                        and r['finish'] is None and r['arrival']>t]
                future += [r['p'].protected_s for r in rows if not r['admitted']
                           and r['finish'] is None and r['p'].protected_s>t]
                if future:
                    steps=min(steps,max(1,math.ceil((min(future)-t)/step)))
                if time_slopes is not None:
                    for r in running:
                        if r['pending_sensitivity']:
                            time_slopes[r['p'].name] += 1.
                            r['pending_sensitivity'] = False
                t+=steps*step
                for r in running:
                    r['decode']-=steps
                    if r['decode']<=0:
                        r['finish']=t;total+=t;finished()
                iterations+=steps-1
                continue
            Q=A=N=0;advanced=[];load=0.
            running.sort(key=lambda r:(r['p'].order,r['p'].name))
            # Match the prototype's logical-token scheduling: skipped queries
            # do not refill the native budget with extra unscheduled work.
            slots=self.sequence_budget-len(running)
            decoding_only=True
            for r in sorted(running,key=lambda r:(r['cursor']<r['end'],r['p'].order)):
                if not budget:
                    break
                load += r['pending_delay']
                r['pending_delay'] = 0.
                if time_slopes is not None and r['pending_sensitivity']:
                    time_slopes[r['p'].name] += 1.
                    r['pending_sensitivity'] = False
                if r['cursor']<r['end']:
                    decoding_only=False
                    begin=r['cursor'];end=min(r['end'],begin+budget)
                    priced_slots = 2 if slots > 0 else 1
                    if self.query_budget:
                        def charge(endpoint):
                            valid = r['valid'] | {j for i in r['loads'] if (i+1)*256<=endpoint
                                                 for j in range(i*16,(i+1)*16)}
                            key=('query-budget',begin,endpoint,frozenset(valid),priced_slots)
                            if key not in cache:
                                cache[key]=sum(b-a for a,b in query_segments(
                                    begin,endpoint,valid,slots=priced_slots))
                            return cache[key]
                        if charge(r['end']) <= budget:
                            end=r['end']
                        else:
                            low,high=end,r['end']
                            while low<high:
                                middle=(low+high+1)//2
                                if charge(middle)<=budget:low=middle
                                else:high=middle-1
                            end=low
                        budget_charge=charge(end)
                    else:
                        budget_charge=end-begin
                    loading={i for i in r['loads']-r['loaded'] if (i+1)*256<=end}
                    r['loaded'].update(loading)
                    load+=len(loading)*(self.costs.load_token*256+self.costs.load_call)
                    r['valid'].update(j for i in loading for j in range(i*16,(i+1)*16))
                    key=('segments',begin,end,frozenset(r['valid']),
                         priced_slots if self.query_budget else max(1,slots+1))
                    if key not in cache:
                        cache[key]=query_segments(begin,end,r['valid'],slots=key[-1])
                    segments=cache[key]
                    slots-=len(segments)-1
                    for a,b in segments:Q+=b-a;A+=(a+b+1)*(b-a)/2;N+=1
                    budget-=budget_charge;r['cursor']=end
                    if end==r['end']:r['decode']-=1
                else:
                    Q+=1;A+=r['end']+(r['p'].output-r['decode'])+1
                    N+=1;budget-=1;r['decode']-=1
                advanced.append(r)
            # The actual backend fences synchronous H2D before attention.
            t+=load
            step=(dict(self.decode_by_width).get(len(advanced),self.decode_step) if decoding_only else
                  self.fixed_step+self.costs.query*Q+self.costs.attention_pair*A+self.costs.segment*N)
            t+=max(step,1e-6)
            for r in advanced:
                if r['cursor']==r['end'] and r['decode'] <= 0:
                    r['finish']=t
                    total+=t;finished()
        return total

    def marginal(self, programs, name, gpu_remove=(), cpu_remove=(), cpu_add=()):
        changed=[replace(p, gpu=p.gpu-frozenset(gpu_remove),
                         cpu=(p.cpu-frozenset(cpu_remove))|frozenset(cpu_add))
                 if p.name==name else p for p in programs]
        return self.objective(changed)-self.objective(programs)


def select_groups(groups, need, loss, max_candidates=8):
    """Conditionally re-evaluate concrete bundles; no static score sum.

    loss receives all pages/chunks selected so far, preserving combination
    effects and telescoping accounting. A bounded shortlist is caller-owned.
    """
    selected=set(); evaluations=0
    while len(selected)<need:
        candidates=[set(g)-selected for g in groups if set(g)-selected]
        if not candidates:
            break
        candidates=candidates[:max_candidates]
        remaining=need-len(selected)
        # Do not evict a whole group when only one of its members is needed.
        candidates=[g if len(g)<=remaining else set(sorted(g)[:remaining]) for g in candidates]
        options=[]
        base=loss(selected)
        for g in candidates:
            evaluations+=1
            options.append(((loss(selected|g)-base)/len(g), min(g),g))
        selected.update(min(options,key=lambda x:x[:2])[2])
    return selected,evaluations
