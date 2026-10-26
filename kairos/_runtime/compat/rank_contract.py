"""Kairos rank contract implementation."""
import math
import statistics


RANK_TAGS = {'rankfcfs': 'fcfs', 'rankcurrent': 'current', 'rankverified':'verified',
             'rankwarm': 'warm',
             'rankgated': 'gated',
             'ranklearned': 'learned', 'ranklearnedgated': 'learnedgated',
             'ranktruthcold': 'truthcold', 'ranktruthwarm': 'truthwarm',
             'ranktruth': 'truth', 'rankaudit': 'predicted', 'rankstage': 'stage',
             'rankcdf': 'cdfmean', 'rankcdfindex': 'cdfindex',
             'rankcdfwarm': 'cdfwarm', 'rankadaptive': 'adaptive',
             'rankprogressive': 'progressive',
             'rankprogressivedebt': 'progressivedebt',
             'rankprogressivewindow': 'progressivewindow'}


def selected_policy(tags):
    chosen = set(tags) & RANK_TAGS.keys()
    if len(chosen) > 1:
        raise ValueError('Choose exactly one scalar rank intervention')
    return RANK_TAGS[next(iter(chosen))] if chosen else None


def phase_policy(policy, completed):
    """Isolate oracle information by observed history, never by future stage.

    These diagnostic arms use exactly the incumbent outside the declared
    phase. First-call running/waiting states are still cold until completion.
    The two masks partition every arrived/idle state of the full truth arm.
    """
    if completed < 0:
        raise ValueError('Completed call count cannot be negative')
    if policy == 'truthcold':
        return 'truth' if completed == 0 else 'predicted'
    if policy == 'truthwarm':
        return 'truth' if completed > 0 else 'predicted'
    return policy


def rank_score(policy, *, current, future, arrival_order, truth_future=None):
    for x in (current, future):
        if not math.isfinite(x) or x < 0:
            raise ValueError('Rank work must be finite nonnegative solo service')
    if policy == 'fcfs':
        return float(arrival_order)
    if policy == 'current':
        return current
    if policy == 'truth':
        if truth_future is None or not math.isfinite(truth_future) or truth_future < 0:
            raise ValueError('Missing diagnostic future service')
        return current + truth_future
    if policy in ('predicted','stage','verified','warm','gated','learned','learnedgated','cdfmean','cdfindex','cdfwarm','adaptive','progressive','progressivedebt','progressivewindow'):
        return current + future
    raise ValueError('Unknown rank intervention: ' + str(policy))


class StageWorkPrior:
    """A strong population-only control, with no individual neighbor features.

    All labels come from explicitly verified complete training paths. The
    current arrived call conditions survival; ended training paths cannot
    be used as zero-sized examples at an impossible attained stage.
    """
    def __init__(self,prior):
        from kairos._runtime.compat.online_path import split_prior
        if prior.get('schema_version')!=2:
            raise ValueError('Stage control requires audited training endpoints')
        self.training_ids=set(prior['heldout_ids'])
        paths=split_prior(prior)
        self.paths=tuple(p for p in paths if all(r['prompt_tokens']+r['output_tokens']<=16384 for r in p))
        if not self.paths:raise ValueError('No context-compatible stage training paths')
        self.cache={}

    def future_work(self,completed,*,idle,beta,step):
        if completed<0 or (idle and completed==0):raise ValueError('Invalid observed stage')
        key=(completed,idle,tuple(beta),step)
        if key not in self.cache:
            start=completed if idle else completed+1
            # An arrived current call k, or known nonterminal completed call
            # k-1 in its tool gap, implies a training path with a call k.
            paths=[p for p in self.paths if len(p)>completed]
            if not paths:
                # Out of support: reuse the final observed transition, not
                # an invented terminal event. This is a marked finite prior.
                paths=self.paths;start=None
            values=[]
            for path in paths:
                indices=(range(start,len(path)) if start is not None else [len(path)-1])
                total=0.
                for i in indices:
                    t=path[i];n=t['prompt_tokens'];keep=min(n-1,path[i-1]['prompt_tokens']) if i else 0
                    q=n-keep
                    total+=max(0.,beta[1]*q+beta[2]*q*(n+keep+1)/2)+t['output_tokens']*step
                values.append(total)
            self.cache[key]=statistics.median(values)
        return self.cache[key]


class ScalarTruth:
    def __init__(self, data, trace_sha):
        if (data.get('future_information') is not True or data.get('deployable') is not False
                or data.get('trace_sha256') != trace_sha):
            raise ValueError('Scalar truth must disclose its exact trace and future information')
        self.values = data.get('suffix_s',{})
        self.terms = data.get('service_terms')
        self.stages = ({p:len(rows) for p,rows in self.terms.items()} if self.terms is not None
                       else {p:len(rows) for p,rows in self.values.items()})

    def after_current(self, program, completed, *, idle=False, beta=None, step=None):
        # In a tool gap there is no current arrived call: all work after the
        # last COMPLETED call belongs to future work (including the next call).
        stage = completed - int(idle)
        if stage < 0 or program not in self.stages or stage >= self.stages[program]:
            raise ValueError('No truth for this arrived stage')
        if self.terms is not None:
            if beta is None or step is None:
                raise ValueError('Runtime scalar truth needs identical current service pricing')
            # Only a scalar escapes this diagnostic object. This accommodates
            # each run's independent service calibration without giving the
            # controller future memory, tool events, or individual turns.
            value = math.fsum(max(0.,beta[1]*q+beta[2]*att)+step*out
                for q,att,out in self.terms[program][stage+1:])
        else:value = float(self.values[program][stage])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid truth service')
        return value
