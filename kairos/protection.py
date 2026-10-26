"""Kairos protection implementation."""
from collections import defaultdict
import math
import statistics

from kairos._runtime.compat.oracle_program_plan import Program, Turn
from kairos._runtime.compat.online_path import _check_history, _check_known_output, split_prior


class CommitmentPrior:
    protection_support = True
    neighbors = 0

    def __init__(self, prior):
        self.paths = list(split_prior(prior))
        self.tool_samples = defaultdict(list)
        for path in self.paths:
            for row in path:
                if not row.get('done') and row.get('tool_s') is not None:
                    self.tool_samples[row.get('tool', '')].append(float(row['tool_s']))
        self.global_tool_samples = tuple(x for values in self.tool_samples.values() for x in values)
        self.cache_hits = self.cache_misses = 0

    def population(self, inputs, outputs, after_completion=False):
        _check_history(inputs, outputs, after_completion)
        stage = len(inputs)-1
        rows = [path for path in self.paths if len(path) > stage+int(after_completion)]
        if not rows: rows = self.paths
        self.cache_misses += 1
        return rows

    def _turns(self, path, inputs, outputs, attained, current_cap, after_completion,
               max_len, known_current_output):
        stage = len(inputs)-1
        tail = path[stage:]
        if after_completion: tail = tail[1:]
        if not tail:
            return ()
        shift = inputs[-1]-path[stage]['prompt_tokens']
        turns=[]; previous=0
        for j,row in enumerate(tail):
            prompt=max(1,min(max_len-1,row['prompt_tokens']+shift))
            output=max(1,int(row['output_tokens']))
            if j==0:
                prompt=inputs[-1]
                if output <= attained: output=attained+1
                if current_cap is not None: output=min(output,max(attained+1,current_cap))
                if known_current_output is not None: output=known_current_output
            output=min(output,max(1,max_len-prompt))
            keep=min(prompt-1,previous) if j else 0
            turns.append(Turn(prompt,output,keep,float(row.get('tool_s') or 0.)))
            previous=prompt
        return tuple(turns)

    def forecast(self, name, inputs, outputs, *, attained=0, current_cap=None,
                 memory_pages=0, page=16, max_len=16384, after_completion=False,
                 known_current_output=None, **_):
        _check_known_output(known_current_output,inputs,attained,current_cap,max_len,after_completion)
        rows=self.population(inputs,outputs,after_completion)
        stage=len(inputs)-1
        turns=[self._turns(path,inputs,outputs,attained,current_cap,after_completion,max_len,known_current_output)
               for path in rows]
        turns=[x for x in turns if x]
        if not turns:
            # A censored/terminal empirical path cannot provide a future
            # envelope. Keep the controller causal and executable by using a
            # population median growth/output atom, rather than indexing an
            # empty path or inventing a complete future trajectory.
            prompt=min(max_len-1, max(1, inputs[-1]))
            output=max(1, int(statistics.median(
                [r[stage]['output_tokens'] for r in self.paths
                 if len(r)>stage] or [attained+1])))
            if current_cap is not None: output=min(output,max(attained+1,current_cap))
            if known_current_output is not None: output=known_current_output
            turns=[(Turn(prompt,min(output,max_len-prompt),0,0.),)]
        chosen=sorted(turns,key=lambda x:sum(t.output for t in x))[len(turns)//2]
        target=math.ceil((chosen[0].prompt+(current_cap or chosen[0].output))/page)
        peak=max(target,memory_pages,*[math.ceil((t.prompt+t.output)/page) for t in chosen])
        remaining=[sum(t.output for t in x) for x in turns]
        return Program(name,0.,chosen),dict(source_training_index=None,neighbors=len(rows),
            stage_tail_fallback=False,peak_pages=peak,current_pages=target,predicted_turns=len(chosen),
            predicted_output=chosen[0].output,predicted_remaining_service=statistics.fmean(remaining),
            current_output_source='contractual' if known_current_output is not None else 'population')

    def scenarios(self,*args,**kwargs):
        # Scenarios preserve joint path samples for the optional diagnostic;
        # the commitment controller consumes only conditional means.
        name,inputs,outputs=args[:3]
        rows=self.population(inputs,outputs,kwargs.get('after_completion',False))
        result=[]
        for path in rows:
            turns=self._turns(path,inputs,outputs,kwargs.get('attained',0),kwargs.get('current_cap'),
                               kwargs.get('after_completion',False),kwargs.get('max_len',16384),
                               kwargs.get('known_current_output'))
            if turns: result.append(Program(name,0.,turns))
        return result

    def idle_bound(self, tool):
        values=self.tool_samples.get(tool) or self.global_tool_samples
        return max(.001,max(values,default=1.))

    def idle_remaining(self, tool, age):
        values=self.tool_samples.get(tool) or self.global_tool_samples
        residual=[v-age for v in values if v>age]
        return max(.001,statistics.fmean(residual or values or [1.]))


class CommitmentHoldPolicy:
    """Observed tool-return distribution, with completion-time benefit quote."""
    mode='r5-conditional-hold'
    fixed=0.0
    k=0
    def __init__(self, history=None, training=None, *, allow_live_tail=False):
        self.allow_live_tail=bool(allow_live_tail)
        self.history=defaultdict(list)
        self.prior_history=defaultdict(list)
        self.live_history=defaultdict(list)
        # The first live return has no samples yet.  Seed only the tool-gap
        # marginal from the disjoint training prior; live observations are
        # appended later and remain the only source for deployment updates.
        for key, values in (training or {}).items():
            rows=[(float(value), 0) for value in values]
            self.history[key].extend(rows)
            self.prior_history[key].extend(rows)
        for key, rows in (history or {}).items():
            copied=[(float(x[0]),int(x[1])) for x in rows]
            self.history[key].extend(copied)
            self.prior_history[key].extend(copied)
        for key in tuple(self.history):
            self.history[key]=self.history[key][-512:]
            self.prior_history[key]=self.prior_history[key][-512:]
        self.global_prior=tuple(x for values in self.prior_history.values() for x in values)

    def observe(self,tool,duration,added_tokens):
        row=(max(0.,float(duration)),int(added_tokens))
        self.history[tool].append(row)
        self.live_history[tool].append(row)
        self.history[tool]=self.history[tool][-512:]
        self.live_history[tool]=self.live_history[tool][-512:]

    def _hold_samples(self, tool):
        """Causal empirical-Bayes support for the optional hold decision.

        A live tool label may be unseen in the deployment prior.  One observed
        interval then updates the population posterior; it does not become a
        one-point replacement distribution.  Tool-specific prior samples are
        preferred when available, while an unseen label is shrunk to the
        global prior plus its live observations.
        """
        prior=self.prior_history.get(tool, ())
        live=self.live_history.get(tool, ())
        if self.allow_live_tail:
            # Retain prior pseudo-observations, but do not assign zero
            # probability forever to a repeatedly observed deployment tail.
            # The existing expected-hold cost still decides whether to keep
            # a long lease; observing a long interval does not mandate one.
            return tuple(prior or self.global_prior)+tuple(live)
        if prior:
            # Online observations update probability mass inside the
            # deployment support, but an uncalibrated live tail does not
            # manufacture a new protection deadline. A support extension
            # would turn one long tool call into a multi-second lease before
            # its frequency is established. Conditional residual estimates
            # still retain every live observation through conditional().
            upper=max(x[0] for x in prior)
            return tuple(prior)+tuple(x for x in live if x[0] <= upper)
        population=tuple(self.global_prior)
        if population:
            upper=max(x[0] for x in population)
            return population+tuple(x for x in live if x[0] <= upper)
        return tuple(self.history.get(tool, ()))

    def conditional(self,tool,elapsed):
        rows=[x for x in self.history[tool] if x[0]>elapsed]
        if not rows: rows=[x for values in self.history.values() for x in values if x[0]>elapsed]
        return [(max(.001,x[0]-elapsed),x[1]) for x in sorted(rows)[:3]] or [(.001,256)]*3

    def ttl(self,tool,benefit,oracle_gap=None,opportunity_rate=1.0):
        rows=sorted(x[0] for x in self._hold_samples(tool))
        if not rows:return 0.,'r5-no-return-evidence'
        if not math.isfinite(opportunity_rate) or opportunity_rate < 0:
            raise ValueError('Opportunity rate must be finite and nonnegative')
        # The only finite candidates are observed event boundaries; between
        # them protection costs increase while return probability is constant.
        best=(0.,0.)
        # A sorted sweep visits each distinct empirical return boundary once.
        # Prefix sums give H(t)=(sum_{d<=t} d + (n-k)t)/n exactly on that
        # support. Stable strict improvement keeps the earliest tied TTL.
        prefix=0.;n=len(rows);index=0
        while index<n:
            tau=rows[index]
            while index<n and rows[index]==tau:
                prefix+=rows[index];index+=1
            # Holding time is only a cost when an observed ready frontier can
            # actually use the capacity.  The scheduler supplies the
            # completion-area marginal rate for that state; it is zero when
            # no competing call is currently blocked and rises continuously
            # with the physical capacity deficit.  This avoids charging a
            # pinned page one second of JCT merely because the owner waits in
            # a tool.
            # All candidates have the same denominator n. Compare their
            # numerators so equal empirical utilities do not acquire a
            # spurious preference for a later deadline from two divisions.
            utility=index*max(0.,benefit)-opportunity_rate*(prefix+(n-index)*tau)
            if utility>best[1]:best=(tau,utility)
        return best[0],'r5-completion-area'
