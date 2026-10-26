"""Kairos online path implementation."""
from collections import OrderedDict,defaultdict
import math
import statistics
import time

from kairos._runtime.compat.oracle_program_plan import Program, Turn
from kairos._runtime.compat.path_calendar import simulate_calendar, admission_gates


def split_prior(prior):
    ids = prior['heldout_ids']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Training identities must be nonempty and unique')
    if prior.get('schema_version', 1) not in (1, 2):
        raise ValueError('Unsupported path prior schema')
    if set(ids) & set(prior['evaluation_ids']): raise ValueError('Prior/evaluation overlap')
    if prior.get('schema_version') == 2:
        metadata = prior['program_metadata']
        if set(metadata) != set(ids): raise ValueError('Training provenance identity mismatch')
        for name in ids:
            source = metadata[name]
            if source.get('endpoint') != 'Submitted' or source.get('censored') is not False:
                raise ValueError('Censored or unverified endpoint in complete-path prior')
            digest = source.get('source_sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Missing raw training provenance')
    paths = []
    for row in prior['rows']:
        if row.get('censored'): raise ValueError('Censored path in complete-path prior')
        if row['stage'] == 0: paths.append([])
        if not paths or row['stage'] != len(paths[-1]): raise ValueError('Broken prior path')
        if paths[-1] and paths[-1][-1]['done']:
            raise ValueError('Rows after a terminal training state')
        for key in ('prompt_tokens', 'output_tokens'):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError('Training token counts must be positive integers')
        gap = row.get('tool_s')
        if gap is not None and (not isinstance(gap, (int, float)) or
                                not math.isfinite(gap) or gap < 0):
            raise ValueError('Tool duration must be finite and nonnegative')
        paths[-1].append(row)
    if len(paths) != len(ids) or any(not p[-1]['done'] for p in paths):
        raise ValueError('Incomplete prior provenance')
    return tuple(tuple(p) for p in paths)


def _check_history(inputs, outputs, after_completion):
    if not inputs or len(outputs) != len(inputs)-1+int(after_completion):
        raise ValueError('Observed outputs must match the before/after completion boundary')
    if any(type(v) is not int or v <= 0 for v in (*inputs, *outputs)):
        raise ValueError('Observed token counts must be positive integers')


def _check_known_output(value, inputs, attained, current_cap, max_len, after_completion):
    if value is None:
        return
    if (after_completion or type(value) is not int or value < 1 or value < attained or
        value > max_len-inputs[-1] or (current_cap is not None and value > current_cap)):
        raise ValueError('Known current output must be an attainable contractual total')


def known_request_output(request):
    """A public cap is exact only when the request disables every early stop.

    This reads the arrived request contract, never a stored future label.
    Ordinary online max_tokens remains an upper bound, including when EOS
    is disabled but a stop string/token is configured.
    """
    params = getattr(request, 'sampling_params', None)
    if (params is not None and getattr(params, 'ignore_eos', False) is True and
        not getattr(params, 'stop', None) and not getattr(params, 'stop_token_ids', None)):
        return request.max_tokens
    return None


class PathForecast:
    def __init__(self, prior, neighbors=8, fast=False, current_output_context=False):
        self.paths = split_prior(prior); self.neighbors = neighbors
        if type(neighbors) is not int or neighbors < 1: raise ValueError('Invalid neighbor count')
        self.protection_support = prior.get('protection_support', True)
        self.fast=fast
        # Opt-in because a public max_tokens cap is not an exact output.
        self.current_output_context=current_output_context
        self.output_tail = max(r['output_tokens'] for p in self.paths for r in p)
        tools=defaultdict(list)
        for path in self.paths:
            for r in path:
                if not r['done'] and r.get('tool_s') is not None:
                    tools[r.get('tool','')].append(float(r['tool_s']))
        self.tool_samples={k:tuple(v) for k,v in tools.items()}
        self.global_tool_samples=tuple(v for xs in tools.values() for v in xs)
        self.clear()

    def clear(self):
        self._population_cache=OrderedDict();self.cache_hits=0;self.cache_misses=0
        self._source_cache=OrderedDict();self._scenario_cache=OrderedDict()

    def population(self, inputs, outputs, after_completion=False, current_output=None):
        """Condition only on current input and actually completed output lengths.

        Distance and neighbor count are frozen before serving. Current output
        is right-censored: paths below attained output cannot declare it done.
        Beyond training stages use a one-call tail, explicitly no certainty
        about true termination. The next real arrival revises that forecast.
        """
        _check_history(inputs, outputs, after_completion)
        if current_output is not None and (after_completion or type(current_output) is not int or current_output < 1):
            raise ValueError('Current output context must be an exact positive pre-completion value')
        key=tuple(inputs),tuple(outputs),after_completion,current_output
        if key in self._population_cache:
            self.cache_hits+=1;value=self._population_cache.pop(key)
            self._population_cache[key]=value
            return value
        self.cache_misses+=1
        stage = len(inputs)-1
        eligible = [(i, p) for i, p in enumerate(self.paths) if len(p) > stage+int(after_completion)]
        fallback = not eligible
        if fallback: eligible = list(enumerate(self.paths))
        if self.fast and self.neighbors>=len(eligible):
            # An all-population model does not use distance to choose any
            # neighbor. Do not compute logarithms/exact means it discards.
            value=[(0.,i,p,min(stage,len(p)-1)) for i,p in eligible],fallback
            self._population_cache[key]=value
            if len(self._population_cache)>512:self._population_cache.popitem(last=False)
            return value
        scored = []
        for i, p in eligible:
            k = min(stage, len(p)-1)
            dg = [abs(math.log1p(max(0, inputs[j]-inputs[0]))-
                      math.log1p(max(0, p[min(j,len(p)-1)]['prompt_tokens']-p[0]['prompt_tokens'])))
                  for j in range(1, len(inputs))]
            dy = [abs(math.log1p(v)-math.log1p(p[min(j,len(p)-1)]['output_tokens']))
                  for j, v in enumerate(outputs)]
            if current_output is not None and not after_completion:
                dy.append(abs(math.log1p(current_output)-
                              math.log1p(p[min(stage,len(p)-1)]['output_tokens'])))
            mean=statistics.fmean if self.fast else statistics.mean
            score = mean(dg) if dg else 0.
            score += mean(dy) if dy else 0.
            scored.append((score, i, p, k))
        ranked=sorted(scored, key=lambda r: (r[0], r[1]))
        # All equally near paths remain eligible at the cutoff. Cold-start
        # predictions must not depend on which eight paths happen to be first.
        cutoff=ranked[min(self.neighbors,len(ranked))-1][0]
        value=[r for r in ranked if r[0]<=cutoff+1e-12],fallback
        self._population_cache[key]=value
        if len(self._population_cache)>512:self._population_cache.popitem(last=False)
        return value

    def forecast(self, name, inputs, outputs, *, attained=0, current_cap=None,
                 memory_pages=0, page=16, max_len=16384, after_completion=False,
                 source_training_index=None, known_current_output=None):
        _check_known_output(known_current_output, inputs, attained, current_cap, max_len, after_completion)
        context = known_current_output if self.current_output_context else None
        selected, fallback = self.population(inputs, outputs, after_completion, context)
        # Choose the median remaining-work observed PATH, preserving the
        # correlation among rounds, output, input growth and tool gaps.
        key=tuple(inputs),tuple(outputs),after_completion,context
        if source_training_index is not None:
            _,source,path,stage=next(r for r in selected if r[1]==source_training_index)
        elif self.fast and key in self._source_cache:
            source,path,stage=self._source_cache[key]
        else:
            rows = sorted(selected, key=lambda z: (sum(r['output_tokens'] for r in z[2][z[3]:]), z[1]))
            _, source, path, stage = rows[len(rows)//2]
            if self.fast:
                self._source_cache[key]=(source,path,stage)
                if len(self._source_cache)>512:self._source_cache.popitem(last=False)
        tail = path[stage:stage+1] if fallback else path[stage:]
        shift = inputs[-1]-tail[0]['prompt_tokens']; turns = []; previous = 0; clipped = 0
        for j, row in enumerate(tail):
            prompt = max(1, min(max_len-1, row['prompt_tokens']+shift))
            clipped += int(prompt != row['prompt_tokens']+shift or row['output_tokens'] > max_len-prompt)
            output = max(1, int(row['output_tokens']))
            if j == 0:
                prompt = inputs[-1]
                if output <= attained:
                    survivors = [p[k]['output_tokens'] for _, _, p, k in selected
                                 if p[k]['output_tokens'] > attained]
                    output = int(statistics.median(survivors)) if survivors else attained+max(1,self.output_tail//4)
                if current_cap is not None: output = min(output, max(attained+1, current_cap))
                if known_current_output is not None: output = known_current_output
            output = min(output, max_len-prompt)
            keep = min(prompt-1, previous) if j else 0
            turns.append(Turn(prompt, max(1,output), keep, float(row.get('tool_s') or 0)))
            previous = prompt
        if after_completion:
            turns = turns[1:]
            if not turns:
                growth = [max(1,p[i+1]['prompt_tokens']-p[i]['prompt_tokens'])
                          for p in self.paths for i in range(len(p)-1)]
                prompt = min(max_len-1, inputs[-1]+int(statistics.median(growth)))
                turns = [Turn(prompt,min(self.output_tail,max_len-prompt),inputs[-1],0.)]
                fallback = True
        # A known per-call cap enters ONLY the current-call memory constraint.
        # Extra capacity does not invent extra decode service in the model.
        target = math.ceil((turns[0].prompt+(current_cap or turns[0].output))/page)
        peak = max(target, memory_pages, *(math.ceil((t.prompt+t.output)/page) for t in turns))
        return Program(name, 0., tuple(turns)), dict(source_training_index=source,
            neighbors=len(selected), stage_tail_fallback=fallback, peak_pages=peak,
            current_pages=target, predicted_turns=len(turns),
            predicted_output=turns[0].output, context_clipped_turns=clipped,
            current_output_source='contractual' if known_current_output is not None else 'predicted')

    def scenarios(self,name,inputs,outputs,*,attained=0,current_cap=None,
                  after_completion=False,page=16,max_len=16384,known_current_output=None):
        """Equally weighted neighboring whole paths, cached by observed history.

        No new learned weights or evaluation labels. Current output censoring
        is revised from attained service; correlations in each future path
        are preserved. The cache contains only independent training paths.
        """
        _check_known_output(known_current_output, inputs, attained, current_cap, max_len, after_completion)
        context = known_current_output if self.current_output_context else None
        # The contractual current output changes the returned current turn
        # even when it is deliberately excluded from neighbor selection.
        key=tuple(inputs),tuple(outputs),after_completion,page,max_len,context,known_current_output
        selected,_=self.population(inputs,outputs,after_completion,context)
        if key not in self._scenario_cache:
            self._scenario_cache[key]=tuple(self.forecast('',inputs,outputs,
                after_completion=after_completion,page=page,max_len=max_len,
                source_training_index=row[1],known_current_output=known_current_output)[0].turns for row in selected)
            if len(self._scenario_cache)>512:self._scenario_cache.popitem(last=False)
        survivors=[p[k]['output_tokens'] for _,_,p,k in selected if p[k]['output_tokens']>attained]
        replacement=int(statistics.median(survivors)) if survivors else attained+max(1,self.output_tail//4)
        result=[]
        for turns in self._scenario_cache[key]:
            if not after_completion:
                first=turns[0];output=first.output
                if output<=attained:output=replacement
                if current_cap is not None:output=min(output,max(attained+1,current_cap))
                if known_current_output is not None:output=known_current_output
                output=max(1,min(output,max_len-first.prompt))
                turns=(Turn(first.prompt,output,first.keep,first.gap),)+turns[1:]
            result.append(Program(name,0.,turns))
        return result

    def idle_bound(self, tool):
        if not self.protection_support: raise ValueError('Prior has no verified tool timing support')
        values=self.tool_samples.get(tool,self.global_tool_samples)
        # Finite training-support lease, not presumed future Program lifetime.
        return max(.001, max(values, default=1.))

    def idle_remaining(self, tool, age):
        if not self.protection_support: raise ValueError('Prior has no verified tool timing support')
        values=self.tool_samples.get(tool,self.global_tool_samples)
        remaining=[v-age for v in values if v>age]
        # Right-censored observation: samples that already returned are not
        # plausible remaining durations. Beyond training support use a finite
        # positive tail; the lease can still expire independently.
        return max(.001, statistics.mean(remaining) if remaining else
                   statistics.mean(values) if values else 1.)


def choose_continuation(programs, order, initial, beta, decode, *, settings,
                        minimum_pages=None, evaluator=None):
    """Compare now with event-delayed admission over COMPLETE residual paths.

    Only currently arrived identities are supplied. Ongoing work is mandatory.
    Four explicit candidates bound CPU work. No optimum outside this family
    is claimed; infeasible growth is rejected, not given a free terminal state.
    """
    started = time.perf_counter(); evaluations = []
    minimum_pages = minimum_pages or {}
    evaluate=evaluator or simulate_calendar
    common = dict(settings, initial=initial, strict_order=False, record=True,
                  minimum_pages=minimum_pages)
    pending = [p for p in order if p not in initial]
    if not pending:
        return dict(admit=[], evaluations=[], seconds=time.perf_counter()-started)
    focus = pending[0]
    peak = evaluate(programs, order, beta, decode, new_peak=True, **common)
    gates = admission_gates(peak['events']) if peak['feasible'] else {}
    candidates = [('now', {})]
    if peak['feasible']:
        next_call = {}; next_program = {}; counts = {}
        for e in peak['events']:
            if e['event'] == 'turn_finish' and e['program'] in initial:
                counts[e['program']] = e['stage']+1
                if not next_call: next_call = dict(counts)
            if e['event'] == 'finish' and e['program'] in initial and not next_program:
                next_program = dict(counts)
        candidates += [('next-call', {focus:next_call}),
                       ('next-program', {focus:next_program}), ('peak-guards', gates)]
    seen = set(); best = None
    for label, guards in candidates:
        key = tuple(sorted((p, tuple(sorted(g.items()))) for p,g in guards.items()))
        if key in seen: continue
        seen.add(key)
        result = evaluate(programs, order, beta, decode, gates=guards, **common)
        evaluations.append(dict(candidate=label, feasible=result['feasible'],
            sum_jct_s=result.get('sum_jct_s'), reason=result.get('reason')))
        if result['feasible'] and (best is None or result['sum_jct_s'] < best[1]['sum_jct_s']-1e-8):
            best = label,result
    admit = [] if best is None else [e['program'] for e in best[1]['events']
                                    if e['event']=='admit' and e['time']<=1e-9]
    return dict(admit=admit, evaluations=evaluations, selected=best[0] if best else 'wait-infeasible',
                seconds=time.perf_counter()-started)
