"""Kairos oracle program plan implementation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


def trace_identity(trace_bytes, plan):
    """Accept the exact registered trace or its declared JSON serialization.

    The original archive pretty-prints the prepared JSON. Legacy plans were
    made from that archive, so byte identity alone incorrectly rejects the
    identical workload. No token, arrival, tool or metadata is ignored.
    """
    actual = hashlib.sha256(trace_bytes).hexdigest()
    parsed = json.loads(trace_bytes)
    canonical = hashlib.sha256(json.dumps(parsed, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    pretty = hashlib.sha256((json.dumps(parsed, indent=2)+'\n').encode()).hexdigest()
    valid = (actual == plan['trace_sha256'] or
             canonical == plan.get('trace_content_sha256') or
             pretty == plan['trace_sha256'])
    if not valid:
        raise ValueError('Oracle plan trace does not match the actual real replay')
    return dict(prepared_bytes_sha256=actual, canonical_json_sha256=canonical,
                planned_bytes_sha256=plan['trace_sha256'],
                identical_bytes=actual == plan['trace_sha256'],
                comparison='all JSON fields; serialization differences only')


@dataclass(frozen=True)
class Turn:
    prompt: int
    output: int
    keep: int
    gap: float


@dataclass(frozen=True)
class Program:
    name: str
    arrival: float
    turns: tuple[Turn, ...]

    def peak_pages(self, page_size=16):
        return max(math.ceil((t.prompt+t.output)/page_size) for t in self.turns)


def parse_trace(trace):
    return tuple(Program(p['id'],float(p['arrival_s']),tuple(
        Turn(int(t['prompt_tokens']),int(t['output_tokens']),
             int(t['keep_prefix_tokens']),float(t.get('gap_after_s',0)))
        for t in p['turns'])) for p in trace['programs'])


def simulate(programs, order, beta, decode_steps, capacity_pages=3071,
             page_size=16, max_seqs=8, query_budget=2048, headroom_pages=32,
             mixed_floor=0., host_step=0., record=False, strict_order=False):
    """Closed-loop, token-iteration simulation of the restricted policy.

    Tools start at the simulated completion of their preceding call. Query
    budgets and full-history reservations are concrete. Mixed execution uses
    a profiled approximation, and is reported as such rather than truth.
    """
    by_name={p.name:p for p in programs}
    if len(order)!=len(programs) or set(order)!=set(by_name):
        raise ValueError('Order must contain every Program exactly once')
    rank={p:i for i,p in enumerate(order)}
    peaks={p.name:p.peak_pages(page_size) for p in programs}
    if any(v>capacity_pages-headroom_pages for v in peaks.values()):
        raise ValueError('A complete Program cannot fit the reservation envelope')
    pending=set(by_name);resident=set();finished={};rows={};events=[];t=0.
    steps=prefill=decode=mixed=0
    engine=0.;area=0.
    def log(event,**kw):
        if record:events.append(dict(time=t,event=event,**kw))
    def open_turn(name,stage,ready):
        turn=by_name[name].turns[stage]
        # The original teacher continuation can replace the generated tail.
        # Only exact keep_prefix_tokens, rounded down to committed pages, hits.
        keep=min(turn.keep//page_size*page_size,max(0,turn.prompt-1)) if stage else 0
        rows[name]=dict(stage=stage,ready=ready,computed=keep,generated=0)
    while len(finished)<len(programs):
        admitted=False
        while len(resident)<max_seqs:
            if strict_order:
                first=min(pending,key=rank.get) if pending else None
                eligible=[first] if first is not None and by_name[first].arrival<=t else []
            else:
                eligible=sorted((p for p in pending if by_name[p].arrival<=t),key=rank.get)
            if not eligible:break
            p=eligible[0]
            used=sum(peaks[q] for q in resident)
            if used+peaks[p]+headroom_pages>capacity_pages:break
            pending.remove(p);resident.add(p);open_turn(p,0,t);admitted=True
            log('admit',program=p,reserved_pages=peaks[p],total_reserved=used+peaks[p])
        ready=sorted((p for p in resident if rows[p]['ready']<=t),key=rank.get)
        if not ready:
            future=[by_name[p].arrival for p in pending if by_name[p].arrival>t]
            future.extend(rows[p]['ready'] for p in resident if rows[p]['ready']>t)
            if not future:raise RuntimeError('Reservation policy stalled')
            end=min(future)
            area+=sum(p.arrival<=t and p.name not in finished for p in programs)*(end-t)
            t=end;continue
        left=query_budget;Q=A=0;members=[];prefill_q=0
        # Native serving runs existing decodes before taking prefill work.
        decoders=[p for p in ready if rows[p]['computed']>=by_name[p].turns[rows[p]['stage']].prompt]
        prefills=[p for p in ready if p not in decoders]
        for p in decoders+prefills:
            if not left:break
            row=rows[p];turn=by_name[p].turns[row['stage']]
            if row['computed']<turn.prompt:
                n=min(left,turn.prompt-row['computed'])
                begin=row['computed']+1;A+=n*(2*begin+n-1)//2
                row['computed']+=n;prefill_q+=n
                if row['computed']==turn.prompt:row['generated']+=1
            else:
                n=1;A+=turn.prompt+row['generated'];row['generated']+=1
            Q+=n;left-=n;members.append(p)
        if not members:raise RuntimeError('No legal execution progress')
        N=len(members)
        ds=float(decode_steps.get(str(N),decode_steps.get(N,.02)))
        if prefill_q:
            duration=max(float(mixed_floor),float(beta[0])+float(beta[1])*Q+
                float(beta[2])*A+float(beta[3])*N,ds)
            mixed+=1
        else:duration=ds
        duration+=host_step
        if not duration>0:raise ValueError('Execution duration must be positive')
        end=t+duration
        # Count arrivals inside the iteration with their exact interval length.
        area+=sum(max(0.,end-max(t,p.arrival)) for p in programs
                  if p.name not in finished and p.arrival<end)
        engine+=duration;steps+=1;prefill+=prefill_q;decode+=Q-prefill_q;t=end
        for p in members:
            row=rows[p];turn=by_name[p].turns[row['stage']]
            if row['generated']<turn.output:continue
            stage=row['stage'];log('turn_finish',program=p,stage=stage)
            if stage+1==len(by_name[p].turns):
                finished[p]=t;resident.remove(p);del rows[p];log('finish',program=p)
            else:open_turn(p,stage+1,t+turn.gap)
    total=sum(finished[p.name]-p.arrival for p in programs)
    if not math.isclose(total,area,rel_tol=1e-8,abs_tol=1e-6):
        raise AssertionError('Completion sum and unfinished-area disagree')
    return dict(mean_jct_s=total/len(programs),sum_jct_s=total,
        makespan_s=t,completion_s=finished,steps=steps,mixed_steps=mixed,
        prefill_queries=prefill,decode_queries=decode,engine_s=engine,
        peak_reservation_pages=peaks,events=events if record else None)


def improve_order(programs,initial,beta,decode_steps,**kwargs):
    """Deterministic full-replay insertion search; feasible upper bound only.

    Searches a restricted reservation policy family, not all possible batch,
    preemption and placement trajectories. No horizon terminal proxy is used.
    """
    order=tuple(initial);evaluations=0;cache={};passes=[]
    def cost(candidate):
        nonlocal evaluations
        key=tuple(candidate)
        if key not in cache:
            cache[key]=simulate(programs,key,beta,decode_steps,**kwargs)
            evaluations+=1
        return cache[key]
    best=cost(order)
    converged=False
    for iteration in range(8):
        choice=order;value=best
        for i in range(len(order)):
            shorter=list(order);p=shorter.pop(i)
            for j in range(len(order)):
                candidate=tuple(shorter[:j]+[p]+shorter[j:])
                result=cost(candidate)
                if result['sum_jct_s']<value['sum_jct_s']-1e-8:
                    choice,value=candidate,result
        passes.append(dict(iteration=iteration,mean_jct_s=value['mean_jct_s']))
        if choice==order:
            converged=True
            break
        order,best=choice,value
    return dict(order=list(order),simulation=best,evaluations=evaluations,
                passes=passes,converged=converged,optimality=(
                    'local optimum in insertion-order family; no global certificate'
                    if converged else
                    'best insertion-search incumbent at iteration limit; no optimality certificate'))
