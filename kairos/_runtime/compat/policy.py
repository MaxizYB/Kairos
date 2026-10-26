"""Kairos policy implementation."""
from __future__ import annotations
import math
import time
from bisect import bisect_right
from collections import defaultdict


class TTLPolicy:
    def __init__(self, mode="paper", fixed=2.0, k=100, history=None):
        self.mode,self.fixed,self.k = mode,fixed,k
        self.history = defaultdict(list)
        for name,rows in (history or {}).items():
            self.history[name] = [(float(r[0]),int(r[1])) for r in rows]

    def observe(self, tool, duration, added_tokens):
        self.history[tool].append((duration,added_tokens))
        self.history[tool] = self.history[tool][-512:]

    def ttl(self, tool, benefit, oracle_gap=None):
        if self.mode == "zero":
            return 0.0,"zero"
        if self.mode == "fixed":
            return self.fixed,"fixed"
        if self.mode == "oracle":
            if oracle_gap is None:
                raise ValueError("Oracle TTL requires the actual recorded tool gap")
            return max(0,float(oracle_gap))+0.02,"oracle-tool-gap"
        if self.mode == "oracle-economic":
            if oracle_gap is None or not math.isfinite(float(oracle_gap)) or float(oracle_gap) < 0:
                raise ValueError("Economic oracle requires a finite nonnegative recorded tool gap")
            # Information ablation of the paper objective, not global optimal
            # JCT. Keep the replay handover margin in both benefit and cost.
            candidate = float(oracle_gap) + .02
            return (candidate if benefit > candidate else 0.), "oracle-economic-tool-gap"
        global_rows = [row for rows in self.history.values() for row in rows]
        if len(global_rows) <= self.k:
            return max(0,math.log(max(benefit,1.0))),"cold-exp-unit-mean"
        rows = self.history[tool] if len(self.history[tool])>self.k else global_rows
        xs = sorted(r[0] for r in rows)
        best,utility = 0.0,0.0
        for candidate in sorted(set([0.0]+xs)):
            prob = bisect_right(xs,candidate)/len(xs)
            u = prob*benefit-candidate
            if u>utility:
                best,utility=candidate,u
        return best,"per-tool-ecdf" if rows is self.history[tool] else "global-ecdf"

    def conditional(self, tool, elapsed):
        rows = [r for r in self.history[tool] if r[0]>elapsed]
        if not rows:
            rows = [r for rs in self.history.values() for r in rs if r[0]>elapsed]
        if not rows:
            return [(max(0.01,self.fixed-elapsed),256)]*3
        rows = sorted(rows)
        return [(rows[int(q*(len(rows)-1))][0]-elapsed,
                 rows[int(q*(len(rows)-1))][1]) for q in (0.2,0.5,0.8)]


def evaluate_prefill(requests, budget=64, beta=(1.0,0.01,0.0001,0.05), deadline=None):
    """Restricted evaluator from Method.md, using interval cursors.

    Units follow beta and arrivals. Shared batch completion sets every F.
    Also usable for MSA layouts in CPU mechanism tests.
    """
    rows=[]
    for r in requests:
        intervals=r.get("intervals")
        if intervals is None:
            intervals=[(x,x) for x in r["queries"]]
        rows.append({**r,"intervals":list(intervals),"segment":0,"position":None})
    t=0.0;finish={};trace=[]
    while len(finish)<len(rows):
        if deadline is not None and time.perf_counter()>=deadline:return None
        for r in rows:
            if r["arrival"]<=t and r["segment"]>=len(r["intervals"]):
                finish.setdefault(r["id"],r["arrival"])
        ready=sorted([r for r in rows if r["arrival"]<=t and r["id"] not in finish],
                     key=lambda r:(r["order"],r["id"]))
        if not ready:
            future=[r["arrival"] for r in rows if r["id"] not in finish and r["arrival"]>t]
            if not future: break
            t=min(future);continue
        left=budget;Q=A=N=0;completed=[]
        for r in ready:
            rq=ra=0
            while left and r["segment"]<len(r["intervals"]):
                u,v=r["intervals"][r["segment"]]
                u=r["position"] if r["position"] is not None else u
                n=min(left,v-u+1)
                rq+=n;ra+=n*(2*u+n-1)//2;left-=n
                if u+n>v:
                    r["segment"]+=1;r["position"]=None
                else:r["position"]=u+n
            if rq:Q+=rq;A+=ra;N+=1
            if r["segment"]>=len(r["intervals"]):completed.append(r["id"])
            if not left:break
        duration=max(1e-9,beta[0]+beta[1]*Q+beta[2]*A+beta[3]*N)
        end=t+duration
        for rid in completed:finish[rid]=end
        trace.append(dict(start=t,end=end,Q=Q,A=A,N=N))
        t=end
    if len(finish)!=len(rows):return None
    return dict(J=sum(finish[r["id"]]-r["arrival"] for r in rows),F=finish,batches=trace)


def select_suffixes(rows,K,block_size,budget,beta,optimize=True,budget_s=.005,max_evals=32):
    """Feasible suffix search for native APC semantic recovery.

    Assumes released suffixes eventually get overwritten: an explicitly
    conservative proxy, not the complete allocator/event simulator in Method.
    Telemetry separates released references from actual misses at admission.
    """
    start=time.perf_counter();rows=sorted(rows,key=lambda r:r["order"])
    selected={r["id"]:0 for r in rows}
    K=min(K,sum(r["limit"] for r in rows))
    for _ in range(K):
        choices=[r for r in rows if selected[r["id"]]<r["limit"]]
        if not choices:break
        def weight(r):
            pos=(r["total"]-selected[r["id"]])*block_size
            gap=sum(x[0] for x in r["scenarios"])/3
            return (beta[1]*block_size+beta[2]*(pos*block_size))/(1+gap)
        victim=min(choices,key=lambda r:(weight(r),r["order"]))
        selected[victim["id"]]+=1
    initial=dict(selected)
    def cost(plan):
        J=0
        for s in range(3):
            requests=[]
            for r in rows:
                missing=plan[r["id"]]*block_size
                old=r["old_tokens"]
                u=max(1,r["scenarios"][s][1])
                start_token=max(1,old-missing+1)
                requests.append(dict(id=r["id"],arrival=r["scenarios"][s][0],order=r["order"],
                                     intervals=[(start_token,old+u)]))
            result=evaluate_prefill(requests,budget,beta)
            if result is None:return math.inf
            J+=result["J"]/3
        return J
    evaluations=0;initial_J=cost(selected);best_J=initial_J
    if optimize:
        while evaluations<max_evals and time.perf_counter()-start<budget_s:
            best=None;best_cost=best_J
            for group in (1,4,16):
                for src in rows:
                    if selected[src["id"]]<group:continue
                    for dst in rows:
                        if src is dst or selected[dst["id"]]+group>dst["limit"]:continue
                        if evaluations>=max_evals or time.perf_counter()-start>=budget_s:break
                        trial=dict(selected);trial[src["id"]]-=group;trial[dst["id"]]+=group
                        value=cost(trial);evaluations+=1
                        if value<best_cost:best=trial;best_cost=value
            if best is None:break
            selected,best_J=best,best_cost
    search_s=time.perf_counter()-start
    if initial_J-best_J<=search_s:selected=initial;best_J=initial_J
    return {p:n for p,n in selected.items() if n},dict(initial_J=initial_J,best_J=best_J,
            evaluations=evaluations,search_s=search_s,recovery_proxy="suffix-overwritten")
