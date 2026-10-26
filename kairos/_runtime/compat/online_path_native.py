"""Kairos online path native implementation."""
import ctypes as C
import hashlib
from pathlib import Path
import subprocess
import tempfile

_lib=None


def warmup():
    global _lib
    if _lib is not None:return
    source=Path(__file__).with_suffix('.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    cache=Path.home()/'.cache/kvcache-online-path';cache.mkdir(parents=True,exist_ok=True)
    library=cache/('calendar-'+digest+'.so')
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix='calendar-build-',dir=cache) as build:
            target=Path(build)/'calendar.so'
            subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(target)],check=True)
            target.replace(library)
    _lib=C.CDLL(str(library));f=_lib.online_calendar
    f.argtypes=[C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_double),C.POINTER(C.c_double),
                C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_double),
                C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_int]
    f.restype=C.c_int


def simulate(programs,order,beta,decode_steps,*,initial=None,minimum_pages=None,gates=None,
             capacity_pages=3071,headroom_pages=32,page_size=16,max_seqs=8,
             query_budget=2048,mixed_floor=0.,host_step=0.,new_peak=False,**unused):
    warmup();by={p.name:p for p in programs};n=len(order)
    if n!=len(by) or set(order)!=set(by) or not n:raise ValueError('Invalid online order')
    if any(p.arrival!=0 for p in programs):raise ValueError('Only arrived residual paths are supported')
    initial=initial or {};minimum_pages=minimum_pages or {};gates=gates or {}
    offsets=[0];turns=[];states=[]
    for p in order:
        turns.extend(v for t in by[p].turns for v in (t.prompt,t.output,t.keep,t.gap))
        offsets.append(len(turns)//4)
        r=initial.get(p)
        states.extend([1,r.get('ready',0),r.get('computed',0),r.get('generated',0),
                       r['pages'],r.get('started',False)] if r else [0]*6)
    I=lambda xs:(C.c_int*len(xs))(*xs)
    D=lambda xs:(C.c_double*len(xs))(*xs)
    settings=[capacity_pages,headroom_pages,page_size,max_seqs,query_budget,new_peak,
              mixed_floor,host_step,*beta]+[0.]+[float(decode_steps.get(str(k),decode_steps.get(k,.02)))
                                               for k in range(1,max(max_seqs,n)+1)]
    out=D([0.]*11);event_capacity=3*offsets[-1]+3*n+4;events=D([0.]*(5*event_capacity))
    ok=_lib.online_calendar(n,I(offsets),D(turns),D(states),I([minimum_pages.get(p,0) for p in order]),
        I([gates.get(p,{}).get(q,0) for p in order for q in order]),D(settings),out,events,event_capacity)
    if not ok:
        code=int(out[10])
        if code not in (1,2,3):raise ValueError('Invalid native calendar input/state: '+str(code))
        return dict(feasible=False,reason={1:'future growth deadlock',2:'initial overcommit',
            3:'forecast path cannot fit available envelope'}[code],events=[])
    result=[];completed={p:0 for p in order};finish={}
    for i in range(int(out[9])):
        k,p,stage,t,used=events[5*i:5*i+5];p=order[int(p)];stage=int(stage)
        e=dict(event=['admit','start_call','turn_finish','finish'][int(k)],program=p,time=t)
        if k==0:e.update(reserved_pages=int(used),completed=dict(completed))
        elif k==1:e.update(stage=stage,reserved_pages=int(used))
        elif k==2:e['stage']=stage;completed[p]=stage+1
        else:finish[p]=t
        result.append(e)
    return dict(feasible=True,mean_jct_s=out[1],sum_jct_s=out[2],makespan_s=out[3],
        prefill_queries=int(out[4]),engine_s=out[5],steps=int(out[6]),peak_reserved_pages=int(out[7]),
        reserved_page_seconds=out[8],completion_s=finish,events=result)
