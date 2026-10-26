"""Frozen hierarchy-aware teacher replay used by the archived measurements."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from kairos._runtime.replay import dump as save


def quantiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    def pct(q):
        position = (len(ordered)-1)*q
        low = int(position)
        high = min(low+1, len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)
    mean = statistics.fmean(values)
    return dict(n=len(values), mean=mean, min=ordered[0], p50=pct(.5), p90=pct(.9),
                p95=pct(.95), p99=pct(.99), max=ordered[-1],
                cv=statistics.pstdev(values)/mean if mean else None)


def synthetic_visible_messages(*args, **kwargs):
    raise ValueError('Only token-verified visible messages are released')

async def replay(args, programs, run_dir):
    import httpx
    records=[];jobs=[]
    done={p['id']:asyncio.Event() for p in programs}
    failure={}
    prior=programs[0].get('tool_prior',{})
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout,connect=10),
            limits=httpx.Limits(max_connections=64)) as client:
        start=time.monotonic();start_wall=time.time()
        async def program(row):
            parent=row.get('predecessor')
            if parent:
                if parent not in done:raise ValueError('Unknown parent Program')
                await done[parent].wait()
                if failure.get(parent):raise RuntimeError('Parent Program failed')
                await asyncio.sleep(row['human_gap_replay_s'])
            else:
                await asyncio.sleep(max(0,row['arrival_s']-(time.monotonic()-start)))
            job_start=time.monotonic();prompt=[];first_end=None
            try:
                for turn,part in enumerate(row['turns']):
                    prompt=prompt[:part.get('keep_prefix_tokens',len(prompt))]+part['added_token_ids']
                    meta=dict(cont_program_id=row['id'],cont_tool=part.get('tool','unclassified-gap'),
                        cont_final=turn==len(row['turns'])-1,cont_oracle_gap=0)
                    if os.environ.get('CONTINUUM_WORK_MESSAGES'):
                        visible=getattr(args,'_work_visible_messages',None)
                        if visible is None:
                            visible=json.loads(Path(os.environ['CONTINUUM_WORK_MESSAGES']).read_bytes())
                            args._work_visible_messages=visible
                        meta['cont_visible_messages']=json.dumps(visible[row['id']][turn],separators=(',',':'))
                    elif os.environ.get('CONTINUUM_SYNTHETIC_VISIBLE') == '1':
                        meta['cont_visible_messages']=json.dumps(
                            synthetic_visible_messages(row,turn),separators=(',',':'))
                    if args.reuse_engine:meta['cont_experiment']=args.current_method
                    if row['id']==programs[0]['id'] and turn==0 and prior:
                        meta['cont_bfcl_reset_history']=json.dumps(prior)
                    payload=dict(model=args.model,prompt=prompt,max_tokens=part['output_tokens'],
                        ignore_eos=True,temperature=0,seed=1,stream=True,return_token_ids=True,
                        stream_options=dict(include_usage=True),cache_salt=row.get('cache_salt',row['id']),vllm_xargs=meta)
                    begin=time.monotonic();stamps=[];output=[];usage={};error=None;request_id=None
                    try:
                        async with client.stream('POST',f'http://127.0.0.1:{args.port}/v1/completions',json=payload) as response:
                            if response.status_code!=200:raise RuntimeError((await response.aread()).decode()[:1200])
                            async for line in response.aiter_lines():
                                if not line.startswith('data: ') or line=='data: [DONE]':continue
                                item=json.loads(line[6:]);request_id=item.get('id',request_id)
                                if 'error' in item:raise RuntimeError(str(item['error']))
                                for choice in item.get('choices',[]):
                                    ids=choice.get('token_ids') or []
                                    if ids:stamps.append((time.monotonic(),len(ids)));output+=ids
                                if item.get('usage'):usage=item['usage']
                    except Exception as exc:error=repr(exc)
                    end=time.monotonic()
                    record=dict(program=row['id'],session=row['session_id'],user=row['synthetic_user'],
                        cohort=row['cohort'],turn=turn,source_turn=row['source_turn_id'],request_id=request_id,
                        prompt_tokens=len(prompt),output_tokens=len(output),output_ids=output,usage=usage,error=error,
                        start_s=begin-start,end_s=end-start,e2e_s=end-begin,
                        ttft_s=stamps[0][0]-begin if stamps else None,
                        itl_s=[b[0]-a[0] for a,b in zip(stamps,stamps[1:]) if a[1]==b[1]==1],
                        stream_chunks=[n for _,n in stamps],
                        prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                        planned_output_tokens=part['output_tokens'],source_gap_s=part['source_gap_s'],
                        replay_gap_s=part['gap_after_s'],source_cached_tokens=part['source_cached_tokens'],
                        source_tool_failures=part['source_tool_failures'])
                    records.append(record)
                    with (run_dir/'requests.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    if error or len(output)!=part['output_tokens']:
                        failure[row['id']]=True
                        raise RuntimeError(f'Incomplete call {row["id"]}:{turn}: {error}; {len(output)} tokens')
                    if turn==0:first_end=end
                    if turn<len(row['turns'])-1:await asyncio.sleep(part['gap_after_s'])
                job_end=time.monotonic()
                jobs.append(dict(program=row['id'],session=row['session_id'],user=row['synthetic_user'],cohort=row['cohort'],
                    jct_s=job_end-job_start,arrival_s=job_start-start,end_s=job_end-start,turns=len(row['turns']),
                    first_turn_end_s=first_end-start,post_initial_jct_s=job_end-first_end,
                    primed_jct_s=None,tool_dependency_s=sum(t['gap_after_s'] for t in row['turns'][:-1]),
                    human_gap_original_s=row['human_gap_original_s'],human_gap_replay_s=row['human_gap_replay_s']))
            finally:done[row['id']].set()
        await asyncio.wait_for(asyncio.gather(*(program(p) for p in programs)),args.run_timeout)
    wall=time.monotonic()-start
    def avg(v):return statistics.fmean(v) if v else None
    def p95(v):return quantiles(v).get('p95')
    good=[r for r in records if r['error'] is None]
    ttft=[r['ttft_s'] for r in good if r['ttft_s'] is not None]
    itl=[t for r in good for t in r['itl_s']];e2e=[r['e2e_s'] for r in good]
    summary=dict(method=args.current_method,wall_s=wall,start_wall_time=start_wall,end_wall_time=time.time(),
        completed=len(good),failed=len(records)-len(good),jobs=len(jobs),
        jct_mean_s=avg([j['jct_s'] for j in jobs]),jct_p95_s=p95([j['jct_s'] for j in jobs]),
        primed_jct_mean_s=None,primed_jct_p95_s=None,ttft_mean_s=avg(ttft),ttft_p95_s=p95(ttft),
        itl_mean_s=avg(itl),itl_p95_s=p95(itl),e2e_mean_s=avg(e2e),e2e_p95_s=p95(e2e),
        request_throughput=len(good)/wall,output_token_throughput=sum(r['output_tokens'] for r in good)/wall,
        total_token_throughput=sum(r['prompt_tokens']+r['output_tokens'] for r in good)/wall,
        hierarchy='Program=one user target; human wait excluded from Program JCT',
        workload_scope='public production metadata, synthetic token identity, bounded serial subset')
    save(run_dir/'jobs.json',jobs);save(run_dir/'summary.json',summary)
    if len(good)!=sum(len(p['turns']) for p in programs):raise RuntimeError('Incomplete replay')
    return summary
