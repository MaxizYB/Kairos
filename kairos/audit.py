"""Recompute released measurements and consumed scheduling/TTL decisions."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from kairos.causal import service
from kairos.protection import CommitmentHoldPolicy, CommitmentPrior

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_bytes())


def near(got,want):
    if not math.isclose(got,want,rel_tol=1e-9,abs_tol=1e-9):
        raise ValueError(f'Mismatch: {got} versus {want}')


def percentile(xs,q):
    xs=sorted(xs);position=(len(xs)-1)*q;a=int(position);b=min(a+1,len(xs)-1)
    return xs[a]+(position-a)*(xs[b]-xs[a])


def recovery(quote,profile):
    """Independent reconstruction from logged physical facts, not serving code."""
    if quote['restored'] or quote['computed_tokens']>=quote['prompt_tokens']:return 0.
    prompt=quote['prompt_tokens'];page=quote['page']
    gpu=min(prompt,max(quote['computed_tokens'],quote['gpu_hit_pages']*page))
    gpu=gpu//page*page
    chunks=int(quote['host'].get('chunks',prompt//256))
    chunk=gpu//256;committed=set(quote['host'].get('cpu_chunks',[]))
    while chunk<chunks and chunk in committed:chunk+=1
    end=min(prompt,max(gpu,chunk*256));loaded=end-gpu
    beta=profile['beta'];result=0.
    for cursor in range(end,prompt,2048):
        count=min(2048,prompt-cursor)
        result+=max(0.,beta[0]+beta[1]*count+beta[2]*count*(2*cursor+count+1)/2+
                    (beta[3] if len(beta)>3 else 0.))
    if loaded:result+=max(0.,profile['reload_fixed_seconds'])+loaded*max(0.,profile['reload_seconds_per_token'])
    return result


def audit_results():
    comparison=read(ROOT/'results/comparison.json');verified=[]
    for row in comparison['rows']:
        name=row['method'];directory=ROOT/'results'/name
        jobs=read(directory/'jobs.json');summary=read(directory/'summary.json')
        with (directory/'requests.jsonl').open() as stream:requests=[json.loads(x) for x in stream]
        if len(jobs)!=16 or len(requests)!=178 or any(r['error'] for r in requests):
            raise ValueError('Incomplete archived replay')
        if sum(r['output_tokens'] for r in requests)!=26164:raise ValueError('Work mismatch')
        jcts=[j['jct_s'] for j in jobs]
        near(summary['jct_mean_s'],sum(jcts)/len(jcts));near(summary['jct_p95_s'],percentile(jcts,.95))
        for key in ('jct_mean_s','jct_p95_s','wall_s'):near(summary[key],row['summary'][key])
        verified.append(dict(method=name,mean_jct_s=summary['jct_mean_s'],
                             p95_jct_s=summary['jct_p95_s'],wall_s=summary['wall_s']))
    reference=read(ROOT/'assets/reference_profile.json')
    profile=read(ROOT/'assets/deployment_profile.json');profile.update(read(ROOT/'assets/reload_profile.json'))
    prior=CommitmentPrior(read(ROOT/'assets/program_prior.json'))
    hold=CommitmentHoldPolicy(training=prior.tool_samples)
    posts={};tools={};checked=ttl=changed=0;consumed=set();arrivals={}
    stream=gzip.open(ROOT/'results/kairos/scheduler.jsonl.gz','rt')
    for line in stream:
        event=json.loads(line);kind=event['event'];p=event.get('program')
        if event.get('future_information') is True:raise ValueError('Future-information flag')
        if kind=='arrival':tools[p]=event['meta'].get('cont_tool')
        elif kind=='r83_observation':arrivals[p]=event['request']
        elif kind=='r82_prediction':posts[p]=event
        elif kind=='tool_observed':hold.observe(event['tool'],event['duration'],event['added_tokens'])
        elif kind=='r8314_dispatch':
            changed+=event['changed']
            for p,quote in event['quotes'].items():
                post=posts[p]
                if not post['request']==quote['request']==arrivals[p]:raise ValueError('Stale forecast consumed')
                current=service(quote['prompt_tokens'],quote['remaining_output'],post['feature_keep'],reference)
                # Quotes are for ungranted calls, hence no generated output has been subtracted.
                near(current,quote['current_s'])
                total=math.fsum(post['mass'])
                progress=math.fsum(w/total*current/(current+r) for r,w in zip(post['atoms_s'],post['mass']))
                restore=recovery(quote,profile)
                actual=restore+quote['remaining_output']*quote['decode_step_s']
                for got,want in ((quote['restore_s'],restore),(quote['progress'],progress),
                                 (quote['actual_s'],actual),(quote['score'],actual/progress)):
                    near(got,want)
                checked+=1;consumed.add(quote['request'])
        elif kind=='pin_decision':
            value,_=hold.ttl(tools[p],event['reload_estimate']+event['wait_mean']*event['continuity_fraction'],
                             opportunity_rate=event['opportunity_rate'])
            near(value,event['ttl']);ttl+=1
    stream.close()
    if (checked,len(consumed),ttl)!=(1700,52,162):raise ValueError('Incomplete mechanism evidence')
    baseline=next(r for r in verified if r['method']=='continuum')
    candidate=next(r for r in verified if r['method']=='kairos')
    return dict(valid=True,results=verified,quotes_checked=checked,ttl_checked=ttl,
        consumed_requests=len(consumed),changed_events=changed,
        reductions_percent={k:100*(1-candidate[k]/baseline[k]) for k in ('mean_jct_s','p95_jct_s','wall_s')})


def verify_files():
    manifest=read(ROOT/'provenance/release_manifest.json')
    for row in manifest['files']:
        path=ROOT/row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Release artifact changed: '+row['path'])
    return len(manifest['files'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-file-hashes',action='store_true')
    args=parser.parse_args()
    result=audit_results()
    if not args.skip_file_hashes:result['files_verified']=verify_files()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
