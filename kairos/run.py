"""Launch one isolated, bounded teacher replay using the frozen configuration."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

from kairos._runtime import replay as harness
from kairos._runtime.activation import RUNTIME_ACTIVATION, RUNTIME_ENVIRONMENT
from kairos._runtime.teacher_replay import replay as teacher_replay

ROOT=Path(__file__).resolve().parents[1]
REVISION='b968826d9c46dd6066d109eabc6255188de91218'

def expand(value, bindings):
    if isinstance(value,list):return [expand(v,bindings) for v in value]
    if isinstance(value,dict):return {k:expand(v,bindings) for k,v in value.items()}
    if isinstance(value,str):
        return re.sub(r'\$\{([A-Z]+)\}',lambda m:bindings[m[1]],value)
    return value


def affinity(spec):
    cpus=set()
    for part in spec.split(','):
        bounds=[int(x) for x in part.split('-')]
        if len(bounds) not in (1,2) or bounds[0]<0 or bounds[-1]<bounds[0]:
            raise ValueError('Invalid CPU affinity')
        cpus.update(range(bounds[0],bounds[-1]+1))
    return cpus


def prepare(args, out, control_socket):
    config=json.loads((ROOT/'configs'/f'{args.method}.json').read_text())
    if config.get('method') != args.method or config.get('runtime_profile') != args.method:
        raise ValueError('Method configuration identity mismatch')
    bindings=dict(ROOT=str(ROOT),OUT=str(out),MODEL=args.model,PYTHON=sys.executable,
                  PROFILE=str(args.profile.resolve()),RELOAD=str(args.reload_profile.resolve()),
                  SOCKET=str(control_socket))
    command=expand(config['command'],bindings)
    # Only explicitly archived policy settings enter the serving process.
    env={k:v for k,v in os.environ.items() if not k.startswith(
        ('CONTINUUM_','CONTINUITY_','KAIROS_','LMCACHE_','VLLM_'))}
    env.update(expand(config['environment'],bindings))
    # Translate the public method name to the frozen runtime protocol without
    # putting implementation-era switches in the public experiment config.
    env.update(expand(RUNTIME_ENVIRONMENT[args.method],bindings))
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',MAX_JOBS='2',
        PYTHONDONTWRITEBYTECODE='1',TRANSFORMERS_OFFLINE='1')
    # Do not let an old research checkout shadow the standalone package.
    env['PYTHONPATH']=str(ROOT)
    engine=['taskset','-c',args.engine_cpus,*command]
    helper=['taskset','-c',args.helper_cpus,sys.executable,'-m',
        'kairos._runtime.compat.lmcache_server','--host','127.0.0.1','--port','18766',
        '--chunk-size','256','--max-workers','2','--l1-size-gb','8.0',
        '--l1-use-lazy','--l1-init-size-gb','4','--eviction-policy','LRU',
        '--prometheus-port','18767','--http-host','127.0.0.1','--http-port','18768',
        '--metrics-sample-rate','1.0']
    return config,engine,helper,env


def recorded_environment(config, env):
    """Never serialize inherited credentials or unrelated host settings."""
    runtime = RUNTIME_ENVIRONMENT.get(config.get('method'), {})
    safe = set(config.get('environment', {})) | set(runtime) | {
        'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'OMP_WAIT_POLICY', 'MAX_JOBS',
        'PYTHONDONTWRITEBYTECODE', 'TRANSFORMERS_OFFLINE', 'PYTHONPATH'}
    return {k: env[k] for k in sorted(safe) if k in env}


def check_model(model):
    path=Path(model)
    config=json.loads((path/'config.json').read_text())
    expected=dict(model_type='qwen3',num_hidden_layers=36,num_key_value_heads=8,head_dim=128)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('This release requires the recorded Qwen3-8B BF16 KV layout')
    from kairos._runtime.compat.replica import is_qwen3_8b_cache_model
    if not is_qwen3_8b_cache_model(str(path)):
        raise ValueError('Use the pinned HF snapshot path (models--Qwen--Qwen3-8B/snapshots/REVISION); arbitrary model aliases are not validated')
    if 'snapshots' in path.parts and path.name!=REVISION:
        raise ValueError('Checkpoint revision differs from the recorded experiment')
    from kairos.verify import verify_model
    return verify_model(model)


def guard(args, processes, gpu_uuid):
    gpu=next(g for g in harness.smi() if int(g['index'])==args.gpu)
    own,foreign=harness.gpu_process_scope(harness.compute_apps(gpu_uuid),
                                         [p.pid for p in processes])
    rss=0
    for pid in own:
        try:
            status=Path(f'/proc/{pid}/status').read_text().splitlines()
            rss+=sum(int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:'))
        except FileNotFoundError:pass
    available=harness.available_ram()
    row=dict(time=time.time(),gpu=gpu,foreign_gpu_pids=sorted(foreign),
             owned_rss_bytes=rss,available_ram_bytes=available)
    if foreign or available<8*1024**3 or rss>32*1024**3 or int(gpu['used'])>30*1024:
        raise RuntimeError('Resource guard violated: '+json.dumps(row))
    if any(p.poll() is not None for p in processes):
        raise RuntimeError('Owned serving process exited')
    return row


def wait_ready(url, timeout, processes, args, uuid, monitoring):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        monitoring.append(guard(args,processes,uuid))
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                if response.status==200:return
        except (OSError,urllib.error.URLError):pass
        time.sleep(1)
    raise RuntimeError('Serving readiness timeout')


async def guarded_replay(args, namespace, programs, out, processes, uuid, monitoring):
    async def watchdog():
        while True:
            monitoring.append(await asyncio.to_thread(guard,args,processes,uuid))
            await asyncio.sleep(2)
    watch=asyncio.create_task(watchdog())
    work=asyncio.create_task(teacher_replay(namespace,programs,out))
    try:
        done,_=await asyncio.wait({watch,work},return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception():raise task.exception()
        return await work
    finally:
        watch.cancel();work.cancel()
        await asyncio.gather(watch,work,return_exceptions=True)


def run(args):
    out=args.output.resolve()
    config,engine,helper,env=prepare(args,out,'/tmp/kairos-placeholder/control.sock')
    if args.dry_run:
        print(json.dumps(dict(engine=engine,helper=helper,environment=recorded_environment(config,env),
            caution='Dry run only; no packages, GPU, checkpoint or ports validated'),indent=2))
        return
    from kairos.verify import verify_checkout, verify_workload
    verify_checkout()
    verify_workload()
    verified_model = check_model(args.model)
    if args.engine_cpus==args.helper_cpus or affinity(args.engine_cpus)&affinity(args.helper_cpus):
        raise ValueError('Engine/helper CPU sets must not overlap')
    if not affinity(args.engine_cpus+', '+args.helper_cpus)<=os.sched_getaffinity(0):
        raise ValueError('Selected CPU affinity is outside the allowed cpuset')
    # Check all shared local ports before creating any process.
    for port in (18765,18766,18767,18768):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:
                raise RuntimeError(f'Port {port} is already occupied')
    gpu=next(g for g in harness.smi() if int(g['index'])==args.gpu)
    if int(gpu['used'])>128 or int(gpu['util'])>5 or harness.compute_apps(gpu['uuid']):
        raise RuntimeError('This reproduction requires an idle dedicated GPU')
    if harness.available_ram()<40*1024**3:
        raise RuntimeError('Require at least 40 GiB available host RAM before startup')
    out.mkdir(parents=True,exist_ok=False)
    processes=[];monitoring=[];files=[]
    state=dict(status='starting',method=args.method,started_at=time.time(),gpu=gpu,
        packages={p:importlib.metadata.version(p) for p in ('torch','vllm','lmcache','transformers','httpx')},
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        reload_sha256=hashlib.sha256(args.reload_profile.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256((ROOT/'data/workload/trace.json').read_bytes()).hexdigest(),
        expected_model_revision=REVISION,verified_model=verified_model,
        packaging_gpu_revalidation=True)
    harness.dump(out/'state.json',state)
    previous_visible=os.environ.get('CONTINUUM_WORK_MESSAGES')
    try:
        with tempfile.TemporaryDirectory(prefix='kairos-control-') as temporary:
            config,engine,helper,env=prepare(args,out,Path(temporary)/'control.sock')
            harness.dump(out/'launch.json',dict(engine=engine,helper=helper,
                environment=recorded_environment(config,env)))
            for name,path in [('profile.json',args.profile),('reload-calibration.json',args.reload_profile)]:
                (out/name).write_bytes(path.read_bytes())
            for command,name,url in ((helper,'helper','http://127.0.0.1:18768/metrics'),
                                     (engine,'server','http://127.0.0.1:18765/health')):
                log=(out/(name+'.log')).open('x');files.append(log)
                processes.append(subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True))
                wait_ready(url,args.start_timeout,processes,args,gpu['uuid'],monitoring)
            namespace=SimpleNamespace(model=args.model,port=18765,max_num_seqs=8,
                request_timeout=120,run_timeout=900,reuse_engine=True,gapped_reuse=True,
                current_method=RUNTIME_ACTIVATION[args.method]+'~heterogeneous',current_root=str(out))
            harness.warmup(namespace)
            from kairos._runtime.layout.correctness import validate
            validate(namespace,processes[0],out)
            harness.reset_owned_lmcache(processes[0],out)
            harness.snapshot_metrics(namespace,out,'before',True)
            programs=json.loads((ROOT/'data/workload/trace.json').read_bytes())['programs']
            # Sidecar is read only by the client, which sends the arrived prefix.
            os.environ['CONTINUUM_WORK_MESSAGES']=str(ROOT/'data/workload/visible_messages.json')
            state['status']='replaying';harness.dump(out/'state.json',state)
            result=asyncio.run(guarded_replay(args,namespace,programs,out,processes,gpu['uuid'],monitoring))
            result['method']=args.method
            harness.dump(out/'summary.json',result)
            harness.snapshot_metrics(namespace,out,'after',True)
            state.update(status='complete',summary=result)
            print(json.dumps(result,indent=2))
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        if previous_visible is None:os.environ.pop('CONTINUUM_WORK_MESSAGES',None)
        else:os.environ['CONTINUUM_WORK_MESSAGES']=previous_visible
        for process in reversed(processes):harness.stop(process)
        for file in files:file.close()
        state['ended_at']=time.time()
        harness.dump(out/'state.json',state);harness.dump(out/'resources.json',monitoring)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('kairos','continuum'),required=True)
    parser.add_argument('--model',required=True,help='Pinned local Qwen3-8B HF snapshot directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--engine-cpus',default='4-5,68-69')
    parser.add_argument('--helper-cpus',default='6-7,70-71')
    parser.add_argument('--profile',type=Path,default=ROOT/'assets/deployment_profile.json')
    parser.add_argument('--reload-profile',type=Path,default=ROOT/'assets/reload_profile.json')
    parser.add_argument('--start-timeout',type=int,default=2400)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    def terminate(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    run(args)


if __name__=='__main__':main()
