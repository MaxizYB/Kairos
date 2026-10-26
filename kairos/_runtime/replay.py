import asyncio, hashlib, json, math, os, signal, subprocess, time, urllib.request
from pathlib import Path

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def snapshot_metrics(args, directory, suffix, lmcache=False):
    endpoints = [(args.port, "metrics")]
    if lmcache:
        endpoints.append((18768, "lmcache-metrics"))
    for port, name in endpoints:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            (directory / f"{name}-{suffix}.txt").write_bytes(response.read())


def drain_owned_lmcache(process):
    """Wait until this run's MP transfers finish without changing cache state."""
    if process.poll() is not None or os.getpgid(process.pid) != process.pid:
        raise RuntimeError("The run no longer owns a live private LMCache helper")
    stable = 0
    until = time.monotonic()+30
    values = {}
    while time.monotonic()<until:
        with urllib.request.urlopen("http://127.0.0.1:18768/metrics",timeout=5) as response:
            raw=response.read().decode()
        values={}
        for line in raw.splitlines():
            if line.startswith("lmcache_mp_num_"):
                key,value=line.rsplit(None,1)
                key=key.split("{",1)[0]
                values[key]=values.get(key,0)+float(value)
        idle = mp_transfers_idle(values)
        stable=stable+1 if idle else 0
        if stable>=3:
            return values
        time.sleep(.2)
    raise RuntimeError(f"Outstanding LMCache transfers; refuse cache operation: {values}")


def mp_transfers_idle(values):
    """OTel omits a counter series until its first event occurs.

    A missing *pair* is zero work, but one missing side or an entirely empty
    exporter is not evidence of a drain. Warmup emits stores before the first
    retrieve; requiring a retrieve series then incorrectly waits forever.
    """
    observed = False
    for kind in ("stores", "retrieves"):
        submitted = values.get(f"lmcache_mp_num_submitted_{kind}_total")
        finished = values.get(f"lmcache_mp_num_finished_{kind}_total")
        if submitted is None and finished is None:
            continue
        observed = True
        if submitted is None or finished is None or submitted != finished:
            return False
    return observed


def reset_owned_lmcache(process, directory, label=None):
    """Drain our private MP helper before clearing L1 between comparisons."""
    values = drain_owned_lmcache(process)
    request=urllib.request.Request("http://127.0.0.1:18768/cache/clear",
        data=b'{"tier":"l1","force":true}',headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=10) as response:
        result=json.load(response)
    if result.get("status")!="ok":raise RuntimeError("LMCache L1 reset failed")
    filename = "lmcache-reset.json" if label is None else f"lmcache-reset-{label}.json"
    dump(directory/filename,dict(private_helper_pid=process.pid,
        drained_counters=values,result=result))


def reset_prefix_cache(port, timeout=30):
    """Reset vLLM APC and require the engine's success acknowledgement."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/reset_prefix_cache",
            data=b"", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.load(response)
            if result.get("success") is True:
                return result
            last = result
        except Exception as exc:
            last = str(exc)
        time.sleep(.2)
    raise RuntimeError(f"vLLM prefix-cache reset did not succeed: {last}")


def warmup(args):
    """Warm identical shapes before timing; measured methods reset APC."""
    for i, length in enumerate((512, 2048, 512)):
        payload = dict(model=args.model, prompt=[100 + j % 1000 for j in range(length)],
                       max_tokens=4, ignore_eos=True, temperature=0, seed=1,
                       cache_salt=f"continuum-warmup-{i}",
                       vllm_xargs=dict(cont_program_id=f"warmup-{i}", cont_final=True))
        if args.reuse_engine:
            payload["vllm_xargs"]["cont_experiment"] = "vllm-warmup"
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            response.read()
    # In the native path there is no experiment switch to clear warmup APC.
    if not args.reuse_engine:
        reset_prefix_cache(args.port)


def smi():
    out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"],text=True,timeout=10)
    return [dict(zip(("index","uuid","total","used","util"),r.split(", "))) for r in out.strip().splitlines()]


def available_ram():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
    return 0


def compute_apps(uuid):
    out=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                 "--format=csv,noheader,nounits"],text=True,timeout=10)
    return {int(row.split(", ")[1]) for row in out.splitlines() if row.startswith(uuid+", ")}


def descendants(pid):
    result={pid}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        try:
            if os.getpgid(int(path.name))==pid:result.add(int(path.name))
        except (ProcessLookupError,PermissionError):pass
    return result


def gpu_process_scope(apps, roots, baseline_foreign=()):
    """Read owned groups AFTER the GPU snapshot, including newly born workers.

    Taking /proc first can mislabel a child that initializes CUDA while the
    subsequent nvidia-smi command runs. Exact private process groups remain
    the ownership boundary; a matching username is never sufficient.
    """
    own = set().union(*(descendants(pid) for pid in roots))
    return own, set(apps)-own-set(baseline_foreign)


def process_identity(pid):
    """Bounded read-only identity evidence before a transient PID disappears."""
    row = dict(pid=pid)
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        for line in status:
            key, _, value = line.partition(":")
            if key in ("Name", "PPid", "Uid"):
                row[key.lower()] = value.strip()
        row["pgid"] = os.getpgid(pid)
        row["argv_prefix"] = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            errors="replace").split("\0")[:4]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        row["identity_unavailable"] = True
    return row


def stop(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)




