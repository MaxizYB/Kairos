"""runtime.replay."""

import asyncio, hashlib, json, os, signal, subprocess, time, urllib.request
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def snapshot_metrics(args, directory, suffix, lmcache=False):
    endpoints = [(args.port, "metrics")]
    if lmcache:
        endpoints.append((18768, "lmcache-metrics"))
    for port, name in endpoints:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=5
        ) as response:
            (directory / f"{name}-{suffix}.txt").write_bytes(response.read())


def drain_owned_lmcache(process):
    if process.poll() is not None or os.getpgid(process.pid) != process.pid:
        raise RuntimeError("The run no longer owns a live private LMCache helper")
    stable = 0
    until = time.monotonic() + 30
    values = {}
    while time.monotonic() < until:
        with urllib.request.urlopen(
            "http://127.0.0.1:18768/metrics", timeout=5
        ) as response:
            raw = response.read().decode()
        values = {}
        for line in raw.splitlines():
            if line.startswith("lmcache_mp_num_"):
                key, value = line.rsplit(None, 1)
                key = key.split("{", 1)[0]
                values[key] = values.get(key, 0) + float(value)
        idle = mp_transfers_idle(values)
        stable = stable + 1 if idle else 0
        if stable >= 3:
            return values
        time.sleep(0.2)
    raise RuntimeError(
        f"Outstanding LMCache transfers; refuse cache operation: {values}"
    )


def mp_transfers_idle(values):
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
    values = drain_owned_lmcache(process)
    request = urllib.request.Request(
        "http://127.0.0.1:18768/cache/clear",
        data=b'{"tier":"l1","force":true}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    if result.get("status") != "ok":
        raise RuntimeError("LMCache L1 reset failed")
    filename = "lmcache-reset.json" if label is None else f"lmcache-reset-{label}.json"
    dump(
        directory / filename,
        dict(private_helper_pid=process.pid, drained_counters=values, result=result),
    )


def reset_prefix_cache(port, timeout=30):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.load(response)
            if result.get("success") is True:
                return result
            last = result
        except Exception as exc:
            last = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"vLLM prefix-cache reset did not succeed: {last}")


def warmup(args):
    for i, length in enumerate((512, 2048, 512)):
        payload = dict(
            model=args.model,
            prompt=[100 + j % 1000 for j in range(length)],
            max_tokens=4,
            ignore_eos=True,
            temperature=0,
            seed=1,
            cache_salt=f"continuum-warmup-{i}",
            vllm_xargs=dict(cont_program_id=f"warmup-{i}", cont_final=True),
        )
        if args.reuse_engine:
            payload["vllm_xargs"]["cont_experiment"] = "vllm-warmup"
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            response.read()
    if (
        getattr(args, "replay_execution", None)
        and args.reuse_engine
        and (not getattr(args, "native_scheduler", False))
    ):
        from kairos.replay.execution import replay_key
        from kairos.replay.boundary import generation_parameters

        name = "warmup-fixed-replay-boundary"
        public = generation_parameters(
            512,
            args.max_model_len,
            int(os.environ.get("CONTINUUM_REPLAY_OUTPUT_LIMIT", "16384")),
        )
        count = min(4, public["max_tokens"])
        payload = dict(
            model=args.model,
            prompt=[100 + j % 1000 for j in range(512)],
            **public,
            temperature=0,
            seed=1,
            return_token_ids=True,
            cache_salt=name,
            vllm_xargs=dict(
                cont_program_id=name,
                cont_tool="unknown",
                cont_experiment="vllm-warmup",
                cont_replay_key=replay_key("vllm-warmup", name, 0),
            ),
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.load(response)
        if (
            result.get("usage", {}).get("completion_tokens") != count
            or sum((len(c.get("token_ids") or []) for c in result.get("choices", [])))
            != count
        ):
            raise RuntimeError(
                "Fixed Replay executor failed real startup boundary check"
            )
        dump(
            Path(args.current_root) / "fixed-replay-preflight.json",
            dict(
                passed=True,
                prescribed_tokens=count,
                public_parameters=public,
                measured_workload=False,
            ),
        )
    if not args.reuse_engine:
        reset_prefix_cache(args.port)


def smi():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return [
        dict(zip(("index", "uuid", "total", "used", "util"), r.split(", ")))
        for r in out.strip().splitlines()
    ]


def available_ram():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def compute_apps(uuid):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    return {
        int(row.split(", ")[1])
        for row in out.splitlines()
        if row.startswith(uuid + ", ")
    }


def descendants(pid):
    result = {pid}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if os.getpgid(int(path.name)) == pid:
                result.add(int(path.name))
        except (ProcessLookupError, PermissionError):
            pass
    return result


def gpu_process_scope(apps, roots, baseline_foreign=()):
    own = set().union(*(descendants(pid) for pid in roots))
    return (own, set(apps) - own - set(baseline_foreign))


def process_identity(pid):
    row = dict(pid=pid)
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        for line in status:
            key, _, value = line.partition(":")
            if key in ("Name", "PPid", "Uid"):
                row[key.lower()] = value.strip()
        row["pgid"] = os.getpgid(pid)
        row["argv_prefix"] = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .decode(errors="replace")
            .split("\x00")[:4]
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        row["identity_unavailable"] = True
    return row


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


async def replay(args, programs, run_dir):
    if programs and programs[0].get("bfcl_task"):
        raise ValueError("Only recorded SWE Replay is supported")
    import httpx
    from kairos.replay.boundary import (
        arrived_metadata,
        generation_parameters,
        validate_arrived_metadata,
    )
    from kairos.replay.execution import replay_key, PROTOCOL

    records = []
    jobs = []
    primed = [row for row in programs if row.get("prime_barrier", False)]
    if primed and len(primed) != len(programs):
        raise ValueError(
            "Priming barrier must include every program in a diagnostic trace"
        )
    prime_barrier = asyncio.Barrier(len(programs)) if primed else None
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.request_timeout, connect=10),
        limits=httpx.Limits(max_connections=max(32, 2 * args.max_num_seqs)),
    ) as client:
        start = time.monotonic()
        start_wall = time.time()

        async def program(row):
            await asyncio.sleep(max(0, row["arrival_s"] - (time.monotonic() - start)))
            job_start = time.monotonic()
            prompt = []
            first_turn_end = None
            prime_end = None
            if row.get("probe_after"):
                if not getattr(args, "recovery_probe", False):
                    raise ValueError(
                        "Diagnostic arrival barrier requires --recovery-probe"
                    )
                target = row["probe_after"]
                path = Path(args.current_root) / "partial/scheduler.jsonl"
                deadline = time.monotonic() + 15
                cursor = 0
                while True:
                    matched = False
                    if path.exists():
                        with path.open() as stream:
                            stream.seek(cursor)
                            for line in stream:
                                event = json.loads(line)
                                if event.get("experiment") != args.current_method:
                                    continue
                                if target["stage"] == "pin":
                                    matched |= (
                                        event["event"] == "pin"
                                        and event.get("program") == target["program"]
                                    )
                                elif target["stage"] == "decode":
                                    matched |= event[
                                        "event"
                                    ] == "batch_schedule" and any(
                                        (
                                            member["program"] == target["program"]
                                            and (not member["prefill_queries"])
                                            for member in event.get("members", [])
                                        )
                                    )
                                else:
                                    raise ValueError("Unknown diagnostic arrival stage")
                            cursor = stream.tell()
                    if matched:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Diagnostic arrival barrier timed out")
                    await asyncio.sleep(0.01)
            for turn, part in enumerate(row["turns"]):
                tool_runtime_s = 0.0
                tool_digest = None
                if part.get("runtime_tool"):
                    raise ValueError("Only recorded SWE Replay is supported")
                if "keep_prefix_tokens" in part:
                    prompt = prompt[: part["keep_prefix_tokens"]]
                prompt += part["added_token_ids"]
                meta = arrived_metadata(row["id"])
                if not getattr(args, "gpu_profile", False):
                    meta["cont_replay_key"] = replay_key(
                        args.current_method, row["id"], turn
                    )
                if os.environ.get("CONTINUUM_WORK_MESSAGES"):
                    visible = getattr(args, "_work_visible_messages", None)
                    if visible is None:
                        visible = json.loads(
                            Path(os.environ["CONTINUUM_WORK_MESSAGES"]).read_bytes()
                        )
                        args._work_visible_messages = visible
                    meta["cont_visible_messages"] = json.dumps(
                        visible[row["id"]][turn], separators=(",", ":")
                    )
                elif "visible_messages" in part:
                    meta["cont_visible_messages"] = json.dumps(
                        part["visible_messages"], separators=(",", ":")
                    )
                if args.reuse_engine:
                    meta["cont_experiment"] = args.current_method
                if (
                    turn == 0
                    and row["id"] == programs[0]["id"]
                    and getattr(args, "synthetic_tool_history", None)
                ):
                    meta["cont_bfcl_reset_history"] = json.dumps(
                        args.synthetic_tool_history
                    )
                validate_arrived_metadata(meta)
                generation = generation_parameters(
                    len(prompt),
                    args.max_model_len,
                    int(os.environ.get("CONTINUUM_REPLAY_OUTPUT_LIMIT", "16384")),
                )
                payload = dict(
                    model=args.model,
                    prompt=prompt,
                    **generation,
                    temperature=0,
                    seed=1,
                    stream=True,
                    return_token_ids=True,
                    stream_options={"include_usage": True},
                    cache_salt=row["id"],
                    vllm_xargs=meta,
                )
                req_start = time.monotonic()
                stamps = []
                output = []
                usage = {}
                error = None
                output_text = []
                finish_reasons = []
                try:
                    async with client.stream(
                        "POST",
                        f"http://127.0.0.1:{args.port}/v1/completions",
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            raise RuntimeError((await response.aread()).decode()[:1200])
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            item = json.loads(line[6:])
                            if "error" in item:
                                raise RuntimeError(str(item["error"]))
                            for choice in item.get("choices", []):
                                if choice.get("finish_reason"):
                                    finish_reasons.append(choice["finish_reason"])
                                output_text.append(choice.get("text", ""))
                                ids = choice.get("token_ids") or []
                                if ids:
                                    stamps.append((time.monotonic(), len(ids)))
                                    output += ids
                            if item.get("usage"):
                                usage = item["usage"]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc!s}"
                end = time.monotonic()
                record = dict(
                    program=row["id"],
                    turn=turn,
                    prompt_tokens=len(prompt),
                    output_tokens=len(output),
                    output_ids=output,
                    usage=usage,
                    error=error,
                    start_s=req_start - start,
                    end_s=end - start,
                    e2e_s=end - req_start,
                    ttft_s=stamps[0][0] - req_start if stamps else None,
                    itl_s=[
                        b[0] - a[0]
                        for a, b in zip(stamps, stamps[1:])
                        if a[1] == b[1] == 1
                    ],
                    stream_chunks=[n for _, n in stamps],
                    output_text="".join(output_text),
                    finish_reasons=finish_reasons,
                    generation_parameters=generation,
                    planned_output_tokens=part["output_tokens"],
                    request_metadata_keys=sorted(meta),
                    request_tool=meta.get("cont_tool"),
                    prompt_sha256=hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
                    tool_runtime_s=tool_runtime_s,
                    tool_sha256=tool_digest,
                )
                records.append(record)
                with (run_dir / "requests.jsonl").open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if error:
                    break
                if len(output) != part["output_tokens"]:
                    raise RuntimeError(
                        f"Incomplete fixed Replay output count: {len(output)} != {part['output_tokens']}"
                    )
                if turn == 0:
                    first_turn_end = end
                if row.get("append_generated_outputs", False):
                    prompt += output
                if turn == 0 and prime_barrier is not None:
                    await prime_barrier.wait()
                    prime_end = time.monotonic()
                if turn < len(row["turns"]) - 1:
                    await asyncio.sleep(part["gap_after_s"])
            job_end = time.monotonic()
            jobs.append(
                dict(
                    program=row["id"],
                    jct_s=job_end - job_start,
                    arrival_s=job_start - start,
                    turns=len(row["turns"]),
                    first_turn_end_s=first_turn_end - start if first_turn_end else None,
                    post_initial_jct_s=job_end - first_turn_end
                    if first_turn_end
                    else None,
                    primed_jct_s=job_end - prime_end if prime_end else None,
                )
            )

        await asyncio.wait_for(
            asyncio.gather(*(program(p) for p in programs)), args.run_timeout
        )
    wall = time.monotonic() - start

    def pct(xs, p):
        xs = sorted(xs)
        if not xs:
            return None
        pos = (len(xs) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo) if hi != lo else xs[lo]

    good = [r for r in records if r["error"] is None]
    primed_jct = [j["primed_jct_s"] for j in jobs if j.get("primed_jct_s") is not None]
    ttft = [r["ttft_s"] for r in good if r["ttft_s"] is not None]
    itl = [x for r in good for x in r["itl_s"]]
    e2e = [r["e2e_s"] for r in good]
    summary = dict(
        method=args.current_method,
        wall_s=wall,
        start_wall_time=start_wall,
        generation_protocol=PROTOCOL,
        public_output_limit=int(
            os.environ.get("CONTINUUM_REPLAY_OUTPUT_LIMIT", "16384")
        ),
        end_wall_time=time.time(),
        completed=len(good),
        failed=len(records) - len(good),
        jobs=len(jobs),
        jct_mean_s=sum((j["jct_s"] for j in jobs)) / len(jobs),
        jct_p95_s=pct([j["jct_s"] for j in jobs], 0.95),
        primed_jct_mean_s=sum(primed_jct) / len(primed_jct) if primed_jct else None,
        primed_jct_p95_s=pct(primed_jct, 0.95),
        ttft_mean_s=sum(ttft) / len(ttft) if ttft else None,
        ttft_p95_s=pct(ttft, 0.95),
        itl_mean_s=sum(itl) / len(itl) if itl else None,
        itl_p95_s=pct(itl, 0.95),
        e2e_mean_s=sum(e2e) / len(e2e) if e2e else None,
        e2e_p95_s=pct(e2e, 0.95),
        request_throughput=len(good) / wall,
        output_token_throughput=sum((r["output_tokens"] for r in good)) / wall,
        total_token_throughput=sum(
            (r["output_tokens"] + r["prompt_tokens"] for r in good)
        )
        / wall,
    )
    dump(run_dir / "jobs.json", jobs)
    dump(run_dir / "summary.json", summary)
    if summary["failed"] or len(good) != sum((len(p["turns"]) for p in programs)):
        raise RuntimeError("Incomplete replay; timings are diagnostic only")
    return summary
