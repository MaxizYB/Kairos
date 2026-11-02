"""Native-engine doubles for exercising the actual release controller on CPU."""

import ast
from collections import deque
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Queue(list):
    def add_request(self, request):
        self.append(request)
        self.sort(key=lambda r: r.priority)


class NativeBoundary:
    def _emit(self, event, **fields):
        self.events.append(dict(event=event, experiment=self.experiment, **fields))

    def program(self, request):
        return request.program

    def metadata(self, request):
        return request.meta

    def _switch_experiment(self, name):
        self.experiment = name
        self.online_mode = "onlinefitresidual"

    def _online_grant(self, request, pages, reason):
        p = request.program
        self.online_entered.add(p)
        self.online_active.add(p)
        self.online_reserved[p] = pages
        self.online_granted.add(request.request_id)
        self._emit(
            "online_path_grant",
            program=p,
            request=request.request_id,
            pages=pages,
            reason=reason,
        )

    def _online_idle_reclaimable(self, p):
        return True

    def _replica_revoke(self, program, preserve_borrowed=False):
        self._replica_call(
            dict(op="revoke", program=program, preserve_borrowed=preserve_borrowed)
        )

    def _replica_grant(self, pin):
        self._replica_call(dict(op="grant", program=pin.program))

    def _clear_continuity_lease(self, program, reason):
        self.continuity_leases.pop(program, None)


def native_release():
    path = ROOT / "kairos/_runtime/compat/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ContinuumScheduler"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_release"
    )
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_release"]


NativeBoundary._release = native_release()


def controller_type():
    module = ModuleType("kairos._runtime.compat.online_path_scheduler")
    module.OnlinePathScheduler = NativeBoundary
    module.FinitePathProtection = lambda policy, forecaster: policy
    queue = ModuleType("vllm.v1.core.sched.request_queue")
    queue.create_request_queue = lambda policy: Queue()
    with patch.dict(sys.modules, {module.__name__: module, queue.__name__: queue}):
        return importlib.import_module("kairos.scheduler").KairosScheduler


def request(p, prompt=64, cap=8192):
    return SimpleNamespace(
        program=p,
        request_id=p + "-call",
        prompt_token_ids=list(range(prompt)),
        num_prompt_tokens=prompt,
        output_token_ids=[],
        num_computed_tokens=0,
        num_output_placeholders=0,
        max_tokens=cap,
        sampling_params=SimpleNamespace(
            ignore_eos=True, stop=[], stop_token_ids=[], extra_args={}
        ),
        priority=0,
        meta={},
        is_finished=lambda: False,
    )


class Pool:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.hash_block_size = 16
        self.blocks = {}
        self.freed = []

    def get_num_free_blocks(self):
        return (
            self.scheduler._capacity()
            - sum(b.ref_cnt > 0 for b in self.blocks.values())
            - sum(self.scheduler.physical.values())
        )

    def free_blocks(self, blocks):
        for block in blocks:
            if block.ref_cnt < 1:
                raise RuntimeError("Double free")
            block.ref_cnt -= 1
            self.freed.append(block.block_id)

    def get_cached_block(self, key, groups):
        matches = [b for b in self.blocks.values() if b.block_hash == key]
        return [matches[0]] if matches else None


def attach_pin(s, p, pages, common=None):
    pool = s.kv_cache_manager.block_pool
    hashes = [f"{p}/old/{k}".encode() for k in range(pages)]
    if p in s.online_requests:
        r = s.online_requests[p]
        common = pages if common is None else common
        r.cache_salt = "private/" + p
        r.block_hashes = [
            hashes[k] if k < min(common, pages) else f"{p}/new/{k}".encode()
            for k in range(r.num_prompt_tokens // 16)
        ]
    blocks = []
    for value in hashes:
        bid = len(pool.blocks) + 1
        block = SimpleNamespace(
            block_id=bid, block_hash=value, ref_cnt=1, is_null=False
        )
        pool.blocks[bid] = block
        blocks.append(block)
    pin = SimpleNamespace(
        program=p,
        blocks=blocks,
        lookup_hashes=hashes,
        deadline=100.0,
        started=1.0,
        tool="observed-tool",
        tokens=pages * 16,
        token_ids=tuple(range(pages * 16)),
        model="cpu-model",
        cache_salt="private/" + p,
    )
    s.pins[p] = pin
    s.online_reserved[p] = pages
    return pin


def fixture(stages=(0, 1, 0), idle=False):
    from kairos.prediction.prior import ServicePrior

    cls = controller_type()
    s = cls.__new__(cls)
    s.program_enabled = True
    s.experiment = s.METHOD + "~heterogeneous"
    s.events, s.order = [], {}
    s.waiting, s.skipped_waiting, s.running = Queue(), Queue(), []
    s.policy = "priority"
    for name in (
        "pins",
        "online_requests",
        "online_inputs",
        "online_outputs",
        "online_reserved",
        "online_cleaning",
        "online_insurance",
        "program_arrivals",
        "program_predictions",
        "program_completed",
        "program_completed_prefix",
        "program_keeps",
        "program_priorities",
        "program_boundaries",
        "forecast_output_cache",
        "physical",
        "continuity_leases",
    ):
        setattr(s, name, {})
    for name in (
        "online_granted",
        "online_entered",
        "online_active",
        "online_ended",
        "program_retries",
        "finished_recving_kv_req_ids",
    ):
        setattr(s, name, set())
    s.online_dirty = True
    s.program_pending = deque()
    s.max_model_len, s.block_size, s.max_num_running_reqs = 16384, 16, 8
    s.limit = 100
    s._capacity = lambda: s.limit + 32
    s.reference_profile = dict(
        beta=[0.0, 0.001, 0.0, 0.0], decode_step_by_seqs={"1": 0.01}
    )
    s.beta = [0.0, 0.002, 0.0, 0.0]
    s.profile = dict(
        pipeline_service=dict(beta=list(s.beta), decode_step_by_seqs={"1": 0.02})
    )
    s.forest = SimpleNamespace(training_ids=set(), sha256="cpu-fixture")
    prior = json.loads((ROOT / "assets/program_prior.json").read_bytes())
    s.service_prior = ServicePrior(prior, s._service, s.max_model_len)
    s._prepare_output_forests = lambda tokenizer=None: None
    s.rpc = []

    def rpc(payload):
        s.rpc.append(payload)
        return {"programs": {}}

    s._replica_call = rpc
    s.connector = SimpleNamespace(
        scheduler_adapter=SimpleNamespace(model_name="cpu-model"), repair_enabled=False
    )
    s.kv_cache_manager = SimpleNamespace(
        get_blocks=lambda rid: SimpleNamespace(
            blocks=[list(range(s.physical.get(rid, 0)))]
        )
    )
    s.kv_cache_manager.block_pool = Pool(s)

    def hits(r):
        blocks = []
        for key in getattr(r, "block_hashes", ())[: (r.num_prompt_tokens - 1) // 16]:
            found = s.kv_cache_manager.block_pool.get_cached_block(key, [0])
            if found is None:
                break
            blocks.append(found[0])
        return SimpleNamespace(blocks=[blocks]), len(blocks) * 16, None

    s.kv_cache_manager.get_computed_blocks = hits
    for p, stage in zip(("a", "b", "c"), stages):
        r = request(p)
        s.order[p] = len(s.order)
        s.online_inputs[p] = [32, 64] if stage else [64]
        s.online_outputs[p] = [3] if stage else []
        s.online_requests[p] = r
        s.program_keeps[p] = r.request_id, 0
        s.program_arrivals[p] = r.request_id
        if stage:
            s.program_predictions[p] = dict(
                request=r.request_id, post=dict(atoms=[100.0], mass=[1.0], pred=100.0)
            )
        post = dict(
            atoms=[3.0, 40.0, 200.0],
            mass=[0.2, 0.6, 0.2],
            source="current-output-forest",
        )
        s.forecast_output_cache[p] = dict(
            request=r.request_id, post=post, marginal=post
        )
        s.waiting.add_request(r)
    if idle:
        r = s.online_requests.pop("c")
        s.waiting.remove(r)
        s.online_outputs["c"].append(10)
        s.program_completed["c"] = dict(
            request=r.request_id,
            stage=1,
            keep=0,
            post=dict(atoms=[2.0, 6.0], mass=[0.4, 0.6], pred=4.4),
            source="stage-conditioned-training-prior",
        )
        s.program_predictions.pop("c", None)
        attach_pin(s, "c", 4)
    return s


def controlled(limit=10):
    """Three mandatory pages, an eight-page idle pin, and a four-page arrival."""
    s = fixture()
    s.waiting.clear()
    s.online_requests.clear()
    s.order = {"guard": 0, "high": 1, "low": 2}
    s.online_reserved.clear()
    s.physical.clear()
    high = request("high", prompt=63)
    s.waiting.add_request(high)
    s.online_requests["high"] = high
    guard = request("guard", prompt=47)
    s.running = [guard]
    s.physical[guard.request_id] = 3
    s.online_reserved["guard"] = 3
    s.limit = limit
    attach_pin(s, "low", 8)
    s._emit = lambda event, **row: NativeBoundary._emit(
        s,
        "forecast_" + event[len("program_") :]
        if event.startswith("program_")
        else event,
        **row,
    )

    def snapshot(queued):
        ready = {r.program: r for r in queued}
        rows = {
            p: dict(
                state="ready" if p in ready else "running" if p == "guard" else "idle"
            )
            for p in s.order
        }
        rank = SimpleNamespace(
            ready=tuple(sorted(ready, key=s.order.__getitem__)),
            priorities=s.order,
            boundaries={},
        )
        return ready, rows, {}, {}, rank

    s._snapshot = snapshot
    return s
