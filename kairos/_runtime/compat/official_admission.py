"""Kairos official admission implementation."""
from __future__ import annotations

import ast
from collections import deque
import hashlib
from pathlib import Path
import time
import types

SOURCE_SHA = "b21b12a07a3ccf473c82acf00820b1aa2073044b"
SOURCE_BLOB = "ba0cd04263ba43c488ead58b916dd7985105af5c"


class ResolvedLookupBase:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self._resolved_result


class AdmissionStats:
    def __init__(self):
        self.rows = []

    def hit(self, event, **counts):
        self.rows.append(dict(event=event, **counts))


def load_official_rule(mode, chunk_tokens, bytes_per_token, threshold=8):
    path = Path(__file__).resolve().parents[1] / (
        "vendor/efficientagent-b21b12a/connector.py")
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise RuntimeError("EfficientAgent source does not match the audited Git blob")
    source = ast.parse(data, filename=str(path))
    original = next(n for n in source.body if isinstance(n, ast.ClassDef)
                    and n.name == "EfficientAgentConnector")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_pressure", "get_num_new_matched_tokens")]
    setter = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                  and n.name == "set_skip_save")
    cls = ast.ClassDef(name=original.name,
        bases=[ast.Name(id="ResolvedLookupBase", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[setter, cls], type_ignores=[]))
    scope = dict(ResolvedLookupBase=ResolvedLookupBase, time=time, json=__import__("json"),
        ADMISSION=mode, CHUNK=chunk_tokens, BYTES_PER_TOKEN=bytes_per_token,
        KAPPA=threshold, THETA=.95, WINDOW_S=60., TELEMETRY_READ_S=.5,
        TELEMETRY_MAX_AGE_S=5., TELEMETRY_FILE=None, DECISION_CACHE=65536)
    exec(compile(module, str(path), "exec"), scope)
    return scope["EfficientAgentConnector"]


class OfficialAdmission:
    def __init__(self, mode, host_gib, chunk_tokens, bytes_per_token, telemetry_file,
                 threshold=8):
        if mode not in ("fixed", "conditioned"):
            raise ValueError("Use an official EfficientAgent admission mode")
        cls = load_official_rule(mode, chunk_tokens, bytes_per_token, threshold)
        # All policy state mirrors upstream __init__; no inherited engine
        # from the other LMCache API is constructed.
        self.rule = object.__new__(cls)
        self.rule._ea_stats = AdmissionStats()
        self.rule._ea_decided = {}
        self.rule._ea_tasks = {}
        self.rule._ea_lens = deque(maxlen=256)
        self.rule._ea_telemetry = (0., None)
        self.rule._ea_host_bytes = host_gib * 2**30
        self.rule._lmcache_engine = types.SimpleNamespace(load_specs={})
        cls._pressure.__globals__["TELEMETRY_FILE"] = str(telemetry_file)
        self.mode = mode

    def resolve(self, request, computed_tokens, result, host_end, task):
        if result[0] is None:
            return None, []
        sp = request.sampling_params
        extra = dict(sp.extra_args or {})
        params = dict(extra.get("kv_transfer_params") or {})
        # The replay's Program ID is the official task ID equivalent.
        params["efficientagent.task_id"] = task
        sp.extra_args = dict(extra, kv_transfer_params=params)
        self.rule._resolved_result = result
        self.rule._lmcache_engine.load_specs[request.request_id] = types.SimpleNamespace(
            lmcache_cached_tokens=host_end)
        start = len(self.rule._ea_stats.rows)
        returned = self.rule.get_num_new_matched_tokens(request, computed_tokens)
        if returned != result:
            raise RuntimeError("Admission unexpectedly changed the native lookup")
        return self.rule._ea_decided[request.request_id], self.rule._ea_stats.rows[start:]
