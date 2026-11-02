"""replay.execution."""

import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache

PROTOCOL = "fixed-work-private-stop-v1"
ENV = "CONTINUUM_REPLAY_EXECUTION_MANIFEST"


def replay_key(experiment, program, turn):
    return hashlib.sha256(
        json.dumps([experiment, program, turn], separators=(",", ":")).encode()
    ).hexdigest()


def prepare_manifest(groups, path, max_model_len, output_limit):
    budgets = {}
    for experiment, programs in groups:
        for program in programs:
            prompt = []
            for stage, part in enumerate(program["turns"]):
                prompt = (
                    prompt[: part.get("keep_prefix_tokens", len(prompt))]
                    + part["added_token_ids"]
                )
                count = part["output_tokens"]
                if (
                    type(count) is not int
                    or count < 1
                    or count > min(output_limit, max_model_len - len(prompt))
                ):
                    raise ValueError(
                        "Replay work does not fit the public context/ceiling"
                    )
                key = replay_key(experiment, program["id"], stage)
                if key in budgets:
                    raise ValueError("Duplicate private replay identity")
                budgets[key] = count
                if program.get("append_generated_outputs"):
                    raise ValueError(
                        "Private fixed replay requires recorded next prompts"
                    )
    path = Path(path)
    path.write_text(json.dumps(dict(protocol=PROTOCOL, budgets=budgets)))
    path.chmod(384)
    return dict(
        protocol=PROTOCOL,
        requests=len(budgets),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@lru_cache(maxsize=4)
def _execution_budgets(path):
    manifest = json.loads(Path(path).read_bytes())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Wrong private replay execution protocol")
    budgets = manifest["budgets"]
    if any((type(v) is not int or v < 1 for v in budgets.values())):
        raise ValueError("Invalid private replay execution budget")
    return budgets


def update_output(request, token_ids, native_update, finished_status, *args, **kwargs):
    metadata = getattr(request.sampling_params, "extra_args", None) or {}
    key = metadata.get("cont_replay_key")
    if key is None:
        return native_update(request, token_ids, *args, **kwargs)
    path = os.environ.get(ENV)
    if not path:
        raise ValueError("Missing private Replay executor manifest")
    count = _execution_budgets(path)[key]
    params = request.sampling_params
    if (
        not params.ignore_eos
        or getattr(params, "stop", None)
        or getattr(params, "stop_token_ids", None)
        or (count > request.max_tokens)
    ):
        raise ValueError(
            "Fixed Replay needs EOS disabled and a sufficient public ceiling"
        )
    remaining = count - len(request.output_token_ids)
    if remaining <= 0:
        raise ValueError("Completed replay request executed again")
    output, stopped = native_update(
        request, list(token_ids[:remaining]), *args, **kwargs
    )
    observed = len(request.output_token_ids)
    if observed == count:
        request.status = finished_status
        return (output, True)
    if stopped:
        raise ValueError(
            "Native execution stopped before prescribed Replay work completed"
        )
    return (output, False)
