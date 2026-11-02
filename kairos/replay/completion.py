"""replay.completion."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
from kairos.replay.execution import replay_key
from kairos.messages import tool_name

PROTOCOL = "completed-current-action-v1"
SOURCE_SCHEMA = "recorded-current-actions-v1"
SOURCE_ENV = "CONTINUUM_COMPLETION_OBSERVATIONS"
ENV = "CONTINUUM_REPLAY_COMPLETION_MANIFEST"
FIELDS = frozenset({"tool", "terminal", "evidence", "assistant_sha256"})


def _commands(message):
    actions = message.get("extra", {}).get("actions", [])
    if actions:
        return [a["command"] for a in actions]
    result = []
    for call in message.get("tool_calls", []) or []:
        arguments = call.get("function", {}).get("arguments", "{}")
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
            result.append(arguments["command"])
    return result


def explicit_submission(message):
    for command in _commands(message):
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            words = list(lexer)
        except ValueError:
            continue
        start = 0
        for index in range(len(words) + 1):
            if index == len(words) or words[index] in ("&&", ";", "||", "|"):
                if words[start:index][:2] == [
                    "echo",
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                ]:
                    return True
                if index < len(words) and words[index] in ("||", "|"):
                    break
                start = index + 1
    return False


def validate_action(action):
    if not isinstance(action, dict) or set(action) != FIELDS:
        raise ValueError("Completion observation contains missing/unreviewed fields")
    if action["tool"] is not None and (
        not isinstance(action["tool"], str)
        or not action["tool"]
        or len(action["tool"]) > 4096
    ):
        raise ValueError("Invalid observed tool category")
    if type(action["terminal"]) is not bool:
        raise ValueError("Completion terminal observation must be Boolean")
    if action["evidence"] not in (
        "explicit-submit-action",
        "current-tool-action",
        "assistant-only",
    ):
        raise ValueError("Unreviewed completion evidence")
    if action["terminal"] != (action["evidence"] == "explicit-submit-action"):
        raise ValueError("Terminal status requires explicit current-action evidence")
    digest = action["assistant_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((c not in "0123456789abcdef" for c in digest))
    ):
        raise ValueError("Missing completed-assistant provenance hash")
    return dict(action)


def recorded_action(message):
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Completion evidence must be the current assistant message")
    tool = tool_name(message)
    terminal = explicit_submission(message)
    return validate_action(
        dict(
            tool=None if tool == "no-action" else tool,
            terminal=terminal,
            evidence="explicit-submit-action"
            if terminal
            else "current-tool-action"
            if tool != "no-action"
            else "assistant-only",
            assistant_sha256=hashlib.sha256(
                json.dumps(message, sort_keys=True).encode()
            ).hexdigest(),
        )
    )


def prepare_manifest(groups, source, path):
    payload = json.loads(Path(source).read_bytes())
    if payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Invalid recorded completion source")
    observations = {}
    for experiment, programs in groups:
        for program in programs:
            if experiment == "vllm-warmup":
                continue
            actions = payload["actions"].get(program["id"])
            if not isinstance(actions, list) or len(actions) != len(program["turns"]):
                raise ValueError(
                    "Missing current-action evidence for Program " + program["id"]
                )
            for stage, action in enumerate(actions):
                key = replay_key(experiment, program["id"], stage)
                if key in observations:
                    raise ValueError("Duplicate completion-observation identity")
                observations[key] = validate_action(action)
    path = Path(path)
    path.write_text(json.dumps(dict(protocol=PROTOCOL, observations=observations)))
    path.chmod(384)
    return dict(
        protocol=PROTOCOL,
        observations=len(observations),
        explicit_submit_actions=sum((v["terminal"] for v in observations.values())),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        availability="after current output completion only",
        trajectory_end_inferred=False,
        future_gap_or_length=False,
    )


@lru_cache(maxsize=4)
def _observations(path):
    payload = json.loads(Path(path).read_bytes())
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Invalid completion observation protocol")
    return payload["observations"]


def observe_replay_completion(request):
    meta = getattr(request.sampling_params, "extra_args", None) or {}
    key, path = (meta.get("cont_replay_key"), os.environ.get(ENV))
    if key is None or not path:
        return None
    status = getattr(getattr(request, "status", None), "name", None)
    if (
        not request.is_finished()
        or not request.output_token_ids
        or status not in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
    ):
        raise ValueError("Cannot observe an unfinished/unsuccessful current action")
    action = _observations(path).get(key)
    if action is None:
        if meta.get("cont_experiment") == "vllm-warmup":
            return None
        raise ValueError(
            "Missing completed-action observation for finished replay call"
        )
    action = validate_action(action)
    previous = getattr(request, "_continuum_completed_action", None)
    if previous is not None:
        if previous != action:
            raise ValueError("Conflicting completion observations for one request")
        return None
    request._continuum_completed_action = action
    return action


def completion_metadata(request):
    params = getattr(request, "sampling_params", None)
    meta = getattr(params, "extra_args", None) or {}
    action = getattr(request, "_continuum_completed_action", None)
    if action is None:
        return meta
    if not request.is_finished():
        raise ValueError("Completed action cannot be visible on an active request")
    return dict(
        meta, cont_tool=action["tool"] or "unknown", cont_final=action["terminal"]
    )


def validate_events(events, expected):
    arrivals, seen, observed_requests, logical_finished, native_admitted = (
        {},
        set(),
        set(),
        set(),
        set(),
    )
    terminal_programs = set()
    for e in events:
        kind, rid = (e["event"], e.get("request"))
        if kind == "arrival":
            meta = e.get("meta", {})
            key = meta.get("cont_replay_key")
            if key in expected:
                from kairos.replay.boundary import validate_arrived_metadata

                validate_arrived_metadata(meta)
                if (e.get("experiment"), e.get("program")) in terminal_programs:
                    raise ValueError(
                        "Program resumed after explicit terminal observation"
                    )
                arrivals[rid] = (key, e["time"])
        elif kind == "admit" and rid in arrivals:
            native_admitted.add(rid)
        elif kind == "completion_observed":
            if rid not in arrivals:
                raise ValueError("Completion observation has no causal arrival")
            key, arrived = arrivals[rid]
            action = {k: e.get(k) for k in FIELDS}
            if (
                key in seen
                or action != expected[key]
                or e["time"] < arrived
                or (rid not in native_admitted)
                or (not e.get("arrived_metadata_unchanged"))
                or e.get("future_information")
                or (e.get("after_output_tokens", 0) < 1)
            ):
                raise ValueError("Invalid/repeated/out-of-order completion observation")
            seen.add(key)
            observed_requests.add(rid)
            if action["terminal"]:
                terminal_programs.add((e.get("experiment"), e.get("program")))
        elif (
            kind == "pin"
            and (e.get("experiment"), e.get("program")) in terminal_programs
        ):
            raise ValueError("Explicitly terminated Program created a new optional pin")
        elif kind == "holding_logical_finish" and rid in arrivals:
            if rid not in observed_requests:
                raise ValueError(
                    "Logical cache lifecycle ran before completion observation"
                )
            logical_finished.add(rid)
    if seen != set(expected) or len(arrivals) != len(expected):
        raise ValueError("Missing completed-action observations")
    return dict(
        valid=True,
        observations=len(seen),
        logical_completions=len(logical_finished),
        current_action_only=True,
        arrived_metadata_unchanged=True,
        trajectory_end_inferred=False,
    )
