"""Shared text projection: usable by preparation and arrived-only serving."""

import math
import re

from kairos.causal import causal_features
from kairos.prediction.features import features

SCHEMA = "kairos-canonical-visible-v1"


def issue_text(text):
    match = re.search(r"<pr_description>(.*?)</pr_description>", text, re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bISSUE:\s*\n(.*?)\n\s*INSTRUCTIONS:", text, re.S)
    return match.group(1).strip() if match else text.strip()


def message_text(message):
    content = message.get("content")
    body = content if isinstance(content, str) else ""
    if message.get("role") != "assistant":
        return body
    for call in message.get("tool_calls") or []:
        arguments = call.get("function", {}).get("arguments", "")
        if isinstance(arguments, str) and arguments and arguments not in body:
            body += "\n" + arguments
    for action in message.get("extra", {}).get("actions", []) or []:
        command = action.get("command") if isinstance(action, dict) else None
        if isinstance(command, str) and command and command not in body:
            body += "\n" + command
    return body


def visible(messages, numeric):
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        raise ValueError("Missing arrived task context")
    previous = [m for m in messages if m.get("role") == "assistant"]
    feedback = [
        m
        for m in messages
        if m.get("role") == "tool" or (m.get("role") == "user" and m is not users[0])
    ]
    return dict(
        numeric=list(numeric),
        issue=issue_text(message_text(users[0]))[:20000],
        previous=message_text(previous[-1])[-16000:] if previous else "",
        feedback=message_text(feedback[-1])[-20000:] if feedback else "",
    )


def observed_input(*, tokenizer, prompt_ids, messages, prompts, outputs, service):
    if (
        not isinstance(messages, list)
        or not messages
        or len(prompts) != len(outputs) + 1
        or prompts[-1] != len(prompt_ids)
        or any(type(x) is not int or x < 1 for x in outputs)
        or not math.isfinite(service)
        or service < 0
    ):
        raise ValueError("Invalid completed-history boundary")
    old = features(
        tokenizer.decode(prompt_ids, skip_special_tokens=False),
        list(prompts),
        list(outputs),
    )
    extra, _ = causal_features(
        messages, list(prompts), list(outputs), (service,), False
    )
    result = visible(messages, old + list(extra.values()))
    if len(result["numeric"]) != 127:
        raise ValueError("Causal feature schema mismatch")
    return result
