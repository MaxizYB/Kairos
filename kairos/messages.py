"""Kairos messages implementation."""
from __future__ import annotations

import json
import math
import re


def model_message(message):
    fields = ("role", "content", "tool_calls", "tool_call_id", "name")
    return {key: message[key] for key in fields if key in message}


def tool_name(message):
    actions = message.get("extra", {}).get("actions", [])
    if actions:
        commands = [item["command"] for item in actions]
    else:
        commands = [json.loads(item["function"]["arguments"])["command"]
                    for item in message.get("tool_calls", [])]
    if not commands:
        commands = re.findall(r"```bash\s*\n(.*?)\n```", message.get("content") or "", re.S)
    # Commands are labels only. Never execute untrusted trajectory content.
    return "+".join(sorted({(command.strip().split() or ["unknown"])[0]
                            for command in commands})) or "no-action"


def rounds(document):
    messages = document["messages"]
    indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not indices:
        raise ValueError("No recorded model rounds")
    result = []
    for number, index in enumerate(indices):
        assistant = messages[index]
        end = indices[number + 1] if number + 1 < len(indices) else len(messages)
        feedback = [message for message in messages[index + 1:end] if message["role"] != "exit"]
        finish = assistant.get("extra", {}).get("timestamp")
        timestamps = [message.get("extra", {}).get("timestamp") for message in feedback]
        if number + 1 < len(indices):
            if finish is None or not timestamps or any(value is None for value in timestamps):
                raise ValueError("Missing actual tool completion timestamps")
            duration = max(timestamps) - finish
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid tool completion interval")
        else:
            duration = 0.0
        prompt = [model_message(message) for message in messages[:index] if message["role"] != "exit"]
        result.append(dict(messages=prompt, assistant=model_message(assistant),
                           tool=tool_name(assistant), gap_after_s=duration,
                           original_assistant_timestamp=finish, feedback_messages=len(feedback)))
    return result


def token_ids(value):
    return list(value["input_ids"] if hasattr(value, "keys") else value)


def encode_rounds(tokenizer, document):
    rows = rounds(document)
    previous = []
    turns = []
    for number, row in enumerate(rows):
        prompt = token_ids(tokenizer.apply_chat_template(row["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        completed = token_ids(tokenizer.apply_chat_template(row["messages"] + [row["assistant"]],
            tokenize=True, add_generation_prompt=False, enable_thinking=False))
        output_lcp = 0
        for a, b in zip(prompt, completed):
            if a != b:
                break
            output_lcp += 1
        output = max(1, len(completed) - output_lcp)
        keep = 0
        for a, b in zip(previous, prompt):
            if a != b:
                break
            keep += 1
        turns.append(dict(keep_prefix_tokens=keep, added_token_ids=prompt[keep:],
            output_tokens=output, gap_after_s=row["gap_after_s"], tool=row["tool"],
            original_round=number, prompt_tokens=len(prompt),
            original_assistant_timestamp=row["original_assistant_timestamp"],
            feedback_messages=row["feedback_messages"]))
        previous = prompt
    return turns
