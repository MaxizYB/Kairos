"""replay.boundary."""

FORBIDDEN_METADATA = frozenset(
    {
        "cont_final",
        "cont_oracle_gap",
        "cont_known_output",
        "cont_output_tokens",
        "cont_planned_output_tokens",
        "cont_remaining_turns",
        "cont_total_turns",
        "cont_cohort",
        "cont_future_messages",
        "cont_prediction_oracle",
    }
)
ALLOWED_METADATA = frozenset(
    {
        "cont_program_id",
        "cont_tool",
        "cont_experiment",
        "cont_visible_messages",
        "cont_replay_key",
        "cont_bfcl_tool_marker",
        "cont_bfcl_think_end_token",
        "cont_bfcl_reset_history",
    }
)


def validate_arrived_metadata(metadata):
    bad = set(metadata) & FORBIDDEN_METADATA
    if bad:
        raise ValueError("Future trace metadata is forbidden: " + ",".join(sorted(bad)))
    extra = set(metadata) - ALLOWED_METADATA
    if extra:
        raise ValueError(
            "Unreviewed request metadata is forbidden: " + ",".join(sorted(extra))
        )
    key = metadata.get("cont_replay_key")
    if key is not None and (
        not isinstance(key, str)
        or len(key) != 64
        or any((c not in "0123456789abcdef" for c in key))
    ):
        raise ValueError("Invalid opaque replay execution key")
    tool = metadata.get("cont_tool")
    if tool not in (None, "", "unknown", "bfcl"):
        raise ValueError("Future tool category cannot be supplied at request arrival")


def generation_parameters(prompt_tokens, max_model_len, output_limit=16384):
    if (
        type(prompt_tokens) is not int
        or type(max_model_len) is not int
        or type(output_limit) is not int
        or (prompt_tokens < 1)
        or (output_limit < 1)
        or (prompt_tokens >= max_model_len)
    ):
        raise ValueError("Invalid public unknown-output generation boundary")
    return dict(
        max_tokens=min(output_limit, max_model_len - prompt_tokens), ignore_eos=True
    )


def arrived_metadata(program):
    return dict(cont_program_id=program, cont_tool="unknown")


def validate_online_method(method):
    tags = method.split("~")
    if any(("oracle" in tag.lower() or tag == "knownoutput" for tag in tags)):
        raise ValueError(
            "Future-truth and known-output methods are forbidden in new tests"
        )
