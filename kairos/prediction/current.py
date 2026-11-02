"""prediction.current."""

import json
import math
from kairos.prediction.metadata import FEATURE_CONTRACT, prefix_features
from kairos.prediction.output_tree import CurrentCallForest
from kairos.scheduling.contracts import CurrentCall

FULL_CONTRACT = "swe-text-and-prefix-127-v1"


def condition_output(post, *, cap, generated=0, marginal=None):
    if (
        type(cap) is not int
        or cap < 1
        or type(generated) is not int
        or (not 0 <= generated <= cap)
    ):
        raise ValueError("Invalid observed output progress/public ceiling")
    if generated == cap:
        return CurrentCall(
            float(cap), 0.0, cap, False, "public-limit-reached", (float(cap),), (1.0,)
        )

    def surviving(p):
        return [
            (float(min(n, cap)), float(w))
            for n, w in zip(p["atoms"], p["mass"])
            if min(n, cap) > generated and w > 0
        ]

    pairs = surviving(post)
    source = post.get("source", "current-output-forest")
    if not pairs:
        if marginal is None:
            raise ValueError("Current forest needs a training-only tail marginal")
        pairs = surviving(marginal)
        source += "/training-tail"
        if not pairs:
            pairs = [
                (float(generated + min(n, cap - generated)), float(w))
                for n, w in zip(marginal["atoms"], marginal["mass"])
                if w > 0
            ]
            source += "/residual-scale"
    merged = {}
    for n, w in pairs:
        if not math.isfinite(n) or not math.isfinite(w) or n <= generated or (w < 0):
            raise ValueError("Invalid current-output posterior")
        merged[n] = merged.get(n, 0.0) + w
    atoms = tuple(sorted(merged))
    total = math.fsum(merged.values())
    if not total > 0:
        raise ValueError("Empty current-output posterior")
    mass = tuple((merged[n] / total for n in atoms))
    estimate = math.fsum((n * w for n, w in zip(atoms, mass)))
    return CurrentCall(estimate, estimate - generated, cap, False, source, atoms, mass)


class OutputForest(CurrentCallForest):
    def __init__(self, path, tokenizer=None, evaluation_ids=()):
        super().__init__(path, tokenizer, evaluation_ids)
        self.feature_contract = self.artifact.get("feature_contract")
        if self.feature_contract not in (FEATURE_CONTRACT, FULL_CONTRACT):
            raise ValueError(
                "Current-output forest needs an explicit causal feature contract"
            )
        self.marginal = self.artifact["output_marginal"]

    def predict_observed(
        self, *, prompt_ids, messages, prompt_lengths, outputs, services
    ):
        if self.feature_contract == FEATURE_CONTRACT:
            values = prefix_features(
                list(prompt_lengths), list(outputs), float(services)
            )
            post = self.predict_features(values)
        else:
            if not isinstance(messages, list) or not messages or self.tokenizer is None:
                raise ValueError(
                    "Text current-output forest requires arrived input messages"
                )
            post = self.predict(
                text=self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
                messages=messages,
                prompt_lengths=list(prompt_lengths),
                outputs=list(outputs),
                services=float(services),
            )
        return dict(
            post,
            source="current-output-forest",
            forest_sha256=self.sha256,
            feature_contract=self.feature_contract,
        )


def visible_messages(raw):
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    return messages if isinstance(messages, list) and messages else None
