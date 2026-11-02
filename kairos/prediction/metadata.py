"""prediction.metadata."""

import math
from kairos.prediction.suffix import SuffixForest

FEATURE_CONTRACT = "common-prefix-metadata-127-v1"
VISIBLE_INDICES = tuple(range(9)) + tuple(range(81, 93))


def mask_features(row):
    values = list(row["old_features"]) + list(row["features"])
    if len(values) != 127:
        raise ValueError("Wrong shared feature schema")
    keep = set(VISIBLE_INDICES)
    masked = [float(x) if j in keep else 0.0 for j, x in enumerate(values)]
    return dict(row, old_features=masked[:81], features=masked[81:])


def prefix_features(prompt_lengths, completed_outputs, completed_service_s):
    if (
        len(prompt_lengths) != len(completed_outputs) + 1
        or not prompt_lengths
        or any((type(x) is not int or x < 1 for x in prompt_lengths))
        or any((type(x) is not int or x < 1 for x in completed_outputs))
        or (not math.isfinite(completed_service_s))
        or (completed_service_s < 0)
    ):
        raise ValueError("Invalid causal prompt/completed-call prefix")
    from kairos.prediction.features import features
    from kairos.causal import causal_features

    old = features("", list(prompt_lengths), list(completed_outputs))
    new, _ = causal_features(
        [], list(prompt_lengths), list(completed_outputs), (completed_service_s,), False
    )
    return old + list(new.values())


class MetadataForest(SuffixForest):
    def __init__(self, artifact_path, evaluation_ids=()):
        super().__init__(artifact_path, None, evaluation_ids)
        if self.artifact.get("feature_contract") != FEATURE_CONTRACT:
            raise ValueError("Forest was not trained for metadata-only inference")

    def predict_metadata(
        self, *, prompt_lengths, completed_outputs, completed_service_s
    ):
        result = self.predict_features(
            prefix_features(prompt_lengths, completed_outputs, completed_service_s)
        )
        result["source"] = self.artifact["predictor_name"]
        return result
