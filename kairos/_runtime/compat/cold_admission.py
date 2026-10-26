"""Kairos cold admission implementation."""
import hashlib
import json
import math
from pathlib import Path

from kairos.features import FEATURE_NAMES, features
from kairos.causal import causal_features


def feature_names():
    values, _ = causal_features([dict(role='user', content='')], [1], [], [], False)
    return ['prompt/' + name for name in FEATURE_NAMES] + ['progress/' + name for name in values]


def feature_source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=('kairos/_runtime/compat/cold_admission.py','kairos/features.py',
           'kairos/causal.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def first_features(text, messages, prompt_tokens):
    if not isinstance(text, str) or type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError('Invalid arrived first prompt')
    if (not isinstance(messages, list) or not messages or
            any(not isinstance(m, dict) or m.get('role') not in ('system', 'developer', 'user')
                for m in messages) or not any(m.get('role') == 'user' for m in messages)):
        raise ValueError('Cold features require a first pre-call message boundary')
    values, _ = causal_features(messages, [prompt_tokens], [], [], False)
    return features(text, [prompt_tokens], []) + list(values.values())


def reorder_fresh(order, candidates, scores):
    """Replace only fresh candidate slots; preserve every other relative order.

    Missing predictions make the complete decision fall back. Partially
    mixing unrelated ordinal and service scores would be meaningless.
    """
    fresh = [p for p in order if p in candidates]
    if not fresh or any(p not in scores or not math.isfinite(scores[p]) for p in fresh):
        return list(order)
    position = {p: i for i, p in enumerate(order)}
    ranked = iter(sorted(fresh, key=lambda p: (scores[p], position[p])))
    return [next(ranked) if p in candidates else p for p in order]


class ColdAdmissionRanker:
    """Validated, portable linear model; score is explicitly ordinal only."""
    def __init__(self, path, evaluation_ids=()):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.model = model = json.loads(raw)
        if (model.get('schema') != 'cold-admission-ordinal-v1' or
                model.get('feature_names') != feature_names() or
                model.get('feature_source_sha256') != feature_source_hashes() or
                model.get('input_contract') != 'first-pre-call-causal-only' or
                model.get('output_contract') != 'ordinal-only-lower-first' or
                model.get('current_service_is_feature') is not False):
            raise ValueError('Cold admission artifact contract mismatch')
        self.training_ids = set(model['training_ids'])
        if (len(self.training_ids) != len(model['training_ids']) or
                self.training_ids & set(evaluation_ids)):
            raise ValueError('Cold admission training/evaluation overlap')
        self.mean = model['mean']; self.scale = model['scale']; self.theta = model['theta']
        if any(len(v) != len(feature_names()) for v in (self.mean, self.scale, self.theta)):
            raise ValueError('Cold admission parameter dimensions')
        if (any(not math.isfinite(x) for v in (self.mean, self.scale, self.theta) for x in v) or
                any(x <= 0 for x in self.scale)):
            raise ValueError('Invalid cold admission parameters')

    def score(self, values):
        if len(values) != len(self.theta) or any(not math.isfinite(x) for x in values):
            raise ValueError('Invalid cold admission features')
        result = math.fsum((x - mean) / scale * weight
                          for x, mean, scale, weight in zip(values, self.mean, self.scale, self.theta))
        if not math.isfinite(result):
            raise ValueError('Nonfinite cold admission score')
        return result
