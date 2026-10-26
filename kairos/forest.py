"""Kairos forest implementation."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path


def _f32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _leaf(tree, x):
    node = 0
    while tree["feature"][node] >= 0:
        node = (tree["left"][node] if x[tree["feature"][node]] <=
                tree["threshold"][node] else tree["right"][node])
    return node


class SuffixForest:
    """Honest CDF forest for post-current standard service ``R+``."""

    schema = "r-only-distribution-v1"

    def __init__(self, artifact_path, tokenizer, evaluation_ids=()):
        path = Path(artifact_path)
        raw = path.read_bytes()
        artifact = json.loads(raw)
        if artifact.get("schema") != self.schema or artifact.get("kind") != "honest_cdf":
            raise ValueError("Kairos requires the honest_cdf distribution artifact")
        if int(artifact.get("features", 0)) != 127:
            raise ValueError("Kairos forest feature schema mismatch")
        training = set(artifact.get("training_ids", ()))
        overlap = training & set(evaluation_ids)
        if overlap:
            raise ValueError("Kairos forest training/evaluation identity overlap")
        self.artifact = artifact
        self.path = str(path)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.tokenizer = tokenizer
        self.training_ids = training
        self.calls = 0

    def predict(self, *, text, messages, prompt_lengths, outputs, services):
        """Return a finite distribution over post-current service seconds."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("missing causal visible messages")
        from kairos.features import features as old_features
        from kairos.causal import causal_features

        old = old_features(text, list(prompt_lengths), list(outputs))
        values, _ = causal_features(messages, list(prompt_lengths), list(outputs),
                                     (float(services),), False)
        # Two targets can share exactly the same causal feature extraction.
        # This is the last ARRIVED request only, never a replay/future label.
        self.last_features = tuple(old + list(values.values()))
        return self.predict_features(self.last_features)

    def predict_features(self, values):
        """Portable numerical export interface, also used for equivalence tests."""
        x = [_f32(v) for v in values]
        if len(x) != 127:
            raise ValueError("Kairos forest causal feature length mismatch")
        merged = {}
        trees = self.artifact["trees"]
        for tree in trees:
            values = tree["value"][str(_leaf(tree, x))]
            for atom, probability in values:
                atom = max(0.0, float(atom))
                merged[atom] = merged.get(atom, 0.0) + float(probability) / len(trees)
        atoms = sorted(merged)
        total = math.fsum(merged.values())
        if not atoms or not math.isfinite(total) or total <= 0:
            raise ValueError("empty Kairos forest posterior")
        mass = [max(0.0, merged[a] / total) for a in atoms]
        normal = math.fsum(mass)
        mass = [p / normal for p in mass]
        self.calls += 1
        return dict(pred=math.fsum(a * p for a, p in zip(atoms, mass)),
                    atoms=atoms, mass=mass, source="honest_cdf157")
