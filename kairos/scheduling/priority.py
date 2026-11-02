"""scheduling.priority."""

from bisect import bisect_left
import math
from kairos.prediction.prior import mean
from kairos.scheduling.contracts import ServiceOrder

PACKAGE = "progress-continuity"


def idle_index(atoms, mass, quantum):
    mean(dict(atoms=list(atoms), mass=list(mass)))
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("Positive causal quantum required")
    survival = math.fsum((w for r, w in zip(atoms, mass) if r > 0))
    if not survival:
        return dict(
            score=0.0,
            progress=0.0,
            actual_s=0.0,
            quantum_s=quantum,
            continuation_mass=0.0,
        )
    actual = (
        math.fsum((min(quantum, r) * w for r, w in zip(atoms, mass) if r > 0))
        / survival
    )
    progress = (
        math.fsum((min(quantum / r, 1.0) * w for r, w in zip(atoms, mass) if r > 0))
        / survival
    )
    return dict(
        score=actual / progress,
        progress=progress,
        actual_s=actual,
        quantum_s=quantum,
        continuation_mass=survival,
    )


def training_quantum(paths, stage, previous_prompt, service, max_len):
    if stage < 1 or previous_prompt < 1:
        raise ValueError("Idle owner requires completed history")
    rows = [(path, stage) for path in paths if len(path) > stage]
    source = "stage-conditioned-training-next-service"
    if not rows:
        rows = [(path, k) for path in paths for k in range(1, len(path))]
        source = "pooled-training-next-service"
    if not rows:
        raise ValueError("No training continuation service evidence")
    values = []
    for path, k in rows:
        shift = previous_prompt - int(path[k - 1]["prompt_tokens"])
        prompt = max(1, min(max_len - 1, int(path[k]["prompt_tokens"]) + shift))
        output = min(max(1, int(path[k]["output_tokens"])), max_len - prompt)
        values.append(service(prompt, output, min(prompt - 1, previous_prompt)))
    q = math.fsum(values) / len(values)
    if not math.isfinite(q) or q <= 0:
        raise ValueError("Invalid training quantum")
    return dict(
        quantum_s=q,
        training_samples=len(values),
        source=source,
        next_stage=stage,
        previous_observed_prompt=previous_prompt,
    )


def order(anchors, ready_scores, cold_scores, arrival, held_scores):
    if cold_scores:
        raise ValueError("Every ready call must use the unified progress score")
    ready = tuple(sorted(ready_scores, key=lambda p: (ready_scores[p], arrival[p])))
    held = set(anchors) - set(ready_scores) - set(cold_scores)
    if held != set(held_scores):
        raise ValueError("Missing nonready J score")
    scores = dict(ready_scores)
    scores.update(held_scores)
    if any((not math.isfinite(v) or v < 0 for v in scores.values())):
        raise ValueError("Invalid continuity J")
    ranked = sorted(scores, key=lambda p: (scores[p], arrival[p]))
    priorities = {p: k for k, p in enumerate(ranked)}
    ready_keys = sorted(((ready_scores[p], arrival[p]) for p in ready_scores))
    boundaries = {
        p: bisect_left(ready_keys, (held_scores[p], arrival[p])) for p in held
    }
    return ServiceOrder(ready, priorities, boundaries)
