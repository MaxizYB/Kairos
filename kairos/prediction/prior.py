"""prediction.prior."""

import math
from kairos._runtime.compat.online_path import split_prior
from kairos._runtime.compat.owner_order import owner_order


def continuity_order(incumbent, scores, cold, backed):
    backed = set(backed)
    if not backed <= set(scores) or backed & set(cold):
        raise ValueError("A backed continuation must be a mature ready call")
    first = sorted(backed, key=lambda p: scores[p])
    rest = owner_order(
        [p for p in incumbent if p not in backed],
        {p: q for p, q in scores.items() if p not in backed},
        cold,
    )
    return first + rest


def distribution(values, source):
    values = sorted((float(x) for x in values))
    if not values or any((not math.isfinite(x) or x < 0 for x in values)):
        raise ValueError("Invalid post-current service distribution")
    return dict(
        atoms=values,
        mass=[1.0 / len(values)] * len(values),
        pred=math.fsum(values) / len(values),
        source=source,
    )


def mean(post):
    atoms, mass = (post["atoms"], post["mass"])
    if (
        not atoms
        or len(atoms) != len(mass)
        or any((not math.isfinite(x) or x < 0 for x in atoms + mass))
        or (not math.isclose(math.fsum(mass), 1.0, abs_tol=1e-08))
    ):
        raise ValueError("Invalid normalized service distribution")
    return math.fsum((a * w for a, w in zip(atoms, mass)))


class ServicePrior:
    def __init__(self, prior, service, max_len=16384):
        self.rows = tuple((tuple((row for row in path)) for path in split_prior(prior)))
        self.service, self.max_len = (service, max_len)
        self.cache = {}
        self.global_post = None

    def predict(self, inputs, outputs, current_output):
        if (
            not inputs
            or len(outputs) != len(inputs) - 1
            or type(current_output) is not int
            or (current_output < 1)
        ):
            raise ValueError("Fallback needs the causal arrived-call landmark")
        stage = len(outputs)
        key = (stage, int(inputs[-1]))
        if key not in self.cache:
            values = [
                self._suffix(path, stage, int(inputs[-1]))
                for path in self.rows
                if len(path) > stage
            ]
            post = (
                distribution(values, "stage-conditioned-training-prior")
                if values
                else self._global()
            )
            self.cache[key] = dict(post, stage=stage, matching_paths=len(values))
        return dict(self.cache[key])

    def _suffix(self, path, stage, current_prompt):
        shift = current_prompt - int(path[stage]["prompt_tokens"])
        previous = current_prompt
        total = 0.0
        for row in path[stage + 1 :]:
            prompt = max(1, min(self.max_len - 1, int(row["prompt_tokens"]) + shift))
            output = min(max(1, int(row["output_tokens"])), self.max_len - prompt)
            keep = min(prompt - 1, previous)
            total += self.service(prompt, output, keep)
            previous = prompt
        return total

    def _global(self):
        if self.global_post is None:
            self.global_post = distribution(
                [
                    self._suffix(
                        path,
                        k,
                        max(1, min(self.max_len - 1, int(path[k]["prompt_tokens"]))),
                    )
                    for path in self.rows
                    for k in range(len(path))
                ],
                "global-training-stage-prior",
            )
        return self.global_post


class ToolMarginal:
    protection_support = True
    neighbors = 0
    cache_hits = cache_misses = 0

    def __init__(self, prior):
        self.tool_samples = {}
        for path in split_prior(prior):
            for row in path:
                if not row.get("done") and row.get("tool_s") is not None:
                    self.tool_samples.setdefault(row.get("tool", ""), []).append(
                        float(row["tool_s"])
                    )
        self.global_tool_samples = tuple(
            (x for v in self.tool_samples.values() for x in v)
        )

    def idle_bound(self, tool):
        return max(
            0.001,
            max(self.tool_samples.get(tool) or self.global_tool_samples, default=1.0),
        )

    def idle_remaining(self, tool, age):
        values = self.tool_samples.get(tool) or self.global_tool_samples
        residual = [v - age for v in values if v > age]
        return (
            max(0.001, math.fsum(residual) / len(residual))
            if residual
            else self.idle_bound(tool)
        )

    def forecast(self, *args, **kwargs):
        raise AssertionError("Kairos must not construct a representative path")

    scenarios = forecast
