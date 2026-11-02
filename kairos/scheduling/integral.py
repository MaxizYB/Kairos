"""Exact nonlinear progress integral with cached probability weights."""

import math
from kairos.scheduling.cost import posterior_mean


class ResidualLaw:
    def __init__(self, post):
        self.post = post
        self.atoms = tuple((float(x) for x in post["atoms"]))
        self.mass = tuple((float(x) for x in post["mass"]))
        self.mean = posterior_mean(self.atoms, self.mass)
        total = math.fsum(self.mass)
        self.weights = tuple((w / total for w in self.mass))

    def quote(self, standard):
        if not math.isfinite(standard) or standard <= 0:
            raise ValueError("Positive finite standard service required")
        progress = math.fsum(
            (w * standard / (standard + r) for r, w in zip(self.atoms, self.weights))
        )
        return dict(
            score=1.0 / progress,
            progress=progress,
            residual_mean_s=self.mean,
            standard_s=standard,
            actual_s=1.0,
            mode="distribution",
        )


IMPLEMENTATION = "kairos-continuity-running-reduced-exact-v2"


class RunningIntegral:
    def _quotes(self, ready, scored, forecasts, calls):
        stats = self._integral_state()
        scope = getattr(self, "_continuity_quote_scope", None)
        if scope is not None:
            for p, request in scope["running"].items():
                forecast, call = (forecasts[p], calls[p])
                post = forecast["post"]
                standard = self._service(
                    request.num_prompt_tokens, call.total_tokens, forecast["keep"]
                )
                saved = self._compiled_progress.get(p)
                if (
                    saved is not None
                    and saved[0] == request.request_id
                    and (saved[1] is post)
                    and (saved[2] == standard)
                ):
                    stats["integral_hits"] += 1
                    continue
                law = self._running_reduced_laws.get(p)
                if law is None or law.post is not post:
                    law = ResidualLaw(post)
                    self._running_reduced_laws[p] = law
                    stats["law_compilations"] += 1
                quote = law.quote(standard)
                self._compiled_progress[p] = (request.request_id, post, standard, quote)
                stats["integral_evaluations"] += 1
        return super()._quotes(ready, scored, forecasts, calls)

    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            self.__dict__.pop("_running_reduced_stats", None)
            self._integral_state()
        return result

    def _integral_state(self):
        if not hasattr(self, "_running_reduced_stats"):
            self._running_reduced_laws = {}
            self._running_reduced_stats = dict(
                law_compilations=0, integral_evaluations=0, integral_hits=0
            )
        return self._running_reduced_stats
