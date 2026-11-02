"""Unified ready queue and continuity positions from a single observation snapshot."""

import time
from kairos.prediction.prior import mean
from kairos.scheduling.priority import (
    PACKAGE as CONTINUITY,
    idle_index,
    order,
    training_quantum,
)

IMPLEMENTATION = "kairos-ready-economic-unified-ready-J-v1"


class UnifiedSnapshot:
    def _snapshot(self, queued):
        started = time.perf_counter()
        if not hasattr(self, "_compiled_stats"):
            from collections import OrderedDict

            self._compiled_calls, self._compiled_service = ({}, OrderedDict())
            self._compiled_progress, self._compiled_means = ({}, {})
            self._compiled_stats = dict(
                call_hits=0,
                call_misses=0,
                service_hits=0,
                service_misses=0,
                index_hits=0,
                index_misses=0,
                mean_hits=0,
                mean_misses=0,
            )
        self._collect()
        ready = {self.program(r): r for r in queued}
        if len(ready) != len(queued):
            raise ValueError("Concurrent requests share a serial Program identity")
        running = {self.program(r): r for r in self.running if not r.is_finished()}
        if set(ready) & set(running):
            raise ValueError("A serial Program cannot be both ready and running")
        rows, forecasts, calls = ({}, {}, {})
        for p in sorted(
            set(ready) | set(running) | set(self.pins), key=self.order.__getitem__
        ):
            request = running.get(p) or ready.get(p)
            if request is None:
                forecast = self._idle_posterior(p)
                current = standard = 0.0
                kind = "idle"
            else:
                forecast = self._posterior(p, request)
                call = self._current_call(request)
                calls[p], forecasts[p] = (call, forecast)
                standard = self._service(
                    request.num_prompt_tokens, call.total_tokens, forecast["keep"]
                )
                current, kind = (standard, "ready")
                if p in running:
                    kind = "running"
                    keep = max(
                        forecast["keep"],
                        min(request.num_prompt_tokens - 1, request.num_computed_tokens),
                    )
                    current = self._service(
                        request.num_prompt_tokens, max(1.0, call.remaining_tokens), keep
                    )
            post = forecast["post"]
            remaining = self._residual_mean(p, post)
            rows[p] = dict(
                request=forecast["request"],
                state=kind,
                stage=forecast.get("stage", len(self.online_outputs[p])),
                current_service_s=current,
                standard_service_s=standard,
                residual_mean_s=remaining,
                total_service_s=current + remaining,
                source=forecast["source"],
                atoms_s=post["atoms"],
                mass=post["mass"],
            )
        if getattr(self, "_continuity_quote_scope", None) is not None:
            raise RuntimeError("Nested unified snapshot")
        scope = dict(running=running)
        self._continuity_quote_scope = scope
        try:
            quotes = (
                self._quotes(ready, set(ready), forecasts, calls)
                if ready or running
                else {}
            )
        finally:
            self._continuity_quote_scope = None
        cold = {}
        anchors = {p: r["total_service_s"] for p, r in rows.items()}
        scores = {p: q["score"] for p, q in quotes.items()}
        self._compiled_snapshot_s = time.perf_counter() - started
        proof = {}
        for p in sorted(set(rows) - set(quotes), key=self.order.__getitem__):
            row = rows[p]
            if row["state"] == "running":
                q = scope["quotes"][p]
                proof[p] = dict(kind="running-current-call", score=q["score"], quote=q)
            elif row["state"] == "idle":
                key = (len(self.online_outputs[p]), self.online_inputs[p][-1])
                cache = self.__dict__.setdefault("_continuity_quantum_cache", {})
                if key not in cache:
                    cache[key] = training_quantum(
                        self.service_prior.rows, *key, self._service, self.max_model_len
                    )
                training = cache[key]
                indexes = self.__dict__.setdefault("_continuity_index_cache", {})
                saved = indexes.get(p)
                if (
                    saved is not None
                    and saved[0] is row["atoms_s"]
                    and (saved[1] is row["mass"])
                    and (saved[2] == training["quantum_s"])
                ):
                    index = saved[3]
                else:
                    index = idle_index(
                        row["atoms_s"], row["mass"], training["quantum_s"]
                    )
                    indexes[p] = (
                        row["atoms_s"],
                        row["mass"],
                        training["quantum_s"],
                        index,
                    )
                proof[p] = dict(kind="idle-suffix-probe", **index, training=training)
            else:
                raise ValueError("Unclassified nonready state")
        result = order(
            anchors, scores, cold, self.order, {p: q["score"] for p, q in proof.items()}
        )
        self._continuity_j_proof = dict(implementation=CONTINUITY, held=proof)
        return (ready, rows, quotes, cold, result)

    def _residual_mean(self, p, post):
        saved = self._compiled_means.get(p)
        if saved is not None and saved[0] is post:
            self._compiled_stats["mean_hits"] += 1
            return saved[1]
        value = mean(post)
        self._compiled_means[p] = (post, value)
        self._compiled_stats["mean_misses"] += 1
        return value
