"""Update predicted remaining output at empirical survival boundaries."""

from kairos.scheduling.contracts import CurrentCall

IMPLEMENTATION = "kairos-continuity-running-survival-events-v1"


class OutputSurvival:
    def _current_call(self, request):
        stats = self._survival_state()
        if not any((r is request and (not r.is_finished()) for r in self.running)):
            stats["ready_passthrough"] += 1
            return super()._current_call(request)
        p = self.program(request)
        observed = getattr(self, "forecast_output_cache", {}).get(p, {})
        generated = len(request.output_token_ids)
        cap = min(
            int(request.max_tokens), self.max_model_len - request.num_prompt_tokens
        )
        saved = self._running_event_calls.get(p)
        if (
            saved is not None
            and observed.get("request") == request.request_id
            and (saved[0] == (request.request_id, cap, request.num_prompt_tokens))
            and (saved[1] is observed.get("post"))
            and (saved[2] is observed.get("marginal"))
            and (saved[3] <= generated < saved[4])
        ):
            call = saved[5]
            stats["interval_hits"] += 1
            return CurrentCall(
                call.total_tokens,
                call.total_tokens - generated,
                call.cap_tokens,
                call.exact,
                call.source,
                call.atoms,
                call.mass,
            )
        call = super()._current_call(request)
        observed = getattr(self, "forecast_output_cache", {}).get(p, {})
        stats["rebuilds"] += 1
        if (
            call.source.endswith("/residual-scale")
            or call.exact
            or (not 0 <= generated < cap)
        ):
            self._running_event_calls.pop(p, None)
            if call.source.endswith("/residual-scale"):
                stats["residual_fallbacks"] += 1
        else:
            self._running_event_calls[p] = (
                (request.request_id, cap, request.num_prompt_tokens),
                observed.get("post"),
                observed.get("marginal"),
                generated,
                min(call.atoms),
                call,
            )
        return call

    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            self.__dict__.pop("_running_event_stats", None)
            self._survival_state()
        return result

    def _survival_state(self):
        if not hasattr(self, "_running_event_stats"):
            self._running_event_calls = {}
            self._running_event_stats = dict(
                interval_hits=0, rebuilds=0, residual_fallbacks=0, ready_passthrough=0
            )
        return self._running_event_stats
