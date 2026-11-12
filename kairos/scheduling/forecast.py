"""scheduling.forecast."""

import os
from pathlib import Path
import time
from kairos.prediction.current import (
    FULL_CONTRACT,
    OutputForest,
    condition_output,
    visible_messages,
)
from kairos.prediction.metadata import FEATURE_CONTRACT
from kairos.replay.boundary import validate_arrived_metadata
from kairos.scheduling.contracts import capacity_pages
from kairos.scheduling.capacity import admission
from kairos.scheduling.program import ProgramControl

VERSION = "kairos"
METHOD = "continuity-eager-a~provenanceprior~onlinefitresidual~finishlogical~" + VERSION


class ForecastControl(ProgramControl):
    VERSION = VERSION
    METHOD = METHOD
    track_frontier_on_allocation = False

    def _prepare_resources(self):
        resources = super()._prepare_resources()
        self._prepare_output_forests(resources[1].tokenizer)
        return resources

    def _prepare_output_forests(self, tokenizer=None):
        if hasattr(self, "forecast_output_forest"):
            return
        models = Path(__file__).resolve().parents[2] / "assets"
        self.forecast_output_forest = OutputForest(
            os.environ.get(
                "CONTINUUM_CURRENT_OUTPUT_FOREST", str(models / "current_output.json")
            ),
            tokenizer,
        )
        self.forecast_output_fallback = OutputForest(
            os.environ.get(
                "CONTINUUM_CURRENT_OUTPUT_FALLBACK",
                str(models / "current_output_metadata.json"),
            )
        )
        if self.forecast_output_fallback.feature_contract != FEATURE_CONTRACT:
            raise ValueError(
                "Missing-message current forest must use metadata-only features"
            )

    def _predict_current_output(self, request):
        self._prepare_output_forests(getattr(self.forest, "tokenizer", None))
        p = self.program(request)
        messages = visible_messages(self.metadata(request).get("cont_visible_messages"))
        forest = self.forecast_output_forest
        if forest.feature_contract == FULL_CONTRACT and (
            messages is None or forest.tokenizer is None
        ):
            forest = self.forecast_output_fallback
        if p in forest.training_ids:
            raise ValueError(
                "Current-output forest training/evaluation identity overlap"
            )
        started = time.perf_counter()
        post = forest.predict_observed(
            prompt_ids=request.prompt_token_ids,
            messages=messages,
            prompt_lengths=self.online_inputs[p],
            outputs=self.online_outputs[p],
            services=getattr(self, "forecast_observed_service", {}).get(p, 0.0),
        )
        if not hasattr(self, "forecast_output_cache"):
            self.forecast_output_cache = {}
        self.forecast_output_cache[p] = dict(
            request=request.request_id, post=post, marginal=forest.marginal
        )
        self._emit(
            "forecast_current_prediction",
            program=p,
            request=request.request_id,
            stage=len(self.online_outputs[p]),
            atoms_tokens=post["atoms"],
            mass=post["mass"],
            output_mean=post["pred"],
            forest_sha256=forest.sha256,
            feature_contract=forest.feature_contract,
            observed_service_s=getattr(self, "forecast_observed_service", {}).get(
                p, 0.0
            ),
            completed_outputs=list(self.online_outputs[p]),
            current_output_known=False,
            future_information=False,
            seconds=time.perf_counter() - started,
        )

    def add_request(self, request):
        incoming = self.metadata(request).get("cont_experiment")
        enforce = (
            self.VERSION in incoming.split("~")
            if incoming is not None
            else self.program_enabled
        )
        if enforce:
            validate_arrived_metadata(self.metadata(request))
            if getattr(
                request.sampling_params, "ignore_eos", False
            ) and "cont_replay_key" not in self.metadata(request):
                raise ValueError(
                    "Fixed Replay requires an opaque private execution key"
                )
        result = super().add_request(request)
        if self.program_enabled:
            cached = getattr(self, "forecast_output_cache", {}).get(
                self.program(request)
            )
            if cached is None or cached["request"] != request.request_id:
                self._predict_current_output(request)
        return result

    def _current_call(self, request):
        if not self.program_enabled:
            return super()._current_call(request)
        p = self.program(request)
        cached = getattr(self, "forecast_output_cache", {}).get(p)
        if cached is None or cached["request"] != request.request_id:
            self._predict_current_output(request)
            cached = self.forecast_output_cache[p]
        return condition_output(
            cached["post"],
            cap=min(
                int(request.max_tokens), self.max_model_len - request.num_prompt_tokens
            ),
            generated=len(request.output_token_ids),
            marginal=cached["marginal"],
        )

    def _emit(self, event, **fields):
        if event.startswith("program_"):
            event = "forecast_" + event[len("program_") :]
        if event == "forecast_registry":
            fields.update(
                capacity="observed native frontier",
                suffix_predictor="Progress-CDF R+ in standard service units",
                backend="joint coverage; mapped boundary / batched native suffix",
                full_output_reservation=False,
                current_output="causal current-output forest",
                capacity_refresh="admission and reference lifecycle boundaries",
                decode_growth_triggers_control=False,
            )
            if getattr(self, "uses_joint_forecasts", False):
                fields.update(
                    score="exp(E[log(t(O) / g(O,R)) | arrived history, O>n])",
                    primary_predictor="continuous joint p(O|H) p(R|O,H)",
                    suffix_predictor="auxiliary forest for idle continuity and TTL",
                    current_output="joint survival posterior for J; forest for auxiliary bookkeeping",
                )
        return super()._emit(event, **fields)

    def _switch_experiment(self, name):
        if name == getattr(self, "experiment", None):
            return
        super()._switch_experiment(name)
        if self.program_enabled:
            self.online_grant_allocation = False
            self.forecast_observed_service = {}
            self.forecast_output_cache = {}
            self.forecast_output_completed = set()

    def _admission(self, *args, **kwargs):
        return admission(*args, **kwargs)

    def _free_request_blocks(self, request):
        p = self.program(request)
        cleaning = bool(getattr(self, "online_cleaning", {}).get(p))
        before = self.online_reserved.get(p)
        result = super()._free_request_blocks(request)
        if (
            self.program_enabled
            and self.online_cleaning.get(p)
            and (before is not None)
        ):
            self.online_reserved[p] = max(before, self.online_reserved.get(p, 0))
        if self.program_enabled and cleaning and (not self.online_cleaning.get(p)):
            self._sync_idle_frontier(p)
            self.online_dirty = True
        return result

    def _continuum_complete_policy(self, request):
        if self.program_enabled:
            p = self.program(request)
            if not hasattr(self, "forecast_output_completed"):
                self.forecast_output_completed, self.forecast_observed_service = (
                    set(),
                    {},
                )
            status = getattr(getattr(request, "status", None), "name", None)
            if (
                self.online_requests.get(p) is request
                and request.request_id not in self.forecast_output_completed
                and request.output_token_ids
                and (status in (None, "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"))
            ):
                rid, keep = self.program_keeps[p]
                if rid != request.request_id:
                    raise ValueError("Stale observed current-output completion")
                increment = self._service(
                    request.num_prompt_tokens, len(request.output_token_ids), keep
                )
                self.forecast_observed_service[p] = (
                    self.forecast_observed_service.get(p, 0.0) + increment
                )
                self.forecast_output_completed.add(request.request_id)
                self._emit(
                    "forecast_current_completed",
                    program=p,
                    request=request.request_id,
                    prompt_tokens=request.num_prompt_tokens,
                    output_tokens=len(request.output_token_ids),
                    standard_keep=keep,
                    service_increment_s=increment,
                    observed_service_s=self.forecast_observed_service[p],
                    future_information=False,
                )
            if self.online_requests.get(p) is request:
                pin = self.pins.get(p)
                pages = max(self._native_pages(request), len(pin.blocks) if pin else 0)
                if self.online_cleaning.get(p):
                    pages = max(pages, self.online_reserved.get(p, 0))
                self.online_reserved[p] = pages
        return super()._continuum_complete_policy(request)

    def _capacity_evidence(self, request):
        return dict(
            prompt=request.num_prompt_tokens,
            cap=min(
                int(request.max_tokens), self.max_model_len - request.num_prompt_tokens
            ),
            generated=len(request.output_token_ids),
            computed=request.num_computed_tokens,
            pending=max(0, getattr(request, "num_output_placeholders", 0)),
            page=self.block_size,
            max_len=self.max_model_len,
        )

    def _online_grant(self, request, pages, reason):
        if reason == "residency-current-call":
            reason = "kairos-current-call"
        return super()._online_grant(request, pages, reason)

    def _release(self, program, *args, **kwargs):
        if args and args[0] == "program_optional_replacement":
            args = ("forecast_optional_replacement",) + args[1:]
        before = self.online_reserved.get(program) if self.program_enabled else None
        result = super()._release(program, *args, **kwargs)
        if self.program_enabled:
            self._sync_idle_frontier(program)
            if self.online_reserved.get(program) != before:
                self.online_dirty = True
        return result

    def _capacity_kind(self, request):
        return "native-frontier"

    def _online_target(self, request):
        if not self.program_enabled:
            return super()._online_target(request)
        return capacity_pages(
            prompt=request.num_prompt_tokens,
            cap=min(
                int(request.max_tokens), self.max_model_len - request.num_prompt_tokens
            ),
            generated=len(request.output_token_ids),
            computed=request.num_computed_tokens,
            pending=max(0, getattr(request, "num_output_placeholders", 0)),
            page=self.block_size,
            exact=False,
            max_len=self.max_model_len,
        )

    def _sync_frontiers(self):
        for p, request in self.online_requests.items():
            if request.request_id in self.online_granted:
                self.online_reserved[p] = max(
                    self._online_target(request), self._native_pages(request)
                )
        for p in self.pins:
            self._sync_idle_frontier(p)

    def _sync_idle_frontier(self, program):
        if program in self.online_cleaning or program in self._occupied():
            return
        pin = self.pins.get(program)
        if pin is not None:
            self.online_reserved[program] = len(pin.blocks)
