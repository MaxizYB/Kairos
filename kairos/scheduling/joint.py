"""One arrived posterior, observed survival updates, and live geometric quotes."""

import os
import time
import numpy as np

from kairos.prediction.current import visible_messages
from kairos.prediction.visible import observed_input


class JointForecasts:
    """Arrived-only joint inference and geometric ready/running quotations."""

    uses_joint_forecasts = True

    def _emit(self, event, **fields):
        if event == "forecast_current_completed":
            saved = getattr(self, "progress_cache", {}).get(fields["program"])
            if saved is None or saved["request"] != fields["request"]:
                raise ValueError("Completion does not own the arrived posterior")
            fields["posterior_reuse"] = dict(saved["posterior"].stats)
        return super()._emit(event, **fields)

    def _prepare_resources(self):
        result = super()._prepare_resources()
        if not hasattr(self, "progress_model"):
            from kairos.prediction.encoder import ContinuousRuntime

            self.progress_model = ContinuousRuntime(
                os.environ["KAIROS_PROGRESS_ENCODER"],
                os.environ["KAIROS_PROGRESS_CHECKPOINT"],
                os.environ["KAIROS_COUPLED_STUDY"],
                os.environ["KAIROS_DENSITY_MODEL"],
            )
            self.progress_cache = {}
            self._emit(
                "continuous_resources_ready",
                checkpoints=self.progress_model.hashes,
                density_path=os.environ["KAIROS_DENSITY_MODEL"],
                reference_step=self.progress_model.reference_step,
                quadrature=3,
                posterior_order=False,
                future_information=False,
            )
            self._emit(
                "continuous_compilation_ready",
                interpolation_knots=192,
                learned_density_unchanged=True,
                exact_state_reuse=True,
                packed_input_copies=2,
                unused_prediction_heads_loaded=0,
                future_information=False,
            )
        return result

    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed and getattr(self, "program_enabled", False):
            self.progress_model.reset()
            self.progress_cache = {}
        return result

    def _predict_current_output(self, request):
        program = self.program(request)
        saved = self.progress_cache.get(program)
        if saved is not None and saved["request"] == request.request_id:
            return None
        result = super()._predict_current_output(request)
        started = time.perf_counter()
        rid, keep = self.program_keeps[program]
        if rid != request.request_id:
            raise ValueError("Stale current request")
        prefill = self._service(request.num_prompt_tokens, 0, keep)
        visible = observed_input(
            tokenizer=self.forest.tokenizer,
            prompt_ids=request.prompt_token_ids,
            messages=visible_messages(
                self.metadata(request).get("cont_visible_messages")
            ),
            prompts=self.online_inputs[program],
            outputs=self.online_outputs[program],
            service=self.forecast_observed_service.get(program, 0.0),
        )
        prediction = self.progress_model.predict(
            key=program, stage=len(self.online_outputs[program]), visible=visible
        )
        cap = min(
            int(request.max_tokens), self.max_model_len - request.num_prompt_tokens
        )
        middle = time.perf_counter()
        posterior = self.progress_model.density.compile(
            (np.array(prediction["projected"]), np.array(prediction["output_mass"])),
            prefill=prefill,
            reference_step=self.progress_model.reference_step,
            cap=cap,
        )
        self.progress_cache[program] = dict(
            prediction, request=request.request_id, posterior=posterior
        )
        self._emit(
            "continuous_prediction",
            program=program,
            request=request.request_id,
            stage=len(self.online_outputs[program]),
            **prediction,
            reference_prefill=prefill,
            public_cap=cap,
            interpolation_nodes=posterior.nodes,
            conditional_compile_seconds=time.perf_counter() - middle,
            arrival_controller_seconds=time.perf_counter() - started,
            current_output_known=False,
            future_information=False,
        )
        return result

    def _quotes(self, ready, scored, forecasts, calls):
        result = super()._quotes(ready, scored, forecasts, calls)
        scope = getattr(self, "_continuity_quote_scope", None)
        groups = [result] + ([scope["quotes"]] if scope is not None else [])
        for group in groups:
            for program, quote in list(group.items()):
                saved = self.progress_cache[program]
                posterior = saved["posterior"]
                if (
                    saved["request"] != quote["request"]
                    or posterior.cap != quote["output_cap"]
                ):
                    raise ValueError("Stale posterior or changed public cap")
                value = posterior.quote(
                    generated=int(quote["generated_output"]),
                    restore=quote["restore_s"],
                    decode_step=quote["decode_step_s"],
                )
                group[program] = dict(
                    quote,
                    forest_score=quote["score"],
                    forest_actual_s=quote["actual_s"],
                    forest_progress=quote["progress"],
                    **value,
                    progress_input_sha256=saved["input_sha256"],
                    progress_source="continuous-joint-geometric",
                )
        return result
