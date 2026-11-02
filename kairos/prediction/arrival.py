"""prediction.arrival."""

import json
import time


class CausalPredictor:
    def __init__(self, forest, service):
        self.forest, self.service = (forest, service)
        self.previous, self.stages, self.seen = ({}, {}, {})

    def observe(self, observation):
        started = time.perf_counter()
        p = observation["p"]
        ids, outputs = (observation["ids"], observation["outputs"])
        stage = len(outputs)
        previous_stage = self.stages.get(p, -1)
        retry = observation.get("same_stage_retry", False) and stage == previous_stage
        if (
            stage != previous_stage + 1
            and (not retry)
            or len(observation["inputs"]) != stage + 1
            or any((type(x) is not int or x < 1 for x in outputs))
        ):
            raise ValueError("Nonsequential causal completed-call observation")
        previous = self.previous.get(p, ())
        increment = 0.0
        if outputs and (not retry):
            if not previous:
                raise ValueError("Completed call has no previously arrived prompt")
            increment = self.service(
                len(previous), outputs[-1], observation["observed_completed_keep"]
            )
            self.seen[p] = self.seen.get(p, 0.0) + increment
        self.previous[p], self.stages[p] = (ids, stage)
        post, inference_s, missing = (None, 0.0, None)
        if outputs:
            raw = observation["raw"]
            try:
                messages = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                messages = None
            if not isinstance(messages, list) or not messages:
                missing = "missing-visible-messages"
            else:
                inference = time.perf_counter()
                post = self.forest.predict(
                    text=self.forest.tokenizer.decode(ids, skip_special_tokens=False),
                    messages=messages,
                    prompt_lengths=list(observation["inputs"]),
                    outputs=list(outputs),
                    services=self.seen[p],
                )
                inference_s = time.perf_counter() - inference
        return dict(
            observation=observation,
            keep=observation["current_keep"],
            services=self.seen.get(p, 0.0),
            prediction=post,
            service_audit=dict(
                source="observed-completed-output",
                output_tokens=outputs[-1] if outputs else None,
                increment_s=increment,
                previous_prompt_tokens=len(previous),
                previous_keep_tokens=observation["observed_completed_keep"],
            ),
            missing=missing,
            seconds=time.perf_counter() - started,
            inference_s=inference_s,
        )
