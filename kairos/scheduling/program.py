"""scheduling.program."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from vllm.v1.core.sched.request_queue import create_request_queue
from kairos.continuity.returns import CommitmentHoldPolicy
from kairos._runtime.compat.online_path import known_request_output
from kairos._runtime.compat.online_path_scheduler import (
    OnlinePathScheduler,
    FinitePathProtection,
)
from kairos.prediction.suffix import SuffixForest
from kairos.prediction.prior import ServicePrior, ToolMarginal
from kairos.scheduling.contracts import NATIVE_HEADROOM_PAGES
from kairos.scheduling.starvation import aged_priority
from kairos.prediction.arrival import CausalPredictor

VERSION = "kairos"
METHOD = "continuity-eager-a~provenanceprior~onlinefitresidual~finishlogical~" + VERSION


class ProgramControl(OnlinePathScheduler):
    VERSION = VERSION
    METHOD = METHOD
    uses_path_evaluator = False
    track_frontier_on_allocation = True

    def __init__(self, *args, **kwargs):
        self.program_enabled = False
        self.program_executor = None
        self.program_resources = None
        self.starvation_guard_enabled = os.environ.get(
            "KAIROS_STARVATION_GUARD", "1"
        ) == "1"
        self.starvation_guard_threshold_s = float(
            os.environ.get("KAIROS_STARVATION_THRESHOLD_S", "4.0")
        )
        self.starvation_guard_quantum_s = float(
            os.environ.get("KAIROS_STARVATION_QUANTUM_S", "1.0")
        )
        self._starvation_guard_seen = set()
        if self.starvation_guard_threshold_s < 0 or self.starvation_guard_quantum_s <= 0:
            raise ValueError("Invalid starvation guard parameters")
        super().__init__(*args, **kwargs)
        if os.environ.get("KAIROS_PREDICTION_PRELOAD") == "1":
            self._prepare_resources()

    def _emit(self, event, **fields):
        if getattr(self, "_program_switching", False) and event in {
            "online_path_registry",
            "holding_economics_registry",
            "scalar_rank_registry",
        }:
            return
        return super()._emit(event, **fields)

    def _prepare_resources(self):
        if self.program_resources is None:
            started = time.perf_counter()
            raw = Path(os.environ["KAIROS_PREDICTOR_PROFILE"]).read_bytes()
            reference = json.loads(raw)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                os.environ["CONTINUUM_MODEL"], local_files_only=True
            )
            forest = SuffixForest(os.environ["KAIROS_PREDICTOR_FOREST"], tokenizer)
            self.program_resources = (
                reference,
                forest,
                hashlib.sha256(raw).hexdigest(),
            )
            self._emit(
                "program_resources_ready",
                seconds=time.perf_counter() - started,
                forest_sha256=forest.sha256,
                profile_sha256=self.program_resources[2],
                request_observations=0,
                future_information=False,
            )
        return self.program_resources

    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        tags = set(name.split("~"))
        enabled = self.VERSION in tags
        if enabled:
            if any(
                value and ("ORACLE" in key or "TRUTH" in key)
                for key, value in os.environ.items()
                if key.startswith(("CONTINUUM_", "KAIROS_"))
            ):
                raise ValueError("Kairos cannot consume future prediction truth")
            if (
                name.split("~", 1)[0] != "continuity-eager-a"
                or not {"onlinefitresidual", "finishlogical"} <= tags
            ):
                raise ValueError(
                    "RESIDENCY requires its registered native/continuity lifecycle"
                )
            allowed = set(self.METHOD.split("~")) | {"heterogeneous"}
            if tags - allowed:
                raise ValueError("Unknown Kairos lifecycle option")
        if self.program_executor is not None:
            self.program_executor.shutdown(wait=True)
            self.program_executor = None
        self._program_switching = enabled
        try:
            super()._switch_experiment(name)
        finally:
            self._program_switching = False
        self.program_enabled = enabled
        if not enabled:
            return
        self._starvation_guard_seen.clear()
        if self.starvation_guard_enabled:
            self._emit(
                "starvation_guard_registry",
                policy="bounded-j-rank-aging",
                threshold_s=self.starvation_guard_threshold_s,
                quantum_s=self.starvation_guard_quantum_s,
                scope="native waiting and skipped-waiting queues only",
                changes_score=False,
                changes_admission=False,
                changes_ttl=False,
                changes_eviction=False,
                future_information=False,
            )
        self.reference_profile, self.forest, reference_sha = self._prepare_resources()
        if self.forest.training_ids & set(self.online_prior["evaluation_ids"]):
            raise ValueError("Forest training/evaluation identity overlap")
        self.forecaster = ToolMarginal(self.online_prior)
        self.policy_model = FinitePathProtection(
            CommitmentHoldPolicy(training=self.forecaster.tool_samples), self.forecaster
        )
        self.service_prior = ServicePrior(
            self.online_prior, self._service, self.max_model_len
        )
        self.program_worker = CausalPredictor(self.forest, self._service)
        self.program_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="residency-predict"
        )
        self.program_pending = deque()
        self.program_arrivals, self.program_predictions = ({}, {})
        self.program_completed, self.program_completed_prefix, self.program_keeps = (
            {},
            {},
            {},
        )
        self.program_retries = set()
        self.program_priorities, self.program_boundaries = ({}, {})
        self.progressive_compete = self.adaptive_compete = False
        self.online_known_output = False
        self._emit(
            "program_registry",
            version=self.VERSION,
            architecture="one ready service queue; revocable continuity; complete admission",
            score="t / E[s/(s+R+)]",
            forest_sha256=self.forest.sha256,
            reference_profile_sha256=reference_sha,
            training_evaluation_disjoint=True,
            representative_paths=False,
            future_information=False,
            current_output="public exact contract or causal empirical service estimate",
            capacity="exact contract: complete call; unknown EOS: observed native frontier",
            running="native allocation, preemption and DMA fences",
            protection="remaining-service continuity boundaries; returned TTL exemption",
            ttl="same empirical economic tool policy; native headroom explicit",
            headroom_pages=NATIVE_HEADROOM_PAGES,
            backend="unchanged continuity",
        )

    def _service(self, prompt, output, keep):
        from kairos.causal import service

        return service(prompt, output, keep, self.reference_profile)

    @staticmethod
    def _common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def add_request(self, request):
        if self.program_enabled:
            previous = self.online_requests.get(self.program(request))
            if (
                previous is not None
                and previous.request_id != request.request_id
                and (not previous.is_finished())
            ):
                raise ValueError("RESIDENCY requires serial calls within each Program")
        super().add_request(request)
        if not self.program_enabled:
            return
        p, rid = (self.program(request), request.request_id)
        if p in self.forest.training_ids:
            raise ValueError("Request identity overlaps forest training")
        old_keep = self.program_keeps.get(p, (None, 0))[1]
        ids = tuple(request.prompt_token_ids)
        keep = (
            min(
                self._common(self.program_completed_prefix.get(p, ()), ids),
                len(ids) - 1,
            )
            // self.block_size
            * self.block_size
        )
        self.program_keeps[p] = (rid, keep)
        raw = self.metadata(request).get("cont_visible_messages")
        if isinstance(raw, list):
            raw = json.dumps(raw)
        observation = dict(
            p=p,
            request_id=rid,
            ids=ids,
            raw=raw,
            inputs=tuple(self.online_inputs[p]),
            outputs=tuple(self.online_outputs[p]),
            observed_completed_keep=old_keep,
            current_keep=keep,
            same_stage_retry=p in self.program_retries,
        )
        if p in self.program_retries:
            self.online_ended.discard(p)
            self.program_retries.discard(p)
        self.program_arrivals[p] = rid
        self.program_predictions.pop(p, None)
        self.program_pending.append(
            self.program_executor.submit(self.program_worker.observe, observation)
        )
        self._emit(
            "program_observation",
            program=p,
            request=rid,
            stage=len(observation["outputs"]),
            current_keep=keep,
            prompt_tokens=request.num_prompt_tokens,
            output_cap=request.max_tokens,
            frozen_at_arrival=True,
            future_information=False,
        )

    def _collect(self):
        while self.program_pending and self.program_pending[0].done():
            result = self.program_pending.popleft().result()
            obs = result["observation"]
            p, rid = (obs["p"], obs["request_id"])
            current = self.program_arrivals.get(p) == rid
            self._emit(
                "program_features",
                program=p,
                request=rid,
                current=current,
                seconds=result["seconds"],
                inference_s=result["inference_s"],
                service_input=result["service_audit"],
                missing=result["missing"],
            )
            if current and result["prediction"] is not None:
                self.program_predictions[p] = dict(
                    request=rid, post=result["prediction"]
                )
                post = result["prediction"]
                self._emit(
                    "program_prediction",
                    program=p,
                    request=rid,
                    atoms_s=post["atoms"],
                    mass=post["mass"],
                    pred_s=post["pred"],
                    observed_service_s=result["services"],
                    future_information=False,
                )

    def _posterior(self, p, request):
        predicted = self.program_predictions.get(p, {})
        if predicted.get("request") == request.request_id:
            post, source = (predicted["post"], "honest_cdf157")
        else:
            post = self.service_prior.predict(
                self.online_inputs[p], self.online_outputs[p], int(request.max_tokens)
            )
            source = post["source"]
        rid, keep = self.program_keeps[p]
        if rid != request.request_id:
            raise ValueError("Stale current-call history boundary")
        return dict(request=rid, keep=keep, post=post, source=source)

    def _idle_posterior(self, p):
        saved = self.program_completed[p]
        fresh = self.program_predictions.get(p, {})
        if fresh.get("request") == saved["request"]:
            return dict(saved, post=fresh["post"], source="honest_cdf157")
        return saved

    def _occupied(self):
        names = {self.program(r) for r in self.running if not r.is_finished()}
        names.update(
            (
                p
                for p, r in self.online_requests.items()
                if r.request_id in self.online_granted
            )
        )
        return names

    def _online_has_slot(self, p):
        if not self.program_enabled:
            return super()._online_has_slot(p)
        occupied = self._occupied()
        return p in occupied or len(occupied) < self.max_num_running_reqs

    def _native_pages(self, request):
        try:
            blocks = self.kv_cache_manager.get_blocks(request.request_id)
        except KeyError:
            return 0
        return len(blocks.blocks[0])

    def _sync_frontiers(self):
        for p, request in self.online_requests.items():
            if (
                request.request_id in self.online_granted
                and known_request_output(request) is None
            ):
                self.online_reserved[p] = max(
                    self._online_target(request), self._native_pages(request)
                )

    def _refresh_priority(self):
        if not self.program_enabled:
            return super()._refresh_priority()
        for name in ("waiting", "skipped_waiting"):
            queue = create_request_queue(self.policy)
            for request in getattr(self, name):
                p = self.program(request)
                request.priority = self.program_priorities.get(p, self.order[p])
                queue.add_request(request)
            setattr(self, name, queue)
        for request in self.running:
            p = self.program(request)
            request.priority = self.program_priorities.get(p, self.order[p])

        if not self.starvation_guard_enabled:
            return
        for name in ("waiting", "skipped_waiting"):
            old = getattr(self, name)
            queue = create_request_queue(self.policy)
            for request in list(old):
                waited = self._starvation_waited_s(request)
                aged = aged_priority(
                    int(request.priority),
                    waited,
                    self.starvation_guard_threshold_s,
                    self.starvation_guard_quantum_s,
                )
                request.priority = int(aged["priority"])
                if aged["aged"] and request.request_id not in self._starvation_guard_seen:
                    self._starvation_guard_seen.add(request.request_id)
                    self._emit(
                        "starvation_guard",
                        request=request.request_id,
                        program=self.program(request),
                        waited_s=aged["waited_s"],
                        base_priority=aged["base_priority"],
                        effective_priority=aged["priority"],
                        credit=aged["credit"],
                        threshold_s=self.starvation_guard_threshold_s,
                        quantum_s=self.starvation_guard_quantum_s,
                        future_information=False,
                    )
                queue.add_request(request)
            setattr(self, name, queue)

    @staticmethod
    def _starvation_waited_s(request):
        """Return queue age without assuming one particular vLLM clock.

        Current vLLM records epoch seconds in arrival_time. A small
        compatibility fallback handles runtimes that use monotonic seconds,
        avoiding a negative age while keeping the decision causal.
        """

        try:
            arrival = float(request.arrival_time)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        now = time.time() if arrival > 100_000_000 else time.monotonic()
        return max(0.0, now - arrival)

    def _continuum_complete_policy(self, request):
        if self.program_enabled:
            self._collect()
            p = self.program(request)
            status = getattr(getattr(request, "status", None), "name", None)
            successful = len(request.output_token_ids) > 0 and status in (
                None,
                "FINISHED_STOPPED",
                "FINISHED_LENGTH_CAPPED",
            )
            if not successful:
                stage = len(self.online_outputs[p])
                result = super()._continuum_complete_policy(request)
                self.online_outputs[p].pop()
                self.online_inputs[p].pop()
                if len(self.online_outputs[p]) != stage:
                    raise RuntimeError(
                        "Abort changed an unrelated completed-call history"
                    )
                self.program_predictions.pop(p, None)
                self.program_retries.add(p)
                self._emit(
                    "program_aborted",
                    program=p,
                    request=request.request_id,
                    stage=stage,
                    output_tokens=len(request.output_token_ids),
                    status=status,
                    future_information=False,
                )
                return result
            forecast = self._posterior(p, request)
            self.program_completed[p] = dict(
                request=request.request_id,
                stage=len(self.online_outputs[p]),
                post=dict(forecast["post"]),
                source=forecast["source"],
            )
            self.program_completed_prefix[p] = tuple(request.prompt_token_ids) + tuple(
                request.output_token_ids[:-1]
            )
            self._emit(
                "program_completed",
                program=p,
                **self.program_completed[p],
                output_tokens=len(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                future_information=False,
            )
        return super()._continuum_complete_policy(request)

    def _release(self, program, *args, **kwargs):
        result = super()._release(program, *args, **kwargs)
        if self.program_enabled and program not in self.pins:
            request = self.online_requests.get(program)
            committed = (
                request is not None and request.request_id in self.online_granted
            )
            if not committed and program not in self.online_cleaning:
                self.online_reserved.pop(program, None)
                self.online_active.discard(program)
            if result and (not committed):
                self.online_dirty = True
        return result

    def _allocate(self, request, num_new_tokens, *args, **kwargs):
        result = super()._allocate(request, num_new_tokens, *args, **kwargs)
        if (
            result is not None
            and self.program_enabled
            and self.track_frontier_on_allocation
            and (request.request_id in self.online_granted)
            and (known_request_output(request) is None)
        ):
            p = self.program(request)
            pages = max(self._online_target(request), self._native_pages(request))
            if self.online_reserved.get(p) != pages:
                self.online_reserved[p] = pages
                self._emit(
                    "program_native_frontier",
                    program=p,
                    request=request.request_id,
                    pages=pages,
                    output_cap=request.max_tokens,
                    scheduled_tokens=num_new_tokens,
                    prediction_used=False,
                    triggers_control=False,
                    online_dirty_after=self.online_dirty,
                )
        return result
