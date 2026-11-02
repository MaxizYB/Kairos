"""Transactional admission, safe prefix retention, and physical idle quotation."""

import hashlib
import json
import math
import time
from kairos.scheduling.contracts import NATIVE_HEADROOM_PAGES
from kairos.scheduling.priority import order, training_quantum
from kairos.scheduling.retention import (
    common_prefix,
    private_suffix,
    reference_scan,
    retention_plan,
)


def profile_digest(profile):
    return hashlib.sha256(
        json.dumps(
            profile, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def physical_index(reference, physical_quantum):
    q = reference["quantum_s"]
    if (
        not math.isfinite(q)
        or q <= 0
        or (not math.isfinite(physical_quantum))
        or (physical_quantum <= 0)
    ):
        raise ValueError("Positive independently calibrated probe prices required")
    ratio = physical_quantum / q
    return dict(
        reference,
        score=reference["score"] * ratio,
        actual_s=reference["actual_s"] * ratio,
    )


def _prefix(self, p, request, pin):
    adapter = getattr(getattr(self, "connector", None), "scheduler_adapter", None)
    if (
        adapter is None
        or not pin.model
        or pin.model != adapter.model_name
        or (pin.cache_salt != str(getattr(request, "cache_salt", "") or ""))
        or (self.kv_cache_manager.block_pool.hash_block_size != self.block_size)
    ):
        return None
    cache = self.__dict__.setdefault("_retention_prefix_cache", {})
    saved = cache.get(p)
    if saved and saved[0] is request and (saved[1] is pin):
        return (min(saved[2], len(pin.blocks)), saved[3])
    old, new = (list(pin.lookup_hashes[: len(pin.blocks)]), list(request.block_hashes))
    if (
        len(old) != len(pin.blocks)
        or len(new) < request.num_prompt_tokens // self.block_size
    ):
        return None
    matched, proof = common_prefix(
        old, new[: request.num_prompt_tokens // self.block_size]
    )
    cache[p] = (request, pin, matched, proof)
    return (matched, proof)


def _release(self, p, count, *, keep):
    pin = self.pins[p]
    blocks = list(pin.blocks[-count:])
    kind = "bounded-replacement" if keep else "whole-replacement"
    row = dict(
        program=p,
        kind=kind,
        references=count,
        pin_pages_before=len(pin.blocks),
        deadline_before=pin.deadline,
        reserved_before=dict(self.online_reserved),
        block_ids=[b.block_id for b in blocks],
        ref_counts=[b.ref_cnt for b in blocks],
        occupied=p in self._occupied(),
        cleaning=bool(self.online_cleaning.get(p)),
        native_pages=self._native_pages(self.online_requests[p])
        if p in self.online_requests
        else 0,
        private_required=bool(keep),
        prefix_proof=None,
        future_information=False,
    )
    if keep and (
        row["occupied"]
        or row["cleaning"]
        or row["native_pages"]
        or any((x != 1 for x in row["ref_counts"]))
        or (len(set(row["block_ids"])) != count)
    ):
        raise RuntimeError("Kairos cannot credit active/shared references")
    self._retention_release_kind = kind
    try:
        freed = self._release(
            p, "partial_pressure" if keep else "program_optional_replacement", count
        )
    finally:
        self._retention_release_kind = None
    retained = self.pins.get(p)
    expected = dict(row["reserved_before"])
    if retained:
        expected[p] = len(retained.blocks)
    else:
        expected.pop(p, None)
    if self.online_reserved != expected or (keep and freed != count):
        raise RuntimeError("Kairos release did not settle its physical ledger")
    row.update(
        physical_freed=freed,
        retained=len(retained.blocks) if retained else 0,
        deadline_after=retained.deadline if retained else None,
        reserved_after=dict(self.online_reserved),
    )
    self._emit("retention_release", **row)


IMPLEMENTATION = "kairos-capacity-transaction"
PRICING = "physical-training-probe"


class CapacityControl:
    def _online_control(self, queued):
        if not getattr(self, "program_enabled", False):
            return super()._online_control(queued)
        if not self.online_dirty:
            return
        started = time.perf_counter()
        self._retention_sequence = getattr(self, "_retention_sequence", 0) + 1
        self._sync_frontiers()
        ready, rows, quotes, cold, ranked = self._snapshot(queued)
        self.program_priorities, self.program_boundaries = (
            dict(ranked.priorities),
            dict(ranked.boundaries),
        )
        self.online_rank = dict(ranked.priorities)
        self.online_victim_rank = dict(ranked.priorities)
        self._emit(
            "program_service",
            rows=rows,
            quotes=quotes,
            cold_scores=cold,
            ready_order=list(ranked.ready),
            protection_boundaries=ranked.boundaries,
            native_priorities=ranked.priorities,
            request_ids={p: r.request_id for p, r in ready.items()},
            future_information=False,
        )
        limit = self._capacity() - NATIVE_HEADROOM_PAGES
        if limit < 1:
            raise ValueError("GPU capacity does not cover native safety headroom")
        initial, occupied = (dict(self.online_reserved), self._occupied())
        cleaning = set(self.online_cleaning)
        optional = {
            p
            for p in self.pins
            if p not in occupied | cleaning and self._online_idle_reclaimable(p)
        }
        calls = [
            dict(
                program=p,
                request=ready[p].request_id,
                target_pages=self._online_target(ready[p]),
                target_kind=self._capacity_kind(ready[p]),
                capacity_evidence=self._capacity_evidence(ready[p]),
            )
            for p in ranked.ready
            if ready[p].request_id not in self.online_granted
        ]
        by_owner = {c["program"]: c for c in calls}
        reference = reference_scan(
            calls,
            initial,
            occupied,
            optional,
            ranked.priorities,
            limit,
            self.max_num_running_reqs,
            cleaning=cleaning,
        )
        safety, selected = ({}, set(reference["selected"]))
        for p in reference["victims"]:
            pin = self.pins[p]
            usable, proof = (len(pin.blocks), None)
            if p in ready:
                matched = _prefix(self, p, ready[p], pin) if p not in selected else None
                usable, proof = matched if matched is not None else (0, None)
            safety[p] = dict(
                total_pages=len(pin.blocks),
                usable_pages=usable,
                safe_suffix_pages=private_suffix(pin) if p not in selected else 0,
                prefix_proof=proof,
            )
        retention = retention_plan(
            reference, initial, safety, ranked.priorities, limit, enabled=True
        )
        granted_at_start = sorted(self.online_granted)
        pins_at_start = {p: len(pin.blocks) for p, pin in self.pins.items()}
        for p in reference["victims"]:
            keep = retention["keep"].get(p, 0)
            count = len(self.pins[p].blocks) - keep
            if count:
                _release(self, p, count, keep=keep)
                self.online_insurance.pop(p, None)
                if not keep:
                    self.online_active.discard(p)
        for p in reference["selected"]:
            self._online_grant(
                ready[p], by_owner[p]["target_pages"], "residency-current-call"
            )
        if self.online_reserved != retention["reserved_at_end"]:
            raise RuntimeError("Kairos committed a different plan")
        self._emit(
            "program_admission",
            ready_order=list(ranked.ready),
            selected=reference["selected"],
            checks=reference["checks"],
            claim=reference["claim"],
            reserved_at_start=initial,
            reserved_at_end=dict(self.online_reserved),
            capacity_limit=limit,
            seconds=time.perf_counter() - started,
            future_information=False,
            retention_checks_scope="read-only original plan; actual commits independently checked",
            retention_reference_end=reference["reserved_at_end"],
            retention_retention=retention,
            retention_safety=safety,
            retention_occupied_at_start=sorted(occupied),
            retention_running_at_start=sorted(
                {self.program(r) for r in self.running if not r.is_finished()}
            ),
            retention_cleaning_at_start=sorted(cleaning),
            retention_pins_at_start=pins_at_start,
            retention_granted_at_start=granted_at_start,
            retention_optional_at_start=sorted(optional),
            max_running=self.max_num_running_reqs,
        )
        self.online_dirty = False
        cache = self.__dict__.get("_retention_prefix_cache", {})
        for p in list(cache):
            if p not in self.pins:
                del cache[p]

    def _emit(self, event, **fields):
        if event in (
            "program_registry",
            "forecast_registry",
            "program_service",
            "forecast_service",
            "program_admission",
            "forecast_admission",
            "retention_release",
        ):
            fields.update(
                retention_implementation=IMPLEMENTATION,
                retention_mode="kairos",
                retention_sequence=getattr(self, "_retention_sequence", 0),
                eager_normalization=False,
                idle_pricing=PRICING if True else "inherited-reference",
            )
        if event == "unpin" and getattr(self, "_retention_release_kind", None):
            fields["retention_action"] = self._retention_release_kind
        return super()._emit(event, **fields)

    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            (
                self._retention_prefix_cache,
                self._retention_pricing_cache,
                self._retention_sequence,
            ) = ({}, None, 0)
        return result

    def _snapshot(self, queued):
        from kairos.causal import service

        ready, rows, quotes, cold, original = super()._snapshot(queued)
        proof = self._continuity_j_proof
        held = dict(proof["held"])
        profile = self.profile.get("pipeline_service")
        if (
            not isinstance(profile, dict)
            or "beta" not in profile
            or "1" not in profile.get("decode_step_by_seqs", {})
        ):
            raise ValueError(
                "Kairos physical pricing requires independent pipeline calibration"
            )
        state = self.__dict__.get("_retention_pricing_cache")
        if state is None or state["profile"] is not profile:
            state = dict(profile=profile, digest=profile_digest(profile), probes={})
            self._retention_pricing_cache = state
        for p, q in proof["held"].items():
            if q["kind"] != "idle-suffix-probe":
                continue
            training = q["training"]
            key = (training["next_stage"], training["previous_observed_prompt"])
            if key not in state["probes"]:
                state["probes"][key] = training_quantum(
                    self.service_prior.rows,
                    *key,
                    lambda n, o, k: service(n, o, k, profile),
                    self.max_model_len,
                )
            physical = state["probes"][key]
            converted = physical_index(q, physical["quantum_s"])
            held[p] = dict(
                converted,
                pricing_implementation=PRICING,
                reference_score=q["score"],
                reference_actual_s=q["actual_s"],
                physical_training=physical,
                ratio=physical["quantum_s"] / q["quantum_s"],
                physical_profile_sha256=state["digest"],
            )
        result = order(
            {p: r["total_service_s"] for p, r in rows.items()},
            {p: q["score"] for p, q in quotes.items()},
            cold,
            self.order,
            {p: q["score"] for p, q in held.items()},
        )
        if result.ready != original.ready:
            raise RuntimeError("Idle conversion changed ready dispatch")
        self._continuity_j_proof = dict(proof, held=held)
        return (ready, rows, quotes, cold, result)
