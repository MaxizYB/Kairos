"""Finite KV protection priced from recovery savings and current competition."""

from __future__ import annotations
import math
import time
from kairos.scheduling.cost import recovery_seconds
from kairos.scheduling.contracts import NATIVE_HEADROOM_PAGES
from kairos.continuity.economics import ReturnLaw, exclusive_pages, marginal_exposure


def frontier_pressure(
    owner,
    unique_pages,
    calls,
    reserved,
    occupied,
    optional,
    priorities,
    limit,
    slots,
    locked=(),
):
    calls = tuple(calls)
    occupied = set(occupied)
    scan_input = dict(
        owner=owner,
        unique_pages=unique_pages,
        calls=list(calls),
        reserved=dict(reserved),
        occupied=sorted(occupied),
        optional=sorted(optional),
        priorities=dict(priorities),
        limit=limit,
        slots=slots,
        locked=sorted(locked),
    )
    if not calls or slots <= len(occupied):
        return dict(
            rate=0.0,
            crossings=0,
            advanced=[],
            displaced=[],
            reason="no-ready-or-slot",
            scan_input=scan_input,
            future_information=False,
        )
    quote = marginal_exposure(
        owner,
        unique_pages,
        calls,
        reserved,
        occupied,
        optional,
        priorities,
        limit,
        slots,
        locked,
    )
    denominator = max(1, min(len(calls), slots - len(occupied)))
    crossings = max(0, len(quote["advanced"]) - len(quote["displaced"]))
    rate = min(1.0, crossings / denominator)
    return dict(
        rate=rate,
        crossings=crossings,
        denominator=denominator,
        advanced=quote["advanced"],
        displaced=quote["displaced"],
        keep_selected=quote["keep"]["selected"],
        release_selected=quote["unpin"]["selected"],
        unique_release_pages=unique_pages,
        reason="frontier-crossing" if rate else "no-crossing",
        scan_input=scan_input,
        exact_JCT_value=False,
        future_information=False,
    )


def _valid_blocks(scheduler, request, length):
    try:
        rows = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    except KeyError:
        return ()
    limit = max(0, int(length) // scheduler.block_size)
    result = []
    for block in list(rows)[:limit]:
        if (
            getattr(block, "is_null", False)
            or getattr(block, "block_hash", None) is None
        ):
            break
        result.append(block)
    return tuple(result)


def _releasable_pages(scheduler, request, length):
    blocks = _valid_blocks(scheduler, request, length)
    try:
        all_blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    except KeyError:
        return ()
    return exclusive_pages(blocks, all_blocks)


def legal_frontier_pressure(scheduler, request, length):
    waiting = [
        r
        for queue in (scheduler.waiting, scheduler.skipped_waiting)
        for r in queue
        if r.request_id not in scheduler.online_granted
        and r.request_id != request.request_id
    ]
    if not waiting:
        return dict(rate=0.0, witness=None, reason="no-ready", future_information=False)
    owner = scheduler.program(request)
    rank = lambda r: (
        scheduler.program_priorities.get(
            scheduler.program(r), scheduler.order.get(scheduler.program(r), 0)
        ),
        scheduler.order.get(scheduler.program(r), 0),
    )
    waiting.sort(key=rank)
    locked = {scheduler.program(r) for r in scheduler.running}
    locked.update(
        (
            p
            for p, r in scheduler.online_requests.items()
            if r.request_id in scheduler.online_granted
        )
    )
    locked.update(scheduler.online_cleaning)
    locked.discard(owner)
    victims = [
        p
        for p in scheduler.pins
        if p != owner
        and p not in locked
        and (p not in scheduler.online_cleaning)
        and scheduler._online_idle_reclaimable(p)
    ]
    victim_rank = (
        getattr(scheduler, "online_victim_rank", None) or scheduler.program_priorities
    )
    victims.sort(
        key=lambda p: victim_rank.get(p, scheduler.order.get(p, 0)), reverse=True
    )
    victims.append(owner)
    reserved = dict(scheduler.online_reserved)
    if owner not in reserved:
        return dict(
            rate=0.0,
            witness=None,
            reason="no-owner-reservation",
            future_information=False,
        )
    blocks = _valid_blocks(scheduler, request, length)
    unique = _releasable_pages(scheduler, request, length)
    reserve_after_cleanup = len(blocks)
    projected_reserved = dict(reserved)
    projected_reserved[owner] = reserve_after_cleanup
    credit = min(len(unique), reserve_after_cleanup)
    if credit <= 0:
        return dict(
            rate=0.0,
            witness=None,
            reason="no-unique-releasable-pages",
            future_information=False,
        )
    occupied = set(scheduler._occupied())
    occupied.discard(owner)
    locked = set(locked)
    locked.discard(owner)
    optional = set(victims)
    optional.add(owner)
    priorities = dict(scheduler.program_priorities)
    for name in (
        set(reserved) | occupied | optional | {scheduler.program(r) for r in waiting}
    ):
        priorities.setdefault(name, scheduler.order[name])
    calls = [(scheduler.program(r), scheduler._online_target(r)) for r in waiting]
    result = frontier_pressure(
        owner,
        credit,
        calls,
        projected_reserved,
        occupied,
        optional,
        priorities,
        scheduler._capacity() - NATIVE_HEADROOM_PAGES,
        scheduler.max_num_running_reqs,
        locked,
    )
    result.update(
        owner=owner,
        unique_releasable_pages=len(unique),
        reserved_pages=reserved[owner],
        post_cleanup_retained_pages=reserve_after_cleanup,
        post_cleanup_projection=True,
        future_information=False,
    )
    return result


IMPLEMENTATION = "kairos-economic-bounded-frontier-economic-hold-v1"


class EconomicRetention:
    def _holding_continuity_fraction(self, request):
        return (
            0.0
            if self._economic_active()
            else super()._holding_continuity_fraction(request)
        )

    def _holding_opportunity_rate(self, request, length):
        if not self._economic_active():
            return super()._holding_opportunity_rate(request, length)
        started = time.perf_counter()
        evidence = legal_frontier_pressure(self, request, length)
        evidence["quote_seconds"] = time.perf_counter() - started
        self.__dict__.setdefault("_economic_last_pressure", {})[
            self.program(request)
        ] = evidence
        return evidence["rate"]

    def _continuation_restore_estimate(self, request, length, default):
        if not self._economic_active():
            return super()._continuation_restore_estimate(request, length, default)
        started = time.perf_counter()
        blocks = _valid_blocks(self, request, length)
        unique = _releasable_pages(self, request, length)
        if not blocks:
            quote = dict(
                benefit_s=0.0,
                keep_restore_s=0.0,
                displaced_restore_s=0.0,
                unique_pages=0,
                source="no-committed-prefix",
            )
        else:
            host = (self._replica_call(dict(op="cpu_residency")) or {}).get(
                "programs", {}
            )
            owner = self.program(request)
            cpu = host.get(owner, {})
            layout = dict(
                cpu_chunks=list(cpu.get("cpu_chunks", ())),
                chunks=cpu.get("chunks", int(length) // 256),
                blocks=[b.block_id for b in blocks],
                tokens=int(length),
            )
            profile = dict(
                beta=list(self.beta),
                reload_fixed_seconds=self.profile.get("reload_fixed_seconds", 0.0),
                reload_seconds_per_token=self.profile.get(
                    "reload_seconds_per_token", 0.0
                ),
            )
            keep = recovery_seconds(layout, (), profile, self.block_size)
            lost = recovery_seconds(layout, unique, profile, self.block_size)
            quote = dict(
                benefit_s=max(0.0, lost - keep),
                keep_restore_s=keep,
                displaced_restore_s=lost,
                unique_pages=len(unique),
                source="economic/observable-layout-marginal-recovery",
                layout=layout,
                removed_pages=sorted(unique),
                profile=profile,
                block_size=self.block_size,
                conditional_on_reuse=True,
            )
        quote["quote_seconds"] = time.perf_counter() - started
        self.__dict__.setdefault("_economic_recovery_quotes", {})[
            request.request_id
        ] = quote
        return quote["benefit_s"]

    def _choose_time_length(self, request, ttl, blocks, source, length):
        if not self._economic_active():
            return (ttl, blocks, source)
        started = time.perf_counter()
        owner = self.program(request)
        recovery = self._economic_recovery_quotes.pop(request.request_id)
        pressure = self._economic_last_pressure.pop(owner)
        original = self.policy_model.original
        tool = str(self.metadata(request)["cont_tool"])
        samples = tuple((float(row[0]) for row in original._hold_samples(tool)))
        law = ReturnLaw(samples, max(samples, default=0.0))
        decision = law.choose(recovery["benefit_s"], pressure["rate"])
        if not math.isclose(decision["ttl_s"], ttl, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "economic audit quote differs from inherited TTL selection"
            )
        seconds = (
            time.perf_counter()
            - started
            + recovery["quote_seconds"]
            + pressure["quote_seconds"]
        )
        self.__dict__.setdefault("_economic_pin_proofs", {})[owner] = dict(
            request=request.request_id,
            ttl_s=decision["ttl_s"],
            benefit_s=recovery["benefit_s"],
            rate=pressure["rate"],
        )
        self._emit(
            "ttl_decision",
            program=owner,
            request=request.request_id,
            phase="completion-post-cleanup-quote",
            ttl_model=IMPLEMENTATION,
            decision=decision,
            recovery=recovery,
            frontier_pressure=pressure,
            return_law=dict(durations=list(law.durations), horizon_s=law.horizon),
            inherited_ttl_s=ttl,
            pricing_compute_s=seconds,
            native_cleanup_deferred=True,
            legacy_wait_reward_removed=True,
            future_information=False,
        )
        return (ttl, blocks, "economic/bounded-frontier-economic-hold")

    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            self.__dict__.pop("_economic_last_pressure", None)
            self.__dict__.pop("_economic_recovery_quotes", None)
            self.__dict__.pop("_economic_pin_proofs", None)
        return result

    def _emit(self, event, **fields):
        if (
            event in ("program_registry", "forecast_registry")
            and self._economic_active()
        ):
            fields.update(
                ttl_model=IMPLEMENTATION,
                ttl_benefit="observable marginal recovery loss",
                ttl_cost="one paired capacity grant/claim/victim scan",
                future_projection=False,
            )
        if event == "pin_decision" and self._economic_active():
            owner = fields.get("program")
            proof = self.__dict__.get("_economic_pin_proofs", {}).pop(owner, None)
            fields.update(
                ttl_model=IMPLEMENTATION,
                economic_model="bounded-frontier-economic-hold",
                alpha_wait_removed=True,
                economic_decision=proof,
            )
        return super()._emit(event, **fields)

    def _economic_active(self):
        return bool(getattr(self, "program_enabled", False))
