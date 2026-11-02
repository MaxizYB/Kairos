"""Frozen decision equivalence and capacity/reference invariants on CPU."""

import json
from pathlib import Path
import random
import unittest

from support import attach_pin, controlled, fixture, request
from kairos.audit import audit_events
from kairos.continuity.economics import ReturnLaw
from kairos.observability.distributions import expand_rows
from kairos.prediction.current import condition_output
from kairos.scheduling.admission import physical_index
from kairos.scheduling.contracts import capacity_pages
from kairos.scheduling.priority import idle_index
from kairos.scheduling.score import service_rank

ROOT = Path(__file__).resolve().parents[1]


def snapshot_view(s):
    _, rows, quotes, _, rank = s._snapshot(s.waiting)
    value = dict(
        rows=rows,
        quotes=quotes,
        ready=rank.ready,
        priorities=rank.priorities,
        boundaries=rank.boundaries,
        held=s._continuity_j_proof["held"],
    )

    def normalize(x):
        if isinstance(x, dict):
            return {
                k: normalize(v)
                for k, v in x.items()
                if k not in ("pricing_implementation", "implementation")
            }
        if isinstance(x, (tuple, list)):
            return [normalize(v) for v in x]
        return x

    return normalize(value)


class ControllerTests(unittest.TestCase):
    def test_running_quote_stays_exact_across_output_survival_boundaries(self):
        s = fixture(stages=(1, 1, 1))
        r = s.online_requests["a"]
        s.waiting.remove(r)
        s.running.append(r)
        for generated in (0, 1, 2, 3, 4, 39, 40, 41, 199, 200, 201, 600):
            with self.subTest(generated=generated):
                r.output_token_ids = [1] * generated
                r.num_computed_tokens = 64 + generated
                s._snapshot(s.waiting)
                q = s._continuity_j_proof["held"]["a"]["quote"]
                post = s.program_predictions["a"]["post"]
                expected = service_rank(
                    q["standard_s"],
                    q["actual_s"],
                    post["atoms"],
                    post["mass"],
                    "distribution",
                )
                self.assertAlmostEqual(q["score"], expected["score"])
                self.assertGreater(q["remaining_output"], 0)

    def test_partial_victim_release_keeps_only_uncommitted_slack(self):
        s = controlled()
        s._online_control(s.waiting)
        self.assertEqual(s.online_reserved, dict(guard=3, high=4, low=3))
        self.assertEqual(len(s.pins["low"].blocks), 3)
        self.assertEqual(s.pins["low"].deadline, 100.0)
        e = next(e for e in s.events if e["event"] == "retention_release")
        self.assertEqual((e["references"], e["physical_freed"]), (5, 5))

    def test_shared_tail_cannot_fund_partial_retention(self):
        s = controlled()
        s.pins["low"].blocks[-1].ref_cnt = 2
        s._online_control(s.waiting)
        self.assertNotIn("low", s.pins)
        self.assertEqual(s.online_reserved, dict(guard=3, high=4))

    def test_capacity_claim_cannot_be_spent_on_partial_retention(self):
        s = controlled()
        low = request("low", prompt=159)
        s.online_requests["low"] = low
        s.waiting.add_request(low)
        s._online_control(s.waiting)
        e = next(e for e in s.events if e["event"] == "forecast_admission")
        self.assertEqual(e["retention_retention"]["claim_credit_pages"], 3)
        self.assertEqual(e["retention_retention"]["retained_pages"], 0)

    def test_frozen_numeric_scores_and_capacity_transactions(self):
        cases = json.loads((ROOT / "tests/fixtures/controller.json").read_text())
        for case in cases:
            with self.subTest(case=case["case"]):
                s = fixture(stages=(1, 1, 1), idle=case["idle"])
                for p, progress in case["progress"].items():
                    r = s.online_requests[p]
                    r.output_token_ids = [1] * progress["generated"]
                    r.num_computed_tokens = progress["computed"]
                if case["pin_a"]:
                    attach_pin(s, "a", 6, common=2)
                s.limit, s.max_num_running_reqs = case["limit"], case["slots"]
                s._online_control(s.waiting)
                e = next(
                    e
                    for e in expand_rows(s.events)
                    if e["event"] == "forecast_admission"
                )
                outcome = dict(
                    selected=e["selected"],
                    checks=e["checks"],
                    claim=e["claim"],
                    reserved=s.online_reserved,
                    pins={p: len(pin.blocks) for p, pin in s.pins.items()},
                    retention=e["retention_retention"],
                )
                self.assertEqual(outcome, case["outcome"])
                self.assertEqual(snapshot_view(s), case["snapshot"])
                self.assertEqual(audit_events(expand_rows(s.events))["transactions"], 1)

    def test_cold_and_warm_share_one_progress_order(self):
        s = fixture()
        ready, _, quotes, cold, rank = s._snapshot(s.waiting)
        self.assertEqual(set(ready), set(quotes))
        self.assertFalse(cold)
        self.assertEqual(
            rank.ready,
            tuple(sorted(ready, key=lambda p: (quotes[p]["score"], s.order[p]))),
        )
        self.assertEqual(rank.ready[0], "a")

    def test_no_control_on_unchanged_decode_ticks(self):
        s = fixture()
        s._online_control(s.waiting)
        count = len(s.events)
        for _ in range(50):
            s._online_control(s.waiting)
        self.assertEqual(count, len(s.events))

    def test_repeated_snapshots_refresh_resources_with_one_residency_read(self):
        s = fixture(idle=True)
        s._snapshot(s.waiting)
        before = len(s.rpc)
        for _ in range(5):
            s._snapshot(s.waiting)
        self.assertEqual(len(s.rpc) - before, 5)

    def test_cap_changes_do_not_reserve_future_output(self):
        a, b = fixture(), fixture()
        for r in b.waiting:
            r.max_tokens = 16384 - r.num_prompt_tokens
        for s in (a, b):
            s._online_control(s.waiting)
        self.assertEqual(a.online_reserved, b.online_reserved)
        self.assertEqual(capacity_pages(prompt=6000, cap=2048), 376)
        self.assertEqual(capacity_pages(prompt=6000, cap=8192), 376)
        self.assertEqual(capacity_pages(prompt=64, cap=8192, generated=16), 6)

    def test_physical_idle_index_is_reference_unit_invariant(self):
        base = physical_index(idle_index([2.0, 6.0], [0.4, 0.6], 1.0), 1.7)
        for scale in (0.001, 0.1, 2.0, 1000.0):
            value = physical_index(
                idle_index([2 * scale, 6 * scale], [0.4, 0.6], scale), 1.7
            )
            self.assertAlmostEqual(value["score"], base["score"])

    def test_ttl_economic_objective_matches_direct_enumeration(self):
        rng = random.Random(8181)
        for _ in range(200):
            durations = tuple(rng.uniform(0.01, 4.0) for _ in range(10))
            benefit, rate = rng.random() * 4, rng.random()
            decision = ReturnLaw(durations, max(durations)).choose(benefit, rate)
            candidates = sorted({0.0, *durations})
            utility = lambda t: (
                sum((benefit if d <= t else 0.0) - rate * min(d, t) for d in durations)
                / len(durations)
            )
            expected = max(candidates, key=utility)
            self.assertAlmostEqual(decision["ttl_s"], expected)

    def test_output_survival_has_no_false_exact_length(self):
        post = dict(atoms=[3.0, 40.0, 200.0], mass=[0.2, 0.6, 0.2])
        for generated in (0, 3, 39, 40, 199, 200, 500):
            call = condition_output(post, cap=8192, generated=generated, marginal=post)
            self.assertFalse(call.exact)
            self.assertGreater(call.remaining_tokens, 0)
            self.assertLess(call.total_tokens, 8192)

    def test_distributional_progress_is_not_mean_substitution(self):
        q = service_rank(2.0, 2.0, [0.0, 10.0], [0.5, 0.5], "distribution")
        self.assertAlmostEqual(q["score"], 2 / (0.5 + 0.5 * 2 / 12))
        self.assertNotEqual(q["score"], 7.0)

    def test_tampered_transaction_is_rejected(self):
        s = fixture(idle=True)
        s._online_control(s.waiting)
        events = list(expand_rows(s.events))
        next(e for e in events if e["event"] == "forecast_admission")["checks"][0][
            "target_pages"
        ] = 999
        with self.assertRaises(ValueError):
            audit_events(events)

    def test_ledger_never_debits_active_or_dma_owner(self):
        from kairos.scheduling.capacity import admission

        ranks = {"a": 0, "running": 1, "dma": 2}
        plan = admission(
            "a",
            5,
            {"running": 4, "dma": 4},
            {"running"},
            {"running", "dma"},
            ranks,
            10,
            8,
            locked={"dma"},
        )
        self.assertFalse(plan.funded)
        self.assertFalse(plan.victims)


if __name__ == "__main__":
    unittest.main()
