"""Public entrypoint and continuous-law contracts, runnable without torch/vLLM."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from support import controller_type
from kairos.prediction.joint import JointDensity

ROOT = Path(__file__).resolve().parents[1]


class ContinuousLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deterministic synthetic law: no learned tensors are needed in Git.
        state = dict(
            log_output_centers=np.log([24.0, 180.0]),
            log_residual_centers=np.log([0.3, 4.0]),
            output_width_raw=np.array([-1.0, -0.8]),
            residual_width_raw=np.array([-1.0, -0.6]),
            output_atoms=np.array([1.0, 256.0]),
        )
        state["project.weight"] = np.zeros((8, 448))
        state["project.bias"] = np.linspace(-1.0, 1.0, 8)
        state["condition.weight"] = np.linspace(-0.1, 0.1, 8)[:, None]
        for name, width in (("output_logits", 2), ("output", 3)):
            state[name + ".1.weight"] = np.ones(8)
            state[name + ".1.bias"] = np.zeros(8)
            state[name + ".2.weight"] = np.arange(width * 8).reshape(width, 8) * 0.01
            state[name + ".2.bias"] = np.linspace(-0.3, 0.3, width)
        cls.density = JointDensity(state)
        projected, mass = cls.density.prepare(np.zeros(448))
        cls.case = dict(
            projected=projected,
            output_mass=mass,
            prefill=0.11,
            reference_step=0.008,
            cap=16384,
        )

    def law(self, case=None, cap=None):
        case = self.case if case is None else case
        return self.density.compile(
            (np.asarray(case["projected"]), np.asarray(case["output_mass"])),
            prefill=case["prefill"],
            reference_step=case["reference_step"],
            cap=cap or case["cap"],
        )

    def test_no_future_service_and_one_output_reduce_to_current_physical_cost(self):
        state = dict(self.density.state)
        state["output.2.bias"] = np.array([0.0, -1000.0, -1000.0])
        density = JointDensity(state)
        law = density.compile(
            density.prepare(np.zeros(448)), prefill=0.11, reference_step=0.008, cap=1
        )
        actual = law.quote(generated=0, restore=0.1, decode_step=0.01)
        self.assertAlmostEqual(actual["score"], 0.11)
        self.assertAlmostEqual(actual["progress"], 1.0)

    def test_generated_tokens_and_live_cost_have_separate_invalidation(self):
        law = self.law()
        first = law.quote(generated=0, restore=0.0, decode_step=0.01)
        second = law.quote(generated=0, restore=1.0, decode_step=0.01)
        self.assertGreater(second["score"], first["score"])
        self.assertEqual(second["progress"], first["progress"])
        self.assertEqual(law.stats["survival_updates"], 1)
        law.quote(generated=1, restore=1.0, decode_step=0.01)
        self.assertEqual(law.stats["survival_updates"], 2)

    def test_unchanged_quote_is_copied_and_exactly_reused(self):
        law = self.law()
        args = dict(generated=0, restore=0.1, decode_step=0.01)
        value = law.quote(**args)
        value["score"] = -1
        self.assertGreater(law.quote(**args)["score"], 0)
        self.assertEqual(law.stats["quote_hits"], 1)

    def test_public_cap_and_observation_are_not_actual_output_labels(self):
        law = self.law(cap=128)
        for n in (-1, 128, 1.5):
            with self.assertRaises(ValueError):
                law.quote(generated=n, restore=0.0, decode_step=0.01)
        with self.assertRaises(TypeError):
            law.quote(
                generated=0, restore=0.0, decode_step=0.01, actual_output_length=10
            )


class PublicAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = controller_type(public=True)
        cls.core = next(
            base for base in cls.public.__mro__ if base.__name__ == "_SchedulingCore"
        )

    def setUp(self):
        self.scheduler = object.__new__(self.public)

    def test_public_entrypoint_includes_joint_and_ownership(self):
        names = [base.__name__ for base in self.public.__mro__]
        self.assertIn("JointForecasts", names)
        self.assertIn("ReservationSettlement", names)
        self.assertIsNot(self.public, self.core)

    def test_same_request_does_not_repeat_inference(self):
        s = self.scheduler
        s.progress_cache = {"p": dict(request="r")}
        with (
            patch.object(self.core, "program", return_value="p"),
            patch.object(
                self.core,
                "_predict_current_output",
                side_effect=AssertionError("Repeated inference"),
            ),
        ):
            self.assertIsNone(
                s._predict_current_output(SimpleNamespace(request_id="r"))
            )

    def test_ready_and_running_use_geometric_not_forest_quotes(self):
        s = self.scheduler
        s._continuity_quote_scope = dict(quotes={})

        class Posterior:
            cap = 100

            def quote(self, **kw):
                return dict(
                    score=kw["decode_step"] * (100 - kw["generated"]) + kw["restore"],
                    actual_s=1.0,
                    progress=0.5,
                    posterior_progress=0.6,
                )

        s.progress_cache = {
            p: dict(request=p, posterior=Posterior(), input_sha256=p)
            for p in ("ready", "running")
        }

        def quote(p, n):
            return dict(
                request=p,
                output_cap=100,
                generated_output=n,
                restore_s=0.2,
                decode_step_s=0.01,
                score=7.0,
                actual_s=2.0,
                progress=0.2,
            )

        def native(*args):
            s._continuity_quote_scope["quotes"] = {"running": quote("running", 20)}
            return {"ready": quote("ready", 0)}

        with patch.object(self.core, "_quotes", side_effect=native):
            ready = s._quotes({}, set(), {}, {})
        self.assertEqual(ready["ready"]["score"], 1.2)
        self.assertEqual(s._continuity_quote_scope["quotes"]["running"]["score"], 1.0)
        self.assertEqual(ready["ready"]["forest_score"], 7.0)
        self.assertEqual(
            ready["ready"]["progress_source"], "continuous-joint-geometric"
        )

    def test_completion_uses_own_posterior_only(self):
        s = self.scheduler
        s.progress_cache = {
            "p": dict(request="r", posterior=SimpleNamespace(stats=dict(quotes=8)))
        }
        with patch.object(self.core, "_emit") as emit:
            s._emit("forecast_current_completed", program="p", request="r")
            self.assertEqual(emit.call_args.kwargs["posterior_reuse"], dict(quotes=8))
            with self.assertRaises(ValueError):
                s._emit("forecast_current_completed", program="p", request="old")

    def test_switch_resets_joint_state_only_when_changed(self):
        s = self.scheduler
        s.experiment, s.program_enabled, s.progress_cache = "one", True, {"p": 1}
        s.progress_model = SimpleNamespace(reset=lambda: None)
        with (
            patch.object(self.core, "_switch_experiment"),
            patch.object(s.progress_model, "reset") as reset,
        ):
            s._switch_experiment("one")
            reset.assert_not_called()
            s._switch_experiment("two")
            reset.assert_called_once()
        self.assertEqual(s.progress_cache, {})
