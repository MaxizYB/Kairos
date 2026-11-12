"""Standalone checkout, release assets, launch routing, and document links."""

import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kairos._runtime.activation import RUNTIME_ACTIVATION
from kairos.run import prepare
from kairos.verify import verify_workload

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_local_imports_form_a_complete_checkout(self):
        for path in (ROOT / "kairos").rglob("*.py"):
            if "vendor" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if not name.startswith("kairos."):
                        continue
                    target = ROOT.joinpath(*name.split("."))
                    self.assertTrue(
                        target.with_suffix(".py").is_file()
                        or (target / "__init__.py").is_file(),
                        (path, name),
                    )

    def test_runtime_needs_no_research_checkout(self):
        for path in (ROOT / "kairos").rglob("*.py"):
            if "vendor" in path.parts:
                continue
            text = path.read_text()
            for name in (
                "Phase_continuum",
                "continuum_compat",
                "gapped_compat",
                "/home/caohuilin",
            ):
                self.assertNotIn(name, text, path)

    def test_causal_activation_and_both_predictors_are_wired(self):
        self.assertNotIn("knownoutput", RUNTIME_ACTIVATION["kairos"])
        args = SimpleNamespace(
            method="kairos",
            model="/model",
            gpu=0,
            profile=ROOT / "assets/deployment_profile.json",
            reload_profile=ROOT / "assets/reload_profile.json",
            engine_cpus="0-3",
            helper_cpus="4-7",
        )
        with patch.dict(
            "os.environ",
            dict(
                CONTINUUM_SCALAR_RANK_TRUTH="/private/truth",
                KAIROS_STALE_FLAG="1",
                PYTHONPATH="/research/tree",
            ),
        ):
            _, engine, helper, env = prepare(
                args, ROOT / "runs/test", "/tmp/kairos-test.sock"
            )
        self.assertNotIn("CONTINUUM_SCALAR_RANK_TRUTH", env)
        self.assertNotIn("KAIROS_STALE_FLAG", env)
        self.assertEqual(env["PYTHONPATH"], str(ROOT))
        self.assertIn("kairos.scheduler.KairosScheduler", engine)
        self.assertEqual(env["KAIROS_FORECAST_BACKEND"], "1")
        for key in (
            "KAIROS_PREDICTOR_FOREST",
            "CONTINUUM_CURRENT_OUTPUT_FOREST",
            "CONTINUUM_CURRENT_OUTPUT_FALLBACK",
        ):
            self.assertTrue(Path(env[key]).is_file())
        self.assertEqual(
            Path(env["KAIROS_DENSITY_MODEL"]), ROOT / "assets/joint/density.npz"
        )
        for key in (
            "KAIROS_PROGRESS_ENCODER",
            "KAIROS_PROGRESS_CHECKPOINT",
            "KAIROS_COUPLED_STUDY",
        ):
            self.assertTrue(Path(env[key]).is_dir())
        self.assertEqual(env["KAIROS_STARVATION_GUARD"], "1")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        args.joint_assets = Path("/external/joint")
        _, _, _, external_env = prepare(
            args, ROOT / "runs/test", "/tmp/kairos-test.sock"
        )
        self.assertEqual(
            external_env["KAIROS_DENSITY_MODEL"], "/external/joint/density.npz"
        )
        self.assertEqual(
            external_env["KAIROS_PROGRESS_CHECKPOINT"], "/external/joint/checkpoint"
        )
        self.assertEqual(helper[helper.index("--l1-size-gb") + 1], "10.0")
        self.assertEqual(
            engine[engine.index("--kv-cache-memory-bytes") + 1], "5935988736"
        )

    def test_model_features_and_training_identity_sets_are_explicit(self):
        for name in (
            "forest.json",
            "current_output.json",
            "current_output_metadata.json",
        ):
            model = json.loads((ROOT / "assets" / name).read_bytes())
            self.assertEqual(model["features"], 127)
            self.assertEqual(len(model["trees"]), 48)
            self.assertGreater(len(model["training_ids"]), 100)
        self.assertEqual(
            json.loads((ROOT / "assets/forest.json").read_bytes())["split_objective"],
            "balanced",
        )

    def test_all_arrivals_keep_exactly_the_same_recorded_work(self):
        self.assertEqual(verify_workload(), 178)
        traces = [
            json.loads((ROOT / "data/workload" / name).read_bytes())["programs"]
            for name in ("trace.json", "poisson_jps1.json", "poisson_jps2.json")
        ]
        payload = lambda ps: [(p["id"], p["turns"]) for p in ps]
        self.assertEqual(payload(traces[0]), payload(traces[1]))
        self.assertEqual(payload(traces[0]), payload(traces[2]))
        self.assertEqual(
            sum(t["output_tokens"] for p in traces[0] for t in p["turns"]), 26164
        )

    def test_documents_have_resolvable_relative_links(self):
        for path in (ROOT / "README.md", ROOT / "Kairos_Method_Design_Section_2_ZH.md"):
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" not in link and not link.startswith("#"):
                    self.assertTrue(
                        (path.parent / link.split("#")[0]).exists(), (path, link)
                    )


if __name__ == "__main__":
    unittest.main()
