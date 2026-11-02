"""Causal request inputs, private stop budgets, and completion observations."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kairos._runtime.replay import replay
from kairos.replay.boundary import generation_parameters, validate_arrived_metadata
from kairos.replay.completion import (
    completion_metadata,
    observe_replay_completion,
    prepare_manifest as prepare_completion,
)
from kairos.replay.execution import prepare_manifest, replay_key, update_output


class BoundaryTests(unittest.TestCase):
    def test_public_limit_depends_only_on_arrived_prompt_and_context(self):
        self.assertEqual(generation_parameters(64, 16384)["max_tokens"], 16320)
        for key in (
            "cont_final",
            "cont_oracle_gap",
            "cont_output_tokens",
            "cont_future_messages",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_arrived_metadata({key: 1})
        with self.assertRaises(ValueError):
            validate_arrived_metadata({"cont_tool": "pytest"})

    def test_private_stop_never_changes_public_max_tokens(self):
        program = dict(id="p", turns=[dict(added_token_ids=[1, 2], output_tokens=3)])
        key = replay_key("kairos", "p", 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.json"
            prepare_manifest([("kairos", [program])], path, 16384, 16384)
            r = SimpleNamespace(
                output_token_ids=[],
                max_tokens=8192,
                status="RUNNING",
                sampling_params=SimpleNamespace(
                    ignore_eos=True,
                    stop=[],
                    stop_token_ids=[],
                    extra_args=dict(cont_replay_key=key),
                ),
            )

            def native(request, ids):
                request.output_token_ids.extend(ids)
                return ids, False

            with patch.dict(os.environ, CONTINUUM_REPLAY_EXECUTION_MANIFEST=str(path)):
                self.assertEqual(
                    update_output(r, [10, 11], native, "FINISHED"), ([10, 11], False)
                )
                self.assertEqual(
                    update_output(r, [12, 13], native, "FINISHED"), ([12], True)
                )
            self.assertEqual(r.max_tokens, 8192)
            self.assertEqual(r.output_token_ids, [10, 11, 12])
            self.assertEqual(r.status, "FINISHED")
            self.assertFalse(hasattr(r, "output_budget"))

    def test_completed_action_becomes_visible_only_after_output(self):
        action = dict(
            tool="pytest",
            terminal=False,
            evidence="current-tool-action",
            assistant_sha256="a" * 64,
        )
        program = dict(id="p", turns=[{}])
        key = replay_key("kairos", "p", 0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "actions.json"
            source.write_text(
                json.dumps(
                    dict(schema="recorded-current-actions-v1", actions={"p": [action]})
                )
            )
            path = Path(directory) / "completion.json"
            prepare_completion([("kairos", [program])], source, path)
            meta = dict(cont_program_id="p", cont_tool="unknown", cont_replay_key=key)
            r = SimpleNamespace(
                sampling_params=SimpleNamespace(extra_args=meta),
                output_token_ids=[],
                status=SimpleNamespace(name="RUNNING"),
                is_finished=lambda: False,
            )
            with patch.dict(os.environ, CONTINUUM_REPLAY_COMPLETION_MANIFEST=str(path)):
                self.assertEqual(completion_metadata(r), meta)
                with self.assertRaises(ValueError):
                    observe_replay_completion(r)
                r.output_token_ids = [1]
                r.status.name = "FINISHED_STOPPED"
                r.is_finished = lambda: True
                self.assertEqual(observe_replay_completion(r), action)
                self.assertIsNone(observe_replay_completion(r))
                self.assertEqual(completion_metadata(r)["cont_tool"], "pytest")
                self.assertEqual(r.sampling_params.extra_args["cont_tool"], "unknown")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_payload_hides_fixed_work_and_future_messages(self):
        payloads = []
        counts = [3, 5]

        class Response:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def aiter_lines(self):
                yield "data: " + json.dumps(
                    dict(choices=[dict(token_ids=[99] * counts[len(payloads) - 1])])
                )
                yield "data: [DONE]"

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def stream(self, *args, **kwargs):
                payloads.append(copy.deepcopy(kwargs["json"]))
                return Response()

        module = SimpleNamespace(
            AsyncClient=Client, Timeout=lambda *a, **k: None, Limits=lambda **k: None
        )
        row = dict(
            id="p",
            arrival_s=0.0,
            turns=[
                dict(added_token_ids=[1, 2], output_tokens=3, gap_after_s=0.0),
                dict(added_token_ids=[3], output_tokens=5, gap_after_s=0.0),
            ],
        )
        messages = {
            "p": [
                [dict(role="user", content="first")],
                [dict(role="user", content="arrived continuation")],
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visible.json"
            path.write_text(json.dumps(messages))
            args = SimpleNamespace(
                model="model",
                port=123,
                max_num_seqs=8,
                max_model_len=16384,
                request_timeout=1,
                run_timeout=2,
                reuse_engine=True,
                current_method="kairos",
            )
            with (
                patch.dict(sys.modules, {"httpx": module}),
                patch.dict(
                    os.environ,
                    CONTINUUM_WORK_MESSAGES=str(path),
                    CONTINUUM_REPLAY_OUTPUT_LIMIT="16384",
                ),
            ):
                result = await replay(args, [row], root)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(payloads[1]["prompt"], [1, 2, 3])
        for index, payload in enumerate(payloads):
            meta = payload["vllm_xargs"]
            validate_arrived_metadata(meta)
            self.assertEqual(meta["cont_tool"], "unknown")
            self.assertNotIn("cont_final", meta)
            self.assertEqual(
                json.loads(meta["cont_visible_messages"]), messages["p"][index]
            )
            self.assertEqual(payload["max_tokens"], 16384 - len(payload["prompt"]))
            self.assertNotEqual(payload["max_tokens"], counts[index])


if __name__ == "__main__":
    unittest.main()
