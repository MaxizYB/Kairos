"""Server CPU check: strict packaged weights and optional encoder equivalence.

Uses torch only in the server environment; does not start vLLM or use a GPU.
"""

import argparse
import json
from pathlib import Path
import time


def check(reference=None, assets=None):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from kairos.prediction.encoder import FeatureEncoder, VisibleTokenizer, sha

    root = Path(__file__).resolve().parents[1]
    assets = Path(assets) if assets is not None else root / "assets/joint"
    lock = json.loads((assets / "checkpoint/SELECTION_LOCK.json").read_bytes())
    for key, expected in lock["checkpoints"].items():
        if sha(assets / "checkpoint" / (key + ".pt")) != expected:
            raise ValueError("Checkpoint changed: " + key)
    artifact = torch.load(
        assets / "checkpoint/expanded.pt", map_location="cpu", weights_only=True
    )
    state = {k: v for k, v in artifact["state"].items() if not k.startswith("head.")}
    torch.set_num_threads(1)
    model = FeatureEncoder(assets / "encoder").eval()
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(assets / "encoder", local_files_only=True)
    inputs = VisibleTokenizer(tokenizer, artifact["mean"], artifact["scale"])
    original = None
    if reference:
        original = AutoModel.from_pretrained(reference, local_files_only=True).eval()
        original.load_state_dict(
            {
                k.removeprefix("encoder."): v
                for k, v in state.items()
                if k.startswith("encoder.")
            },
            strict=True,
        )
        prior_tokenizer = AutoTokenizer.from_pretrained(
            reference, local_files_only=True
        )
    errors, cases = [], 0
    started = time.perf_counter()
    with torch.inference_mode():
        for text in (
            "",
            "Fix the failing test in the repository.",
            "tool output " * 100,
        ):
            visible = dict(
                issue=text, previous=text[::-1], feedback=text, numeric=[0.0] * 127
            )
            data = inputs.one(visible)
            if original is not None:
                old = VisibleTokenizer(
                    prior_tokenizer, artifact["mean"], artifact["scale"]
                ).one(visible)
                if (
                    old["input_ids"] != data["input_ids"]
                    or old["token_type_ids"] != data["token_type_ids"]
                ):
                    raise ValueError("Packaged tokenization changed")
            ids = torch.zeros((1, 384), dtype=torch.long)
            types, mask = ids.clone(), ids.clone()
            n = len(data["input_ids"])
            ids[0, :n] = torch.tensor(data["input_ids"])
            types[0, :n] = torch.tensor(data["token_type_ids"])
            mask[0, :n] = 1
            batch = dict(input_ids=ids, token_type_ids=types, attention_mask=mask)
            hidden = model.encoder(**batch).last_hidden_state
            if not torch.isfinite(hidden).all():
                raise ValueError("Nonfinite encoder output")
            if original is not None:
                expected = original(**batch).last_hidden_state
                torch.testing.assert_close(hidden, expected, rtol=0.0, atol=0.0)
                errors.append(float(torch.max(torch.abs(hidden - expected))))
            numeric = model.numeric(torch.from_numpy(data["numeric"])[None])
            if numeric.shape != (1, 64) or hidden.shape[-1] != 384:
                raise ValueError("Joint feature dimensions differ")
            cases += 1
    return dict(
        valid=True,
        device="cpu",
        strict_state_dict=True,
        cases=cases,
        reference_compared=reference is not None,
        max_absolute_error=max(errors) if errors else None,
        seconds=time.perf_counter() - started,
        training_programs=len(artifact["training_ids"]),
        full_inference_gpu_test=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-encoder", type=Path)
    parser.add_argument("--joint-assets", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.reference_encoder, args.joint_assets)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
