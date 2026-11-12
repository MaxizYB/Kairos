"""CPU reconstruction of recorded causal quotes; never performs scheduling.

Use --write-fixture only when maintaining the small public regression fixture.
The input log is an explicit archive path, not a dependency of the package.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from kairos.prediction.joint import JointDensity

ROOT = Path(__file__).resolve().parents[1]


def check(path, fixture=False):
    with np.load(ROOT / "assets/joint/density.npz") as data:
        density = JointDensity({k: data[k] for k in data.files})
    step = None
    laws, cases = {}, {}
    quotes = calls = 0
    maximum = 0.0
    with Path(path).open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "continuous_resources_ready":
                step = event["reference_step"]
            if event["event"] == "continuous_prediction":
                if step is None or event.get("future_information"):
                    raise ValueError("Missing causal resource provenance")
                args = dict(
                    prefill=event["reference_prefill"],
                    reference_step=step,
                    cap=event["public_cap"],
                )
                laws[event["request"]] = density.compile(
                    (np.asarray(event["projected"]), np.asarray(event["output_mass"])),
                    **args,
                )
                if fixture and len(cases) < 12:
                    cases[event["request"]] = dict(
                        projected=event["projected"],
                        output_mass=event["output_mass"],
                        **args,
                        quotes=[],
                    )
                calls += 1
            elif event["event"] == "forecast_service":
                group = list(event["quotes"].values())
                group += [
                    h["quote"]
                    for h in event.get("continuity_j", {}).get("held", {}).values()
                    if h["kind"] == "running-current-call"
                ]
                for q in group:
                    args = dict(
                        generated=int(q["generated_output"]),
                        restore=q["restore_s"],
                        decode_step=q["decode_step_s"],
                    )
                    value = laws[q["request"]].quote(**args)
                    for key, actual in value.items():
                        if not math.isclose(
                            actual, q[key], rel_tol=1.0e-10, abs_tol=1.0e-12
                        ):
                            raise ValueError(
                                "Migrated quote differs: " + q["request"] + "/" + key
                            )
                        maximum = max(maximum, abs(actual / q[key] - 1))
                    case = cases.get(q["request"])
                    if case is not None and len(case["quotes"]) < 12:
                        case["quotes"].append(
                            dict(input=args, output={k: q[k] for k in value})
                        )
                    quotes += 1
    if not calls or not quotes:
        raise ValueError("No joint predictions/quotes in supplied log")
    report = dict(
        valid=True,
        calls=calls,
        quotes=quotes,
        max_relative_error=maximum,
        labels_used=False,
        replay="recorded causal quotation only; no new oracle decisions",
    )
    return report, [case for case in cases.values() if case["quotes"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    report, cases = check(args.log, args.write_fixture)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.write_fixture:
        target = ROOT / "tests/data/joint_quotes.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
