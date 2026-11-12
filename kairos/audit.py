"""Validate recorded work and recompute released-controller transactions."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from kairos.scheduling.retention import reference_scan, retention_plan
from kairos.scheduling.score import service_rank

ROOT = Path(__file__).resolve().parents[1]


def audit_events(events):
    counts = dict(snapshots=0, transactions=0, grants=0, bounded_releases=0)
    current_priority = None
    joint_laws, joint_density, reference_step = {}, None, None
    for event in events:
        if event.get("future_information"):
            raise ValueError("Future information in serving evidence")
        if event["event"] == "continuous_resources_ready":
            reference_step = event["reference_step"]
        elif event["event"] == "continuous_prediction":
            import numpy as np
            from kairos.prediction.joint import JointDensity

            if reference_step is None:
                raise ValueError("Joint prediction has no resource provenance")
            if joint_density is None:
                with np.load(ROOT / "assets/joint/density.npz") as data:
                    joint_density = JointDensity({k: data[k] for k in data.files})
            joint_laws[event["request"]] = joint_density.compile(
                (np.asarray(event["projected"]), np.asarray(event["output_mass"])),
                prefill=event["reference_prefill"],
                reference_step=reference_step,
                cap=event["public_cap"],
            )
        elif event["event"] == "forecast_service":
            quotes = list(event["quotes"].values())
            quotes += [
                row["quote"]
                for row in event.get("continuity_j", {}).get("held", {}).values()
                if row["kind"] == "running-current-call"
            ]
            for quote in quotes:
                if quote["output_exact"]:
                    raise ValueError("Current output was treated as a true length")
                if quote.get("progress_source") == "continuous-joint-geometric":
                    expected = joint_laws[quote["request"]].quote(
                        generated=int(quote["generated_output"]),
                        restore=quote["restore_s"],
                        decode_step=quote["decode_step_s"],
                    )
                else:
                    expected = service_rank(
                        quote["standard_s"],
                        quote["actual_s"],
                        quote["atoms_s"],
                        quote["mass"],
                        "distribution",
                    )
                if not math.isclose(expected["score"], quote["score"], rel_tol=1e-10):
                    raise ValueError("Progress score differs")
            counts["snapshots"] += 1
        elif event["event"] == "forecast_admission":
            checks = event["checks"]
            calls = [
                {
                    k: row[k]
                    for k in (
                        "program",
                        "request",
                        "target_pages",
                        "target_kind",
                        "capacity_evidence",
                    )
                }
                for row in checks
            ]
            # Priorities cover running and idle owners as well; bind to the
            # resource order recorded in the same service snapshot below.
            if current_priority is None:
                raise ValueError("Capacity transaction has no service snapshot")
            priorities = current_priority
            reference = reference_scan(
                calls,
                event["reserved_at_start"],
                set(event["retention_occupied_at_start"]),
                set(event["retention_optional_at_start"]),
                priorities,
                event["capacity_limit"],
                event["max_running"],
                cleaning=set(event["retention_cleaning_at_start"]),
            )
            retained = retention_plan(
                reference,
                event["reserved_at_start"],
                event["retention_safety"],
                priorities,
                event["capacity_limit"],
                enabled=True,
            )
            for key in ("selected", "checks", "claim"):
                if reference[key] != event[key]:
                    raise ValueError("Capacity transaction differs: " + key)
            if (
                retained != event["retention_retention"]
                or retained["reserved_at_end"] != event["reserved_at_end"]
            ):
                raise ValueError("Unsafe or inconsistent retained prefix")
            counts["transactions"] += 1
            counts["grants"] += len(reference["selected"])
        elif event["event"] == "retention_release":
            if event["retained"]:
                if (
                    event["occupied"]
                    or event["cleaning"]
                    or event["native_pages"]
                    or any(v != 1 for v in event["ref_counts"])
                    or event["physical_freed"] != event["references"]
                    or event["deadline_before"] != event["deadline_after"]
                ):
                    raise ValueError("Partial release violated native reference safety")
                counts["bounded_releases"] += 1
        if event["event"] == "forecast_service":
            current_priority = event["native_priorities"]
    return counts


def audit_benchmarks():
    report = json.loads((ROOT / "results/benchmark/summary.json").read_bytes())
    cells = []
    for cell in report["cells"]:
        directory = ROOT / "results/benchmark" / cell["workload"] / cell["method"]
        summary = json.loads((directory / "summary.json").read_bytes())
        jobs = json.loads((directory / "jobs.json").read_bytes())
        requests = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        if (len(jobs), len(requests), summary["completed"], summary["failed"]) != (
            16,
            178,
            178,
            0,
        ):
            raise ValueError("Incomplete benchmark cell")
        if any(
            r["error"] or r["output_tokens"] != r["planned_output_tokens"]
            for r in requests
        ):
            raise ValueError("Recorded work differs")
        mean = statistics.fmean(j["jct_s"] for j in jobs)
        if not math.isclose(mean, summary["jct_mean_s"], rel_tol=1e-12):
            raise ValueError("Program JCT aggregation differs")
        cells.append(
            dict(workload=cell["workload"], method=cell["method"], mean_jct_s=mean)
        )
    joint_path = ROOT / "results/benchmark/cg1/summary.json"
    joint_cells = []
    if joint_path.exists():
        joint_report = json.loads(joint_path.read_bytes())
        for cell in joint_report["cells"]:
            paths = {}
            for artifact in cell["artifacts"]:
                path = ROOT / artifact["path"]
                if (
                    not path.resolve().is_relative_to(ROOT)
                    or hashlib.sha256(path.read_bytes()).hexdigest()
                    != artifact["sha256"]
                ):
                    raise ValueError("Modified CG1 experiment artifact")
                paths[path.name] = path
            summary = json.loads(paths["summary.json"].read_bytes())
            if (summary["jobs"], summary["completed"], summary["failed"]) != (
                cell["programs"],
                cell["calls"],
                0,
            ):
                raise ValueError("Incomplete CG1 experiment")
            if not math.isclose(
                summary["jct_mean_s"], cell["metrics"]["mean_jct_s"], rel_tol=1e-12
            ):
                raise ValueError("CG1 summary differs")
            if "jobs.json" in paths:
                jobs = json.loads(paths["jobs.json"].read_bytes())
                if len(jobs) != cell["programs"] or not math.isclose(
                    statistics.fmean(j["jct_s"] for j in jobs),
                    summary["jct_mean_s"],
                    rel_tol=1e-12,
                ):
                    raise ValueError("CG1 per-program aggregation differs")
            if "requests.jsonl" in paths:
                requests = [
                    json.loads(line)
                    for line in paths["requests.jsonl"].read_text().splitlines()
                ]
                if len(requests) != cell["calls"] or any(
                    r["error"] or r["output_tokens"] != r["planned_output_tokens"]
                    for r in requests
                ):
                    raise ValueError("CG1 fixed work differs")
            joint_cells.append(
                dict(group=cell["group"], method=cell["method"], calls=cell["calls"])
            )
    return dict(
        valid=True,
        cells=cells,
        repetitions=report["repetitions"],
        cg1_cells=joint_cells,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        from kairos.observability.distributions import expand_rows

        path = args.run / "scheduler.jsonl"
        result = audit_events(expand_rows(json.loads(line) for line in path.open()))
    else:
        result = audit_benchmarks()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
