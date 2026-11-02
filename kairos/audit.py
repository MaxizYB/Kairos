"""Validate recorded work and recompute released-controller transactions."""

import argparse
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
    for event in events:
        if event.get("future_information"):
            raise ValueError("Future information in serving evidence")
        if event["event"] == "forecast_service":
            for quote in event["quotes"].values():
                if quote["output_exact"]:
                    raise ValueError("Current output was treated as a true length")
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
    return dict(valid=True, cells=cells, repetitions=report["repetitions"])


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
