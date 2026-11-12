"""Maintainer: export compact measured CG1 evidence from a research workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]


def export(workspace):
    source = workspace / "Phase_continuum"
    destination = ROOT / "results/benchmark/cg1"
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def copy(path, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return dict(
            path=str(target.relative_to(ROOT)),
            source=str(path.relative_to(workspace)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def cell(label, path, group, programs):
        folder = destination / group / label
        artifacts = []
        for name in (
            "summary.json",
            "audit.json",
            "completion-audit-strict.json",
            "jobs.json",
            "requests.jsonl",
        ):
            if (path / name).exists():
                artifacts.append(copy(path / name, folder / name))
        summary = json.loads((path / "summary.json").read_bytes())
        if summary["jobs"] != programs or summary["failed"] != 0:
            raise ValueError("Incomplete cell: " + str(path))
        metrics = dict(
            mean_jct_s=summary["jct_mean_s"],
            p95_jct_s=summary["jct_p95_s"],
            wall_s=summary["wall_s"],
            mean_ttft_s=summary["ttft_mean_s"],
            p95_ttft_s=summary["ttft_p95_s"],
        )
        if (path / "jobs.json").exists():
            jobs = json.loads((path / "jobs.json").read_bytes())
            if (
                abs(statistics.fmean(j["jct_s"] for j in jobs) - summary["jct_mean_s"])
                > 1.0e-8
            ):
                raise ValueError("JCT aggregation differs")
            metrics["makespan_s"] = max(
                j["arrival_s"] + j["jct_s"] for j in jobs
            ) - min(j["arrival_s"] for j in jobs)
        records.append(
            dict(
                group=group,
                method=label,
                programs=programs,
                calls=summary["completed"],
                metrics=metrics,
                artifacts=artifacts,
            )
        )

    common = source / "cg1_common16_20261111/results"
    for method in ("forest", "cg1"):
        cell(
            method,
            common / "downloaded-r2/results/gpu7-pair-r2" / method,
            "common16_gpu7",
            16,
        )
    cell("continuum", common / "downloaded-continuum-r2/continuum", "common16_gpu7", 16)
    reports = (
        (
            "baseline32_gpu2",
            source / "baseline_comparison_20261111/results-summary/comparison.json",
        ),
        (
            "cg1_cleanup32_gpu1",
            source / "continuous_geometric_20261111/results-summary/comparison.json",
        ),
        (
            "remaining_work32_gpu1",
            source / "remaining_work_20261111/results-summary-gpu1/comparison.json",
        ),
    )
    for group, path in reports:
        report = json.loads(path.read_bytes())
        copy(path, destination / group / "comparison.json")
        for method, row in report["cells"].items():
            cell(method, workspace / row["directory"], group, 32)
    for size, run in ((16, "common16-8b-gpu3-r1"), (32, "common32-8b-gpu3-r2")):
        for method in ("cg1", "cg1_fair"):
            cell(
                method,
                source / "starvation_guard_20261111/results" / run / method,
                "guard" + str(size) + "_gpu3",
                size,
            )
    copy(
        common / "downloaded-continuum-r2/comparison-continuum.json",
        destination / "common16_gpu7/comparison-continuum.json",
    )
    copy(
        common / "downloaded-r2/results-summary-r2/comparison.json",
        destination / "common16_gpu7/comparison.json",
    )
    output = dict(
        schema=1,
        repetitions=1,
        fixed_work=True,
        groups_are_separate_comparisons=True,
        packaged_entrypoint_gpu_remeasured=False,
        cells=records,
        guard_clock_correction="Historical guard used monotonic minus epoch arrival time. Zero activations do not prove waits below threshold; corrected implementation has separate queue/clock tests.",
    )
    (destination / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )
    mappings = {
        "kairos/prediction/joint.py": "continuous_geometric_20261111/posterior.py",
        "kairos/prediction/encoder.py": "continuous_geometric_20261111/runtime.py",
        "kairos/prediction/visible.py": "consistent_prediction_20261111/visible.py",
        "kairos/scheduling/joint.py": "continuous_geometric_20261111/scheduler.py",
        "kairos/scheduling/ownership.py": "reservation_lifecycle_20261111/settlement.py",
        "assets/joint/density.npz": "continuous_joint_20261111/results/selected-test-r1/density-head.npz",
    }
    for name in ("expanded.pt", "distribution.pt", "SELECTION_LOCK.json"):
        mappings["assets/joint/checkpoint/" + name] = (
            "coupled_service_20261110/results/encoder-joint-r1/encoder/" + name
        )
    for name in ("coupled.pt", "SELECTION_LOCK.json"):
        mappings["assets/joint/study/" + name] = (
            "coupled_service_20261110/results/encoder-joint-r1/" + name
        )
    for path in (ROOT / "assets/joint/encoder").iterdir():
        mappings[str(path.relative_to(ROOT))] = (
            "transformer_progress_20261110/encoder_stage/" + path.name
        )
    files = []
    for packaged, original in sorted(mappings.items()):
        path, reference = ROOT / packaged, source / original
        files.append(
            dict(
                path=packaged,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source=str(reference.relative_to(workspace)),
                source_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            )
        )
    provenance = dict(
        method="Kairos-CG1",
        entrypoint="kairos.scheduler.KairosScheduler",
        joint_predictor="p(O|H) p(R|O,H), continuous mixtures with residual zero mass",
        score="exp(E[log(t(O) / (s(O)/(s(O)+R))) | H, O>generated])",
        guard="bounded queue aging, epoch/monotonic clock correction",
        code_changes="Responsibility-based relocation and formatting. Encoder constructs from packaged config and strictly loads the complete trained state instead of loading redundant base weights; CPU equivalence checked.",
        assets_identical=True,
        weights_tracked_in_git=False,
        external_weight_files=[
            "checkpoint/expanded.pt",
            "checkpoint/distribution.pt",
            "study/coupled.pt",
            "density.npz",
        ],
        files=files,
        training_programs=1646,
        training_states=21114,
        validation_programs=169,
        validation_states=2023,
        evaluation_programs=198,
        evaluation_states=2566,
        evaluation_previously_observed=True,
        inference_requires_research_tree=False,
        packaged_entrypoint_gpu_remeasured=False,
    )
    if any(
        row["sha256"] != row["source_sha256"]
        for row in files
        if row["path"].startswith("assets/")
    ):
        raise ValueError("Packaged learned assets differ")
    (ROOT / "provenance/joint_release.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        json.dumps(dict(cells=len(records), groups=len({r["group"] for r in records})))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    export(parser.parse_args().workspace.resolve())
