"""Reproduce the method figures with CPU-only, held-out evaluations.

Run from the checkout: python scripts/build_figures.py
External raw records are needed only to refresh the compact workload snapshot:
python scripts/build_figures.py --refresh-workloads --source-root ..
No fitted parameters, scheduling decisions, or runtime artifacts are changed.
"""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kairos.metrics import crps, quantile
from kairos.prediction.current import OutputForest, condition_output
from kairos.prediction.suffix import SuffixForest

FIG = ROOT / "figures"
# Adapted from the blue / apricot / green grouping in CacheBlend Fig. 1
# and the pastel module grouping in Continuum Fig. 7. See sources.json.
BLUE = "#557AAB"
ORANGE = "#C08A5B"
GREEN = "#618C78"
PURPLE = "#8A7B9D"
INK = "#253647"
PALE = {BLUE: "#E9EFF6", ORANGE: "#F5ECE3", GREEN: "#EAF1EC", PURPLE: "#F0EDF4"}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "lines.linewidth": 2,
            "axes.axisbelow": True,
        }
    )


def save(fig, name):
    path = FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            path.with_suffix("." + ext), dpi=220, bbox_inches="tight", pad_inches=0.10
        )
    plt.close(fig)


def grid(ax):
    ax.grid(axis="y", color=BLUE, alpha=0.12, linewidth=0.7)


def quantiles(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": len(x),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "maximum": float(max(x)),
        "mean": float(np.mean(x)),
    }


def ccdf(x):
    x = np.sort(x)
    return x, (len(x) - np.arange(len(x))) / len(x)


def refresh_workloads(source):
    """Export aggregate observations, not text, identifiers, or scaled replay data."""
    cop_path = (
        source
        / "Phase_continuum/information_prediction_20261020/results/copilot_adaptation_r1/copilot_states.json.gz"
    )
    trace_path = (
        source
        / "Phase_continuum/production_workload_20261014/data/raw/syfi_coding_trace.jsonl.gz"
    )
    cop = [p for p in read(cop_path)["programs"] if p["split"] == "test"]
    calls = [sum(not r["after"] for r in p["states"]) for p in cop]
    tools, prompts, outputs, reuse = [], [], [], []
    session = {}
    target_calls = []
    users = set()
    n = 0
    with gzip.open(trace_path, "rt") as stream:
        for line in stream:
            r = json.loads(line)
            n += 1
            sid = r["session_id"]
            users.add(r["user"])
            human = any(
                e.get("event_type") == "user_message"
                for e in r.get("timing_events", [])
            )
            if human or sid not in session:
                if sid in session:
                    target_calls.append(session[sid])
                session[sid] = 0
            session[sid] += 1
            p = r.get("input_tokens_total") or 0
            o = r.get("output_tokens") or 0
            if p > 0:
                prompts.append(p)
                reuse.append(min(1, max(0, (r.get("prefix_tokens") or 0) / p)))
            if o > 0:
                outputs.append(o)
            for tool in r.get("tools") or []:
                latency = tool.get("tool_wall_latency_ms")
                if latency is not None and latency >= 0:
                    tools.append(latency / 1000)
    target_calls.extend(session.values())
    pos = np.sort([x for x in tools if x > 0])
    # Exact empirical quantiles on a dense probability grid; no tail truncation.
    probs = np.unique(np.r_[np.linspace(0, 0.99, 401), np.linspace(0.99, 1, 401)])
    rank = np.minimum(len(pos) - 1, np.floor(probs * (len(pos) - 1)).astype(int))
    top = max(1, int(np.ceil(0.05 * len(tools))))
    result = {
        "copilot": {
            "calls": calls,
            "summary": quantiles(calls),
            "single_call_fraction": float(np.mean(np.asarray(calls) == 1)),
            "definition": "Held-out bounded user-turns, not verified autonomous completions; source extraction excludes gaps >30 s.",
        },
        "tracelab": {
            "rounds": n,
            "users": len(users),
            "sessions": len(session),
            "inferred_targets": len(target_calls),
            "calls_histogram": sorted(Counter(target_calls).items()),
            "positive_tool_ccdf": {
                "x": pos[rank].tolist(),
                "y": ((len(pos) - rank) / len(pos)).tolist(),
            },
            "tools": quantiles(tools),
            "positive_tools": len(pos),
            "slowest5_time_share": float(sum(sorted(tools)[-top:]) / sum(tools)),
            "prompt": quantiles(prompts),
            "output": quantiles(outputs),
            "provider_prefix_fraction": quantiles(reuse),
            "definition": "Raw unscaled volunteer records; target boundaries inferred from user_message; tool_wall_latency_ms is per recorded tool, not inter-call gap; provider prefix is realized accounting, not token-identity proof.",
        },
        "sources": [
            {"path": str(p.relative_to(source)), "sha256": digest(p)}
            for p in (cop_path, trace_path)
        ],
    }
    write(FIG / "data/workload_observations.json.gz", result)
    print("Workload snapshot refreshed", flush=True)


def swe_programs(data, split="blind"):
    return [
        p
        for p in data["programs"]
        if p["split"] == split and p["complete"] and p["within_16k"]
    ]


def before(p):
    return sorted([r for r in p["states"] if not r["after"]], key=lambda r: r["stage"])


def workload_figures(data, extra):
    programs = swe_programs(data)
    calls = [len(before(p)) for p in programs]
    gaps = [
        r["label_gap_s"]
        for p in programs
        if p["timing_valid"]
        for r in before(p)[:-1]
        if r["label_gap_s"] > 0
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    for x, label, color, marker in [
        (calls, "SWE: completed Program", BLUE, "o"),
        (extra["copilot"]["calls"], "Copilot: bounded user-turn", GREEN, "s"),
    ]:
        x = np.asarray(x)
        xx = np.arange(1, max(x) + 1)
        yy = np.asarray([np.mean(x >= v) for v in xx])
        ax.step(xx, yy, where="post", color=color, label=label)
        ax.plot(xx[::2], yy[::2], ls="none", marker=marker, ms=4, color=color)
    ax.set(
        xlabel="LLM calls in one recorded unit",
        ylabel="Records with at least x calls",
        xlim=(0.5, 25),
        ylim=(0, 1.03),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    grid(ax)
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.55,
        "Different endpoint definitions\nSWE: n = 36   |   Copilot: n = 1,241",
        transform=ax.transAxes,
        ha="right",
        fontsize=9.5,
    )
    save(fig, "workload/call_span")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), layout="constrained")
    top = max(1, int(np.ceil(0.05 * len(gaps))))
    slow_share = sum(sorted(gaps)[-top:]) / sum(gaps)
    trace = extra["tracelab"]
    series = [
        (*ccdf(gaps), "SWE: inter-call gap", BLUE, quantiles(gaps), slow_share),
        (
            trace["positive_tool_ccdf"]["x"],
            np.asarray(trace["positive_tool_ccdf"]["y"])
            * trace["positive_tools"]
            / trace["tools"]["n"],
            "TraceLab: per-tool latency",
            GREEN,
            trace["tools"],
            trace["slowest5_time_share"],
        ),
    ]
    for ax, (x, y, title, color, q, share) in zip(axes, series):
        ax.step(x, y, where="post", color=color)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Observed duration (s)",
            ylabel="Fraction exceeding duration",
            title=title,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{100 * v:g}%"))
        grid(ax)
        ax.text(
            0.04,
            0.08,
            f"Median {q['median']:.3g} s   |   P95 {q['p95']:.3g} s\nSlowest ~5% account for {share:.1%} of time",
            transform=ax.transAxes,
            fontsize=9.4,
        )
    save(fig, "workload/tool_wait_tail")

    warm = [r for p in programs for r in before(p) if r["stage"] > 0]
    frac = np.asarray([r["keep"] / r["prompt"] for r in warm])
    # Observed progress is an offline alignment variable, never a predictor input.
    knots = np.linspace(0, 1, 11)
    paths = np.asarray(
        [
            np.interp(
                knots,
                np.linspace(0, 1, len(before(p))),
                [r["prompt"] for r in before(p)],
            )
            for p in programs
            if len(before(p)) > 1
        ]
    )
    lo, mid, hi = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
    ax = axes[0]
    ax.fill_between(
        knots,
        lo / 1000,
        hi / 1000,
        color=BLUE,
        alpha=0.15,
        label="10th–90th percentile",
    )
    ax.plot(knots, mid / 1000, color=BLUE, marker="o", ms=4, label="Median Program")
    ax.set(
        xlabel="Relative observed call position",
        ylabel="Arrived prompt (k tokens)",
        title="Context grows across calls",
        ylim=(0, 16.5),
    )
    grid(ax)
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    x = np.sort(frac)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=GREEN)
    ax.set(
        xlabel="Token-identical prefix / current prompt",
        ylabel="Fraction of returning calls",
        title="History is a large part of the input",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    grid(ax)
    ax.text(
        0.04,
        0.8,
        f"Median reusable fraction: {np.median(frac):.1%}\n{len(warm)} returning calls",
        transform=ax.transAxes,
        fontsize=9.7,
    )
    save(fig, "workload/context_reuse")

    # Positive conditioning would hide terminal mass; retain all surviving calls.
    by_stage = defaultdict(list)
    for p in programs:
        for r in before(p):
            by_stage[r["stage"]].append(r["target"])
    stages = [k for k, v in sorted(by_stage.items()) if len(v) >= 8]
    values = np.asarray([np.quantile(by_stage[k], [0.1, 0.5, 0.9]) for k in stages])
    fig, ax = plt.subplots(figsize=(6.5, 3.65), layout="constrained")
    ax.fill_between(
        stages,
        values[:, 0],
        values[:, 2],
        alpha=0.16,
        color=PURPLE,
        label="10th–90th percentile",
    )
    ax.plot(stages, values[:, 1], color=PURPLE, marker="o", ms=4, label="Median")
    for j in (0, len(stages) // 2, len(stages) - 1):
        ax.annotate(
            f"n={len(by_stage[stages[j]])}",
            (stages[j], values[j, 2]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set(
        xlabel="Previously completed calls",
        ylabel="Service after current call (reference s)",
        ylim=(0, max(values[:, 2]) * 1.2),
    )
    grid(ax)
    ax.legend(loc="upper right")
    save(fig, "workload/residual_by_stage")
    return {
        "swe": {
            "programs": len(programs),
            "states": sum(calls),
            "calls": quantiles(calls),
            "timing_programs": sum(bool(p["timing_valid"]) for p in programs),
            "gaps": quantiles(gaps),
            "slowest5_gap_time_share": slow_share,
            "warm_prefix_fraction": quantiles(frac),
            "context_first_median": float(np.median(paths[:, 0])),
            "context_last_median": float(np.median(paths[:, -1])),
            "residual_by_stage": {
                str(k): dict(
                    n=len(v),
                    **{f"q{q}": float(np.quantile(v, q / 100)) for q in [10, 50, 90]},
                )
                for k, v in by_stage.items()
            },
        },
        "copilot": {k: v for k, v in extra["copilot"].items() if k != "calls"},
        "tracelab": {
            k: v
            for k, v in trace.items()
            if k not in ["positive_tool_ccdf", "calls_histogram"]
        },
    }


def feature_associations(data):
    from scipy.stats import rankdata

    rows = [before(p)[0] for p in swe_programs(data)]
    y = np.asarray([r["target"] for r in rows])
    rng = np.random.default_rng(20261102)
    indices = rng.integers(0, len(rows), (2000, len(rows)))

    def corr(x, y):
        a = rankdata(x, axis=-1)
        b = rankdata(y, axis=-1)
        a -= a.mean(axis=-1, keepdims=True)
        b -= b.mean(axis=-1, keepdims=True)
        denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
        return np.divide(
            (a * b).sum(axis=-1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

    result = []
    for field in [
        "prompt_log",
        "issue_chars_log",
        "issue_lines_log",
        "issue_code_fraction",
        "issue_error_words_log",
    ]:
        j = data["feature_names"].index(field)
        x = np.asarray([r["features"][j] for r in rows])
        result.append(
            {
                "feature": field,
                "rho": float(corr(x, y)),
                "bootstrap95": np.quantile(
                    corr(x[indices], y[indices]), [0.025, 0.975]
                ).tolist(),
            }
        )
    return {"n": len(rows), "resamples": 2000, "seed": 20261102, "rows": result}


def macro(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row[key])
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def evaluate_forests(data):
    suffix = SuffixForest(ROOT / "assets/forest.json", None)
    output = OutputForest(ROOT / "assets/current_output.json", None)
    result = {}
    metrics = {}
    for name, split in [("SWE-36", "blind"), ("SWE-14", "development")]:
        programs = swe_programs(data, split)
        assert not {p["id"] for p in programs} & (
            suffix.training_ids | output.training_ids
        )
        result[name] = {}
        metrics[name] = {}
        for target, model in [("output", output), ("suffix", suffix)]:
            rows = []
            for p in programs:
                for state in before(p):
                    post = model.predict_features(
                        state["old_features"] + state["features"]
                    )
                    if target == "output":
                        call = condition_output(
                            post, cap=16384 - state["prompt"], marginal=output.marginal
                        )
                        post = {
                            "pred": call.total_tokens,
                            "atoms": call.atoms,
                            "mass": call.mass,
                        }
                    truth = state["output"] if target == "output" else state["target"]
                    row = {
                        "id": p["id"],
                        "stage": state["stage"],
                        "truth": truth,
                        "mean": post["pred"],
                        "atoms": list(post["atoms"]),
                        "mass": list(post["mass"]),
                        "mae": abs(post["pred"] - truth),
                        "crps": crps(post["atoms"], post["mass"], truth),
                        "within2": float(truth / 2 <= post["pred"] <= truth * 2)
                        if truth > 0
                        else None,
                    }
                    row["low"] = quantile(post["atoms"], post["mass"], 0.05)
                    row["high"] = quantile(post["atoms"], post["mass"], 0.95)
                    row["coverage"] = float(row["low"] <= truth <= row["high"])
                    row["width"] = row["high"] - row["low"]
                    rows.append(row)
            result[name][target] = rows
            m = dict(
                programs=len(programs),
                states=len(rows),
                unit="tokens" if target == "output" else "reference s",
                **{
                    key: macro(rows, key)
                    for key in ["mae", "crps", "coverage", "width"]
                },
                covered_states=sum(r["coverage"] for r in rows),
                state_coverage=float(np.mean([r["coverage"] for r in rows])),
                within_factor2_positive=macro(
                    [r for r in rows if r["truth"] > 0], "within2"
                ),
            )
            m["calibration"] = []
            for nominal in np.arange(0.1, 0.96, 0.05):
                for r in rows:
                    lo = quantile(r["atoms"], r["mass"], (1 - nominal) / 2)
                    hi = quantile(r["atoms"], r["mass"], (1 + nominal) / 2)
                    r["interval_hit"] = float(lo <= r["truth"] <= hi)
                m["calibration"].append([float(nominal), macro(rows, "interval_hit")])
            for row in rows:
                row.pop("interval_hit")
            if target == "output":
                m["service_mae_reference_s"] = (
                    m["mae"] * data["profile"]["decode_step_by_seqs"]["1"]
                )
            metrics[name][target] = m
    return result, metrics


def predictor_figures(data, forecasts, metrics):
    groups = forecasts["SWE-36"]
    titles = {"output": "Current output length", "suffix": "Service after current call"}
    units = {"output": "tokens", "suffix": "reference s"}
    colors = {"output": BLUE, "suffix": GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), layout="constrained")
    for ax, target in zip(axes, ["output", "suffix"]):
        rows = groups[target]
        color = colors[target]
        t = np.asarray([r["truth"] for r in rows])
        p = np.asarray([r["mean"] for r in rows])
        ax.scatter(t, p, s=16, c=color, alpha=0.42, edgecolors="none", rasterized=True)
        if target == "output":
            bound = max(max(t), max(p)) * 1.15
            ax.set(xscale="log", yscale="log", xlim=(1, bound), ylim=(1, bound))
            lo = 1
        else:
            bound = max(max(t), max(p)) * 1.08
            lo = 0
            ax.set(xlim=(-1, bound), ylim=(-1, bound))
        ax.plot(
            [lo, bound],
            [lo, bound],
            ls="--",
            color=ORANGE,
            lw=1.5,
            label="Exact agreement",
        )
        ax.set(
            xlabel=f"Observed value ({units[target]})",
            ylabel=f"Predicted mean ({units[target]})",
            title=titles[target],
        )
        m = metrics["SWE-36"][target]
        ax.text(
            0.04,
            0.95,
            f"MAE {m['mae']:.1f} {units[target]}\nWithin 2×: {m['within_factor2_positive']:.1%}"
            + (" of positive labels" if target == "suffix" else ""),
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
        )
        ax.legend(loc="lower right", fontsize=9)
        grid(ax)
    save(fig, "prediction/observed_vs_predicted")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    for ax, target in zip(axes, ["output", "suffix"]):
        m = metrics["SWE-36"][target]
        x, y = np.asarray(m["calibration"]).T
        ax.plot([0, 1], [0, 1], ls="--", lw=1.3, color=ORANGE, label="Nominal coverage")
        ax.plot(
            x, y, color=colors[target], marker="o", ms=3.5, label="Observed coverage"
        )
        ax.plot(
            0.9, m["coverage"], "o", color=colors[target], ms=8, mec="white", mew=1.2
        )
        ax.set(
            xlabel="Nominal central interval",
            ylabel="Observed coverage",
            title=titles[target],
            xlim=(0, 1),
            ylim=(0, 1.04),
            aspect="equal",
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        grid(ax)
        ax.text(
            0.03,
            0.96,
            f"90% interval: {m['coverage']:.1%} coverage\nMean width: {m['width']:.1f} {units[target]}",
            transform=ax.transAxes,
            va="top",
            fontsize=9.1,
        )
        ax.legend(loc="lower right", fontsize=8.8)
    save(fig, "prediction/interval_calibration")

    # Chosen by workload structure only: lower median call count, ID tie-break.
    programs = sorted(swe_programs(data), key=lambda p: (len(before(p)), p["id"]))
    chosen = programs[(len(programs) - 1) // 2]
    ident = chosen["id"]
    case = {t: [r for r in groups[t] if r["id"] == ident] for t in groups}
    stage = (len(case["suffix"]) - 1) // 2
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), layout="constrained")
    for ax, target in zip(axes, ["output", "suffix"]):
        row = case[target][stage]
        atoms = np.asarray(row["atoms"])
        mass = np.asarray(row["mass"])
        color = colors[target]
        # CDF is exact and avoids arbitrary density smoothing for discrete forests.
        ax.axvspan(
            row["low"],
            row["high"],
            color=color,
            alpha=0.13,
            label="Central 90% interval",
        )
        ax.step(
            np.r_[0, atoms],
            np.r_[0, np.cumsum(mass)],
            where="post",
            color=color,
            label="Predicted CDF",
        )
        ax.axvline(row["truth"], color=ORANGE, ls="--", lw=1.8, label="Observed value")
        xmax = max(quantile(row["atoms"], row["mass"], 0.99), row["truth"]) * 1.05
        ax.set(
            xlim=(0, xmax),
            ylim=(0, 1.03),
            xlabel=f"{titles[target]} ({units[target]})",
            ylabel="Cumulative probability",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        grid(ax)
        ax.text(
            0.98,
            0.3,
            f"Observed: {row['truth']:.1f}\nPredicted mean: {row['mean']:.1f}\n90% interval: [{row['low']:.1f}, {row['high']:.1f}]",
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1, -0.025), fontsize=8.6)
    save(fig, "prediction/call_distribution_example")

    rows = case["suffix"]
    x = np.asarray([r["stage"] + 1 for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 3.7), layout="constrained")
    ax.fill_between(
        x,
        [r["low"] for r in rows],
        [r["high"] for r in rows],
        color=GREEN,
        alpha=0.16,
        label="Predicted central 90% interval",
    )
    ax.plot(
        x,
        [r["mean"] for r in rows],
        color=GREEN,
        marker="o",
        ms=4,
        label="Predicted mean",
    )
    ax.plot(
        x,
        [r["truth"] for r in rows],
        color=ORANGE,
        marker="D",
        ms=4,
        label="Observed residual service",
    )
    ax.axvline(stage + 1, color=PURPLE, ls=":", lw=1.2)
    ax.set(
        xlabel="Current call index (one Program)",
        ylabel="Service after current call (reference s)",
        xticks=x,
        ylim=(0, None),
    )
    grid(ax)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "prediction/program_evolution")
    return {
        "program_id": ident,
        "selection": "Lower median of (call count, Program ID) across SWE-36; call example is lower middle call. No selection by prediction error.",
        "calls": len(rows),
        "current_call": stage + 1,
        "snapshots": {
            t: {
                k: v
                for k, v in case[t][stage].items()
                if k not in ["atoms", "mass", "interval_hit"]
            }
            for t in case
        },
    }


def performance_figures():
    cells = read(ROOT / "results/benchmark/summary.json")["cells"]
    table = {(c["workload"], c["method"]): c["summary"]["jct_mean_s"] for c in cells}
    methods = ["native", "continuum", "kairos"]
    labels = ["vLLM + LMCache", "Continuum", "Kairos"]
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    y = [table[("dense", m)] for m in methods]
    axes[0].bar(np.arange(3), y, color=colors, width=0.58)
    for i, v in enumerate(y):
        axes[0].text(i, v + 2, f"{v:.2f}", ha="center", fontsize=10)
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=labels,
        ylabel="Mean Program JCT (s)",
        title="Dense burst",
        ylim=(0, 150),
    )
    for m, label, color, marker in zip(methods, labels, colors, ["s", "^", "o"]):
        y = [table[(f"poisson_jps{j}", m)] for j in [1, 2]]
        axes[1].plot([1, 2], y, color=color, marker=marker, ms=6, label=label)
        for j, v in zip([1, 2], y):
            axes[1].annotate(
                f"{v:.2f}",
                (j, v),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )
    axes[1].set(
        xticks=[1, 2],
        xlim=(0.8, 2.2),
        ylim=(65, 145),
        xlabel="Program arrivals per second (JPS)",
        ylabel="Mean Program JCT (s)",
        title="Poisson arrivals",
    )
    axes[1].legend(loc="upper right", fontsize=8.8)
    for ax in axes:
        grid(ax)
    save(fig, "evaluation/mean_jct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-workloads", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    if args.refresh_workloads:
        refresh_workloads(args.source_root.resolve())
    configure()
    data = read(ROOT / "data/training/causal_states.json.gz")
    extra = read(FIG / "data/workload_observations.json.gz")
    workload = workload_figures(data, extra)
    forecasts, metrics = evaluate_forests(data)
    case = predictor_figures(data, forecasts, metrics)
    performance_figures()
    write(FIG / "data/predictor_predictions.json.gz", forecasts)
    write(
        FIG / "data/analysis.json",
        {
            "workload": workload,
            "feature_associations": feature_associations(data),
            "predictors": metrics,
            "example": case,
            "aggregation": "Macro: equal weight per Program, mean over that Program's pre-call states; raw counts and state-micro coverage also provided.",
            "profile_units": "Frozen reference service seconds, not measured 8B end-to-end latency.",
            "model_hashes": {
                p.name: digest(p)
                for p in [
                    ROOT / "assets/forest.json",
                    ROOT / "assets/current_output.json",
                ]
            },
            "states_sha256": digest(ROOT / "data/training/causal_states.json.gz"),
        },
    )
    write(
        FIG / "data/sources.json",
        {
            "visual_references": [
                {
                    "paper": "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live",
                    "url": "https://arxiv.org/abs/2511.02230",
                    "figures": "Fig. 5 (tail / accumulated time), Fig. 7 (module grouping), Fig. 11 (arrival-rate curves)",
                    "pages": [4, 7, 10],
                },
                {
                    "paper": "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion",
                    "url": "https://arxiv.org/abs/2405.16444",
                    "figures": "Fig. 1: blue / apricot / green cache grouping and schematic state transformations",
                    "pages": [2],
                },
            ],
            "style": "Muted derivatives of the referenced blue/apricot/green groups; white background, direct labels, vector text. Original vector drawings; no reproduced paper graphics or external icon license dependency.",
            "workload_sources": extra["sources"],
            "no_internal_version_comparisons": True,
            "evaluation": "Frozen released models only; held-out identity checks; true labels used after prediction for offline scoring only; no oracle scheduling experiments.",
            "diagrams": "Conceptual, not measured traces. All shapes and text in method/Kairos_Method_Figures.pptx are editable.",
        },
    )
    print(
        json.dumps(
            {"predictors": metrics, "example": case, "workload": workload}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
