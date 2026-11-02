"""runtime.compat.continuity bridge model."""

from dataclasses import dataclass
import math
from bisect import bisect_right


def common_history_pages(previous, current):
    count = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        count += 1
    return count


def continuation_queue_rank(
    restored,
    preempted,
    pinned,
    returning,
    common_pages,
    resident_pages,
    all_returns=False,
):
    if restored:
        return -1
    if preempted:
        return 0
    covered = common_pages > 0 and resident_pages >= common_pages
    return 1 if pinned or (returning and (all_returns or covered)) else 2


def conditional_samples(values, attained=0.0, count=8):
    xs = sorted((float(x) for x in values if math.isfinite(float(x)) and x > 0))
    return residual_strata(xs, attained, count)


def residual_strata(xs, attained=0.0, count=8):
    first = bisect_right(xs, attained)
    n = len(xs) - first
    if not n:
        scale = max(1.0, xs[-1] if xs else 128.0)
        return ((scale,) * count, "exhausted-tail-fallback")
    return (
        tuple(
            (
                xs[first + min(n - 1, int((i + 0.5) * n / count))] - attained
                for i in range(count)
            )
        ),
        "conditional-ecdf",
    )


def decode_samples(rows, stage, prompt, attained=0):
    xs, info = decode_population(rows, stage, prompt)
    samples, source = residual_strata(xs, attained)
    return (samples, dict(info, source=source))


def decode_population(rows, stage, prompt):
    complete = [r for r in rows if not r.get("censored", False)]
    group = [
        r
        for r in complete
        if min(3, r["stage"]) == min(3, stage)
        and r["prompt_tokens"] // 4096 == prompt // 4096
    ]
    if len(group) < 8:
        group = [r for r in complete if min(3, r["stage"]) == min(3, stage)]
    if len(group) < 8:
        group = complete
    return (
        tuple(
            sorted((float(r["output_tokens"]) for r in group if r["output_tokens"] > 0))
        ),
        dict(rows=len(group), excluded_censored=len(rows) - len(complete)),
    )


def completion_reserve(rows, page=16, quantile=0.9):
    result = {}
    for rid, prompt, generated, owned, samples in rows:
        ordered = sorted(samples)
        remaining = ordered[
            min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
        ]
        target = math.ceil((prompt + generated + max(0, remaining) - 1) / page)
        result[rid] = max(0, target - owned)
    return result


def native_completion_reserve(admission, incremental, native_reserved, growth):
    base_incremental = incremental - native_reserved
    return max(native_reserved, growth + max(0, admission - base_incremental))


def continuation_target(prompt, context_deltas, max_length, page=16):
    values = [
        max(0, int(value)) for value in context_deltas if math.isfinite(float(value))
    ]
    growth = sum(values) / len(values) if values else 0.0
    return math.ceil(min(max_length, prompt + growth) / page)


def continuation_growth_debt(
    pins, targets, ready_prompts, incoming, max_length, page=16
):
    result = {}
    for program, resident_pages in pins.items():
        if program == incoming:
            continue
        target = (
            math.ceil(min(max_length, ready_prompts[program]) / page)
            if program in ready_prompts
            else targets.get(program, resident_pages)
        )
        debt = max(0, target - resident_pages)
        if debt:
            result[program] = debt
    return result


def choose_fitting_follower(
    candidates,
    free_pages,
    live_growth_pages,
    running_count,
    sequence_limit,
    bypasses,
    limit=2,
    head_deficit_pages=None,
):
    if head_deficit_pages is not None and head_deficit_pages <= 0:
        return None
    if running_count >= sequence_limit or bypasses >= limit:
        return None
    for row in sorted(candidates, key=lambda r: (r["order"], r["request"])):
        if row["wait_s"] < row["service_s"]:
            continue
        required = row["target_pages"] + live_growth_pages
        if required <= free_pages:
            return dict(
                row, required_pages=required, live_growth_pages=live_growth_pages
            )
    return None


def oracle_program_service(turns, stage, beta, decode_step, restore_s, generated=0):
    if stage < 0 or stage >= len(turns):
        raise ValueError("Unknown oracle stage")
    decode = sum((int(t["output_tokens"]) for t in turns[stage:])) - generated
    prefill = 0.0
    for turn in turns[stage + 1 :]:
        length = int(turn["prompt_tokens"])
        prefix = min(max(0, int(turn["keep_prefix_tokens"])), max(0, length - 1))
        queries = length - prefix
        prefill += max(
            0.0,
            beta[0]
            + beta[1] * queries
            + beta[2] * queries * (prefix + length + 1) / 2
            + beta[3],
        )
    return dict(
        seconds=max(0.0, restore_s) + max(0, decode) * decode_step + prefill,
        remaining_output_tokens=max(0, decode),
        future_prefill_s=prefill,
        remaining_turns=len(turns) - stage,
        source="explicit-future-program-service-oracle",
    )


def program_service_population(rows, stage, prompt, beta, min_paths=8):
    paths = []
    for row in rows:
        if row["stage"] == 0:
            paths.append([])
        if not paths or row.get("censored", False):
            continue
        if row["stage"] != len(paths[-1]):
            raise ValueError("Incomplete Program prior")
        paths[-1].append(row)
    paths = [p for p in paths if p and p[-1].get("done", False)]
    if not paths:
        raise ValueError("No completed training Programs")
    supported = [
        s for s in range(stage + 1) if sum((len(p) > s for p in paths)) >= min_paths
    ]
    fitted = max(supported, default=0)
    eligible = [p for p in paths if len(p) > fitted]
    group = [
        p for p in eligible if p[fitted]["prompt_tokens"] // 4096 == prompt // 4096
    ]
    if len(group) >= min_paths:
        eligible = group
    result = []
    for path in eligible:
        current = path[fitted]
        shift = prompt - current["prompt_tokens"]
        prefill = 0.0
        for i in range(fitted + 1, len(path)):
            previous, pathrow = (path[i - 1], path[i])
            length = max(1, pathrow["prompt_tokens"] + shift)
            prefix = max(
                0,
                min(
                    length - 1,
                    previous["prompt_tokens"] + previous["output_tokens"] + shift,
                ),
            )
            queries = length - prefix
            prefill += max(
                0.0,
                beta[0]
                + beta[1] * queries
                + beta[2] * queries * (prefix + length + 1) / 2
                + beta[3],
            )
        result.append(
            (
                current["output_tokens"],
                sum((r["output_tokens"] for r in path[fitted + 1 :])),
                prefill,
                len(path) - fitted,
            )
        )
    return (
        tuple(result),
        dict(
            paths=len(result),
            stage=stage,
            fitted_stage=fitted,
            stage_backoff=fitted != stage,
            source="disjoint-completed-Program-service-mixture",
        ),
    )


def program_service_quote(population, attained, decode_step, restore_s):
    survivors = [r for r in population if r[0] > attained]
    tail_fallback = not survivors
    if tail_fallback:
        scale = max(1, max((r[0] for r in population)))
        survivors = [(attained + scale, r[1], r[2], r[3]) for r in population]
    seconds = restore_s + sum(
        ((r[0] - attained + r[1]) * decode_step + r[2] for r in survivors)
    ) / len(survivors)
    return dict(
        seconds=seconds,
        conditional_paths=len(survivors),
        tail_fallback=tail_fallback,
        remaining_turns_mean=sum((r[3] for r in survivors)) / len(survivors),
    )


def prefix_restore(length, gpu_tokens, cpu_chunks, beta, load_fixed, load_per_token):
    end = min(gpu_tokens, max(0, length - 1))
    loaded = 0
    while end // 256 in cpu_chunks and (end // 256 + 1) * 256 < length:
        end = (end // 256 + 1) * 256
        loaded += 256
    queries = max(0, length - end)
    compute = (
        max(
            0.0,
            beta[0]
            + beta[1] * queries
            + beta[2] * queries * (end + length + 1) / 2
            + beta[3],
        )
        if queries
        else 0.0
    )
    load = max(0.0, load_fixed + load_per_token * loaded) if loaded else 0.0
    return dict(
        seconds=compute + load,
        compute_s=compute,
        load_s=load,
        queries=queries,
        loaded_tokens=loaded,
        end=end,
    )


@dataclass(frozen=True)
class Release:
    program: str
    pages: int
    loss_s: float
    expires_s: float


def pressure_choice(candidates, deficit, running_releases, waiters=1):
    if deficit <= 0:
        return dict(action="none", selected=(), gain_s=0.0, loss_s=0.0)
    selected = []
    freed = 0
    loss = 0.0
    for c in sorted(candidates, key=lambda c: (c.loss_s / max(1, c.pages), c.program)):
        if c.pages <= 0:
            continue
        selected.append(c)
        freed += c.pages
        loss += c.loss_s
        if freed >= deficit:
            break
    if freed < deficit:
        return dict(
            action="wait",
            selected=(),
            gain_s=0.0,
            loss_s=loss,
            reason="insufficient-legal-idle-pages",
        )
    timelines = []
    samples = max((len(times) for _, times in running_releases), default=1)
    for i in range(samples):
        events = [(c.expires_s, c.pages) for c in candidates]
        events += [
            (times[i % len(times)], pages) for pages, times in running_releases if times
        ]
        released = 0
        for when, pages in sorted(events):
            released += pages
            if released >= deficit:
                timelines.append(max(0.0, when))
                break
    gain = sum(timelines) / len(timelines) if timelines else 0.0
    margin = max(0.005, 0.1 * gain)
    action = "release" if gain > loss + margin else "wait"
    return dict(
        action=action,
        selected=tuple(selected) if action == "release" else (),
        gain_s=gain,
        loss_s=loss,
        margin_s=margin,
        freed_pages=freed,
        reason="next-admission-delay-proxy",
        waiters=waiters,
    )
