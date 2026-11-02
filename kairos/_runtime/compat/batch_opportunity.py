"""runtime.compat.batch opportunity."""

import math


def backed_suffix(total_pages, required_pages, private_pages, cpu_chunks, chunks):
    if required_pages <= 0 or required_pages >= total_pages:
        return None
    retained = (total_pages - required_pages) // 16 * 16
    released = total_pages - retained
    if released > private_pages or retained <= 0:
        return None
    full_end = total_pages // 16
    if full_end > chunks or any(
        (i not in cpu_chunks for i in range(retained // 16, full_end))
    ):
        return None
    return dict(
        release_pages=released,
        retained_pages=retained,
        cpu_restore_tokens=(full_end - retained // 16) * 256,
        tail_recompute_tokens=(total_pages - full_end * 16) * 16,
    )


def local_gain(
    step1,
    step2,
    ttl_remaining,
    active_cap_steps,
    incoming_cap_steps,
    incoming_delay_s,
    return_cost_s,
):
    values = (step1, step2, ttl_remaining, incoming_delay_s, return_cost_s)
    if any((not math.isfinite(v) or v < 0 for v in values)) or min(step1, step2) <= 0:
        return None
    window = max(0.0, ttl_remaining - incoming_delay_s)
    paired_steps = min(
        max(0, active_cap_steps), max(0, incoming_cap_steps), int(window / step2)
    )
    saved = paired_steps * max(0.0, 2 * step1 - step2)
    return dict(
        paired_steps_upper_bound=paired_steps,
        opportunity_window_s=window,
        saved_gpu_work_upper_bound_s=saved,
        estimated_return_cost_s=return_cost_s,
        accept=paired_steps >= 2 and saved > 1.5 * return_cost_s + step1,
    )
