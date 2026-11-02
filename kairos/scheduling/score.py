"""scheduling.score."""

import math
from kairos.scheduling.cost import (
    distribution_benefit,
    posterior_mean,
    recovery_seconds,
)


def service_rank(standard, actual, atoms, mass, mode):
    if any((not math.isfinite(x) or x <= 0 for x in (standard, actual))):
        raise ValueError("Positive finite standard and actual service required")
    mean = posterior_mean(atoms, mass)
    if mode == "mean":
        progress = standard / (standard + mean)
    elif mode == "distribution":
        progress = distribution_benefit(standard, atoms, mass)
    else:
        raise ValueError("Unknown posterior consumption rule")
    return dict(
        score=actual / progress,
        progress=progress,
        residual_mean_s=mean,
        standard_s=standard,
        actual_s=actual,
        mode=mode,
    )


def current_service(
    *, prompt, output, computed, gpu_pages, host, profile, page, step, restored=False
):
    if prompt < 1 or output < 1 or computed < 0 or (step <= 0) or (page < 1):
        raise ValueError("Invalid current call")
    hit_tokens = min(prompt, max(computed, gpu_pages * page))
    layout = dict(host, blocks=list(range(hit_tokens // page)), tokens=prompt)
    restore = (
        0.0
        if restored or computed >= prompt
        else recovery_seconds(layout, (), profile, page)
    )
    return (restore + output * step, restore)
