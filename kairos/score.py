"""Kairos score implementation."""
import math

from kairos.cost import distribution_benefit, posterior_mean, recovery_seconds


def service_rank(standard, actual, atoms, mass, mode):
    """Return t/progress (seconds); smaller is better.

    This is a local progress index, NOT completion probability or predicted
    JCT. With no suffix it is current SJF; with t=s and mean consumption it
    becomes s+E[R+]. No theorem is asserted for self-suspending GPU batches.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (standard, actual)):
        raise ValueError('Positive finite standard and actual service required')
    mean = posterior_mean(atoms, mass)
    if mode == 'mean':
        progress = standard / (standard + mean)
    elif mode == 'distribution':
        progress = distribution_benefit(standard, atoms, mass)
    else:
        raise ValueError('Unknown Kairos posterior consumption')
    return dict(score=actual / progress, progress=progress, residual_mean_s=mean,
                standard_s=standard, actual_s=actual, mode=mode)


def current_service(*, prompt, output, computed, gpu_pages, host, profile, page, step,
                    restored=False):
    """Price existing GPU prefix + committed CPU interval + recompute.

    CPU rows are the same advisory v21 current-prefix registry used by its
    victim planner; writing chunks are not counted. This is not a promise
    that an asynchronous restore will see the identical future layout.
    """
    if prompt < 1 or output < 1 or computed < 0 or step <= 0 or page < 1:
        raise ValueError('Invalid current call')
    hit_tokens = min(prompt, max(computed, gpu_pages * page))
    layout = dict(host, blocks=list(range(hit_tokens // page)), tokens=prompt)
    # Completed async receive is authoritative even before native promotes
    # WAITING_FOR_REMOTE_KVS and materializes its computed-token count.
    restore = 0. if restored or computed >= prompt else recovery_seconds(
        layout, (), profile, page)
    return restore + output * step, restore
