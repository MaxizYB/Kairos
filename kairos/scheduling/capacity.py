"""scheduling.capacity."""

from kairos.scheduling.contracts import Admission, admission as base_admission


def admission(
    owner,
    target,
    reserved,
    occupied,
    optional,
    priorities,
    limit,
    max_running,
    *,
    claim_pages=0,
    claim_pending=False,
    locked=(),
):
    if owner in locked and owner not in occupied:
        return Admission(
            False,
            (),
            "dma-cleanup",
            max(0, target - reserved.get(owner, 0)),
            max(0, limit - sum(reserved.values())),
        )
    return base_admission(
        owner,
        target,
        reserved,
        occupied,
        optional,
        priorities,
        limit,
        max_running,
        claim_pages=claim_pages,
        claim_pending=claim_pending,
        locked=locked,
    )
