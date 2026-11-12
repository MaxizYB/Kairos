"""Causal forecasts, progress ordering, and revocable KV continuity.

Each mixin owns a decision boundary. Admission commits capacity, snapshots
quote ready/running/idle owners, holding prices completed calls, and numerical
caches reuse probability laws. Native allocation and DMA remain in the adapter.
"""

from kairos.continuity.holding import EconomicRetention
from kairos.observability.compact import CompactDistributions
from kairos.prediction.survival import OutputSurvival
from kairos.scheduling.admission import CapacityControl
from kairos.scheduling.cache import ServiceCache
from kairos.scheduling.continuity_quotes import ContinuityQuotes
from kairos.scheduling.forecast import ForecastControl
from kairos.scheduling.integral import RunningIntegral
from kairos.scheduling.snapshot import UnifiedSnapshot
from kairos.scheduling.joint import JointForecasts
from kairos.scheduling.ownership import ReservationSettlement


class _SchedulingCore(
    CapacityControl,
    UnifiedSnapshot,
    EconomicRetention,
    CompactDistributions,
    RunningIntegral,
    OutputSurvival,
    ContinuityQuotes,
    ServiceCache,
    ForecastControl,
):
    """Auxiliary forests, capacity transactions, TTL and native lifecycle."""

    _forecast_compact_log_installed = True


class KairosScheduler(JointForecasts, ReservationSettlement, _SchedulingCore):
    """Continuous joint prediction, geometric J, and bounded queue aging."""
