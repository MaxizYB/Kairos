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


class KairosScheduler(
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
    """The complete Kairos policy and its native reference lifecycle."""

    _forecast_compact_log_installed = True
