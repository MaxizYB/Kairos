"""runtime.compat.layered policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    admission: str | None
    host: str
    gpu: bool
    chunk_admission: bool = False
    device_leaf: bool = False
    leaf_reuse_nodes: bool = False
    recovery_gpu: bool = False
    recovery_weight: str = "time"
    prefill_budget: int | None = None
    recovery_route: str = "segmented"
    segment_capacity: bool = False
    compact_recovery: bool = False
    plan_refinement: int = 0
    cost_route: bool = False
    executable_route: bool = False
    executable_gpu: bool = False
    executable_cpu: bool = False
    event_gpu_history: bool = False
    control_stream: bool = False
    control_readonly_stream: bool = False
    restore_fence: str = "before-forward"


LAYERS = {
    "layered-ea-duplicate": Layer("conditioned", "hicache", False),
    "layered-ea-duplicate-gpu": Layer("conditioned", "hicache", True),
    "layered-ea-duplicate-leaf": Layer(
        "conditioned", "hicache", False, device_leaf=True
    ),
    "layered-ea-duplicate-leaf-cached": Layer(
        "conditioned", "hicache", False, device_leaf=True, leaf_reuse_nodes=True
    ),
    "layered-ea-gpu": Layer("conditioned", "lru", True),
    "layered-ea-joint": Layer("conditioned", "joint", True, True),
    "gapped-layered-ea-joint": Layer("conditioned", "joint", True, True),
    "gapped-layered-ea-continuation": Layer("conditioned", "continuation", True, True),
    "gapped-layered-ea-continuation-repair": Layer(
        "conditioned", "continuation-repair", True, True
    ),
    "gapped-layered-ea-recovery": Layer(
        "conditioned", "continuation-repair", True, True, recovery_gpu=True
    ),
    "gapped-layered-ea-recovery-gpu": Layer(
        "conditioned", "joint", True, True, recovery_gpu=True
    ),
    "gapped-layered-ea-recovery-gpu-wide": Layer(
        "conditioned", "joint", True, True, recovery_gpu=True, prefill_budget=2048
    ),
    "gapped-layered-ea-recovery-gpu-hybrid": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
    ),
    "gapped-layered-ea-joint-hybrid": Layer(
        "conditioned", "joint", True, True, prefill_budget=2048, recovery_route="layout"
    ),
    "gapped-layered-ea-recovery-static-hybrid": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        recovery_weight="uniform",
        prefill_budget=2048,
        recovery_route="layout",
    ),
    "gapped-layered-ea-recovery-compact-hybrid": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
    ),
    "gapped-layered-ea-recovery-plan-hybrid": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        plan_refinement=3,
    ),
    "gapped-layered-ea-recovery-compact-cost": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
    ),
    "gapped-layered-ea-recovery-compact-firstwrite": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        restore_fence="first-write",
    ),
    "gapped-layered-ea-recovery-compact-event": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        event_gpu_history=True,
    ),
    "gapped-layered-ea-recovery-compact-stream": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        control_stream=True,
    ),
    "gapped-layered-ea-recovery-compact-observe-stream": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        control_readonly_stream=True,
    ),
    "gapped-layered-ea-executable-route": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        executable_route=True,
    ),
    "gapped-layered-ea-executable-gpu": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        executable_route=True,
        executable_gpu=True,
    ),
    "gapped-layered-ea-executable-joint": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        compact_recovery=True,
        cost_route=True,
        executable_route=True,
        executable_gpu=True,
        executable_cpu=True,
    ),
    "gapped-layered-ea-recovery-segments": Layer(
        "conditioned",
        "joint",
        True,
        True,
        recovery_gpu=True,
        prefill_budget=2048,
        recovery_route="layout",
        segment_capacity=True,
    ),
    "gapped-layered-ea-recovery-static": Layer(
        "conditioned", "joint", True, True, recovery_gpu=True, recovery_weight="uniform"
    ),
    "gapped-layered-ea-recovery-current": Layer(
        "conditioned", "continuation-current", True, True, recovery_gpu=True
    ),
    "gapped-layered-ea-recovery-current-static": Layer(
        "conditioned",
        "continuation-current",
        True,
        True,
        recovery_gpu=True,
        recovery_weight="uniform",
    ),
    "layered-duplicate-joint": Layer(None, "hicache", True, True),
}
SEGMENTED_LAYERS = tuple((name for name in LAYERS if name.startswith("gapped-")))
CONTINUATION_LAYERS = tuple(
    (name for name in SEGMENTED_LAYERS if LAYERS[name].host.startswith("continuation"))
)


def cpu_flags(layer):
    return dict(
        partial_admission=True,
        allocation_reclaim=False,
        hicache_reclaim=layer.host
        in ("hicache", "continuation", "continuation-repair", "continuation-current"),
        demand_lru_reclaim=layer.host == "lru",
        demand_zero_reclaim=layer.host == "joint",
        complementary_admission=layer.chunk_admission,
        pressure_write_bridge=layer.chunk_admission,
        duplicate_admission=False,
        shelter_arrived_tail=False,
        reference_first_reclaim=False,
        official_admission_telemetry=layer.admission is not None,
    )


def withdraw_chunks(index, victims):
    result = {}
    for bid in victims:
        for program, page in index.by_block.get(bid, ()):
            result.setdefault(program, set()).add(page // 16)
    return {p: sorted(indices) for p, indices in result.items()}
