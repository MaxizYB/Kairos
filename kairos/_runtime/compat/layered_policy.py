"""Kairos layered policy implementation."""
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
    recovery_weight: str = 'time'
    prefill_budget: int | None = None
    recovery_route: str = 'segmented'
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
    restore_fence: str = 'before-forward'


LAYERS = {
    "layered-ea-duplicate": Layer("conditioned", "hicache", False),
    "layered-ea-duplicate-gpu": Layer("conditioned", "hicache", True),
    "layered-ea-duplicate-leaf": Layer("conditioned", "hicache", False, device_leaf=True),
    "layered-ea-duplicate-leaf-cached": Layer("conditioned", "hicache", False,
                                            device_leaf=True, leaf_reuse_nodes=True),
    "layered-ea-gpu": Layer("conditioned", "lru", True),
    "layered-ea-joint": Layer("conditioned", "joint", True, True),
    # Same official request admission and pressure-write budget as the
    # strongest prefix baseline, on a common exact segmented backend.
    "gapped-layered-ea-joint": Layer("conditioned", "joint", True, True),
    "gapped-layered-ea-continuation": Layer("conditioned", "continuation", True, True),
    "gapped-layered-ea-continuation-repair": Layer("conditioned", "continuation-repair", True, True),
    "gapped-layered-ea-recovery": Layer("conditioned", "continuation-repair", True, True,
                                       recovery_gpu=True),
    # Factorial and information controls on the identical execution path.
    # The static arm is a matched cost-only ablation, not an upstream system.
    "gapped-layered-ea-recovery-gpu": Layer("conditioned", "joint", True, True,
                                           recovery_gpu=True),
    # Matched execution diagnostic: same victims, admission, TTL and DMA;
    # retain the registered native query cap instead of the 512-token shrink.
    "gapped-layered-ea-recovery-gpu-wide": Layer("conditioned", "joint", True, True,
                                                recovery_gpu=True,prefill_budget=2048),
    # Same placement and shared 2048 cap; native prefix protocol when the
    # current bounded segmented plan offers no additional query reuse.
    "gapped-layered-ea-recovery-gpu-hybrid": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout'),
    # Shared execution for the strongest mature layout baseline. A joint
    # strategy comparison must not reserve the protocol improvement for us.
    "gapped-layered-ea-joint-hybrid": Layer("conditioned", "joint", True, True,
        prefill_budget=2048,recovery_route='layout'),
    "gapped-layered-ea-recovery-static-hybrid": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,recovery_weight='uniform',prefill_budget=2048,recovery_route='layout'),
    # Indexed masks preserve conditional chunk choices. The separate plan
    # arm additionally quotes three final equal-capacity alternatives only.
    "gapped-layered-ea-recovery-compact-hybrid": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True),
    "gapped-layered-ea-recovery-plan-hybrid": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        plan_refinement=3),
    "gapped-layered-ea-recovery-compact-cost": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True),
    # Mechanism control, not a new placement algorithm: whole-range RETRIEVE
    # may overlap embedding/QKV, fenced BEFORE the first cache writer.
    "gapped-layered-ea-recovery-compact-firstwrite": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,restore_fence='first-write'),
    "gapped-layered-ea-recovery-compact-event": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,event_gpu_history=True),
    "gapped-layered-ea-recovery-compact-stream": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,control_stream=True),
    "gapped-layered-ea-recovery-compact-observe-stream": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,control_readonly_stream=True),
    "gapped-layered-ea-executable-route": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,executable_route=True),
    "gapped-layered-ea-executable-gpu": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,executable_route=True,executable_gpu=True),
    "gapped-layered-ea-executable-joint": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',compact_recovery=True,
        cost_route=True,executable_route=True,executable_gpu=True,executable_cpu=True),
    # Exact recovery descriptors no longer consume the real-request quota.
    # Same 8 real requests, full-KV capacity, original TTL and GPU victims.
    "gapped-layered-ea-recovery-segments": Layer("conditioned", "joint", True, True,
        recovery_gpu=True,prefill_budget=2048,recovery_route='layout',segment_capacity=True),
    "gapped-layered-ea-recovery-static": Layer("conditioned", "joint", True, True,
                                              recovery_gpu=True,recovery_weight='uniform'),
    "gapped-layered-ea-recovery-current": Layer("conditioned", "continuation-current", True, True,
                                               recovery_gpu=True),
    "gapped-layered-ea-recovery-current-static": Layer("conditioned", "continuation-current", True, True,
                                                      recovery_gpu=True,recovery_weight='uniform'),
    "layered-duplicate-joint": Layer(None, "hicache", True, True),
}

SEGMENTED_LAYERS = tuple(name for name in LAYERS if name.startswith('gapped-'))
CONTINUATION_LAYERS = tuple(name for name in SEGMENTED_LAYERS
                            if LAYERS[name].host.startswith('continuation'))


def cpu_flags(layer):
    return dict(partial_admission=True, allocation_reclaim=False,
        hicache_reclaim=layer.host in ("hicache", "continuation", "continuation-repair", "continuation-current"),
        demand_lru_reclaim=layer.host == "lru",
        demand_zero_reclaim=layer.host == "joint",
        complementary_admission=layer.chunk_admission,
        pressure_write_bridge=layer.chunk_admission,
        duplicate_admission=False, shelter_arrived_tail=False,
        reference_first_reclaim=False,
        official_admission_telemetry=layer.admission is not None)


def withdraw_chunks(index, victims):
    """Only actual upcoming native victims can invalidate a GPU proof."""
    result = {}
    for bid in victims:
        for program, page in index.by_block.get(bid, ()):
            result.setdefault(program, set()).add(page // 16)
    return {p: sorted(indices) for p, indices in result.items()}
