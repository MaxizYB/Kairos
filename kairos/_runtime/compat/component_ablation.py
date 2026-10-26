"""Kairos component ablation implementation."""


def native_lru_payload(payload):
    if payload.get('op')!='experiment':return payload
    return dict(payload,mode='observe',allocation_reclaim=False,
        partial_admission=True,complementary_admission=False,
        duplicate_admission=False,demand_zero_reclaim=False,
        pressure_write_bridge=False,demand_lru_reclaim=True,
        shelter_arrived_tail=False,reference_first_reclaim=False)
