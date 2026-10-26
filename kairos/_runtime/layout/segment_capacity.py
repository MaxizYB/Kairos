"""Kairos segment capacity implementation."""


def segment_room(real_limit, runner_limit, running, reserved, incoming):
    if not 0 < real_limit <= runner_limit:
        raise ValueError('Invalid request/runner capacity')
    return max(0, runner_limit - running - reserved - int(incoming))


def query_slots(real_limit, runner_limit, running, reserved, incoming, maximum):
    if maximum < 1:
        raise ValueError('Invalid segment limit')
    return min(maximum, 1 + segment_room(real_limit, runner_limit,
                                        running, reserved, incoming))
