"""Fail closed if an unshipped experimental predictor is requested."""


def unavailable(*args, **kwargs):
    raise ValueError('This branch is not part of the Kairos release configuration')


CompactPredictor = unavailable
SuffixPredictor = unavailable
completion_cost = unavailable
