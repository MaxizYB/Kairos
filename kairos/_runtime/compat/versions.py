"""Kairos versions implementation."""

V20 = "replica-joint-demand-zero-admit-partial"
V21 = "replica-joint-demand-bridge-admit-partial"


def expand_version(name):
    # Expand before parsing independent ablation and repeat suffixes.
    for version, implementation in (("v20", V20), ("v21", V21)):
        if name == version or name.startswith(version + "-"):
            return implementation + name[len(version):]
    return name
