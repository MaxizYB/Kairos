"""runtime.compat.versions."""

RESIDENCY = "replica-joint-demand-zero-admit-partial"
CONTINUITY = "replica-joint-demand-bridge-admit-partial"


def expand_version(name):
    for version, implementation in (
        ("residency", RESIDENCY),
        ("continuity", CONTINUITY),
    ):
        if name == version or name.startswith(version + "-"):
            return implementation + name[len(version) :]
    return name
