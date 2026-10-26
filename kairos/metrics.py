"""Distribution evaluation in standard service seconds; no ML dependencies."""
from collections import defaultdict
import math


def distribution(atoms, mass):
    if len(atoms) != len(mass) or not atoms:
        raise ValueError('Nonempty equal-length support and weights required')
    merged = defaultdict(float)
    for value, weight in zip(atoms, mass):
        if not math.isfinite(value) or not math.isfinite(weight) or min(value, weight) < 0:
            raise ValueError('Invalid distribution')
        merged[float(value)] += float(weight)
    total = math.fsum(merged.values())
    if total <= 0:
        raise ValueError('Zero total mass')
    support = sorted(merged)
    return support, [merged[value]/total for value in support]


def crps(atoms, mass, truth):
    """Exact finite-distribution CRPS using an O(K log K) sorted scan."""
    if not math.isfinite(truth) or truth < 0:
        raise ValueError('Invalid target')
    atoms, mass = distribution(atoms, mass)
    cumulative = 0.
    half_pair = 0.
    for value, weight in zip(atoms, mass):
        half_pair += weight*value*(2*cumulative+weight-1)
        cumulative += weight
    return math.fsum(p*abs(x-truth) for x,p in zip(atoms,mass))-half_pair


def quantile(atoms, mass, q):
    if not 0 <= q <= 1:
        raise ValueError('Invalid probability')
    atoms, mass = distribution(atoms, mass)
    cumulative = 0.
    for value, weight in zip(atoms, mass):
        cumulative += weight
        if cumulative >= q-1e-12:
            return value
    return atoms[-1]


def evaluate(prediction, truth):
    low = quantile(prediction['atoms'], prediction['mass'], .05)
    high = quantile(prediction['atoms'], prediction['mass'], .95)
    return dict(crps_s=crps(prediction['atoms'],prediction['mass'],truth),
                mae_s=abs(prediction['pred']-truth),coverage90=float(low<=truth<=high),
                width90_s=high-low)


def macro_average(rows):
    """First average states within program; then programs equally."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['id']].append(row['metrics'])
    if not grouped:
        raise ValueError('Empty evaluation')
    keys = next(iter(grouped.values()))[0].keys()
    return {k:math.fsum(math.fsum(r[k] for r in states)/len(states)
              for states in grouped.values())/len(grouped) for k in keys}
