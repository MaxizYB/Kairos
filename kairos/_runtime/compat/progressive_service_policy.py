"""Kairos progressive service policy implementation."""

import math


def _valid(values, name):
    values = tuple(float(v) for v in values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def _quantile(atoms, mass, q):
    cumulative = 0.0
    for atom, probability in zip(atoms, mass):
        cumulative += probability
        if cumulative >= q - 1e-12:
            return atom
    return atoms[-1]


def choose(*, current, residual_atoms=None, residual_mass=None, warm,
           observed_s=0.0):
    current = _valid(current, "current")
    if type(warm) is not bool:
        raise TypeError("warm must be bool")
    if not math.isfinite(observed_s) or observed_s < 0:
        raise ValueError("observed_s must be finite and non-negative")
    current_atoms = sorted(current)
    current_q25 = current_atoms[int((len(current_atoms) - 1) * .25)]
    current_median = current_atoms[int((len(current_atoms) - 1) * .5)]
    current_q75 = current_atoms[int((len(current_atoms) - 1) * .75)]
    source = current_atoms
    mode = "cold-probe"
    if residual_atoms is not None or residual_mass is not None:
        if residual_atoms is None or residual_mass is None:
            raise ValueError("residual atoms and mass must be provided together")
        atoms = _valid(residual_atoms, "residual_atoms")
        mass = _valid(residual_mass, "residual_mass")
        if len(atoms) != len(mass) or any(a > b for a, b in zip(atoms, atoms[1:])):
            raise ValueError("invalid residual support")
        total = math.fsum(mass)
        if total <= 0:
            raise ValueError("residual mass must be positive")
        mass = tuple(p / total for p in mass)
        if warm:
            source = list(atoms)
            mode = "warm-cdf-residual"
    q25 = _quantile(source, [1.0 / len(source)] * len(source), .25) if mode == "cold-probe" else _quantile(source, mass, .25)
    median = _quantile(source, [1.0 / len(source)] * len(source), .5) if mode == "cold-probe" else _quantile(source, mass, .5)
    q75 = _quantile(source, [1.0 / len(source)] * len(source), .75) if mode == "cold-probe" else _quantile(source, mass, .75)
    return dict(mode=mode, score_s=float(median), q25_s=float(q25),
                q75_s=float(q75), uncertainty_s=float(max(0.0, q75-q25)),
                current_median_s=float(current_median),
                observed_s=float(observed_s), predictor='suffix-cdf')
