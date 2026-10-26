from __future__ import annotations
import math

def distribution_benefit(current, atoms, mass):
    """Expected completion-share benefit under a discrete residual posterior.

    A mean plug-in is not equivalent for this nonlinear objective. Keep the
    posterior finite and normalized; evaluate each atom before averaging.
    """
    current = float(current)
    atoms = tuple(float(v) for v in atoms)
    mass = tuple(float(v) for v in mass)
    if current <= 0 or len(atoms) != len(mass) or not atoms:
        raise ValueError('Invalid residual posterior')
    if any(not math.isfinite(v) or v < 0 for v in atoms + mass):
        raise ValueError('Residual posterior must be finite and nonnegative')
    total = math.fsum(mass)
    if total <= 0:
        raise ValueError('Residual posterior has zero mass')
    return math.fsum((p / total) * current / (current + r)
                     for r, p in zip(atoms, mass))


def posterior_mean(atoms, mass):
    """Finite posterior mean, kept separate for logging and diagnostics."""
    atoms = tuple(float(v) for v in atoms)
    mass = tuple(float(v) for v in mass)
    if len(atoms) != len(mass) or not atoms:
        raise ValueError('Invalid posterior')
    total = math.fsum(mass)
    if total <= 0 or any(not math.isfinite(v) or v < 0 for v in atoms + mass):
        raise ValueError('Invalid posterior')
    return math.fsum(r * p for r, p in zip(atoms, mass)) / total


def prefix_pages(ids, removed=()):
    removed = set(removed)
    return next((i for i, bid in enumerate(ids) if bid is None or bid in removed), len(ids))


def recovery_seconds(layout, removed, profile, page=16):
    """One contiguous GPU prefix plus one contiguous host interval, like v21.

    The whole resulting layout is priced together. A CPU hole stops the
    restoration frontier, so independently summing per-page costs is wrong.
    No new offload is requested by a release plan; CPU contents therefore
    stay unchanged in this local comparison. Later stores may change them.
    """
    gpu = prefix_pages(layout['blocks'], removed)
    tokens = int(layout.get('tokens', len(layout['blocks']) * page))
    tokens = max(0, tokens)
    gpu_tokens = min(tokens, gpu * page)
    host = set(layout.get('cpu_chunks', ()))
    chunks = int(layout.get('chunks', tokens // 256))
    chunk = gpu_tokens // 256
    while chunk < chunks and chunk in host:
        chunk += 1
    end = min(tokens, max(gpu_tokens, chunk * 256))
    load = max(0, end - gpu_tokens)
    beta = profile['beta']
    # Chunked recompute agrees with the serving reference service convention.
    prefill = 0.0
    cursor = end
    while cursor < tokens:
        q = min(2048, tokens - cursor)
        prefill += max(0., beta[0] + beta[1] * q +
                       beta[2] * q * (2 * cursor + q + 1) / 2 +
                       (beta[3] if len(beta) > 3 else 0.))
        cursor += q
    if load:
        prefill += max(0., profile.get('reload_fixed_seconds', 0.))
        prefill += load * max(0., profile.get('reload_seconds_per_token', 0.))
    return prefill


