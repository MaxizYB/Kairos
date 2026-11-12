"""Continuous joint law, survival conditioning, and geometric service quotes.

The learned density remains continuous. Numerical interpolation approximates
conditional integrals, never replaces the distribution with learned buckets.
An unchanged observation/cost state reuses the exact previous calculation.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import erf, log_ndtr, logsumexp, ndtri_exp, softmax


@lru_cache(maxsize=4)
def quadrature(order):
    z, w = np.polynomial.hermite.hermgauss(order)
    u, v = np.polynomial.legendre.leggauss(order)
    return z * math.sqrt(2), w / math.sqrt(math.pi), (u + 1) / 2, v / 2


class JointDensity:
    """p(O | arrived history) p(R | O, arrived history). No outcome inputs."""

    def __init__(self, state):
        self.state = {k: np.asarray(v, float) for k, v in state.items()}
        if not all(np.isfinite(v).all() for v in self.state.values()):
            raise ValueError("Nonfinite density weights")
        self.mu_o = self.state["log_output_centers"]
        self.mu_r = self.state["log_residual_centers"]
        self.sigma_o = 0.02 + np.logaddexp(0, self.state["output_width_raw"])
        self.sigma_r = 0.02 + np.logaddexp(0, self.state["residual_width_raw"])

    def _network(self, value, name):
        x = 0.5 * value * (1 + erf(value / math.sqrt(2)))
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1.0e-5)
        x = x * self.state[name + ".1.weight"] + self.state[name + ".1.bias"]
        return softmax(
            x @ self.state[name + ".2.weight"].T + self.state[name + ".2.bias"], axis=-1
        )

    def prepare(self, feature):
        feature = np.asarray(feature, float)
        if feature.shape != (448,) or not np.isfinite(feature).all():
            raise ValueError("Expected one arrived encoder/numeric feature")
        projected = (
            feature @ self.state["project.weight"].T + self.state["project.bias"]
        )
        return projected, self._network(projected, "output_logits")

    def compile(self, prepared, *, prefill, reference_step, cap, nodes=192, order=3):
        if (
            type(cap) is not int
            or cap < 1
            or not math.isfinite(prefill)
            or prefill < 0
            or not math.isfinite(reference_step)
            or reference_step <= 0
        ):
            raise ValueError("Invalid visible reference state")
        projected, marginal = prepared
        output = np.unique(np.rint(np.geomspace(1, cap, nodes)))
        if len(output) == 1:
            output = np.array([1.0, 1.0 + 1.0e-8])
        coordinate = np.log(output) / max(
            1.0, float(np.log(self.state["output_atoms"][-1]))
        )
        conditional = self._network(
            projected[None, :]
            + coordinate[:, None] * self.state["condition.weight"][:, 0],
            "output",
        )
        z, w, _, _ = quadrature(order)
        residual = np.exp(self.mu_r[:, None] + self.sigma_r[:, None] * z)
        reference = prefill + reference_step * output
        ratio = residual[None, :, :] / reference[:, None, None]
        continuation = conditional[:, 1:]
        log_inverse = np.sum(continuation * (np.log1p(ratio) @ w), axis=1)
        mean_progress = conditional[:, 0] + np.sum(
            continuation * ((1 / (1 + ratio)) @ w), axis=1
        )
        functions = PchipInterpolator(
            np.log(output), np.column_stack([log_inverse, mean_progress]), axis=0
        )
        return CallPosterior(
            self, np.asarray(marginal), functions, cap, order, len(output)
        )


@dataclass(frozen=True)
class SurvivalView:
    output: np.ndarray
    mass: np.ndarray
    expected_log_inverse: float
    mean_progress: float


class CallPosterior:
    """One arrived call; bounded caches keyed only by actual observation/cost."""

    def __init__(self, density, marginal, functions, cap, order, nodes):
        self.density, self.marginal, self.functions = density, marginal, functions
        self.cap, self.order, self.nodes = cap, order, nodes
        self._generated = self._view = self._quote_key = self._quote = None
        self.stats = dict(quotes=0, quote_hits=0, survival_updates=0, survival_hits=0)

    def observe(self, generated):
        if type(generated) is not int or not 0 <= generated < self.cap:
            raise ValueError("Invalid actual observed generation")
        if generated == self._generated:
            self.stats["survival_hits"] += 1
            return self._view
        d = self.density
        z, w, u, v = quadrature(self.order)
        marginal = self.marginal
        if generated:
            tail = log_ndtr(-(math.log(generated + 0.5) - d.mu_o) / d.sigma_o)
            log_mass = np.log(marginal) + tail
            marginal = np.exp(log_mass - logsumexp(log_mass))
            zo, wo = -ndtri_exp(tail[:, None] + np.log1p(-u)), v
        else:
            zo, wo = z[None, :], w
        latent = d.mu_o[:, None] + d.sigma_o[:, None] * zo
        output = np.clip(np.rint(np.exp(np.minimum(latent, 700))), 1, self.cap).ravel()
        weights = (marginal[:, None] * wo).ravel()
        output, inverse = np.unique(output, return_inverse=True)
        mass = np.bincount(inverse, weights=weights, minlength=len(output))
        mass /= mass.sum()
        if (output <= generated).any() or not np.isfinite(mass).all():
            raise ValueError("Invalid continuous survival")
        functions = self.functions(np.log(output))
        self._view = SurvivalView(
            output, mass, float(mass @ functions[:, 0]), float(mass @ functions[:, 1])
        )
        self._generated = generated
        self._quote_key = None
        self.stats["survival_updates"] += 1
        return self._view

    def quote(self, *, generated, restore, decode_step):
        if (
            type(generated) is not int
            or not 0 <= generated < self.cap
            or not math.isfinite(restore)
            or restore < 0
            or not math.isfinite(decode_step)
            or decode_step <= 0
        ):
            raise ValueError("Invalid observed progress or live physical cost")
        self.stats["quotes"] += 1
        key = generated, restore, decode_step
        if key == self._quote_key:
            self.stats["quote_hits"] += 1
            return dict(self._quote)
        view = self.observe(generated)
        current = restore + decode_step * (view.output - generated)
        if (current <= 0).any() or not np.isfinite(current).all():
            raise ValueError("Invalid visible cost")
        log_current = float(view.mass @ np.log(current))
        self._quote = dict(
            actual_s=float(np.exp(log_current)),
            score=float(np.exp(log_current + view.expected_log_inverse)),
            progress=float(np.exp(-view.expected_log_inverse)),
            posterior_progress=view.mean_progress,
        )
        self._quote_key = key
        return dict(self._quote)
