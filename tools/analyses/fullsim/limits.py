"""Counting-experiment statistics for a signal window with expected background b.

The Run-1A analysis' limit and discovery (Run-1A-Analysis/stats/profile2d.py
and profile.py), which are what results are quoted with:

  cls_upper_limit          CLs upper limit on the signal for n observed:
                           exact Poisson CLs without systematics, else the
                           asymptotic profile-likelihood CLs with Gaussian
                           constraints on b and on the signal efficiency
  required_signal          signal for a Z sigma discovery, Z sqrt(b + sigma_b^2)

and RefAna/pyCount's SensitivityAnalyzer, kept for comparison with it (no
systematics):

  fc_interval              Feldman-Cousins interval on the signal for n observed
  expected_upper_limit     classical Poisson upper limit, averaged over n ~ Poisson(b)
  fc_table_upper_limit     the 90% CL FC upper limit for n = b, from FC.csv

The FC construction is SensitivityAnalyzer's (likelihood-ratio ordering over
n = 0..N_MAX), but the interval edges are found by bisection on s rather than
read off a 200,000-point grid, which took over a minute per call. The edges
agree with the grid's to its 1e-4 spacing.
"""

from pathlib import Path

import numpy as np

# The 90% CL Feldman-Cousins upper limit on s for n = b observed, tabulated in
# b (columns b, s90), from RefAna/pyCount FC.csv. Covers b = 0..260.
FC_TABLE = Path(__file__).resolve().parent / "FC.csv"

# Largest count the FC acceptance sets range over (SensitivityAnalyzer's
# n_max_override). Far beyond any b a Mu2e signal window sees.
N_MAX = 2000
# Bisection steps for an interval edge or a required signal: enough to get
# far below any precision quoted.
BISECTION_STEPS = 100


def _fc_accepts(s: float, b: float, n_obs: int, cl: float) -> bool:
    """Is n_obs in the FC acceptance set for signal s, background b?"""
    from scipy.stats import poisson

    n = np.arange(N_MAX + 1)
    s_best = np.maximum(0.0, n - b)
    ranks = poisson.logpmf(n, s + b) - poisson.logpmf(n, s_best + b)
    probs = poisson.pmf(n, s + b)
    order = np.argsort(ranks)[::-1]
    cum = np.cumsum(probs[order])
    accepted = order[: int(np.searchsorted(cum, cl)) + 1]
    return bool(np.any(accepted == n_obs))


def _edge(inside: float, outside: float, accepts) -> float:
    """The boundary between a point `accepts` holds at and one it does not."""
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (inside + outside)
        if accepts(mid):
            inside = mid
        else:
            outside = mid
    return inside


def fc_interval(n_obs: int, b: float, cl: float = 0.9) -> tuple[float, float]:
    """Feldman-Cousins interval [s_low, s_high] on the signal, in events."""
    def accepts(s):
        return _fc_accepts(s, b, n_obs, cl)

    # The best-fit signal always accepts n_obs: it ranks first there.
    s_hat = max(0.0, n_obs - b)
    low = 0.0 if accepts(0.0) else _edge(s_hat, 0.0, accepts)
    outside = max(20.0, 2.0 * s_hat + 10.0)
    while accepts(outside):
        outside *= 2.0
    return low, _edge(s_hat, outside, accepts)


def poisson_upper_limit(n_obs: int, b: float, cl: float = 0.9) -> float:
    """Classical upper limit: the s with P(n <= n_obs | s + b) = 1 - cl,
    or 0 when even s = 0 is excluded."""
    from scipy.stats import poisson

    target = 1.0 - cl
    if poisson.cdf(n_obs, b) < target:
        return 0.0
    high = 1.0
    while poisson.cdf(n_obs, b + high) > target:
        high *= 2.0
    return _edge(0.0, high, lambda s: poisson.cdf(n_obs, b + s) > target)


def expected_upper_limit(b: float, cl: float = 0.9) -> float:
    """The classical upper limit averaged over n ~ Poisson(b)."""
    from scipy.stats import poisson

    n_max = 50 if b <= 0 else int(max(50, b + 10 * np.sqrt(b) + 20))
    n = np.arange(n_max + 1)
    weights = poisson.pmf(n, b)
    limits = np.array([poisson_upper_limit(int(k), b, cl) for k in n])
    total = float(np.dot(weights, limits))
    tail = 1.0 - float(weights.sum())
    if tail > 1e-8:
        total += tail * poisson_upper_limit(n_max + 1, b, cl)
    return total


def fc_table_upper_limit(b: float, table: Path = FC_TABLE) -> float:
    """The tabulated 90% CL FC upper limit for n = b, interpolated in b."""
    data = np.loadtxt(table, delimiter=",")
    return float(np.interp(b, data[:, 0], data[:, 1]))


def _root(f, low: float) -> float:
    """The root of a decreasing f above `low`, widening the bracket upward
    until it holds one (profile2d's fixed [0.001, 100] can miss it)."""
    from scipy.optimize import brentq

    high = max(2.0 * low, 1.0)
    while f(high) > 0.0:
        high *= 2.0
        if high > 1e7:
            raise ValueError("no upper limit below 1e7 signal events")
    return float(brentq(f, low, high, xtol=1e-10))


# Lower bound on a profiled nuisance parameter, as profile2d.py has it.
NUISANCE_FLOOR = 1e-3


def cls_upper_limit(n_obs: int, b: float, cl: float = 0.9, b_sigma: float = 0.0,
                    eff_sigma: float = 0.0) -> float:
    """CLs upper limit on the signal events, as Run-1A's profile2d.py sets it.

    With no uncertainty, exact Poisson CLs: the s solving
    P(n <= n_obs | s + b) / P(n <= n_obs | b) = 1 - cl.

    Otherwise the likelihood Pois(n_obs | eps s + b') x G(b'; b, b_sigma) x
    G(eps; 1, eff_sigma) is profiled over b' and eps, the one-sided q_mu
    (0 below the best-fit signal) gives the asymptotic CLs
    2 (1 - Phi(sqrt(q_mu))), and the limit is where that is 1 - cl.
    `b_sigma` is absolute (events), `eff_sigma` a fraction of the efficiency.

    One difference from profile2d.py: a nuisance whose sigma is 0 stays
    fixed. profile2d drops the background's constraint when b_sigma = 0 but
    eff_sigma > 0, which leaves b' free and roughly triples the limit.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm, poisson

    alpha = 1.0 - cl
    if b_sigma <= 0 and eff_sigma <= 0:
        cl_b = poisson.cdf(n_obs, b)
        return _root(lambda s: poisson.cdf(n_obs, s + b) / cl_b - alpha, 0.0)

    free_b, free_eff = b_sigma > 0, eff_sigma > 0

    def nll(s, params):
        params = list(params)
        b_fit = params.pop(0) if free_b else b
        eff = params.pop(0) if free_eff else 1.0
        mu = s * eff + b_fit
        # mu = 0 is allowed for n_obs = 0: no background and no signal.
        if mu < 0 or (mu == 0 and n_obs > 0):
            return 1e10
        value = -poisson.logpmf(n_obs, mu)
        if free_b:
            value += 0.5 * ((b_fit - b) / b_sigma) ** 2
        if free_eff:
            value += 0.5 * ((eff - 1.0) / eff_sigma) ** 2
        return value

    start = [b] * free_b + [1.0] * free_eff
    bounds = [(NUISANCE_FLOOR, None)] * len(start)

    def profiled(s):
        fit = minimize(lambda p: nll(s, p), x0=start, bounds=bounds)
        return nll(s, fit.x)

    s_hat = max(0.0, n_obs - b)
    nll_best = profiled(s_hat)

    def cls_minus_alpha(s):
        q = 0.0 if s < s_hat else max(0.0, 2.0 * (profiled(s) - nll_best))
        return 2.0 * norm.sf(np.sqrt(q)) - alpha

    return _root(cls_minus_alpha, max(s_hat, NUISANCE_FLOOR))


def required_signal(z: float, b: float, b_sigma: float = 0.0) -> float:
    """Signal events for a z sigma discovery, z sqrt(b + b_sigma^2), as
    Run-1A's profile.py has it. NaN with no background at all, where that
    says 0 and means nothing."""
    variance = b + b_sigma ** 2
    return float(z * np.sqrt(variance)) if variance > 0 else float("nan")
