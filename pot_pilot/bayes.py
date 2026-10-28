"""Bayesian fits of the GEV and POT models, used from step 15 on.

Both models are fitted the same way: the same likelihoods as the maximum-likelihood fits in
steps 04 and 08, a prior on the shape, flat priors on everything else, and the same sampler.
That way the only difference between a fit with a constrained tail and one without is the
prior on the shape.

Parameters. Everything here works in the textbook shape xi (xi > 0 is a heavy tail), like
pot.py. SciPy's genextreme uses c = -xi, so the GEV code below does not use it.
    GEV   location mu, log scale, xi                       (fitted to the 45 annual maxima)
    POT   log scale, xi of the GPD, and the event rate     (fitted to the event peaks)

Priors. Flat on mu, on log scale and on log rate; the shape prior is one of SHAPE_PRIORS,
which match output/tail_priors.csv from step 14. In POT, the event rate has nothing in common
with the GPD parameters once the threshold is fixed, so its posterior is drawn directly:
with a flat prior on log rate it is a gamma distribution with shape n_events and rate n_years.

Sampler. Random-walk Metropolis, run as 4 independent chains from scattered starting points.
During warm-up the proposal is tuned to the shape of the posterior (adaptive Metropolis,
Haario et al. 2001), then fixed for the draws that are kept. Convergence is checked with
split R-hat and the effective sample size (Gelman et al., Bayesian Data Analysis, 3rd ed.,
chapter 11).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma

import pot
from paths import DAILY_CSV, WATER_YEARS

# The shape priors from step 14 (all on xi).
SHAPE_PRIORS = {
    "flat": {"kind": "uniform", "low": -1.0, "high": 1.0},
    "main": {"kind": "normal", "mean": -0.10, "sd": 0.12},
    "wider": {"kind": "normal", "mean": -0.10, "sd": 0.20},
    "narrower": {"kind": "normal", "mean": -0.10, "sd": 0.06},
    "zero": {"kind": "normal", "mean": 0.0, "sd": 0.12},
    "zero_wider": {"kind": "normal", "mean": 0.0, "sd": 0.20},
}
SMALL_XI = 1e-6    # below this, use the xi = 0 (Gumbel or exponential) form


def log_shape_prior(xi, prior):
    """Log prior density of xi, up to a constant."""
    if prior["kind"] == "uniform":
        return 0.0 if prior["low"] < xi < prior["high"] else -np.inf
    return -0.5 * ((xi - prior["mean"]) / prior["sd"]) ** 2


def gev_log_likelihood(params, maxima):
    """GEV log-likelihood of the annual maxima. params = (mu, log scale, xi)."""
    mu, log_scale, xi = params
    y = (maxima - mu) / np.exp(log_scale)
    n = maxima.size
    if abs(xi) < SMALL_XI:
        return -n * log_scale - y.sum() - np.exp(-y).sum()
    t = 1.0 + xi * y
    if np.any(t <= 0):
        return -np.inf
    log_t = np.log(t)
    return -n * log_scale - (1.0 + 1.0 / xi) * log_t.sum() - np.exp(-log_t / xi).sum()


def gpd_log_likelihood(params, excesses):
    """GPD log-likelihood of the excesses over the threshold. params = (log scale, xi)."""
    log_scale, xi = params
    y = excesses / np.exp(log_scale)
    n = excesses.size
    if abs(xi) < SMALL_XI:
        return -n * log_scale - y.sum()
    t = 1.0 + xi * y
    if np.any(t <= 0):
        return -np.inf
    return -n * log_scale - (1.0 + 1.0 / xi) * np.log(t).sum()


def gev_log_posterior(params, maxima, prior):
    lp = log_shape_prior(params[2], prior)
    return lp if lp == -np.inf else lp + gev_log_likelihood(params, maxima)


def gpd_log_posterior(params, excesses, prior):
    lp = log_shape_prior(params[1], prior)
    return lp if lp == -np.inf else lp + gpd_log_likelihood(params, excesses)


def posterior_mode(log_post, start):
    """The highest point of the posterior, and the curvature there (for the first proposal).

    With the flat prior this is the maximum-likelihood fit, which step 15 checks against the
    fits from steps 04 and 08.
    """
    def negative(p):
        value = log_post(p)
        return np.inf if not np.isfinite(value) else -value

    best = None
    for jitter in (0.0, 0.05, -0.05):
        x0 = np.array(start, float)
        x0[-1] += jitter
        result = minimize(negative, x0, method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000})
        if best is None or result.fun < best.fun:
            best = result
    curvature = pot.hessian(negative, best.x)
    return best.x, curvature


def sample(log_post, mode, curvature, seed, n_chains=4, n_warmup=4000, n_keep=6000):
    """Draws from the posterior: an array of shape (chains, draws, parameters).

    The chains start at points scattered around the mode, twice as wide as the posterior is
    expected to be, so that R-hat can tell whether they found the same distribution.
    """
    rng = np.random.default_rng(seed)
    d = len(mode)
    try:
        cov = np.linalg.inv(curvature)
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = np.diag(np.full(d, 1e-3))
    step_scale = 2.38**2 / d    # the usual scaling for a random-walk proposal
    draws = np.empty((n_chains, n_keep, d))
    acceptance = np.empty(n_chains)

    for chain in range(n_chains):
        start_factor = np.linalg.cholesky(cov)
        current = mode + 2.0 * start_factor @ rng.standard_normal(d)
        current_lp = log_post(current)
        while not np.isfinite(current_lp):
            current = mode + start_factor @ rng.standard_normal(d)
            current_lp = log_post(current)
        factor = np.linalg.cholesky(step_scale * cov)    # proposal = current + factor @ z
        steps = rng.standard_normal((n_warmup + n_keep, d))
        log_u = np.log(rng.random(n_warmup + n_keep))
        history = np.empty((n_warmup, d))
        accepted = 0
        for i in range(n_warmup + n_keep):
            # Warm-up: every 500 steps, tune the proposal to the draws so far.
            if i < n_warmup and i >= 1000 and i % 500 == 0:
                recent = history[i // 2:i]
                factor = np.linalg.cholesky(step_scale * (np.cov(recent.T) + 1e-10 * np.eye(d)))
            candidate = current + factor @ steps[i]
            candidate_lp = log_post(candidate)
            if log_u[i] < candidate_lp - current_lp:
                current, current_lp = candidate, candidate_lp
                if i >= n_warmup:
                    accepted += 1
            if i < n_warmup:
                history[i] = current
            else:
                draws[chain, i - n_warmup] = current
        acceptance[chain] = accepted / n_keep
    return draws, acceptance


def split_rhat(chains):
    """Split R-hat for one quantity. chains has shape (chains, draws). Close to 1 is good."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]])
    within = halves.var(axis=1, ddof=1).mean()
    between = n * halves.mean(axis=1).var(ddof=1)
    pooled = (n - 1) / n * within + between / n
    return float(np.sqrt(pooled / within))


def effective_sample_size(chains):
    """Effective number of independent draws for one quantity. chains: (chains, draws).

    Uses the autocorrelation averaged over chains, summed in pairs until a pair turns negative
    (Geyer's initial positive sequence).
    """
    m, n = chains.shape
    centered = chains - chains.mean(axis=1, keepdims=True)
    # autocovariance of each chain at every lag, through the FFT
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centered, size, axis=1)
    autocov = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n] / n
    within = chains.var(axis=1, ddof=1).mean()
    pooled = (n - 1) / n * within + chains.mean(axis=1).var(ddof=1)
    rho = 1.0 - (within - autocov.mean(axis=0)) / pooled
    rho[0] = 1.0
    total = 0.0
    for lag in range(0, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0:
            break
        total += pair
    tau = max(2.0 * total - 1.0, 1.0)
    return float(m * n / tau)


def gev_return_levels(return_periods, mu, scale, xi):
    """GEV return levels for each draw. Returns an array of shape (draws, return periods)."""
    y = -np.log(1.0 - 1.0 / np.asarray(return_periods, float))
    mu, scale, xi = (np.asarray(v, float)[:, None] for v in (mu, scale, xi))
    small = np.abs(xi) < SMALL_XI
    safe_xi = np.where(small, 1.0, xi)
    heavy = mu - scale / safe_xi * (1.0 - y ** (-safe_xi))
    gumbel = mu - scale * np.log(y)
    return np.where(small, gumbel, heavy)


def pot_return_levels(return_periods, threshold, rate, scale, xi):
    """POT return levels for each draw, the same formula as pot.return_level."""
    y = -np.log(1.0 - 1.0 / np.asarray(return_periods, float))
    rate, scale, xi = (np.asarray(v, float)[:, None] for v in (rate, scale, xi))
    m = rate / y
    small = np.abs(xi) < SMALL_XI
    safe_xi = np.where(small, 1.0, xi)
    heavy = threshold + scale / safe_xi * (m**safe_xi - 1.0)
    exponential = threshold + scale * np.log(m)
    return np.where(small, exponential, heavy)


def pilot_series(cells):
    """The daily values (days x cells) and water-year maxima (years x cells) at the pilot cells."""
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    values = daily[[str(c) for c in cells.cell_id]].to_numpy(np.float32).astype(np.float64)
    water_year = np.where(daily.date.dt.month >= 10, daily.date.dt.year + 1, daily.date.dt.year)
    annual_max = np.array([values[water_year == year].max(axis=0) for year in WATER_YEARS])
    return values, annual_max


def fit_version(model, prior_name, data, start, seed, return_periods, n_years, threshold=None,
                n_warmup=4000, n_keep=6000):
    """Fit one version at one cell and summarize the posterior.

    model is "GEV" (data = the annual maxima) or "POT" (data = the excesses over threshold).
    start is a starting point for finding the mode (the maximum-likelihood fit, or close to it).
    n_warmup and n_keep are the chain lengths passed to sample(). Returns a dictionary with
    the posterior mode, the draws of every reported quantity as (chains, draws) arrays, the
    return-level draws (draws x return periods), R-hat and ESS for each quantity, and the
    acceptance rate.
    """
    prior = SHAPE_PRIORS[prior_name]
    if model == "GEV":
        log_post = lambda p: gev_log_posterior(p, data, prior)
        names = ["location", "log_scale", "xi"]
    else:
        log_post = lambda p: gpd_log_posterior(p, data, prior)
        names = ["log_scale", "xi"]

    mode, curvature = posterior_mode(log_post, start)
    draws, acceptance = sample(log_post, mode, curvature, seed, n_warmup=n_warmup, n_keep=n_keep)
    n_chains, n_draws, _ = draws.shape

    quantities = {name: draws[:, :, i] for i, name in enumerate(names)}
    quantities["scale"] = np.exp(quantities.pop("log_scale"))
    flat = {name: q.ravel() for name, q in quantities.items()}
    if model == "GEV":
        levels = gev_return_levels(return_periods, flat["location"], flat["scale"], flat["xi"])
    else:
        # The event rate is drawn on its own (see the top of this file), one per GPD draw.
        rng = np.random.default_rng(seed + 500)
        rate = gamma.rvs(data.size, scale=1.0 / n_years, size=n_chains * n_draws,
                         random_state=rng)
        quantities["rate"] = rate.reshape(n_chains, n_draws)
        flat["rate"] = rate
        levels = pot_return_levels(return_periods, threshold, rate, flat["scale"], flat["xi"])
    for j, period in enumerate(return_periods):
        quantities[f"rp{period}"] = levels[:, j].reshape(n_chains, n_draws)

    return {
        "mode": mode, "quantities": quantities, "flat": flat, "levels": levels,
        "rhat": {name: split_rhat(q) for name, q in quantities.items()},
        "ess": {name: effective_sample_size(q) for name, q in quantities.items()},
        "acceptance": float(acceptance.mean()),
    }
