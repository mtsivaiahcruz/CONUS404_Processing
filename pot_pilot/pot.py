"""Peaks-over-threshold (POT) method for the daily wind series at one grid cell.

The steps, in the order pot_fit() runs them:

1. Threshold. A percentile (95, 97, 98 or 99) of all the daily values at the cell.
2. Events. Days above the threshold are grouped into events. An event ends once `separation`
   days in a row are at or below the threshold. Each event is represented by its highest
   daily value, the peak.
3. Model. A generalized Pareto distribution (GPD) with scale sigma and shape xi is fitted by
   maximum likelihood to the excesses (peak minus threshold).
4. Event rate. rate = number of events / number of years.
5. Return level. The wind speed with a 1-in-N chance of being exceeded in any one year. This
   is the definition the annual-maximum GEV uses, so the two can be compared directly:
       m   = rate / (-ln(1 - 1/N))
       x_N = threshold + sigma / xi * (m**xi - 1)      if xi is not 0
       x_N = threshold + sigma * ln(m)                 if xi is 0
6. Standard error, by the delta method. It combines the uncertainty in the event rate with
   the uncertainty in sigma and xi. The threshold is treated as fixed.

xi is the usual GPD shape (xi > 0 means a heavy tail). scipy.stats.genpareto calls it c.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto, kstest

Z95 = 1.96    # a 95% interval is the estimate plus or minus 1.96 standard errors


def decluster(values, threshold, separation):
    """Group the days above the threshold into events and keep each event's peak.

    Returns the day index of each peak, the peak values, and the number of days above the
    threshold.
    """
    days_above = np.flatnonzero(values > threshold)
    if days_above.size == 0:
        return days_above, values[days_above], 0

    # A new event starts wherever two days above the threshold are more than `separation`
    # days apart. That gap means at least `separation` days in between were at or below it.
    breaks = np.flatnonzero(np.diff(days_above) > separation)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, days_above.size]

    peak_days = []
    for start, end in zip(starts, ends):
        event_days = days_above[start:end]
        peak_days.append(event_days[np.argmax(values[event_days])])
    peak_days = np.array(peak_days)
    return peak_days, values[peak_days], int(days_above.size)


def gpd_nll(params, excesses):
    """Negative log-likelihood of the GPD for params = (sigma, xi). Lower means a better fit."""
    sigma, xi = params
    if sigma <= 0:
        return np.inf
    z = 1.0 + xi * excesses / sigma
    if np.any(z <= 0):
        return np.inf    # with these parameters one of the observed values would be impossible
    n = excesses.size
    if abs(xi) < 1e-8:
        return n * np.log(sigma) + excesses.sum() / sigma    # xi = 0: exponential distribution
    return n * np.log(sigma) + (1.0 + 1.0 / xi) * np.log(z).sum()


def fit_gpd(excesses):
    """Fit the GPD to the excesses by maximum likelihood.

    The optimizer is started from several points and the best result is kept. scipy's own fit
    is run as well, as an independent check, and is used if it turns out better.
    """
    mean = excesses.mean()
    variance = excesses.var()

    # Method-of-moments estimates make a good first starting point.
    if variance > 0:
        xi_start = 0.5 * (1.0 - mean * mean / variance)
        sigma_start = 0.5 * mean * (mean * mean / variance + 1.0)
    else:
        xi_start = 0.0
        sigma_start = mean
    starts = [
        (sigma_start, xi_start),
        (mean, 0.0),
        (mean, 0.1),
        (mean, -0.1),
        (1.5 * mean, -0.3),
        (0.7 * mean, 0.3),
    ]

    best = None
    for start in starts:
        if not np.isfinite(gpd_nll(np.array(start), excesses)):
            continue
        result = minimize(gpd_nll, x0=np.array(start), args=(excesses,), method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    scipy_xi, _, scipy_sigma = genpareto.fit(excesses, floc=0)
    scipy_nll = gpd_nll(np.array([scipy_sigma, scipy_xi]), excesses)

    if best is None or scipy_nll < best.fun - 1e-9:
        sigma, xi, nll, source = float(scipy_sigma), float(scipy_xi), float(scipy_nll), "scipy"
    else:
        sigma, xi, nll, source = float(best.x[0]), float(best.x[1]), float(best.fun), "multistart"
    return {"sigma": sigma, "xi": xi, "nll": nll, "nll_scipy": float(scipy_nll), "source": source}


def hessian(func, point, relative_step=1e-4):
    """Second derivatives of func at point, by central finite differences.

    For a negative log-likelihood at its minimum, the inverse of this matrix is the usual
    large-sample estimate of the covariance matrix of the fitted parameters.
    """
    n = len(point)
    step = relative_step * np.maximum(1.0, np.abs(point))
    result = np.empty((n, n))
    for a in range(n):
        for b in range(a, n):
            step_a = np.eye(n)[a] * step[a]    # a small move in parameter a only
            step_b = np.eye(n)[b] * step[b]
            result[a, b] = result[b, a] = (
                func(point + step_a + step_b) - func(point + step_a - step_b)
                - func(point - step_a + step_b) + func(point - step_a - step_b)
            ) / (4 * step[a] * step[b])
    return result


def return_level(return_period, threshold, rate, sigma, xi):
    """Wind speed with a 1-in-return_period chance of being exceeded in any one year."""
    # m is close to the number of events expected in return_period years
    m = rate / (-np.log(1.0 - 1.0 / return_period))
    if abs(xi) < 1e-8:
        return threshold + sigma * np.log(m)
    return threshold + sigma / xi * (m**xi - 1.0)


def return_level_grad(return_period, rate, sigma, xi):
    """How fast the return level changes with rate, sigma and xi (its three derivatives).

    The delta method uses these to turn the uncertainty in the three fitted values into the
    uncertainty of the return level.
    """
    m = rate / (-np.log(1.0 - 1.0 / return_period))
    log_m = np.log(m)
    if abs(xi) < 1e-6:
        return np.array([sigma / rate, log_m, sigma * log_m * log_m / 2.0])
    m_xi = m**xi
    return np.array([
        sigma * m_xi / rate,
        (m_xi - 1.0) / xi,
        -sigma * (m_xi - 1.0) / xi**2 + sigma * m_xi * log_m / xi,
    ])


def pot_fit(values, percentile, separation, n_years, return_periods):
    """Run the whole POT method on one daily series, for one threshold and separation rule.

    Returns a dictionary with the threshold, the event counts, the fitted GPD and, for every
    return period N, the estimate (rpN_ms), its standard error and its 95% interval.
    """
    threshold = float(np.percentile(values, percentile))
    _, peaks, n_days_above = decluster(values, threshold, separation)
    excesses = (peaks - threshold).astype(np.float64)
    n_events = int(excesses.size)
    rate = n_events / n_years
    out = {
        "threshold_pct": percentile, "separation_days": separation, "threshold_ms": threshold,
        "n_exceedance_days": n_days_above, "n_events": n_events, "events_per_year": rate,
        "mean_days_per_event": n_days_above / n_events if n_events else np.nan,
        "max_peak_ms": float(peaks.max()) if n_events else np.nan,
    }

    fit = fit_gpd(excesses)
    sigma, xi = fit["sigma"], fit["xi"]

    # The covariance of (sigma, xi) is estimated by the inverse of the Hessian of the negative
    # log-likelihood. It can only be used if the Hessian is positive definite (every eigenvalue
    # above zero).
    params = np.array([sigma, xi])
    hess = hessian(lambda p: gpd_nll(p, excesses), params)
    hessian_ok = bool(np.all(np.isfinite(hess)) and np.all(np.linalg.eigvalsh(hess) > 0))
    cov = np.linalg.inv(hess) if hessian_ok else np.full((2, 2), np.nan)

    out.update({
        "gpd_scale": sigma, "gpd_shape_xi": xi,
        "se_scale": float(np.sqrt(cov[0, 0])), "se_shape_xi": float(np.sqrt(cov[1, 1])),
        "nll": fit["nll"], "nll_minus_scipy": fit["nll"] - fit["nll_scipy"],
        "fit_source": fit["source"],
        "hessian_ok": hessian_ok,
        # with a negative shape the fitted distribution has a highest possible value
        "upper_bound_ms": float(threshold - sigma / xi) if xi < 0 else np.inf,
        # Kolmogorov-Smirnov test of the fit: a small p-value means the GPD does not match the
        # excesses. It is a rough check, because the GPD was fitted to the same excesses.
        "ks_pvalue": float(kstest(excesses, genpareto(xi, 0, sigma).cdf).pvalue),
    })

    for N in return_periods:
        estimate = return_level(N, threshold, rate, sigma, xi)
        grad = return_level_grad(N, rate, sigma, xi)
        # Delta method. grad holds the derivatives for (rate, sigma, xi). The first term comes
        # from the event rate: the number of events is treated as a Poisson count, so the
        # variance of the rate is rate / n_years. The second term comes from sigma and xi. The
        # rate is treated as independent of sigma and xi, so the two terms add.
        variance = grad[0] ** 2 * rate / n_years + grad[1:] @ cov @ grad[1:]
        se = float(np.sqrt(variance)) if hessian_ok and variance >= 0 else np.nan
        out[f"rp{N}_ms"] = estimate
        out[f"rp{N}_se_ms"] = se
        out[f"rp{N}_lower_ms"] = estimate - Z95 * se
        out[f"rp{N}_upper_ms"] = estimate + Z95 * se

    flags = []
    if n_events < 30:
        flags.append("fewer than 30 events")
    if xi <= -0.5:
        flags.append("shape <= -0.5 (delta-method SE unreliable)")
    if not hessian_ok:
        flags.append("information matrix not positive definite")
    if xi < 0 and out["upper_bound_ms"] < out["max_peak_ms"] * 1.02:
        flags.append("fitted upper bound within 2% of the largest observed peak")
    out["flags"] = "; ".join(flags)
    return out
