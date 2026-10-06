"""Checks of pot.py, and of the quicker routines in 09_pot_bootstrap.py, on cases where the
right answer is known.

Usage:
    python test_pot.py      (about 1 minute; prints PASS or FAIL for each check)
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import genextreme, genpareto

import pot

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# 1. Grouping days into events, on a short hand-made series.
series = np.array([0, 5, 6, 0, 7, 0, 0, 8, 0, 0, 0, 9], float)
for separation, expected in ((1, [6, 7, 8, 9]), (2, [7, 8, 9]), (3, [8, 9])):
    peak_days, peaks, n_above = pot.decluster(series, 1.0, separation)
    check(f"decluster r={separation}", list(peaks) == expected and n_above == 5,
          f"peaks {list(peaks)}")
peak_days, peaks, _ = pot.decluster(np.array([0, 3, 9, 9, 2, 0.0]), 1.0, 1)
check("decluster keeps one peak per event, first day on ties",
      list(peak_days) == [2] and list(peaks) == [9])
peak_days, peaks, n_above = pot.decluster(np.zeros(5), 1.0, 1)
check("decluster with no exceedances", peak_days.size == 0 and n_above == 0)

# 2. The GPD fit recovers parameters that are known, and the standard errors from the Hessian
#    are close to the textbook large-sample formulas:
#    se(xi) = (1 + xi) / sqrt(n),  se(sigma) = sigma * sqrt(2 * (1 + xi) / n).
for true_sigma, true_xi in ((2.0, 0.1), (1.5, -0.2), (1.0, 0.0)):
    sample = genpareto.rvs(true_xi, 0, true_sigma, size=20000, random_state=rng)
    fit = pot.fit_gpd(sample)
    hess = pot.hessian(lambda p: pot.gpd_nll(p, sample), np.array([fit["sigma"], fit["xi"]]))
    cov = np.linalg.inv(hess)
    n = sample.size
    se_xi_formula = (1 + fit["xi"]) / np.sqrt(n)
    se_sigma_formula = fit["sigma"] * np.sqrt(2 * (1 + fit["xi"]) / n)
    check(f"GPD recovery sigma={true_sigma} xi={true_xi}",
          abs(fit["sigma"] - true_sigma) < 4 * se_sigma_formula
          and abs(fit["xi"] - true_xi) < 4 * se_xi_formula,
          f"got sigma {fit['sigma']:.3f} xi {fit['xi']:.3f}")
    # The Hessian varies more from sample to sample when the tail has an upper bound (xi < 0).
    # In a separate trial of 12 samples at xi = -0.2 (not part of this script) the ratio ran
    # from 0.95 to 1.06, and the Hessian standard error matched the spread of the fitted xi
    # better than the formula did. So the tolerance is wider there.
    tolerance = 0.05 if true_xi >= 0 else 0.10
    ratio_xi = np.sqrt(cov[1, 1]) / se_xi_formula
    ratio_sigma = np.sqrt(cov[0, 0]) / se_sigma_formula
    check("  Hessian SE vs asymptotic formula",
          abs(ratio_xi - 1) < tolerance and abs(ratio_sigma - 1) < tolerance,
          f"ratio xi {ratio_xi:.3f} sigma {ratio_sigma:.3f}")
    check("  multistart no worse than scipy", fit["nll"] <= fit["nll_scipy"] + 1e-6,
          f"diff {fit['nll'] - fit['nll_scipy']:.2e}")

# 3. The POT return level equals the GEV quantile of the annual maximum that the same model
#    implies (both are the level with a 1-in-N chance of being exceeded in a year). A Poisson
#    number of GPD events per year gives an annual maximum that follows a GEV with the
#    location and scale below, and scipy's GEV shape is -xi.
for rate, sigma, xi, threshold in ((3.0, 1.5, 0.08, 10.0), (8.0, 1.2, -0.15, 9.0),
                                   (2.0, 2.0, 0.0, 11.0)):
    for N in (10, 20, 50, 100, 500):
        got = pot.return_level(N, threshold, rate, sigma, xi)
        if xi == 0:
            expected = (threshold + sigma * np.log(rate)) - sigma * np.log(-np.log(1 - 1 / N))
        else:
            gev_loc = threshold + sigma * (rate**xi - 1) / xi
            gev_scale = sigma * rate**xi
            expected = genextreme.isf(1 / N, -xi, gev_loc, gev_scale)
        check(f"return level lam={rate} xi={xi} N={N}", abs(got - expected) < 1e-9,
              f"{got:.6f} vs {expected:.6f}")

# 4. The derivative formulas equal numerical derivatives, including close to xi = 0.
for rate, sigma, xi in ((3.0, 1.5, 0.08), (8.0, 1.2, -0.15), (2.0, 2.0, 1e-9), (4.0, 1.0, 5e-7)):
    grad = pot.return_level_grad(100, rate, sigma, xi)
    eps = 1e-6
    params = np.array([rate, sigma, xi])
    grad_numeric = []
    for direction in np.eye(3):
        above = pot.return_level(100, 0, *(params + direction * eps))
        below = pot.return_level(100, 0, *(params - direction * eps))
        grad_numeric.append((above - below) / (2 * eps))
    grad_numeric = np.array(grad_numeric)
    check(f"gradient xi={xi}", np.allclose(grad, grad_numeric, rtol=1e-4, atol=1e-5),
          f"{np.round(grad, 4)} vs {np.round(grad_numeric, 4)}")

# 5. The delta-method standard error is within 25% of the spread of the estimate over many
#    simulated 45-year records.
n_years, true_rate, true_sigma, true_xi, threshold = 45, 3.0, 1.5, 0.05, 10.0
n_records = 1500
estimates = {20: [], 50: [], 100: []}
standard_errors = {20: [], 50: [], 100: []}
n_covered = {20: 0, 50: 0, 100: 0}
truth = {N: pot.return_level(N, threshold, true_rate, true_sigma, true_xi) for N in estimates}
for b in range(n_records):
    n = rng.poisson(true_rate * n_years)
    sample = genpareto.rvs(true_xi, 0, true_sigma, size=n, random_state=rng)
    fit = pot.fit_gpd(sample)
    hess = pot.hessian(lambda p: pot.gpd_nll(p, sample), np.array([fit["sigma"], fit["xi"]]))
    cov = np.linalg.inv(hess)
    rate = n / n_years
    for N in estimates:
        estimate = pot.return_level(N, threshold, rate, fit["sigma"], fit["xi"])
        grad = pot.return_level_grad(N, rate, fit["sigma"], fit["xi"])
        se = np.sqrt(grad[0] ** 2 * rate / n_years + grad[1:] @ cov @ grad[1:])
        estimates[N].append(estimate)
        standard_errors[N].append(se)
        n_covered[N] += (estimate - 1.96 * se <= truth[N] <= estimate + 1.96 * se)
for N in estimates:
    spread = np.std(estimates[N], ddof=1)
    median_se = np.median(standard_errors[N])
    check(f"delta SE vs Monte Carlo SD, N={N}", 0.75 < median_se / spread < 1.25,
          f"median SE {median_se:.3f}, MC SD {spread:.3f}, ratio {median_se / spread:.2f}, "
          f"95% interval coverage {n_covered[N] / n_records:.1%}")

# 6. The whole method, on a made-up daily series with storms that last three days.
n_days = 16437
background = rng.gamma(6, 0.9, size=n_days)
storm_starts = rng.random(n_days) < 3.0 / 365.25
series = background.copy()
for day in np.flatnonzero(storm_starts):
    storm_peak = 9 + genpareto.rvs(0.05, 0, 1.5, random_state=rng)
    for offset, share in enumerate((0.85, 1.0, 0.8)):
        if day + offset < n_days:
            series[day + offset] = max(series[day + offset], storm_peak * share)
all_periods = [10, 20, 25, 50, 100, 200, 500]
n_events = {}
for percentile in (95, 97, 98, 99):
    for separation in (1, 2, 3):
        fit = pot.pot_fit(series, percentile, separation, 45, all_periods)
        n_events[(percentile, separation)] = fit["n_events"]
        levels = [fit[f"rp{N}_ms"] for N in all_periods]
        check(f"end-to-end q={percentile} r={separation}",
              fit["hessian_ok"] and np.all(np.diff(levels) > 0)
              and fit["n_events"] <= fit["n_exceedance_days"],
              f"u {fit['threshold_ms']:.2f} events {fit['n_events']} "
              f"({fit['events_per_year']:.1f}/yr) xi {fit['gpd_shape_xi']:+.3f} "
              f"rp100 {fit['rp100_ms']:.2f} +/- {1.96 * fit['rp100_se_ms']:.2f}")
check("more separation never gives more events",
      all(n_events[(percentile, 1)] >= n_events[(percentile, 2)] >= n_events[(percentile, 3)]
          for percentile in (95, 97, 98, 99)))
check("higher threshold never gives more exceedance days",
      all(pot.pot_fit(series, lower, 1, 45, [100])["n_exceedance_days"]
          >= pot.pot_fit(series, higher, 1, 45, [100])["n_exceedance_days"]
          for lower, higher in ((95, 97), (97, 98), (98, 99))))
days_above_99 = pot.pot_fit(series, 99, 1, 45, [100])["n_exceedance_days"]
check("99th percentile leaves ~1% of days above", abs(days_above_99 - 0.01 * n_days) <= 1,
      f"{days_above_99} of {n_days}")


# 7. The quicker routines used by the bootstraps give the same answers as the full ones.
#    09_pot_bootstrap.py cannot be imported the usual way because its name starts with a digit.
spec = importlib.util.spec_from_file_location(
    "boot", Path(__file__).resolve().parent / "09_pot_bootstrap.py")
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)
worst_nll_gap = 0.0
worst_level_gap = 0.0
cases = ((1.2, -0.15, 150), (1.0, 0.29, 143), (0.9, 0.0, 300), (1.3, -0.1, 650), (1.1, 0.2, 140),
         (0.8, -0.17, 500))
for sigma, xi, mean_events in cases:
    for b in range(60):
        sample = genpareto.rvs(xi, 0, sigma, size=rng.poisson(mean_events), random_state=rng)
        fast_sigma, fast_xi = boot.fit_fast(sample, (sigma, xi))
        fit = pot.fit_gpd(sample)
        fast_nll = pot.gpd_nll(np.array([fast_sigma, fast_xi]), sample)
        worst_nll_gap = max(worst_nll_gap, fast_nll - fit["nll"])
        rate = len(sample) / 45
        fast_level = pot.return_level(100, 10, rate, fast_sigma, fast_xi)
        full_level = pot.return_level(100, 10, rate, fit["sigma"], fit["xi"])
        worst_level_gap = max(worst_level_gap, abs(fast_level - full_level))
check("fast GPD fit equals the full fit (360 samples)",
      worst_nll_gap < 1e-6 and worst_level_gap < 1e-3,
      f"worst likelihood gap {worst_nll_gap:.1e}, "
      f"worst 100-year difference {worst_level_gap:.1e} m/s")
other_series = rng.gamma(4, 2, size=5000)
check("fast event grouping equals pot.decluster",
      all(np.array_equal(boot.decluster_fast(other_series, 12.0, separation),
                         pot.decluster(other_series, 12.0, separation)[1])
          for separation in (1, 2, 3)))

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
