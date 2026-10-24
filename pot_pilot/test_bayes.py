"""Checks of bayes.py on cases where the right answer is known.

Usage:
    python test_bayes.py      (about 1 minute; prints PASS or FAIL for each check)
"""
import warnings

import numpy as np
from scipy.stats import genextreme, genpareto

import bayes
import pot

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


# 1. The log-likelihoods equal SciPy's. genextreme takes c = -xi; genpareto takes c = xi.
maxima = genextreme.rvs(c=0.1, loc=20.0, scale=2.0, size=45, random_state=rng)
excesses = genpareto.rvs(c=0.1, scale=1.5, size=200, random_state=rng)
worst_gev = worst_gpd = 0.0
for xi in (-0.3, -0.05, -1e-7, 0.0, 1e-7, 0.05, 0.3):
    ours = bayes.gev_log_likelihood([20.0, np.log(2.0), xi], maxima)
    theirs = genextreme.logpdf(maxima, c=-xi, loc=20.0, scale=2.0).sum()
    if np.isfinite(theirs):
        worst_gev = max(worst_gev, abs(ours - theirs))
    ours = bayes.gpd_log_likelihood([np.log(1.5), xi], excesses)
    theirs = genpareto.logpdf(excesses, c=xi, scale=1.5).sum()
    if np.isfinite(theirs):
        worst_gpd = max(worst_gpd, abs(ours - theirs))
check("GEV log-likelihood matches scipy (c = -xi)", worst_gev < 1e-5,
      f"largest gap {worst_gev:.1e}")
check("GPD log-likelihood matches scipy (c = xi)", worst_gpd < 1e-5, f"largest gap {worst_gpd:.1e}")
check("GEV outside its support is impossible",
      bayes.gev_log_likelihood([20.0, np.log(2.0), -0.5], np.array([40.0])) == -np.inf)

# 2. The shape priors.
check("flat prior is zero outside -1 to 1",
      bayes.log_shape_prior(1.2, bayes.SHAPE_PRIORS["flat"]) == -np.inf
      and bayes.log_shape_prior(0.5, bayes.SHAPE_PRIORS["flat"]) == 0.0)
main = bayes.SHAPE_PRIORS["main"]
gap = bayes.log_shape_prior(0.14, main) - bayes.log_shape_prior(-0.10, main)
check("main prior: 2 sd from the mean is exp(-2) as likely", abs(gap + 2.0) < 1e-12)

# 3. Return levels equal scipy (GEV) and pot.return_level (POT), including xi = 0.
periods = [20, 50, 100]
worst = 0.0
for xi in (-0.2, 0.0, 1e-8, 0.2):
    ours = bayes.gev_return_levels(periods, [20.0], [2.0], [xi])[0]
    theirs = genextreme.isf(1.0 / np.array(periods), c=-xi, loc=20.0, scale=2.0)
    worst = max(worst, np.abs(ours - theirs).max())
    ours = bayes.pot_return_levels(periods, 10.0, [5.0], [1.5], [xi])[0]
    theirs = np.array([pot.return_level(p, 10.0, 5.0, 1.5, xi) for p in periods])
    worst = max(worst, np.abs(ours - theirs).max())
check("return levels match scipy and pot.py", worst < 1e-6, f"largest gap {worst:.1e}")

# 4. The sampler recovers a known posterior: a correlated two-dimensional normal.
target_mean = np.array([1.0, -2.0])
target_cov = np.array([[1.0, 0.8], [0.8, 2.0]])
precision = np.linalg.inv(target_cov)


def normal_log_density(p):
    d = p - target_mean
    return -0.5 * d @ precision @ d


draws, acceptance = bayes.sample(normal_log_density, target_mean, precision, seed=3)
flat = draws.reshape(-1, 2)
mean_gap = np.abs(flat.mean(axis=0) - target_mean).max()
cov_gap = np.abs(np.cov(flat.T) - target_cov).max()
check("sampler recovers the mean", mean_gap < 0.05, f"largest gap {mean_gap:.3f}")
check("sampler recovers the covariance", cov_gap < 0.15, f"largest gap {cov_gap:.3f}")
check("sampler acceptance is reasonable (0.15 to 0.5)", 0.15 < acceptance.mean() < 0.5,
      f"{acceptance.mean():.2f}")

# 5. The convergence checks.
iid = rng.standard_normal((4, 5000))
check("R-hat is close to 1 for well-mixed chains", bayes.split_rhat(iid) < 1.005,
      f"{bayes.split_rhat(iid):.4f}")
shifted = iid + np.array([0.0, 0.0, 0.0, 1.0])[:, None]
check("R-hat flags a chain stuck somewhere else", bayes.split_rhat(shifted) > 1.05,
      f"{bayes.split_rhat(shifted):.3f}")
ess_iid = bayes.effective_sample_size(iid)
check("ESS of independent draws is close to their number", 0.9 < ess_iid / iid.size < 1.1,
      f"{ess_iid:.0f} of {iid.size}")
phi = 0.9    # AR(1) chains, whose ESS is known: n (1 - phi) / (1 + phi)
ar = np.zeros((4, 20000))
noise = rng.standard_normal(ar.shape)
for t in range(1, ar.shape[1]):
    ar[:, t] = phi * ar[:, t - 1] + noise[:, t]
expected = ar.size * (1 - phi) / (1 + phi)
ess_ar = bayes.effective_sample_size(ar)
check("ESS of correlated draws matches theory", abs(ess_ar / expected - 1) < 0.2,
      f"{ess_ar:.0f}, expected about {expected:.0f}")

# 6. Known-answer recovery: with a long record, the flat-prior fit lands on the true values.
truth = np.array([25.0, np.log(3.0), 0.1])
long_record = genextreme.rvs(c=-truth[2], loc=truth[0], scale=np.exp(truth[1]), size=2000,
                             random_state=rng)
log_post = lambda p: bayes.gev_log_posterior(p, long_record, bayes.SHAPE_PRIORS["flat"])
mode, curvature = bayes.posterior_mode(log_post, [24.0, np.log(2.5), 0.0])
draws, _ = bayes.sample(log_post, mode, curvature, seed=5, n_warmup=2000, n_keep=3000)
medians = np.median(draws.reshape(-1, 3), axis=0)
gaps = np.abs(medians - truth)
check("GEV fit recovers known parameters (2,000 years)", (gaps < [0.3, 0.05, 0.04]).all(),
      f"gaps {np.round(gaps, 3)}")
true_peaks = genpareto.rvs(c=0.1, scale=2.0, size=3000, random_state=rng)
log_post = lambda p: bayes.gpd_log_posterior(p, true_peaks, bayes.SHAPE_PRIORS["flat"])
mode, curvature = bayes.posterior_mode(log_post, [0.5, 0.0])
draws, _ = bayes.sample(log_post, mode, curvature, seed=6, n_warmup=2000, n_keep=3000)
medians = np.median(draws.reshape(-1, 2), axis=0)
gaps = np.abs(medians - [np.log(2.0), 0.1])
check("GPD fit recovers known parameters (3,000 events)", (gaps < [0.05, 0.04]).all(),
      f"gaps {np.round(gaps, 3)}")

# 7. The main prior pulls a heavy-tailed fit toward lighter tails, as intended.
short_record = long_record[:45]
modes = {}
for name in ("flat", "main"):
    log_post = lambda p, pr=bayes.SHAPE_PRIORS[name]: bayes.gev_log_posterior(p, short_record, pr)
    modes[name], _ = bayes.posterior_mode(log_post, [25.0, np.log(3.0), 0.1])
check("main prior pulls the shape toward -0.10", modes["main"][2] < modes["flat"][2],
      f"flat {modes['flat'][2]:+.3f}, main {modes['main'][2]:+.3f}")

print("\nAll checks passed." if ok else "\nSome checks FAILED.")
raise SystemExit(0 if ok else 1)
