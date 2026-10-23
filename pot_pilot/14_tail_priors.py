"""Set the priors on the tail shape for the constrained-tail fits.

Writes the priors on the shape parameter, checks the sign conventions, and compares the priors
with the shapes fitted in steps 06 and 08.

Shape convention. Everything here uses xi, the textbook shape (Coles 2001): xi > 0 is a heavy
upper tail, xi = 0 is the Gumbel case, and xi < 0 is a tail with an upper bound. SciPy is not
consistent about this:
    scipy.stats.genextreme (GEV)    c = -xi
    scipy.stats.genpareto (GPD)     c = xi
so the GEV code has to flip the sign and the POT code does not. pot.py already works in xi.
Martins & Stedinger (2000) write the shape as kappa, Hosking's convention, which is the same
as genextreme's c. Their prior has a mean kappa of -0.10, which is xi = +0.10.

The GEV for annual maxima and the GPD for events above a threshold have the same shape in
theory, so each prior is used for both models.

The priors, all on xi:
    flat               uniform from -1 to 1. Used for the fits without a constrained tail.
                       The lower bound keeps the posterior proper.
    main               normal, mean -0.10, sd 0.12. Used for the constrained-tail fits.
    wider              normal, mean -0.10, sd 0.20. Sensitivity test.
    narrower           normal, mean -0.10, sd 0.06. Sensitivity test.
    martins_stedinger  beta(6, 9) on kappa from -0.5 to 0.5. For comparison only, not used.
Location and log scale get flat priors in every version.

Where the numbers come from:
    mean -0.10   Gross, Heckert, Lechner & Simiu (1995), NISTIR 5531: annual maximum winds at
                 115 U.S. stations not dominated by hurricanes. The weighted mean tail
                 parameter is close to -0.1, and they take -0.2 as a working value.
    xi below 0   Storm intensity has a physical upper limit (Simiu, Heckert & Whalen 1998;
                 Emanuel's potential intensity for hurricanes), so the center sits below 0.
    sd 0.12      The width of the Martins & Stedinger prior (sd 0.122), which is the usual
                 strength for a prior meant to stop unstable small-sample fits.
    normal       No hard bounds on the shape, so a cell whose data clearly show a heavier
                 tail can still get one.

Reads:
    output/pilot_gev_parameters.csv (step 06)
    output/pilot_pot_results.csv (step 08)
Writes:
    output/tail_priors.csv
    output/plots/fig5_tail_priors.png
"""
import matplotlib

matplotlib.use("Agg")    # draw straight to files; no window is opened

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, genextreme, genpareto, norm, uniform

from paths import OUT, require

PLOTS = OUT / "plots"
POT_PERCENTILE = 98    # the provisional POT setting from step 08's tests
POT_SEPARATION = 2

require(OUT / "pilot_gev_parameters.csv", OUT / "pilot_pot_results.csv")

# 1. The sign conventions. A shape that should give a bounded tail must give one.
assert np.isfinite(genextreme(c=0.2).support()[1]), "genextreme: c = 0.2 should be bounded"
assert np.isinf(genextreme(c=-0.2).support()[1]), "genextreme: c = -0.2 should be unbounded"
assert np.isfinite(genpareto(c=-0.2).support()[1]), "genpareto: c = -0.2 should be bounded"
assert np.isinf(genpareto(c=0.2).support()[1]), "genpareto: c = 0.2 should be unbounded"

# 2. The priors, as SciPy distributions on xi.
# Martins & Stedinger: kappa + 0.5 follows beta(6, 9). Since xi = -kappa, xi + 0.5 follows
# beta(9, 6).
martins_stedinger_kappa = beta(6, 9, loc=-0.5, scale=1.0)
assert abs(martins_stedinger_kappa.mean() - (-0.10)) < 1e-9
assert abs(martins_stedinger_kappa.std() - 0.122) < 0.001    # the published sd

PRIORS = {
    "flat": dict(
        dist=uniform(loc=-1.0, scale=2.0), used_in="Fits without a constrained tail",
        distribution="Uniform(-1, 1)",
        source="",
        justification="Lets the data decide. The lower bound only keeps the posterior proper.",
    ),
    "main": dict(
        dist=norm(-0.10, 0.12), used_in="Constrained-tail fits",
        distribution="Normal(mean -0.10, sd 0.12)",
        source="Gross et al. 1995 (NISTIR 5531); Simiu, Heckert & Whalen 1998; "
               "Martins & Stedinger 2000",
        justification="Centered on the published weighted mean for U.S. annual maximum winds "
                      "(about -0.1), which fits storm intensity having an upper limit. Width "
                      "of the Martins & Stedinger prior. No hard bounds.",
    ),
    "wider": dict(
        dist=norm(-0.10, 0.20), used_in="Sensitivity test",
        distribution="Normal(mean -0.10, sd 0.20)",
        source="As for main",
        justification="A weaker pull, so heavy tails at hurricane-exposed cells stay likely.",
    ),
    "narrower": dict(
        dist=norm(-0.10, 0.06), used_in="Sensitivity test",
        distribution="Normal(mean -0.10, sd 0.06)",
        source="As for main",
        justification="Half the main width: what happens if the published estimates are "
                      "trusted strongly. Its range includes the NIST working value of -0.2.",
    ),
    "martins_stedinger": dict(
        dist=beta(9, 6, loc=-0.5, scale=1.0), used_in="Not used, for comparison",
        distribution="Beta(6, 9) on kappa over (-0.5, 0.5), which is Beta(9, 6) on xi",
        source="Martins & Stedinger 2000",
        justification="Built for floods and rainfall, which are heavy-tailed, so it is centered "
                      "on the wrong side of 0 for wind. Its shape bounds are hard limits.",
    ),
}

# 3. The pilot fits, as a check on the priors (not a source for them).
gev = pd.read_csv(OUT / "pilot_gev_parameters.csv")
assert np.allclose(gev.xi, -gev.shape_scipy_c, atol=1e-4), "pilot GEV xi should equal -c"
pot_fits = pd.read_csv(OUT / "pilot_pot_results.csv")
pot_fits = pot_fits[(pot_fits.threshold_pct == POT_PERCENTILE)
                    & (pot_fits.separation_days == POT_SEPARATION)]
assert len(gev) == 13 and len(pot_fits) == 13
gev_xi = gev.xi.to_numpy()
pot_xi = pot_fits.gpd_shape_xi.to_numpy()
print(f"pilot GEV xi:  mean {gev_xi.mean():+.3f}, range {gev_xi.min():+.3f} to {gev_xi.max():+.3f}")
print(f"pilot POT xi ({POT_PERCENTILE}th, {POT_SEPARATION}-day):  mean {pot_xi.mean():+.3f}, "
      f"range {pot_xi.min():+.3f} to {pot_xi.max():+.3f}")

rows = []
for name, prior in PRIORS.items():
    dist = prior["dist"]
    lower, upper = dist.ppf([0.025, 0.975])
    rows.append({
        "prior": name, "used_in": prior["used_in"], "distribution": prior["distribution"],
        "mean_xi": dist.mean(), "sd_xi": dist.std(),
        "lower_95_xi": lower, "upper_95_xi": upper,
        "prob_xi_above_0": dist.sf(0.0),
        "pilot_gev_cells_in_95": int(((gev_xi >= lower) & (gev_xi <= upper)).sum()),
        "pilot_pot_cells_in_95": int(((pot_xi >= lower) & (pot_xi <= upper)).sum()),
        "source": prior["source"], "justification": prior["justification"],
    })
table = pd.DataFrame(rows)
numbers = ["mean_xi", "sd_xi", "lower_95_xi", "upper_95_xi", "prob_xi_above_0"]
table[numbers] = table[numbers].round(3)
table.to_csv(OUT / "tail_priors.csv", index=False)
print(table.drop(columns=["source", "justification"]).to_string(index=False))
print(f"wrote {OUT / 'tail_priors.csv'}")

# 4. The plot: each prior's density, with the pilot fits marked underneath.
TEAL, BLUE, TAN, GREY = "#2F8F7F", "#205196", "#D9A341", "#333333"
MUTED_TEXT, AXIS_COLOR, ICE = "#5C5C5C", "#CCCCCC", "#E9ECEF"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
    "font.size": 10.5, "text.color": GREY, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": AXIS_COLOR,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": ICE, "axes.axisbelow": True, "legend.frameon": False,
})
lines = {
    "main": dict(color=GREY, linewidth=2.4, label="Main: normal, sd 0.12"),
    "wider": dict(color="#8A8A8A", linewidth=1.6, label="Wider: normal, sd 0.20"),
    "narrower": dict(color="#000000", linewidth=1.2, linestyle=(0, (4, 2)),
                     label="Narrower: normal, sd 0.06"),
    "flat": dict(color="#8A8A8A", linewidth=1.2, linestyle=":", label="Flat: uniform, -1 to 1"),
    "martins_stedinger": dict(color=TAN, linewidth=1.6, linestyle=(0, (6, 2)),
                              label="Martins & Stedinger (floods), for comparison"),
}
xi = np.linspace(-0.6, 0.6, 601)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.14)
for name, style in lines.items():
    ax.plot(xi, PRIORS[name]["dist"].pdf(xi), **style)
top = ax.get_ylim()[1]
ax.plot(gev_xi, np.full(13, -0.04 * top), "|", color=TEAL, markersize=14, markeredgewidth=2,
        label="Pilot GEV fits (13 cells)")
ax.plot(pot_xi, np.full(13, -0.10 * top), "|", color=BLUE, markersize=14, markeredgewidth=2,
        label=f"Pilot POT fits ({POT_PERCENTILE}th percentile, {POT_SEPARATION}-day)")
ax.axvline(0, color=AXIS_COLOR, linewidth=1)
ax.set_ylim(-0.14 * top, top)
ax.set_xlim(-0.6, 0.6)
ax.set_xlabel("Shape \u03be  (below 0: bounded tail, above 0: heavy tail)")
ax.set_ylabel("Prior density")
ax.legend(loc="upper right", fontsize=9)
fig.text(0.08, 0.96, "Priors on the tail shape", fontsize=15, fontweight="bold", va="top")
fig.text(0.08, 0.905, "All on the textbook shape \u03be. The GEV code uses SciPy's c = "
         "\u2212\u03be. The POT code uses \u03be directly.", fontsize=10.5, color=MUTED_TEXT,
         va="top")
PLOTS.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS / "fig5_tail_priors.png", dpi=200)
plt.close(fig)
print(f"wrote {PLOTS / 'fig5_tail_priors.png'}")
