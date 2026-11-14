
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import PROC, save_json, save_fig, style, C, SEED

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

psi = pd.read_csv(PROC / "rbi_psi.csv", parse_dates=["month"]).set_index("month").asfreq("MS")
DONORS = ["debit_pos", "debit_oth", "credit_pos", "credit_oth", "ppi_card", "imps", "neft",
          "aeps_ft", "nach_cr", "nach_dr", "netc", "bhim_aadhaar", "cts", "atm_cash"]
TREATED = "ppi_wallet"
PRE = ("2020-07-01", "2023-03-01")
POST = ("2023-04-01", "2024-01-01")


def panel(kind):
    out = {}
    for u in [TREATED] + DONORS:
        v, val = psi[f"{u}_vol_lakh"], psi[f"{u}_val_cr"]
        if kind == "vol":
            out[u] = np.log(v)
        elif kind == "val":
            out[u] = np.log(val)
        else:
            out[u] = np.log(val / v)
    return pd.DataFrame(out).loc[PRE[0]:POST[1]].dropna(axis=1)


def synth(Y, treated, pre_mask):
    """Choose the weights of the comparison group so that it is as close as possible to wallets before April 2023."""
    donors = [c for c in Y.columns if c != treated]
    Yd = Y - Y[pre_mask].mean()
    X0, x1 = Yd.loc[pre_mask, donors].values, Yd.loc[pre_mask, treated].values
    w = cp.Variable(len(donors), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x1 - X0 @ w)), [cp.sum(w) == 1])
    prob.solve(solver=cp.CLARABEL)
    wv = np.clip(w.value, 0, None)
    wv /= wv.sum()
    synth_path = Yd[donors].values @ wv
    gap = Yd[treated].values - synth_path
    return pd.Series(gap, index=Y.index), pd.Series(wv, index=donors)


def rmspe(g, m):
    return np.sqrt(np.mean(g[m] ** 2))


def sc_analysis(kind, pre=PRE, post=POST, label=""):
    Y = panel(kind).loc[pre[0]:post[1]]
    pre_mask = (Y.index >= pre[0]) & (Y.index <= pre[1])
    post_mask = (Y.index >= post[0]) & (Y.index <= post[1])
    gap, w = synth(Y, TREATED, pre_mask)
    att = gap[post_mask].mean()
    ratio_t = rmspe(gap, post_mask) / rmspe(gap, pre_mask)
    placebo = {}
    for d in [c for c in Y.columns if c != TREATED]:
        Yp = Y.drop(columns=[TREATED])
        gp, _ = synth(Yp, d, pre_mask)
        placebo[d] = {"att": gp[post_mask].mean(), "pre_rmspe": rmspe(gp, pre_mask),
                      "ratio": rmspe(gp, post_mask) / rmspe(gp, pre_mask), "gap": gp}
    ratios = np.array([p["ratio"] for p in placebo.values()])
    atts = np.array([p["att"] for p in placebo.values()])
    p_ratio = (1 + np.sum(ratios >= ratio_t)) / (1 + len(ratios))
    p_att = (1 + np.sum(np.abs(atts) >= abs(att))) / (1 + len(atts))
    # Check 1: other payment methods treated as if charged (those that fitted well before April 2023)
    pre_t = rmspe(gap, pre_mask)
    good = [d for d, p in placebo.items() if p["pre_rmspe"] <= 2 * pre_t]
    atts_g = np.array([placebo[d]["att"] for d in good])
    p_att_g = (1 + np.sum(np.abs(atts_g) >= abs(att))) / (1 + len(atts_g))
    # Likely range of the effect (Chernozhukov, Wuthrich and Zhu 2021)
    grid = np.linspace(-0.4, 0.4, 801)
    keep = []
    for tau in grid:
        Ya = Y.copy()
        Ya.loc[post_mask, TREATED] = Ya.loc[post_mask, TREATED] - tau
        ga, _ = synth(Ya, TREATED, pre_mask)
        u = ga.values
        T, Tp = len(u), post_mask.sum()
        stat = np.mean(np.abs(u[-Tp:]))
        stats_perm = [np.mean(np.abs(np.roll(u, k)[-Tp:])) for k in range(T)]
        pval = np.mean(np.array(stats_perm) >= stat)
        if pval > 0.10:
            keep.append(tau)
    ci90 = (min(keep), max(keep)) if keep else (np.nan, np.nan)
    return {"kind": kind, "att": att, "pre_rmspe": pre_t, "post_pre_ratio": ratio_t,
            "p_ratio_rank": p_ratio, "p_att_rank": p_att, "p_att_rank_goodfit": p_att_g,
            "n_goodfit": len(good), "ci90_conformal": ci90,
            "weights": w[w > 0.01].round(3).to_dict(), "gap": gap, "placebo": placebo,
            "pre_mask": pre_mask, "post_mask": post_mask}


def its(kind):
    Y = panel(kind)[TREATED].loc[PRE[0]:POST[1]]
    t = np.arange(len(Y))
    post = (Y.index >= POST[0]).astype(float)
    t0 = np.argmax(post)
    X = pd.DataFrame({"const": 1.0, "t": t, "post": post, "post_t": post * (t - t0)}, index=Y.index)
    for m in range(2, 13):
        X[f"m{m}"] = (Y.index.month == m).astype(float)
    res = sm.OLS(Y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    # average change after April 2023
    k = post.sum()
    L = np.zeros(X.shape[1])
    L[list(X.columns).index("post")] = 1.0
    L[list(X.columns).index("post_t")] = (k - 1) / 2
    eff = float(L @ res.params.values)
    se = float(np.sqrt(L @ res.cov_params().values @ L))
    return {"avg_post_effect": eff, "se": se, "ci95": (eff - 1.96 * se, eff + 1.96 * se),
            "level_shift": float(res.params["post"]), "slope_change": float(res.params["post_t"]),
            "p_level": float(res.pvalues["post"]), "p_slope": float(res.pvalues["post_t"]), "r2": float(res.rsquared)}


results = {}
for kind in ("vol", "val", "tick"):
    r = sc_analysis(kind)
    results[kind] = r
    results[kind]["its"] = its(kind)

# Check 2: a fake start date of April 2022, when nothing happened
pt = {k: sc_analysis(k, pre=("2020-07-01", "2022-03-01"), post=("2022-04-01", "2023-01-01")) for k in ("vol", "val", "tick")}

# Upper limit on how strongly people react to a fee, allowing for wallets becoming usable
# on UPI at the same time and for only part of wallet money being charged
lo_val = results["val"]["ci90_conformal"][0]
bounds = {}
for BI in (0.0, 0.05, 0.10):
    for phi in (0.10, 0.20, 0.40):
        beta_fee_lo = lo_val - BI
        bounds[f"BI={BI:.2f},phi={phi:.2f}"] = float(-beta_fee_lo / (1.1 * phi))

save_json({
    "window": {"pre": PRE, "post": POST, "donors": DONORS},
    "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("gap", "placebo", "pre_mask", "post_mask")}
                for k, v in results.items()},
    "placebo_time": {k: {kk: vv for kk, vv in v.items() if kk not in ("gap", "placebo", "pre_mask", "post_mask")}
                     for k, v in pt.items()},
    "semi_elasticity_upper_bound_per_pp": bounds,
}, "fee_response")

# Print the results shown in the graph
NAMES = {"vol": "Number of payments", "val": "Value of payments", "tick": "Average payment size"}
print("WALLET FEE STUDY: change in wallets after April 2023, compared with the comparison group")
for k in ("vol", "val", "tick"):
    print(f"  {NAMES[k]:22s} {100 * results[k]['att']:+5.1f}%")

# Graph
style()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
ttl = {"vol": "(a) Number of payments", "val": "(b) Value of payments", "tick": "(c) Average payment size"}
for ax, kind in zip(axes, ("vol", "val", "tick")):
    r = results[kind]
    for d, p in r["placebo"].items():
        if p["pre_rmspe"] <= 2 * r["pre_rmspe"]:
            ax.plot(p["gap"].index, 100 * p["gap"].values, color=C["muted"], lw=0.5, alpha=0.6)
    ax.plot(r["gap"].index, 100 * r["gap"].values, color=C["red"], lw=1.6, label="Wallets")
    ax.axvline(pd.Timestamp("2023-04-01"), color=C["ink"], lw=0.8, ls="--")
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.set_title(ttl[kind])
    ax.set_ylim(-45, 45)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].set_ylabel("Difference from comparison group (%)")
axes[0].plot([], [], color=C["muted"], lw=0.8, label="Other payment methods")
axes[0].legend(loc="lower left", fontsize=7)
save_fig(fig, "simple_synth")