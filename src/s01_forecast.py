
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from common import PROC, save_json, save_fig, style, C, SEED, fy_of, TAB
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

upi = pd.read_csv(PROC / "upi_monthly.csv", parse_dates=["month"]).set_index("month")
upi = upi.asfreq("MS")
y_all = np.log(upi["volume_mn"] / upi["days"])
y = y_all.loc["2019-04-01":]
LOCK = pd.Series(0.0, index=y_all.index); LOCK.loc["2020-04-01":"2020-05-01"] = 1.0  # 2020 lockdown months
END_FCST = pd.Timestamp("2031-03-01")
H_MAX = 24


# The six forecasting models
def fit_sarima(ytr, h):
    best = None
    for p in (0, 1, 2):
        for q in (0, 1):
            for P in (0, 1):
                for Q in (0, 1):
                    try:
                        m = SARIMAX(ytr, exog=LOCK.loc[ytr.index], order=(p, 1, q),
                                    seasonal_order=(P, 1, Q, 12), trend="n").fit(disp=False)
                        k = p + q + P + Q + 1
                        n = len(ytr)
                        aicc = m.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
                        if best is None or aicc < best[0]:
                            best = (aicc, m, (p, q, P, Q))
                    except Exception:
                        pass
    f = best[1].get_forecast(h, exog=np.zeros((h, 1))).predicted_mean.values
    return f, {"order": best[2]}


def fit_ets(ytr, h):
    m = ExponentialSmoothing(ytr, trend="add", damped_trend=True, seasonal="add",
                             seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
    return m.forecast(h).values, {"phi": float(m.params["damping_trend"])}


def seasonal_dummies(idx):
    D = np.zeros((len(idx), 11))
    for i, t in enumerate(idx):
        if t.month > 1:
            D[i, t.month - 2] = 1.0
    return D


def fit_growth_decay(ytr, h):
    """Growth model: yearly growth falls towards a long-run level, g = g_inf + (g0 - g_inf) * exp(-k * years)."""
    g = (ytr - ytr.shift(12)).dropna()
    g = g.loc["2019-04-01":]
    s = (g.index - pd.Timestamp("2019-04-01")).days.values / 365.25

    def resid(th):
        ginf, g0, k = th
        return g.values - (ginf + (g0 - ginf) * np.exp(-k * s))
    sol = least_squares(resid, x0=[0.10, 1.2, 0.5], bounds=([0.0, 0, 0.01], [0.5, 3.0, 5.0]),
                        loss="soft_l1", f_scale=0.05)
    ginf, g0, k = sol.x
    # each month = same month last year + growth
    hist = list(ytr.values)
    idx = pd.date_range(ytr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    out = []
    for t in idx:
        st = (t - pd.Timestamp("2019-04-01")).days / 365.25
        gt = ginf + (g0 - ginf) * np.exp(-k * st)
        val = hist[-12] + gt
        hist.append(val)
        out.append(val)
    return np.array(out), {"g_inf": ginf, "g0": g0, "kappa": k}


def fit_recent_decay(ytr, h, window=36):
    """Growth model fitted on the last 36 months only: g = g0 * exp(-k * years)."""
    g = (ytr - ytr.shift(12)).dropna().iloc[-window:]
    s = (g.index - g.index[-1]).days.values / 365.25

    def resid(th):
        ga, k = th
        return g.values - ga * np.exp(-k * s)
    sol = least_squares(resid, x0=[g.values[-1], 0.3], bounds=([0.0, -1.0], [2.0, 3.0]),
                        loss="soft_l1", f_scale=0.03)
    ga, k = sol.x
    hist = list(ytr.values)
    idx = pd.date_range(ytr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    out = []
    for t in idx:
        st = (t - g.index[-1]).days / 365.25
        val = hist[-12] + ga * np.exp(-k * st)
        hist.append(val)
        out.append(val)
    return np.array(out), {"g_last": ga, "kappa": k, "half_life_years": np.log(2) / k if k > 0 else np.inf}


def fit_richards(ytr, h):
    """S-shaped (logistic) curve that levels off at a maximum, with a monthly pattern."""
    s = (ytr.index - pd.Timestamp("2019-04-01")).days.values / 365.25
    D = seasonal_dummies(ytr.index)

    def model(th, s, D):
        lK, lQ, r, lnu = th[:4]
        sea = D @ th[4:]
        nu = np.exp(lnu)
        return lK - (1 / nu) * np.log1p(np.exp(lQ - r * s)) + sea

    def resid(th):
        return ytr.values - model(th, s, D)
    best = None
    for lK0 in (np.log(1500), np.log(3000), np.log(6000)):
        for lnu0 in (-1.0, 0.0, 1.0):
            x0 = np.r_[lK0, 8.0, 1.0, lnu0, np.zeros(11)]
            try:
                sol = least_squares(resid, x0=x0, max_nfev=20000)
                if best is None or sol.cost < best.cost:
                    best = sol
            except Exception:
                pass
    idx = pd.date_range(ytr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    sf = (idx - pd.Timestamp("2019-04-01")).days.values / 365.25
    f = model(best.x, sf, seasonal_dummies(idx))
    return f, {"K_daily_mn": float(np.exp(best.x[0])), "r": float(best.x[2]),
               "nu": float(np.exp(best.x[3]))}


def lgb_features(yser):
    df = pd.DataFrame({"y": yser})
    df["g12"] = df["y"] - df["y"].shift(12)
    for L in (1, 2, 3, 6, 12):
        df[f"g12_l{L}"] = df["g12"].shift(L)
    df["m1"] = df["y"] - df["y"].shift(1)
    df["m1_l1"] = df["m1"].shift(1)
    df["month"] = df.index.month
    df["trend"] = np.arange(len(df))
    return df


def fit_lgb(ytr, h):
    """LightGBM: predicts future growth from past growth, with one model for each number of months ahead."""
    df = lgb_features(ytr)
    feats = [c for c in df.columns if c not in ("y",)]
    hist = list(ytr.values)
    idx = pd.date_range(ytr.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    preds = []
    for k in range(1, h + 1):
        tgt = (df["y"].shift(-k) - df["y"].shift(-k + 12)).rename("tgt")
        d = pd.concat([df[feats], tgt], axis=1).dropna()
        d = d.loc["2020-06-01":]
        if len(d) < 18 or k > H_MAX:
            preds.append(np.nan)
            continue
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                min_child_samples=6, subsample=0.9, subsample_freq=1,
                                colsample_bytree=0.9, reg_lambda=1.0, verbose=-1,
                                random_state=SEED)
        mdl.fit(d[feats].values, d["tgt"].values)
        x_last = df[feats].iloc[[-1]].values
        preds.append(float(mdl.predict(x_last)[0]))
    out = []
    for k, t in enumerate(idx, 1):
        base = hist[len(ytr) - 12 + k - 1] if k <= 12 else out[k - 13]
        out.append(base + preds[k - 1])
    return np.array(out), {}


MODELS = {"SARIMA": fit_sarima, "ETS": fit_ets, "GrowthDecay": fit_growth_decay,
          "RecentDecay": fit_recent_decay, "Richards": fit_richards, "LightGBM": fit_lgb}

# Test each model from 11 past dates (every 3 months, March 2022 to September 2024)
origins = pd.date_range("2022-03-01", "2024-09-01", freq="3MS")
cv_rows = []
for o in origins:
    ytr = y.loc[:o]
    yte = y.loc[o + pd.offsets.MonthBegin():o + pd.offsets.MonthBegin(H_MAX)]
    for name, fn in MODELS.items():
        f, _ = fn(ytr, H_MAX)
        for k in range(min(H_MAX, len(yte))):
            cv_rows.append({"origin": o, "model": name, "h": k + 1,
                            "err": yte.values[k] - f[k]})
    print(f"  tested from {o:%b %Y}")
cv = pd.DataFrame(cv_rows)

# Average percentage error of each model, 1-6, 7-12 and 13-24 months ahead
cv["ape"] = np.abs(np.exp(cv["err"]) - 1)
summary = []
for name, g in cv.groupby("model"):
    row = {"model": name}
    for lo, hi in ((1, 6), (7, 12), (13, 24)):
        gg = g[(g.h >= lo) & (g.h <= hi)]
        row[f"MAPE_{lo}_{hi}"] = 100 * gg["ape"].mean()
        row[f"bias_{lo}_{hi}"] = 100 * (np.exp(gg["err"]) - 1).mean()
    row["RMSE_log_all"] = np.sqrt((g["err"] ** 2).mean())
    gr = g[g.origin >= pd.Timestamp("2023-09-01")]
    row["MAPE_recent_13_24"] = 100 * gr[(gr.h >= 13)]["ape"].mean()
    row["bias_recent_13_24"] = 100 * (np.exp(gr[(gr.h >= 13)]["err"]) - 1).mean()
    summary.append(row)
summary = pd.DataFrame(summary).sort_values("MAPE_recent_13_24")

# Test whether the best model is really more accurate than the others (Diebold-Mariano test)
from scipy import stats
best_model = summary.iloc[0]["model"]
dm = {}
for name in MODELS:
    if name == best_model:
        continue
    a = cv[(cv.model == name) & (cv.h == 12)].set_index("origin")["err"] ** 2
    b = cv[(cv.model == best_model) & (cv.h == 12)].set_index("origin")["err"] ** 2
    d = (a - b).dropna()
    n = len(d)
    gam = [np.cov(d.values[l:], d.values[:n - l])[0, 1] if l else d.var(ddof=0) for l in range(min(4, n - 1))]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    stat = d.mean() / np.sqrt(max(var, 1e-12))
    dm[name] = {"DM_stat": float(stat), "p_value": float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))),
                "n_origins": n}

dm24 = {}
for name in MODELS:
    if name == "RecentDecay":
        continue
    a = cv[(cv.model == name) & (cv.h >= 13)].groupby("origin")["err"].apply(lambda e: (e ** 2).mean())
    b = cv[(cv.model == "RecentDecay") & (cv.h >= 13)].groupby("origin")["err"].apply(lambda e: (e ** 2).mean())
    d = (a - b).dropna()
    d = d[d.index >= pd.Timestamp("2023-09-01")]
    n = len(d)
    if n >= 3:
        stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        dm24[name] = {"mean_loss_diff": float(d.mean()), "t": float(stat),
                      "p_one_sided": float(1 - stats.t.cdf(stat, df=n - 1)), "n": n}
# Weights for averaging the models; recent tests count more (Stock and Watson 2004)
DELTA = 0.75
age = {o: (origins[-1] - o).days / 91.3 for o in origins}
cv["dw"] = cv["origin"].map(lambda o: DELTA ** age[o])
mse_h = cv.groupby(["model", "h"]).apply(lambda g: np.sum(g["dw"] * g["err"] ** 2) / np.sum(g["dw"])).unstack(0)
mse_h_equal = cv.groupby(["model", "h"])["err"].apply(lambda e: (e ** 2).mean()).unstack(0)
excluded = [m for m in MODELS if summary.set_index("model").loc[m, "MAPE_13_24"] > 1.6 * summary["MAPE_13_24"].min()]
use = [m for m in MODELS if m not in excluded]
w_h = (1 / mse_h[use]).div((1 / mse_h[use]).sum(axis=1), axis=0)
w_avg = w_h.mean()

ens_err = []
for (o, h), g in cv[cv.model.isin(use)].groupby(["origin", "h"]):
    e = (g.set_index("model")["err"] * w_h.loc[h]).sum()
    ens_err.append({"origin": o, "h": h, "err": e})
ens_err = pd.DataFrame(ens_err)
ens_mape = {f"{lo}_{hi}": 100 * np.abs(np.exp(ens_err[(ens_err.h >= lo) & (ens_err.h <= hi)]["err"]) - 1).mean()
            for lo, hi in ((1, 6), (7, 12), (13, 24))}

ens_err["dw"] = ens_err["origin"].map(lambda o: DELTA ** age[o])
sd_h = ens_err.groupby("h").apply(lambda g: np.sqrt(np.sum(g["dw"] * g["err"] ** 2) / np.sum(g["dw"])))
bias_h = ens_err.groupby("h").apply(lambda g: np.sum(g["dw"] * g["err"]) / np.sum(g["dw"]))
hh = sd_h.index.values.astype(float)
coef = np.polyfit(np.log(hh), np.log(sd_h.values), 1)
b_exp, a_sc = coef[0], np.exp(coef[1])

# Final forecast
H_FINAL = (END_FCST.year - y.index[-1].year) * 12 + END_FCST.month - y.index[-1].month
fc = {}
params = {}
for name in use:
    f, p = MODELS[name](y, H_FINAL)
    fc[name] = f
    params[name] = p
idx_f = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=H_FINAL, freq="MS")
w_long = w_h.loc[13:24].mean()
w_long = w_long / w_long.sum()
W = np.array([w_h.loc[h].values if h <= H_MAX else w_long.values for h in range(1, H_FINAL + 1)])
F = np.column_stack([fc[m] for m in use])
Wm = np.where(np.isnan(F), 0.0, W)
Wm = Wm / Wm.sum(axis=1, keepdims=True)
y_ens = (Wm * np.nan_to_num(F)).sum(axis=1)
h_arr = np.arange(1, H_FINAL + 1)
sd_f = a_sc * h_arr ** b_exp

# Simulate 4,000 possible future paths to get the likely range: the speed at which
# growth slows, the current growth rate and the month-to-month noise are varied
G_all = (y - y.shift(12)).dropna()
roll = []
for o in pd.date_range("2023-09-01", y.index[-1], freq="3MS"):
    _, pr = fit_recent_decay(y.loc[:o], 1)
    roll.append(pr["kappa"])
roll = np.array(roll)
k_mu, k_sd = float(roll.mean()), float(2 * roll.std(ddof=1))
_, p4 = fit_recent_decay(y, 1)
g36 = G_all.iloc[-36:]
s36 = (g36.index - g36.index[-1]).days.values / 365.25
res36 = g36.values - p4["g_last"] * np.exp(-p4["kappa"] * s36)
sig_m = float(res36.std(ddof=2))
N_PATHS = 4000
kap = np.clip(rng.normal(k_mu, k_sd, N_PATHS), 0.08, 0.5)
ga = p4["g_last"] + rng.choice(res36, N_PATHS) * 0.5
st_f = (idx_f - y.index[-1]).days.values / 365.25
paths_log = np.zeros((N_PATHS, H_FINAL))
base = np.r_[y.values[-12:]]
for i in range(N_PATHS):
    hist = list(base)
    noise = rng.normal(0, sig_m, H_FINAL)
    for k in range(H_FINAL):
        v = hist[-12] + ga[i] * np.exp(-kap[i] * st_f[k]) + noise[k]
        hist.append(v)
    paths_log[i] = hist[12:]
# The main forecast is the growth model fitted on the last 3 years; the average of
# the models is kept as an upper case in which growth stops slowing
y_comb = y_ens.copy()
y_prim = fc["RecentDecay"] if "RecentDecay" in fc else fit_recent_decay(y, H_FINAL)[0]
paths_log = paths_log - np.median(paths_log, axis=0)[None, :] + y_prim[None, :]
y_ens = y_prim
phi = float(np.corrcoef(paths_log[:, 11], paths_log[:, 23])[0, 1])
days = idx_f.days_in_month.values
paths_vol = np.exp(paths_log) * days[None, :]  # monthly transactions, million
sd_sim = paths_log.std(axis=0)

out = pd.DataFrame({"month": idx_f, "ens_volume_mn": np.exp(y_ens) * days, "comb_volume_mn": np.exp(y_comb) * days,
                    "p05": np.quantile(paths_vol, 0.05, axis=0), "p25": np.quantile(paths_vol, 0.25, axis=0),
                    "p50": np.quantile(paths_vol, 0.50, axis=0), "p75": np.quantile(paths_vol, 0.75, axis=0),
                    "p95": np.quantile(paths_vol, 0.95, axis=0)})
for m in use:
    out[m] = np.exp(fc[m]) * days

# Yearly totals (2026-27 includes the months already observed)
obs_fy27 = upi.loc["2026-04-01":, "volume_mn"].sum()
fy_idx = np.array([fy_of(t) for t in idx_f])
annual = {}
for fy in range(2027, 2032):
    msk = fy_idx == fy
    tot = paths_vol[:, msk].sum(axis=1) + (obs_fy27 if fy == 2027 else 0.0)
    annual[fy] = {"p05": np.quantile(tot, .05) / 10, "p50": np.quantile(tot, .5) / 10,
                  "p95": np.quantile(tot, .95) / 10}  # crore transactions
for fy in range(2027, 2032):
    msk = fy_idx == fy
    annual[fy]["comb"] = (np.sum(np.exp(y_comb[msk]) * days[msk]) + (obs_fy27 if fy == 2027 else 0.0)) / 10
annual_df = pd.DataFrame(annual).T
annual_df.index.name = "FY"
fy_hist = upi.groupby("fy")["volume_mn"].sum() / 10
annual_df["yoy_median_%"] = 100 * (annual_df["p50"] / annual_df["p50"].shift(1) - 1)
annual_df.loc[2027, "yoy_median_%"] = 100 * (annual_df.loc[2027, "p50"] / fy_hist.loc[2026] - 1)
np.save(PROC / "forecast_annual_paths_crore.npy",
        np.column_stack([(paths_vol[:, fy_idx == fy].sum(axis=1) + (obs_fy27 if fy == 2027 else 0)) / 10
                         for fy in range(2027, 2032)]))

# How often the actual values fell inside the likely range in the tests
rec = cv[(cv.model == "RecentDecay") & (cv.origin >= pd.Timestamp("2023-09-01"))]
cov80 = float(np.mean(np.abs(rec["err"]) <= 1.2816 * sd_sim[rec["h"].values - 1]))
cov95 = float(np.mean(np.abs(rec["err"]) <= 1.96 * sd_sim[rec["h"].values - 1]))
ens_mape_recent = 100 * float(np.abs(np.exp(rec[rec.h >= 13]["err"]) - 1).mean())

save_json({
    "cv_summary": summary.to_dict(orient="records"), "dm_vs_best_h12": dm, "dm_recent_vs_M4_h13_24": dm24, "best_model": best_model,
    "ensemble_members": use, "excluded": excluded, "ensemble_weights_avg": w_avg.to_dict(), "ensemble_weights_long": w_long.to_dict(),
    "ensemble_cv_mape": ens_mape, "sd_scale_a": a_sc, "sd_scale_b": b_exp, "ar1_phi_errors": phi,
    "coverage80_recent": cov80, "coverage95_recent": cov95, "ensemble_mape_recent_13_24": ens_mape_recent,
    "kappa_rolling": roll.tolist(), "kappa_mu": k_mu, "kappa_sd": k_sd, "monthly_noise_sd": sig_m,
    "recent_decay_full": p4, "discount_delta": DELTA, "params_full_sample": params,
    "annual_crore": annual_df.reset_index().to_dict(orient="records"),
    "fy26_observed_crore": float(fy_hist.loc[2026]), "fy27_first7_obs_crore": float(obs_fy27 / 10),
    "n_cv_origins": len(origins),
}, "forecast")

# Plain names of the models
PLAIN = {"SARIMA": "SARIMA", "ETS": "Exponential smoothing", "LightGBM": "LightGBM",
         "GrowthDecay": "Growth model (all years)", "Richards": "S-shaped curve",
         "RecentDecay": "Growth model (last 3 years)"}
# Graphs
import matplotlib.dates as mdates
style()
# Monthly forecast and yearly growth rate
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
h = upi.loc["2019-04-01":, "volume_mn"] / 1e3
ax[0].plot(h.index, h.values, color=C["ink"], lw=1.2, label="Actual")
ax[0].fill_between(out["month"], out["p05"] / 1e3, out["p95"] / 1e3, color=C["blue"], alpha=0.2, lw=0, label="Likely range")
ax[0].plot(out["month"], out["p50"] / 1e3, color=C["blue"], lw=1.4, label="Forecast")
ax[0].set_ylabel("UPI transactions per month (billion)")
ax[0].set_title("(a) Monthly UPI transactions")
ax[0].legend(loc="upper left")
dd = upi["volume_mn"] / upi["days"]
gr = (100 * (dd / dd.shift(12) - 1)).loc["2021-04-01":]
ax[1].plot(gr.index, gr.values, color=C["ink"], lw=1.0, label="Actual")
tt = pd.date_range(gr.index[-36], "2031-03-01", freq="MS")
ss = (tt - upi.index[-1]).days.values / 365.25
ax[1].plot(tt, 100 * (np.exp(p4["g_last"] * np.exp(-p4["kappa"] * ss)) - 1), color=C["orange"], ls="--", label="Growth model")
ax[1].set_ylabel("Growth over last year (%)")
ax[1].set_title("(b) Yearly growth rate")
ax[1].set_ylim(0, 120)
for a in ax:
    a.xaxis.set_major_locator(mdates.YearLocator(2))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax[1].legend()
save_fig(fig, "simple_forecast")

# Forecast error of each model by months ahead
COLS = {"SARIMA": C["muted"], "ETS": C["teal"], "LightGBM": C["green"], "GrowthDecay": C["orange"],
        "Richards": C["purple"], "RecentDecay": C["blue"]}
mape = (100 * cv["ape"]).groupby([cv["model"], cv["h"]]).mean().unstack(0)
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for m in PLAIN:
    ax.plot(mape.index, mape[m], color=COLS[m], lw=2.4 if m == "RecentDecay" else 1.1, label=PLAIN[m])
ax.set_xlabel("Months ahead")
ax.set_ylabel("Average error (%)")
ax.set_title("Error grows further ahead")
ax.legend(ncol=1, fontsize=6.5, loc="upper left")
ax.set_ylim(0, 70)
save_fig(fig, "simple_errors")

# Error of each model on the recent tests (from September 2023), 1-2 years ahead
rec_err = summary.set_index("model")["MAPE_recent_13_24"].reindex(list(PLAIN)).sort_values()
fig, ax = plt.subplots(figsize=(3.6, 2.8))
bars = ax.barh([PLAIN[m] for m in rec_err.index][::-1], rec_err.values[::-1],
               color=[C["blue"] if m == "RecentDecay" else C["muted"] for m in rec_err.index][::-1], height=0.6)
for bar, v in zip(bars, rec_err.values[::-1]):
    ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.5)
ax.set_xlabel("Average error, 1-2 years ahead (%)")
ax.set_title("Recent tests (from September 2023)")
ax.set_xlim(0, 50)
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="y", visible=False)
save_fig(fig, "simple_errors_recent")

# Print the results shown in the graphs
print("\nAVERAGE FORECAST ERROR BY MONTHS AHEAD (%)")
print(f"  {'Model':30s} " + " ".join(f"{h:>4d}" for h in (6, 12, 18, 24)))
for m in PLAIN:
    print(f"  {PLAIN[m]:30s} " + " ".join(f"{mape.loc[h, m]:4.0f}" for h in (6, 12, 18, 24)))
print("\nAVERAGE FORECAST ERROR 1-2 YEARS AHEAD, RECENT TESTS FROM SEPTEMBER 2023 (%)")
for m, v in rec_err.items():
    print(f"  {PLAIN[m]:30s} {v:5.1f}")
print(f"  Chosen model: {PLAIN[best_model]}; growth halves every {np.log(2) / p4['kappa']:.1f} years")
print("\nFORECAST OF YEARLY UPI TRANSACTIONS (crore)")
print(f"  {'Year':8s} {'Low':>8s} {'Expected':>9s} {'High':>8s} {'Growth':>7s}")
for fy, r in annual_df.iterrows():
    print(f"  {fy - 1}-{str(fy)[2:]:3s} {r['p05']:8,.0f} {r['p50']:9,.0f} {r['p95']:8,.0f} {r['yoy_median_%']:6.1f}%")