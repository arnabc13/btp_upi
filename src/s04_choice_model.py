
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
from upimodel import Economy, SEGMENTS, BASE, DEFAULTS
from common import load_json, save_json, save_fig, style, C, SEED, PROC, TAB, anchors

rng = np.random.default_rng(SEED)
A = anchors().set_index("key")["value"]
fc = load_json("forecast")
fy27 = [r for r in fc["annual_crore"] if r["FY"] == 2027][0]
P2M_SHARE = 0.633
P2M_FY27 = fy27["p50"] * P2M_SHARE  # crore UPI shop payments in 2026-27

# 1. How similar card and bank transfer are to UPI
TARGET_DIV_CASH = A["consumer_alt_cash"] / (A["consumer_alt_cash"] + A["consumer_alt_credit"] + A["consumer_alt_debit"]
                                            + A["consumer_alt_neft"] + A["consumer_alt_avoid"])


def diversion_to_cash(mu, x=5000.0, overrides=None):
    from upimodel import base_shares
    su, sc, sk, sa = base_shares("STD", np.array([x]), overrides)
    a = su / (1 - sc)
    beta, rp = 1.0, 1e-4

    def shares(rp):
        S = 1 - a + a * np.exp(-beta * rp / mu)
        Dn = (1 - sc) * S ** mu + sc
        return (1 - sc) * S ** mu / Dn * a * np.exp(-beta * rp / mu) / S, sc / Dn
    u0, c0 = shares(0.0)
    u1, c1 = shares(rp)
    return float(((c1 - c0) / (u0 - u1))[0])


mu_star = brentq(lambda m: diversion_to_cash(m) - TARGET_DIV_CASH, 0.05, 1.0)

# 2. Fee sensitivity: combine the survey with the wallet result
fr = load_json("fee_response")
b_hat = fr["results"]["val"]["att"]
lo, hi = fr["results"]["val"]["ci90_conformal"]
se_b = (hi - lo) / (2 * 1.645)
M = 200_000
# Starting range: half from the survey (corrected for people over-stating how they will
# react), half from a wide range around 0.3
H = np.exp(rng.uniform(np.log(1.5), np.log(10.0), M))
unwilling = rng.uniform(0.78, 0.88, M)  # 83% of merchants would not accept the fee (with a margin for rounding)
eta_sp = -np.log(1 - unwilling / H) / 0.4
eta_diffuse = np.exp(rng.normal(np.log(0.3), 1.0, M))
eta_prior = np.where(rng.uniform(size=M) < 0.5, eta_sp, eta_diffuse)
bI = rng.uniform(0.0, 0.10, M)
phi = rng.uniform(0.10, 0.40, M)
b_model = bI - (1 - np.exp(-1.1 * eta_prior)) * phi
logw = norm.logpdf(b_hat, loc=b_model, scale=se_b)
w = np.exp(logw - logw.max())
w /= w.sum()
ess = 1 / np.sum(w ** 2)
post_idx = rng.choice(M, size=20000, replace=True, p=w)
eta_post = eta_prior[post_idx]
q = lambda v: np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
np.save(PROC / "eta_posterior.npy", eta_post)

# 3. Test the designs in 1,000 runs
mix_draws = np.load(PROC / "ticket_mixture_draws.npy")
N_DRAWS = 1000


def perturb_base(scale):
    out = {}
    for g, tab in BASE.items():
        out[g] = {k: list(np.clip(np.array(v) * scale[k], 0, 0.6)) for k, v in tab.items()}
    return out


def draw_params():
    sh = rng.dirichlet(np.array([0.07, 0.585, 0.22, 0.125]) * 60)
    p = dict(
        eta=float(rng.choice(eta_post)),
        theta=float(rng.uniform(0.5, 1.0)),
        m_P2PM=float(rng.uniform(1.0, 3.0)),
        e0_cash=float(np.exp(rng.normal(np.log(0.75), 0.5))),
        eps_cash=float(np.exp(rng.normal(np.log(0.0025), 0.5))),
        c_upi=float(rng.uniform(0.2, 1.0)),
        mit_b=float(rng.uniform(0.002, 0.006)),
        share_gt2000=dict(zip(SEGMENTS, sh)),
    )
    scale = {"cash": rng.uniform(0.8, 1.2), "card": rng.uniform(0.8, 1.2), "a2a": rng.uniform(0.8, 1.2)}
    ov = perturb_base(scale)
    p["mu"] = float(brentq(lambda m: diversion_to_cash(m, overrides=ov) - TARGET_DIV_CASH * rng.uniform(0.9, 1.1), 0.05, 1.0))
    th = mix_draws[rng.integers(len(mix_draws))]
    vol = P2M_FY27 * rng.uniform(fy27["p05"] / fy27["p50"], fy27["p95"] / fy27["p50"]) * rng.uniform(0.62, 0.645) / P2M_SHARE
    return p, ov, th, vol


def run_policies(E):
    out = {}
    gov = E.evaluate(E.policy_rates("govt2026"))
    out["GOV"] = gov
    for tag, rr in (("T05", 0.0005), ("T07", 0.0007), ("T30", 0.0030)):
        out[tag] = E.evaluate(E.policy_rates("threshold", rate=rr, threshold=2000))
    R = gov["revenue"]
    f = lambda rr: E.evaluate(E.policy_rates("flat_all", rate=rr))["revenue"] - R
    try:
        rr = brentq(f, 1e-6, 0.02)
        out["FLAT"] = E.evaluate(E.policy_rates("flat_all", rate=rr))
        out["FLAT"]["rate"] = rr
    except ValueError:
        out["FLAT"] = None
    g = lambda fee: E.evaluate(E.policy_rates("per_txn", fee=fee, threshold=2000))["revenue"] - R
    try:
        fee = brentq(g, 0.01, 500)
        out["PERTX"] = E.evaluate(E.policy_rates("per_txn", fee=fee, threshold=2000))
        out["PERTX"]["fee"] = fee
    except ValueError:
        out["PERTX"] = None
    # government plan with small merchants also charged
    r = E.policy_rates("govt2026")
    c = E.cells
    m = (c["seg"].values == "P2PM") & (c["lo"].values >= 2000)
    r2 = r.copy()
    r2[m] = np.minimum(0.004 * c["x"].values[m], 300) / c["x"].values[m]
    out["GOV+P2PM"] = E.evaluate(r2)
    # share of charged payments where the fee is higher than the merchant's cost of cash
    over = (r > E.mit + 1e-12) & (r > 0)
    out["GOV"]["mit_violation_share"] = float(np.sum(c["upi0"].values[over]) / np.sum(c["upi0"].values[r > 0]))
    return out


rows = []
central = None
for i in range(N_DRAWS + 1):
    if i == 0:  # the first run uses the middle values
        p = dict(eta=float(np.median(eta_post)), mu=mu_star)
        E = Economy(p, annual_p2m_txn_crore=P2M_FY27)
    else:
        p, ov, th, vol = draw_params()
        E = Economy(p, th=th, annual_p2m_txn_crore=vol, base_overrides=ov)
    res = run_policies(E)
    if i == 0:
        central = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if v else None) for k, v in res.items()}
        central["params"] = {k: v for k, v in E.P.items() if k != "p2m_annual_txn_crore"}
        continue
    for pol, o in res.items():
        if o is None:
            continue
        rec = {"draw": i, "policy": pol, **{k: v for k, v in o.items() if not k.startswith("_")}}
        rec.update({f"par_{k}": v for k, v in p.items() if not isinstance(v, dict)})
        rows.append(rec)
mc = pd.DataFrame(rows)
mc.to_csv(PROC / "mc_policy_eval.csv", index=False)

summ = mc.groupby("policy")[["revenue", "dwl", "dwl_per_rupee", "upi_loss_pct_charged", "cash_txn_added",
                              "upi_loss_pct_STD", "upi_loss_pct_P2PM"]].quantile([0.05, 0.5, 0.95]).unstack()
summ.to_csv(TAB / "mc_policy_summary.csv")

# compare designs within the same run
piv = mc.pivot_table(index="draw", columns="policy", values="dwl")
rv = mc.pivot_table(index="draw", columns="policy", values="revenue")
comp = {}
for alt in ("FLAT", "PERTX"):
    d = (piv[alt] - piv["GOV"]) / piv[alt]
    comp[f"GOV_vs_{alt}_dwl_saving_pct"] = (100 * d.quantile([0.05, 0.5, 0.95])).tolist()
    comp[f"P(GOV better than {alt})"] = float((piv["GOV"] < piv[alt]).mean())
# extra loss per extra rupee from charging small merchants
dR = rv["GOV+P2PM"] - rv["GOV"]
dD = piv["GOV+P2PM"] - piv["GOV"]
comp["P2PM_marginal_dwl_per_rupee_q"] = (dD / dR).quantile([0.05, 0.5, 0.95]).tolist()
comp["GOV_avg_dwl_per_rupee_q"] = (piv["GOV"] / rv["GOV"]).quantile([0.05, 0.5, 0.95]).tolist()
comp["P(P2PM marginal cost > GOV average)"] = float(((dD / dR) > (piv["GOV"] / rv["GOV"])).mean())
comp["P2PM_extra_revenue_q"] = dR.quantile([0.05, 0.5, 0.95]).tolist()
lossP = mc[mc.policy == "GOV+P2PM"].set_index("draw")["upi_loss_pct_P2PM"]
lossS = mc[mc.policy == "GOV"].set_index("draw")["upi_loss_pct_STD"]
# which inputs matter most (rank regression)
g = mc[mc.policy == "GOV"].copy()
pars = [c for c in g.columns if c.startswith("par_")]
from scipy.stats import rankdata
X = np.column_stack([rankdata(g[c]) for c in pars])
X = (X - X.mean(0)) / X.std(0)
src = {}
for yname in ("revenue", "dwl"):
    yv = rankdata(g[yname])
    yv = (yv - yv.mean()) / yv.std()
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(yv)), X], yv, rcond=None)
    src[yname] = dict(zip([p.replace("par_", "") for p in pars], coef[1:].round(3)))

save_json({"mu_star": mu_star, "diversion_target": TARGET_DIV_CASH, "mnl_diversion": diversion_to_cash(1.0),
           "eta_prior_q": q(eta_prior), "eta_post_q": q(eta_post), "eta_ess": ess,
           "b_hat": b_hat, "se_b": se_b, "p2m_fy27_crore": P2M_FY27, "central": central,
           "comparisons": comp, "srrc": src, "n_draws": N_DRAWS}, "choice_model")

# Share of charged payments lost with a 0.4% fee
lost = lambda e: 100 * (1 - np.exp(-0.4 * np.asarray(e)))
lp, lq = lost(q(eta_prior)), lost(q(eta_post))
est = [("Survey, taken at face value", 83.0, None, None),  # 83% of merchants would not accept a 0.4% fee
       ("Survey, adjusted for over-statement", lp[2], lp[0], lp[4]),
       ("Final estimate (survey + wallet data)", lq[2], lq[0], lq[4])]
g = central["GOV"]

print("CHARGED PAYMENTS LOST WITH A 0.4% FEE")
for n_, m_, a_, b_ in est:
    print(f"  {n_:40s} around {m_:.0f}%")
print("\nGOVERNMENT PLAN, 2026-27")
print(f"  Fee raised:            Rs {g['revenue']:,.0f} crore a year")
print(f"  Economic loss:         Rs {g['dwl']:,.0f} crore a year")

# Graph
PAL = {"teal": "#1b7f79", "amber": "#e09f3e", "plum": "#7b3f8c", "slate": "#4a5566", "coral": "#d1495b", "sand": "#d9cbb3"}
style()
plt.rcParams.update({"font.family": "sans-serif", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linestyle": "-"})
fig, ax = plt.subplots(figsize=(5.2, 2.6))
yy = np.arange(len(est))[::-1]
for y_, (n_, m_, a_, b_), col in zip(yy, est, (PAL["slate"], PAL["sand"], PAL["teal"])):
    if a_ is not None:
        ax.plot([a_, b_], [y_, y_], color=col, lw=6, solid_capstyle="round", alpha=0.5)
    ax.plot(m_, y_, "o", ms=11, color=col, markeredgecolor="black", markeredgewidth=0.6)
    ax.text(m_, y_ + 0.32, f"{m_:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_yticks(yy)
ax.set_yticklabels([e[0] for e in est], fontsize=8)
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, len(est) - 0.3)
ax.set_xlabel("Charged payments lost with a 0.4% fee (%)")
ax.set_title("How strongly merchants react to the fee")
ax.grid(axis="y", visible=False)
save_fig(fig, "fee_sensitivity")