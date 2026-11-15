
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
import matplotlib.pyplot as plt
from common import anchors, save_json, save_fig, style, C, SEED, PROC

rng = np.random.default_rng(SEED)
A = anchors().set_index("key")["value"]
TARGET = {"F500": A["tk_vol_le500"], "F2000": 1 - A["tk_vol_gt2000"],
          "V500": A["tk_val_le500"], "V2000": A["tk_val_le500"] + A["tk_val_500_2000"],
          "mean": A["p2m_mean_ticket"]}


def cdf(x, th):
    w, m1, s1, m2, s2 = th
    return w * norm.cdf((np.log(x) - m1) / s1) + (1 - w) * norm.cdf((np.log(x) - m2) / s2)


def pexp(a, th):
    """Value of payments up to size a, for the fitted curve."""
    w, m1, s1, m2, s2 = th
    e1 = np.exp(m1 + s1 ** 2 / 2) * norm.cdf((np.log(a) - m1 - s1 ** 2) / s1)
    e2 = np.exp(m2 + s2 ** 2 / 2) * norm.cdf((np.log(a) - m2 - s2 ** 2) / s2)
    return w * e1 + (1 - w) * e2


def mean(th):
    w, m1, s1, m2, s2 = th
    return w * np.exp(m1 + s1 ** 2 / 2) + (1 - w) * np.exp(m2 + s2 ** 2 / 2)


def resid(th, T):
    mu = mean(th)
    return np.array([
        (cdf(500, th) - T["F500"]) / 0.01,
        (cdf(2000, th) - T["F2000"]) / 0.01,
        (pexp(500, th) / mu - T["V500"]) / 0.01,
        (pexp(2000, th) / mu - T["V2000"]) / 0.01,
        (np.log(mu) - np.log(T["mean"])) / 0.01,
    ])


LB, UB = [0.5, 2, 0.2, 4.5, 0.3], [0.995, 6.5, 2.5, 11, 3]


def fit(T, x0s=None):
    best = None
    if x0s is None:
        x0s = [[0.9, m1, 1.0, m2, s2] for m1 in (4.0, 4.5, 5.0) for m2 in (5.5, 6.5, 7.8) for s2 in (0.8, 1.5, 2.0)]
    for x0 in x0s:
        if True:
            try:
                r = least_squares(resid, x0, args=(T,), bounds=(LB, UB))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best


best = fit(TARGET)
th = best.x

draws = []
for _ in range(300):
    T = dict(TARGET)
    for k in ("F500", "F2000", "V500"):
        T[k] = TARGET[k] + rng.uniform(-0.005, 0.005)
    T["V2000"] = T["V500"] + A["tk_val_500_2000"] + rng.uniform(-0.005, 0.005)
    T["mean"] = TARGET["mean"] * (1 + rng.uniform(-0.01, 0.01))
    r = fit(T, x0s=[th])
    if r is not None and np.abs(r.fun).max() < 0.5:
        draws.append(r.x)
draws = np.array(draws)
np.save(PROC / "ticket_mixture_draws.npy", draws)

# Quantities used by the later steps
mu = mean(th)
q75k_vol = 1 - cdf(75000, th)
q75k_val = 1 - pexp(75000, th) / mu
gt2000_val = 1 - pexp(2000, th) / mu
mean_gt2000 = (mu - pexp(2000, th)) / (1 - cdf(2000, th))
# Share of the 0.4% fee kept after the Rs 300 cap, for payments above Rs 2,000
xs = np.exp(np.linspace(np.log(2000), np.log(5e6), 20001))
w, m1, s1, m2, s2 = th
pdf = (w * norm.pdf((np.log(xs) - m1) / s1) / (s1 * xs) + (1 - w) * norm.pdf((np.log(xs) - m2) / s2) / (s2 * xs))
num = np.trapezoid(np.minimum(0.004 * xs, 300) * pdf, xs)
den = np.trapezoid(xs * pdf, xs)
cap_factor = num / (0.004 * den)
d_q75 = np.array([1 - pexp(75000, d) / mean(d) for d in draws])

save_json({"params": {"w": th[0], "mu1": th[1], "s1": th[2], "mu2": th[3], "s2": th[4]},
           "fit_max_abs_resid_pp": float(np.abs(best.fun).max()), "targets": TARGET,
           "n_draws": len(draws), "mean_ticket": mu, "share_value_ge75k": q75k_val,
           "share_value_ge75k_p05_p95": np.quantile(d_q75, [.05, .95]).tolist(),
           "share_vol_ge75k": q75k_vol, "share_value_gt2000": gt2000_val, "mean_ticket_gt2000": mean_gt2000,
           "cap_factor_0p4_300": cap_factor,
           "component_medians": [float(np.exp(th[1])), float(np.exp(th[3]))]}, "ticket_distribution")

# Share of payments and of money in each size group
edges = [0, 500, 2000, 10000, 75000, np.inf]
names = ["Up to 500", "500-2,000", "2,000-10,000", "10,000-75,000", "Above 75,000"]
cnt = np.diff([0.0] + [cdf(e, th) for e in edges[1:-1]] + [1.0])
val = np.diff([0.0] + [pexp(e, th) / mu for e in edges[1:-1]] + [1.0])

print("SHOP PAYMENTS BY SIZE (Rs)")
print(f"  {'Size group':15s} {'Share of payments':>18s} {'Share of money':>15s}")
for n_, c_, v_ in zip(names, cnt, val):
    print(f"  {n_:15s} {100 * c_:17.1f}% {100 * v_:14.1f}%")
print(f"  Largest gap between the fitted curve and NPCI's figures: {np.abs(best.fun).max():.2f} percentage points")

# Graph
PAL = {"teal": "#1b7f79", "amber": "#e09f3e", "plum": "#7b3f8c", "slate": "#4a5566", "coral": "#d1495b", "sand": "#d9cbb3"}
style()
plt.rcParams.update({"font.family": "sans-serif", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linestyle": "-"})
fig, ax = plt.subplots(figsize=(5.2, 3.0))
xg = np.arange(len(names))
ax.bar(xg - 0.2, 100 * cnt, width=0.4, color=PAL["teal"], label="Share of payments", edgecolor="white")
ax.bar(xg + 0.2, 100 * val, width=0.4, color=PAL["amber"], label="Share of money", edgecolor="white", hatch="//")
for x_, c_, v_ in zip(xg, cnt, val):
    ax.text(x_ - 0.2, 100 * c_ + 1.5, f"{100 * c_:.0f}%" if c_ >= 0.01 else f"{100 * c_:.1f}%", ha="center", fontsize=7)
    ax.text(x_ + 0.2, 100 * v_ + 1.5, f"{100 * v_:.0f}%", ha="center", fontsize=7)
ax.set_xticks(xg)
ax.set_xticklabels(names, fontsize=7)
ax.set_xlabel("Payment size (Rs)")
ax.set_ylabel("Share (%)")
ax.set_ylim(0, 100)
ax.set_title("Shop payments by size")
ax.legend(frameon=False)
save_fig(fig, "payment_sizes")
