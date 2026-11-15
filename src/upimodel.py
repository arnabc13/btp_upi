"""Model of how shop payments are made (UPI, card, bank transfer or cash) when merchants pay a fee on UPI.

Shop payments are divided into cells by merchant group and payment size. In each cell the
payer chooses in two steps: digital or cash, then UPI, card or bank transfer. A fee makes UPI
less attractive, so some payments move to other methods. The model works out the money the
fee raises and the economic loss it causes:

    Loss = convenience lost - fee collected - UPI cost saved + extra cost of cash
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from common import load_json

SEGMENTS = ["P2PM", "STD", "FLAT", "CAP"]
EDGES = np.array([1, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500,
                  10000, 15000, 25000, 50000, 75000, 100000, 200000, 500000, 1e6, 1e7], float)

# Starting shares of cash, card and bank transfer for each merchant group at a few payment
# sizes; shares in between are found by interpolation on a log scale
ANCH_X = np.array([100, 1000, 5000, 50000])
BASE = {  # cash, card, bank transfer; UPI gets the rest
    "STD":  {"cash": [0.42, 0.36, 0.30, 0.22], "card": [0.02, 0.06, 0.15, 0.22], "a2a": [0.005, 0.02, 0.03, 0.08]},
    "P2PM": {"cash": [0.45, 0.40, 0.35, 0.35], "card": [0.00, 0.01, 0.02, 0.02], "a2a": [0.002, 0.01, 0.02, 0.03]},
    "FLAT": {"cash": [0.20, 0.15, 0.10, 0.08], "card": [0.03, 0.10, 0.15, 0.15], "a2a": [0.05, 0.10, 0.15, 0.20]},
    "CAP":  {"cash": [0.00, 0.00, 0.00, 0.00], "card": [0.00, 0.00, 0.00, 0.00], "a2a": [0.30, 0.30, 0.35, 0.40]},
}

DEFAULTS = dict(
    eta=0.40,  # share of UPI payments lost per 1% of fee at a Rs 2,000 payment (normal merchants)
    theta=0.75,  # 1: merchants look at the % rate; 0: at the rupee amount
    mu=0.70,  # how similar card and bank transfer are to UPI (set in s04 from the customer survey)
    m_P2PM=2.0, m_STD=1.0, m_FLAT=1.0, m_CAP=0.7,  # how much more each merchant group reacts to the fee
    e0_cash=0.75,  # cost of one cash payment to the country, Rs (printing and moving cash)
    eps_cash=0.0025,  # plus this share of the payment (tax lost on unreported cash sales)
    c_upi=0.50,  # cost of processing one UPI payment, Rs
    mit_a=1.0, mit_b=0.003,  # merchant's own cost of handling cash: Rs a + b x payment
    share_gt2000=dict(P2PM=0.07, STD=0.585, FLAT=0.22, CAP=0.125),  # share of shop money above Rs 2,000 in each merchant group
    share_le2000=dict(P2PM=0.35, STD=0.595, FLAT=0.05, CAP=0.005),  # share of shop payments up to Rs 2,000 in each merchant group
    p2m_annual_txn_crore=None,  # set from the forecast
)


def mixture_bins(th=None, edges=EDGES):
    """Share of payments, share of money and average payment in each size group, from the fitted curve."""
    if th is None:
        p = load_json("ticket_distribution")["params"]
        th = [p["w"], p["mu1"], p["s1"], p["mu2"], p["s2"]]
    w, m1, s1, m2, s2 = th
    lo, hi = np.log(edges[:-1]), np.log(edges[1:])

    def F(z, m, s):
        return norm.cdf((z - m) / s)

    def E(z, m, s):
        return np.exp(m + s * s / 2) * norm.cdf((z - m - s * s) / s)
    cnt = w * (F(hi, m1, s1) - F(lo, m1, s1)) + (1 - w) * (F(hi, m2, s2) - F(lo, m2, s2))
    cnt[0] += w * F(lo[0], m1, s1) + (1 - w) * F(lo[0], m2, s2)
    cnt[-1] += w * (1 - F(hi[-1], m1, s1)) + (1 - w) * (1 - F(hi[-1], m2, s2))
    val = w * (E(hi, m1, s1) - E(lo, m1, s1)) + (1 - w) * (E(hi, m2, s2) - E(lo, m2, s2))
    val[0] += w * E(lo[0], m1, s1) + (1 - w) * E(lo[0], m2, s2)
    mean_all = w * np.exp(m1 + s1 ** 2 / 2) + (1 - w) * np.exp(m2 + s2 ** 2 / 2)
    val[-1] += mean_all - val.sum()
    x = val / cnt
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "cnt_share": cnt, "val_share": val / mean_all,
                         "x": x})


def base_shares(seg, x, overrides=None):
    lx = np.log(np.clip(x, ANCH_X[0], ANCH_X[-1]))
    la = np.log(ANCH_X)
    tab = BASE[seg] if overrides is None or seg not in overrides else overrides[seg]
    sc = np.interp(lx, la, tab["cash"])
    sk = np.interp(lx, la, tab["card"])
    sa = np.interp(lx, la, tab["a2a"])
    su = 1 - sc - sk - sa
    return su, sc, sk, sa


class Economy:
    """All the cells of the model for one set of input values."""

    def __init__(self, params=None, th=None, annual_p2m_txn_crore=18600.0, base_overrides=None):
        P = dict(DEFAULTS)
        if params:
            P.update(params)
        self.P = P
        bins = mixture_bins(th)
        self.bins = bins
        B = len(bins)
        self.B = B
        x = bins["x"].values
        gt = bins["lo"].values >= 2000
        rows = []
        N_upi_total = annual_p2m_txn_crore  # crore UPI shop payments a year
        for g in SEGMENTS:
            sh_le = P["share_le2000"][g]
            sh_gt = P["share_gt2000"][g]
            seg_share = np.where(gt, sh_gt, sh_le)
            upi_cnt = N_upi_total * bins["cnt_share"].values * seg_share
            su, sc, sk, sa = base_shares(g, x, base_overrides)
            N = upi_cnt / su  # all shop payments in each cell (any method), crore a year
            m = P[f"m_{g}"]
            a = su / (1 - sc) if g != "CAP" else su / (1 - 1e-12)
            beta_raw = P["eta"] * m * (x / 2000.0) ** (1 - P["theta"])
            denom = sc * a + (1 - a) / P["mu"]
            beta = beta_raw / denom
            for b in range(B):
                rows.append(dict(seg=g, b=b, x=x[b], lo=bins["lo"].values[b], hi=bins["hi"].values[b],
                                 N=N[b], upi0=upi_cnt[b], su0=su[b], sc0=sc[b], sk0=sk[b], sa0=sa[b],
                                 a=a[b], beta=beta[b]))
        self.cells = pd.DataFrame(rows)
        self.mit = self.P["mit_a"] / self.cells["x"].values + self.P["mit_b"]

    # Money raised and economic loss for a set of fee rates
    def evaluate(self, rate):
        """Money raised, economic loss and payments lost (Rs crore a year) for a set of fee rates, one per cell."""
        c = self.cells
        P = self.P
        r = np.asarray(rate, float)
        rp = 100.0 * r
        mu = P["mu"]
        a, scz, beta, x, N = c["a"].values, c["sc0"].values, c["beta"].values, c["x"].values, c["N"].values
        S = 1 - a + a * np.exp(-beta * rp / mu)
        Dn = (1 - scz) * S ** mu + scz
        PD = (1 - scz) * S ** mu / Dn
        su = PD * a * np.exp(-beta * rp / mu) / S
        sc = scz / Dn
        lam = beta * 100.0 / x
        dcs = np.where(rp > 0, np.log(Dn) / lam, 0.0)
        rev = su * r * x
        dext = (su - c["su0"].values) * P["c_upi"] + (sc - scz) * (P["e0_cash"] + P["eps_cash"] * x)
        dW = dcs + rev - dext
        out = {
            "revenue": float(np.sum(N * rev)),
            "dwl": float(-np.sum(N * dW)),
            "dcs": float(np.sum(N * dcs)),
            "upi_txn": float(np.sum(N * su)),
            "upi_txn0": float(np.sum(c["upi0"].values)),
            "cash_txn_added": float(np.sum(N * (sc - scz))),
            "upi_value_lost": float(np.sum(N * (c["su0"].values - su) * x)),
            "cash_ext_cost_added": float(np.sum(N * (sc - scz) * (P["e0_cash"] + P["eps_cash"] * x))),
            "merchant_fee_paid": float(np.sum(N * rev)),
        }
        seg = c["seg"].values
        for g in SEGMENTS:
            m = seg == g
            out[f"upi_loss_pct_{g}"] = float(100 * (1 - np.sum(N[m] * su[m]) / np.sum(c["upi0"].values[m])))
            out[f"revenue_{g}"] = float(np.sum(N[m] * rev[m]))
        charged = r > 0
        out["upi_loss_pct_charged"] = float(100 * (1 - np.sum(N[charged] * su[charged]) / max(np.sum(c["upi0"].values[charged]), 1e-12)))
        out["dwl_per_rupee"] = out["dwl"] / out["revenue"] if out["revenue"] > 0 else np.nan
        out["_su"] = su
        out["_sc"] = sc
        return out

    # Fee designs
    def policy_rates(self, kind, **kw):
        c = self.cells
        x, lo, seg = c["x"].values, c["lo"].values, c["seg"].values
        r = np.zeros(len(c))
        above = lo >= kw.get("threshold", 2000)
        if kind == "govt2026":
            std = (seg == "STD") & above
            r[std] = np.minimum(0.004 * x[std], 300.0) / x[std]
            fl = (seg == "FLAT") & above
            r[fl] = 5.0 / x[fl]
            cp = (seg == "CAP")
            r[cp] = np.minimum(0.0002 * x[cp], 300.0) / x[cp]
        elif kind == "flat_all":  # one rate on all shop payments (small merchants free)
            m = seg != "P2PM"
            r[m] = kw["rate"]
        elif kind == "threshold":  # one rate above a payment size, with an optional cap
            m = above & (seg != "P2PM") if kw.get("exempt_p2pm", True) else above
            r[m] = kw["rate"]
            if kw.get("cap"):
                r[m] = np.minimum(kw["rate"] * x[m], kw["cap"]) / x[m]
        elif kind == "per_txn":  # fixed rupee fee on each payment above a payment size
            m = above & (seg != "P2PM")
            r[m] = kw["fee"] / x[m]
        return r


def eta_semi_elasticity(beta, sc0, a, mu):
    return beta * (sc0 * a + (1 - a) / mu)
