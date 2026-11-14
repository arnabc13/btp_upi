
import re
import numpy as np
import pandas as pd
from common import RAW, MAN, PROC, num, save_json, fy_of

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_month(s):
    s = str(s).strip().lower()
    m = re.match(r"([a-z]+)-(\d{4})", s)
    if not m:
        return None
    mon = MONTHS.get(m.group(1)[:3])
    return pd.Timestamp(year=int(m.group(2)), month=mon, day=1) if mon else None


# NPCI monthly UPI data
npci = pd.read_csv(RAW / "npci_upi_stats.csv", dtype=str)
rows = []
for _, r in npci.iterrows():
    # The first two years (2016-17 and 2017-18) are laid out differently in the file
    t = parse_month(r.iloc[2])
    if t is not None:
        rows.append((t, num(r.iloc[3]), num(r.iloc[4]), num(r.iloc[5])))
        continue
    t = parse_month(r.iloc[7])
    if t is not None:
        rows.append((t, num(r.iloc[8]), num(r.iloc[9]), num(r.iloc[10])))
npci_df = (pd.DataFrame(rows, columns=["month", "banks", "volume_mn", "value_cr"])
           .drop_duplicates("month").set_index("month").sort_index())
# A spreadsheet turned four 2018 value cells into dates; these are read as missing
corrupted = npci_df.index[npci_df["value_cr"].isna()].strftime("%Y-%m").tolist()

# RBI monthly data for 22 payment methods
rbi = pd.read_csv(RAW / "rbi_old.csv", header=None, dtype=str)
cols = {  # each method has a volume column followed by a value column
    "rtgs": 15, "aeps_ft": 23, "imps": 29, "nach_cr": 31, "neft": 33, "upi": 35,
    "bhim_aadhaar": 41, "nach_dr": 45, "netc": 47, "cards": 49, "credit": 51,
    "credit_pos": 53, "credit_oth": 55, "debit": 57, "debit_pos": 59, "debit_oth": 61,
    "ppi": 63, "ppi_wallet": 65, "ppi_card": 67, "cts": 75, "atm_cash": 97, "micro_atm": 111,
}
recs = []
for i in range(7, rbi.shape[0]):
    t = parse_month(rbi.iloc[i, 0])
    if t is None:
        continue
    rec = {"month": t}
    for k, j in cols.items():
        rec[f"{k}_vol_lakh"] = num(rbi.iloc[i, j])
        rec[f"{k}_val_cr"] = num(rbi.iloc[i, j + 1])
    recs.append(rec)
psi = pd.DataFrame(recs).set_index("month").sort_index()
psi.to_csv(PROC / "rbi_psi.csv")

# Join the NPCI data with the recent months typed in by hand
upi = npci_df[["volume_mn", "value_cr"]].copy()
upi["source"] = "NPCI product statistics (TransactIQ compilation)"
rbi_upi = pd.DataFrame({"volume_mn": psi["upi_vol_lakh"] / 10.0, "value_cr": psi["upi_val_cr"]})
overlap = upi.index.intersection(rbi_upi.index)
rel_err_vol = (rbi_upi.loc[overlap, "volume_mn"] / upi.loc[overlap, "volume_mn"] - 1).abs()
rel_err_val = (rbi_upi.loc[overlap, "value_cr"] / upi.loc[overlap, "value_cr"] - 1).abs()
fill = rbi_upi.index.difference(upi.index)
add = rbi_upi.loc[fill].copy()
add["source"] = "RBI Payment System Indicators"
recent = pd.read_csv(MAN / "upi_monthly_recent.csv")
recent["month"] = pd.to_datetime(recent["month"] + "-01")
recent = recent.set_index("month")
upi = pd.concat([upi, add, recent[["volume_mn", "value_cr", "source"]]]).sort_index()
upi = upi[~upi.index.duplicated(keep="first")]
full_idx = pd.date_range(upi.index.min(), upi.index.max(), freq="MS")
assert len(full_idx) == len(upi), "gap in monthly UPI series"
upi["fy"] = [fy_of(t) for t in upi.index]
upi["days"] = upi.index.days_in_month
upi.index.name = "month"
upi.to_csv(PROC / "upi_monthly.csv")

# Check the yearly totals against the totals published by the government
fy = upi.groupby("fy")["volume_mn"].sum()
published = {  # crore transactions, published by PIB and the Ministry of Finance
    2022: 4597.0, 2024: 13116.0, 2026: 24161.69,
}
checks = {str(k): {"computed_crore": fy[k] / 10, "published_crore": v,
                   "rel_error": fy[k] / 10 / v - 1} for k, v in published.items()}
val = {
    "n_months": len(upi), "first": str(upi.index.min().date()), "last": str(upi.index.max().date()),
    "npci_rbi_overlap_months": len(overlap),
    "npci_rbi_max_rel_err_volume": float(rel_err_vol.max()),
    "npci_rbi_median_rel_err_volume": float(rel_err_vol.median()),
    "npci_rbi_max_rel_err_value": float(rel_err_val.max()),
    "fy_totals_vs_published": checks,
    "npci_value_cells_corrupted": corrupted,
    "psi_months": len(psi), "psi_first": str(psi.index.min().date()), "psi_last": str(psi.index.max().date()),
    "fy_volume_crore": {str(k): v / 10 for k, v in fy.items()},
}
save_json(val, "data_validation")

print("DATA CHECKS")
print(f"  UPI monthly data: {val['n_months']} months ({upi.index.min():%b %Y} to {upi.index.max():%b %Y})")
print(f"  RBI data for 22 payment methods: {val['psi_months']} months ({psi.index.min():%b %Y} to {psi.index.max():%b %Y})")
print(f"  NPCI vs RBI: {val['npci_rbi_overlap_months']} common months, largest difference {100 * val['npci_rbi_max_rel_err_volume']:.2f}%")
for k, c in checks.items():
    print(f"  {int(k) - 1}-{k[2:]} total: {c['computed_crore']:,.0f} crore (published {c['published_crore']:,.0f}, "
          f"difference {100 * c['rel_error']:+.3f}%)")
print(f"  Value cells set to missing (spreadsheet error): {', '.join(corrupted)}")