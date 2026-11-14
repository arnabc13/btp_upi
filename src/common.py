"""Folders, graph style and small helper functions shared by all the steps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "manual"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
RES = ROOT / "results"
for p in (PROC, FIG, TAB, RES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20261105

C = {
    "ink": "#1f2933", "muted": "#7b8794", "grid": "#e4e7eb",
    "blue": "#1f5fa8", "orange": "#d9730d", "green": "#2e7d4f",
    "red": "#b3261e", "purple": "#6b4fa3", "teal": "#0f7c80",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["muted"], "axes.grid": True, "grid.color": C["grid"],
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.4,
        "legend.frameon": False,
    })


def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(RES / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(name):
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def fy_of(ts):
    """Financial year of a date, named by the year it ends in (April 2025 -> 2026)."""
    return ts.year + 1 if ts.month >= 4 else ts.year


def anchors():
    return pd.read_csv(MAN / "anchors.csv")


def num(s):
    """Convert a number written in Indian style, such as '20,60,735.57', to a float."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def solver(name, required=True):
    """Find the Ipopt or Bonmin optimisation solver on this computer."""
    import os
    import shutil
    exe = name + (".exe" if os.name == "nt" else "")
    dirs = [os.environ.get("SOLVER_DIR")]
    try:
        import idaes
        dirs.append(getattr(idaes, "bin_directory", None))
    except Exception:
        pass
    dirs += [os.path.expanduser("~/.idaes/bin"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "idaes", "bin")]
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, exe)):
            return os.path.join(d, exe)
    p = shutil.which(name)
    if p:
        return p
    if required:
        raise FileNotFoundError(
            f"{name} not found. Run `idaes get-extensions` (see README) or set SOLVER_DIR to the folder containing {exe}.")
    return None