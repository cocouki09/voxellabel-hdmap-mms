"""Figure for R1 Section 4.1: ground-model predictors on the same map vertices.

(a) cumulative distribution of |error| (log x), (b) median |error| with 95%
block-bootstrap CI. Reads W05 cells.csv + summary.json.

  python eval/revision/fig_ground_baselines.py --w05 <W05 dir> --out <png>
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_sys.stdout.reconfigure(encoding="utf-8")
# reference categorical palette, fixed order (slots 1-4); constant baseline in muted ink
METHODS = [("nearest", "Nearest vertex (Eq. 2)", "#2a78d6"),
           ("idw", "IDW", "#eb6834"),
           ("tin", "TIN", "#1baf7a"),
           ("plane", "Local plane", "#eda100"),
           ("constant", "Constant", "#8a8984")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w05", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.w05)
    cells = pd.read_csv(d / "cells.csv")
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))["common_mask"]
    common = np.all(np.isfinite(cells[[f"pred_{m}" for m, _, _ in METHODS]].to_numpy()), axis=1)
    ref = cells["ref_z"].to_numpy()[common]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5 / 2.54, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    for m, name, col in METHODS:
        e = np.sort(np.abs(cells[f"pred_{m}"].to_numpy()[common] - ref))
        y = np.arange(1, len(e) + 1) / len(e)
        a.plot(np.maximum(e, 1e-3), y, color=col, lw=2 if m != "constant" else 1.5,
               ls="-" if m != "constant" else "--", label=name)
    a.set_xscale("log"); a.set_xlim(2e-3, 12); a.set_ylim(0, 1)
    a.set_xlabel("|error| (m), log scale"); a.set_ylabel("Fraction of reference cells")
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.axvline(0.2, color=MUTED, lw=0.8, ls=":")
    a.text(0.22, 0.52, "0.2 m", color=MUTED, fontsize=6.5)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    a.set_title("(a) Error distribution, 26,240 cells", loc="left", fontsize=7, color=INK)

    vm = [x for x in METHODS if x[0] != "constant"]
    for i, (m, name, col) in enumerate(vm):
        s = summ[m]; lo, hi = s["median_abs_ci95"]
        b.plot([lo * 100, hi * 100], [i, i], color=col, lw=2, solid_capstyle="round")
        b.plot([s["median_abs"] * 100], [i], "o", color=col, ms=8, mec="white", mew=2)
        b.text(hi * 100 + 0.15, i, f"{s['median_abs']*100:.1f} cm", va="center", fontsize=6.5, color=INK)
    b.set_yticks(range(len(vm))); b.set_yticklabels([n.replace(" (Eq. 2)", "\n(Eq. 2)") for _, n, _ in vm])
    b.set_xlim(4.5, 9.2); b.set_ylim(len(vm) - 0.5, -0.5)
    b.set_xlabel("Median |error| (cm)\nwith 95% block-bootstrap CI")
    b.grid(True, axis="x", color=GRID, lw=0.6); b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.set_title(f"(b) Median error\n(constant: {summ['constant']['median_abs']:.2f} m, off scale)",
                loc="left", fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
