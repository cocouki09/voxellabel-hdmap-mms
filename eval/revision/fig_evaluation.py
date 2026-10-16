"""Evaluation-design figure for R1 (Section 3.7): sample, blind labeling, re-checks, three-level scoring.

  python eval/revision/fig_evaluation.py --out <png>
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
IN_FILL, STEP_FILL, CORE_FILL, OUT_FILL, FIX_FILL = "#eef3fb", "#f4f4f2", "#dde8f7", "#eaf5ec", "#fbf1e3"


def box(ax, x, y, w, h, title, sub, fill, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fill, ec=MUTED, lw=1.0))
    ax.text(x + w / 2, y + h * 0.80, title, ha="center", va="center", fontsize=7.5,
            fontweight="bold" if bold else "normal", color=INK)
    ax.text(x + w / 2, y + h * 0.38, sub, ha="center", va="center", fontsize=5.8, color=MUTED, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, ls=ls, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.set_xlim(0, 1.005); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.10, 0.96, "Design", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(0.43, 0.96, "Reference data (blind, one annotator)", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(0.84, 0.96, "Scoring by use", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    W = 0.175
    box(ax, 0.01, 0.58, W, 0.30, "Population", "all 554,610 occupied cells\nincl. filtered cells;\nparameters frozen before", IN_FILL, True)
    box(ax, 0.01, 0.14, W, 0.34, "Stratified random\nsample", "12 strata by outcome\n1,090 drawn (fixed seed)\nlabeled in random order\nto fixed quotas: 728", IN_FILL, True)
    box(ax, 0.235, 0.58, 0.18, 0.30, "Blind cell labeling", "raw LiDAR (3D, plan, side)\n+ survey photos\n14 categories; 695 judgeable", CORE_FILL, True)
    box(ax, 0.235, 0.14, 0.18, 0.34, "Object review", "all 91 output objects,\nname hidden, random order\nmain content + purity", CORE_FILL, True)
    box(ax, 0.445, 0.58, 0.18, 0.30, "Consistency", "49 cells labeled again\nin random order\nintra-rater κ = 0.86", STEP_FILL)
    box(ax, 0.445, 0.14, 0.18, 0.34, "Yes/no re-checks", "post: 124 cells\ntunnel barrier vs. wall:\n40 predicted guardrail\n+ 27 not predicted", FIX_FILL, True)
    ax.text(0.535, 0.05, "blind labels frozen; corrections in separate files,\nresults reported before and after", ha="center", fontsize=5.8, color=MUTED, style="italic")
    sx, sw = 0.70, 0.28
    box(ax, sx, 0.66, sw, 0.22, "1  Static occupancy", "share of output cells that are\nstatic solid structure", OUT_FILL, True)
    box(ax, sx, 0.39, sw, 0.22, "2  Objects", "object precision (Wilson CI), purity,\nrecall vs. mapped features", OUT_FILL, True)
    box(ax, sx, 0.10, sw, 0.24, "3  Cells (semantic)", "Horvitz–Thompson P / R / F1 / IoU\nwith stratified bootstrap CI;\ndelivered + strict definitions; ablations", OUT_FILL, True)
    arrow(ax, 0.10, 0.58, 0.10, 0.48)
    arrow(ax, 0.185, 0.40, 0.235, 0.66)
    arrow(ax, 0.415, 0.73, 0.445, 0.73)
    arrow(ax, 0.325, 0.58, 0.47, 0.48, ls="--")
    arrow(ax, 0.415, 0.31, 0.445, 0.31, ls="--")
    ax.plot([0.645, 0.645], [0.22, 0.77], color=MUTED, lw=1.0)
    arrow(ax, 0.625, 0.50, 0.645, 0.50)
    for y in (0.77, 0.50, 0.22):
        arrow(ax, 0.645, y, sx, y)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
