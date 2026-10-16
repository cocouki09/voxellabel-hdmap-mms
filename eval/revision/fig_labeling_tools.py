"""Evaluation figure for R1 (Section 3.7): real screenshots of the labeling tools with a flow strip.

Screenshots come from the labeling viewer in its read-only figure mode (?fig=1&sid=N), which shows English labels.

  python eval/revision/fig_labeling_tools.py --caps <dir with cell.png obj.png recheck.png> --out <png>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"]})
INK, MUTED = "#0b0b0b", "#52514e"
FILLS = ["#eef3fb", "#dde8f7", "#f4f4f2", "#dde8f7", "#fbf1e3", "#eaf5ec"]


def strip(ax, tags=("(a) ", "(b) ", "(c) "), panels=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    p = panels or ("", "", "")
    items = [("Stratified random\nsample", "554,610 cells, 12 strata\n1,090 drawn"),
             (tags[0] + "Blind cell labeling", "728 cells (695 judgeable)\npipeline label hidden" + p[0]),
             ("Repeat labeling", "49 cells, κ = 0.86"),
             (tags[1] + "Object review", "all 91 output objects" + p[1]),
             (tags[2] + "Yes/no re-checks", "post 124; tunnel barrier\n40 + 27 (both directions)" + p[2]),
             ("Scoring by use", "occupancy → objects\n→ cells")]
    w, gap = 0.152, 0.018
    for i, (t, s) in enumerate(items):
        x = 0.005 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.08), w, 0.84, boxstyle="round,pad=0.005,rounding_size=0.02", fc=FILLS[i], ec=MUTED, lw=0.8))
        three = s.count("\n") > 1
        ax.text(x + w / 2, 0.76 if three else 0.66, t, ha="center", va="center", fontsize=7, fontweight="bold", color=INK, linespacing=1.0)
        ax.text(x + w / 2, 0.33 if three else 0.28, s, ha="center", va="center", fontsize=6, color=MUTED, linespacing=1.05)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + w + gap - 0.002, 0.5), xytext=(x + w + 0.002, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))


def label(ax, t):
    ax.text(0.008, 0.985, t, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1f6feb", ec="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.caps)
    cell = np.asarray(Image.open(d / "cell.png").convert("RGB"))[0:1000, 0:1600]
    obj = np.asarray(Image.open(d / "obj.png").convert("RGB"))[30:960, 0:1195]
    rc = np.asarray(Image.open(d / "recheck.png").convert("RGB"))
    rc = np.vstack([rc[30:910, 800:1600], np.full((10, 800, 3), 30, np.uint8),
                    np.hstack([rc[915:960, 0:500], np.full((45, 300, 3), 30, np.uint8)])])
    W = 170 / 25.4
    h_strip = 0.62
    h_a = W * cell.shape[0] / cell.shape[1]
    wb = W * 0.6; wc = W - wb
    h_bc = max(wb * obj.shape[0] / obj.shape[1], wc * rc.shape[0] / rc.shape[1])
    H = h_strip + h_a + h_bc + 0.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 1 - h_strip / H, 1, h_strip / H]); strip(ax)
    ax = fig.add_axes([0, 1 - (h_strip + h_a + 0.04) / H, 1, h_a / H]); ax.imshow(cell); ax.axis("off")
    label(ax, "(a) Cell labeling: 3D, plan and side views of the raw LiDAR, survey photos; yellow box = cell")
    y0 = 1 - (h_strip + h_a + 0.10 + h_bc) / H
    ax = fig.add_axes([0, y0, wb / W - 0.005, h_bc / H]); ax.imshow(obj); ax.axis("off")
    label(ax, "(b) Object review: pink = cells of one output object")
    ax = fig.add_axes([wb / W + 0.005, y0, wc / W - 0.005, h_bc / H]); ax.imshow(rc); ax.axis("off")
    label(ax, "(c) Re-check: barrier or tunnel wall?")
    fig.savefig(a.out, dpi=300)
    print(a.out)


if __name__ == "__main__":
    main()
