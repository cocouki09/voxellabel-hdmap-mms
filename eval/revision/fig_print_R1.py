"""Print-size versions of the remaining figures for R1.

- Figure 4: spatial distribution of ground-model errors, redrawn from the error
  tables with print-size fonts (shared colour scale, clipped at 3 m).
- Figures 6 and 7 (viewer screenshots of the submitted version): the original
  screenshots with larger panel labels and a colour legend of the categories in
  the viewer's palette (web/src/core/fsdPalette.ts).

  python eval/revision/fig_print_R1.py --out paper/revision_R1/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
PAL = {"Unknown (occupied)": "#d6c08a", "Guardrail": "#8a8f7e", "Traffic light": "#22b8d6", "Sign": "#f0c040",
       "Post / street light": "#6a5d4e", "Overpass (height barrier)": "#7c8aa0", "Tunnel": "#b5723e"}
W_MM = 170


def load(name):
    x, y, e = [], [], []
    with open(ROOT / "eval" / name, newline="") as f:
        for r in csv.DictReader(f):
            x.append(float(r["center_x"])); y.append(float(r["center_y"])); e.append(abs(float(r["error_m"])))
    return np.array(x), np.array(y), np.clip(np.array(e), 0, 3.0)


def fig4(out):
    fig = plt.figure(figsize=(W_MM / 25.4, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1.0], height_ratios=[1, 0.05], wspace=0.25, hspace=0.55)
    for i, (name, title) in enumerate([("ground_z_errors.csv", "(a) Deoksong interchange"),
                                       ("ground_z_errors_road86.csv", "(b) Road-86")]):
        ax = fig.add_subplot(gs[0, i])
        x, y, e = load(name)
        o = np.argsort(e)
        sc = ax.scatter(x[o] - x.min(), y[o] - y.min(), c=e[o], s=1.2, cmap="magma", vmin=0, vmax=3, lw=0)
        ax.set_aspect("equal"); ax.set_title(title, loc="left")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    cax = fig.add_subplot(gs[1, 0])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("|ground-model error| (m), clipped at 3 m; coordinates from the ROI corner")
    fig.savefig(out / "Fig4_ground_errmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def screenshot(src, dst, labels, legend, ncol, crop=0.93):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    a = np.asarray(im)
    # crop away the old "(a)/(b)" captions at the bottom: keep rows until the last non-white row of the image area
    fig_h = W_MM / 25.4 * h / w
    fig = plt.figure(figsize=(W_MM / 25.4, fig_h * crop + 0.45))
    ax = fig.add_axes([0, 0.45 / (fig_h * crop + 0.45), 1, fig_h * crop / (fig_h * crop + 0.45)])
    ax.imshow(a[: int(h * crop)]); ax.axis("off")
    for x, t in labels:
        ax.text(x * w, 0.03 * h, t, fontsize=9, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if legend:
        fig.legend(handles=[Patch(fc=PAL[k], ec="0.3", lw=0.4, label=k) for k in legend], loc="lower center",
                   ncol=ncol, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.2)
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    fig4(out)
    screenshot(out / "Fig8_input_output_v1.png", out / "Fig6_input_output.png",
               [(0.005, "(a) Raw LiDAR + map"), (0.497, "(b) Labeled voxels")],
               ["Unknown (occupied)", "Guardrail", "Traffic light", "Sign", "Post / street light"], 5)
    screenshot(out / "Fig9_fullviews_v1.png", out / "Fig7_fullviews.png",
               [], list(PAL), 4, crop=1.0)
    print("ok")


if __name__ == "__main__":
    main()
