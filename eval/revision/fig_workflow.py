"""Figure 1 for R1: workflow with explicit inputs, processing steps and output.

Drawn at its print width (13.5 cm) so that the smallest text is at least 6 pt.

  python eval/revision/fig_workflow.py --out <png>
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"]})
INK, MUTED = "#0b0b0b", "#52514e"
IN_FILL, STEP_FILL, CORE_FILL, OUT_FILL = "#eef3fb", "#f4f4f2", "#dde8f7", "#eaf5ec"


def box(ax, x, y, w, h, title, sub, fill, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fill, ec=MUTED, lw=0.8))
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=7.2,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.0)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=6.2, color=MUTED, linespacing=1.1)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(13.5 / 2.54, 2.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-0.012, 1.012); ax.set_ylim(-0.01, 1); ax.axis("off")
    for x, t in ((0.105, "Inputs"), (0.50, "Processing (no training)"), (0.895, "Output")):
        ax.text(x, 0.95, t, ha="center", fontsize=7.6, color=INK, fontweight="bold")
    w, h = 0.20, 0.33
    box(ax, 0.005, 0.42, w, h, "Vector HD map", "11 SHP layers\nwith 3D vertices", IN_FILL)
    box(ax, 0.005, 0.04, w, h, "MMS LiDAR", "unclassified points\nROI clip, stride 4", IN_FILL)
    box(ax, 0.27, 0.42, w, h, "Ground model", "nearest map vertex\n(Eq. 2)", CORE_FILL, True)
    box(ax, 0.27, 0.04, w, h, "Voxelization", "0.5 m cells, nearest\nreal point (Eq. 1)", STEP_FILL)
    box(ax, 0.53, 0.42, w, h, "Elevation-gated\nlabeling", "radius r_c and\nwindow (Eq. 4)", CORE_FILL, True)
    box(ax, 0.53, 0.04, w, h, "Vehicle filter,\nretention, objects", "pole protection;\nshared poles", CORE_FILL, True)
    box(ax, 0.79, 0.18, 0.205, 0.46, "Labeled voxel\nscene", "objects + unknown\noccupancy; each voxel\nkeeps a LiDAR point", OUT_FILL, True)
    arrow(ax, 0.205, 0.585, 0.27, 0.585)
    arrow(ax, 0.205, 0.20, 0.27, 0.20)
    arrow(ax, 0.47, 0.585, 0.53, 0.585)
    arrow(ax, 0.47, 0.20, 0.53, 0.50)
    arrow(ax, 0.63, 0.42, 0.63, 0.37)
    arrow(ax, 0.73, 0.20, 0.79, 0.33)
    ax.annotate("", xy=(0.60, 0.755), xytext=(0.15, 0.755),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, ls="--", connectionstyle="arc3,rad=-0.22", shrinkA=0, shrinkB=0))
    ax.text(0.245, 0.865, "feature geometry and categories", fontsize=6.2, color=MUTED, ha="center")
    fig.savefig(args.out, dpi=300)
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
