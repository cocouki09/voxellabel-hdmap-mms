"""Figure 2 for R1 (R2 Methods comment): one road stretch, one viewpoint,
(a) precision road map, (b) raw MMS LiDAR, (c) labeled output, (d) the same
output with the elevation gate switched off (branches over the guardrail
become guardrail). Category colors follow the scene figures.

  python eval/revision/fig_three_panel.py --center 337410 4173210 --half 22 --out <png>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys as _sys
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from shapely.geometry import box  # noqa: E402

_sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/lidar/deoksong/SEC001_서별내IC_내곡교차로_240213"
# viewer palette (web/src/core/fsdPalette.ts), as in Figures 7 and 8
CAT_COL = {"unknown": "#d6c08a", "guardrail": "#8a8f7e", "traffic_light": "#22b8d6",
           "sign": "#f0c040", "street_light": "#6a5d4e", "height_barrier": "#7c8aa0"}
NAMES = {"unknown": "unknown", "guardrail": "guardrail", "traffic_light": "traffic light", "sign": "sign",
         "street_light": "post / street light", "height_barrier": "overpass (height barrier)"}
MAP_LAYERS = {"B2_SURFACELINEMARK": ("#9e9e9e", 0.8), "A2_LINK": ("#2a78d6", 1.2),
              "A3_DRIVEWAYSECTION": ("#bdbdbd", 0.6), "C3_VEHICLEPROTECTIONSAFETY": (CAT_COL["guardrail"], 2.2),
              "C1_TRAFFICLIGHT": (CAT_COL["traffic_light"], 0), "B1_SAFETYSIGN": (CAT_COL["sign"], 0),
              "C6_POSTPOINT": (CAT_COL["street_light"], 0)}


def style(ax, c, half, zr, title):
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_zlim(*zr)
    ax.set_box_aspect((1, 1, (zr[1] - zr[0]) / (2 * half)), zoom=1.45)
    ax.view_init(elev=28, azim=-62)
    ax.set_axis_off()
    ax.text2D(0.03, 0.93, title, transform=ax.transAxes, fontsize=7.2, va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", nargs=2, type=float, required=True)
    ap.add_argument("--half", type=float, default=22.0)
    ap.add_argument("--full", default=str(ROOT / "../output/ijgi_revision/W07_RUNS/A0_full_sign2m/stages.npz"))
    ap.add_argument("--gateoff", default=str(ROOT / "../output/ijgi_revision/W07_RUNS/A2_gate_off/stages.npz"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = np.array(args.center); h = args.half
    win = box(c[0] - h, c[1] - h, c[0] + h, c[1] + h)

    # voxel outputs
    def cells(path):
        st = np.load(path)
        xyz = st["origin"] + (st["cells"] + 0.5) * float(st["voxel_size"])
        m = st["keep"] & (np.abs(xyz[:, 0] - c[0]) <= h) & (np.abs(xyz[:, 1] - c[1]) <= h)
        return xyz[m], st["labels_final"][m].astype(str)
    fx, fl = cells(args.full)
    gx, gl = cells(args.gateoff)
    z0 = np.percentile(fx[:, 2], 2); zr = (z0 - 1, z0 + 16)

    # raw LiDAR
    pts, col = [], []
    for f in glob.glob(str(RAW / "SURV0[125]*/TRACK*/Laser01/*.las")):
        with laspy.open(f) as r:
            hd = r.header
            if hd.maxs[0] < c[0] - h or hd.mins[0] > c[0] + h or hd.maxs[1] < c[1] - h or hd.mins[1] > c[1] + h:
                continue
            las = r.read()
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        m = (np.abs(x - c[0]) <= h) & (np.abs(y - c[1]) <= h) & (z >= zr[0]) & (z <= zr[1])
        m &= np.random.default_rng(0).random(len(x)) < 0.35
        if not m.any():
            continue
        rgb = np.column_stack([las.red[m], las.green[m], las.blue[m]]).astype(float)
        pts.append(np.column_stack([x[m], y[m], z[m]])); col.append(rgb / (65535.0 if rgb.max() > 255 else 255.0))
    P = np.vstack(pts); C = np.clip(np.vstack(col) * 1.3, 0, 1)

    fig = plt.figure(figsize=(13.5 / 2.54, 4.7))
    ax = [fig.add_subplot(2, 2, k + 1, projection="3d") for k in range(4)]
    # (a) map
    for layer, (colr, lw) in MAP_LAYERS.items():
        g = gpd.read_file(RAW / "HDMap_UTM52N_타원체고" / f"{layer}.shp")
        g = g[g.intersects(win)]
        for geom in g.geometry.clip(win):
            parts = getattr(geom, "geoms", [geom])
            for p in parts:
                if p.geom_type == "Polygon":
                    p = p.exterior
                co = np.asarray(p.coords)
                if co.shape[1] < 3:
                    continue
                if lw == 0:
                    zc = co[:, 2].mean()
                    ax[0].plot([co[:, 0].mean()] * 2, [co[:, 1].mean()] * 2, [z0, zc], color=colr, lw=1.5)
                    ax[0].scatter([co[:, 0].mean()], [co[:, 1].mean()], [zc], color=colr, s=18, depthshade=False)
                else:
                    ax[0].plot(co[:, 0], co[:, 1], co[:, 2], color=colr, lw=lw)
    style(ax[0], c, h, zr, "(a) Precision road map (3D vertices)")
    ax[1].scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=0.5, depthshade=False, linewidths=0)
    style(ax[1], c, h, zr, "(b) Raw MMS LiDAR (unclassified)")
    for k, (X, L, t) in enumerate([(fx, fl, "(c) Labeled output (elevation gate on)"),
                                   (gx, gl, "(d) Elevation gate off")]):
        cc = [CAT_COL.get(l, "#c8b98a") for l in L]
        ax[2 + k].scatter(X[:, 0], X[:, 1], X[:, 2], c=cc, s=2.2, depthshade=False, linewidths=0, marker="s")
        style(ax[2 + k], c, h, zr, t)
    hand = [Line2D([], [], marker="s", ls="", color=v, label=NAMES[k]) for k, v in CAT_COL.items()
            if k in set(fl) | set(gl)]
    fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False, fontsize=6.4, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=-0.04, right=1.04, top=0.99, bottom=0.08, wspace=-0.08, hspace=0.02)
    fig.savefig(args.out, dpi=300)
    print(f"[SAVE] {args.out}  raw pts {len(P):,}  cells {len(fx):,}")


if __name__ == "__main__":
    main()
