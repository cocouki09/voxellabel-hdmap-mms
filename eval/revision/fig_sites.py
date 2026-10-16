"""Figure 3 for R1 (R2: overview map followed by detailed site maps).

(a) overview of both sites at one scale (map road geometry, ROIs, distance,
geographic coordinates), (b) Deoksong detail, (c) Road-86 detail. Reuses the
drawing code of eval/make_fig_study_area.py; no external basemap.

  python eval/revision/fig_sites.py --out <png>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from pyproj import Transformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_fig_study_area as sa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ov = {k: sa.load_overlays(p) for k, p in sa.SCENES.items()}
    fig = plt.figure(figsize=(13.5 / 2.54, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.8], height_ratios=[1.25, 1.0])
    a0 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a1 = fig.add_subplot(gs[1, :])
    # (a) overview
    for k in ("deoksong", "road86"):
        for o in ov[k]:
            if o["category"] in sa.ROAD_CATS:
                for xs, ys in sa.geom_xy(o["geometry"]):
                    a0.plot(xs, ys, color="#7d7d7d", lw=0.6)
    tr = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
    for roi, name in ((sa.DEOKSONG_ROI, "Deoksong (b)"), (sa.ROAD86_ROI, "Road-86 (c)")):
        x0, y0, x1, y1 = roi
        a0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=1.4, ls="--"))
        lon, lat = tr.transform((x0 + x1) / 2, (y0 + y1) / 2)
        a0.text(x0 - 400, (y0 + y1) / 2, f"{name}\n{lat:.3f}° N\n{lon:.3f}° E", fontsize=8, va="center", ha="right")
    cd = ((sa.DEOKSONG_ROI[0] + sa.DEOKSONG_ROI[2]) / 2, (sa.DEOKSONG_ROI[1] + sa.DEOKSONG_ROI[3]) / 2)
    cr = ((sa.ROAD86_ROI[0] + sa.ROAD86_ROI[2]) / 2, (sa.ROAD86_ROI[1] + sa.ROAD86_ROI[3]) / 2)
    dist = ((cd[0] - cr[0]) ** 2 + (cd[1] - cr[1]) ** 2) ** 0.5
    a0.plot([cd[0], cr[0]], [cd[1], cr[1]], color="#2a78d6", lw=1, ls=":")
    a0.text((cd[0] + cr[0]) / 2 - 300, (cd[1] + cr[1]) / 2, f"{dist / 1000:.1f} km", color="#2a78d6", fontsize=8, ha="right")
    a0.set_aspect("equal"); a0.ticklabel_format(style="plain", useOffset=False); a0.tick_params(labelsize=7)
    xs = [sa.DEOKSONG_ROI[0], sa.DEOKSONG_ROI[2], sa.ROAD86_ROI[0], sa.ROAD86_ROI[2]]
    ys = [sa.DEOKSONG_ROI[1], sa.DEOKSONG_ROI[3], sa.ROAD86_ROI[1], sa.ROAD86_ROI[3]]
    a0.set_xlim(min(xs) - 6500, max(xs) + 1200); a0.set_ylim(min(ys) - 900, max(ys) + 1600)
    a0.set_xlabel("UTM-52N easting (m)", fontsize=8); a0.set_ylabel("UTM-52N northing (m)", fontsize=8)
    bx, by = min(xs) - 6000, min(ys) - 500
    a0.plot([bx, bx + 2000], [by, by], color="black", lw=2.5); a0.text(bx + 1000, by + 200, "2 km", ha="center", fontsize=7)
    a0.annotate("N", xy=(0.9, 0.97), xytext=(0.9, 0.88), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    a0.set_title("(a) Overview, Gyeonggi Province", fontsize=10)
    seen = sa.draw_site(a1, ov["deoksong"], "(b) Deoksong expressway interchange (primary site)", sa.DEOKSONG_ROI)
    seen |= sa.draw_site(a2, ov["road86"], "(c) Road-86 intersection", sa.ROAD86_ROI, label_roi=sa.ROAD86_LABEL_ROI)
    handles = [Line2D([], [], color="#7d7d7d", lw=1.2, label="road link / driveway"),
               Line2D([], [], color="#cfcfcf", lw=0.8, label="lane marking"),
               Line2D([], [], color=sa.STYLE["guardrail"]["color"], lw=1.6, label="guardrail"),
               Line2D([], [], color=sa.STYLE["height_barrier"]["color"], lw=2.0, label="height barrier / overpass (C5)")]
    for cat, st in sa.POINT_STYLE.items():
        if cat in seen:
            lab = {"street light": "post / street light (C6)"}.get(st["label"], st["label"])
            handles.append(Line2D([], [], marker=st["marker"], ls="", color=st["color"], markersize=5, label=lab))
    handles += [Line2D([], [], color="black", lw=1.4, ls="--", label="evaluation ROI"),
                Line2D([], [], color="#d81b9a", lw=1.8, label="spot-check ROI (80 m × 80 m)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, frameon=False, handlelength=1.6, columnspacing=1.0)
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        f = t.get_fontsize()
        t.set_fontsize(7.2 if f >= 9.5 else max(6.2, min(6.6, f * 0.8)))
    for a in (a0, a1, a2):
        a.tick_params(labelsize=6.2)
    from matplotlib.ticker import MaxNLocator
    a0.xaxis.set_major_locator(MaxNLocator(2)); a2.xaxis.set_major_locator(MaxNLocator(2))
    a1.set_title("(b) Deoksong expressway interchange (primary site)", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(f"[SAVE] {args.out}  distance {dist:.0f} m")


if __name__ == "__main__":
    main()
