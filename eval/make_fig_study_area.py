"""Fig. study-area - two-panel site map for the paper (§5.1 / Data).

Panel (a) Deoksong expressway interchange (primary site; evaluation ROI
          1,250 m x 600 m marked).
Panel (b) road86 national-road intersection (second site; evaluation ROI and
          the focused spot-check ROI, magenta, marked).

Drawn entirely from the pipeline's own scene_preview.json overlays (HD-map
geometry) - no external basemap/tiles, so it is fully reproducible and free of
licensing concerns. Coordinates are UTM zone 52N (EPSG:32652); grid north ~=
true north at this longitude (convergence < 1 deg, negligible at map scale).

Run:
  cd voxel_scene_pipeline
  set PYTHONIOENCODING=utf-8
  python eval/make_fig_study_area.py
Output: eval/figs/fig_study_area.png (220 dpi)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "deoksong": ROOT / "data/output_deoksong/preview_json/scene_preview.json",
    "road86": ROOT / "data/output_road86/preview_json/scene_preview.json",
}
# evaluation ROIs (UTM-52N) - keep in sync with eval/eval_ground_z.py
DEOKSONG_ROI = (337100.0, 4172950.0, 338350.0, 4173550.0)   # 1,250 x 600 m
ROAD86_ROI = (339850.0, 4163880.0, 340200.0, 4164480.0)     # 350 x 600 m
# focused spot-check ROI on road86 (matches eval/inject_label_roi.py cage)
ROAD86_LABEL_ROI = (339974.0, 4164305.0, 340054.0, 4164385.0)
OUT = ROOT / "eval/figs/fig_study_area.png"

ROAD_CATS = {"link", "driveway", "subsidiary", "node"}
LANE_CATS = {"lane", "crosswalk", "stopline", "surface_mark"}
STYLE = {
    "guardrail": dict(color="#2e7d32", lw=1.4, zorder=4),
    "height_barrier": dict(color="#e65100", lw=2.0, zorder=4),
}
POINT_STYLE = {
    "traffic_light": dict(marker="^", color="#c62828", s=34, zorder=6, label="traffic light"),
    "sign": dict(marker="s", color="#c9a800", s=22, zorder=6, label="sign"),
    "street_light": dict(marker="o", color="#1565c0", s=26, zorder=6, label="street light"),
}


def load_overlays(path: Path):
    with open(path, encoding="utf-8") as f:
        scene = json.load(f)
    return scene["overlays"]


def geom_xy(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "Point":
        yield [c[0]], [c[1]]
    elif t == "LineString":
        yield [p[0] for p in c], [p[1] for p in c]
    elif t == "MultiLineString":
        for line in c:
            yield [p[0] for p in line], [p[1] for p in line]
    elif t == "Polygon":
        for ring in c:
            yield [p[0] for p in ring], [p[1] for p in ring]
    elif t == "MultiPolygon":
        for poly in c:
            for ring in poly:
                yield [p[0] for p in ring], [p[1] for p in ring]


def draw_site(ax, overlays, title, roi, label_roi=None, pad_frac=0.10):
    pt_seen = set()
    for o in overlays:
        cat = o["category"]
        if cat == "label_roi":
            continue
        for xs, ys in geom_xy(o["geometry"]):
            if cat in POINT_STYLE:
                st = dict(POINT_STYLE[cat]); st.pop("label")
                ax.scatter(xs, ys, **st, edgecolors="white", linewidths=0.4)
                pt_seen.add(cat)
            elif cat in STYLE:
                ax.plot(xs, ys, **STYLE[cat], solid_capstyle="round")
            elif cat in LANE_CATS:
                ax.plot(xs, ys, color="#cfcfcf", lw=0.5, zorder=2)
            elif cat in ROAD_CATS:
                ax.plot(xs, ys, color="#7d7d7d", lw=1.1, zorder=3)
    x0, y0, x1, y1 = roi
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor="black", lw=1.4, linestyle="--", zorder=7))
    if label_roi:
        lx0, ly0, lx1, ly1 = label_roi
        ax.add_patch(Rectangle((lx0, ly0), lx1 - lx0, ly1 - ly0, fill=False,
                               edgecolor="#d81b9a", lw=1.8, zorder=8))
    # view = ROI + padding (overlays extend beyond; ROI is what the paper evaluates)
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    ax.set_xlim(x0 - pw, x1 + pw)
    ax.set_ylim(y0 - ph, y1 + ph)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
    ax.set_xlabel("UTM-52N easting (m)", fontsize=8)
    ax.set_ylabel("UTM-52N northing (m)", fontsize=8)
    ax.ticklabel_format(style="plain", useOffset=False)
    # scale bar (200 m), bottom-left inside the ROI (above the dashed edge)
    bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.06
    ax.plot([bx, bx + 200], [by, by], color="black", lw=2.5, zorder=9)
    ax.text(bx + 100, by + (y1 - y0) * 0.035, "200 m", ha="center", fontsize=7,
            zorder=9)
    # north arrow - axes-fraction coords so it never blows up in data units
    ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
    return pt_seen


def main():
    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.9),
        gridspec_kw={"width_ratios": [2.08, 0.72]})
    seen = set()
    seen |= draw_site(axes[0], load_overlays(SCENES["deoksong"]),
                      "(a) Deoksong expressway interchange - primary site",
                      DEOKSONG_ROI)
    seen |= draw_site(axes[1], load_overlays(SCENES["road86"]),
                      "(b) Road-86 intersection - second site",
                      ROAD86_ROI, label_roi=ROAD86_LABEL_ROI)

    handles = [
        Line2D([], [], color="#7d7d7d", lw=1.2, label="road link / driveway"),
        Line2D([], [], color="#cfcfcf", lw=0.8, label="lane marking"),
        Line2D([], [], color=STYLE["guardrail"]["color"], lw=1.6, label="guardrail"),
        Line2D([], [], color=STYLE["height_barrier"]["color"], lw=2.0, label="height barrier"),
    ]
    for cat, st in POINT_STYLE.items():
        if cat in seen:
            handles.append(Line2D([], [], marker=st["marker"], linestyle="",
                                  color=st["color"], markersize=6, label=st["label"]))
    handles.append(Line2D([], [], color="black", lw=1.4, linestyle="--",
                          label="evaluation ROI"))
    handles.append(Line2D([], [], color="#d81b9a", lw=1.8,
                          label="spot-check ROI (80 m x 80 m)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Study sites - Korean precision-road-map geometry, Gyeonggi Province, "
                 "Republic of Korea (UTM zone 52N)", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
