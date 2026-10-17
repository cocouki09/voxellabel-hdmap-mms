"""Ground-Z reconstruction accuracy — SHP-vertex model vs LIDAR-measured surface.

Quantitative validation of the pipeline's ground-elevation model. The Korean
precision road-map (SHP) vertices give a PREDICTED local ground Z at any (x, y)
— this is exactly the model `run_build_scene_lidar.py` queries via
`build_ground_z_index` / `query_ground_z` to compute each voxel's elevation.

We compare it against an INDEPENDENT reference: the actual road surface as
measured by the MMS LIDAR, taken as a robust low percentile of point Z per
small XY cell inside the SHP road corridor. The corridor mask uses only SHP
geometry (XY) to pick which LIDAR points are road — it does NOT use the SHP Z,
so the Z comparison is genuinely independent (not circular).

No manual labeling required — fully automatic. A large signed bias would also
flag a vertical-datum mismatch between the map and the LIDAR.

Outputs: RMSE / MAE / bias summary + a per-cell CSV for the error-histogram and
spatial-error-map figures.

Run:
  cd voxel_scene_pipeline
  set PYTHONPATH=.
  python eval/eval_ground_z.py                 # auto: raw if present, else sample
  python eval/eval_ground_z.py --source sample # force the bundled sample
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from src.readers.shp_reader import read_shp
from src.readers.las_reader import read_las_xyz
from src.semantic.shp_matcher import (
    build_ground_z_index,
    query_ground_z,
    GROUND_SURFACE_CATEGORIES,
)

# --- config mirrored from run/run_build_scene_lidar.py (keep in sync) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Data folders (NGII delivery names). Override with VOXELLABEL_DEOKSONG_DIR / VOXELLABEL_ROAD86_DIR.
LIDAR_ROOT = Path(os.environ.get("VOXELLABEL_DEOKSONG_DIR",
                                 PROJECT_ROOT / "data/raw/lidar/deoksong/SEC001_서별내IC_내곡교차로_240213"))
SAMPLE_ROOT = PROJECT_ROOT / "data/sample"
ROI_BBOX = (337100.0, 4172950.0, 338350.0, 4173550.0)
ROAD_CORRIDOR_BUFFER_M = 3.0
ROAD_CORRIDOR_CATEGORIES = {"link", "driveway", "subsidiary"}

# cross-region #2 (mirrors run/run_build_scene_lidar.py). SAME eval params as
# deoksong (buffer / percentile / bin / min-points) — no re-tuning, so this is a transfer test.
ROAD86_ROOT = Path(os.environ.get("VOXELLABEL_ROAD86_DIR",
                                  PROJECT_ROOT / "data/raw/lidar/road86/SEC203_이패동494-2_진안사거리_240213"))
ROAD86_ROI = (339850.0, 4163880.0, 340200.0, 4164480.0)


def resolve_sources_road86() -> tuple[Path, list[Path]]:
    if not ROAD86_ROOT.is_dir():
        raise RuntimeError(
            f"road86 data not found at {ROAD86_ROOT} — set VOXELLABEL_ROAD86_DIR"
        )
    las = sorted(ROAD86_ROOT.glob("SURV*/TRACK*/Laser01/*.las"))
    if not las:
        raise RuntimeError(f"no .las under {ROAD86_ROOT}")
    print(f"[SOURCE] road86: {len(las)} LAS from {ROAD86_ROOT}")
    return ROAD86_ROOT / "HDMap_UTM52N_타원체고", las

LAYER_FILES = {
    "A1_NODE": "A1_NODE.shp",
    "A2_LINK": "A2_LINK.shp",
    "A3_DRIVEWAYSECTION": "A3_DRIVEWAYSECTION.shp",
    "A4_SUBSIDIARYSECTION": "A4_SUBSIDIARYSECTION.shp",
    "B2_SURFACELINEMARK": "B2_SURFACELINEMARK.shp",
    "B3_SURFACEMARK": "B3_SURFACEMARK.shp",
}
LAYER_TO_CATEGORY = {
    "B2_SURFACELINEMARK": "lane",
    "A2_LINK": "link",
    "A1_NODE": "node",
    "A3_DRIVEWAYSECTION": "driveway",
    "A4_SUBSIDIARYSECTION": "subsidiary",
    "B3_SURFACEMARK": "surface_mark",  # dispatched per-feature below
}
B3_KIND_CROSSWALK_SET = {5111, 5321}
B3_KIND_STOPLINE_SET = {5371, 5372, 5381}


def b3_category_for(row) -> str:
    kind = row.get("Kind")
    try:
        kind_int = int(kind)
    except (TypeError, ValueError):
        return "surface_mark"
    if kind_int in B3_KIND_CROSSWALK_SET:
        return "crosswalk"
    if kind_int in B3_KIND_STOPLINE_SET:
        return "stopline"
    return "surface_mark"


def resolve_sources(prefer: str) -> tuple[Path, list[Path]]:
    """Return (HDMAP_DIR, LAS files), mirroring the pipeline's selection."""
    use_raw = prefer == "raw" or (prefer == "auto" and LIDAR_ROOT.is_dir())
    if use_raw and LIDAR_ROOT.is_dir():
        las: list[Path] = []
        for surv_track in [
            "SURV05_내곡교차로/TRACK11",
            "SURV05_내곡교차로/TRACK12",
            "SURV01_서별내IC_내곡교차로/TRACK01",
            "SURV02_내곡교차로_서별내IC/TRACK02",
        ]:
            las += sorted((LIDAR_ROOT / surv_track / "Laser01").glob("*.las"))
        if las:
            print(f"[SOURCE] raw dataset under {LIDAR_ROOT}")
            return LIDAR_ROOT / "HDMap_UTM52N_타원체고", las

    sample_lidar = SAMPLE_ROOT / "lidar"
    sample_shp = SAMPLE_ROOT / "shp"
    if sample_lidar.is_dir() and sample_shp.is_dir():
        las = sorted(sample_lidar.glob("*.laz")) + sorted(sample_lidar.glob("*.las"))
        if las:
            print(f"[SOURCE] bundled sample under {SAMPLE_ROOT}")
            return sample_shp, las

    raise RuntimeError("No LIDAR data found (raw or sample).")


def load_ground_features(hdmap_dir: Path, roi) -> list[dict]:
    """Load + ROI-clip the SHP layers that define ground surface, returning
    {category, geometry} dicts (same categories the pipeline feeds its ground
    model and road corridor)."""
    feats: list[dict] = []
    roi_box = box(*roi)
    for layer_key, default_cat in LAYER_TO_CATEGORY.items():
        p = hdmap_dir / LAYER_FILES[layer_key]
        if not p.exists():
            print(f"[SHP] skip missing: {p.name}")
            continue
        gdf = read_shp(str(p))
        if gdf is None or gdf.empty:
            continue
        gdf = gdf[gdf.intersects(roi_box)]
        n = 0
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            cat = b3_category_for(row) if layer_key == "B3_SURFACEMARK" else default_cat
            feats.append({"category": cat, "geometry": geom})
            n += 1
        print(f"[SHP] {layer_key}: {n} features in ROI")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["deoksong", "road86"], default="deoksong",
                    help="road86 = cross-region transfer test (same params).")
    ap.add_argument("--source", choices=["auto", "raw", "sample"], default="auto")
    ap.add_argument("--stride", type=int, default=4,
                    help="LAS decimation (keep every Nth point)")
    ap.add_argument("--bin", type=float, default=1.0,
                    help="XY cell size (m) for the LIDAR reference grid")
    ap.add_argument("--percentile", type=float, default=20.0,
                    help="per-cell low percentile of point Z used as the "
                         "measured road-surface elevation (robust to vehicles, "
                         "guardrails, and shoulder points that sit higher)")
    ap.add_argument("--min-points", type=int, default=8,
                    help="minimum LIDAR points per cell to form a reference")
    ap.add_argument("--buffer", type=float, default=ROAD_CORRIDOR_BUFFER_M,
                    help="road-corridor buffer (m) used to select road-surface "
                         "LIDAR points. Tighter = closer to the carriageway, "
                         "excludes sloped shoulders/embankment.")
    ap.add_argument("--out", default=None,
                    help="per-cell CSV path (default: eval/ground_z_errors[_<region>].csv)")
    args = ap.parse_args()

    if args.region == "road86":
        roi = ROAD86_ROI
        hdmap_dir, las_files = resolve_sources_road86()
        default_out = PROJECT_ROOT / "eval/ground_z_errors_road86.csv"
    else:
        roi = ROI_BBOX
        hdmap_dir, las_files = resolve_sources(args.source)
        default_out = PROJECT_ROOT / "eval/ground_z_errors.csv"
    if args.out is None:
        args.out = str(default_out)
    print(f"[REGION] {args.region}")
    print(f"[ROI] {roi}")
    print(f"[LAS] {len(las_files)} file(s), stride={args.stride}")

    # 1) SHP ground model — identical to the pipeline.
    feats = load_ground_features(hdmap_dir, roi)
    ground_feats = [f for f in feats if f["category"] in GROUND_SURFACE_CATEGORIES]
    tree, z_arr = build_ground_z_index(ground_feats)
    if tree is None:
        raise RuntimeError("no SHP ground vertices found — cannot evaluate")
    print(f"[MODEL] SHP ground vertices: {len(z_arr):,}")

    # 2) Road corridor (XY only) to isolate road-surface LIDAR points.
    corridor_geoms = [
        f["geometry"].buffer(args.buffer)
        for f in feats
        if f["category"] in ROAD_CORRIDOR_CATEGORIES
        and f["geometry"] is not None and not f["geometry"].is_empty
    ]
    if not corridor_geoms:
        raise RuntimeError("no corridor geometry (link/driveway/subsidiary)")
    corridor = unary_union(corridor_geoms)

    # 3) LIDAR points, clipped to ROI + corridor.
    pts = read_las_xyz(las_files, roi_bbox=roi, stride=args.stride)
    print(f"[LAS] {len(pts):,} points in ROI")
    if len(pts) == 0:
        raise RuntimeError("no LIDAR points in ROI")
    inside = shapely.contains_xy(corridor, pts[:, 0], pts[:, 1])
    pts = pts[inside]
    print(f"[LAS] {len(pts):,} points inside road corridor "
          f"({args.buffer}m buffer)")
    if len(pts) == 0:
        raise RuntimeError("no LIDAR points inside corridor")

    # 4) Per-cell measured ground = low percentile of point Z.
    ox, oy = roi[0], roi[1]
    ix = np.floor((pts[:, 0] - ox) / args.bin).astype(np.int64)
    iy = np.floor((pts[:, 1] - oy) / args.bin).astype(np.int64)
    df = pd.DataFrame({"ix": ix, "iy": iy, "z": pts[:, 2]})
    grp = df.groupby(["ix", "iy"])["z"]
    ref = grp.quantile(args.percentile / 100.0)
    # Per-cell vertical spread of points (p95 - p05). A large spread is the
    # signature of stacked surfaces at one (x,y) — i.e. a grade-separated
    # crossing (viaduct over an IC) where the corridor catches both the
    # lower road and the upper deck. We use it to test whether the large
    # ground-Z errors concentrate exactly where the road is multi-level.
    z_p05 = grp.quantile(0.05)
    z_p95 = grp.quantile(0.95)
    cnt = grp.size()
    keep = cnt >= args.min_points
    ref = ref[keep]
    if len(ref) == 0:
        raise RuntimeError("no cells met --min-points threshold")

    idx = ref.index.to_frame(index=False)
    centers = np.column_stack((
        ox + (idx["ix"].to_numpy() + 0.5) * args.bin,
        oy + (idx["iy"].to_numpy() + 0.5) * args.bin,
    ))
    ref_z = ref.to_numpy()
    z_spread = (z_p95[keep] - z_p05[keep]).to_numpy()

    # 5) Predicted ground from the SHP model at each cell center.
    pred_z = query_ground_z(tree, z_arr, centers)
    err = pred_z - ref_z  # signed: positive = SHP sits above LIDAR surface
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(err))
    std = float(np.std(err))
    med = float(np.median(abs_err))
    p90 = float(np.percentile(abs_err, 90))
    p95 = float(np.percentile(abs_err, 95))
    emax = float(abs_err.max())

    print("\n=== GROUND-Z RECONSTRUCTION ACCURACY ===")
    print(f"compared cells     : {len(err):,}  "
          f"(bin={args.bin}m, p{args.percentile:g}, min_pts={args.min_points})")
    print(f"RMSE               : {rmse:.3f} m")
    print(f"MAE                : {mae:.3f} m")
    print(f"bias (pred - ref)  : {bias:+.3f} m   (datum/offset check)")
    print(f"std                : {std:.3f} m")
    print(f"median |err|       : {med:.3f} m")
    print(f"90th pct |err|     : {p90:.3f} m")
    print(f"95th pct |err|     : {p95:.3f} m")
    print(f"max |err|          : {emax:.3f} m")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ix", "iy", "center_x", "center_y",
                    "pred_z", "ref_z", "n_points", "z_spread", "error_m"])
        n_arr = cnt[keep].to_numpy()
        for i in range(len(err)):
            w.writerow([
                int(idx["ix"].iloc[i]), int(idx["iy"].iloc[i]),
                f"{centers[i, 0]:.3f}", f"{centers[i, 1]:.3f}",
                f"{pred_z[i]:.3f}", f"{ref_z[i]:.3f}",
                int(n_arr[i]), f"{z_spread[i]:.3f}", f"{err[i]:.4f}",
            ])
    print(f"\n[SAVE] per-cell errors → {out_path}")


if __name__ == "__main__":
    main()
