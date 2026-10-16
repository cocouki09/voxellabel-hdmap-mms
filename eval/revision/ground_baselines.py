"""IJGI revision W05 — ground model vs. geometric baselines on the SAME inputs.

Reviewer requests (R1-3, R2 Results, R3-5/8): compare the map-vertex ground
model with common geometric alternatives instead of a constant only, and
report a full error distribution with uncertainty.

All methods use the same SHP ground-surface vertices (the map's 3D Z) and are
evaluated against the same LiDAR reference cells that eval/eval_ground_z.py
builds (1 m cells, p20 of road-corridor point Z, >= 8 points). The reference
uses only the map's XY to select road points, so no method sees the reference.

Methods (parameters fixed before looking at results):
  constant  single z = median(ref)              (submitted baseline, kept as is)
  nearest   nearest map vertex Z                (= proposed model, Eq. 2)
  idw       inverse-distance weighting, k=8, p=2, search radius 20 m
  tin       linear interpolation on a Delaunay TIN of the vertices
            (duplicate XY within 1 cm averaged; NaN outside the convex hull)
  plane     local least-squares plane on the k=12 nearest vertices within 20 m

Outputs (to --out-dir, never to the submitted eval/ files):
  cells.csv      per-cell ref and every method's prediction
  summary.json   stats per method, own coverage and common-mask comparison,
                 block-bootstrap 95% CIs, pre-defined strata
  summary.md     the same as tables

Run (from voxel_scene_pipeline/):
  set PYTHONPATH=.
  python eval/revision/ground_baselines.py --region deoksong --source raw --out-dir ../output/ijgi_revision/W05_GROUND_BASELINES_deoksong
"""
from __future__ import annotations

import argparse
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/
import eval_ground_z as egz  # noqa: E402  (reuse the submitted loaders as-is)

from src.readers.las_reader import read_las_xyz  # noqa: E402
from src.semantic.ground_models import (  # noqa: E402
    IDW_K, IDW_P, PLANE_K, SEARCH_R, predict_idw, predict_plane, predict_tin,
)
from src.semantic.shp_matcher import (  # noqa: E402
    GROUND_SURFACE_CATEGORIES,
    build_ground_z_index,
    query_ground_z,
)

BLOCK_M = 50.0
N_BOOT = 1000
SEED = 20261014
# Strata fixed a priori: per-cell vertical spread of LiDAR points (p95-p05)
# flags stacked surfaces; map-derived local grade flags ramps.
SPREAD_BINS = [(0.0, 0.3, "single surface (spread<0.3 m)"),
               (0.3, 1.0, "mixed (0.3-1.0 m)"),
               (1.0, np.inf, "stacked/overhead (spread>=1.0 m)")]
GRADE_BINS = [(0.0, 0.02, "flat (<2%)"),
              (0.02, 0.05, "moderate (2-5%)"),
              (0.05, np.inf, "steep (>=5%)")]


def reference_cells(pts, roi, bin_m, pct, min_pts):
    ox, oy = roi[0], roi[1]
    ix = np.floor((pts[:, 0] - ox) / bin_m).astype(np.int64)
    iy = np.floor((pts[:, 1] - oy) / bin_m).astype(np.int64)
    df = pd.DataFrame({"ix": ix, "iy": iy, "z": pts[:, 2]})
    g = df.groupby(["ix", "iy"])["z"]
    out = pd.DataFrame({
        "ref_z": g.quantile(pct / 100.0),
        "p05": g.quantile(0.05),
        "p95": g.quantile(0.95),
        "n_points": g.size(),
    })
    out = out[out["n_points"] >= min_pts].reset_index()
    out["center_x"] = ox + (out["ix"] + 0.5) * bin_m
    out["center_y"] = oy + (out["iy"] + 0.5) * bin_m
    out["z_spread"] = out["p95"] - out["p05"]
    return out.drop(columns=["p05", "p95"])


def stats(err):
    e = err[np.isfinite(err)]
    a = np.abs(e)
    if len(e) == 0:
        return {"n": 0}
    q1, q3 = np.percentile(e, [25, 75])
    return {
        "n": int(len(e)),
        "bias": float(e.mean()),
        "sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
        "median_abs": float(np.median(a)),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "iqr_signed": float(q3 - q1),
        "p90_abs": float(np.percentile(a, 90)),
        "p95_abs": float(np.percentile(a, 95)),
        "p99_abs": float(np.percentile(a, 99)),
        "max_abs": float(a.max()),
        "frac_le_0p2": float((a <= 0.2).mean()),
    }


def block_bootstrap(err, block_id, rng):
    """95% CI for median|e|, MAE, RMSE resampling 50 m spatial blocks."""
    ok = np.isfinite(err)
    err, block_id = err[ok], block_id[ok]
    blocks = np.unique(block_id)
    groups = {b: np.abs(err[block_id == b]) for b in blocks}
    sq = {b: err[block_id == b] ** 2 for b in blocks}
    med, mae, rmse = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        a = np.concatenate([groups[b] for b in pick])
        s = np.concatenate([sq[b] for b in pick])
        med.append(np.median(a)); mae.append(a.mean()); rmse.append(np.sqrt(s.mean()))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"n_blocks": int(len(blocks)), "block_m": BLOCK_M, "n_boot": N_BOOT,
            "median_abs_ci95": ci(med), "mae_ci95": ci(mae), "rmse_ci95": ci(rmse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["deoksong", "road86"], default="deoksong")
    ap.add_argument("--source", choices=["auto", "raw", "sample"], default="raw")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--bin", type=float, default=1.0)
    ap.add_argument("--percentile", type=float, default=20.0)
    ap.add_argument("--min-points", type=int, default=8)
    ap.add_argument("--buffer", type=float, default=egz.ROAD_CORRIDOR_BUFFER_M)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default=None,
                    help="voxel_scene_pipeline folder holding data/ (default: this checkout)")
    args = ap.parse_args()
    t0 = time.time()
    if args.data_root:
        root = Path(args.data_root)
        egz.LIDAR_ROOT = root / egz.LIDAR_ROOT.relative_to(egz.PROJECT_ROOT)
        egz.SAMPLE_ROOT = root / egz.SAMPLE_ROOT.relative_to(egz.PROJECT_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run

    if args.region == "road86":
        roi = egz.ROAD86_ROI
        hdmap_dir, las_files = egz.resolve_sources_road86()
    else:
        roi = egz.ROI_BBOX
        hdmap_dir, las_files = egz.resolve_sources(args.source)
        # resolve_sources silently falls back to the sample when raw is absent
        if args.source == "raw" and "data/raw" not in Path(hdmap_dir).as_posix():
            raise RuntimeError(f"raw requested but resolved to {hdmap_dir}")

    feats = egz.load_ground_features(hdmap_dir, roi)
    gfeats = [f for f in feats if f["category"] in GROUND_SURFACE_CATEGORIES]
    tree, zv = build_ground_z_index(gfeats)
    xyv = tree.data.copy()
    print(f"[MODEL] ground vertices: {len(zv):,}")

    corridor = unary_union([f["geometry"].buffer(args.buffer) for f in feats
                            if f["category"] in egz.ROAD_CORRIDOR_CATEGORIES
                            and f["geometry"] is not None and not f["geometry"].is_empty])
    pts = read_las_xyz(las_files, roi_bbox=roi, stride=args.stride)
    n_roi = len(pts)
    pts = pts[shapely.contains_xy(corridor, pts[:, 0], pts[:, 1])]
    print(f"[LAS] {n_roi:,} pts in ROI, {len(pts):,} in corridor")
    cells = reference_cells(pts, roi, args.bin, args.percentile, args.min_points)
    q = cells[["center_x", "center_y"]].to_numpy()
    ref = cells["ref_z"].to_numpy()
    print(f"[REF] {len(cells):,} reference cells")

    preds = {}
    preds["constant"] = np.full(len(q), float(np.median(ref)))
    preds["nearest"] = query_ground_z(tree, zv, q)
    preds["idw"] = predict_idw(tree, zv, q)
    preds["tin"], n_dup = predict_tin(xyv, zv, q)
    preds["plane"], grade = predict_plane(tree, xyv, zv, q)
    for k, v in preds.items():
        cells[f"pred_{k}"] = v
    cells["local_grade"] = grade

    block_id = (np.floor((q[:, 0] - roi[0]) / BLOCK_M).astype(np.int64) * 100000
                + np.floor((q[:, 1] - roi[1]) / BLOCK_M).astype(np.int64))
    common = np.all(np.column_stack([np.isfinite(v) for v in preds.values()]), axis=1)
    rng = np.random.default_rng(SEED)

    res = {"own_coverage": {}, "common_mask": {}, "strata": {}}
    for k, v in preds.items():
        e = v - ref
        res["own_coverage"][k] = {"coverage": float(np.isfinite(v).mean()), **stats(e)}
        ec = np.where(common, e, np.nan)
        res["common_mask"][k] = {**stats(ec), **block_bootstrap(ec, block_id, rng)}
    spread = cells["z_spread"].to_numpy()
    for lo, hi, name in SPREAD_BINS:
        m = common & (spread >= lo) & (spread < hi)
        res["strata"][name] = {k: stats(np.where(m, v - ref, np.nan)) for k, v in preds.items()}
    for lo, hi, name in GRADE_BINS:
        m = common & np.isfinite(grade) & (grade >= lo) & (grade < hi)
        res["strata"][name] = {k: stats(np.where(m, v - ref, np.nan)) for k, v in preds.items()}

    meta = {
        "experiment_id": f"W05_GROUND_BASELINES_{args.region}",
        "region": args.region, "roi": list(roi), "hdmap_dir": str(hdmap_dir),
        "n_las": len(las_files), "las_files": [str(p) for p in las_files],
        "args": vars(args), "n_points_roi": int(n_roi), "n_points_corridor": int(len(pts)),
        "n_vertices": int(len(zv)), "tin_duplicate_xy_merged": n_dup,
        "n_cells": int(len(cells)), "n_common": int(common.sum()),
        "params": {"idw_k": IDW_K, "idw_p": IDW_P, "search_r": SEARCH_R, "plane_k": PLANE_K,
                   "block_m": BLOCK_M, "n_boot": N_BOOT, "seed": SEED},
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "runtime_s": round(time.time() - t0, 1),
    }
    cells.to_csv(out_dir / "cells.csv", index=False, float_format="%.4f")
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, **res}, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["coverage", "n", "bias", "sd", "median_abs", "mae", "rmse", "iqr_signed",
            "p90_abs", "p95_abs", "p99_abs", "max_abs", "frac_le_0p2"]
    lines = [f"# {meta['experiment_id']}", "",
             f"cells {meta['n_cells']:,}, common mask {meta['n_common']:,}, vertices {meta['n_vertices']:,}", "",
             "## Own coverage", "", "| method | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for k, s in res["own_coverage"].items():
        lines.append(f"| {k} | " + " | ".join(f"{s.get(c, float('nan')):.4f}" for c in cols) + " |")
    lines += ["", "## Common mask with 95% block-bootstrap CI", "",
              "| method | n | median_abs [CI] | MAE [CI] | RMSE [CI] | bias | P95 | P99 | max |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, s in res["common_mask"].items():
        lines.append(
            f"| {k} | {s['n']} | {s['median_abs']:.3f} [{s['median_abs_ci95'][0]:.3f}, {s['median_abs_ci95'][1]:.3f}]"
            f" | {s['mae']:.3f} [{s['mae_ci95'][0]:.3f}, {s['mae_ci95'][1]:.3f}]"
            f" | {s['rmse']:.3f} [{s['rmse_ci95'][0]:.3f}, {s['rmse_ci95'][1]:.3f}]"
            f" | {s['bias']:+.3f} | {s['p95_abs']:.3f} | {s['p99_abs']:.3f} | {s['max_abs']:.3f} |")
    lines += ["", "## Strata (common mask, median_abs / MAE / RMSE, n)", ""]
    for name, d in res["strata"].items():
        lines.append(f"- {name}: " + "; ".join(
            f"{k} {s['median_abs']:.3f}/{s['mae']:.3f}/{s['rmse']:.3f} (n={s['n']})"
            if s.get("n") else f"{k} n=0" for k, s in d.items()))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[DONE] {meta['runtime_s']} s → {out_dir}")


if __name__ == "__main__":
    main()
