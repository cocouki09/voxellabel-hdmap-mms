"""IJGI revision W06 — exact denominators and uncertainty for the unknown audit.

Reviewer requests (R1-5, R2, R3-3/4): report what the 93.5% / 2.9% are a
fraction OF, and give confidence intervals.

Reads the manual audit file (cell-edits-deoksong-0526.json) directly. Each
record is one 0.5 m cell with cellKey "ix,iy,iz", the category the pipeline
gave it (originalCategory) and the operator's category (overrideCategory).

Two denominators are reported:
  submitted   all 96,249 records, as in the submitted manuscript
  unknown     only records whose originalCategory == "unknown" (the quantity
              the text describes: "composition of audited unknown cells")

Two CIs per proportion:
  wilson      binomial Wilson 95% CI, treating cells as independent (what R3
              asked for; too narrow because cells were painted in clusters)
  block       spatial block bootstrap 95% CI (blocks of B m in XY; 25 m and
              50 m), resampling blocks with replacement

Both CIs are conditional on the operator-selected subset; they quantify
sampling noise within that subset, not how representative the subset is of
all unknown cells (selection bias is not removed by a CI).

Run:
  python eval/revision/audit_ci.py --labels ../HANDOFF/labels/cell-edits-deoksong-0526.json --out-dir ../output/ijgi_revision/W06_AUDIT_CI
"""
from __future__ import annotations

import argparse
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import json
from collections import Counter
from pathlib import Path

import numpy as np

VOXEL_M = 0.5
N_BOOT = 2000
SEED = 20261014
BUCKET = {
    "식물": "vegetation (no map class)",
    "height_barrier": "viaduct structure (map has centerline only)",
    "차량": "vehicle (escaped filter)",
    "guardrail": "map-representable miss",
    "street_light": "map-representable miss",
    "traffic_light": "map-representable miss",
    "sign": "map-representable miss",
    "표지판2": "map-representable miss",
}
OUT_OF_MAP = {"vegetation (no map class)", "viaduct structure (map has centerline only)"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def block_ci(ix, iy, flag, block_m, rng):
    b = (np.floor(ix * VOXEL_M / block_m).astype(np.int64) * 1_000_000
         + np.floor(iy * VOXEL_M / block_m).astype(np.int64))
    ub, inv = np.unique(b, return_inverse=True)
    k_b = np.bincount(inv, weights=flag.astype(float))
    n_b = np.bincount(inv).astype(float)
    est = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ub), len(ub))
        est.append(k_b[pick].sum() / n_b[pick].sum())
    return {"block_m": block_m, "n_blocks": int(len(ub)),
            "ci95": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]}


def analyse(records, rng):
    keys = np.array([[int(v) for v in r["cellKey"].split(",")] for r in records])
    ix, iy = keys[:, 0], keys[:, 1]
    buckets = np.array([BUCKET.get(r["overrideCategory"], "other") for r in records])
    n = len(records)
    out = {"n": n, "unique_cells": int(len({r["cellKey"] for r in records})),
           "override_counts": dict(Counter(r["overrideCategory"] for r in records)),
           "proportions": {}}
    groups = {name: buckets == name for name in sorted(set(buckets))}
    groups["out-of-map total (vegetation + viaduct)"] = np.isin(buckets, list(OUT_OF_MAP))
    for name, flag in groups.items():
        k = int(flag.sum())
        out["proportions"][name] = {
            "k": k, "p": k / n, "wilson95": wilson(k, n),
            "block": [block_ci(ix, iy, flag, bm, rng) for bm in (25.0, 50.0)],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rec = json.load(open(args.labels, encoding="utf-8"))["cell_overrides"]
    rng = np.random.default_rng(SEED)
    res = {
        "labels": str(Path(args.labels).resolve()),
        "original_category_counts": dict(Counter(r["originalCategory"] for r in rec)),
        "submitted_all_records": analyse(rec, rng),
        "unknown_only": analyse([r for r in rec if r["originalCategory"] == "unknown"], rng),
        "params": {"voxel_m": VOXEL_M, "n_boot": N_BOOT, "seed": SEED},
    }
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# W06 audit composition with CIs", "",
             f"original categories: {res['original_category_counts']}", ""]
    for scope in ("submitted_all_records", "unknown_only"):
        s = res[scope]
        lines += [f"## {scope} (n={s['n']:,}, unique cells {s['unique_cells']:,})", "",
                  "| bucket | k | % | Wilson 95% | block 25 m 95% | block 50 m 95% |",
                  "|---|---|---|---|---|---|"]
        for name, p in s["proportions"].items():
            w = p["wilson95"]; b25 = p["block"][0]; b50 = p["block"][1]
            lines.append(f"| {name} | {p['k']:,} | {100*p['p']:.2f} | {100*w[0]:.2f}–{100*w[1]:.2f}"
                         f" | {100*b25['ci95'][0]:.2f}–{100*b25['ci95'][1]:.2f} ({b25['n_blocks']} blk)"
                         f" | {100*b50['ci95'][0]:.2f}–{100*b50['ci95'][1]:.2f} ({b50['n_blocks']} blk) |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
