"""Object-level results under the delivered category definitions (R1 object evaluation).

Delivered definitions: 4 street_light = any post / street-light pole (map C6),
5 height_barrier = the overhead structure itself (viaduct / overpass, so
'viaduct_structure' answers count as correct). A lenient column also accepts
any pole-mounted category (traffic light / sign / post) for objects of those
three categories, because they share poles in the field. A post-hoc map Type
filter drops tunnel- and structure-mounted types (C3 Type 8, C1 Type 9,
C5 Type 1) from both the output objects and the mapped features.

  python eval/revision/eval_objects_delivered.py --review-dir <dir> --hdmap <dir> --out <md>
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

_sys.stdout.reconfigure(encoding="utf-8")
CATS = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier"]
LAYER = {"guardrail": "C3_VEHICLEPROTECTIONSAFETY", "traffic_light": "C1_TRAFFICLIGHT", "sign": "B1_SAFETYSIGN",
         "street_light": "C6_POSTPOINT", "height_barrier": "C5_HEIGHTBARRIER"}
ACCEPT = {"height_barrier": {"height_barrier", "viaduct_structure"}}
POLE = {"traffic_light", "sign", "street_light"}
TYPE_FILTER = {"C3_VEHICLEPROTECTIONSAFETY": {"8"}, "C1_TRAFFICLIGHT": {"9"}, "C5_HEIGHTBARRIER": {"1"}}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--hdmap", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.review_dir)
    K = json.load(open(d / "KEY_do_not_open_while_labeling.json", encoding="utf-8"))
    lab = json.load(open(args.labels or d / "labels.json", encoding="utf-8"))
    st = np.load(K["stages"].replace("\\", "/") if Path(K["stages"].replace("\\", "/")).exists() else K["stages"])
    xyz = st["origin"] + (st["cells"] + 0.5) * float(st["voxel_size"])
    roi = box(xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max())
    gdf = {L: gpd.read_file(Path(args.hdmap) / f"{L}.shp") for L in LAYER.values()}
    for k in K["key"]:
        L, i = k["shp_feature_id"].rsplit("_", 1)
        k["type"] = str(gdf[L].loc[int(i), "Type"])
        k["layer"] = L
    lines = ["# Object-level review — delivered definitions", "",
             f"Objects {len(K['key'])}; with a category {sum(1 for v in lab.values() if v.get('cat'))}.", ""]
    res = {}
    for filt in (False, True):
        tag = "type_filtered" if filt else "all"
        lines += [f"## {'Map Type filter (C3-8, C1-9, C5-1 removed)' if filt else 'All output objects'}", "",
                  "| Category | Mapped in ROI | Output objects | Reviewed | Correct (precision, Wilson 95% CI) | Correct incl. shared pole | Purity of correct: mostly / half / mostly other / not given | Object recall (correct / mapped) |",
                  "|---|---:|---:|---:|---|---|---|---|"]
        res[tag] = {}
        for c in CATS:
            L = LAYER[c]; g = gdf[L]
            inroi = g[g.intersects(roi)]
            if filt and L in TYPE_FILTER:
                inroi = inroi[~inroi["Type"].astype(str).isin(TYPE_FILTER[L])]
            items = [k for k in K["key"] if k["category"] == c and not (filt and k["type"] in TYPE_FILTER.get(L, ()))]
            rev = [(k, lab[str(k["sample_id"])]) for k in items if lab.get(str(k["sample_id"]), {}).get("cat")]
            acc = ACCEPT.get(c, {c})
            ok = [(k, v) for k, v in rev if v["cat"] in acc or v.get("cat2") in acc]
            accp = acc | (POLE if c in POLE else set())
            okp = [(k, v) for k, v in rev if v["cat"] in accp or v.get("cat2") in accp]
            pur = Counter(v.get("purity") for _, v in ok)
            lo, hi = wilson(len(ok), len(rev)); lo2, hi2 = wilson(len(okp), len(rev))
            nm = len(inroi)
            res[tag][c] = {"mapped": nm, "objects": len(items), "reviewed": len(rev), "correct": len(ok),
                           "correct_shared_pole": len(okp), "ci95": [lo, hi], "ci95_pole": [lo2, hi2],
                           "purity": {str(k): v for k, v in pur.items()},
                           "wrong_as": dict(Counter(v["cat"] for k, v in rev if (k, v) not in ok)),
                           "recall": len(ok) / nm if nm else None}
            p = len(ok) / len(rev) if rev else float("nan"); p2 = len(okp) / len(rev) if rev else float("nan")
            lines.append(f"| {c} | {nm} | {len(items)} | {len(rev)} | {len(ok)}/{len(rev)} = {p:.2f} [{lo:.2f}, {hi:.2f}] | "
                         f"{len(okp)}/{len(rev)} = {p2:.2f} [{lo2:.2f}, {hi2:.2f}] | "
                         f"{pur.get('mostly', 0)} / {pur.get('about_half', 0)} / {pur.get('mostly_other', 0)} / {pur.get(None, 0)} | "
                         f"{len(ok)}/{nm} = {len(ok) / nm if nm else float('nan'):.2f} |")
        R = res[tag]
        tr = sum(v["reviewed"] for v in R.values()); tc = sum(v["correct"] for v in R.values())
        tp = sum(v["correct_shared_pole"] for v in R.values()); tm = sum(v["mapped"] for v in R.values())
        lo, hi = wilson(tc, tr)
        lines += [f"| **all** | {tm} | {sum(v['objects'] for v in R.values())} | {tr} | {tc}/{tr} = {tc / tr:.2f} [{lo:.2f}, {hi:.2f}] | "
                  f"{tp}/{tr} = {tp / tr:.2f} | | {tc}/{tm} = {tc / tm:.2f} |", "",
                  "Wrong objects were reviewed as: " + "; ".join(f"{c}: {v['wrong_as']}" for c, v in R.items() if v["wrong_as"]), ""]
    unrev = [k["sample_id"] for k in K["key"] if not lab.get(str(k["sample_id"]), {}).get("cat")]
    lines += [f"Objects without a category answer: {unrev}"]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
