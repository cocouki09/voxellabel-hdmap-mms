"""Object-level results from the blind object review.

Object precision per category = share of output objects whose reviewed main
content is that category (Wilson 95% CI; all objects are reviewed, so this is
a census of the output objects, not a sample). Purity = reviewer's share
judgement (mostly / about half / mostly other). Map-conditioned recall =
mapped features of the category that produced an object.

  python eval/revision/eval_objects.py --review-dir <dir> [--labels labels.json] --out <md>
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_sys.stdout.reconfigure(encoding="utf-8")
CATS = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier"]
MAPPED = {"guardrail": 33, "traffic_light": 26, "sign": 25, "street_light": 16, "height_barrier": 2}


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.review_dir)
    key = json.load(open(d / "KEY_do_not_open_while_labeling.json", encoding="utf-8"))["key"]
    lab = json.load(open(args.labels or d / "labels.json", encoding="utf-8"))
    by = defaultdict(list)
    for k in key:
        L = lab.get(str(k["sample_id"]))
        by[k["category"]].append(L if L and L.get("cat") else None)
    lines = ["# Object-level review", "",
             "| Category | Output objects | Reviewed | Correct main category (precision, Wilson 95% CI) | Purity: mostly / half / mostly other | Map-conditioned recall |",
             "|---|---:|---:|---|---|---|"]
    res = {}
    for c in CATS:
        items = by.get(c, [])
        rev = [x for x in items if x]
        ok = [x for x in rev if x["cat"] == c or x.get("cat2") == c]
        pur = Counter(x.get("purity") for x in ok)
        lo, hi = wilson(len(ok), len(rev))
        det = len(items)
        rec = min(det, MAPPED[c]) / MAPPED[c]
        res[c] = {"objects": len(items), "reviewed": len(rev), "correct": len(ok), "precision": len(ok) / len(rev) if rev else None,
                  "ci95": [lo, hi], "purity_of_correct": dict(pur), "wrong_as": dict(Counter(x["cat"] for x in rev if x not in ok)),
                  "mapped": MAPPED[c], "map_recall": rec}
        lines.append(f"| {c} | {len(items)} | {len(rev)} | {len(ok)}/{len(rev)} = {res[c]['precision'] if rev else float('nan'):.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"{pur.get('mostly', 0)} / {pur.get('about_half', 0)} / {pur.get('mostly_other', 0)} | {min(det, MAPPED[c])}/{MAPPED[c]} |")
    lines += ["", "Wrong objects were reviewed as: " + "; ".join(f"{c}: {v['wrong_as']}" for c, v in res.items() if v["wrong_as"])]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
