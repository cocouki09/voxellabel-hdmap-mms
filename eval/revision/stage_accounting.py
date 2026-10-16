"""IJGI revision W08 / W02 — stage-by-stage cell accounting and run comparison.

  python eval/revision/stage_accounting.py --stages A.npz [--compare B.npz] --out-dir <dir>

Accounting (R3-7): every occupied cell gets exactly one primary removal reason,
applied in pipeline order, so that
  occupied = kept + sum(primary removals)
holds exactly. Overlaps between the two unknown filters (low elevation and
outside the retention buffer) are reported separately.

Comparison (W02): cell-by-cell transition table of final outcomes between two
runs on the same cells (e.g. sign radius 2 m vs 1 m), plus object counts.
"""
from __future__ import annotations

import argparse
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(p):
    st = dict(np.load(p, allow_pickle=False))
    side = Path(p).with_suffix(".objects.json")
    st["_objects"] = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    return st


def outcome(st):
    o = np.where(st["keep"], np.char.add("kept:", st["labels_final"].astype(str)), "")
    o = o.astype(object)
    o[st["drop_ground"]] = "drop:ground (map overlay)"
    o[st["drop_dynamic"]] = "drop:vehicle filter"
    low = st["drop_off_road_low"] & (o == "")
    o[low] = "drop:unknown below 0.3 m"
    far = st["drop_out_of_shp"] & (o == "")
    o[far] = "drop:unknown outside 12 m buffer"
    assert (o != "").all()
    return o


def accounting(st):
    n = len(st["cells"])
    o = outcome(st)
    rows = [("occupied cells after voxelization", n, None)]
    lab0 = Counter(st["labels_labeling"].astype(str))
    rows.append(("labeled (non-unknown) after Z-aware labeling", n - lab0.get("unknown", 0), None))
    rows.append(("vehicle candidates (Eq. 5, before protection)", int(st["vehicle_candidate"].sum()), None))
    rows.append(("  of which pole-protected (kept)", int(st["pole_protected"].sum()), None))
    remaining = n
    for reason in ["drop:ground (map overlay)", "drop:vehicle filter",
                   "drop:unknown below 0.3 m", "drop:unknown outside 12 m buffer"]:
        k = int((o == reason).sum())
        remaining -= k
        rows.append((f"− {reason}", k, remaining))
    kept = int(st["keep"].sum())
    assert remaining == kept, (remaining, kept)
    rows.append(("final scene cells", kept, None))
    overlap = int((st["drop_off_road_low"] & st["drop_out_of_shp"]).sum())
    comp = Counter(st["labels_final"][st["keep"]].astype(str))
    return {"rows": rows, "overlap_low_and_outside": overlap,
            "removed_total": n - kept, "removed_pct": 100 * (n - kept) / n,
            "pre_filter_labels": dict(lab0), "final_composition": dict(comp),
            "n_points": int(st["n_points"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    A = load(args.stages)
    acc = accounting(A)
    lines = ["# Stage-by-stage cell accounting", "",
             f"points after ROI clip and stride: {acc['n_points']:,}", "",
             "| stage | cells | remaining |", "|---|---:|---:|"]
    for name, k, rem in acc["rows"]:
        lines.append(f"| {name} | {k:,} | {'' if rem is None else f'{rem:,}'} |")
    lines += ["", f"removed in total: {acc['removed_total']:,} ({acc['removed_pct']:.1f}%)",
              f"cells meeting both unknown filters (counted once above, under 'below 0.3 m'): "
              f"{acc['overlap_low_and_outside']:,}", "",
              "final composition: " + ", ".join(f"{k} {v:,}" for k, v in
                                               sorted(acc["final_composition"].items(), key=lambda x: -x[1]))]
    res = {"A": {"stages": args.stages, **{k: v for k, v in acc.items() if k != "rows"},
                 "rows": [list(r) for r in acc["rows"]]}}

    if args.compare:
        B = load(args.compare)
        assert np.array_equal(A["cells"], B["cells"]), "runs must share the same occupied cells"
        oa, ob = outcome(A), outcome(B)
        trans = Counter(zip(oa, ob))
        changed = {f"{a} -> {b}": n for (a, b), n in trans.items() if a != b}
        objs = lambda S: Counter(v["category"] for v in S["_objects"]["objects"].values()) if S["_objects"] else {}  # noqa: E731
        res["compare"] = {"B": args.compare, "n_changed": int((oa != ob).sum()),
                          "transitions_changed": changed,
                          "objects_A": dict(objs(A)), "objects_B": dict(objs(B)),
                          "composites_A": len(A["_objects"]["composites"]) if A["_objects"] else None,
                          "composites_B": len(B["_objects"]["composites"]) if B["_objects"] else None,
                          "final_A": acc["final_composition"],
                          "final_B": dict(Counter(B["labels_final"][B["keep"]].astype(str)))}
        lines += ["", f"## Comparison A = {Path(args.stages).parent.name}  B = {Path(args.compare).parent.name}", "",
                  f"cells whose final outcome differs: {res['compare']['n_changed']:,}", ""]
        lines += [f"- {k}: {v:,}" for k, v in sorted(changed.items(), key=lambda x: -x[1])]
        lines += ["", f"objects A {res['compare']['objects_A']}", f"objects B {res['compare']['objects_B']}",
                  f"composites A {res['compare']['composites_A']} / B {res['compare']['composites_B']}"]
    (out / "accounting.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "accounting.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
