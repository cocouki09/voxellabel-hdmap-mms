"""Static-occupancy accuracy of the output scene (simulation-oriented metric).

For each predicted output class (and for all kept cells), the Horvitz–Thompson
share of cells whose reference label is static solid content, i.e. anything
except a vehicle or an empty / noise cell. 'cannot_judge' cells are left out.
95% CI from the same stratified bootstrap as eval_sample.py.

  python eval/revision/eval_occupancy.py --sample-dir <dir> --labels <json> [--corrections a.json b.json] --stages <npz> --out <md>
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from collections import Counter
from pathlib import Path

import numpy as np

_sys.stdout.reconfigure(encoding="utf-8")
NON_STATIC = {"vehicle", "noise_or_empty"}
N_BOOT, SEED = 2000, 20261016


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--corrections", nargs="*", default=None)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = [k for k in json.load(open(Path(a.sample_dir) / "KEY_do_not_open_while_labeling.json", encoding="utf-8"))["key"]
           if "retest_of" not in k]
    lab = json.load(open(a.labels, encoding="utf-8"))
    for cf in a.corrections or []:
        for sid, c in json.load(open(cf, encoding="utf-8"))["corrections"].items():
            lab[sid] = {**(lab.get(sid) or {}), "cat": c["cat"]}
    st = np.load(a.stages)
    rows, ok, strata, N_h = [], [], [], {}
    for k in key:
        N_h[k["stratum"]] = k["N_h"]
        c = (lab.get(str(k["sample_id"])) or {}).get("cat")
        if not c or c == "cannot_judge":
            continue
        rows.append(k["row"]); ok.append(c not in NON_STATIC); strata.append(k["stratum"])
    rows = np.asarray(rows); ok = np.asarray(ok); strata = np.asarray(strata)
    n_h = Counter(strata.tolist())
    w = np.asarray([N_h[h] / n_h[h] for h in strata])
    kept = st["keep"][rows] & ~st["drop_dynamic"][rows]
    pred = np.where(kept, st["labels_final"][rows].astype(str), "<removed>")
    groups = sorted(set(pred[kept])) + ["all kept cells"]

    def est(idx):
        out = {}
        for g in groups:
            m = kept[idx] if g == "all kept cells" else pred[idx] == g
            out[g] = float(w[idx][m & ok[idx]].sum() / w[idx][m].sum()) if w[idx][m].sum() > 0 else np.nan
        return out

    pt = est(np.arange(len(rows)))
    rng = np.random.default_rng(SEED)
    boots = [est(np.concatenate([rng.choice(np.where(strata == h)[0], n, replace=True) for h, n in n_h.items()]))
             for _ in range(N_BOOT)]
    lines = ["# Static-occupancy accuracy (share of output cells that are static solid content)", "",
             f"stages {a.stages}; non-static = {sorted(NON_STATIC)}; cannot_judge excluded", "",
             "| Output class | Sample cells | Static-occupancy accuracy [95% CI] |", "|---|---:|---|"]
    res = {}
    for g in groups:
        b = [x[g] for x in boots if np.isfinite(x[g])]
        lo, hi = np.percentile(b, [2.5, 97.5])
        n = int(kept.sum()) if g == "all kept cells" else int((pred == g).sum())
        res[g] = {"acc": pt[g], "ci95": [float(lo), float(hi)], "n": n}
        lines.append(f"| {g} | {n} | {pt[g]:.3f} [{lo:.3f}, {hi:.3f}] |")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
