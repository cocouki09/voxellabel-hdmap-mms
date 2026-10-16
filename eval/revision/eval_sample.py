"""IJGI revision W04/W07/W08 — design-based accuracy from the blind cell sample.

The sample (sample_cells.py) is a stratified random sample of occupied cells
of the base run; each sample i has design weight w_i = N_h / n_h. Any run on
the same occupied cells (base, ablations, sensitivity) is scored on the same
labeled sample with the Horvitz–Thompson estimator, per output category c:

  pred_i = c  iff cell i is kept in the final scene with label c
  TP_c = sum_i w_i [pred_i = c][true_i = c]
  FP_c = sum_i w_i [pred_i = c][true_i != c]
  FN_c = sum_i w_i [pred_i != c][true_i = c]    (a filtered real pole is a miss)
  P = TP/(TP+FP), R = TP/(TP+FN), F1, IoU = TP/(TP+FP+FN)

Samples labeled cannot_judge / noise_or_empty are excluded and their weight
is redistributed within the stratum (reported). 95% CIs: stratified
bootstrap (resample labeled samples within each stratum), 2000 replicates.

Also: composition of the final unknown cells, vehicle-filter removal
precision and false removal of infrastructure, residual vehicles.

Run:
  python eval/revision/eval_sample.py --sample-dir S --out-dir O                # base run
  python eval/revision/eval_sample.py --sample-dir S --runs A=a.npz B=b.npz --out-dir O
"""
from __future__ import annotations

import argparse
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EVAL_CATS = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier"]
TRUE_ALIAS = {"tunnel_structure": "tunnel"}
IGNORE = {"cannot_judge", "noise_or_empty"}
INFRA = set(EVAL_CATS) | {"tunnel"}
N_BOOT = 2000
SEED = 20261016


def predicted(st, rows):
    keep = st["keep"][rows]
    lab = st["labels_final"][rows].astype(str)
    pred = np.where(keep, lab, "<dropped>")
    pred = np.where(st["drop_dynamic"][rows], "<vehicle_removed>", pred)
    return pred


def ht(pred, true, w):
    """Horvitz–Thompson per-category metrics for arrays of equal length."""
    out = {}
    for c in EVAL_CATS:
        pc, tc = pred == c, true == c
        tp, fp, fn = w[pc & tc].sum(), w[pc & ~tc].sum(), w[~pc & tc].sum()
        P = tp / (tp + fp) if tp + fp > 0 else np.nan
        R = tp / (tp + fn) if tp + fn > 0 else np.nan
        F1 = 2 * P * R / (P + R) if np.isfinite(P) and np.isfinite(R) and P + R > 0 else np.nan
        if tp == 0 and tp + fn > 0:  # category present in the reference but nothing correct: F1 = 0, not undefined
            F1 = 0.0
        IoU = tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan
        out[c] = {"TP": tp, "FP": fp, "FN": fn, "precision": P, "recall": R, "f1": F1, "iou": IoU,
                  "n_pred_samples": int(pc.sum()), "n_true_samples": int(tc.sum())}
    f1 = [v["f1"] for v in out.values() if np.isfinite(v["f1"])]
    iou = [v["iou"] for v in out.values() if np.isfinite(v["iou"])]
    out["_macro"] = {"f1": float(np.mean(f1)) if f1 else np.nan,
                     "iou": float(np.mean(iou)) if iou else np.nan}
    unk = pred == "unknown"
    if unk.any():
        comp = defaultdict(float)
        for t, ww in zip(true[unk], w[unk]):
            comp[t] += ww
        tot = sum(comp.values())
        out["_unknown_composition"] = {t: v / tot for t, v in comp.items()}
        out["_unknown_cells_est"] = tot
    rem = pred == "<vehicle_removed>"
    if rem.any():
        wr = w[rem].sum()
        out["_vehicle_filter"] = {
            "removed_cells_est": float(wr),
            "removal_precision": float(w[rem & (true == "vehicle")].sum() / wr),
            "false_removal_infra_share": float(w[rem & np.isin(true, list(INFRA))].sum() / wr),
        }
    kept = ~np.isin(pred, ["<dropped>", "<vehicle_removed>"])
    out["_residual_vehicle_cells"] = float(w[kept & (true == "vehicle")].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="NAME=stages.npz pairs; default = the base run the sample came from")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--promote-secondary", action="store_true",
                    help="sensitivity: if the primary label is noise/cannot-judge but a real second category was given, use it")
    ap.add_argument("--definition", choices=["strict", "delivered"], default="strict",
                    help="delivered: score the categories as the viewer defines them - height_barrier = overhead "
                         "structure (viaduct), street_light = post/street light (any pole, from --post-labels), "
                         "plus the viewer's tunnel category")
    ap.add_argument("--corrections", nargs="*", default=None, help="JSON files {corrections: {sample_id: {cat}}} applied over --labels, in order")
    ap.add_argument("--post-labels", default=None, help="labels.json of the post re-check (post_related / not_post)")
    ap.add_argument("--exclude-ambiguous", action="store_true",
                    help="sensitivity: drop samples marked unsure or given a second category")
    args = ap.parse_args()
    sd = Path(args.sample_dir)
    K = json.load(open(sd / "KEY_do_not_open_while_labeling.json", encoding="utf-8"))
    key = K["key"]
    lab = json.load(open(args.labels or sd / "labels.json", encoding="utf-8"))
    if args.corrections:
        for cf in args.corrections:
          for sid, c in json.load(open(cf, encoding="utf-8"))["corrections"].items():
            lab[sid] = {**(lab.get(sid) or {}), "cat": c["cat"], "cat2": c.get("cat2"), "corrected": True}

    rows, true, strata, unsure = [], [], [], []
    alts = []   # acceptable categories per sample (delivered definition)
    post = json.load(open(args.post_labels, encoding="utf-8")) if args.post_labels else {}
    if args.definition == "delivered" and "tunnel" not in EVAL_CATS:
        EVAL_CATS.append("tunnel")
    true2 = []  # optional second category for mixed cells (Shift+key in the labeler)
    n_lab_h, N_h = Counter(), {}
    ignored = Counter(); unlabeled = 0
    retest = [k for k in key if "retest_of" in k]
    key = [k for k in key if "retest_of" not in k]
    for k in key:
        N_h[k["stratum"]] = k["N_h"]
        L = lab.get(str(k["sample_id"]))
        if L is None or not L.get("cat"):
            unlabeled += 1
            continue
        if args.promote_secondary and L["cat"] in IGNORE and L.get("cat2") and L["cat2"] not in IGNORE:
            L = {**L, "cat": L["cat2"], "cat2": None}
        if L["cat"] in IGNORE:
            ignored[k["stratum"]] += 1
            continue
        if args.exclude_ambiguous and (L.get("unsure") or L.get("cat2")):
            ignored[k["stratum"]] += 1
            continue
        rows.append(k["row"]); true.append(TRUE_ALIAS.get(L["cat"], L["cat"]))
        alts.append({TRUE_ALIAS.get(x, x) for x in (L["cat"], L.get("cat2")) if x})
        if args.definition == "delivered":
            if "viaduct_structure" in alts[-1]:
                alts[-1].add("height_barrier")
            if post.get(str(k["sample_id"]), {}).get("cat") == "post_related":
                alts[-1].add("street_light")
        true2.append(TRUE_ALIAS.get(L.get("cat2") or "", L.get("cat2") or ""))
        strata.append(k["stratum"]); unsure.append(bool(L.get("unsure")))
        n_lab_h[k["stratum"]] += 1
    rows = np.asarray(rows); true = np.asarray(true); strata = np.asarray(strata)
    true_base = true.copy()
    if args.definition == "delivered":  # the viewer's 'height_barrier' (C5) is the overhead structure itself
        true_base = np.where(true_base == "viaduct_structure", "height_barrier", true_base)
    true2 = np.asarray(true2)
    w = np.asarray([N_h[h] / n_lab_h[h] for h in strata], dtype=float)

    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else {"base": K["stages"]}
    rng = np.random.default_rng(SEED)
    boot_idx = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(np.where(strata == h)[0], size=n, replace=True)
                              for h, n in n_lab_h.items()])
        boot_idx.append(idx)

    report = {"sample_dir": str(sd.resolve()), "n_samples": len(key), "n_used": int(len(rows)),
              "n_unlabeled": unlabeled, "ignored_by_stratum": dict(ignored),
              "n_unsure": int(np.sum(unsure)),
              "strata": {h: {"N_h": N_h[h], "n_used": int(n_lab_h.get(h, 0)),
                             "true_counts": dict(Counter(true[strata == h]))} for h in N_h},
              "runs": {}}
    md = ["# Blind-sample evaluation (Horvitz–Thompson, stratified bootstrap 95% CI)", "",
          f"samples {len(key)}, used {len(rows)}, unlabeled {unlabeled}, ignored {sum(ignored.values())}, "
          f"marked unsure {int(np.sum(unsure))}", ""]
    for name, path in runs.items():
        st = dict(np.load(path, allow_pickle=False))
        pred = predicted(st, rows)
        if args.definition == "delivered":  # a prediction among the acceptable categories counts as its truth
            true = np.asarray([p_ if p_ in a_ else t_ for p_, a_, t_ in zip(pred, alts, true_base)])
        point = ht(pred, true, w)
        boots = [ht(pred[i], true[i], w[i]) for i in boot_idx]

        def ci(c, m):
            v = [b[c][m] for b in boots if c in b and np.isfinite(b[c][m])]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [np.nan, np.nan]

        r = {"stages": path, "per_category": {}}
        for c in EVAL_CATS:
            r["per_category"][c] = {**{k: float(v) for k, v in point[c].items()},
                                    **{f"{m}_ci95": ci(c, m) for m in ("precision", "recall", "f1", "iou")}}
        r["macro"] = {**point["_macro"], "f1_ci95": ci("_macro", "f1"), "iou_ci95": ci("_macro", "iou")}
        for extra in ("_unknown_composition", "_unknown_cells_est", "_vehicle_filter", "_residual_vehicle_cells"):
            if extra in point:
                r[extra.strip("_")] = point[extra]
        if "_vehicle_filter" in point:
            r["vehicle_filter"]["removal_precision_ci95"] = ci("_vehicle_filter", "removal_precision")
            r["vehicle_filter"]["false_removal_infra_share_ci95"] = ci("_vehicle_filter", "false_removal_infra_share")
        rv = [b["_residual_vehicle_cells"] for b in boots]
        r["residual_vehicle_cells_ci95"] = [float(np.percentile(rv, 2.5)), float(np.percentile(rv, 97.5))]
        # lenient scoring: a mixed cell counts as correct if the prediction is
        # either of its two labelled categories (strict = primary label only)
        true_len = np.where((true2 != "") & (pred == true2), true2, true)
        pl = ht(pred, true_len, w)
        r["lenient_mixed"] = {c: {m: float(pl[c][m]) for m in ("precision", "recall", "f1", "iou")} for c in EVAL_CATS}
        r["lenient_mixed"]["n_mixed_samples"] = int((true2 != "").sum())
        report["runs"][name] = r

        fmt = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if np.isfinite(v) else "n/a"  # noqa: E731
        md += [f"## run: {name}", "", f"`{path}`", "",
               "| category | P | R | F1 | IoU | est. TP / FP / FN cells |", "|---|---|---|---|---|---|"]
        for c, m in r["per_category"].items():
            md.append(f"| {c} | {fmt(m['precision'], m['precision_ci95'])} | {fmt(m['recall'], m['recall_ci95'])}"
                      f" | {fmt(m['f1'], m['f1_ci95'])} | {fmt(m['iou'], m['iou_ci95'])}"
                      f" | {m['TP']:.0f} / {m['FP']:.0f} / {m['FN']:.0f} |")
        md += ["", f"macro-F1 {fmt(r['macro']['f1'], r['macro']['f1_ci95'])}; "
                   f"macro-IoU {fmt(r['macro']['iou'], r['macro']['iou_ci95'])} over {EVAL_CATS}"]
        if "unknown_composition" in r:
            md += ["", f"final unknown cells (est. {r['unknown_cells_est']:.0f}) composition:"] + \
                  [f"- {t}: {100*v:.1f}%" for t, v in sorted(r["unknown_composition"].items(), key=lambda x: -x[1])]
        if "vehicle_filter" in r:
            v = r["vehicle_filter"]
            md += ["", f"vehicle filter: removed ≈{v['removed_cells_est']:.0f} cells; removal precision "
                       f"{fmt(v['removal_precision'], v['removal_precision_ci95'])}; infrastructure among removed "
                       f"{fmt(v['false_removal_infra_share'], v['false_removal_infra_share_ci95'])}"]
        md += [f"residual vehicle cells in final scene ≈{r['residual_vehicle_cells']:.0f} "
               f"[{r['residual_vehicle_cells_ci95'][0]:.0f}, {r['residual_vehicle_cells_ci95'][1]:.0f}]", ""]
    # Intra-rater agreement on the hidden re-test items (same rater, second pass).
    pairs = [(lab[str(k["retest_of"])]["cat"], lab[str(k["sample_id"])]["cat"]) for k in retest
             if (lab.get(str(k["retest_of"])) or {}).get("cat") and (lab.get(str(k["sample_id"])) or {}).get("cat")]
    if pairs:
        a = np.asarray([p[0] for p in pairs]); b = np.asarray([p[1] for p in pairs])
        po = float((a == b).mean())
        cats = sorted(set(a) | set(b))
        pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
        kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
        kb = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(a), len(a))
            aa, bb_ = a[i], b[i]
            po_ = (aa == bb_).mean()
            pe_ = sum((aa == c).mean() * (bb_ == c).mean() for c in cats)
            kb.append((po_ - pe_) / (1 - pe_) if pe_ < 1 else np.nan)
        kb = [v for v in kb if np.isfinite(v)]
        report["intra_rater"] = {"n_pairs": len(pairs), "agreement": po, "cohen_kappa": kappa,
                                 "kappa_ci95": [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))],
                                 "disagreements": dict(Counter(f"{x} / {y}" for x, y in pairs if x != y))}
        ir = report["intra_rater"]
        md += ["## Intra-rater agreement (hidden re-test, same annotator)", "",
               f"pairs {ir['n_pairs']}, raw agreement {ir['agreement']:.3f}, Cohen's kappa {ir['cohen_kappa']:.3f} "
               f"[{ir['kappa_ci95'][0]:.3f}, {ir['kappa_ci95'][1]:.3f}]",
               "disagreements: " + ", ".join(f"{k} ×{v}" for k, v in ir["disagreements"].items()), ""]
    md += ["## Strata", ""] + [f"- {h}: N={s['N_h']:,} used={s['n_used']} {s['true_counts']}"
                               for h, s in report["strata"].items()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "eval_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                                           encoding="utf-8")
    (out / "eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
