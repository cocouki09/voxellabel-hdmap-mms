"""Collect the final evaluation outputs into manuscript-ready tables (markdown).

  python eval/revision/build_final_tables.py --final-dir <W04_FINAL_...> --out <TABLES.md>
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path

_sys.stdout.reconfigure(encoding="utf-8")
NAMES = {"guardrail": "Guardrail", "traffic_light": "Traffic light", "sign": "Sign",
         "street_light": "Post / street light", "height_barrier": "Overpass (height barrier)", "tunnel": "Tunnel"}
VARIANTS = [("submitted", "Evaluated configuration (sign 2.0 m)"), ("A1_ground_constant", "Constant ground model"),
            ("A6_ground_idw", "IDW ground model"), ("A2_gate_off", "Elevation gate off"),
            ("A3_uniform_radius2", "One radius (2.0 m) for all categories"), ("A4_vehicle_filter_off", "Vehicle filter off"),
            ("A5_pole_protection_off", "Pole protection off"), ("S1_sign1m_deployed", "Sign radius 1.0 m"),
            ("S2_radius_x0.75", "All radii ×0.75"), ("S3_radius_x1.25", "All radii ×1.25"),
            ("S4_window_narrow0.25", "Windows narrowed 0.25 m"), ("S5_window_wide0.25", "Windows widened 0.25 m"),
            ("P1_overhead_sign_window_posthoc", "Post hoc: overhead-sign window"),
            ("P3_maptype_C1_9_posthoc", "Post hoc: exclude tunnel signals (C1 type 9, 4 objects)"),
            ("P2_maptype_filter_posthoc", "Post hoc: exclude map types C3-8, C1-9, C5-1 (10 objects)")]


def f(x, nd=2):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def ci(v, k, nd=2):
    c = v.get(k + "_ci95")
    return f"{f(v.get(k), nd)} [{f(c[0], nd)}, {f(c[1], nd)}]" if c and v.get(k) == v.get(k) and v.get(k) is not None else f(v.get(k), nd)


def table6(run, cats, pred_cells=None):
    rows = ["| Category | Precision | Recall | F1 | IoU |", "|---|---|---|---|---|"]
    for c in cats:
        v = run["per_category"][c]
        rows.append(f"| {NAMES[c]} | {ci(v, 'precision')} | {ci(v, 'recall')} | {ci(v, 'f1')} | {ci(v, 'iou')} |")
    m = run["macro"]
    rows.append(f"| Macro average | | | {ci(m, 'f1')} | {ci(m, 'iou')} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = Path(a.final_dir)
    J = lambda p: json.load(open(F / p, encoding="utf-8"))
    dl = J("cell_delivered/eval_summary.json"); sub = dl["runs"]["submitted"]
    pre = J("cell_delivered_before_tunnel_recheck/eval_summary.json")["runs"]["submitted"]
    strict = J("cell_strict_reference/eval_summary.json")["runs"]["submitted"]
    cats_d = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier", "tunnel"]
    cats_s = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier"]
    out = [f"# Final tables ({F.name})", "",
           f"Sample: {dl['n_samples']} drawn, {dl['n_used']} used, {dl['n_unlabeled']} not labeled (beyond quota), "
           f"{sum(dl['ignored_by_stratum'].values())} excluded (cannot judge / isolated returns).", "",
           "## Table 6 — cell level, delivered definitions (final labels incl. corrections)", ""]
    out += table6(sub, cats_d)
    out += ["", "## Table 6 (reference) — cell level, strict definitions", ""] + table6(strict, cats_s)
    one = J("cell_delivered_onesided_tunnel_recheck/eval_summary.json")["runs"]["submitted"]
    out += ["", "## Effect of the tunnel–guardrail re-check (delivered)", "",
            "| Category | P / R / F1 before | after 40 predicted-guardrail cells | after both directions (40 + 27) |", "|---|---|---|---|"]
    for c in cats_d:
        b, o, x = pre["per_category"][c], one["per_category"][c], sub["per_category"][c]
        out.append(f"| {NAMES[c]} | " + " | ".join(f"{f(v['precision'])} / {f(v['recall'])} / {f(v['f1'])}" for v in (b, o, x)) + " |")
    out.append(f"| Macro-F1 | {ci(pre['macro'], 'f1')} | {ci(one['macro'], 'f1')} | {ci(sub['macro'], 'f1')} |")
    out += ["", "## Sensitivities (delivered)", "", "| Scoring | Macro-F1 | Macro-IoU |", "|---|---|---|",
            f"| Primary label | {ci(sub['macro'], 'f1')} | {ci(sub['macro'], 'iou')} |"]
    lm = sub.get("lenient_mixed") or {}
    if lm:
        import numpy as np
        f1s = [v["f1"] for v in lm.values() if isinstance(v, dict) and v.get("f1") == v.get("f1") and v.get("f1") is not None]
        out.append(f"| Lenient (either label) | {np.mean(f1s):.2f} | |")
    for s, nm in (("exclude-ambiguous", "Ambiguous cells excluded"), ("promote-secondary", "Second label when primary is noise")):
        m = J(f"cell_delivered_{s}/eval_summary.json")["runs"]["submitted"]["macro"]
        out.append(f"| {nm} | {ci(m, 'f1')} | {ci(m, 'iou')} |")
    out += ["", "## Table 7 — ablation and sensitivity (delivered; P / R)", "",
            "| Variant | Guardrail | Traffic light | Sign | Post / street light | Tunnel | Macro-F1 |", "|---|---|---|---|---|---|---|"]
    for k, nm in VARIANTS:
        r = dl["runs"][k]; pc = r["per_category"]
        cells = " | ".join(f"{f(pc[c]['precision'])} / {f(pc[c]['recall'])}" for c in ["guardrail", "traffic_light", "sign", "street_light", "tunnel"])
        out.append(f"| {nm} | {cells} | {ci(r['macro'], 'f1')} |")
    vf = sub["vehicle_filter"]
    out += ["", "## Vehicle filter and unknown composition (delivered, final labels)", "",
            f"- removed ≈{vf['removed_cells_est']:.0f} cells; removal precision {ci(vf, 'removal_precision')}; "
            f"structure among removed {ci(vf, 'false_removal_infra_share')}",
            f"- residual vehicle cells ≈{sub['residual_vehicle_cells']:.0f} {[round(x) for x in sub['residual_vehicle_cells_ci95']]}",
            f"- unknown cells (HT estimate) ≈{sub['unknown_cells_est']:.0f}; composition: " +
            ", ".join(f"{('viaduct' if k == 'height_barrier' else k)} {v * 100:.1f}%" for k, v in sorted(sub["unknown_composition"].items(), key=lambda x: -x[1]))]
    ir = J("../W04_FINAL_strict/eval_summary.json").get("intra_rater") if (F / "../W04_FINAL_strict/eval_summary.json").exists() else None
    if ir:
        out += ["", f"Intra-rater (blind first-pass labels, before any correction): {ir['n_pairs']} pairs, agreement {ir['agreement']:.3f}, "
                    f"κ {ir['cohen_kappa']:.3f} [{ir['kappa_ci95'][0]:.3f}, {ir['kappa_ci95'][1]:.3f}]"]
    out += ["", (F / "occupancy.md").read_text(encoding="utf-8").replace("# ", "## ", 1), "",
            (F / "objects_delivered.md").read_text(encoding="utf-8").replace("# ", "## ", 1)]
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
