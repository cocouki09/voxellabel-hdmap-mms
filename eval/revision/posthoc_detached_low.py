"""Post-hoc variant: remove detached low fragments from pole-mounted objects.

The object review found vehicle cells merged into traffic-light objects: the
signal head sits 5–7 m above the road and a passing vehicle 0.5–2 m, both
inside the 4 m matching radius, so they become one object although they do not
touch. For every object of a pole category (traffic light, sign, post), its
cells are split into 26-connected components; every component other than the
main one that lies entirely below a height limit and inside the road
corridor is detached from the object and kept as unknown occupancy (default),
or deleted like the vehicle filter (--mode remove). The main component (for signals and posts the one holding the highest
cell, for signs the largest) is never removed.

  python eval/revision/posthoc_detached_low.py --run A0/stages.npz --out P5/stages.npz [--max-elev 3.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

POLE = ("traffic_light", "sign", "street_light")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-elev", type=float, default=3.0)
    ap.add_argument("--mode", choices=["remove", "unknown"], default="unknown",
                    help="remove: delete the fragments like the vehicle filter; unknown: detach them from the object and "
                         "keep them as unknown occupancy (default; a detached low fragment can also be real structure)")
    a = ap.parse_args()
    st = dict(np.load(a.run))
    oid = st["object_id"].astype(str)
    lab = st["labels_final"].astype(str)
    keep = st["keep"]; el = st["elevation"]; road = st["in_road"]
    drop = np.zeros(len(oid), bool)
    report = []
    for o in np.unique(oid[keep]):
        if not o or o.split("__")[0] not in POLE:
            continue
        m = np.where((oid == o) & keep)[0]
        if len(m) < 2:
            continue
        c = st["cells"][m]; c0 = c - c.min(0)
        g = np.zeros(c0.max(0) + 1, bool); g[tuple(c0.T)] = True
        comp, n = ndimage.label(g, structure=np.ones((3, 3, 3)))
        if n < 2:
            continue
        L = comp[tuple(c0.T)]
        if o.split("__")[0] == "sign":  # roadside signs: keep the largest component
            main_k = int(np.argmax(np.bincount(L)[1:])) + 1
        else:  # signals and posts are anchored at the top: keep the component holding the highest cell
            main_k = int(L[np.argmax(el[m])])
        for k in range(1, n + 1):
            if k == main_k:
                continue
            sel = m[L == k]
            if el[sel].max() < a.max_elev and road[sel].all():
                drop[sel] = True
                report.append({"object": o, "cells": int(len(sel)), "elev": [float(el[sel].min()), float(el[sel].max())]})
    if a.mode == "remove":
        st["keep"] = keep & ~drop
        st["drop_dynamic"] = st["drop_dynamic"] | drop
    lab = lab.astype(object); lab[drop] = "unknown"; st["labels_final"] = lab.astype(str)
    oid = oid.astype(object); oid[drop] = ""; st["object_id"] = oid.astype(str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **st)
    Path(a.out).with_suffix(".removed.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"removed {int(drop.sum())} cells in {len({r['object'] for r in report})} objects")
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
