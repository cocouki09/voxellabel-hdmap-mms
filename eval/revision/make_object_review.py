"""Object-level review package (R1 object evaluation; R3-6).

Every output object (one per map feature) of the evaluated run is shown with
its cells highlighted over the raw LiDAR and the nearest survey photos. The
category name is hidden and the order is random. The annotator states what
the object mainly is (same categories as the cell sample) and how pure it is.

  python eval/revision/make_object_review.py --stages <A0 stages.npz> --raw <survey root> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys as _sys
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

_sys.stdout.reconfigure(encoding="utf-8")
FX, CX, CY, W, H = 1122.997311, 784.0, 882.0, 1568, 1764
CTX_M = 8.0          # context margin around the object's bounding box
MAX_CTX = 40000
SEED = 20261015


def load_eo(raw):
    names, C, R, dirs = [], [], [], []
    for csv in glob.glob(os.path.join(raw, "SURV*", "TRACK*", "Camera0[1-4]", "External Orientation.csv")):
        d = os.path.dirname(csv)
        for line in open(csv, encoding="utf-8"):
            f = line.strip().split(";")
            if len(f) < 17:
                continue
            names.append(f[0]); dirs.append(d)
            C.append([float(v) for v in f[2:5]])
            R.append(np.array([float(v) for v in f[8:17]]).reshape(3, 3))
    return names, np.asarray(C), np.asarray(R), dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--remap", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False); (out / "photos").mkdir()
    st = np.load(args.stages)
    objs = json.loads(Path(args.stages).with_suffix(".objects.json").read_text(encoding="utf-8"))["objects"]
    vs = float(st["voxel_size"]); org = st["origin"]
    xyz_c = org + (st["cells"] + 0.5) * vs
    oid = st["object_id"]
    ids = [o for o, v in objs.items() if v["category"] != "unknown"]
    rng = np.random.default_rng(SEED)
    ids = list(rng.permutation(ids))
    cells_of = {o: xyz_c[oid == o] for o in ids}
    centers = np.array([c.mean(axis=0) for c in cells_of.values()])
    lo = np.array([c.min(axis=0) for c in cells_of.values()]) - CTX_M
    hi = np.array([c.max(axis=0) for c in cells_of.values()]) + CTX_M
    ctx = [[] for _ in ids]
    las_paths = [str(x) for x in st["las_sources"]]
    if args.remap:
        a, b = args.remap.split("=", 1)
        las_paths = [p.replace(a, b) for p in las_paths]
    for p in las_paths:
        with laspy.open(p) as f:
            hd = f.header
            if hd.maxs[0] < lo[:, 0].min() or hd.mins[0] > hi[:, 0].max() or hd.maxs[1] < lo[:, 1].min() or hd.mins[1] > hi[:, 1].max():
                continue
            las = f.read()
        P = np.column_stack([las.x, las.y, las.z])
        try:
            C = np.column_stack([las.red, las.green, las.blue]).astype(float)
            C = (C / (256.0 if C.max() > 255 else 1.0)).astype(np.uint8)
        except Exception:  # noqa: BLE001
            C = np.full((len(P), 3), 160, np.uint8)
        for i in range(len(ids)):
            m = np.all((P >= lo[i]) & (P <= hi[i]), axis=1)
            if m.any():
                k = np.where(m)[0]
                if len(k) > MAX_CTX:
                    k = rng.choice(k, MAX_CTX, replace=False)
                ctx[i].append((P[k], C[k]))
        print(f"[LAS] {Path(p).name}")
    names, CE, RE, dirs = load_eo(args.raw)
    meta, blobs, off, key = [], [], 0, []
    for i, o in enumerate(ids):
        c0 = centers[i]
        if ctx[i]:
            Pc = np.vstack([a for a, _ in ctx[i]]); Cc = np.vstack([b for _, b in ctx[i]])
            if len(Pc) > MAX_CTX:
                s = rng.choice(len(Pc), MAX_CTX, replace=False); Pc, Cc = Pc[s], Cc[s]
        else:
            Pc = np.zeros((0, 3)); Cc = np.zeros((0, 3), np.uint8)
        rel = (Pc - c0).astype(np.float32)
        blob = rel.tobytes() + Cc.astype(np.uint8).tobytes()
        cells_rel = np.round(cells_of[o] - c0, 3).tolist()
        # photos: project the object centre
        photos = []
        d = np.linalg.norm(CE[:, :2] - c0[:2], axis=1)
        cand = []
        for j in np.where(d < 40)[0]:
            q = (c0 - CE[j]) @ RE[j]
            x, y, z = q[0], -q[1], -q[2]
            if 3 < z < 40:
                u, v = CX + FX * x / z, CY + FX * y / z
                if 80 <= u < W - 80 and 80 <= v < H - 80:
                    cand.append((z, j, u, v))
        cand.sort()
        seen = set()
        for z, j, u, v in cand:
            if len(photos) >= 2:
                break
            if dirs[j] in seen:
                continue
            seen.add(dirs[j])
            try:
                img = Image.open(os.path.join(dirs[j], names[j])).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            ext = np.ptp(cells_of[o], axis=0)
            half = int(min(700, max(220, FX * (max(ext[0], ext[1], ext[2]) + 3) / z / 2)))
            crop = img.crop((int(u) - half, int(v) - half, int(u) + half, int(v) + half))
            dr = ImageDraw.Draw(crop)
            r = max(8, FX * 0.6 / z)
            dr.ellipse((half - r, half - r, half + r, half + r), outline=(255, 40, 220), width=3)
            fn = f"{i}_{len(photos)}.jpg"
            crop.resize((420, 420)).save(out / "photos" / fn, quality=82)
            photos.append({"file": fn, "range_m": round(float(z), 1)})
        meta.append({"sample_id": i, "n": int(len(Pc)), "offset": off, "voxel_m": vs,
                     "obj_cells": cells_rel, "photos": photos, "rep": [0, 0, 0]})
        blobs.append(blob); off += len(blob)
        key.append({"sample_id": i, "object_id": o, "category": objs[o]["category"],
                    "shp_feature_id": objs[o].get("shp_feature_id"), "n_cells": objs[o]["n_cells"],
                    "center": c0.tolist()})
    (out / "points.bin").write_bytes(b"".join(blobs))
    cats = ["guardrail", "traffic_light", "sign", "street_light", "height_barrier", "tunnel_structure",
            "viaduct_structure", "vegetation", "vehicle", "ground_or_road_surface", "building_or_wall",
            "other_structure", "noise_or_empty", "cannot_judge"]
    (out / "samples.json").write_text(json.dumps({"mode": "object", "categories": cats, "samples": meta}), encoding="utf-8")
    (out / "KEY_do_not_open_while_labeling.json").write_text(json.dumps({"stages": args.stages, "seed": SEED, "key": key}, indent=1), encoding="utf-8")
    print(f"[DONE] {len(ids)} objects -> {out}")


if __name__ == "__main__":
    main()
