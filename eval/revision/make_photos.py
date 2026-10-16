"""Attach MMS camera photos to each blind sample (labeling aid only).

For every sample cell, find the survey photos in which the cell centre
projects inside the frame, 2–35 m in front of the camera, and save up to
--per-sample crops around the projected point with a marker. Occlusion is not
checked, so a photo may show whatever stands in front of the cell; the
labeler states this. Uses only the exterior orientation delivered with the
survey (Camera0k/External Orientation.csv: name; t; X; Y; Z; ω; φ; κ; r11..r33)
and the internal orientation (pinhole, no distortion in this delivery).

Projection (found by maximising LiDAR-colour vs photo consistency, r = 0.65):
  q = (X − C) · R,  x = q0, y = −q1, z = −q2,  u = cx + f·x/z,  v = cy + f·y/z

  python eval/revision/make_photos.py --sample-dir <dir> --raw <survey root>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys as _sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_sys.stdout.reconfigure(encoding="utf-8")
FX, CX, CY, W, H = 1122.997311, 784.0, 882.0, 1568, 1764
CROP = 640
OUT_PX = 420


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
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--per-sample", type=int, default=2)
    args = ap.parse_args()
    sd = Path(args.sample_dir)
    key = json.load(open(sd / "KEY_do_not_open_while_labeling.json", encoding="utf-8"))
    centers = np.asarray(key["centers"])
    samples = json.load(open(sd / "samples.json", encoding="utf-8"))
    names, C, R, dirs = load_eo(args.raw)
    print(f"[EO] {len(names)} photos")
    (sd / "photos").mkdir(exist_ok=True)
    n_main = len(centers)
    photos_of = {}
    for i in range(n_main):
        p = centers[i]
        d = np.linalg.norm(C[:, :2] - p[:2], axis=1)
        cand = np.where(d < 35)[0]
        scored = []
        for j in cand:
            q = (p - C[j]) @ R[j]
            x, y, z = q[0], -q[1], -q[2]
            if z < 2 or z > 35:
                continue
            u, v = CX + FX * x / z, CY + FX * y / z
            if not (60 <= u < W - 60 and 60 <= v < H - 60):
                continue
            off = np.hypot((u - CX) / W, (v - CY) / H)  # prefer near the image centre
            scored.append((z * (1 + 2 * off), j, u, v, z))
        scored.sort()
        out = []
        used_dirs = set()
        for _, j, u, v, z in scored:
            if len(out) >= args.per_sample:
                break
            if (dirs[j], round(C[j][0]), round(C[j][1])) in used_dirs:
                continue
            used_dirs.add((dirs[j], round(C[j][0]), round(C[j][1])))
            try:
                img = Image.open(os.path.join(dirs[j], names[j])).convert("RGB")
            except Exception:  # noqa: BLE001  unreadable/missing frame
                continue
            half = CROP // 2
            box = (int(u) - half, int(v) - half, int(u) + half, int(v) + half)
            crop = img.crop(box)
            dr = ImageDraw.Draw(crop)
            r_px = max(6, FX * 0.25 / z)  # half a 0.5 m cell at that range
            dr.rectangle((half - r_px, half - r_px, half + r_px, half + r_px), outline=(255, 204, 51), width=3)
            dr.ellipse((half - 4 * r_px, half - 4 * r_px, half + 4 * r_px, half + 4 * r_px), outline=(255, 204, 51), width=1)
            crop = crop.resize((OUT_PX, OUT_PX))
            fn = f"{i}_{len(out)}.jpg"
            crop.save(sd / "photos" / fn, quality=82)
            out.append({"file": fn, "range_m": round(float(z), 1)})
        photos_of[i] = out
        if i % 100 == 0:
            print(f"[PHOTO] {i}/{n_main}")
    for s in samples["samples"]:
        k = next(x for x in key["key"] if x["sample_id"] == s["sample_id"])
        src = k.get("retest_of", s["sample_id"])
        s["photos"] = photos_of.get(src, [])
    (sd / "samples.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
    have = sum(1 for v in photos_of.values() if v)
    print(f"[DONE] photos for {have}/{n_main} samples")


if __name__ == "__main__":
    main()
