"""IJGI revision W03 — stratified random cell sample for blind manual labeling.

Design (fixed before any labeling):
  population  every occupied 0.5 m cell of one pipeline run (--dump-stages
              npz), INCLUDING cells the filters dropped, so real
              infrastructure removed by a filter still counts as a miss
  strata      the cell's final pipeline outcome:
                kept:<category>  for each kept output category
                drop:vehicle     removed by the dynamic-vehicle filter
                drop:low_unknown unknown below 0.3 m elevation
                drop:out_of_shp  unknown outside the 12 m retention buffer
                drop:ground      ground category (represented by overlay)
  allocation  n per stratum = min(N_h, --n-per-stratum), with --n-unknown
              for the large kept:unknown stratum (recall needs it)
  selection   simple random sample without replacement within each
              stratum, numpy default_rng(--seed)
  weights     w = N_h / n_h  (design weights for P/R estimation)

The annotation package contains only the sample id and the surrounding raw
LiDAR points — never the pipeline label — so labeling is blind. The key
(stratum, weight, cell) is written to a separate file.

Run (after a --dump-stages build):
  python eval/revision/sample_cells.py --stages <run>/stages.npz --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys as _sys
_sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

CROP_XY_M = 20.0      # half-width of the context crop around the cell
CROP_Z_M = 15.0       # +- vertical window around the cell
LOCAL_M = 1.5         # dense neighbourhood around the cell
MID_M = 6.0           # medium-density ring
MAX_LOCAL = 6000      # per-sample caps (random thinning, display only)
MAX_MID = 12000
MAX_CTX = 14000
MAX_PTS = MAX_LOCAL + MAX_MID + MAX_CTX
CTX_KEEP = 0.04       # fraction of points kept for the 6–20 m context
STRATA_ORDER = ["kept:guardrail", "kept:traffic_light", "kept:sign", "kept:street_light",
                "kept:height_barrier", "kept:unknown:near_infra", "kept:unknown:far",
                "drop:vehicle", "drop:low_unknown:near_infra", "drop:low_unknown:far",
                "drop:out_of_shp", "drop:ground"]
# Allocation fixed before sampling (cells per stratum, capped at N_h).
ALLOC = {"kept:guardrail": 80, "kept:traffic_light": 80, "kept:sign": 80,
         "kept:street_light": 80, "kept:height_barrier": 60,
         "kept:unknown:near_infra": 300, "kept:unknown:far": 150,
         "drop:vehicle": 80, "drop:low_unknown:near_infra": 60, "drop:low_unknown:far": 20,
         "drop:out_of_shp": 40, "drop:ground": 60}


# "near" = within ~1.5x the matching radius of a mapped infrastructure feature
# (XY). Missed infrastructure concentrates there, so unknown cells are split
# into near/far sub-strata to make recall estimable for rare categories.
NEAR_M = {"traffic_light": 6.0, "sign": 3.0, "street_light": 4.0,
          "guardrail": 1.0, "height_barrier": 4.5}


def near_infra(st):
    m = np.zeros(len(st["cells"]), dtype=bool)
    for cat, r in NEAR_M.items():
        k = f"dist_{cat}"
        if k in st:
            m |= st[k] <= r
    return m


def stratum_of(st):
    lab = st["labels_final"]
    keep = st["keep"]
    near = near_infra(st)
    s = np.full(len(lab), "", dtype=object)
    s[keep] = np.char.add("kept:", lab[keep].astype(str))
    unk = keep & (lab == "unknown")
    s[unk & near] = "kept:unknown:near_infra"
    s[unk & ~near] = "kept:unknown:far"
    s[st["drop_dynamic"]] = "drop:vehicle"
    low = st["drop_off_road_low"]
    s[low & near] = "drop:low_unknown:near_infra"
    s[low & ~near] = "drop:low_unknown:far"
    s[st["drop_out_of_shp"] & ~low] = "drop:out_of_shp"
    s[st["drop_ground"]] = "drop:ground"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=60,
                    help="allocation for any stratum not listed in ALLOC")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--remap", default=None, help="OLD_PREFIX=NEW_PREFIX for LAS paths")
    ap.add_argument("--n-retest", type=int, default=100,
                    help="main samples repeated at the end for intra-rater agreement")
    ap.add_argument("--las-stride", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)

    st = dict(np.load(args.stages, allow_pickle=False))
    strata = stratum_of(st)
    assert (strata != "").all(), "every cell must fall in exactly one stratum"
    rng = np.random.default_rng(args.seed)
    vs = float(st["voxel_size"]); origin = st["origin"]
    cells = st["cells"]

    picks, key = [], []
    names = [s for s in STRATA_ORDER if (strata == s).any()] + \
            sorted(set(strata) - set(STRATA_ORDER))
    for name in names:
        idx = np.where(strata == name)[0]
        n = min(ALLOC.get(name, args.n_per_stratum), len(idx))
        chosen = rng.choice(idx, size=n, replace=False)
        for i in chosen:
            picks.append(int(i))
            key.append({"stratum": name, "N_h": int(len(idx)), "n_h": int(n),
                        "weight": len(idx) / n, "row": int(i),
                        "cell": cells[i].tolist(),
                        "object_id": str(st["object_id"][i]),
                        "label_labeling": str(st["labels_labeling"][i])})
        print(f"[STRATUM] {name:<20} N={len(idx):>8,}  n={n}")
    order = rng.permutation(len(picks))  # shuffle so strata are not grouped
    picks = [picks[o] for o in order]
    key = [key[o] for o in order]
    for sid, k in enumerate(key):
        k["sample_id"] = sid

    centers = np.column_stack([origin[0] + (cells[picks, 0] + 0.5) * vs,
                               origin[1] + (cells[picks, 1] + 0.5) * vs,
                               origin[2] + (cells[picks, 2] + 0.5) * vs])
    buckets = [[] for _ in picks]
    thin_rng = np.random.default_rng(args.seed + 1)  # display thinning only; selection uses rng
    las_paths = [str(x) for x in st["las_sources"]]
    if args.remap:  # read the same files from another copy (e.g. local disk)
        old, new = args.remap.split("=", 1)
        las_paths = [q.replace(old, new) if q.startswith(old) else q for q in las_paths]
    for p in las_paths:
        with laspy.open(p) as f:
            las = f.read()
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])[::args.las_stride]
        try:
            rgb = np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)])[::args.las_stride]
            rgb = (rgb / (256.0 if rgb.max() > 255 else 1.0)).astype(np.uint8)
        except Exception:  # noqa: BLE001
            rgb = np.full((len(xyz), 3), 160, np.uint8)
        # Square XY crop (Chebyshev radius) around every sample centre.
        lo = centers[:, :2].min(axis=0) - CROP_XY_M
        hi = centers[:, :2].max(axis=0) + CROP_XY_M
        inbox = np.all((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi), axis=1)
        xyz, rgb = xyz[inbox], rgb[inbox]
        if len(xyz):
            # full density only near the cell; the wide context comes from a
            # pre-thinned copy so 20 m neighbourhoods stay small
            near_tree = cKDTree(xyz[:, :2])
            sub = thin_rng.random(len(xyz)) < CTX_KEEP
            ctx_idx_all = np.where(sub)[0]
            ctx_tree = cKDTree(xyz[ctx_idx_all, :2]) if len(ctx_idx_all) else None
            for h in range(len(centers)):
                c = centers[h]
                idx = np.asarray(near_tree.query_ball_point(c[:2], r=MID_M, p=np.inf), dtype=np.int64)
                cidx = (ctx_idx_all[np.asarray(ctx_tree.query_ball_point(c[:2], r=CROP_XY_M, p=np.inf),
                                               dtype=np.int64)] if ctx_tree is not None else idx[:0])
                if len(idx):
                    idx = idx[np.abs(xyz[idx, 2] - c[2]) <= CROP_Z_M]
                if len(cidx):
                    cidx = cidx[(np.abs(xyz[cidx, 2] - c[2]) <= CROP_Z_M)
                                & (np.max(np.abs(xyz[cidx, :2] - c[:2]), axis=1) > MID_M)]
                if not len(idx) and not len(cidx):
                    continue
                cheb = np.max(np.abs(xyz[idx] - c), axis=1) if len(idx) else np.zeros(0)
                tiers = []
                for t, cap in ((idx[cheb <= LOCAL_M], MAX_LOCAL),
                               (idx[cheb > LOCAL_M], MAX_MID),
                               (cidx, MAX_CTX)):
                    if len(t) > cap:
                        t = thin_rng.choice(t, cap, replace=False)
                    tiers.append(t)
                # merge into the running per-tier buffers and re-thin, so memory
                # stays bounded by the caps regardless of how many files overlap
                cur = buckets[h] or [(np.zeros((0, 3)), np.zeros((0, 3), np.uint8))] * 3
                merged = []
                for ti, (t, cap) in enumerate(zip(tiers, (MAX_LOCAL, MAX_MID, MAX_CTX))):
                    P_ = np.vstack([cur[ti][0], xyz[t]]); C_ = np.vstack([cur[ti][1], rgb[t]])
                    if len(P_) > cap:
                        s = thin_rng.choice(len(P_), cap, replace=False); P_, C_ = P_[s], C_[s]
                    merged.append((P_, C_))
                buckets[h] = merged
        print(f"[LAS] {Path(p).name}: {int(inbox.sum()):,} pts near samples")

    n_main = len(picks)
    meta, blobs, off = [], [], 0
    caps = (MAX_LOCAL, MAX_MID, MAX_CTX)
    for sid in range(n_main):
        b = buckets[sid]
        P = np.vstack([t[0] for t in b]) if b else np.zeros((0, 3))
        C = np.vstack([t[1] for t in b]) if b else np.zeros((0, 3), np.uint8)
        rel = (P - centers[sid]).astype(np.float32)
        blob = rel.tobytes() + C.astype(np.uint8).tobytes()
        meta.append({"sample_id": sid, "n": int(len(P)), "offset": off, "voxel_m": vs})
        blobs.append(blob); off += len(blob)
    # Intra-rater re-test: a random subset of main samples is shown again at the
    # end, in shuffled order, under new ids (same points, no hint). Used only
    # for intra-rater agreement (Cohen's kappa), never in the accuracy estimate.
    retest_src = rng.choice(n_main, size=min(args.n_retest, n_main), replace=False)
    for j, src in enumerate(retest_src):
        sid = n_main + j
        meta.append({**meta[src], "sample_id": sid})
        key.append({"sample_id": sid, "retest_of": int(src), "stratum": "_retest"})
    (out / "points.bin").write_bytes(b"".join(blobs))
    (out / "samples.json").write_text(json.dumps({
        "categories": ["guardrail", "traffic_light", "sign", "street_light", "height_barrier",
                       "tunnel_structure", "viaduct_structure", "vegetation", "vehicle",
                       "ground_or_road_surface", "building_or_wall", "other_structure",
                       "noise_or_empty", "cannot_judge"],
        "samples": meta}, indent=1), encoding="utf-8")
    (out / "KEY_do_not_open_while_labeling.json").write_text(json.dumps({
        "stages": str(Path(args.stages).resolve()), "seed": args.seed,
        "args": vars(args), "crop": {"xy_m": CROP_XY_M, "z_m": CROP_Z_M, "max_pts": MAX_PTS},
        "centers": centers.tolist(), "key": key}, indent=1), encoding="utf-8")
    print(f"[DONE] {n_main} samples + {len(meta) - n_main} re-test → {out}")


if __name__ == "__main__":
    main()
