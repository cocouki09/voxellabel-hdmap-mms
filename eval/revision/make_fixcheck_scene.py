"""Viewer check scene for the post-hoc 'detach low fragments' fix (Text S3).

Copies the Deoksong viewer preview and moves the cells that the fix detaches
from traffic-light, sign and post objects into a separate category
'fix_detached' (magenta), so the fix can be inspected in the web viewer next to
the unchanged Deoksong scene. Baked geometry of the affected 25 m cells is
dropped so the viewer rebuilds them from voxel_objects. The original scene is
not modified.

  python eval/revision/make_fixcheck_scene.py --before A0/stages.npz --after P5/stages.npz \
      --preview data/output_deoksong/preview_json/scene_preview.json --out data/output_deoksong_fixcheck
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

PER_VOXEL = ("voxel_positions", "voxel_colors", "voxel_colors_raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--preview", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = np.load(a.before); f = np.load(a.after)
    ob, oa = b["object_id"].astype(str), f["object_id"].astype(str)
    ch = np.where(ob != oa)[0]
    moved = defaultdict(set)
    for i in ch:
        moved[ob[i]].add(tuple(int(v) for v in b["cells"][i]))
    src = Path(a.preview)
    d = json.load(open(src, encoding="utf-8"))
    mcell = int(d["grid_config"]["macro_grid_size"] / d["grid_config"]["voxel_size"])
    touched = set(); new_objs = []; n_moved = 0
    for obj in d["voxel_objects"]:
        cells = moved.get(obj["object_id"])
        if not cells:
            continue
        keep_idx, out_idx = [], []
        for k, v in enumerate(obj["voxels"]):
            (out_idx if tuple(v[:3]) in cells else keep_idx).append(k)
        if not out_idx:
            continue
        n_moved += len(out_idx)
        det = {"object_id": "fix_detached__" + obj["object_id"], "category": "fix_detached", "color": "#ff00ff",
               "voxels": [obj["voxels"][k] for k in out_idx], "attributes": dict(obj["attributes"])}
        for key in PER_VOXEL:
            arr = obj["attributes"].get(key)
            if isinstance(arr, list) and len(arr) == len(obj["voxels"]):
                det["attributes"][key] = [arr[k] for k in out_idx]
                obj["attributes"][key] = [arr[k] for k in keep_idx]
        det["attributes"]["cell_count"] = len(out_idx)
        det["attributes"]["detached_from"] = obj["object_id"]
        det["attributes"].pop("composite_id", None)
        obj["voxels"] = [obj["voxels"][k] for k in keep_idx]
        obj["attributes"]["cell_count"] = len(keep_idx)
        for v in det["voxels"]:
            touched.add(f"{v[0] // mcell},{v[1] // mcell}")
        new_objs.append(det)
    d["voxel_objects"].extend(new_objs)
    baked = d.get("baked_geometry_by_cell") or {}
    for k in touched:
        baked.pop(k, None)
    out = Path(a.out)
    (out / "preview_json").mkdir(parents=True, exist_ok=True)
    (out / "scene_all").mkdir(parents=True, exist_ok=True)
    json.dump(d, open(out / "preview_json" / "scene_preview.json", "w", encoding="utf-8"))
    shutil.copy2(src.parents[1] / "scene_all" / "scene_all.json", out / "scene_all" / "scene_all.json")
    print(f"moved {n_moved} voxels from {len(new_objs)} objects; dropped baked cells {sorted(touched)}")


if __name__ == "__main__":
    main()
