"""Per-SHP-feature clustering of labeled LIDAR cells + composite detection.

Motivation:
  The semantic labeler assigns every LIDAR cell a category string, but it
  doesn't tell us WHICH physical object that cell belongs to. With
  "traffic_light" cells merged into one big VoxelObject, clicking any
  single pole in the viewer selects all 20 traffic lights at once.

  This module splits labeled cells into one VoxelObject per SHP feature
  (i.e., one traffic light pole → one object) and then detects when
  multiple objects share a physical pole (e.g., a traffic light and a
  street light on the same post) and groups them into a composite.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point


def cluster_labeled_cells(
    cells: np.ndarray,
    labels: np.ndarray,
    rep_xyz: np.ndarray,
    rep_rgb: np.ndarray,
    cell_centers_xy: np.ndarray,
    shp_features: list[dict],
    *,
    keep_merged_categories: set[str],
    rep_rgb_raw: np.ndarray | None = None,
) -> dict[str, dict]:
    """Groups (cells, rep_xyz, rep_rgb) by nearest SHP feature of the same
    category. Returns a dict mapping object_id → {category, cells,
    rep_xyz, rep_rgb, shp_feature_id}.

    `rep_rgb_raw` (optional) is the same-indexed pre-ortho/pre-enhance
    snapshot of LIDAR-baked RGB. When supplied, each group's entry also
    carries `rep_rgb_raw` so downstream emitters can ship both the
    processed and the raw colour side-by-side (lets the viewer toggle
    between them without a rebuild).

    Categories in `keep_merged_categories` are emitted as a single merged
    object (used for 'unknown' — splitting it into thousands of tiny
    objects would blow up picking costs).

    `shp_features` is the same list fed to the semantic matcher; each
    entry should carry at minimum {category, geometry}. If an entry has
    'shp_feature_id' it's preserved in the output.
    """
    # Group SHP by category for fast lookup
    shp_by_cat: dict[str, list[dict]] = {}
    for f in shp_features:
        shp_by_cat.setdefault(f["category"], []).append(f)

    out: dict[str, dict] = {}

    for cat in np.unique(labels):
        cat_s = str(cat)
        mask = labels == cat
        if not mask.any():
            continue

        cat_cells = cells[mask]
        cat_xyz = rep_xyz[mask]
        cat_rgb = rep_rgb[mask]
        cat_rgb_raw = rep_rgb_raw[mask] if rep_rgb_raw is not None else None
        cat_centers = cell_centers_xy[mask]

        if cat_s in keep_merged_categories:
            out[f"lidar_{cat_s}"] = {
                "category": cat_s,
                "cells": cat_cells,
                "rep_xyz": cat_xyz,
                "rep_rgb": cat_rgb,
                "rep_rgb_raw": cat_rgb_raw,
                "shp_feature_id": None,
            }
            continue

        shp_feats = shp_by_cat.get(cat_s, [])
        if not shp_feats:
            # No SHP reference for this category — merge
            out[f"lidar_{cat_s}"] = {
                "category": cat_s,
                "cells": cat_cells,
                "rep_xyz": cat_xyz,
                "rep_rgb": cat_rgb,
                "rep_rgb_raw": cat_rgb_raw,
                "shp_feature_id": None,
            }
            continue

        # Geometry type drives clustering strategy
        first_geom = shp_feats[0]["geometry"]
        gt = first_geom.geom_type

        if gt in ("Point", "MultiPoint"):
            # KDTree over SHP Point XY, then assign each cell to nearest.
            shp_xy = np.array(
                [[f["geometry"].centroid.x, f["geometry"].centroid.y] for f in shp_feats],
                dtype=np.float64,
            )
            tree = cKDTree(shp_xy)
            _, nearest_idx = tree.query(cat_centers, k=1)
            _emit_groups(
                cat_s, shp_feats, nearest_idx,
                cat_cells, cat_xyz, cat_rgb, cat_rgb_raw, out,
            )
        else:
            # LineString / Polygon — use shapely .distance() per cell
            # (small feature count per category in practice).
            assignments = _assign_by_geom_distance(cat_centers, shp_feats)
            _emit_groups(
                cat_s, shp_feats, assignments,
                cat_cells, cat_xyz, cat_rgb, cat_rgb_raw, out,
            )

    return out


def _assign_by_geom_distance(
    cell_centers_xy: np.ndarray, shp_feats: list[dict]
) -> np.ndarray:
    """For each cell center, picks the index of the nearest SHP feature
    by shapely geom.distance(point)."""
    n = len(cell_centers_xy)
    assignments = np.zeros(n, dtype=np.int32)
    for i in range(n):
        pt = Point(cell_centers_xy[i, 0], cell_centers_xy[i, 1])
        best_i = 0
        best_d = float("inf")
        for j, f in enumerate(shp_feats):
            d = f["geometry"].distance(pt)
            if d < best_d:
                best_d = d
                best_i = j
        assignments[i] = best_i
    return assignments


def _emit_groups(
    cat_s: str,
    shp_feats: list[dict],
    assignments: np.ndarray,
    cat_cells: np.ndarray,
    cat_xyz: np.ndarray,
    cat_rgb: np.ndarray,
    cat_rgb_raw: np.ndarray | None,
    out: dict[str, dict],
) -> None:
    for shp_i in np.unique(assignments):
        hit = assignments == shp_i
        f = shp_feats[int(shp_i)]
        feat_id = f.get("shp_feature_id") or f"{cat_s}_{int(shp_i)}"
        obj_id = f"{cat_s}__{feat_id}"
        # Anchor XY = SHP feature centroid. This beats the cell
        # centroid for composite detection because a signal with a 3m
        # horizontal arm would otherwise drag its "location" 1.5m off
        # the pole, making sibling objects on the same pole fall
        # outside the merge radius.
        geom = f["geometry"]
        if hasattr(geom, "x") and hasattr(geom, "y"):
            anchor_xy = (float(geom.x), float(geom.y))
        else:
            c = geom.centroid
            anchor_xy = (float(c.x), float(c.y))
        out[obj_id] = {
            "category": cat_s,
            "cells": cat_cells[hit],
            "rep_xyz": cat_xyz[hit],
            "rep_rgb": cat_rgb[hit],
            "rep_rgb_raw": cat_rgb_raw[hit] if cat_rgb_raw is not None else None,
            "shp_feature_id": feat_id,
            "anchor_xy": anchor_xy,
        }


def _union_find_groups(
    cell_xyz: np.ndarray,
    cell_obj_ids: list[str] | np.ndarray,
    radius: float,
) -> list[list[str]]:
    """Cell-level Union-Find. Returns groups of object IDs whose cells
    transitively touch within `radius` (Euclidean, 3D)."""
    if len(cell_xyz) == 0:
        return []
    unique_ids = list(dict.fromkeys(
        cell_obj_ids.tolist() if isinstance(cell_obj_ids, np.ndarray)
        else cell_obj_ids
    ))
    parent: dict[str, str] = {k: k for k in unique_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(cell_xyz) > 1:
        tree = cKDTree(cell_xyz)
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        ids = (cell_obj_ids if isinstance(cell_obj_ids, np.ndarray)
               else np.asarray(cell_obj_ids, dtype=object))
        for i, j in pairs:
            oi, oj = ids[i], ids[j]
            if oi != oj:
                union(oi, oj)

    buckets: dict[str, list[str]] = {}
    for obj_id in unique_ids:
        buckets.setdefault(find(obj_id), []).append(obj_id)
    return list(buckets.values())


def detect_composites(
    objects: dict[str, dict],
    *,
    compositable_categories: Iterable[str],
    max_cell_distance_m: float = 1.0,
    max_xy_extent_m: float | None = None,
    split_cell_distance_m: float = 0.3,
    debug_out: dict | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Groups objects that share a physical structure by checking whether
    their LIDAR cells come within max_cell_distance_m of each other in 3D.

    Two-stage union-find (when max_xy_extent_m is set):
      1. Outer pass at max_cell_distance_m groups loosely-connected
         objects. Sparse LIDAR stacks on the same pole still get
         picked up here.
      2. If an outer group's (x,y) footprint diameter exceeds
         max_xy_extent_m, it's almost certainly a head-to-head bridge
         across separate intersection poles (SHP traffic_light is
         anchored at the signal HEAD, hanging over the road from one
         corner; opposite corners' heads can come within 1m+ of each
         other and let outer union-find transitively merge four poles
         into one group). We re-cluster JUST that group's cells at the
         tighter split_cell_distance_m (~voxel-touching only). Each
         sub-cluster is a candidate composite; the ones whose own
         footprint stays under max_xy_extent_m are kept, the rest are
         discarded (members fall back to standalone).

    The two-stage split is what salvages real same-pole structure
    inside a bundled super-group — single-stage hard-reject lost the
    valid sub-poles along with the bridge.
    """
    compositable = set(compositable_categories)

    # Flatten all cells from compositable objects with a parallel
    # object_id array.
    xyz_list: list[np.ndarray] = []
    obj_per_cell: list[str] = []
    for obj_id, data in objects.items():
        if data["category"] not in compositable:
            continue
        rep_xyz = data.get("rep_xyz")
        if rep_xyz is None or len(rep_xyz) == 0:
            continue
        xyz_list.append(np.asarray(rep_xyz, dtype=np.float64))
        obj_per_cell.extend([obj_id] * len(rep_xyz))

    if not xyz_list:
        return [], {}

    all_xyz = np.concatenate(xyz_list, axis=0)
    obj_arr = np.asarray(obj_per_cell, dtype=object)

    # Stage 1: outer clustering
    outer_groups = _union_find_groups(all_xyz, obj_arr, max_cell_distance_m)
    if debug_out is not None:
        # For evaluation: expose the first-stage groups as they are (no behaviour change)
        debug_out["stage1_groups"] = [list(g) for g in outer_groups]
        debug_out["params"] = {
            "max_cell_distance_m": max_cell_distance_m,
            "max_xy_extent_m": max_xy_extent_m,
            "split_cell_distance_m": split_cell_distance_m,
        }

    # Per-object cell-index buckets for fast subset slicing in stage 2.
    obj_to_idx: dict[str, np.ndarray] = {}
    for obj_id in set(obj_per_cell):
        obj_to_idx[obj_id] = np.where(obj_arr == obj_id)[0]

    # Per-object (x,y) bbox for extent checks.
    obj_xy_bbox: dict[str, tuple[float, float, float, float]] = {}
    for obj_id, idx in obj_to_idx.items():
        sub = all_xyz[idx]
        obj_xy_bbox[obj_id] = (
            float(sub[:, 0].min()), float(sub[:, 1].min()),
            float(sub[:, 0].max()), float(sub[:, 1].max()),
        )

    def group_extent(members: list[str]) -> float:
        mn_x = min(obj_xy_bbox[m][0] for m in members)
        mn_y = min(obj_xy_bbox[m][1] for m in members)
        mx_x = max(obj_xy_bbox[m][2] for m in members)
        mx_y = max(obj_xy_bbox[m][3] for m in members)
        return max(mx_x - mn_x, mx_y - mn_y)

    # Stage 2: extent gate + sub-cluster fallback.
    final_groups: list[list[str]] = []
    rejected_outer = 0
    rescued_subs = 0
    rejected_subs = 0
    for members in outer_groups:
        if len(members) < 2:
            continue
        if max_xy_extent_m is None or group_extent(members) <= max_xy_extent_m:
            final_groups.append(members)
            continue
        # Over-extent → re-cluster this group's cells at the tighter
        # split distance and keep sub-clusters that fit.
        rejected_outer += 1
        sub_idx = np.concatenate([obj_to_idx[m] for m in members])
        sub_xyz = all_xyz[sub_idx]
        sub_obj = obj_arr[sub_idx]
        sub_groups = _union_find_groups(sub_xyz, sub_obj, split_cell_distance_m)
        for sg in sub_groups:
            if len(sg) < 2:
                continue
            if group_extent(sg) <= max_xy_extent_m:
                final_groups.append(sg)
                rescued_subs += 1
            else:
                rejected_subs += 1

    composites: list[dict] = []
    obj_to_composite: dict[str, str] = {}
    for comp_idx, members in enumerate(final_groups, start=1):
        cats = [objects[m]["category"] for m in members]
        primary = max(set(cats), key=cats.count)
        comp_id = f"pole_{comp_idx}"
        composites.append({
            "composite_id": comp_id,
            "member_object_ids": sorted(members),
            "display_name": f"Pole #{comp_idx}",
            "primary_category": primary,
        })
        for m in members:
            obj_to_composite[m] = comp_id

    if rejected_outer or rescued_subs or rejected_subs:
        print(f"[COMPOSITE] outer over-extent={rejected_outer}, "
              f"rescued sub-groups={rescued_subs}, "
              f"sub-groups still over-extent={rejected_subs}")

    return composites, obj_to_composite
