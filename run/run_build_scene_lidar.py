"""LIDAR-aware scene builder for regions that have both MMS point clouds and
precise HD map SHP.

Overview:
  1. Load SHP layers for semantic source (same 11 categories we use for
     Yeouido, plus A4_SUBSIDIARYSECTION added for expressways).
  2. Load LIDAR points from one or more .las files, clipping to ROI.
  3. Voxelize points → occupancy cells (dedupe at voxel_size).
  4. Label each occupancy cell by nearest SHP feature within per-category
     radius (config/object_dimensions.json).
  5. Export as the same scene_preview.json / scene_chunks / scene_all.json
     triplet that the web already consumes — no web changes needed beyond
     registering a new scene descriptor.

Usage (defaults are tuned for a quick MVP test on the Deoksong dataset):
  python run/run_build_scene_lidar.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import box, Polygon

from src.utils.config_loader import load_all_configs
from src.core.grid_config import GridConfig
from src.core.scene_data import SceneData
from src.models.overlay_item import OverlayItem
from src.models.voxel_object import VoxelObject
from src.models.scene_chunk import SceneChunk
from src.readers.shp_reader import read_shp
from src.readers.las_reader import read_las_xyz_rgb, read_las_header
from src.voxel.pointcloud_voxelizer import voxelize_points_with_representative
from src.semantic.shp_matcher import (
    build_sources_from_shp,
    label_cells,
    build_ground_z_index,
    query_ground_z,
    build_macro_ground_grid,
)
from src.semantic.clustering import (
    cluster_labeled_cells,
    detect_composites,
)
from src.exporters.json_exporter import save_json
from src.exporters.chunk_geometry_baker import (
    bake_all_chunks,
    bake_preview_by_macro_cell,
)
from src.utils.ortho_loader import (
    try_load_ortho_for_region,
    try_load_rgb_enhance,
    try_load_area_radius,
)
from src.colors.rgb_enhance import apply_rgb_enhance

# ---------- scene config ----------
PROJECT_DATA_ROOT = Path(__file__).resolve().parents[1]

# ---- the work area (region) is defined by config/regions/<id>.json ----
# Only what differs per region (SHP/LAS paths, ROI, output, CRS, stride) is in the config;
# the tuning parameters (object_dimensions radii, corridor/pole/composite constants, voxel size, ...)
# are shared and unchanged. main() calls load_region() for --region and fills the globals below.
# New region = one JSON file in config/regions, no code change.
SCENE_ID: str = "deoksong"  # overwritten in main() with region.scene_id
OUTPUT_ROOT: Path | None = None
HDMAP_DIR: Path | None = None
LAS_SOURCES: list[Path] | None = None
ROI_BBOX: tuple[float, float, float, float] | None = None  # None = auto(LAS bounds)
CENTER_ROI_SIZE_M = 1250.0  # used only when ROI_BBOX is None (auto)
LIDAR_STRIDE = 4  # overwritten in main() with region.lidar_stride
CLIP_OVERLAYS_TO_ROI = False  # overwritten in main() from the region
ROI_POLY = None  # rotated ROI polygon (shapely Polygon); filled in main() from region.roi_poly

# SHP layer files
# Layer -> file, layer -> category, forwarded attributes and Kind dispatch
# come from config/hdmap_specs/<id>.json (HD map spec). main() calls load_hdmap_spec()
# for region.hdmap_spec and fills this global. If a new region has different layers or Kind
# codes, only the spec file changes (no code).
HDMAP_SPEC = None  # HdmapSpec, set in main()

# LIDAR semantic labels that represent GROUND-level features. These are
# rendered on the web via SHP overlays only — we drop their LIDAR voxel
# output to keep the occupancy focused on vertical/structural objects.
GROUND_CATEGORIES_LIDAR = {
    "lane", "crosswalk", "link", "node", "driveway", "subsidiary",
    "speedbump", "surface_mark", "stopline",
}

# B3_SURFACEMARK Kind dispatch (5111/5321 -> crosswalk, 537x -> stopline, otherwise
# -> surface_mark) is handled by the dispatch of the HD map spec (config/hdmap_specs).

# Categories whose assignments deserve a "is this actually a vehicle?"
# second look. A cell labeled as one of these is reclassified to
# dynamic_vehicle (and dropped) if it sits inside the road corridor and
# its elevation is in the typical vehicle range — i.e., what we see is
# almost certainly a car/truck that was in the lane during the scan,
# not the actual structure.
DYNAMIC_SUSPECT_CATEGORIES = {"traffic_light", "sign", "street_light"}
VEHICLE_ELEVATION_MIN_M = 0.3
VEHICLE_ELEVATION_MAX_M = 4.0   # trucks/buses can be ~4m

# Off-road floor filter — drop "unknown" cells (no SHP match) whose
# elevation above local ground is below this threshold. These are
# sidewalks, planted strips, parking-lot pavement, and the like:
# physically real but the user prefers them represented by the
# ground grid only, not as voxel boxes that clutter the off-road
# space. Vertical structures (poles, trees, guardrails) aren't
# affected because their elevation is well above the threshold,
# and SHP-matched ground (lane/crosswalk/etc.) is already dropped
# elsewhere — this fills the SHP-bounds gap.
OFF_ROAD_GROUND_ELEVATION_M = 0.3

# SHP-bounds filter — drop unknown cells whose XY sits outside the
# union of (road polygon ∪ road centerline buffer). After widening
# the ROI to pull SURV01/02 main-line traversals into the build,
# stretches of bare hillside / median greenery off the carriageway
# came along for the ride. The user wants the scene to feel like
# "what the SHP describes plus the LIDAR objects standing on it"
# — anything truly off-map should not be voxelized.
#
# Buffer is generous enough to retain shoulders, guardrails, and
# small road-adjacent structures without including the verge.
# 12m covers the road shoulder + guardrail + a sliver of grass
# strip on each side without pulling in hillside / off-road terrain.
SHP_RETENTION_BUFFER_M = 12.0
SHP_RETENTION_CATEGORIES = {
    "link", "driveway", "subsidiary",
    "lane", "crosswalk", "stopline", "sidewalk", "surface_mark",
}
# Corridor built from:
#   link buffer 3m (covers 6m wide → typical 2-lane road width)
#   + driveway / subsidiary polygons (actual road surface from SHP)
# Intentionally EXCLUDES B2 lane marking buffering — that bleeds past the
# curb into the sidewalk and would delete legitimate poles (learned from
# prior iteration).
ROAD_CORRIDOR_BUFFER_M = 3.0
ROAD_CORRIDOR_CATEGORIES = {"link", "driveway", "subsidiary"}
# Pole-protection: candidates that would otherwise be filtered as vehicles
# are spared if their XY column has a tall labeled structure above. A
# gantry leg's 0.3~4m LIDAR points get saved because the gantry crossbar
# at >5m anchors the column as a real structure.
POLE_PROTECTION_MIN_ELEVATION_M = 4.5
POLE_PROTECTION_CATEGORIES = {
    "traffic_light", "sign", "street_light", "height_barrier", "guardrail",
}
# Radius (in voxel cells, not meters) around a column to count as "same
# column". A value of 1 means the 3×3 XY neighborhood — absorbs minor
# LIDAR noise and voxel quantization between the leg and its top.
POLE_PROTECTION_XY_RADIUS_CELLS = 1


# ---------- helpers ----------
def safe_geom_to_geojson_dict(geom) -> dict:
    return geom.__geo_interface__


def get_chunk_id(cx: int, cy: int) -> str:
    return f"chunk_{cx}_{cy}"


def ensure_scene_chunk(chunk_map, cx: int, cy: int) -> SceneChunk:
    key = (cx, cy)
    if key not in chunk_map:
        chunk_map[key] = SceneChunk(chunk_id=get_chunk_id(cx, cy), chunk_x=cx, chunk_y=cy)
    return chunk_map[key]


def overlay_chunk_indices(geom, grid_config: GridConfig) -> List[Tuple[int, int]]:
    minx, miny, maxx, maxy = geom.bounds
    scx, scy = grid_config.world_to_macro_chunk_index(minx, miny)
    ecx, ecy = grid_config.world_to_macro_chunk_index(maxx, maxy)
    return [(cx, cy) for cx in range(scx, ecx + 1) for cy in range(scy, ecy + 1)]


def add_overlay_to_chunks(overlay, geom, chunk_map, grid_config):
    for cx, cy in overlay_chunk_indices(geom, grid_config):
        ensure_scene_chunk(chunk_map, cx, cy).overlays.append(overlay)


def add_voxel_object_to_chunks(vx_obj: VoxelObject, chunk_map, grid_config):
    touched = set()
    for gx, gy, gz in vx_obj.voxels:
        wx, wy, _ = grid_config.grid_index_to_world(gx, gy, gz)
        cx, cy = grid_config.world_to_macro_chunk_index(wx, wy)
        touched.add((cx, cy))
    for cx, cy in touched:
        ensure_scene_chunk(chunk_map, cx, cy).voxel_objects.append(vx_obj)


def clip_gdf(gdf, roi_bbox):
    if gdf is None or gdf.empty:
        return gdf
    return gdf[gdf.intersects(box(*roi_bbox))].copy()


def clip_gdf_geom(gdf, roi_bbox):
    """Clip geometries to the ROI box (intersection). shapely 2.x interpolates Z at
    the cut points, so road elevations are kept. In SHP-only regions without LiDAR,
    where the viewer's coverage clip cannot apply, this keeps roads from extending
    outside the box (region flag clip_overlays_to_roi)."""
    if gdf is None or gdf.empty:
        return gdf
    # Clip to the rotated ROI (ROI_POLY) if set, otherwise to the axis-aligned box.
    clip_geom = ROI_POLY if ROI_POLY is not None else box(*roi_bbox)
    clipped = gdf.clip(clip_geom, keep_geom_type=True)
    clipped = clipped[~clipped.geometry.is_empty]
    if clipped.empty:
        return clipped.copy()
    # Clipping can create multi-part geometries; explode them into single parts.
    return clipped.explode(index_parts=False).reset_index(drop=True).copy()


def load_hdmap_layers() -> dict:
    out = {}
    for key, filename in HDMAP_SPEC.layer_files().items():
        p = HDMAP_DIR / filename
        if not p.exists():
            print(f"[SHP] skip missing: {p.name}")
            continue
        gdf = read_shp(str(p))
        out[key] = gdf
        print(f"[SHP] loaded {key}: {len(gdf)} features")
    return out


def determine_roi() -> tuple[float, float, float, float]:
    if ROI_BBOX is not None:
        return ROI_BBOX
    if not LAS_SOURCES:
        raise RuntimeError("no LAS files specified and no ROI_BBOX set")
    hdr = read_las_header(LAS_SOURCES[0])
    cx = (hdr["mins"][0] + hdr["maxs"][0]) / 2
    cy = (hdr["mins"][1] + hdr["maxs"][1]) / 2
    half = CENTER_ROI_SIZE_M / 2
    return (cx - half, cy - half, cx + half, cy + half)


# ---------- main ----------
def main():
    import argparse
    from src.utils.region_registry import load_region, list_regions
    from src.utils.hdmap_spec import load_hdmap_spec

    _regions = list_regions(PROJECT_DATA_ROOT)
    ap = argparse.ArgumentParser(description="LIDAR scene builder (+ ablation flags)")
    ap.add_argument("--region", default="deoksong", choices=_regions or None,
                    help="region to build (config/regions/*.json). "
                         f"available: {', '.join(_regions) or '(none)'}")
    ap.add_argument("--no-zaware", action="store_true",
                    help="ABLATION: disable elevation gating in label_cells "
                         "(pure XY matching). Elevation is still computed for "
                         "the dynamic/off-road filters so only labeling changes.")
    ap.add_argument("--no-pole-protection", action="store_true",
                    help="ABLATION: skip pole-protection so all in-corridor "
                         "low-elevation structure candidates are filtered.")
    ap.add_argument("--measure-only", action="store_true",
                    help="Skip the 164MB export/bake; print label + object "
                         "counts only. For ablation runs.")
    ap.add_argument("--no-bake", action="store_true",
                    help="Skip baked-geometry emission + per-chunk JSONs. "
                         "scene_preview.json keeps voxel positions/colors but "
                         "drops the wireframe pre-bake (web falls back to "
                         "dynamic build). Cuts a large LIDAR-only scene's disk "
                         "footprint ~30x (baked geometry dominates). For "
                         "preview-mode display scenes without driving-focus.")
    ap.add_argument("--dump-labels", default=None,
                    help="Save (cells, labels_labeling, labels_final) to this "
                         ".npz for offline ablation analysis.")
    ap.add_argument("--voxel-bin", action="store_true",
                    help="Dump voxels to a compact binary (voxels.bin, ~16B/cell) "
                         "and strip voxel_objects from scene_preview.json (overlay-"
                         "only light JSON). Lets a full-island 0.5m voxelization "
                         "load at once (~80MB bin vs ~1.2GB JSON). Same voxel logic "
                         "— only output format differs. Implies no-bake/no-chunks.")
    # --- IJGI revision experiment flags (defaults = unchanged behaviour) ---
    ap.add_argument("--dimensions", default=None,
                    help="REVISION: object_dimensions JSON to use instead of "
                         "config/object_dimensions.json (e.g. sign radius 2 m "
                         "as in the submitted experiments) — shared config untouched.")
    ap.add_argument("--ground-model", default="nearest",
                    choices=["nearest", "idw", "tin", "plane", "constant"],
                    help="REVISION ablation: ground predictor used for elevations "
                         "(nearest = submitted model; NaN predictions fall back to nearest).")
    ap.add_argument("--no-vehicle-filter", action="store_true",
                    help="REVISION ablation: skip the dynamic-vehicle filter.")
    ap.add_argument("--overhead-sign-window", action="store_true",
                    help="REVISION post-hoc variant: add an elevation window from the "
                         "map's panel height for signs mounted above the roadside window.")
    ap.add_argument("--no-connected-pole-protection", action="store_true",
                    help="Reproduce the submitted pole protection (column test only). By default a "
                         "protected vehicle candidate must also be 26-connected, through pole-category "
                         "cells, to the tall structure that protects it; candidates that are not lose "
                         "the pole label and are kept as unknown (not deleted). See Supplementary Text S3.")
    ap.add_argument("--dump-stages", default=None,
                    help="REVISION: save per-cell stage masks, labels, elevations "
                         "and kept-object ids to this .npz (stage accounting / eval).")
    args = ap.parse_args()

    # ---- region selection (config/regions/<id>.json; tuning parameters are shared and unchanged) ----
    global SCENE_ID, OUTPUT_ROOT, ROI_BBOX, HDMAP_DIR, LAS_SOURCES, LIDAR_STRIDE
    global HDMAP_SPEC, CLIP_OVERLAYS_TO_ROI, ROI_POLY
    region = load_region(args.region, PROJECT_DATA_ROOT)
    SCENE_ID = region.scene_id
    OUTPUT_ROOT = region.output_root
    ROI_BBOX = region.roi
    HDMAP_DIR = region.hdmap_dir
    LAS_SOURCES = region.las_sources
    LIDAR_STRIDE = region.lidar_stride
    CLIP_OVERLAYS_TO_ROI = region.clip_overlays_to_roi
    ROI_POLY = Polygon(region.roi_poly) if region.roi_poly else None
    if ROI_POLY is not None:
        print(f"[ROI] oriented polygon ({len(region.roi_poly)} corners) — geom clip")
    HDMAP_SPEC = load_hdmap_spec(region.hdmap_spec, PROJECT_DATA_ROOT)
    print(f"[REGION] {SCENE_ID}  output={OUTPUT_ROOT.name}  "
          f"source={region.source_name}  CRS={region.crs}  spec={HDMAP_SPEC.spec_id}")

    # Lightweight stage timing (deterministic runtime measurement, §Results).
    import time
    _t_last = [time.perf_counter()]
    _timings: list[tuple[str, float]] = []

    def mark(name: str):
        now = time.perf_counter()
        _timings.append((name, now - _t_last[0]))
        _t_last[0] = now

    def print_timings():
        total = sum(d for _, d in _timings)
        print("\n[TIMING] per-stage wall-clock (s):")
        for name, d in _timings:
            print(f"    {name:<22} {d:8.2f}  ({100*d/max(total,1e-9):4.1f}%)")
        print(f"    {'TOTAL':<22} {total:8.2f}")

    configs = load_all_configs(str(PROJECT_DATA_ROOT))
    base_grid_config = GridConfig.from_dict(configs["grid"])
    color_map = configs["color_map"]
    dimensions_path = (Path(args.dimensions) if args.dimensions
                       else PROJECT_DATA_ROOT / "config/object_dimensions.json")
    print(f"[DIMENSIONS] {dimensions_path}")
    with open(dimensions_path, encoding="utf-8") as f:
        dimensions = json.load(f)

    roi = determine_roi()
    print(f"[ROI] {roi}  (size ≈ {roi[2]-roi[0]:.1f}m × {roi[3]-roi[1]:.1f}m)")

    # 1) Load + clip SHP layers to ROI
    _t_last[0] = time.perf_counter()  # start timing the build proper
    gdfs = load_hdmap_layers()
    _clip = clip_gdf_geom if CLIP_OVERLAYS_TO_ROI else clip_gdf
    for k, v in gdfs.items():
        gdfs[k] = _clip(v, roi)
        n = 0 if gdfs[k] is None else len(gdfs[k])
        print(f"[ROI-clip{'(geom)' if CLIP_OVERLAYS_TO_ROI else ''}] {k}: {n}")

    # 2) Compute auto origin from ROI
    # If the region sets voxel_size, use it instead of the shared grid_config (to reduce
    # the cell count of a large LiDAR-only ROI). Otherwise the default is used.
    vsize = region.voxel_size if region.voxel_size else base_grid_config.voxel_size
    grid_config = GridConfig(
        macro_grid_size=base_grid_config.macro_grid_size,
        voxel_size=vsize,
        ground_cell_size=vsize,
        origin=(float(roi[0]), float(roi[1]), 0.0),
        default_scene_height=base_grid_config.default_scene_height,
        allow_float_size=base_grid_config.allow_float_size,
    )
    print(f"[ORIGIN] {grid_config.origin}  voxel_size={vsize}")

    scene = SceneData(grid_config=grid_config)
    chunk_map: Dict[Tuple[int, int], SceneChunk] = {}

    # 3) SHP overlays (road map as-is)
    shp_features_for_semantic: list[dict] = []
    for layer_key in HDMAP_SPEC.layers:
        gdf = gdfs.get(layer_key)
        if gdf is None or gdf.empty:
            continue
        role = HDMAP_SPEC.role(layer_key)
        fwd_keys = HDMAP_SPEC.attributes_for(layer_key)  # SHP columns forwarded to the viewer
        per_cat_counts: dict[str, int] = {}
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            # Category: per-code branch if the spec has a dispatch (e.g. Kind of B3),
            # otherwise the fixed category. Unmapped codes get dispatch_default (surface_mark).
            cat = HDMAP_SPEC.category_for(layer_key, row)
            color = color_map.get(cat, "#bdbdbd")
            attrs: dict = {}
            for key in fwd_keys:
                val = row.get(key)
                if val is not None:
                    attrs[key] = val
            overlay = OverlayItem(
                overlay_id=f"{cat}_{idx}",
                category=cat,
                geometry_type=geom.geom_type,
                geometry=safe_geom_to_geojson_dict(geom),
                color=color,
                style={"mode": "geometry_only" if role == "overlay" else "point_marker"},
                attributes=attrs,
            )
            scene.add_overlay(overlay)
            add_overlay_to_chunks(overlay, geom, chunk_map, grid_config)
            shp_features_for_semantic.append({
                "category": cat,
                "geometry": geom,
                "shp_feature_id": f"{layer_key}_{idx}",
            })
            per_cat_counts[cat] = per_cat_counts.get(cat, 0) + 1
        for cat, n in sorted(per_cat_counts.items()):
            print(f"[OK] overlays({cat}) += {n}")
    mark("shp_load+overlay")

    # 4) Load LIDAR + voxelize
    print(f"[LAS] loading {len(LAS_SOURCES)} file(s), stride={LIDAR_STRIDE}")
    pts, rgb = read_las_xyz_rgb(LAS_SOURCES, roi_bbox=roi, stride=LIDAR_STRIDE)
    print(f"[LAS] total kept: {len(pts):,} points")
    mark("las_load")

    if len(pts) == 0:
        print("[WARN] no LIDAR points in ROI — skipping occupancy voxelization")
    else:
        cells, rep_xyz, rep_rgb = voxelize_points_with_representative(
            pts, rgb,
            voxel_size=grid_config.voxel_size,
            origin=grid_config.origin,
        )
        print(f"[VOXEL] {len(cells):,} occupied cells from {len(pts):,} points "
              f"({len(pts)/max(1,len(cells)):.1f} pts/cell avg)")
        mark("voxelize")

        # Snapshot the raw LIDAR-baked RGB before any ortho/enhance
        # touches rep_rgb in place. This snapshot is emitted alongside
        # voxel_colors as voxel_colors_raw so the viewer can flip
        # between "ortho + boost" (default) and "as the LAS shipped it"
        # without a rebuild. Same index space as rep_rgb itself.
        rep_rgb_raw = rep_rgb.copy()

        # Orthophoto recolour — swap each voxel's LIDAR-measured RGB
        # for the matching pixel from a user-supplied aerial capture,
        # falling back to LIDAR RGB outside the captured sub-bbox.
        # The viewer just consumes voxel_colors as before, so this
        # bake-time substitution gives the scene a photographic look
        # with zero web changes. Skipped silently when no ortho is
        # configured / enabled for this region (see config/ortho.json).
        ortho = try_load_ortho_for_region(SCENE_ID, PROJECT_DATA_ROOT)
        if ortho is not None:
            area_radius = try_load_area_radius(SCENE_ID, PROJECT_DATA_ROOT)
            ortho_rgb, ortho_mask = ortho.sample(
                rep_xyz[:, 0],
                rep_xyz[:, 1],
                area_radius_m=area_radius,
            )
            n_hit = int(ortho_mask.sum())
            if n_hit > 0:
                rep_rgb = np.where(
                    ortho_mask[:, None], ortho_rgb, rep_rgb
                ).astype(np.uint8)
            print(
                f"[ORTHO] recoloured {n_hit:,}/{len(rep_rgb):,} voxels "
                f"from {Path(ortho.image_path).name} "
                f"(centre={ortho.centre_utm} size={ortho.size_m} "
                f"rot={ortho.rotation_deg:+.2f}° "
                f"area_radius_m={area_radius})"
            )
            mark("ortho_recolour")

        # RGB enhancement — brightness/saturation/contrast boost
        # applied to whatever rep_rgb ended up being (ortho-recoloured
        # or raw LIDAR). Lifts the dim, grey-leaning tone the MMS LAS
        # tends to bake in (mean ~90/255, sat ~0.38 in Deoksong).
        # Off by default; per-region opt-in via config/ortho.json's
        # `rgb_enhance` block.
        enhance_cfg = try_load_rgb_enhance(SCENE_ID, PROJECT_DATA_ROOT)
        if enhance_cfg:
            before_mean = rep_rgb.mean()
            rep_rgb = apply_rgb_enhance(rep_rgb, **enhance_cfg)
            # Apply the SAME enhancement to the raw snapshot. Without
            # this the viewer's 'raw' source mode dropped the boost
            # while ortho mode's outside-rectangle voxels kept it,
            # producing dim raw vs vivid ortho-fallback even though
            # both paths sample the same LAS pixels. Both flavours
            # should display under the same tone curve so the user can
            # compare them apples-to-apples.
            rep_rgb_raw = apply_rgb_enhance(rep_rgb_raw, **enhance_cfg)
            print(
                f"[ENHANCE] rep_rgb + rep_rgb_raw boosted {enhance_cfg} "
                f"(mean {before_mean:.1f} -> {rep_rgb.mean():.1f})"
            )
            mark("rgb_enhance")

        # 5) Semantic labeling (Z-aware)
        ox, oy, oz = grid_config.origin
        vs = grid_config.voxel_size
        cell_centers_xy = np.column_stack((
            ox + (cells[:, 0] + 0.5) * vs,
            oy + (cells[:, 1] + 0.5) * vs,
        ))
        cell_centers_z = oz + (cells[:, 2] + 0.5) * vs  # world Z

        # Build ground-Z index from ground-category SHP overlays so we can
        # compute each cell's elevation above local ground. Without this
        # we can't distinguish a guardrail vs a tree branch sitting 5 m
        # above the same XY.
        ground_tree, ground_z_arr = build_ground_z_index(
            shp_features_for_semantic
        )
        if ground_tree is None:
            print("[WARN] no ground SHP samples — Z-aware labeling disabled")
            elevations = None
        else:
            local_ground = query_ground_z(
                ground_tree, ground_z_arr, cell_centers_xy
            )
            if args.ground_model != "nearest":
                from src.semantic import ground_models as _gm
                if args.ground_model == "constant":
                    alt = np.full(len(cells), float(np.median(ground_z_arr)))
                elif args.ground_model == "idw":
                    alt = _gm.predict_idw(ground_tree, ground_z_arr, cell_centers_xy)
                elif args.ground_model == "tin":
                    alt, _ = _gm.predict_tin(ground_tree.data, ground_z_arr, cell_centers_xy)
                else:
                    alt, _ = _gm.predict_plane(ground_tree, ground_tree.data,
                                               ground_z_arr, cell_centers_xy)
                n_nan = int(np.isnan(alt).sum())
                local_ground = np.where(np.isnan(alt), local_ground, alt)
                print(f"[ABLATION] ground model = {args.ground_model} "
                      f"({n_nan:,} cells fell back to nearest)")
            elevations = cell_centers_z - local_ground
            print(
                f"[GROUND] elevation computed for {len(cells):,} cells "
                f"(min {elevations.min():.2f}m, max {elevations.max():.2f}m, "
                f"median {float(np.median(elevations)):.2f}m)"
            )
        mark("ground_elevation")

        sources = build_sources_from_shp(
            shp_features_for_semantic,
            dimensions,
            linestring_sample_step=vs,
            polygon_sample_step=vs * 2,
        )
        if args.overhead_sign_window and ground_tree is not None:
            # POST-HOC VARIANT (not the evaluated configuration): signs whose
            # mapped panel sits above the roadside window get a second sign
            # source with a window taken from the map's own panel height.
            from src.semantic.shp_matcher import SemanticSource, sample_linestring
            sign_r = float(dimensions["sign"]["semantic_match_radius_m"])
            lo_hi, samp = [], []
            for f in shp_features_for_semantic:
                g = f["geometry"]
                if f["category"] != "sign" or g is None or g.is_empty or not g.has_z:
                    continue
                zs = np.array([c[2] for c in (g.exterior.coords if g.geom_type == "Polygon" else g.coords)])
                c0 = np.array([[g.centroid.x, g.centroid.y]])
                panel = float(zs.mean() - query_ground_z(ground_tree, ground_z_arr, c0)[0])
                if panel <= dimensions["sign"]["z_range_m"][1]:
                    continue
                xy = (sample_linestring(np.array(list(g.exterior.coords))[:, :2], vs * 2)
                      if g.geom_type == "Polygon" else c0)
                samp.append(xy); lo_hi.append(panel)
            if samp:
                lo = min(lo_hi) - 1.5; hi = max(lo_hi) + 1.5
                sources.append(SemanticSource(category="sign", samples_xy=np.vstack(samp),
                                              match_radius_m=sign_r, z_range_m=(lo, hi)))
                print(f"[POST-HOC] overhead-sign source: {len(samp)} signs, window [{lo:.1f}, {hi:.1f}] m")
        labels = label_cells(
            cell_centers_xy,
            sources,
            default_label="unknown",
            # ABLATION: with --no-zaware the elevation gate is removed from
            # labeling only (elevations stays available for the filters).
            elevations=(None if args.no_zaware else elevations),
        )
        if args.no_zaware:
            print("[ABLATION] Z-aware labeling DISABLED (pure XY matching)")
        # Snapshot the labels right after labeling, before the dynamic filter
        # mutates some to dynamic_vehicle — needed for the Z-gate ablation diff.
        labels_after_labeling = labels.copy()
        mark("label_cells")
        print("[SEM] label distribution (pre-dynamic filter):")
        for lbl in np.unique(labels):
            print(f"    {lbl}: {(labels == lbl).sum():,}")

        # Dynamic-object filter — drop cells that look like vehicles
        # (low elevation, inside road corridor, currently labeled as a
        # vertical structure). Classic MMS false positive.
        n_cells_all = len(labels)
        st_in_road = np.zeros(n_cells_all, dtype=bool)
        st_vehicle_candidate = np.zeros(n_cells_all, dtype=bool)
        st_pole_protected = np.zeros(n_cells_all, dtype=bool)
        if args.no_vehicle_filter:
            print("[ABLATION] dynamic-vehicle filter DISABLED")
        if elevations is not None and not args.no_vehicle_filter:
            corridor_geoms = [
                f["geometry"].buffer(ROAD_CORRIDOR_BUFFER_M)
                for f in shp_features_for_semantic
                if f["category"] in ROAD_CORRIDOR_CATEGORIES
                and f["geometry"] is not None
                and not f["geometry"].is_empty
            ]
            if corridor_geoms:
                from shapely.ops import unary_union
                from shapely import vectorized as shp_vec
                corridor = unary_union(corridor_geoms)
                in_road = shp_vec.contains(
                    corridor, cell_centers_xy[:, 0], cell_centers_xy[:, 1]
                )
                in_vehicle_z = (
                    (elevations >= VEHICLE_ELEVATION_MIN_M)
                    & (elevations <= VEHICLE_ELEVATION_MAX_M)
                )
                suspect_mask = np.zeros(len(labels), dtype=bool)
                for cat in DYNAMIC_SUSPECT_CATEGORIES:
                    suspect_mask |= (labels == cat)
                vehicle_mask = in_road & in_vehicle_z & suspect_mask
                n_pre = int(vehicle_mask.sum())
                st_in_road = in_road.copy()
                st_vehicle_candidate = vehicle_mask.copy()

                # Pole protection: any XY that has a tall structural cell
                # above (e.g. gantry crossbar, traffic-light head) is
                # treated as a real column, so its lower cells are NOT
                # filtered as vehicles. Without this, gantry legs
                # vanished and the overhead signal appeared to float.
                tall_struct = np.zeros(len(labels), dtype=bool)
                for cat in POLE_PROTECTION_CATEGORIES:
                    tall_struct |= (labels == cat)
                tall_struct &= (elevations >= POLE_PROTECTION_MIN_ELEVATION_M)
                # Build set of (gx, gy) that anchor a tall structure.
                protected_xy: set[tuple[int, int]] = set()
                for i in np.where(tall_struct)[0]:
                    gxi = int(cells[i, 0])
                    gyi = int(cells[i, 1])
                    r = POLE_PROTECTION_XY_RADIUS_CELLS
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            protected_xy.add((gxi + dx, gyi + dy))

                if args.no_pole_protection:
                    print("[ABLATION] pole-protection DISABLED")
                elif protected_xy:
                    restored = 0
                    for idx in np.where(vehicle_mask)[0]:
                        key = (int(cells[idx, 0]), int(cells[idx, 1]))
                        if key in protected_xy:
                            vehicle_mask[idx] = False
                            restored += 1
                    if restored:
                        print(
                            f"[DYNAMIC] pole-protected {restored:,} cells "
                            f"(XY has structure above {POLE_PROTECTION_MIN_ELEVATION_M}m)"
                        )

                st_pole_protected = st_vehicle_candidate & ~vehicle_mask

                # Connected pole protection (default): the column test above also
                # protects a vehicle that passes under a signal arm, because the
                # signal head lies in the same column. Keep the protection only for
                # candidates that are 26-connected, through pole-category cells, to
                # a tall structure cell; the others lose the pole label and stay as
                # unknown occupancy (not deleted, since a detached low fragment can
                # also be real structure such as a barrier).
                if not args.no_pole_protection and not args.no_connected_pole_protection and st_pole_protected.any():
                    pole_cat = np.zeros(len(labels), dtype=bool)
                    for cat in POLE_PROTECTION_CATEGORIES | DYNAMIC_SUSPECT_CATEGORIES:
                        pole_cat |= (labels == cat)
                    idx_pole = np.where(pole_cat)[0]
                    pos = {tuple(int(v) for v in cells[i, :3]): k for k, i in enumerate(idx_pole)}
                    parent = list(range(len(idx_pole)))

                    def _find(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a
                    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                            if (dx, dy, dz) > (0, 0, 0)]
                    for (x, y, z), k in pos.items():
                        for dx, dy, dz in offs:
                            j = pos.get((x + dx, y + dy, z + dz))
                            if j is not None:
                                ra, rb = _find(k), _find(j)
                                if ra != rb:
                                    parent[ra] = rb
                    anchored = {_find(k) for k, i in enumerate(idx_pole) if tall_struct[i]}
                    comp_of = np.full(len(labels), -1, dtype=np.int64)
                    for k, i in enumerate(idx_pole):
                        comp_of[i] = _find(k)
                    detach = st_pole_protected & ~np.isin(comp_of, list(anchored))
                    n_det = int(detach.sum())
                    if n_det:
                        labels = labels.copy()
                        labels[detach] = "unknown"
                        st_pole_protected = st_pole_protected & ~detach
                        print(f"[DYNAMIC] connected pole protection: {n_det:,} protected cells not connected "
                              f"to the structure above -> unknown (kept)")
                n_dyn = int(vehicle_mask.sum())
                if n_dyn:
                    labels = labels.copy()
                    labels[vehicle_mask] = "dynamic_vehicle"
                    print(
                        f"[DYNAMIC] reclassified {n_dyn:,} cells as dynamic_vehicle "
                        f"(of {n_pre:,} candidates; {n_pre - n_dyn:,} pole-protected)"
                    )
        mark("dynamic_filter")

        # 6) Drop ground & dynamic labels, then cluster the remaining
        # labeled cells into one VoxelObject per SHP feature (so each
        # pole becomes an independently selectable object). 'unknown'
        # stays merged to keep picking-box count bounded.
        # Off-road floor: unknown + low elevation. Sidewalks, planted
        # strips, parking-lot pavement that sit outside any SHP
        # polygon. Dropped outright — the user prefers them
        # represented by the ground grid only, not as voxel boxes
        # that clutter the off-road space.
        DROPPED_LABELS = GROUND_CATEGORIES_LIDAR | {"dynamic_vehicle"}
        keep_mask = ~np.isin(labels, list(DROPPED_LABELS))
        off_road_ground_mask = np.zeros(len(labels), dtype=bool)
        if elevations is not None:
            off_road_ground_mask = (
                (labels == "unknown")
                & (elevations < OFF_ROAD_GROUND_ELEVATION_M)
            )
            keep_mask &= ~off_road_ground_mask

        # SHP-bounds filter: drop unknown cells outside the union of
        # SHP road geometries (polygons + line buffers). Reuses the
        # corridor union we already build for the dynamic-vehicle
        # check, but widened and broadened so the keep-set is a
        # superset of "what looks like road from the SHP".
        out_of_shp_mask = np.zeros(len(labels), dtype=bool)
        retention_geoms = [
            f["geometry"].buffer(SHP_RETENTION_BUFFER_M)
            for f in shp_features_for_semantic
            if f["category"] in SHP_RETENTION_CATEGORIES
            and f["geometry"] is not None
            and not f["geometry"].is_empty
        ]
        if retention_geoms:
            from shapely.ops import unary_union as _unary_union
            from shapely import vectorized as _shp_vec
            retention_corridor = _unary_union(retention_geoms)
            in_shp = _shp_vec.contains(
                retention_corridor,
                cell_centers_xy[:, 0],
                cell_centers_xy[:, 1],
            )
            out_of_shp_mask = (labels == "unknown") & ~in_shp
            keep_mask &= ~out_of_shp_mask
        kept_cells = cells[keep_mask]
        kept_xyz = rep_xyz[keep_mask]
        kept_rgb = rep_rgb[keep_mask]
        # rep_rgb_raw snapshot — same index space as rep_rgb, so the
        # same keep_mask gives the matching raw-RGB subset. Shipped
        # alongside voxel_colors so the viewer can flip processed/raw.
        kept_rgb_raw = rep_rgb_raw[keep_mask]
        kept_labels = labels[keep_mask]
        kept_centers = cell_centers_xy[keep_mask]
        dropped_ground = int(
            np.isin(labels, list(GROUND_CATEGORIES_LIDAR)).sum()
        )
        dropped_dynamic = int((labels == "dynamic_vehicle").sum())
        dropped_off_road = int(off_road_ground_mask.sum())
        if dropped_ground:
            print(f"[FILTER] dropped {dropped_ground:,} ground-labeled cells "
                  f"(represented by SHP overlays instead)")
        if dropped_dynamic:
            print(f"[FILTER] dropped {dropped_dynamic:,} dynamic_vehicle cells "
                  f"(in-road low-elevation false positives)")
        if dropped_off_road:
            print(f"[FILTER] dropped {dropped_off_road:,} off-road ground cells "
                  f"(unknown + elevation < {OFF_ROAD_GROUND_ELEVATION_M}m)")
        dropped_out_of_shp = int(out_of_shp_mask.sum())
        if dropped_out_of_shp:
            print(f"[FILTER] dropped {dropped_out_of_shp:,} out-of-SHP cells "
                  f"(unknown outside SHP buffer {SHP_RETENTION_BUFFER_M}m)")
        mark("noise_filters")

        # ABLATION dump: cells + labels (post-labeling and post-dynamic-filter)
        # for offline analysis against the manual audit.
        if args.dump_labels:
            np.savez(
                args.dump_labels,
                cells=cells.astype(np.int32),
                labels_labeling=labels_after_labeling.astype("U24"),
                labels_final=labels.astype("U24"),
            )
            print(f"[DUMP] labels → {args.dump_labels}")

        # Per-SHP-feature clustering. 'unknown' merged; the rest split
        # so each physical pole/line becomes one VoxelObject.
        clustered = cluster_labeled_cells(
            kept_cells,
            kept_labels,
            kept_xyz,
            kept_rgb,
            kept_centers,
            shp_features_for_semantic,
            keep_merged_categories={"unknown"},
            rep_rgb_raw=kept_rgb_raw,
        )
        print(f"[CLUSTER] produced {len(clustered):,} VoxelObjects from {len(kept_cells):,} cells")
        mark("clustering")

        if args.dump_stages:
            # Per-cell object id for kept cells ("" for dropped cells).
            key_to_row = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(cells)}
            obj_of_cell = np.full(len(cells), "", dtype=object)
            for obj_id, data in clustered.items():
                for c in data["cells"]:
                    obj_of_cell[key_to_row[(int(c[0]), int(c[1]), int(c[2]))]] = obj_id
            # XY distance from each cell centre to the nearest sampled map
            # feature of every non-ground category (for evaluation strata).
            from scipy.spatial import cKDTree as _KD
            dist_extra = {}
            for src in sources:
                if src.category in GROUND_CATEGORIES_LIDAR or len(src.samples_xy) == 0:
                    continue
                d_src, _ = _KD(src.samples_xy).query(cell_centers_xy, k=1)
                key = f"dist_{src.category}"
                dist_extra[key] = np.minimum(dist_extra.get(key, np.inf), d_src).astype(np.float32)
            np.savez_compressed(
                args.dump_stages,
                cells=cells.astype(np.int32),
                rep_xyz=rep_xyz.astype(np.float64),
                elevation=(elevations if elevations is not None
                           else np.full(len(cells), np.nan)).astype(np.float32),
                labels_labeling=labels_after_labeling.astype("U24"),
                labels_final=labels.astype("U24"),
                in_road=st_in_road,
                vehicle_candidate=st_vehicle_candidate,
                pole_protected=st_pole_protected,
                drop_ground=np.isin(labels, list(GROUND_CATEGORIES_LIDAR)),
                drop_dynamic=(labels == "dynamic_vehicle"),
                drop_off_road_low=off_road_ground_mask,
                drop_out_of_shp=out_of_shp_mask,
                keep=keep_mask,
                object_id=obj_of_cell.astype("U64"),
                origin=np.asarray(grid_config.origin, dtype=np.float64),
                voxel_size=np.float64(grid_config.voxel_size),
                n_points=np.int64(len(pts)),
                args_json=np.str_(json.dumps(vars(args), ensure_ascii=False)),
                dimensions_path=np.str_(str(dimensions_path)),
                las_sources=np.asarray([str(p) for p in LAS_SOURCES]),
                **dist_extra,
            )
            print(f"[DUMP] stages → {args.dump_stages}")

        # Per-category VoxelObject counts (ablation signal: do whole
        # poles/gantries disappear under a variant?).
        from collections import Counter as _Counter
        obj_cat_counts = _Counter(d["category"] for d in clustered.values())
        print("[OBJECTS] per-category VoxelObject counts:")
        for c, n in obj_cat_counts.most_common():
            print(f"    {c}: {n}")
        # Post-filter cell composition (comparable to §5.3 output composition).
        _cell_comp = _Counter(str(l) for l in kept_labels)
        _tot = sum(_cell_comp.values())
        print(f"[CELLS] post-filter composition ({_tot:,} cells):")
        for c, n in _cell_comp.most_common():
            print(f"    {c}: {n:,} ({100*n/max(_tot,1):.1f}%)")

        # Composite detection — pole-sharing neighbors get bundled so
        # the viewer can offer Ctrl+Shift+Click "select whole pole".
        # VOXEL_DEBUG_COMPOSITES=<path.json> dumps the first-stage groups (evaluation only; off by default)
        _comp_debug: dict | None = {} if os.environ.get("VOXEL_DEBUG_COMPOSITES") else None
        composite_list, obj_to_composite = detect_composites(
            clustered,
            debug_out=_comp_debug,
            compositable_categories={
                "traffic_light", "sign", "street_light", "height_barrier",
            },
            # Cell distance stays permissive (1.5m) so sparse-LIDAR
            # poles still get their stack picked up. Cross-pole
            # bridging is rejected by max_xy_extent_m below — a real
            # pole's member cells share a < 1m (x,y) footprint, two
            # poles at an intersection are ≥ 3m apart.
            max_cell_distance_m=1.5,
            # 6.0m is calibrated from observed extent distribution on
            # deoksong: real single-pole composites span 1.5–5.0m
            # (sign arms + multi-stack), cross-pole bridged groups
            # land at 11–22m. 6.0m sits in the natural gap and
            # rejects all bridged super-groups without breaking any
            # real shared-pole.
            # Real single-pole composites on deoksong span 1.5–5m
            # (sign arms + multi-stack). Cross-pole bridged super-
            # groups land at 11–22m. 6m sits in the natural gap.
            # When the outer pass over-shoots, _union_find_groups
            # re-clusters internally at split_cell_distance_m (0.3m,
            # voxel-touching only) and keeps the sub-clusters whose
            # own footprint stays under 6m.
            max_xy_extent_m=6.0,
        )
        if _comp_debug is not None:
            with open(os.environ["VOXEL_DEBUG_COMPOSITES"], "w", encoding="utf-8") as f:
                json.dump(_comp_debug, f, ensure_ascii=False, indent=1)
        if composite_list:
            total_members = sum(len(c["member_object_ids"]) for c in composite_list)
            print(f"[COMPOSITE] {len(composite_list)} composites covering "
                  f"{total_members} objects (shared poles)")
        scene.composite_objects = composite_list
        mark("composite")
        if args.dump_stages:
            side = Path(args.dump_stages).with_suffix(".objects.json")
            side.write_text(json.dumps({
                "objects": {oid: {"category": d["category"],
                                  "shp_feature_id": d.get("shp_feature_id"),
                                  "n_cells": int(len(d["cells"])),
                                  "composite_id": obj_to_composite.get(oid)}
                            for oid, d in clustered.items()},
                "composites": composite_list,
                "timings_s": dict(_timings),
            }, ensure_ascii=False, default=str), encoding="utf-8")
            print(f"[DUMP] objects/composites → {side}")

        if args.measure_only:
            print_timings()
            print("[MEASURE-ONLY] skipping export/bake/save.")
            return

        # Emit VoxelObjects
        for obj_id, data in clustered.items():
            cat = data["category"]
            cells_np = data["cells"]
            xyz_np = data["rep_xyz"]
            rgb_np = data["rep_rgb"]
            if len(cells_np) == 0:
                continue
            color = color_map.get(cat, color_map.get("unknown", "#bdbdbd"))
            vx = VoxelObject(
                object_id=obj_id,
                category=cat,
                color=color,
                attributes={
                    "source": "lidar",
                    "shp_geom": "pointcloud",
                    "shp_feature_id": data.get("shp_feature_id"),
                    "composite_id": obj_to_composite.get(obj_id),
                    "cell_count": int(len(cells_np)),
                    "voxel_positions": np.asarray(xyz_np).reshape(-1).tolist(),
                    "voxel_colors": np.asarray(rgb_np).reshape(-1).tolist(),
                    # Raw LIDAR-baked RGB (pre-ortho, pre-enhance) — the
                    # the viewer's "original LAS color" option reads this when set.
                    "voxel_colors_raw": (
                        np.asarray(data["rep_rgb_raw"]).reshape(-1).tolist()
                        if data.get("rep_rgb_raw") is not None
                        else None
                    ),
                },
            )
            vx.extend_voxels([(int(c[0]), int(c[1]), int(c[2])) for c in cells_np])
            scene.add_voxel_object(vx)
            add_voxel_object_to_chunks(vx, chunk_map, grid_config)

    # 7) Export (same layout as run_build_scene.py)
    preview_dir = OUTPUT_ROOT / "preview_json"
    chunk_dir = OUTPUT_ROOT / "scene_chunks"
    scene_all_dir = OUTPUT_ROOT / "scene_all"
    # Clear stale chunks from a previous build so the directory only
    # holds files from the current ROI / cell layout. Without this
    # the dir balloons (sample run grew from 90 to 2,460 files in
    # five rebuilds) and the viewer's manifest filters them out, so
    # they're pure dead weight on disk.
    if chunk_dir.exists():
        import shutil
        shutil.rmtree(chunk_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    scene_all_dir.mkdir(parents=True, exist_ok=True)

    # --voxel-bin: serialise voxels to compact binary (voxels.bin) + strip
    # voxel_objects from the JSON (overlay-only light preview). Same voxel data
    # as scene.voxel_objects — only the format differs (web decodes voxels.bin).
    voxel_bin_meta = None
    if args.voxel_bin:
        from run.run_export_pointcloud import write_voxel_bin
        cats_seen, cat_to_id = [], {}
        cells_parts, rgb_parts, cat_parts, obj_parts = [], [], [], []
        obj_table = []  # u16 index -> VoxelObject identity (for click picking / selection)
        for vx in scene.voxel_objects:
            if not vx.voxels:
                continue
            if vx.category not in cat_to_id:
                cat_to_id[vx.category] = len(cats_seen)
                cats_seen.append(vx.category)
            cells_np = np.asarray(vx.voxels, dtype=np.int64)  # (M,3) gx,gy,gz
            cols = vx.attributes.get("voxel_colors")
            rgb_np = (np.asarray(cols, dtype=np.uint8).reshape(-1, 3)
                      if cols else np.full((len(cells_np), 3), 180, dtype=np.uint8))
            obj_idx = len(obj_table)
            if obj_idx > 0xFFFF:
                raise SystemExit("[ERR] --voxel-bin: more than 65535 VoxelObjects (exceeds u16)")
            obj_table.append({
                "idx": obj_idx,
                "object_id": vx.object_id,
                "category": vx.category,
                "cell_count": int(len(cells_np)),
                "composite_id": vx.attributes.get("composite_id"),
            })
            cells_parts.append(cells_np)
            rgb_parts.append(rgb_np)
            cat_parts.append(np.full(len(cells_np), cat_to_id[vx.category], dtype=np.uint8))
            obj_parts.append(np.full(len(cells_np), obj_idx, dtype=np.uint16))
        cells_all = np.vstack(cells_parts)
        rgb_all = np.vstack(rgb_parts)
        cat_all = np.concatenate(cat_parts)
        obj_all = np.concatenate(obj_parts)
        vs = grid_config.voxel_size
        ox, oy, oz = grid_config.origin
        # grid cell center, RELATIVE to origin (web re-adds origin − sceneOrigin)
        centers_rel = ((cells_all.astype(np.float64) + 0.5) * vs).astype(np.float32)
        vb_path = OUTPUT_ROOT / "voxels.bin"
        write_voxel_bin(vb_path, centers_rel, rgb_all, cat_all,
                        (float(ox), float(oy), float(oz)), float(vs), obj_all)
        voxel_bin_meta = {
            "file": "voxels.bin",
            "count": int(len(cells_all)),
            "voxel_size": float(vs),
            "origin": [float(ox), float(oy), float(oz)],
            "categories": [
                {"id": i, "name": c,
                 "color": color_map.get(c, color_map.get("unknown", "#bdbdbd"))}
                for i, c in enumerate(cats_seen)
            ],
            # Per-cell u16 object index -> this table. Click picking resolves to object_id
            # and follows the same selection / highlight / info-panel flow as Deoksong.
            "objects": obj_table,
        }
        scene.voxel_objects = []  # strip heavy voxels → overlay-only light JSON
        print(f"[VOXEL-BIN] {len(cells_all):,} cells → {vb_path.name} "
              f"({vb_path.stat().st_size / 1024 / 1024:.1f}MB), {len(cats_seen)} cats")

    # Per-macro-cell bake for preview-mode scenes (deoksong loads
    # scene_preview.json directly and never touches per-chunk JSONs).
    # Pre-build the renderable geometry the web's CellChunk would
    # otherwise assemble at cell-mount time — eliminates the
    # per-voxel JS loop that spiked ~80 ms on driving-focus cell
    # crossings.
    if args.no_bake or args.voxel_bin:
        scene.baked_geometry_by_cell = {}
        print("[BAKE] skipped (--no-bake/--voxel-bin): web uses dynamic/binary geometry")
    else:
        preview_baked, cells_baked, preview_voxels = bake_preview_by_macro_cell(
            scene.voxel_objects, grid_config
        )
        scene.baked_geometry_by_cell = preview_baked
        print(
            f"[BAKE] preview: {cells_baked} macro cells, "
            f"{preview_voxels:,} voxels of pre-built geometry"
        )

    preview_path = preview_dir / "scene_preview.json"
    preview_dict = scene.to_dict()
    if voxel_bin_meta is not None:
        # legend(id→category/color) + bin meta so the web voxel layer can decode
        # voxels.bin and color/toggle by category without the heavy voxel_objects.
        preview_dict["voxel_bin"] = voxel_bin_meta
    save_json(preview_dict, str(preview_path))
    print(f"[SAVE] {preview_path}")

    # Same bake also stamped onto per-chunk JSONs so streaming-mode
    # scenes (yeouido, etc.) get the same speedup once they switch
    # over. Cheap addition since the data is already computed per
    # voxel — the chunk path just re-buckets per chunk_x/chunk_y.
    # Per-chunk JSONs (streaming mode). Skipped with --no-bake — a preview-mode
    # display scene loads scene_preview.json directly and never fetches chunks,
    # so writing them (esp. with baked geometry) is pure disk cost.
    chunk_entries = []
    if args.no_bake or args.voxel_bin:
        print("[SAVE] chunks skipped (--no-bake/--voxel-bin; preview-mode display scene)")
    else:
        chunks_baked, voxels_baked = bake_all_chunks(chunk_map, grid_config)
        print(f"[BAKE] chunks: {chunks_baked} chunks, {voxels_baked:,} voxels of pre-built geometry")
        for (cx, cy), sc in chunk_map.items():
            p = chunk_dir / f"{sc.chunk_id}.json"
            save_json(sc.to_dict(), str(p))
            chunk_entries.append({
                "chunk_id": sc.chunk_id,
                "chunk_x": cx,
                "chunk_y": cy,
                "path": f"data/{OUTPUT_ROOT.name}/scene_chunks/{sc.chunk_id}.json",
            })
        print(f"[SAVE] {len(chunk_entries)} chunks")

    # Per-macro-cell ground Z lets the viewer place the wireframe
    # grid on the actual road surface for every region of the scene
    # — replacing the single-scalar groundZOffset that broke as soon
    # as the ROI started spanning multi-level roads.
    ground_grid = build_macro_ground_grid(
        shp_features_for_semantic,
        macro_cell_size=grid_config.macro_grid_size,
        roi_bbox=roi,
    )
    ground_z_grid_serialized = {f"{cx},{cy}": z for (cx, cy), z in ground_grid.items()}
    print(f"[GROUND-GRID] {len(ground_z_grid_serialized):,} macro-cells filled")

    scene_all = {
        "mode": "meta_reference",
        "grid_config": {
            "macro_grid_size": grid_config.macro_grid_size,
            "voxel_size": grid_config.voxel_size,
            "ground_cell_size": grid_config.ground_cell_size,
            "origin": list(grid_config.origin),
            "default_scene_height": grid_config.default_scene_height,
            "allow_float_size": grid_config.allow_float_size,
        },
        "global_bounds": {"minx": roi[0], "miny": roi[1], "maxx": roi[2], "maxy": roi[3]},
        "roi": {
            "mode": "poly" if ROI_POLY is not None else "bbox",
            "bbox": list(roi),
            # oriented crop: store the 4 corners so the viewer can draw the rotated box
            **(
                {"poly": [[float(x), float(y)]
                          for (x, y) in list(ROI_POLY.exterior.coords)[:-1]]}
                if ROI_POLY is not None else {}
            ),
        },
        "counts": {
            "overlays": len(scene.overlays),
            "voxel_objects": len(scene.voxel_objects),
            "chunks": len(chunk_entries),
        },
        "ground_z_grid": ground_z_grid_serialized,
        "chunks": chunk_entries,
    }
    save_json(scene_all, str(scene_all_dir / "scene_all.json"))
    print(f"[SAVE] scene_all.json")

    # Record the total bytes (and count) of the source LAS in the output directory so the viewer's
    # /pipeline/sizes can compute the compression ratio for any region (no glob in JS). Source
    # resolution is done by region_registry, so only the resolved LAS_SOURCES are summed here.
    try:
        src_bytes = sum(p.stat().st_size for p in LAS_SOURCES if p.exists())
        save_json(
            {"scene_id": SCENE_ID, "source_bytes": src_bytes,
             "las_count": len(LAS_SOURCES)},
            str(OUTPUT_ROOT / "source_meta.json"),
        )
        print(f"[SAVE] source_meta.json (source_bytes={src_bytes:,})")
    except Exception as e:  # noqa: BLE001 — meta is auxiliary; do not fail the build
        print(f"[WARN] could not write source_meta.json: {e}")

    print(f"\n=== BUILD DONE [{SCENE_ID}] ===")
    print(f"overlays      : {len(scene.overlays)}")
    print(f"voxel_objects : {len(scene.voxel_objects)}")
    print(f"chunks        : {len(chunk_entries)}")


if __name__ == "__main__":
    main()
