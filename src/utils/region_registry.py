"""Region registry — a work area (region) is defined by config/regions/<id>.json.

New region = one JSON file, no code. Only what differs per region (SHP/LAS paths, ROI,
output, CRS, stride) is in the config; all tuning parameters are shared.

config/regions/<id>.json schema:
  {
    "id": "deoksong",
    "crs": "EPSG:32652",                 # recorded only (no reprojection)
    "output_dir": "data/output_deoksong",# relative to the project root, or absolute
    "roi": [minx, miny, maxx, maxy] | "auto",   # auto = from the SHP/LAS bounds
    "lidar_stride": 4,
    "sources": [                         # tried in order; the first source with LAS files is used
      { "name": "raw",    "hdmap_dir": "...", "lidar_globs": ["...*.las", ...] },
      { "name": "sample", "hdmap_dir": "...", "lidar_globs": ["...*.laz", ...] }
    ]
  }
"""
from __future__ import annotations

import glob as globmod
import json
from dataclasses import dataclass
from pathlib import Path

ROI = tuple[float, float, float, float]


@dataclass
class ResolvedRegion:
    """Result of load_region(); the build script uses only these values."""
    scene_id: str
    hdmap_dir: Path
    las_sources: list[Path]
    roi: ROI | None  # None = auto (from the SHP/LAS bounds)
    output_root: Path
    lidar_stride: int
    crs: str | None
    hdmap_spec: str  # HD map spec id to use (config/hdmap_specs/<id>.json)
    source_name: str  # name of the source used (raw / sample / external ...)
    voxel_size: float | None  # None = shared grid_config. A region can enlarge it
    # to reduce the cell count of a large LiDAR-only ROI (e.g. 1.0 m). Unset for Deoksong -> 0.5 m.
    clip_overlays_to_roi: bool  # True: clip SHP geometries to the ROI box
    # (Z kept), so roads do not extend outside the box in SHP-only regions without
    # LiDAR. Unset for LiDAR regions such as Deoksong, where the viewer's coverage clip applies.
    roi_poly: list[tuple[float, float]] | None  # rotated ROI polygon (4 corners, UTM).
    # If set, clip to this polygon instead of the axis-aligned bbox (oriented crop); roi (bbox)
    # is derived from its bounds (for the origin and camera).


def _resolve_path(p: str, root: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def list_regions(project_root: Path) -> list[str]:
    d = project_root / "config" / "regions"
    return sorted(f.stem for f in d.glob("*.json")) if d.is_dir() else []


def load_region(region_id: str, project_root: Path) -> ResolvedRegion:
    """Read the config of region_id and resolve its sources. Fallbacks such as raw -> sample
    are expressed by the order of sources; the first source with LAS files is used."""
    cfg_path = project_root / "config" / "regions" / f"{region_id}.json"
    if not cfg_path.exists():
        avail = ", ".join(list_regions(project_root)) or "(none)"
        raise FileNotFoundError(
            f"region config not found: {cfg_path}\n  available regions: {avail}"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    roi_raw = cfg.get("roi")
    roi: ROI | None = (
        None if roi_raw in (None, "auto")
        else tuple(float(x) for x in roi_raw)  # type: ignore[assignment]
    )
    output_root = _resolve_path(cfg["output_dir"], project_root)
    stride = int(cfg.get("lidar_stride", 1))
    crs = cfg.get("crs")
    hdmap_spec = cfg.get("hdmap_spec", "korean_hdmap_2020")
    voxel_size = cfg.get("voxel_size")
    voxel_size = float(voxel_size) if voxel_size is not None else None
    clip_overlays_to_roi = bool(cfg.get("clip_overlays_to_roi", False))
    roi_poly_raw = cfg.get("roi_poly")
    roi_poly = (
        [(float(p[0]), float(p[1])) for p in roi_poly_raw]
        if roi_poly_raw else None
    )
    # With a polygon, derive roi (bbox) from its bounds (for the origin, camera and existing code).
    if roi_poly is not None and roi is None:
        xs = [p[0] for p in roi_poly]
        ys = [p[1] for p in roi_poly]
        roi = (min(xs), min(ys), max(xs), max(ys))

    for src in cfg.get("sources", []):
        las: list[Path] = []
        for g in src.get("lidar_globs", []):
            gp = _resolve_path(g, project_root)
            las += sorted(Path(p) for p in globmod.glob(str(gp)))
        if las:
            hdmap = _resolve_path(src["hdmap_dir"], project_root)
            print(
                f"[REGION] {region_id}: source '{src.get('name', '?')}' "
                f"({len(las)} LAS)  hdmap={hdmap}"
            )
            return ResolvedRegion(
                scene_id=region_id, hdmap_dir=hdmap, las_sources=las,
                roi=roi, output_root=output_root, lidar_stride=stride,
                crs=crs, hdmap_spec=hdmap_spec, source_name=src.get("name", "?"),
                voxel_size=voxel_size,
                clip_overlays_to_roi=clip_overlays_to_roi,
                roi_poly=roi_poly,
            )

    # SHP-only fallback: if no source has LAS files but hdmap_dir exists, build overlays only
    # (a region whose LAS comes later). run_build_scene_lidar handles 0 points
    # gracefully and builds the SHP overlays with their real Z.
    for src in cfg.get("sources", []):
        hd = src.get("hdmap_dir")
        if not hd:
            continue
        hdmap = _resolve_path(hd, project_root)
        if hdmap.exists():
            print(
                f"[REGION] {region_id}: SHP-only (no LAS) — building overlays/Z only  "
                f"hdmap={hdmap}"
            )
            return ResolvedRegion(
                scene_id=region_id, hdmap_dir=hdmap, las_sources=[],
                roi=roi, output_root=output_root, lidar_stride=stride,
                crs=crs, hdmap_spec=hdmap_spec,
                source_name=src.get("name", "shp-only"),
                voxel_size=voxel_size,
                clip_overlays_to_roi=clip_overlays_to_roi,
                roi_poly=roi_poly,
            )

    raise RuntimeError(
        f"region '{region_id}': no LAS files and no valid hdmap_dir. "
        f"config/regions/{region_id}.json  — check the paths (e.g. external drive connected)."
    )
