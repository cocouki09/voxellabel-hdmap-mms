"""Semantic labeling of occupancy cells by nearest SHP feature.

For each occupied voxel cell (world coords), we search per-category SHP
indices for the nearest feature and assign a label if within a
category-specific radius. Cells with no match fall back to "unknown".

SHP features come in 3 flavors:
  - Point (C1 traffic light, B1 sign, C6 post): nearest point distance.
  - LineString (C3 guardrail, C5 height barrier, A2 link): nearest point
    on any segment.
  - Polygon (A3 driveway, C4 speedbump, B3 crosswalk): point-in-polygon
    or nearest edge.

For efficiency we keep all categories as point-sample sets (sampling
lines/polygons at a fine step) and query one unified scipy cKDTree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SemanticSource:
    """A bag of XY samples representing one SHP category."""
    category: str
    samples_xy: np.ndarray  # (K, 2) float64
    match_radius_m: float
    # Optional elevation window (above local ground) where this category is
    # plausible. If None, the category matches at any Z. Example:
    # guardrail = (0.0, 1.5) means a cell whose elevation is between 0 and
    # 1.5 m above the local ground can be labeled guardrail; a tree
    # branch at elevation 5 m won't qualify.
    z_range_m: tuple[float, float] | None = None


GROUND_SURFACE_CATEGORIES = frozenset(
    {"lane", "crosswalk", "link", "node", "driveway", "subsidiary"}
)


def build_ground_z_index(shp_features: Iterable[dict]):
    """Collects 3D vertices from ground-category SHP features and builds
    an XY cKDTree for nearest-ground-Z lookups.

    The returned index lets us estimate "local ground height" at any
    query (x, y) by snapping to the nearest ground overlay vertex's Z.
    This is how we derive per-cell elevation for Z-aware labeling on
    roads whose DEM height varies over the ROI (highways, ramps, etc.).

    Returns: (cKDTree, z_array) or (None, None) if no ground samples.
    """
    samples_xy: list[tuple[float, float]] = []
    samples_z: list[float] = []

    def visit(coord_list):
        if not isinstance(coord_list, (list, tuple)):
            return
        if len(coord_list) > 0 and isinstance(coord_list[0], (int, float)):
            if len(coord_list) >= 3 and np.isfinite(coord_list[2]):
                samples_xy.append((float(coord_list[0]), float(coord_list[1])))
                samples_z.append(float(coord_list[2]))
            return
        for child in coord_list:
            visit(child)

    for f in shp_features:
        if f.get("category") not in GROUND_SURFACE_CATEGORIES:
            continue
        geom = f.get("geometry")
        if geom is None or geom.is_empty:
            continue
        # Use shapely's coord iteration via __geo_interface__
        gi = geom.__geo_interface__
        visit(gi.get("coordinates"))

    if not samples_xy:
        return None, None
    return cKDTree(np.asarray(samples_xy, dtype=np.float64)), np.asarray(
        samples_z, dtype=np.float64
    )


def query_ground_z(
    tree,
    z_array: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    """Returns (N,) array of nearest-ground Z values for each query XY."""
    if tree is None or z_array is None or len(query_xy) == 0:
        return np.full(len(query_xy), np.nan)
    _, idx = tree.query(query_xy, k=1)
    return z_array[idx]


def build_macro_ground_grid(
    shp_features: Iterable[dict],
    *,
    macro_cell_size: float,
    roi_bbox: tuple[float, float, float, float] | None,
    sample_step: float = 1.0,
    k_neighbors: int = 4,
) -> dict[tuple[int, int], float]:
    """Compute one ground-Z value per macro-cell across the ROI.

    For each macro-cell (cx, cy) the result is the mean of the K nearest
    SHP-ground vertex Z values at the cell center. Vertices come from a
    *dense* sampling of every ground-category SHP geometry (LineStrings
    sampled at ~1m, Polygon rings sampled at ~1m, Points kept as-is)
    so a sparsely-vertexed centerline still gives every nearby cell a
    sensible Z.

    K-mean (k=4 by default) smooths over single-vertex outliers — a
    rogue Z spike on one centerline point can't tilt the cell on its
    own. The grid is built for every cell in the ROI so the runtime
    has zero fallback branches; cells fully off the SHP corridor pull
    the nearest in-corridor vertices and end up at their road's Z.

    Args:
        shp_features: list of {'category', 'geometry'} dicts.
        macro_cell_size: world meters per macro-cell side.
        roi_bbox: (minx, miny, maxx, maxy) covering every cell to seed.
            None means union the SHP geometries' bounds.
        sample_step: dense sampling step (m) along lines and polygon
            edges. 1m is dense enough for cell-level smoothing.
        k_neighbors: number of nearest vertices to mean per cell.

    Returns:
        Dict from (cx, cy) macro-grid index → ground Z (world meters).
        Empty if no ground vertices were found at all.
    """
    samples_xy: list[tuple[float, float]] = []
    samples_z: list[float] = []

    def _emit(x: float, y: float, z: float) -> None:
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return
        samples_xy.append((float(x), float(y)))
        samples_z.append(float(z))

    def _z_for_xy(query_xy: tuple[float, float], rings_xyz: list[np.ndarray]) -> float | None:
        # Linear interp of nearby ring vertices' Z by inverse distance.
        # Used only inside _walk_polygon to back-fill a sampled boundary
        # point's Z; LineString sampling carries Z directly via the
        # parametric blend below.
        if not rings_xyz:
            return None
        all_pts = np.vstack(rings_xyz)
        d = np.hypot(
            all_pts[:, 0] - query_xy[0], all_pts[:, 1] - query_xy[1]
        )
        if d.size == 0:
            return None
        order = np.argsort(d)[:4]
        w = 1.0 / np.maximum(d[order], 1e-3)
        return float(np.sum(all_pts[order, 2] * w) / np.sum(w))

    def _walk_linestring(coords: np.ndarray) -> None:
        # coords: (N, >=3). Sample at sample_step, blending Z along
        # each segment so we don't lose the gradient information that
        # the centerline carries between sparse vertices.
        if len(coords) < 2:
            for c in coords:
                if c.shape[0] >= 3:
                    _emit(c[0], c[1], c[2])
            return
        for i in range(len(coords) - 1):
            a = coords[i]
            b = coords[i + 1]
            if a.shape[0] < 3 or b.shape[0] < 3:
                continue
            seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if seg_len <= 0:
                _emit(a[0], a[1], a[2])
                continue
            n = max(2, int(np.ceil(seg_len / sample_step)) + 1)
            ts = np.linspace(0.0, 1.0, n)
            for t in ts:
                _emit(
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                )

    def _walk_polygon(rings: list[np.ndarray]) -> None:
        # Each ring: (N, >=3). Sample boundary the same way as a line;
        # interior fill is not needed — cells inside the polygon will
        # find boundary samples within k=4 anyway.
        for ring in rings:
            if ring.shape[1] >= 3:
                _walk_linestring(ring)

    for f in shp_features:
        if f.get("category") not in GROUND_SURFACE_CATEGORIES:
            continue
        geom = f.get("geometry")
        if geom is None or geom.is_empty:
            continue
        gtype = geom.geom_type
        gi = geom.__geo_interface__
        coords = gi.get("coordinates")
        if gtype == "Point":
            if coords and len(coords) >= 3:
                _emit(coords[0], coords[1], coords[2])
        elif gtype == "MultiPoint":
            for p in coords or []:
                if len(p) >= 3:
                    _emit(p[0], p[1], p[2])
        elif gtype == "LineString":
            arr = np.asarray(coords, dtype=np.float64)
            if arr.ndim == 2:
                _walk_linestring(arr)
        elif gtype == "MultiLineString":
            for line in coords or []:
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim == 2:
                    _walk_linestring(arr)
        elif gtype == "Polygon":
            rings = [np.asarray(r, dtype=np.float64) for r in (coords or [])]
            rings = [r for r in rings if r.ndim == 2]
            _walk_polygon(rings)
        elif gtype == "MultiPolygon":
            for poly in coords or []:
                rings = [np.asarray(r, dtype=np.float64) for r in poly]
                rings = [r for r in rings if r.ndim == 2]
                _walk_polygon(rings)

    if not samples_xy:
        return {}

    pts_xy = np.asarray(samples_xy, dtype=np.float64)
    z_arr = np.asarray(samples_z, dtype=np.float64)
    tree = cKDTree(pts_xy)

    # Determine the cell range we need to fill.
    if roi_bbox is None:
        minx = float(pts_xy[:, 0].min())
        miny = float(pts_xy[:, 1].min())
        maxx = float(pts_xy[:, 0].max())
        maxy = float(pts_xy[:, 1].max())
    else:
        minx, miny, maxx, maxy = roi_bbox
    cx_min = int(np.floor(minx / macro_cell_size))
    cy_min = int(np.floor(miny / macro_cell_size))
    cx_max = int(np.floor(maxx / macro_cell_size))
    cy_max = int(np.floor(maxy / macro_cell_size))

    # Build the centers grid in one batch query for speed.
    centers: list[tuple[int, int]] = []
    centers_xy: list[tuple[float, float]] = []
    for cy in range(cy_min, cy_max + 1):
        for cx in range(cx_min, cx_max + 1):
            centers.append((cx, cy))
            centers_xy.append((
                (cx + 0.5) * macro_cell_size,
                (cy + 0.5) * macro_cell_size,
            ))
    if not centers:
        return {}

    k = min(k_neighbors, len(z_arr))
    _, idx = tree.query(np.asarray(centers_xy), k=k)
    if k == 1:
        means = z_arr[idx]
    else:
        means = z_arr[idx].mean(axis=1)

    return {centers[i]: float(means[i]) for i in range(len(centers))}


def sample_linestring(coords: np.ndarray, step: float) -> np.ndarray:
    """Evenly sample a polyline at `step` meters. coords: (N, 2+) array."""
    if len(coords) < 2:
        return coords[:, :2].astype(np.float64)
    segs = np.diff(coords[:, :2], axis=0)
    lengths = np.linalg.norm(segs, axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return coords[:1, :2].astype(np.float64)
    n = max(2, int(np.ceil(total / step)) + 1)
    ts = np.linspace(0.0, total, n)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(ts):
        seg_idx = int(np.searchsorted(cumulative, t, side="right") - 1)
        seg_idx = min(seg_idx, len(lengths) - 1)
        seg_len = lengths[seg_idx]
        if seg_len <= 0:
            out[i] = coords[seg_idx, :2]
        else:
            local_t = (t - cumulative[seg_idx]) / seg_len
            out[i] = coords[seg_idx, :2] + local_t * segs[seg_idx]
    return out


def build_sources_from_shp(
    shp_features: Iterable[dict],
    dimensions: dict,
    linestring_sample_step: float = 0.25,
    polygon_sample_step: float = 0.5,
) -> list[SemanticSource]:
    """Convert SHP features into per-category point sample sets.

    shp_features: list of {'category': str, 'geometry': shapely geometry}
    dimensions: loaded object_dimensions.json (for match_radius_m per cat).
    """
    by_cat: dict[str, list[np.ndarray]] = {}
    for f in shp_features:
        cat = f["category"]
        geom = f["geometry"]
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt == "Point":
            pts = np.array([[geom.x, geom.y]], dtype=np.float64)
        elif gt == "MultiPoint":
            pts = np.array([[p.x, p.y] for p in geom.geoms], dtype=np.float64)
        elif gt == "LineString":
            coords = np.array(list(geom.coords), dtype=np.float64)
            pts = sample_linestring(coords, linestring_sample_step)
        elif gt == "MultiLineString":
            all_pts = [
                sample_linestring(np.array(list(g.coords), dtype=np.float64), linestring_sample_step)
                for g in geom.geoms
            ]
            pts = np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 2))
        elif gt == "Polygon":
            # sample the exterior + interiors perimeter
            coords = np.array(list(geom.exterior.coords), dtype=np.float64)
            pts = sample_linestring(coords, polygon_sample_step)
        elif gt == "MultiPolygon":
            all_pts = [
                sample_linestring(np.array(list(g.exterior.coords), dtype=np.float64), polygon_sample_step)
                for g in geom.geoms
            ]
            pts = np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 2))
        else:
            continue
        by_cat.setdefault(cat, []).append(pts)

    out: list[SemanticSource] = []
    for cat, chunks in by_cat.items():
        samples = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2))
        if len(samples) == 0:
            continue
        dims = dimensions.get(cat, {})
        radius = float(dims.get("semantic_match_radius_m", 1.0))
        z_range_raw = dims.get("z_range_m")
        z_range: tuple[float, float] | None = None
        if isinstance(z_range_raw, (list, tuple)) and len(z_range_raw) == 2:
            z_range = (float(z_range_raw[0]), float(z_range_raw[1]))
        out.append(
            SemanticSource(
                category=cat,
                samples_xy=samples,
                match_radius_m=radius,
                z_range_m=z_range,
            )
        )
    return out


def label_cells(
    cell_centers_xy: np.ndarray,
    sources: list[SemanticSource],
    default_label: str = "unknown",
    elevations: np.ndarray | None = None,
) -> np.ndarray:
    """Assigns a category to each cell by nearest source within its radius
    AND (optionally) within its elevation range.

    For every cell we:
      1. Optionally gate by elevation — a cell whose elevation is outside
         source.z_range_m cannot be labeled with that category.
      2. Compute XY distance to every eligible source.
      3. Pick the category whose sample set has the smallest distance and
         is within that category's match radius.

    Why Z-aware matching matters:
      - A tree branch at elevation 4 m sitting above a guardrail line
        was previously labeled 'guardrail' (XY match, no Z check). Now
        it falls to 'unknown' because 4 m is outside guardrail's
        [0, 1.5] Z range.
      - Ground paint on a lane was labeled 'sign' if a sign pole was
        within radius. Z range [0.8, 3.5] for sign excludes ground
        points so they fall to 'lane' (ground category) instead.

    elevations can be None — then behavior matches pure XY matching.

    Returns: (N,) array of labels.
    """
    n = len(cell_centers_xy)
    if n == 0:
        return np.array([], dtype=object)

    best_dist = np.full(n, np.inf)
    best_cat = np.full(n, default_label, dtype=object)

    for src in sources:
        if len(src.samples_xy) == 0:
            continue

        # Elevation gate — only cells whose elevation falls in the
        # category's Z range are eligible to be matched here.
        if elevations is not None and src.z_range_m is not None:
            z_lo, z_hi = src.z_range_m
            eligible = (elevations >= z_lo) & (elevations <= z_hi)
            if not eligible.any():
                continue
            eligible_idx = np.where(eligible)[0]
            eligible_xy = cell_centers_xy[eligible]
        else:
            eligible_idx = np.arange(n)
            eligible_xy = cell_centers_xy

        tree = cKDTree(src.samples_xy)
        dists, _ = tree.query(
            eligible_xy, k=1, distance_upper_bound=src.match_radius_m
        )
        hit_local = (dists <= src.match_radius_m) & (
            dists < best_dist[eligible_idx]
        )
        hit_global = eligible_idx[hit_local]
        best_dist[hit_global] = dists[hit_local]
        best_cat[hit_global] = src.category

    return best_cat
