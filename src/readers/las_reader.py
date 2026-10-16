"""LAS / LAZ reader for MMS LIDAR point clouds.

Responsibilities:
- Stream a single LAS file (or list of files) into a flat Nx3 numpy array of
  (x, y, z) world coordinates in the file's CRS.
- Optionally clip by ROI bbox (minx, miny, maxx, maxy) during load to keep
  memory bounded — a single 600MB LAS holds ~20M points, so ROI clipping is
  essential for anything larger than a small intersection.
- Optionally downsample by keeping every Nth point (deterministic stride).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import laspy


def _tile_outside_roi(
    header, roi_bbox: tuple[float, float, float, float] | None
) -> bool:
    """True if the XY bbox of the LAS file misses roi_bbox; the file is then skipped unread.

    Only the header (mins/maxs) is read, so loading the body (tens to hundreds of MB) is
    avoided. Such tiles would be dropped entirely by the ROI mask, so the result is identical.
    This greatly reduces I/O when a small ROI is built from a large survey (several GB).
    """
    if roi_bbox is None:
        return False
    minx, miny, maxx, maxy = roi_bbox
    tminx, tminy = float(header.mins[0]), float(header.mins[1])
    tmaxx, tmaxy = float(header.maxs[0]), float(header.maxs[1])
    return tmaxx < minx or tminx > maxx or tmaxy < miny or tminy > maxy


def read_las_xyz(
    paths: Iterable[Path | str],
    roi_bbox: tuple[float, float, float, float] | None = None,
    stride: int = 1,
    z_min: float | None = None,
    z_max: float | None = None,
) -> np.ndarray:
    """Returns an (N, 3) float64 array of world (x, y, z) points.

    Args:
        paths: one or more .las/.laz paths to concatenate.
        roi_bbox: (minx, miny, maxx, maxy) in the LAS CRS; points outside are
            dropped before accumulation. None = keep all.
        stride: keep every `stride`-th point (1 = all, 10 = 10% of data).
        z_min/z_max: optional Z filter (useful for excluding points below
            road surface or above reasonable structure height).
    """
    chunks: list[np.ndarray] = []
    n_skipped = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[LAS] skip missing: {p}")
            continue
        with laspy.open(p) as f:
            if _tile_outside_roi(f.header, roi_bbox):
                n_skipped += 1
                continue
            las = f.read()
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)
        zs = np.asarray(las.z, dtype=np.float64)
        total = len(xs)
        mask = np.ones(total, dtype=bool)
        if roi_bbox is not None:
            minx, miny, maxx, maxy = roi_bbox
            mask &= (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        if z_min is not None:
            mask &= zs >= z_min
        if z_max is not None:
            mask &= zs <= z_max
        if stride > 1:
            stride_mask = np.zeros(total, dtype=bool)
            stride_mask[::stride] = True
            mask &= stride_mask
        xyz = np.column_stack((xs[mask], ys[mask], zs[mask]))
        print(f"[LAS] {p.name}: kept {len(xyz):,} / {total:,} points")
        chunks.append(xyz)
        del xs, ys, zs, mask, las
    if n_skipped:
        print(f"[LAS] skipped {n_skipped} tile(s) outside ROI (header-only, no point read)")
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def read_las_xyz_rgb(
    paths: Iterable[Path | str],
    roi_bbox: tuple[float, float, float, float] | None = None,
    stride: int = 1,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as read_las_xyz but also returns per-point RGB.

    Returns:
        points_xyz: (N, 3) float64 world coordinates.
        points_rgb: (N, 3) uint8 [0..255] colors. If a LAS file lacks RGB,
            fills with mid-gray (128).
    """
    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    n_skipped = 0

    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[LAS] skip missing: {p}")
            continue
        with laspy.open(p) as f:
            if _tile_outside_roi(f.header, roi_bbox):
                n_skipped += 1
                continue
            las = f.read()
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)
        zs = np.asarray(las.z, dtype=np.float64)
        total = len(xs)

        dim_names = set(las.point_format.dimension_names)
        if {"red", "green", "blue"}.issubset(dim_names):
            rs = np.asarray(las.red, dtype=np.uint32)
            gs = np.asarray(las.green, dtype=np.uint32)
            bs = np.asarray(las.blue, dtype=np.uint32)
            # LAS stores RGB as uint16 — either true 16-bit values
            # (0–65535) or 0–255 in the low byte. Detect by max.
            #
            # NOTE: this MMS dataset's 16-bit RGB is already sRGB-encoded
            # by the camera ISP, just stretched into uint16. A plain
            # >>8 is correct. Applying a *second* sRGB curve on top
            # compressed everything into the highlight band and washed
            # the cloud near-white — don't gamma-correct here.
            max_val = 0
            if total:
                max_val = int(max(rs.max(), gs.max(), bs.max()))
            if max_val > 255:
                rs = (rs >> 8).astype(np.uint8)
                gs = (gs >> 8).astype(np.uint8)
                bs = (bs >> 8).astype(np.uint8)
            else:
                rs = rs.astype(np.uint8)
                gs = gs.astype(np.uint8)
                bs = bs.astype(np.uint8)
        else:
            rs = np.full(total, 128, dtype=np.uint8)
            gs = rs.copy()
            bs = rs.copy()

        mask = np.ones(total, dtype=bool)
        if roi_bbox is not None:
            minx, miny, maxx, maxy = roi_bbox
            mask &= (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        if z_min is not None:
            mask &= zs >= z_min
        if z_max is not None:
            mask &= zs <= z_max
        if stride > 1:
            stride_mask = np.zeros(total, dtype=bool)
            stride_mask[::stride] = True
            mask &= stride_mask

        xyz = np.column_stack((xs[mask], ys[mask], zs[mask]))
        rgb = np.column_stack((rs[mask], gs[mask], bs[mask]))
        print(f"[LAS] {p.name}: kept {len(xyz):,} / {total:,} points (with RGB)")
        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)
        del xs, ys, zs, rs, gs, bs, las, mask

    if n_skipped:
        print(f"[LAS] skipped {n_skipped} tile(s) outside ROI (header-only, no point read)")
    if not chunks_xyz:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )
    return (
        np.concatenate(chunks_xyz, axis=0),
        np.concatenate(chunks_rgb, axis=0),
    )


def read_las_header(path: Path | str) -> dict:
    """Lightweight header inspection — no point load."""
    with laspy.open(Path(path)) as f:
        h = f.header
        return {
            "version": str(h.version),
            "point_format": h.point_format.id,
            "point_count": h.point_count,
            "mins": [float(v) for v in h.mins],
            "maxs": [float(v) for v in h.maxs],
            "scales": [float(v) for v in h.scales],
            "offsets": [float(v) for v in h.offsets],
        }
