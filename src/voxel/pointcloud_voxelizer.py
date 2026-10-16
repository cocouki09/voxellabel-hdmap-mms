"""Point cloud → occupancy voxelization.

Each LIDAR point gets bucketed into a voxel cell of a fixed side (voxel_size).
Any cell that receives at least `min_points_per_cell` points is marked
occupied. Returns a (M, 3) int32 array of unique (gx, gy, gz) indices.

This matches the 'voxelization occupancy principle' in project memory:
each voxel cell with any presence becomes 1 cell occupied.
"""
from __future__ import annotations

import numpy as np


def voxelize_points(
    points_xyz: np.ndarray,
    voxel_size: float,
    origin: tuple[float, float, float],
    min_points_per_cell: int = 1,
) -> np.ndarray:
    """Bucket points into occupancy cells.

    Args:
        points_xyz: (N, 3) world coordinates.
        voxel_size: cell side length in world units (meters).
        origin: world (ox, oy, oz) — grid index 0 corresponds to this corner.
        min_points_per_cell: reject cells with fewer than this many points
            (1 = keep any cell that saw a point; higher = noise suppression).

    Returns:
        (M, 3) int32 array of unique (gx, gy, gz) cell indices.
    """
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.int32)

    ox, oy, oz = origin
    gx = np.floor((points_xyz[:, 0] - ox) / voxel_size).astype(np.int64)
    gy = np.floor((points_xyz[:, 1] - oy) / voxel_size).astype(np.int64)
    gz = np.floor((points_xyz[:, 2] - oz) / voxel_size).astype(np.int64)

    if min_points_per_cell <= 1:
        # Fast path: unique rows only
        keys = np.stack([gx, gy, gz], axis=1)
        unique = np.unique(keys, axis=0)
        return unique.astype(np.int32)

    # Count-based path: dense dict keyed by a 64-bit hash of (gx, gy, gz).
    # Uses pandas-free approach via numpy unique with return_counts.
    keys = np.stack([gx, gy, gz], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    keep = counts >= min_points_per_cell
    return unique[keep].astype(np.int32)


def voxelize_points_with_representative(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    voxel_size: float,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize and also report, per unique cell, the ACTUAL LIDAR point
    that landed closest to the geometric cell center plus its RGB color.

    Rationale: drawing the occupancy marker at the real LIDAR point
    preserves the point-cloud look while still deduplicating per cell.
    Using that point's RGB makes the scene visually resemble the raw
    point cloud rather than a flat category-color blob.

    Returns:
        cells:       (M, 3) int32 voxel indices
        positions:   (M, 3) float64 world coords of the representative LIDAR point
        colors:      (M, 3) uint8 [0..255]
    """
    import pandas as pd

    if len(points_xyz) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )

    ox, oy, oz = origin
    gx = np.floor((points_xyz[:, 0] - ox) / voxel_size).astype(np.int64)
    gy = np.floor((points_xyz[:, 1] - oy) / voxel_size).astype(np.int64)
    gz = np.floor((points_xyz[:, 2] - oz) / voxel_size).astype(np.int64)

    cx = ox + (gx + 0.5) * voxel_size
    cy = oy + (gy + 0.5) * voxel_size
    cz = oz + (gz + 0.5) * voxel_size
    dist_sq = (
        (points_xyz[:, 0] - cx) ** 2
        + (points_xyz[:, 1] - cy) ** 2
        + (points_xyz[:, 2] - cz) ** 2
    )

    df = pd.DataFrame({
        "gx": gx, "gy": gy, "gz": gz,
        "d": dist_sq,
        "x": points_xyz[:, 0],
        "y": points_xyz[:, 1],
        "z": points_xyz[:, 2],
        "r": points_rgb[:, 0],
        "g": points_rgb[:, 1],
        "b": points_rgb[:, 2],
    })
    # Position + colour from the same closest-to-cell-centre point.
    # Two alternatives were tried and rejected:
    #   - mean-RGB per cell (averaged across all ~62 points): killed
    #     saturation (0.38 -> 0.16), grey wash.
    #   - most-saturated-pick per cell (argmax of max-min channel):
    #     picked up noise / outlier vivid points in dim regions like
    #     tunnels, causing a strong green bias in pixels that should
    #     have been concrete grey.
    # Closest-to-centre is the boring-but-correct choice; rgb_enhance
    # handles brightness/saturation lifting downstream.
    best_idx = df.groupby(["gx", "gy", "gz"], sort=False)["d"].idxmin()
    best = df.loc[best_idx]
    cells = best[["gx", "gy", "gz"]].to_numpy().astype(np.int32)
    positions = best[["x", "y", "z"]].to_numpy()
    colors = best[["r", "g", "b"]].to_numpy().astype(np.uint8)
    return cells, positions, colors


def group_cells_by_semantic(
    cells: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Buckets cells by their semantic label string.

    Args:
        cells: (M, 3) int32 voxel indices.
        labels: (M,) array of category strings (e.g. "traffic_light",
            "guardrail", "unknown").

    Returns:
        {category: (K, 3) int32 cells} per unique label.
    """
    out: dict[str, np.ndarray] = {}
    for cat in np.unique(labels):
        mask = labels == cat
        out[str(cat)] = cells[mask]
    return out
