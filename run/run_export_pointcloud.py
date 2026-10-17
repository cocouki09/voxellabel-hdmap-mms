"""Export the ROI-clipped + strided raw LIDAR point cloud as a flat
binary the web viewer can stream and render directly with Three.js
Points — independent of the voxelization step.

Why this exists:
  The main pipeline (run_build_scene_lidar.py) reads LAS in memory,
  voxelises, then drops the raw points. The viewer's "voxel centre"
  layer is the *voxel centroid* (1 point per occupied voxel), not the
  full LIDAR cloud. This script runs the same LAS_SOURCES / ROI_BBOX /
  LIDAR_STRIDE the main pipeline uses but serialises the raw points
  instead of voxelising — so the viewer can offer an "original LIDAR" layer
  alongside voxel/center for size-vs-fidelity comparison.

Output format (little-endian, all offsets in bytes):

    header (36 B)
      0   magic       'PCRG' (4 bytes ASCII, no terminator)
      4   version     uint32                     (=1)
      8   point_count uint32                     N
      12  origin_x    float64                    UTM origin (subtracted
      20  origin_y    float64                    from each point so the
      28  origin_z    float64                    relative coords fit
                                                  in float32 precision)

    xyz block (12 * N B)
      per point: float32 rel_x, rel_y, rel_z

    rgb block (3 * N B)
      per point: uint8 r, g, b

Origin is the scene's grid origin (the same one used by GridConfig)
so the viewer can subtract its frame.sceneOriginWorld and feed the
result straight into a BufferAttribute without re-projecting.

Decoder side: web/src/rendering/RawLidarLayer.tsx.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# The work area (LAS / ROI / stride / output) is resolved through the region registry,
# reading the same config as the build script so that sources, ROI and stride match.
# (The build module's top-level globals are None until its main() runs, so --region is
#  resolved directly here.)
from src.readers.las_reader import read_las_xyz_rgb, read_las_header


MAGIC = b"PCRG"
VERSION = 1
HEADER_BYTES = 36


def write_pcrg_bin(
    out_path: Path,
    rel_xyz_f32: np.ndarray,
    rgb_u8: np.ndarray,
    origin: tuple[float, float, float],
) -> int:
    """Serialise a point cloud to the PCRG binary format (see module docstring).

    Args:
        out_path: destination .bin path (parent dirs created).
        rel_xyz_f32: (N, 3) float32 coords RELATIVE to `origin` (caller already
            subtracted the origin so the values fit in float32 precision).
        rgb_u8: (N, 3) uint8 colors.
        origin: (ox, oy, oz) world UTM origin baked into the header so the
            viewer can re-add it / subtract its own scene origin.

    Returns the point count written. Decoder: web/src/core/pcrg.ts
    (and web/src/rendering/RawLidarLayer.tsx).
    """
    n = int(len(rel_xyz_f32))
    rel = np.ascontiguousarray(rel_xyz_f32, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, n))
        f.write(struct.pack("<ddd", float(origin[0]), float(origin[1]), float(origin[2])))
        f.write(rel.tobytes(order="C"))
        f.write(rgb.tobytes(order="C"))
    return n


VOXEL_MAGIC = b"VXBN"
VOXEL_VERSION = 2
VOXEL_HEADER_BYTES = 40  # magic(4) ver(4) count(4) origin(24) voxel_size(4)


def write_voxel_bin(
    out_path: Path,
    centers_rel_f32: np.ndarray,
    rgb_u8: np.ndarray,
    cat_u8: np.ndarray,
    origin: tuple[float, float, float],
    voxel_size: float,
    obj_u16: np.ndarray,
) -> int:
    """Serialise voxelized cells to a compact binary (PCRG sibling) so the web
    can render a full-island 0.5m voxelization at once (~18 B/cell) instead of a
    multi-GB JSON. Same voxel LOGIC as the JSON path — only the serialization
    differs. Layout (little-endian), v2:

        header: 'VXBN'(4) · version u32 · count u32 · origin f64×3 · voxel_size f32
        body:   centers xyz f32×3×N (RELATIVE to origin, = grid cell centers)
                rgb  u8×3×N  (representative LiDAR color)
                cat  u8×N    (category id; legend lives in scene_preview.json)
                obj  u16×N   (VoxelObject index → object table in JSON meta;
                              added in v2 — object identity for click picking / selection)

    Decoder: web/src/core/voxelBin.ts. Category legend + object table live in
    the light scene_preview.json under the `voxel_bin` key (caller writes it).
    """
    n = int(len(centers_rel_f32))
    rel = np.ascontiguousarray(centers_rel_f32, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
    cat = np.ascontiguousarray(cat_u8, dtype=np.uint8)
    obj = np.ascontiguousarray(obj_u16, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(VOXEL_MAGIC)
        f.write(struct.pack("<II", VOXEL_VERSION, n))
        f.write(struct.pack("<ddd", float(origin[0]), float(origin[1]), float(origin[2])))
        f.write(struct.pack("<f", float(voxel_size)))
        f.write(rel.tobytes(order="C"))
        f.write(rgb.tobytes(order="C"))
        f.write(cat.tobytes(order="C"))
        f.write(obj.tobytes(order="C"))
    return n


def export_pointcloud(
    out_path: Path,
    las_sources: list,
    roi_bbox: tuple | None,
    stride: int,
) -> None:
    if not las_sources:
        raise RuntimeError("no LAS files resolved; check region config")

    print(f"[pcloud] reading {len(las_sources)} LAS file(s) (stride={stride}, roi={roi_bbox})")
    pts, rgb = read_las_xyz_rgb(las_sources, roi_bbox=roi_bbox, stride=stride)
    n = len(pts)
    if n == 0:
        raise RuntimeError("LAS read returned 0 points; nothing to export")
    print(f"[pcloud] {n:,} points after ROI + stride")

    # Use the ROI bbox lower corner + LAS Z minimum as the origin — same
    # convention as the main pipeline's GridConfig. Keeping the origin
    # implicit-from-data lets this script run independently without
    # parsing scene_preview.json.
    if roi_bbox is not None:
        ox, oy = float(roi_bbox[0]), float(roi_bbox[1])
    else:
        ox, oy = float(pts[:, 0].min()), float(pts[:, 1].min())
    oz = float(pts[:, 2].min())
    print(f"[pcloud] origin = ({ox:.3f}, {oy:.3f}, {oz:.3f})")

    rel = np.empty_like(pts, dtype=np.float32)
    rel[:, 0] = (pts[:, 0] - ox).astype(np.float32)
    rel[:, 1] = (pts[:, 1] - oy).astype(np.float32)
    rel[:, 2] = (pts[:, 2] - oz).astype(np.float32)

    rgb_u8 = rgb.astype(np.uint8, copy=False)

    write_pcrg_bin(out_path, rel, rgb_u8, (ox, oy, oz))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[pcloud] wrote {out_path}  ({n:,} points, {size_mb:.1f} MB)")


def main() -> None:
    import argparse

    from run.run_build_scene_lidar import PROJECT_DATA_ROOT, CENTER_ROI_SIZE_M
    from src.utils.region_registry import load_region, list_regions

    ap = argparse.ArgumentParser(
        description="Export the ROI-clipped, strided original LIDAR point cloud as a viewer binary."
    )
    ap.add_argument(
        "--region", required=True, choices=list_regions(PROJECT_DATA_ROOT),
        help="work area defined in config/regions/<id>.json",
    )
    ap.add_argument(
        "--stride", type=int, default=None,
        help="stride to use instead of region.lidar_stride, e.g. to export a lighter raw binary "
             "for a large ROI independently of the (denser) voxel build.",
    )
    args = ap.parse_args()

    region = load_region(args.region, PROJECT_DATA_ROOT)
    stride = args.stride if args.stride else region.lidar_stride
    roi = region.roi
    if roi is None:
        # Same as the build script's auto-ROI: centre of the first LAS header + CENTER_ROI_SIZE_M.
        hdr = read_las_header(region.las_sources[0])
        cx = (hdr["mins"][0] + hdr["maxs"][0]) / 2
        cy = (hdr["mins"][1] + hdr["maxs"][1]) / 2
        half = CENTER_ROI_SIZE_M / 2
        roi = (cx - half, cy - half, cx + half, cy + half)

    out = region.output_root / "raw_lidar.bin"
    print(f"[pcloud] region={region.scene_id} source={region.source_name} out={out} stride={stride}")
    export_pointcloud(out, region.las_sources, roi, stride)


if __name__ == "__main__":
    main()
