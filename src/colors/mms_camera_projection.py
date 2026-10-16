"""
MMS multi-camera projection for voxel recolour.

Leica Pegasus Two Ultimate dumps each captured frame with full External
Orientation (camera pose: world position + 3x3 rotation matrix in UTM52N)
and Internal Orientation (calibration: focal in pixels, principal point,
distortion — already corrected to zero in this dataset). For each voxel
we want a colour that *actually* matches what the camera saw, not the
desaturated grey that Leica's "colourise LAS" step bakes in.

Algorithm:
  1. Parse every External Orientation CSV + Internal Orientation TXT
     for every Camera{1..5}/Track{N} combination under the SURV/TRACK
     directory tree.
  2. Optionally filter frames to those whose camera position falls
     inside (or near) the ROI bbox — cuts ~80% of frames.
  3. For each voxel (x, y, z) world UTM52N:
       a. Find the K closest camera frames by Euclidean distance.
       b. For each candidate, transform voxel into camera coords via
          P_cam = R^T · (P_world - C_world). Reject if Z_cam ≤ 0
          (behind camera).
       c. Project to image plane: u = fx·X_cam/Z_cam + cx, same for v.
          Reject if (u, v) outside [0, W) × [0, H).
       d. Pick the closest *valid* frame. Sample its RGB at (u, v).
  4. If no camera sees the voxel, fall back to the LIDAR-baked RGB
     that was already in rep_rgb.

Conventions used here (verified by sanity test on a known voxel):
  - R from External Orientation is camera→world (its columns are the
    camera-frame axes expressed in world coords).
  - So P_cam = R^T · (P_w - C_w).
  - Camera Y axis points DOWN in image coords (image row grows
    downward), Z axis points FORWARD.

Performance: ~2000 ROI frames × 500k voxels would be 1e9 ops if done
naively. We invert: for each frame, project ALL its voxel candidates
once, sample, and emit. One image is loaded at a time.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# CSV column layout (semicolon-separated, no header):
#  0  filename
#  1  timestamp
#  2  X (UTM East, m)
#  3  Y (UTM North, m)
#  4  Z (ellipsoidal height, m)
#  5  yaw  (omega? degrees)   ┐
#  6  pitch (phi? degrees)    │ unused — we have the full 3×3 matrix
#  7  roll  (kappa? degrees)  ┘
#  8..16  R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]
EXTERNAL_COLS = 17

_CAMERA_ID_RE = re.compile(r"Camera(\d+)_External_Orientation", re.IGNORECASE)


@dataclass
class CameraIntrinsics:
    """OpenCV-style pinhole calibration for a single Camera{N}."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def parse_internal_orientation(path: str | Path) -> CameraIntrinsics:
    """
    Parse one *_Internal_Orientation.txt into a CameraIntrinsics.

    The file is `key=value` lines, possibly with whitespace; we only
    need the Dist_CV_* keys since K1..K6/P1/P2 are all 0 in this
    dataset (image already lens-corrected).
    """
    p = Path(path)
    kv: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

    def f(key: str) -> float:
        if key not in kv:
            raise KeyError(f"missing key '{key}' in {path}")
        return float(kv[key])

    return CameraIntrinsics(
        fx=f("Dist_CV_FX"),
        fy=f("Dist_CV_FY"),
        cx=f("Dist_CV_CX"),
        cy=f("Dist_CV_CY"),
        width=int(f("Dist_CV_WIDTH")),
        height=int(f("Dist_CV_HEIGHT")),
    )


def parse_external_orientation(
    csv_path: str | Path, camera_id: int, track_label: str
) -> pd.DataFrame:
    """
    Parse one *_External_Orientation.csv. Returns a DataFrame with:
      filename, X, Y, Z, R (3x3 as numpy.float64 stored per row),
      camera_id, track_label, image_path (resolved).
    """
    p = Path(csv_path)
    df = pd.read_csv(p, sep=";", header=None, dtype=str)
    if df.shape[1] != EXTERNAL_COLS:
        raise ValueError(
            f"{p}: expected {EXTERNAL_COLS} columns, got {df.shape[1]}"
        )
    out = pd.DataFrame(
        {
            "filename": df[0],
            "X": df[2].astype(float),
            "Y": df[3].astype(float),
            "Z": df[4].astype(float),
        }
    )
    # 3x3 rotation matrix from cols 8..16, row-major.
    R = df.iloc[:, 8:17].astype(float).to_numpy().reshape(-1, 3, 3)
    out["R"] = list(R)  # store one matrix per row
    out["camera_id"] = camera_id
    out["track_label"] = track_label
    # Resolve image path: ../Camera{NN}/<filename> relative to CSV.
    cam_dir = p.parent.parent / f"Camera{camera_id:02d}"
    out["image_path"] = out["filename"].map(lambda fn: str(cam_dir / fn))
    return out


def discover_mms_dataset(root: str | Path) -> tuple[pd.DataFrame, dict[int, CameraIntrinsics]]:
    """
    Walk a SEC*/SURV*/TRACK*/CameraPos tree and load everything.

    Returns
    -------
    poses : DataFrame with one row per (frame, camera). All five cameras
            of a frame are distinct rows. Sorted by track + filename.
    intrinsics : Dict[camera_id, CameraIntrinsics] — one shared calib per
                 camera (all tracks reuse the same per-camera calibration).
    """
    root = Path(root)
    csv_paths = sorted(root.glob("SURV*/TRACK*/CameraPos/*_External_Orientation.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no External_Orientation CSVs under {root}")

    intrinsics: dict[int, CameraIntrinsics] = {}
    dfs: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        m = _CAMERA_ID_RE.search(csv_path.name)
        if not m:
            print(f"[MMS] skip unrecognised CSV name: {csv_path.name}")
            continue
        cam_id = int(m.group(1))
        # Camera5 on Leica Pegasus is a 360° spherical/panoramic unit
        # (PanoramaHotSpot in its Internal Orientation; image is
        # equirectangular). We only handle the four perspective
        # cameras (1–4) here — spherical projection is a different
        # math path, can be added later if needed.
        if cam_id == 5:
            continue
        track_label = csv_path.parent.parent.name  # TRACK01 etc.
        df = parse_external_orientation(csv_path, cam_id, track_label)
        dfs.append(df)
        # Load matching Internal Orientation once per camera. Convention:
        # the file lives in the same CameraPos folder under the name
        # TRACK{NN}_Camera{NN}_Internal_Orientation.txt; we cache by
        # camera_id so subsequent tracks reuse the same calibration.
        if cam_id not in intrinsics:
            int_path = csv_path.with_name(
                csv_path.name.replace(
                    "External_Orientation.csv", "Internal_Orientation.txt"
                )
            )
            if int_path.exists():
                intrinsics[cam_id] = parse_internal_orientation(int_path)

    poses = pd.concat(dfs, ignore_index=True)
    return poses, intrinsics


def project_voxels_to_camera(
    voxel_xyz_world: np.ndarray,
    camera_pos: np.ndarray,
    R: np.ndarray,
    intr: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project (N, 3) voxel world coords into a single camera frame.

    Returns
    -------
    uv     : (N, 2) float — pixel coords (col, row)
    z_cam  : (N,) float — depth in camera coords (>0 means in front)
    valid  : (N,) bool — Z>0 AND uv inside image bounds

    Convention: P_cam = R^T · (P_w - C_w).
    """
    rel = voxel_xyz_world - camera_pos[None, :]  # (N, 3)
    # R is camera→world, so world→camera is R^T:
    p_cam = rel @ R  # equivalent to (R.T @ rel.T).T
    z_cam = p_cam[:, 2]
    # Perspective projection, avoiding divide-by-zero with a tiny epsilon
    # on the depth (caller filters Z<=0 anyway).
    inv_z = 1.0 / np.where(z_cam == 0.0, 1e-9, z_cam)
    u = intr.fx * p_cam[:, 0] * inv_z + intr.cx
    v = intr.fy * p_cam[:, 1] * inv_z + intr.cy
    uv = np.stack([u, v], axis=1)
    valid = (
        (z_cam > 0.0)
        & (u >= 0.0)
        & (u < intr.width)
        & (v >= 0.0)
        & (v < intr.height)
    )
    return uv, z_cam, valid


def recolour_voxels_from_mms(
    rep_xyz: np.ndarray,
    rep_rgb_fallback: np.ndarray,
    mms_root: str | Path,
    *,
    roi_bbox: Optional[tuple[float, float, float, float]] = None,
    max_distance_m: float = 50.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace voxel RGB with samples from MMS vehicle-camera frames.

    For every camera frame inside the ROI, project ALL voxels into the
    image, keep only voxels that (a) land in front of the camera, (b)
    fall inside the image bounds, and (c) are within max_distance_m
    metres of that camera. Among all frames that see a given voxel,
    pick the closest one — that's the "least blurry, least oblique"
    view. Voxels never seen by any camera keep their input rep_rgb.

    Parameters
    ----------
    rep_xyz : (N, 3) float — voxel world coords (UTM52N).
    rep_rgb_fallback : (N, 3) uint8 — current LIDAR-baked RGB; used
        where no camera sees the voxel.
    mms_root : path to the SEC* dataset folder containing SURV*/TRACK*/
        CameraPos/ + Camera{N}/.
    roi_bbox : (x_west, y_south, x_east, y_north) — keep only frames
        whose camera position is inside this rectangle. If None, uses
        all frames (slow).
    max_distance_m : voxels further than this from a frame are skipped
        for that frame (long-range pixels are big + blurry).

    Returns
    -------
    rgb : (N, 3) uint8 — recoloured where a camera saw the voxel,
          fallback otherwise.
    seen_mask : (N,) bool — True where a camera was found.
    """
    poses, intrinsics = discover_mms_dataset(mms_root)
    if verbose:
        print(
            f"[MMS] loaded {len(poses):,} frames over cameras "
            f"{sorted(intrinsics.keys())}"
        )
    if roi_bbox is not None:
        x0, y0, x1, y1 = roi_bbox
        m = (
            (poses["X"] >= x0)
            & (poses["X"] <= x1)
            & (poses["Y"] >= y0)
            & (poses["Y"] <= y1)
        )
        poses = poses[m].reset_index(drop=True)
        if verbose:
            print(f"[MMS] ROI filter -> {len(poses):,} frames")
    if len(poses) == 0:
        return rep_rgb_fallback.copy(), np.zeros(len(rep_xyz), dtype=bool)

    n = len(rep_xyz)
    rgb = rep_rgb_fallback.copy().astype(np.uint8)
    chosen_dist = np.full(n, np.inf, dtype=np.float32)
    seen_mask = np.zeros(n, dtype=bool)

    cam_positions = poses[["X", "Y", "Z"]].to_numpy()
    rotations = np.stack(poses["R"].to_numpy(), axis=0)  # (F, 3, 3)
    cam_ids = poses["camera_id"].to_numpy()
    image_paths = poses["image_path"].to_numpy()

    n_recoloured_log = 0
    for f_idx in range(len(poses)):
        cam_pos = cam_positions[f_idx]
        R = rotations[f_idx]
        intr = intrinsics[int(cam_ids[f_idx])]

        # Cheap proximity filter — skip far voxels for this frame.
        dxyz = rep_xyz - cam_pos[None, :]
        d = np.sqrt((dxyz * dxyz).sum(axis=1))
        near = d < max_distance_m
        # Only voxels closer than what we've already picked for them
        better = near & (d < chosen_dist)
        if not better.any():
            continue

        # Project just the candidates (saves work for the long tail
        # of frames that share many neighbours with already-picked ones).
        cand_idx = np.where(better)[0]
        cand_xyz = rep_xyz[cand_idx]
        uv, z_cam, valid = project_voxels_to_camera(
            cand_xyz, cam_pos, R, intr
        )
        if not valid.any():
            continue

        # Resolve final mask among the candidates.
        local_valid = valid
        update_idx = cand_idx[local_valid]
        if update_idx.size == 0:
            continue
        # Sample image once for this frame at the valid uv batch.
        sampled = sample_image_at_uv(image_paths[f_idx], uv[local_valid])
        rgb[update_idx] = sampled
        chosen_dist[update_idx] = d[update_idx]
        seen_mask[update_idx] = True
        n_recoloured_log += int(update_idx.size)

        if verbose and (f_idx + 1) % 200 == 0:
            print(
                f"[MMS] {f_idx + 1}/{len(poses)} frames processed, "
                f"{seen_mask.sum():,}/{n:,} voxels recoloured so far"
            )

    if verbose:
        print(
            f"[MMS] done: {seen_mask.sum():,}/{n:,} voxels recoloured "
            f"({100.0 * seen_mask.mean():.1f}%)"
        )
    return rgb, seen_mask


def sample_image_at_uv(image_path: str | Path, uv: np.ndarray) -> np.ndarray:
    """
    Bilinear sample an RGB image at (N, 2) float pixel coords.

    Returns (N, 3) uint8.
    """
    p = Path(image_path)
    img = Image.open(p).convert("RGB")
    pix = np.asarray(img, dtype=np.uint8)
    h, w = pix.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = (u - u0).astype(np.float32)[:, None]
    fv = (v - v0).astype(np.float32)[:, None]
    p00 = pix[v0, u0].astype(np.float32)
    p10 = pix[v0, u0 + 1].astype(np.float32)
    p01 = pix[v0 + 1, u0].astype(np.float32)
    p11 = pix[v0 + 1, u0 + 1].astype(np.float32)
    top = p00 * (1 - fu) + p10 * fu
    bot = p01 * (1 - fu) + p11 * fu
    out = top * (1 - fv) + bot * fv
    return np.clip(out, 0, 255).astype(np.uint8)
