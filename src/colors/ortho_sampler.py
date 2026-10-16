"""
Orthophoto sampling for voxel recolouring.

The pipeline normally colours each voxel cell with the RGB of its
representative LIDAR point. That RGB is what the MMS camera measured at
capture time — typically muted, exposure-locked, and visibly different
from what an aerial photograph of the same place looks like. To get a
photographic look in the viewer, we sample a user-supplied orthophoto
(a screen capture of the work area, treated as north-up + axis-aligned
for now) at each voxel centre's world (x, y) and replace the LIDAR
colour with the image pixel colour.

Voxels whose (x, y) falls outside the image's declared sub-bbox keep
their LIDAR colour as a graceful fallback — covers the case where the
capture only includes part of the ROI (e.g. tunnel area not visible
from above).

Design:
  - PIL for image I/O (light dep, no GDAL needed; we're not reading
    GeoTIFF metadata, the bbox comes from config).
  - Bilinear sample in numpy — vectorised over all voxel centres.
  - Returns RGB uint8 + a boolean mask, so the caller can `np.where`
    blend cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class OrthoSampler:
    """Loaded orthophoto + placement metadata, ready to sample.

    Placement is parameterised as **centre + size + rotation** of the
    image rectangle in world coords. The legacy north-up sub_bbox is
    just a special case (rotation_deg = 0); ortho_loader.py knows how
    to derive centre/size from a sub_bbox entry for backward compat.

    rotation_deg > 0 means the image is rotated CCW (math convention)
    in the world XY plane around centre_utm. sample() inverse-rotates
    incoming world points into the image's own axis-aligned frame for
    a plain bounds-check + bilinear sample.
    """

    image_path: str
    pixels: np.ndarray  # (H, W, 3) uint8
    centre_utm: Tuple[float, float]
    size_m: Tuple[float, float]   # (width, height) along the image's own axes
    rotation_deg: float = 0.0

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    def sample(
        self,
        xs_world: np.ndarray,
        ys_world: np.ndarray,
        *,
        area_radius_m: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the image at each (x, y) world coordinate.

        area_radius_m > 0 turns the per-voxel sample into a 3×3 grid in
        a square of side 2*radius around (x, y), and returns the
        **most-saturated** pick among the nine. Use this when the cell
        size matches roughly one image pixel — a single-point sample
        misses sparse vivid signal (lane stripes, kerb paint, small
        rooftops) because it lands on the surrounding asphalt instead.
        Vivid-pick over a cell-sized neighbourhood lets those rare
        pixels actually drive the voxel colour.

        area_radius_m == 0 (default) falls back to a single bilinear
        sample at the exact (x, y).

        Returns
        -------
        rgb : (N, 3) uint8 — sampled colours; rows where mask is False
              are zero-filled (caller is expected to use mask).
        mask : (N,) bool   — True where (x, y) lies inside the (possibly
              rotated) image rectangle (any of the 9 offsets count when
              area_radius_m > 0).
        """
        if area_radius_m > 0.0:
            return self._sample_vivid_area(xs_world, ys_world, area_radius_m)
        return self._sample_point(xs_world, ys_world)

    def _sample_vivid_area(
        self,
        xs_world: np.ndarray,
        ys_world: np.ndarray,
        radius_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3×3 grid sample around each (x,y); per-voxel keep the max-sat pick."""
        r = float(radius_m)
        # The (0,0) offset goes first so it acts as a deterministic
        # tie-breaker when every neighbour has the same saturation.
        offsets = [
            (0.0, 0.0),
            (-r, 0.0), (r, 0.0), (0.0, -r), (0.0, r),
            (-r, -r), (r, -r), (-r, r), (r, r),
        ]
        rgbs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for dx, dy in offsets:
            rgb, mask = self._sample_point(xs_world + dx, ys_world + dy)
            rgbs.append(rgb)
            masks.append(mask)
        stack_rgb = np.stack(rgbs, axis=0)        # (9, N, 3) uint8
        stack_mask = np.stack(masks, axis=0)      # (9, N) bool

        # Saturation proxy: max channel - min channel per sample.
        sat = (
            stack_rgb.max(axis=2).astype(np.int16)
            - stack_rgb.min(axis=2).astype(np.int16)
        )
        # Invalid samples (out of image) shouldn't win the argmax — push
        # their saturation below any possible real value.
        sat = np.where(stack_mask, sat, -1)
        best_offset = np.argmax(sat, axis=0)      # (N,)
        n = stack_rgb.shape[1]
        out_rgb = stack_rgb[best_offset, np.arange(n)]   # (N, 3)
        out_mask = stack_mask.any(axis=0)
        # Zero out RGB where no offset was valid so caller's mask-guard
        # produces clean fallback behaviour.
        out_rgb[~out_mask] = 0
        return out_rgb, out_mask

    def _sample_point(
        self, xs_world: np.ndarray, ys_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single bilinear sample at each (x, y) — the original code path."""
        xs = np.asarray(xs_world, dtype=np.float64)
        ys = np.asarray(ys_world, dtype=np.float64)
        n = xs.shape[0]
        if ys.shape[0] != n:
            raise ValueError("xs_world and ys_world must have the same length")

        cx, cy = self.centre_utm
        w_m, h_m = self.size_m
        if not (w_m > 0 and h_m > 0):
            raise ValueError(f"size_m must be positive: {self.size_m}")

        # Inverse-rotate world coords by -rotation_deg around (cx, cy)
        # so the rectangle becomes axis-aligned in the rotated frame.
        theta = math.radians(self.rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = xs - cx
        dy = ys - cy
        x_img = dx * cos_t + dy * sin_t
        y_img = -dx * sin_t + dy * cos_t

        half_w = w_m / 2.0
        half_h = h_m / 2.0
        mask = (
            (x_img >= -half_w)
            & (x_img <= half_w)
            & (y_img >= -half_h)
            & (y_img <= half_h)
        )

        rgb = np.zeros((n, 3), dtype=np.uint8)
        if not mask.any():
            return rgb, mask

        # Image-frame coords -> pixel coords. Row 0 = NORTH edge (image
        # y = +half_h), col 0 = WEST edge (image x = -half_w). UTM y
        # grows north but image row grows south, so we flip.
        W = self.width
        H = self.height
        px = (x_img[mask] + half_w) / w_m * (W - 1)
        py = (half_h - y_img[mask]) / h_m * (H - 1)

        # NEAREST-NEIGHBOUR sampling (replaces bilinear). User reported
        # that vivid spots (pink lane patch) looked desaturated on the
        # voxel cloud even though the source pixels were colourful — the
        # bilinear blend was averaging the vivid pixel with its grey
        # neighbours, killing the saturation that area_radius's max-sat
        # pick then had nothing strong to choose. Nearest sampling keeps
        # the raw camera pixel; saturation survives the area-pick.
        x0 = np.clip(np.round(px).astype(np.int64), 0, W - 1)
        y0 = np.clip(np.round(py).astype(np.int64), 0, H - 1)
        rgb[mask] = self.pixels[y0, x0]
        return rgb, mask



def load_ortho(
    image_path: str,
    *,
    sub_bbox: Tuple[float, float, float, float] | None = None,
    centre_utm: Tuple[float, float] | None = None,
    size_m: Tuple[float, float] | None = None,
    rotation_deg: float = 0.0,
) -> OrthoSampler:
    """
    Load an orthophoto PNG/JPG into an OrthoSampler.

    Pass EITHER:
      - sub_bbox=(x_west, y_south, x_east, y_north) for legacy north-up
        axis-aligned placement, OR
      - centre_utm + size_m + rotation_deg for a rotated rectangle.

    RGBA inputs are flattened to RGB (alpha composited on white).
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"ortho image not found: {image_path}")

    # Resolve placement: centre+size+rotation wins, sub_bbox is legacy.
    if centre_utm is not None and size_m is not None:
        cx, cy = float(centre_utm[0]), float(centre_utm[1])
        w_m, h_m = float(size_m[0]), float(size_m[1])
        rot = float(rotation_deg)
    elif sub_bbox is not None:
        x_west, y_south, x_east, y_north = (float(v) for v in sub_bbox)
        if not (x_east > x_west and y_north > y_south):
            raise ValueError(f"sub_bbox is degenerate: {sub_bbox}")
        cx = (x_west + x_east) / 2.0
        cy = (y_south + y_north) / 2.0
        w_m = x_east - x_west
        h_m = y_north - y_south
        rot = 0.0
    else:
        raise ValueError(
            "load_ortho requires either sub_bbox or centre_utm+size_m"
        )

    img = Image.open(p)
    if img.mode == "RGBA":
        # Drop alpha — we just want colour. Composite against white so
        # any transparent edges read as background rather than black.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    pixels = np.asarray(img, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(
            f"ortho image must be HxWx3 RGB, got shape {pixels.shape}"
        )

    return OrthoSampler(
        image_path=str(p),
        pixels=pixels,
        centre_utm=(cx, cy),
        size_m=(w_m, h_m),
        rotation_deg=rot,
    )
