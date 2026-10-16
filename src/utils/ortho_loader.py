"""
Region-aware orthophoto + RGB-enhancement loader.

`try_load_ortho_for_region(region, project_root)` reads config/ortho.json,
returns an OrthoSampler — or None if the region is unconfigured,
disabled (`"enabled": false`), or the image file is missing.

`try_load_rgb_enhance(region, project_root)` reads the same config and
returns the per-region `rgb_enhance` dict (brightness/saturation/contrast)
or None if unset.

Both designed to fail soft: a missing config / entry / file just returns
None with a one-line notice, so the pipeline keeps running on the raw
LIDAR-measured RGB instead of hard-crashing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.colors.ortho_sampler import OrthoSampler, load_ortho


def try_load_area_radius(
    region: str, project_root: str | Path
) -> float:
    """
    Return the per-region area_radius_m for vivid-pick area sampling,
    or 0.0 (point sample) if unset / not configured.
    """
    cfg = _load_config(project_root)
    if cfg is None:
        return 0.0
    entry = cfg.get(region)
    if not isinstance(entry, dict):
        return 0.0
    v = entry.get("area_radius_m")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_config(project_root: str | Path) -> Optional[dict]:
    root = Path(project_root)
    cfg_path = root / "config" / "ortho.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ORTHO] config/ortho.json unreadable ({e})")
        return None


def try_load_ortho_for_region(
    region: str, project_root: str | Path
) -> Optional[OrthoSampler]:
    cfg = _load_config(project_root)
    if cfg is None:
        return None

    entry = cfg.get(region)
    if not isinstance(entry, dict):
        return None  # region simply isn't configured — silent skip

    # `enabled: false` lets the user keep the image path + bbox in the
    # config (for documentation / future re-enable) while taking the
    # ortho stage out of the bake — useful when the user concludes
    # the source orthophoto is worse than the raw LIDAR RGB.
    if entry.get("enabled") is False:
        print(f"[ORTHO] '{region}' disabled in config; keeping LIDAR RGB")
        return None

    rel_image = entry.get("image")
    if not isinstance(rel_image, str) or not rel_image:
        return None

    # Two placement modes supported:
    #   1. centre_utm + size_m + rotation_deg (preferred — comes straight
    #      from the in-viewer 'P' georeference key output).
    #   2. sub_bbox (legacy north-up axis-aligned).
    centre = entry.get("centre_utm")
    size = entry.get("size_m")
    rotation = entry.get("rotation_deg", 0.0)
    bbox = entry.get("sub_bbox")

    image_path = Path(project_root) / rel_image
    if not image_path.exists():
        print(
            f"[ORTHO] image not found for '{region}' at {image_path}; "
            f"keeping LIDAR RGB. Drop the capture there and rerun to enable."
        )
        return None

    try:
        if (
            isinstance(centre, list)
            and len(centre) == 2
            and isinstance(size, list)
            and len(size) == 2
        ):
            return load_ortho(
                str(image_path),
                centre_utm=(float(centre[0]), float(centre[1])),
                size_m=(float(size[0]), float(size[1])),
                rotation_deg=float(rotation),
            )
        if isinstance(bbox, list) and len(bbox) == 4:
            return load_ortho(
                str(image_path),
                sub_bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
            )
        print(
            f"[ORTHO] '{region}' has neither centre_utm+size_m nor sub_bbox; "
            f"skipping recolour"
        )
        return None
    except Exception as e:
        print(f"[ORTHO] failed to load '{image_path}' ({e}); skipping recolour")
        return None


def try_load_rgb_enhance(
    region: str, project_root: str | Path
) -> Optional[dict]:
    """
    Return the per-region `rgb_enhance` config dict, or None.

    Expected shape:
      {"brightness": float, "saturation": float, "contrast": float}
    Any subset is fine; missing keys default to 1.0 in apply_rgb_enhance.
    """
    cfg = _load_config(project_root)
    if cfg is None:
        return None
    entry = cfg.get(region)
    if not isinstance(entry, dict):
        return None
    enhance = entry.get("rgb_enhance")
    if not isinstance(enhance, dict):
        return None
    # Filter to the keys we know how to apply, so a typo in config can't
    # silently bleed into apply_rgb_enhance kwargs.
    allowed = ("gamma", "brightness", "saturation", "contrast")
    cleaned = {k: float(enhance[k]) for k in allowed if k in enhance}
    return cleaned or None
