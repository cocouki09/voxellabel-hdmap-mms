"""
Per-voxel RGB enhancement — brightness / saturation / contrast boost.

The MMS LIDAR pipeline (Leica Pegasus Two Ultimate in our case) bakes
the vehicle camera colour into each LAS point's RGB at export time. By
the time `voxelize_points_with_representative` picks the closest point
to each cell centre, every voxel already has a "real photo" colour.

The catch is that those colours come out *dim* (mean ~90/255 in
Deoksong) and *grey-leaning* despite ~38% per-pixel saturation —
exposure-locked daytime capture compresses everything into the lower
half of the dynamic range. Boosting brightness lifts the whole frame,
saturation pumps the existing colour signal, contrast widens the gap
between asphalt/lane/grass/vehicle so they read as distinct cubes on
screen instead of a wash of muted earth tones.

All operations are vectorised numpy uint8 → uint8, run on the
`rep_rgb` array right after voxelisation. ~10ms for 500k voxels.
"""

from __future__ import annotations

import numpy as np


def apply_rgb_enhance(
    rgb: np.ndarray,
    *,
    gamma: float = 1.0,
    brightness: float = 1.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Boost an (N, 3) uint8 RGB array.

    Order of application: gamma → brightness → saturation → contrast.
    Each pass clips to [0, 255] before the next so we don't waste
    precision on silently saturated channels.

    - gamma < 1.0 nonlinearly lifts dark pixels while leaving highlights
      near-untouched (e.g. gamma=0.75 turns 50→100 but 200→215). This is
      the right knob for dim MMS captures because plain brightness
      multiplies highlights into 255-clip, which destroys saturation in
      a second pass.
    - brightness is a linear scale; OK for fine-tuning after gamma.
    - saturation > 1 pumps colour around the Rec.709 luma axis,
      preserving perceived brightness.
    - contrast > 1 deepens shadows / brightens highlights around mid-grey.

    Defaults (1.0 each) are a no-op so callers without enhancement
    config get identical output.
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected (N, 3) RGB array, got {rgb.shape}")

    out = rgb.astype(np.float32)

    if gamma != 1.0:
        # Output of x**g for x in [0,1] is bounded in [0,1]; no clip needed.
        out = (out / 255.0) ** float(gamma) * 255.0

    if brightness != 1.0:
        out *= float(brightness)
        np.clip(out, 0, 255, out=out)

    if saturation != 1.0:
        # Luminance via Rec.709 weights — desaturating toward this grey
        # axis (rather than the per-pixel mean) keeps perceived
        # brightness stable when saturation > 1.
        gray = (
            0.2126 * out[:, 0]
            + 0.7152 * out[:, 1]
            + 0.0722 * out[:, 2]
        )
        out[:, 0] = gray + (out[:, 0] - gray) * float(saturation)
        out[:, 1] = gray + (out[:, 1] - gray) * float(saturation)
        out[:, 2] = gray + (out[:, 2] - gray) * float(saturation)
        np.clip(out, 0, 255, out=out)

    if contrast != 1.0:
        # Pivot around mid-grey: pushes darks darker, lights lighter.
        out = (out - 128.0) * float(contrast) + 128.0
        np.clip(out, 0, 255, out=out)

    return out.astype(np.uint8)
