"""Bakes per-chunk renderable geometry into scene_chunks JSON.

Storage format
--------------
Position / RGB arrays are emitted as base64-encoded little-endian
Float32 binary, NOT JSON text. A 72-float wireframe slot per voxel
as JSON text averages ~15 chars per float; as base64-encoded
Float32 it's 5.33 chars per float (1.33× the raw 4 bytes). For
~280k voxels that's the difference between a 520 MB and a 160 MB
preview JSON — i.e. the difference between "browser hangs parsing"
and "loads in well under a second".

The serialised key gains a `_b64` suffix so the web side knows
to decode rather than read as a JSON number array. See
`web/src/core/bakedGeometry.ts` for the decoder.


Why this exists
---------------
The web viewer's cell-crossing fps dip ultimately came from JS-side
BufferGeometry assembly: each macro cell that the driving-focus zone
swept into had to walk N voxels, append 72 floats per voxel for cube
edges (12 edges × 2 vertices × 3 axes), then materialise a Float32Array
and a THREE.BufferAttribute synchronously on the main thread. With 8
cells mounting at once that hit ~80 ms spikes — visible as stutter
during a drive.

This module shifts that work to scene-build time. For every chunk the
pipeline already emits, we additionally pre-compute the flat float
arrays that the renderer would have built anyway, bucketed by category
so the web side can splice them straight into per-category lineSegments
/ points geometries without a per-voxel loop.

Output format (attached as chunk.baked_geometry, then serialised to
JSON via SceneChunk.to_dict):

    {
      "voxel_wireframe": {
        "by_category": {
          "guardrail": { "positions": [x0,y0,z0, x1,y1,z1, ...] },
          ...
        }
      },
      "voxel_centers": {
        "by_category": {
          "guardrail": {
            "positions": [...],  # 3N floats, local scene-origin space
            "rgb": [...]         # 3N floats, normalised 0..1
          },
          ...
        }
      }
    }

Position values are in local (scene-origin) space — same as what the
web's existing `(gx + 0.5) * vs` math produces — so the GPU upload
side requires zero post-processing.

Cell-edit overrides are intentionally NOT baked. When the user
re-categorises a cell at runtime, the web side falls back to dynamic
geometry for that voxel only. Override use is rare and localised, so
the baked fast path still covers ~all cell-crossing traffic.

Color baking strategy
---------------------
- LIDAR voxels (have attributes.voxel_colors): bake the per-voxel RGB
  the LAS file recorded. VoxelCenterLayer renders that as the dot
  colour, preserving the "photographic" look of the LIDAR cloud.
- SHP-symbolic voxels (sign / pole / etc.): bake the category's
  default colour repeated per voxel. The category default is what
  the renderer would have used anyway absent a per-voxel override.

Wireframe colour stays category-level — VoxelLayer renders one
`<lineSegments>` per category with a material colour, and that
material colour reacts to the user's `categoryColorOverrides` at
render time. So colour edits don't invalidate the baked positions.
"""
from __future__ import annotations

import base64
from typing import Iterable

import numpy as np

from src.core.grid_config import GridConfig
from src.models.scene_chunk import SceneChunk
from src.models.voxel_object import VoxelObject


def _floats_to_b64(values: Iterable[float]) -> str:
    """Pack a flat float array into a base64 string of Float32 LE
    bytes. Web side decodes via the inverse: base64 → Uint8Array →
    Float32Array view. ~4x smaller than the same floats expressed as
    JSON text, with no precision loss for the values the renderer
    actually needs (sub-mm cell centres / wireframe corners)."""
    arr = np.asarray(list(values), dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    """'#a5d6a7' → (0.647, 0.839, 0.654). Falls back to mid-grey on
    parse failure so a bad colour string never crashes a build."""
    if not isinstance(hex_str, str) or not hex_str.startswith("#"):
        return (0.5, 0.5, 0.5)
    s = hex_str.lstrip("#")
    if len(s) == 3:
        s = "".join(c + c for c in s)
    if len(s) != 6:
        return (0.5, 0.5, 0.5)
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return (0.5, 0.5, 0.5)
    return (r, g, b)


def _append_box_edges(out: list[float], cx: float, cy: float, cz: float, size: float) -> None:
    """Mirrors web/src/rendering/VoxelLayer.tsx::appendBoxEdges. Emits
    12 cube edges (24 vertex floats × 3 axes = 72 floats per cube).
    Vertex order is irrelevant for line-segment rendering — pairs are
    interpreted as (a→b) edges — so as long as each consecutive pair
    forms one of the cube's 12 edges, the renderer draws the wireframe
    correctly."""
    h = size / 2.0
    x0, x1 = cx - h, cx + h
    y0, y1 = cy - h, cy + h
    z0, z1 = cz - h, cz + h
    # bottom 4 edges
    out.extend([x0, y0, z0, x1, y0, z0])
    out.extend([x1, y0, z0, x1, y1, z0])
    out.extend([x1, y1, z0, x0, y1, z0])
    out.extend([x0, y1, z0, x0, y0, z0])
    # top 4 edges
    out.extend([x0, y0, z1, x1, y0, z1])
    out.extend([x1, y0, z1, x1, y1, z1])
    out.extend([x1, y1, z1, x0, y1, z1])
    out.extend([x0, y1, z1, x0, y0, z1])
    # 4 verticals
    out.extend([x0, y0, z0, x0, y0, z1])
    out.extend([x1, y0, z0, x1, y0, z1])
    out.extend([x1, y1, z0, x1, y1, z1])
    out.extend([x0, y1, z0, x0, y1, z1])


def bake_chunk_geometry(chunk: SceneChunk, grid_config: GridConfig) -> dict:
    """Build the baked_geometry payload for one chunk.

    Iterates each voxel_object's voxels and routes those whose macro-
    chunk index matches this chunk into per-category position / colour
    arrays. The pipeline currently attaches each voxel_object to every
    chunk it touches, so a single voxel of a poke-out object emits
    geometry only in the chunk it actually sits in.

    Returns the dict that's serialised under chunk.baked_geometry. An
    empty result (no voxels in this chunk) still returns the structural
    keys so the web side's "if (chunk.baked_geometry)" check is simple.
    """
    vs = grid_config.voxel_size
    cx, cy = chunk.chunk_x, chunk.chunk_y
    ox, oy, oz = grid_config.origin

    # cat -> [floats]
    wireframe_by_cat: dict[str, list[float]] = {}
    # cat -> {"positions": [...], "rgb": [...]}
    centers_by_cat: dict[str, dict[str, list[float]]] = {}

    for obj in chunk.voxel_objects:
        cat = obj.category
        attrs = obj.attributes or {}
        obj_ground = attrs.get("ground_z_local", 0.0)
        if not isinstance(obj_ground, (int, float)):
            obj_ground = 0.0

        # Optional per-voxel arrays — parallel to obj.voxels, 3N each.
        # Length mismatch (e.g., stale legacy data) → drop, fall back
        # to the category default colour and the computed grid centre.
        n = len(obj.voxels)
        raw_colors = attrs.get("voxel_colors")
        raw_positions = attrs.get("voxel_positions")
        has_colors = isinstance(raw_colors, list) and len(raw_colors) == n * 3
        has_positions = (
            isinstance(raw_positions, list) and len(raw_positions) == n * 3
        )

        # Category default for any voxel that lacks per-voxel RGB.
        default_rgb = _hex_to_rgb01(obj.color)

        for i, (gx, gy, gz) in enumerate(obj.voxels):
            # Filter — only voxels physically inside this chunk.
            wx, wy, _ = grid_config.grid_index_to_world(gx, gy, gz)
            vcx, vcy = grid_config.world_to_macro_chunk_index(wx, wy)
            if (vcx, vcy) != (cx, cy):
                continue

            # Local centre (scene-origin space, what the web renderer
            # actually plots).
            local_cx = (gx + 0.5) * vs
            local_cy = (gy + 0.5) * vs
            local_cz = (gz + 0.5) * vs + obj_ground

            # Wireframe — 72 floats per voxel.
            wf = wireframe_by_cat.get(cat)
            if wf is None:
                wf = []
                wireframe_by_cat[cat] = wf
            _append_box_edges(wf, local_cx, local_cy, local_cz, vs)

            # Centres — 3 position floats + 3 RGB floats per voxel.
            cd = centers_by_cat.get(cat)
            if cd is None:
                cd = {"positions": [], "rgb": []}
                centers_by_cat[cat] = cd
            if has_positions:
                # LIDAR raw world position, shifted into local space.
                wx_lidar = raw_positions[i * 3] - ox
                wy_lidar = raw_positions[i * 3 + 1] - oy
                wz_lidar = raw_positions[i * 3 + 2] - oz
                cd["positions"].extend([wx_lidar, wy_lidar, wz_lidar])
            else:
                cd["positions"].extend([local_cx, local_cy, local_cz])
            if has_colors:
                cd["rgb"].extend([
                    raw_colors[i * 3] / 255.0,
                    raw_colors[i * 3 + 1] / 255.0,
                    raw_colors[i * 3 + 2] / 255.0,
                ])
            else:
                cd["rgb"].extend(default_rgb)

    return {
        "voxel_wireframe": {
            "by_category": {
                cat: {"positions_b64": _floats_to_b64(pos)}
                for cat, pos in wireframe_by_cat.items()
            }
        },
        "voxel_centers": {
            "by_category": {
                cat: {
                    "positions_b64": _floats_to_b64(data["positions"]),
                    "rgb_b64": _floats_to_b64(data["rgb"]),
                }
                for cat, data in centers_by_cat.items()
            }
        },
    }


def bake_all_chunks(
    chunk_map: dict, grid_config: GridConfig
) -> tuple[int, int]:
    """Walks every chunk in `chunk_map` and attaches a baked_geometry
    dict to each. Returns (chunks_baked, total_voxels_baked) for the
    build log so a regression in the bake stage is visible without
    needing to scrape the JSON afterwards.
    """
    chunks_baked = 0
    total_voxels = 0
    for chunk in chunk_map.values():
        baked = bake_chunk_geometry(chunk, grid_config)
        # SceneChunk.to_dict will serialise this — see scene_chunk.py
        # patch for the new field.
        chunk.baked_geometry = baked
        chunks_baked += 1
        # Voxel count for log = (base64 char length / 5.33) / 3 floats
        # per voxel. base64 inflates 4 bytes → 5.33 chars; each voxel's
        # centre is 3 Float32 = 12 bytes = 16 chars. Approximate count
        # via len / 16 is enough for the log line.
        for cat_data in baked["voxel_centers"]["by_category"].values():
            b64 = cat_data.get("positions_b64", "")
            total_voxels += len(b64) // 16
    return chunks_baked, total_voxels


def bake_preview_by_macro_cell(
    voxel_objects: list, grid_config: GridConfig
) -> tuple[dict[str, dict], int, int]:
    """Per-macro-cell baking for preview-mode scenes.

    The web's deoksong path loads `scene_preview.json` and never fetches
    the per-chunk JSON files. So baking into `chunk.baked_geometry`
    alone (above) doesn't reach the renderer — preview-mode needs the
    same bake materialised under a different roof.

    This produces a dict keyed by `"cx,cy"` macro-cell key (matching
    the web's `voxelObjectsByCell` map) so the renderer can look up
    pre-built geometry directly on cell mount. Each entry has the
    same `voxel_wireframe` / `voxel_centers` structure as the
    per-chunk bake.

    Why not just emit the per-chunk bake into the preview JSON: chunks
    can group multiple macro cells, but the web side keys everything by
    individual macro cell. Direct per-macro-cell baking saves the
    renderer from a second re-bucketing step at load time.

    Returns (by_cell map, cells_baked, total_voxels_baked).
    """
    vs = grid_config.voxel_size
    m = grid_config.macro_grid_size
    ox, oy, oz = grid_config.origin

    by_cell: dict[str, dict] = {}
    # cell key → {cat → {"wireframe": [...], "centers_pos": [...], "centers_rgb": [...]}}
    accum: dict[str, dict[str, dict[str, list[float]]]] = {}
    voxel_count = 0

    for obj in voxel_objects:
        cat = obj.category
        attrs = obj.attributes or {}
        obj_ground = attrs.get("ground_z_local", 0.0)
        if not isinstance(obj_ground, (int, float)):
            obj_ground = 0.0
        n = len(obj.voxels)
        raw_colors = attrs.get("voxel_colors")
        raw_positions = attrs.get("voxel_positions")
        has_colors = isinstance(raw_colors, list) and len(raw_colors) == n * 3
        has_positions = (
            isinstance(raw_positions, list) and len(raw_positions) == n * 3
        )
        default_rgb = _hex_to_rgb01(obj.color)

        for i, (gx, gy, gz) in enumerate(obj.voxels):
            # Local cell centre — matches web's
            # `(gx + 0.5) * vs` math.
            local_cx = (gx + 0.5) * vs
            local_cy = (gy + 0.5) * vs
            local_cz = (gz + 0.5) * vs + obj_ground

            # Macro-cell key — same scheme as web's
            # splitVoxelObjectsByCell so the keys line up.
            mcx = int(local_cx // m)
            mcy = int(local_cy // m)
            key = f"{mcx},{mcy}"

            cell = accum.get(key)
            if cell is None:
                cell = {}
                accum[key] = cell
            cat_data = cell.get(cat)
            if cat_data is None:
                cat_data = {
                    "wireframe": [],
                    "centers_pos": [],
                    "centers_rgb": [],
                }
                cell[cat] = cat_data

            _append_box_edges(
                cat_data["wireframe"], local_cx, local_cy, local_cz, vs
            )

            if has_positions:
                cat_data["centers_pos"].extend([
                    raw_positions[i * 3] - ox,
                    raw_positions[i * 3 + 1] - oy,
                    raw_positions[i * 3 + 2] - oz,
                ])
            else:
                cat_data["centers_pos"].extend([local_cx, local_cy, local_cz])
            if has_colors:
                cat_data["centers_rgb"].extend([
                    raw_colors[i * 3] / 255.0,
                    raw_colors[i * 3 + 1] / 255.0,
                    raw_colors[i * 3 + 2] / 255.0,
                ])
            else:
                cat_data["centers_rgb"].extend(default_rgb)
            voxel_count += 1

    # Re-shape into the wire format the web expects (base64 floats).
    for key, cell in accum.items():
        by_cell[key] = {
            "voxel_wireframe": {
                "by_category": {
                    cat: {"positions_b64": _floats_to_b64(data["wireframe"])}
                    for cat, data in cell.items()
                }
            },
            "voxel_centers": {
                "by_category": {
                    cat: {
                        "positions_b64": _floats_to_b64(data["centers_pos"]),
                        "rgb_b64": _floats_to_b64(data["centers_rgb"]),
                    }
                    for cat, data in cell.items()
                }
            },
        }

    return by_cell, len(by_cell), voxel_count
