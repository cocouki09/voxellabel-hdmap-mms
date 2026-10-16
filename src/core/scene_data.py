from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.grid_config import GridConfig
from src.models.ground_cell import GroundCell
from src.models.overlay_item import OverlayItem
from src.models.voxel_object import VoxelObject


@dataclass
class SceneData:
    grid_config: GridConfig
    ground_cells: List[GroundCell] = field(default_factory=list)
    overlays: List[OverlayItem] = field(default_factory=list)
    voxel_objects: List[VoxelObject] = field(default_factory=list)
    path_sources: Dict = field(default_factory=dict)
    # Optional higher-level grouping: each composite references multiple
    # voxel_object ids. Emitted by the LIDAR pipeline when e.g. a traffic
    # light and a street light share the same physical pole.
    composite_objects: List[dict] = field(default_factory=list)
    # Per-macro-cell pre-built renderable geometry, keyed by "cx,cy"
    # cellKey. Populated by src/exporters/chunk_geometry_baker for the
    # preview JSON path (deoksong + others using preview mode). The
    # web side reads this map and skips its dynamic per-voxel JS loop
    # entirely — see baker module docstring for schema.
    baked_geometry_by_cell: Optional[Dict[str, dict]] = None

    def add_ground_cell(self, cell: GroundCell) -> None:
        self.ground_cells.append(cell)

    def add_overlay(self, overlay: OverlayItem) -> None:
        self.overlays.append(overlay)

    def add_voxel_object(self, voxel_object: VoxelObject) -> None:
        self.voxel_objects.append(voxel_object)

    def to_dict(self) -> dict:
        out: dict = {
            "grid_config": {
                "macro_grid_size": self.grid_config.macro_grid_size,
                "voxel_size": self.grid_config.voxel_size,
                "ground_cell_size": self.grid_config.ground_cell_size,
                "origin": self.grid_config.origin,
                "default_scene_height": self.grid_config.default_scene_height,
                "allow_float_size": self.grid_config.allow_float_size
            },
            "ground_cells": [cell.to_dict() for cell in self.ground_cells],
            "overlays": [overlay.to_dict() for overlay in self.overlays],
            "voxel_objects": [obj.to_dict() for obj in self.voxel_objects],
            "composite_objects": self.composite_objects,
            "path_sources": self.path_sources,
        }
        if self.baked_geometry_by_cell is not None:
            out["baked_geometry_by_cell"] = self.baked_geometry_by_cell
        return out