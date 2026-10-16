from dataclasses import dataclass, field
from typing import List, Optional

from src.models.ground_cell import GroundCell
from src.models.overlay_item import OverlayItem
from src.models.voxel_object import VoxelObject


@dataclass
class SceneChunk:
    chunk_id: str
    chunk_x: int
    chunk_y: int
    ground_cells: List[GroundCell] = field(default_factory=list)
    overlays: List[OverlayItem] = field(default_factory=list)
    voxel_objects: List[VoxelObject] = field(default_factory=list)
    # Per-chunk pre-built geometry arrays the renderer can upload to the
    # GPU without per-voxel JS work. Populated by
    # src/exporters/chunk_geometry_baker.bake_all_chunks() during the
    # build. Optional so legacy chunks without baking still serialise
    # (and the web side falls back to dynamic build when this is
    # missing). See baker module docstring for schema.
    baked_geometry: Optional[dict] = None

    def to_dict(self) -> dict:
        out: dict = {
            "chunk_id": self.chunk_id,
            "chunk_x": self.chunk_x,
            "chunk_y": self.chunk_y,
            "ground_cells": [cell.to_dict() for cell in self.ground_cells],
            "overlays": [overlay.to_dict() for overlay in self.overlays],
            "voxel_objects": [obj.to_dict() for obj in self.voxel_objects],
        }
        if self.baked_geometry is not None:
            out["baked_geometry"] = self.baked_geometry
        return out