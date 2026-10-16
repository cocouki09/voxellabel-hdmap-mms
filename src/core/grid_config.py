from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridConfig:
    macro_grid_size: float = 5.0
    voxel_size: float = 0.25
    ground_cell_size: float = 0.25
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_scene_height: float = 10.0
    allow_float_size: bool = True

    def world_to_grid_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        gx = int((x - self.origin[0]) // self.voxel_size)
        gy = int((y - self.origin[1]) // self.voxel_size)
        gz = int((z - self.origin[2]) // self.voxel_size)
        return gx, gy, gz

    def grid_index_to_world(self, gx: int, gy: int, gz: int) -> Tuple[float, float, float]:
        x = self.origin[0] + gx * self.voxel_size
        y = self.origin[1] + gy * self.voxel_size
        z = self.origin[2] + gz * self.voxel_size
        return x, y, z

    def world_to_macro_chunk_index(self, x: float, y: float) -> Tuple[int, int]:
        cx = int((x - self.origin[0]) // self.macro_grid_size)
        cy = int((y - self.origin[1]) // self.macro_grid_size)
        return cx, cy
    
    @classmethod
    def from_dict(cls, data: dict) -> "GridConfig":
        return cls(
            macro_grid_size=data.get("macro_grid_size", 5.0),
            voxel_size=data.get("voxel_size", 0.25),
            ground_cell_size=data.get("ground_cell_size", 0.25),
            origin=tuple(data.get("origin", [0.0, 0.0, 0.0])),
            default_scene_height=data.get("default_scene_height", 10.0),
            allow_float_size=data.get("allow_float_size", True),
        )