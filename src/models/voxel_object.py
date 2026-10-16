from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class VoxelObject:
    object_id: str
    category: str
    color: str
    voxels: List[Tuple[int, int, int]] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)

    def add_voxel(self, gx: int, gy: int, gz: int) -> None:
        self.voxels.append((gx, gy, gz))

    def extend_voxels(self, voxel_list: List[Tuple[int, int, int]]) -> None:
        self.voxels.extend(voxel_list)

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "color": self.color,
            "voxels": self.voxels,
            "attributes": self.attributes
        }