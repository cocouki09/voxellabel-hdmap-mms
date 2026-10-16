from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Feature:
    feature_id: str
    category: str
    role: str
    geometry_type: str
    geometry: Any
    attributes: Dict[str, Any] = field(default_factory=dict)

    def is_voxel_object(self) -> bool:
        return self.role == "voxel_object"

    def is_road_overlay(self) -> bool:
        return self.role == "road_overlay"

    def is_ground_base(self) -> bool:
        return self.role == "ground_base"