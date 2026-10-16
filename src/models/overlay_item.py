from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class OverlayItem:
    overlay_id: str
    category: str
    geometry_type: str
    geometry: Any
    color: str
    style: Dict[str, Any] = field(default_factory=dict)
    # Free-form passthrough for source SHP attributes that downstream
    # (the web viewer) needs. For A2_LINK we ship LinkID / FromNodeID /
    # ToNodeID / LaneNo / RoadType so the viewer can build a link
    # graph and run a simulator vehicle along connected links.
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overlay_id": self.overlay_id,
            "category": self.category,
            "geometry_type": self.geometry_type,
            "geometry": self.geometry,
            "color": self.color,
            "style": self.style,
            "attributes": self.attributes,
        }