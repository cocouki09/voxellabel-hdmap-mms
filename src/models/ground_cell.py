from dataclasses import dataclass
from typing import Optional


@dataclass
class GroundCell:
    gx: int
    gy: int
    base_type: str = "ground_base"
    color: str = "#2b2b2b"
    z_hint: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "gx": self.gx,
            "gy": self.gy,
            "base_type": self.base_type,
            "color": self.color,
            "z_hint": self.z_hint
        }