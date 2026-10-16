"""HD map spec loader — interprets the precision road map layers via config/hdmap_specs/<id>.json.

Layer -> file, layer -> category (+role), forwarded attributes and Kind dispatch live in
the config, not in code. If a new region has different layers or surface-mark Kind codes,
only the spec file changes (no code). "hdmap_spec" in the region config selects the spec.

spec JSON:
  {
    "id": "korean_hdmap_2020",
    "dispatch_default": "surface_mark",   # fallback category for unmapped dispatch codes
    "layers": {
      "<LAYER>": {
        "file": "<LAYER>.shp",
        "category": "link",               # fixed category (when there is no dispatch)
        "role": "overlay" | "point",
        "attributes": ["ID", ...],        # SHP columns forwarded to the viewer
        "dispatch": { "by": "Kind",       # per-code category branch (e.g. B3)
                      "map": { "5111": "crosswalk", ... } }
      }, ...
    }
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HdmapSpec:
    spec_id: str
    layers: dict[str, dict[str, Any]]  # insertion order = processing order
    dispatch_default: str

    def layer_files(self) -> dict[str, str]:
        """Layer name -> SHP file name (for load_hdmap_layers)."""
        return {n: d["file"] for n, d in self.layers.items() if "file" in d}

    def role(self, layer: str) -> str:
        return self.layers.get(layer, {}).get("role", "overlay")

    def attributes_for(self, layer: str) -> list[str]:
        return list(self.layers.get(layer, {}).get("attributes", []))

    def category_for(self, layer: str, row: Any) -> str:
        """Category of one feature: per-code branch if there is a dispatch (unmapped ->
        dispatch_default), otherwise the fixed category."""
        d = self.layers.get(layer, {})
        disp = d.get("dispatch")
        if disp:
            col = disp.get("by", "Kind")
            try:
                code = str(int(row.get(col)))
            except (TypeError, ValueError):
                return self.dispatch_default
            return disp.get("map", {}).get(code, self.dispatch_default)
        return d.get("category", self.dispatch_default)


def load_hdmap_spec(spec_id: str, project_root: Path) -> HdmapSpec:
    p = project_root / "config" / "hdmap_specs" / f"{spec_id}.json"
    if not p.exists():
        avail = (
            ", ".join(sorted(f.stem for f in (p.parent).glob("*.json")))
            if p.parent.is_dir() else "(none)"
        )
        raise FileNotFoundError(
            f"hdmap spec not found: {p}\n  available specs: {avail}"
        )
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return HdmapSpec(
        spec_id=cfg.get("id", spec_id),
        layers=cfg.get("layers", {}),
        dispatch_default=cfg.get("dispatch_default", "surface_mark"),
    )
