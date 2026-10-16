import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_configs(project_root: str) -> dict:
    root = Path(project_root)
    return {
        "grid": load_json(str(root / "config" / "grid_config.json")),
        "object_rules": load_json(str(root / "config" / "object_rules.json")),
        "color_map": load_json(str(root / "config" / "color_map.json")),
        "export": load_json(str(root / "config" / "export_config.json")),
    }