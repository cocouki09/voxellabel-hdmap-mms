from pathlib import Path
import geopandas as gpd


def read_shp(path: str) -> gpd.GeoDataFrame:
    shp_path = Path(path)
    if not shp_path.exists():
        raise FileNotFoundError(f"SHP file not found: {path}")
    return gpd.read_file(shp_path)


def select_first_n(gdf: gpd.GeoDataFrame, n: int = 5) -> gpd.GeoDataFrame:
    return gdf.head(n).copy()