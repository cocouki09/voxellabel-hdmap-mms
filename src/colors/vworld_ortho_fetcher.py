"""
V-World Satellite (orthophoto) WMTS fetcher.

Given a UTM bbox + API key, downloads the V-World Satellite tiles that
cover the area, stitches them into a single PIL image, and crops to the
exact bbox. The result drops in next to ortho.png and feeds the same
ortho_sampler pipeline path that the manual-capture flow uses — so the
recolour stage is unchanged.

V-World WMTS is OGC-compliant, Web Mercator (EPSG:3857) tile coords,
OSM/Google-compatible z/x/y indexing. Default zoom 18 ≈ 0.6 m/pixel,
well above the 0.5 m voxel size — bilinear sampling in ortho_sampler
will gently downsample without aliasing.

URL pattern (V-World public spec):
  http://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg

Layer aliases worth knowing if "Satellite" returns blanks for a region:
  Satellite (orthophoto)  ← default here
  Hybrid    (orthophoto + labels)
  Base      (map)
  midnight  (dark-mode map)
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

# V-World WMTS endpoint. Note: HTTP (not HTTPS) is the documented one;
# they redirect transparently if HTTPS is preferred by the client.
VWORLD_WMTS_URL = (
    "http://api.vworld.kr/req/wmts/1.0.0/{key}/{layer}/{z}/{y}/{x}.{ext}"
)
TILE_SIZE = 256  # OGC WMTS standard tile size


@dataclass
class FetchedOrtho:
    """Bundles the stitched PIL image with the exact UTM bbox it covers."""

    image: Image.Image
    sub_bbox_utm: Tuple[float, float, float, float]
    zoom: int
    n_tiles: int


def _lonlat_to_tile_fractional(lon: float, lat: float, z: int) -> Tuple[float, float]:
    """OSM/Google web-mercator tile coords (top-left origin at (-180, 85.05N))."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (
        1.0
        - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi
    ) / 2.0 * n
    return x, y


def _tile_to_lonlat(tx: float, ty: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lon, lat


def fetch_vworld_ortho(
    sub_bbox_utm: Tuple[float, float, float, float],
    api_key: str,
    *,
    zoom: int = 18,
    src_crs: str = "EPSG:32652",  # UTM52N
    layer: str = "Satellite",
    ext: str = "jpeg",
    request_delay_sec: float = 0.05,
) -> FetchedOrtho:
    """
    Fetch V-World Satellite imagery covering the given UTM bbox.

    Parameters
    ----------
    sub_bbox_utm : (x_west, y_south, x_east, y_north) in UTM coords
    api_key      : V-World API key
    zoom         : WMTS zoom level. 18 ≈ 0.6 m/pixel; 19 ≈ 0.3 m/pixel
                   (higher zoom = more tiles = slower + bigger image).
    src_crs      : Source CRS for sub_bbox. Defaults to UTM52N.
    layer        : V-World layer name (Satellite for orthophoto).
    ext          : Tile format (jpeg for Satellite, png for vector layers).

    Returns
    -------
    FetchedOrtho — stitched + cropped image + the exact UTM bbox it covers.
                   (sub_bbox_utm is echoed back unchanged so callers can
                   wire it straight into ortho.json.)
    """
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    x_west, y_south, x_east, y_north = sub_bbox_utm

    # UTM → WGS84 lon/lat for the four corners (just need NW + SE here).
    lon_w, lat_n = transformer.transform(x_west, y_north)
    lon_e, lat_s = transformer.transform(x_east, y_south)

    # Fractional tile coords for NW and SE — then integer tile range.
    tx_nw_f, ty_nw_f = _lonlat_to_tile_fractional(lon_w, lat_n, zoom)
    tx_se_f, ty_se_f = _lonlat_to_tile_fractional(lon_e, lat_s, zoom)
    tx0, ty0 = math.floor(tx_nw_f), math.floor(ty_nw_f)
    tx1, ty1 = math.floor(tx_se_f), math.floor(ty_se_f)

    n_x = tx1 - tx0 + 1
    n_y = ty1 - ty0 + 1
    total = n_x * n_y
    print(
        f"[VWORLD] zoom={zoom} tiles={n_x}x{n_y}={total}  "
        f"canvas≈{n_x * TILE_SIZE}x{n_y * TILE_SIZE}px"
    )

    if total > 1024:
        raise ValueError(
            f"refusing to fetch {total} tiles (zoom={zoom} too high "
            f"for this bbox); pick a lower zoom"
        )

    canvas = Image.new("RGB", (n_x * TILE_SIZE, n_y * TILE_SIZE))
    sess = requests.Session()

    for ix, tx in enumerate(range(tx0, tx1 + 1)):
        for iy, ty in enumerate(range(ty0, ty1 + 1)):
            url = VWORLD_WMTS_URL.format(
                key=api_key, layer=layer, z=zoom, y=ty, x=tx, ext=ext
            )
            try:
                resp = sess.get(url, timeout=20)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise RuntimeError(f"V-World tile fetch failed: {url}\n  {e}")
            try:
                tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                # Some "blank" V-World responses come back as text/html on
                # auth errors — surface that clearly.
                snippet = resp.content[:200].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"V-World returned non-image content for {url}:\n"
                    f"  {snippet}\n  ({e})"
                )
            canvas.paste(tile_img, (ix * TILE_SIZE, iy * TILE_SIZE))
            if request_delay_sec > 0:
                time.sleep(request_delay_sec)

    # Canvas covers tile range [tx0..tx1+1, ty0..ty1+1] in fractional coords.
    # Convert our exact bbox corners to canvas pixel coords and crop.
    cw, ch = canvas.size
    px_w_f = (tx_nw_f - tx0) * TILE_SIZE
    py_n_f = (ty_nw_f - ty0) * TILE_SIZE
    px_e_f = (tx_se_f - tx0) * TILE_SIZE
    py_s_f = (ty_se_f - ty0) * TILE_SIZE

    crop_box = (
        max(0, int(round(px_w_f))),
        max(0, int(round(py_n_f))),
        min(cw, int(round(px_e_f))),
        min(ch, int(round(py_s_f))),
    )
    cropped = canvas.crop(crop_box)
    print(
        f"[VWORLD] cropped to {cropped.size[0]}x{cropped.size[1]}px "
        f"covering UTM bbox {sub_bbox_utm}"
    )

    return FetchedOrtho(
        image=cropped,
        sub_bbox_utm=tuple(float(v) for v in sub_bbox_utm),  # type: ignore[arg-type]
        zoom=zoom,
        n_tiles=total,
    )


def save_fetched_ortho(fetched: FetchedOrtho, out_path: str | Path) -> None:
    """Save the stitched image as PNG (lossless — orthophoto recolour
    happens off this file)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fetched.image.save(p, format="PNG")
    print(f"[VWORLD] saved → {p}")
