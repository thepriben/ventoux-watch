"""What OpenStreetMap knows around a camera, kept on disk.

One camera, one cache. Nothing here is specific to Mont Serein: give it a
position and a radius and it answers with the ways and the landmarks around,
plus the ground elevation of every point it had to look up.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("ventoux.osm")

AGENT = "ventoux-watch/0.1 (github.com/thepriben/ventoux-watch)"
MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)
ELEVATION = "https://api.open-meteo.com/v1/elevation"

SURFACE_TAGS = {
    ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest",
    ("natural", "scrub"): "forest",
    ("natural", "tree_row"): "forest",
    ("landuse", "meadow"): "meadow",
    ("landuse", "grass"): "meadow",
    ("natural", "grassland"): "meadow",
    ("natural", "heath"): "meadow",
    ("natural", "scree"): "scree",
    ("leisure", "playground"): "playground",
    ("leisure", "swimming_pool"): "pool",
    ("amenity", "parking"): "parking",
    ("amenity", "parking_space"): "parking",
    ("building", "*"): "building",
}

ROAD_WIDTH_M = {
    "motorway": 12.0,
    "trunk": 10.0,
    "primary": 8.0,
    "secondary": 7.0,
    "tertiary": 6.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 5.0,
    "service": 4.0,
    "track": 3.0,
    "path": 1.6,
    "footway": 1.6,
    "cycleway": 2.0,
    "steps": 1.4,
}


def bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / 110_540.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def around(lat: float, lon: float, radius_m: float, cache: Path, max_age_s: int = 30 * 86400) -> dict:
    """Ways and landmarks around a point. The disk copy is used when fresh."""
    if cache.is_file() and time.time() - cache.stat().st_mtime < max_age_s:
        return json.loads(cache.read_text(encoding="utf-8"))
    south, west, north, east = bbox(lat, lon, radius_m)
    box = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    query = (
        "[out:json][timeout:120];("
        f'way["highway"]({box});'
        f'way["building"]({box});'
        f'way["landuse"]({box});'
        f'way["natural"]({box});'
        f'way["amenity"~"^(parking|parking_space)$"]({box});'
        # Somewhere children play is not somewhere a car drives, and the yellow
        # frame of one is the brightest thing in this picture after the sky.
        f'way["leisure"]({box});'
        f'node["tourism"="artwork"]({box});'
        f'node["historic"]({box});'
        f'node["natural"="tree"]({box});'
        # Masts carry a red lamp for aircraft. At night it blinks in place and
        # reads like the first flame of a fire, so it has to be on the map.
        f'way["man_made"]({box});'
        f'node["man_made"]({box});'
        ");out body geom;"
    )
    payload = _ask(query)
    if not (payload.get("elements") or []):
        # Overpass answers an empty set instead of an error when it is being
        # restarted. Written to disk, that emptiness becomes the map: every
        # surface turns unknown and the watcher stops recognising the road it
        # has been watching for weeks. A blank answer is never an answer.
        log.warning("OpenStreetMap a repondu a vide, la carte en place est conservee")
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        raise RuntimeError("OpenStreetMap n'a rien renvoye et il n'y a pas de carte en cache")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def surface_of(tags: dict) -> str:
    if tags.get("highway"):
        if tags.get("junction") == "roundabout":
            return "roundabout"
        if tags["highway"] in {"path", "footway", "steps", "cycleway", "track"}:
            return "path"
        return "road"
    if tags.get("building"):
        return "building"
    for (key, value), name in SURFACE_TAGS.items():
        if value == "*":
            continue
        if tags.get(key) == value:
            return name
    return ""


def road_width_m(tags: dict) -> float:
    try:
        return float(str(tags.get("width")).split()[0])
    except (TypeError, ValueError):
        pass
    lanes = tags.get("lanes")
    kind = tags.get("highway", "")
    width = ROAD_WIDTH_M.get(kind, 4.0)
    try:
        return max(width, float(lanes) * 3.0)
    except (TypeError, ValueError):
        return width


class Ground:
    """Ground elevation, asked once per point and remembered."""

    def __init__(self, cache: Path, default: float = 0.0):
        self.cache = cache
        self.default = default
        self.known: dict[str, float] = {}
        if cache.is_file():
            self.known = json.loads(cache.read_text(encoding="utf-8"))

    def key(self, lat: float, lon: float) -> str:
        return f"{lat:.5f},{lon:.5f}"

    def at(self, lat: float, lon: float) -> float:
        return self.known.get(self.key(lat, lon), self.default)

    def learn(self, points: list[tuple[float, float]], batch: int = 90, pause_s: float = 1.5) -> int:
        missing = []
        seen = set()
        for lat, lon in points:
            key = self.key(lat, lon)
            if key in self.known or key in seen:
                continue
            seen.add(key)
            missing.append((lat, lon))
        added = 0
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            query = (
                f"{ELEVATION}?latitude={','.join(f'{a:.5f}' for a, _b in chunk)}"
                f"&longitude={','.join(f'{b:.5f}' for _a, b in chunk)}"
            )
            try:
                request = urllib.request.Request(query, headers={"User-Agent": AGENT})
                with urllib.request.urlopen(request, timeout=30) as response:
                    values = json.loads(response.read().decode()).get("elevation") or []
            except Exception:
                time.sleep(10)
                continue
            for (lat, lon), value in zip(chunk, values):
                self.known[self.key(lat, lon)] = float(value)
                added += 1
            time.sleep(pause_s)
        if added:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text(json.dumps(self.known), encoding="utf-8")
        return added


def _ask(query: str) -> dict:
    last = None
    for attempt in range(6):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            request = urllib.request.Request(
                url,
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": AGENT},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode())
        except Exception as exc:
            last = exc
            time.sleep(8)
    raise RuntimeError(f"Overpass n'a pas répondu: {last}")
