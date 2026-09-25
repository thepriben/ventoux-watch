"""Draw the map into the frame of one camera.

    .venv/bin/python -m scripts.build_scene [--fresh]

`--fresh` throws away the copy of OpenStreetMap kept on disk and asks again,
which is what to do after drawing something new on the map. The elevations are
kept: the ground does not change.

This is the install step of a camera. It reads the pose declared in
OpenStreetMap, fits the direction, the tilt and the field of view on a few
landmarks pointed at in both the map and the picture, then answers one question
for every point of the frame: what is on the ground there, and how far.

The answer is found by following the line of sight until it meets the terrain,
then reading what OpenStreetMap has drawn at that spot — forest, meadow, scree,
roadway, roundabout, car park, path, building. Nothing here is specific to Mont
Serein: another camera only needs its own position and its own landmarks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watcher.config import load_config
from watcher.frustum import Pose, distance_m, fit, march, project
from watcher.geometry import load_zones
from watcher.osm import around, road_width_m, surface_of
from watcher.scenemap import CODES
from watcher.terrain import Terrain

GRID_W, GRID_H = 192, 108
REACH_W, REACH_H = 96, 54
LANDMARK_SIZE_M = 2.0
LANDMARK_BIGGEST_M = 9.0
# Fine enough to hold the island of a roundabout sixteen metres across.
MAP_STEP_M = 1.0
ROAD_SLACK_M = 2.5
# Later wins. A car park is drawn corner by corner and a road is only a centre
# line with a guessed width, so the car park has the last word over the tarmac.
PAINT_ORDER = ["meadow", "scree", "forest", "path", "road", "roundabout", "island", "parking", "building"]


def main() -> int:
    cfg = load_config()
    root = Path(cfg["_root"])
    camera = cfg["camera"]
    marks = camera.get("marks") or []
    if not marks:
        print("Aucun repère dans config.camera.marks : impossible de caler la vue.")
        return 1

    pose = Pose(
        lat=float(camera["lat"]),
        lon=float(camera["lon"]),
        ele=float(camera.get("ele") or 0),
        yaw=float(camera.get("bearing") or 0),
        pitch=float(camera.get("pitch") or 0),
        hfov=float(camera.get("fov") or 90),
        height_m=float(camera.get("height_m") or 4),
    )
    pose, score = fit(pose, marks)
    pose, score = _fit_position(pose, marks)
    rms = (score / len(marks)) ** 0.5
    print(f"Calage : cap {pose.yaw:.1f}° site {pose.pitch:.1f}° champ {pose.hfov:.1f}° écart {rms:.4f}")

    reach = float(camera.get("reach_m", 2500))
    apron = float(camera.get("apron_m", 60))
    terrain = Terrain(float(camera["lat"]), float(camera["lon"]), reach, cache=root / "data" / "osm" / "terrain.json")
    terrain.anchor(pose.lat, pose.lon)
    holes = len(terrain.missing())
    if holes:
        print(f"Terrain : {holes} altitudes à demander, environ {holes // 100 + 1} appels", flush=True)
        terrain.build(say=lambda done, total: print(f"  {done}/{total}", flush=True))
        left = len(terrain.missing())
        if left:
            print(f"Terrain : {left} altitudes manquent encore, relance la commande pour les finir")
    near = [mark for mark in marks if distance_m(pose, mark["lat"], mark["lon"]) <= 300]
    if near:
        print(f"Terrain : recalé de {terrain.settle(near):+.1f} m sur {len(near)} repères")
    levels = sorted(float(mark["ele"]) for mark in marks if distance_m(pose, mark["lat"], mark["lon"]) <= apron)
    if apron > 0 and levels:
        terrain.level(apron, levels[len(levels) // 2])
        print(f"Terrain : replat à {levels[len(levels) // 2]:.0f} m sur {apron:.0f} m")

    fresh = "--fresh" in sys.argv
    data = around(pose.lat, pose.lon, reach, root / "data" / "osm" / "around.json", max_age_s=0 if fresh else 30 * 86400)
    if fresh:
        print("Carte   : relue depuis OpenStreetMap")
    land, codes = _land(data, pose, reach)
    print(f"Carte   : {int((land > 0).sum() * MAP_STEP_M ** 2 / 10_000)} ha de surfaces connues autour")

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    far = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    sky = _sky_mask(load_zones(root / cfg["zones"]), GRID_W, GRID_H)
    span = int(reach / MAP_STEP_M)
    for row in range(GRID_H):
        for column in range(GRID_W):
            hit = march(pose, (column + 0.5) / GRID_W, (row + 0.5) / GRID_H, terrain, reach)
            if hit is None:
                continue
            east, north, distance = hit
            far[row, column] = distance
            if sky[row, column] and distance > 0.8 * reach:
                continue
            x = int(round(east / MAP_STEP_M)) + span
            y = span - int(round(north / MAP_STEP_M))
            if 0 <= x < land.shape[1] and 0 <= y < land.shape[0]:
                grid[row, column] = land[y, x]

    back = {index: name for name, index in codes.items()}
    back[len(codes) + 1] = "sky"
    grid[far == 0] = len(codes) + 1
    rows = ["".join("." if cell == 0 else CODES[back[cell]] for cell in line) for line in grid]

    out = root / "config" / "scene.json"
    out.write_text(
        json.dumps(
            {
                "pose": {**pose.as_dict(), "rms": round(rms, 5), "reach_m": reach},
                "grid": rows,
                "reach": _shrink(far),
                "landmarks": _landmarks(data, pose, terrain, rms, reach),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for line in rows:
        for cell in line:
            counts[cell] = counts.get(cell, 0) + 1
    share = {key: round(100 * value / (GRID_W * GRID_H)) for key, value in sorted(counts.items())}
    seen = far[far > 0]
    print(f"Surfaces : {share}")
    print(f"Portée   : de {seen.min():.0f} m à {seen.max():.0f} m, médiane {np.median(seen):.0f} m")
    print(f"Écrit    : {out}")
    return 0


def _land(data: dict, pose: Pose, reach: float) -> tuple[np.ndarray, dict]:
    """Paint the map itself, seen from above, in metres around the camera."""
    span = int(reach / MAP_STEP_M)
    image = np.zeros((2 * span + 1, 2 * span + 1), dtype=np.uint8)
    codes = {name: index + 1 for index, name in enumerate(PAINT_ORDER)}
    scale = math.cos(math.radians(pose.lat))

    def to_pixels(points):
        out = []
        for lat, lon in points:
            east = (lon - pose.lon) * 111_320.0 * scale
            north = (lat - pose.lat) * 110_540.0
            out.append((int(round(east / MAP_STEP_M)) + span, span - int(round(north / MAP_STEP_M))))
        return np.array(out, dtype=np.int32)

    shapes: dict[str, list] = {name: [] for name in PAINT_ORDER}
    for element in data.get("elements") or []:
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags") or {}
        name = surface_of(tags)
        if not name:
            continue
        shapes[name].append((tags, [(point["lat"], point["lon"]) for point in geometry]))

    for name in PAINT_ORDER:
        for tags, points in shapes[name]:
            shape = to_pixels(points)
            if tags.get("highway"):
                # OpenStreetMap draws a centre line and the width is a guess, so
                # the tarmac is painted with a verge on either side. Erring wide
                # keeps a car on the road; erring narrow calls it off-road.
                thick = max(1, int(round((road_width_m(tags) + 2 * ROAD_SLACK_M) / MAP_STEP_M)))
                cv2.polylines(image, [shape], False, int(codes[name]), thick)
                _island(image, shape, tags, codes)
            elif len(points) > 3 and points[0] == points[-1]:
                cv2.fillPoly(image, [shape], int(codes[name]))
            else:
                cv2.polylines(image, [shape], False, int(codes[name]), 2)
    return image, codes


def _island(image: np.ndarray, shape: np.ndarray, tags: dict, codes: dict) -> None:
    """The planted middle of a roundabout, which the tarmac has just covered.

    A roundabout is one closed ring sixteen metres across, and painting it with
    a road's width plus its verges fills the circle. What belongs in the middle
    is not roadway: it is the planted island, with its stones and its wooden
    figures, where nothing drives and nobody stands still. Only the carriageway
    is taken back out here, without the verge — a verge may spill onto grass,
    never onto the island.
    """
    if tags.get("junction") != "roundabout" or len(shape) < 4 or not np.array_equal(shape[0], shape[-1]):
        return
    carriage = max(1, int(round(road_width_m(tags) / MAP_STEP_M)))
    inner = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(inner, [shape], 1)
    cv2.polylines(inner, [shape], True, 0, carriage)
    image[inner.astype(bool)] = int(codes["island"])


def _landmarks(data: dict, pose: Pose, terrain: Terrain, rms: float, reach_m: float) -> list[dict]:
    """The small fixed things a camera keeps mistaking for a passer-by.

    A wooden statue, a hut, a shelter: they never move, so a box drawn tightly
    around one is a light or a shadow sweeping over it, never an event. Only
    objects small enough to be confused with a person or a car are kept; a
    whole chalet is a surface, not a landmark.
    """
    out = []
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        spot, size = _fixture(element, tags)
        if spot is None or size > LANDMARK_BIGGEST_M:
            continue
        lat, lon = spot
        span = distance_m(pose, lat, lon)
        if span > reach_m:
            continue
        east = (lon - pose.lon) * 111_320.0 * math.cos(math.radians(pose.lat))
        north = (lat - pose.lat) * 110_540.0
        seen = project(pose, lat, lon, terrain.height(east, north) + size / 2)
        if seen is None or not (0 <= seen[0] <= 1 and 0 <= seen[1] <= 1):
            continue
        half = math.degrees(math.atan2(max(size, LANDMARK_SIZE_M) / 2, max(span, 5.0)))
        width = math.tan(math.radians(half)) / (2 * math.tan(math.radians(pose.hfov / 2)))
        out.append(
            {
                "name": tags.get("name") or _plain(tags),
                "kind": tags.get("artwork_type") or tags.get("building") or tags.get("historic") or tags.get("tourism") or "",
                "osm": f"{element['type']}/{element['id']}",
                "x": round(seen[0], 4),
                "y": round(seen[1], 4),
                # Widened by how well the view is calibrated: we know where the
                # statue is to within that much, no better.
                "r": round(max(0.010, min(0.08, width + rms)), 4),
                "size_m": round(size, 1),
                "distance_m": round(span, 1),
            }
        )
    return out


WORDS = {"kiosk": "cabane", "hut": "cabane", "shed": "cabane", "shelter": "abri", "statue": "statue",
         "memorial": "mémorial", "wayside_cross": "croix", "service": "local technique", "yes": "bâtiment"}


def _plain(tags: dict) -> str:
    """A landmark reads back in the history, so it needs a plain French word."""
    for key in ("artwork_type", "building", "historic", "tourism"):
        word = tags.get(key)
        if word:
            return WORDS.get(word, word)
    return "repère"


def _fixture(element: dict, tags: dict) -> tuple[tuple[float, float] | None, float]:
    """Where a fixed thing stands and how wide it is, in metres."""
    if element.get("type") == "node":
        if not (tags.get("tourism") == "artwork" or tags.get("historic")):
            return None, 0.0
        return (element["lat"], element["lon"]), LANDMARK_SIZE_M
    if not tags.get("building"):
        return None, 0.0
    points = element.get("geometry") or []
    if len(points) < 3:
        return None, 0.0
    lat = sum(point["lat"] for point in points) / len(points)
    lon = sum(point["lon"] for point in points) / len(points)
    scale = math.cos(math.radians(lat))
    across = [(point["lon"] - lon) * 111_320.0 * scale for point in points]
    along = [(point["lat"] - lat) * 110_540.0 for point in points]
    return (lat, lon), max(max(across) - min(across), max(along) - min(along))


def _shrink(far: np.ndarray) -> list[list[int]]:
    """How far the ground is, kept coarse: it changes slowly across the frame."""
    small = cv2.resize(far, (REACH_W, REACH_H), interpolation=cv2.INTER_AREA)
    return [[int(round(value)) for value in row] for row in small]


def _sky_mask(zones: dict, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = (zones.get("polygons") or {}).get("sky")
    if not polygon:
        return mask
    shape = np.array([[point[0] * width, point[1] * height] for point in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [shape], 1)
    return mask


def _fit_position(pose: Pose, marks: list[dict]):
    """A camera node is placed by hand; let the landmarks move it a little."""
    from watcher.frustum import _residual

    best, score = pose, _residual(pose, marks)
    step = {"yaw": 8.0, "pitch": 6.0, "hfov": 12.0, "height_m": 4.0, "lat": 0.0003, "lon": 0.0004}
    for _ in range(7):
        for _sweep in range(60):
            moved = False
            for key, size in step.items():
                for way in (1, -1):
                    values = {
                        "lat": best.lat, "lon": best.lon, "ele": best.ele, "yaw": best.yaw,
                        "pitch": best.pitch, "hfov": best.hfov, "height_m": best.height_m,
                        "aspect": best.aspect,
                    }
                    values[key] += way * size
                    if key == "hfov":
                        values[key] = max(20.0, min(140.0, values[key]))
                    if key == "height_m":
                        values[key] = max(2.0, min(20.0, values[key]))
                    trial = Pose(**values)
                    value = _residual(trial, marks)
                    if value < score - 1e-12:
                        best, score, moved = trial, value, True
            if not moved:
                break
        step = {key: value / 2.5 for key, value in step.items()}
    return best, score


if __name__ == "__main__":
    raise SystemExit(main())
