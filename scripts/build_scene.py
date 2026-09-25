"""Draw the map into the frame of one camera.

    .venv/bin/python -m scripts.build_scene

Reads the camera pose and the landmarks that can be pointed at in both the map
and the picture, fits the direction, the tilt and the field of view on them,
then paints every OpenStreetMap surface into a grid the watcher can read.
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
from watcher.frustum import Pose, distance_m, fit, project
from watcher.geometry import load_zones
from watcher.osm import Ground, around, road_width_m, surface_of
from watcher.scenemap import CODES

GRID_W, GRID_H = 192, 108
LANDMARK_SIZE_M = 2.0
PAINT_ORDER = ["meadow", "scree", "forest", "parking", "path", "road", "roundabout", "building"]


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

    data = around(pose.lat, pose.lon, float(camera.get("scene_radius_m") or 900), root / "data" / "osm" / "around.json")
    ground = Ground(root / "data" / "osm" / "elevation.json", default=pose.ele)

    # Beyond the near field the ground model is a coarse elevation grid, and a
    # few metres of error there throw a shape halfway up the mountain. Only the
    # apron around the camera is painted; the rest stays unknown on purpose.
    radius = float(camera.get("scene_radius_m") or 250)
    ways, landmarks = _sort(data, pose, radius)
    wanted = [point for _kind, _tags, points in ways for point in points]
    wanted += [(mark["lat"], mark["lon"]) for mark in landmarks]
    added = ground.learn(wanted)
    if camera.get("flat_apron", True):
        # The apron around a station is flat to within a metre or two, while the
        # public elevation grid is ninety metres wide. Levelling it on the marks
        # that were used to fit the view removes more error than it adds.
        levels = sorted(
            float(mark["ele"]) for mark in marks
            if distance_m(pose, mark["lat"], mark["lon"]) <= radius
        )
        if levels:
            ground.default = levels[len(levels) // 2]
            ground.known = {}
            print(f"Terrain : replat à {ground.default:.0f} m")
    print(f"Terrain : {len(ways)} tracés, {added} altitudes nouvelles")

    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    codes = {name: index + 1 for index, name in enumerate(PAINT_ORDER)}
    for name in PAINT_ORDER:
        for kind, tags, points in ways:
            if kind != name:
                continue
            _paint(grid, codes[name], pose, ground, tags, points, closed=_is_closed(points))

    # The elevation grid is coarse on a steep slope, so a path a hundred metres
    # away can land above the skyline. Nothing on the ground belongs to the sky.
    sky = _sky_mask(load_zones(root / cfg["zones"]))
    codes["sky"] = max(codes.values()) + 1
    grid[sky > 0] = codes["sky"]

    rows = []
    back = {index: name for name, index in codes.items()}
    for row in grid:
        rows.append("".join("." if cell == 0 else CODES[back[cell]] for cell in row))

    marks_out = []
    for mark in landmarks:
        seen = project(pose, mark["lat"], mark["lon"], ground.at(mark["lat"], mark["lon"]) + 1.0)
        if seen is None or not (0 <= seen[0] <= 1 and 0 <= seen[1] <= 1):
            continue
        span = distance_m(pose, mark["lat"], mark["lon"])
        half = math.degrees(math.atan2(LANDMARK_SIZE_M / 2, max(span, 5.0)))
        reach = math.tan(math.radians(half)) / (2 * math.tan(math.radians(pose.hfov / 2)))
        marks_out.append(
            {
                "name": mark.get("name") or mark.get("kind") or "landmark",
                "kind": mark.get("kind") or "",
                "osm": mark.get("osm") or "",
                "x": round(seen[0], 4),
                "y": round(seen[1], 4),
                # Widened by how well the view is calibrated: we know where the
                # statue is to within that much, no better.
                "r": round(max(0.010, min(0.08, reach + rms)), 4),
                "distance_m": round(span, 1),
            }
        )

    out = root / "config" / "scene.json"
    out.write_text(
        json.dumps(
            {
                "pose": {**pose.as_dict(), "rms": round(rms, 5)},
                "grid": rows,
                "landmarks": marks_out,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in rows:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    share = {key: round(100 * value / (GRID_W * GRID_H)) for key, value in sorted(counts.items())}
    print(f"Surfaces : {share}")
    print(f"Repères  : {[mark['name'] for mark in marks_out]}")
    print(f"Écrit    : {out}")
    return 0


def _sky_mask(zones: dict) -> np.ndarray:
    mask = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    polygon = (zones.get("polygons") or {}).get("sky")
    if not polygon:
        return mask
    shape = np.array([[point[0] * GRID_W, point[1] * GRID_H] for point in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [shape], 1)
    return mask


def _sort(data: dict, pose: Pose, radius: float) -> tuple[list, list]:
    ways, landmarks = [], []
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        if element["type"] == "node":
            if not (tags.get("tourism") == "artwork" or tags.get("historic")):
                continue
            if distance_m(pose, element["lat"], element["lon"]) > radius:
                continue
            landmarks.append(
                {
                    "lat": element["lat"],
                    "lon": element["lon"],
                    "name": tags.get("name") or tags.get("artwork_type") or tags.get("historic") or "artwork",
                    "kind": tags.get("artwork_type") or tags.get("historic") or tags.get("tourism") or "",
                    "osm": f"node/{element['id']}",
                }
            )
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        name = surface_of(tags)
        if not name:
            continue
        points = [(point["lat"], point["lon"]) for point in geometry]
        if min(distance_m(pose, lat, lon) for lat, lon in points) > radius:
            continue
        ways.append((name, tags, points))
    return ways, landmarks


def _is_closed(points: list) -> bool:
    return len(points) > 3 and points[0] == points[-1]


def _in_reach(seen) -> bool:
    """A point projected far outside the frame means the shape wraps the camera."""
    return seen is not None and -1.0 <= seen[0] <= 2.0 and -1.0 <= seen[1] <= 2.0


def _paint(grid, code, pose, ground, tags, points, closed: bool) -> None:
    if closed:
        shape = []
        for lat, lon in points:
            seen = project(pose, lat, lon, ground.at(lat, lon))
            if not _in_reach(seen):
                return
            shape.append((seen[0] * GRID_W, seen[1] * GRID_H))
        polygon = np.array(shape, dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(grid, [polygon], int(code))
        return
    half = road_width_m(tags) / 2
    for first, second in zip(points, points[1:]):
        quad = _ribbon(pose, ground, first, second, half)
        if quad is not None:
            cv2.fillPoly(grid, [quad], int(code))


def _ribbon(pose, ground, first, second, half):
    import math

    lat1, lon1 = first
    lat2, lon2 = second
    scale = math.cos(math.radians(lat1))
    dx = (lon2 - lon1) * 111_320.0 * scale
    dy = (lat2 - lat1) * 110_540.0
    span = math.hypot(dx, dy)
    if span < 1e-6:
        return None
    ox = -dy / span * half
    oy = dx / span * half
    corners = []
    for lat, lon in ((lat1, lon1), (lat2, lon2)):
        base = ground.at(lat, lon)
        for way in (1, -1):
            plat = lat + way * oy / 110_540.0
            plon = lon + way * ox / (111_320.0 * scale)
            seen = project(pose, plat, plon, base)
            if not _in_reach(seen):
                return None
            corners.append((seen[0] * GRID_W, seen[1] * GRID_H))
    quad = np.array([corners[0], corners[1], corners[3], corners[2]], dtype=np.int32)
    return quad


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
