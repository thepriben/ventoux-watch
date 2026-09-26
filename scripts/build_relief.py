"""Turn OpenStreetMap and the elevation model into a scene you can walk round.

    .venv/bin/python -m scripts.build_relief

`build_scene` answers one question per point of the picture: what is there, and
how far. That is all the watcher needs, and it is flat by construction — it
only knows the world along the camera's own lines of sight.

This writes the other half: the ground as a shape, in metres east and north of
the camera, with the roads as ribbons, the buildings as volumes and the masts
as columns, every one of them sitting at the height the terrain gives it. From
that you can look at Mont Serein from the webcam's angle, and then from any
other. Nothing here is specific to this camera.
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
from watcher.frustum import Pose, distance_m, fit
from watcher.osm import around, road_width_m, surface_of
from watcher.terrain import Terrain
from scripts.build_scene import MASTS, _fit_position, _lift, _plain

# The elevation model is a 25 m one, so a vector drawn finer than this carries
# no relief anyway. Far away it may be coarser still: half a metre of wiggle a
# kilometre off is a tenth of a pixel.
BEND_M = 0.8
# A storey, and the roof above the last one. Used when the map gives levels but
# no height, which is the common case here.
STOREY_M = 3.0
ROOF_M = 2.5
PLAIN_BUILDING_M = 6.0
# The webcam is bolted to a building, so that building surrounds the eye. Drawn,
# it would fill the frame with its own wall and hide everything it was put there
# to watch. Anything this close is what the camera is standing on.
UNDERFOOT_M = 4.0
BROAD = {"forest": 1, "meadow": 2, "scree": 3}
# One cell per elevation post: finer would invent a relief the model has not
# got. This is only the broad cover, the roads come as vectors.
COVER_STEP_M = 40.0


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

    reach = float(camera.get("reach_m", 2500))
    apron = float(camera.get("apron_m", 60))
    terrain = Terrain(float(camera["lat"]), float(camera["lon"]), reach, cache=root / "data" / "osm" / "terrain.json")
    terrain.anchor(pose.lat, pose.lon)
    if terrain.missing():
        print("Terrain incomplet : lance d'abord scripts.build_scene")
        return 1
    near = [mark for mark in marks if distance_m(pose, mark["lat"], mark["lon"]) <= 300]
    if near:
        terrain.settle(near)
    levels = sorted(float(mark["ele"]) for mark in marks if distance_m(pose, mark["lat"], mark["lon"]) <= apron)
    if apron > 0 and levels:
        terrain.level(apron, levels[len(levels) // 2])

    eye = pose.ele + pose.height_m
    data = around(pose.lat, pose.lon, reach, root / "data" / "osm" / "around.json")
    ways = list(_ways(data, pose, reach))

    payload = {
        "pose": {**pose.as_dict(), "rms": round(rms, 5), "reach_m": reach},
        "terrain": _heightfield(terrain, eye, reach),
        "cover": _cover(ways, reach),
        "roads": _roads(ways, terrain, eye),
        "ribbons": _ribbons(ways, terrain, eye),
        "buildings": _buildings(ways, terrain, eye),
        "woods": _woods(ways),
        "trees": _trees(data, pose, terrain, eye, reach),
        "masts": _masts(data, pose, terrain, eye, reach),
        "figures": _figures(data, pose, terrain, eye, reach),
    }

    out = root / "config" / "relief.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Terrain  : {len(payload['terrain']['grid'])} lignes au pas de {terrain.step_m:.0f} m")
    print(f"Routes   : {len(payload['roads'])} tronçons")
    print(f"Surfaces : {len(payload['ribbons'])} aires pavées")
    print(f"Bâtiments: {len(payload['buildings'])}")
    print(f"Bois     : {len(payload['woods'])} parcelles, {len(payload['trees'])} arbres isolés")
    print(f"Mâts     : {len(payload['masts'])}")
    print(f"Écrit    : {out}  ({out.stat().st_size // 1024} ko)")
    return 0


def _ways(data: dict, pose: Pose, reach: float):
    """Every closed or open way, in metres east and north of the camera."""
    scale = math.cos(math.radians(pose.lat))
    for element in data.get("elements") or []:
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        points = np.array(
            [
                (
                    (point["lon"] - pose.lon) * 111_320.0 * scale,
                    (point["lat"] - pose.lat) * 110_540.0,
                )
                for point in geometry
            ],
            dtype=np.float32,
        )
        if np.min(np.hypot(points[:, 0], points[:, 1])) > reach:
            continue
        yield element.get("tags") or {}, points


def _simplify(points: np.ndarray) -> np.ndarray:
    """Drop the corners that say nothing, more of them the further out it is."""
    span = float(np.median(np.hypot(points[:, 0], points[:, 1])))
    slack = max(BEND_M, span / 500.0)
    shut = len(points) > 3 and bool(np.allclose(points[0], points[-1]))
    kept = cv2.approxPolyDP(points.reshape(-1, 1, 2), slack, shut).reshape(-1, 2)
    if len(kept) < 2:
        return points
    if shut and not np.allclose(kept[0], kept[-1]):
        kept = np.vstack([kept, kept[:1]])
    return kept


def _plan(points) -> list[list[float]]:
    """A line or a ring seen from above, in metres east and north of the eye.

    No heights. They are read off the heightfield when the scene is drawn, so
    that a road lies on exactly the surface the page has under it. Computed
    here instead, they came from a finer reading of the same model and sank
    half a metre into the ground the page had drawn.
    """
    return [[round(float(east), 1), round(float(north), 1)] for east, north in points]


def _roads(ways, terrain: Terrain, eye: float) -> list[dict]:
    """Every way a vehicle or a walker follows, as a centre line and a width."""
    out = []
    for tags, points in ways:
        kind = tags.get("highway")
        if not kind:
            continue
        out.append(
            {
                "k": kind,
                "w": round(road_width_m(tags), 1),
                "r": tags.get("junction") == "roundabout",
                "p": _plan(_simplify(points)),
            }
        )
    return out


def _ribbons(ways, terrain: Terrain, eye: float) -> list[dict]:
    """The flat surfaces drawn corner by corner: car parks, and the island.

    A roundabout comes as a single ring, which is its centre line and not its
    outline. Filled, it gives the whole disc; the carriageway is then laid over
    it as a ribbon, and what stays visible in the middle is the planted island.
    Without it the roundabout is a grey band among grey bands, and the one
    feature everybody recognises here cannot be made out at all.
    """
    out = []
    for tags, points in ways:
        if len(points) < 4 or not np.allclose(points[0], points[-1]):
            continue
        if tags.get("junction") == "roundabout":
            out.append({"k": "island", "p": _plan(_simplify(points))})
        elif not tags.get("highway") and surface_of(tags) in {"parking", "playground"}:
            out.append({"k": surface_of(tags), "p": _plan(_simplify(points))})
    return out


def _buildings(ways, terrain: Terrain, eye: float) -> list[dict]:
    """Footprints lifted to their own height.

    OpenStreetMap rarely gives a height up here, and a chalet guessed at six
    metres is far closer than a chalet left flat on the ground.
    """
    out = []
    for tags, points in ways:
        if surface_of(tags) != "building":
            continue
        if len(points) < 4 or not np.allclose(points[0], points[-1]):
            continue
        if float(np.min(np.hypot(points[:, 0], points[:, 1]))) <= UNDERFOOT_M:
            continue
        out.append({"h": round(_tall(tags), 1), "p": _plan(_simplify(points))})
    return out


def _tall(tags: dict) -> float:
    try:
        return float(str(tags.get("height") or "").split()[0])
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return float(tags["building:levels"]) * STOREY_M + ROOF_M
    except (KeyError, TypeError, ValueError):
        return PLAIN_BUILDING_M


def _woods(ways) -> list[dict]:
    """The wooded parcels, as outlines.

    Only the boundary travels. The trees inside are sown by the page itself, on
    a fixed pattern: sending a hundred thousand positions over the wire would
    cost more than the shape they fill.
    """
    out = []
    for tags, points in ways:
        if tags.get("highway") or surface_of(tags) != "forest":
            continue
        if len(points) < 4 or not np.allclose(points[0], points[-1]):
            continue
        ring = _simplify(points)
        out.append({"p": [[round(float(east), 1), round(float(north), 1)] for east, north in ring]})
    return out


def _trees(data: dict, pose: Pose, terrain: Terrain, eye: float, reach: float) -> list[list[float]]:
    """The trees standing on their own, which the map has bothered to record.

    On a ski slope a lone tree is a landmark, and one of them is the pine the
    camera keeps finding a walker under.
    """
    scale = math.cos(math.radians(pose.lat))
    out = []
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        if tags.get("natural") != "tree":
            continue
        lat, lon = element.get("lat"), element.get("lon")
        if lat is None or lon is None:
            continue
        east = (lon - pose.lon) * 111_320.0 * scale
        north = (lat - pose.lat) * 110_540.0
        if math.hypot(east, north) > reach:
            continue
        try:
            tall = float(str(tags.get("height") or "").split()[0])
        except (TypeError, ValueError, IndexError):
            tall = 9.0
        out.append([round(east, 1), round(north, 1), round(tall, 1)])
    return out


def _masts(data: dict, pose: Pose, terrain: Terrain, eye: float, reach: float) -> list[dict]:
    """The towers and the aerials, which are the skyline up here."""
    scale = math.cos(math.radians(pose.lat))
    out = []
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        if tags.get("man_made") not in MASTS:
            continue
        spot = element.get("center") or element
        lat, lon = spot.get("lat"), spot.get("lon")
        if lat is None or lon is None:
            continue
        east = (lon - pose.lon) * 111_320.0 * scale
        north = (lat - pose.lat) * 110_540.0
        if math.hypot(east, north) > reach:
            continue
        out.append(
            {
                "name": tags.get("name") or _plain(tags),
                "at": [round(east, 1), round(north, 1)],
                "h": round(_lift(tags) or 12.0, 1),
            }
        )
    return out


def _figures(data: dict, pose: Pose, terrain: Terrain, eye: float, reach: float) -> list[dict]:
    """The carved figures and the memorials, which are what people point at.

    Three metres of wood standing on the island of a roundabout is the one thing
    everybody here recognises, and it reads from any angle, which a painted
    surface does not.
    """
    scale = math.cos(math.radians(pose.lat))
    out = []
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        if tags.get("tourism") != "artwork" and tags.get("historic") != "memorial":
            continue
        spot = element.get("center") or element
        lat, lon = spot.get("lat"), spot.get("lon")
        if lat is None or lon is None:
            continue
        east = (lon - pose.lon) * 111_320.0 * scale
        north = (lat - pose.lat) * 110_540.0
        if math.hypot(east, north) > reach:
            continue
        out.append(
            {
                "name": tags.get("name") or _plain(tags),
                "at": [round(east, 1), round(north, 1)],
                "h": 2.6,
            }
        )
    return out


def _heightfield(terrain: Terrain, eye: float, reach: float) -> dict:
    """The ground itself, as a square of heights in metres above the eye."""
    step = terrain.step_m
    side = int(2 * reach / step) + 1
    grid = []
    for row in range(side):
        north = reach - row * step
        grid.append([round(terrain.height(col * step - reach, north) - eye, 1) for col in range(side)])
    return {"step_m": step, "reach_m": reach, "grid": grid}


def _cover(ways, reach: float) -> dict:
    """What grows on the ground, coarse, one cell per elevation post.

    Only the three broad covers are kept. Everything paved is drawn as a shape
    of its own, at its own width: a road three cells wide in a picture like this
    would be a motorway.
    """
    side = int(2 * reach / COVER_STEP_M) + 1
    image = np.zeros((side, side), dtype=np.uint8)
    for name in ("meadow", "scree", "forest"):
        for tags, points in ways:
            if tags.get("highway") or surface_of(tags) != name:
                continue
            if len(points) < 4 or not np.allclose(points[0], points[-1]):
                continue
            shape = np.array(
                [
                    (
                        int(round((east + reach) / COVER_STEP_M)),
                        int(round((reach - north) / COVER_STEP_M)),
                    )
                    for east, north in points
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [shape], int(BROAD[name]))
    return {"step_m": COVER_STEP_M, "reach_m": reach, "grid": [row.tolist() for row in image]}


if __name__ == "__main__":
    raise SystemExit(main())
