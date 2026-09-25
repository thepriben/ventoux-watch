"""Where a point on the ground lands in the picture.

A webcam is published in OpenStreetMap with a position, an elevation and a
direction. That is enough to place the world in the frame, once the pitch and
the real field of view are fitted on a few landmarks that can be pointed at in
both the map and the image. The same fit is meant to run for every camera, so
nothing here is specific to Mont Serein.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_M_PER_DEG_LAT = 110_540.0
EARTH_M_PER_DEG_LON = 111_320.0


@dataclass
class Pose:
    lat: float
    lon: float
    ele: float
    yaw: float
    pitch: float = 0.0
    hfov: float = 90.0
    height_m: float = 4.0
    aspect: float = 16 / 9

    def as_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "ele": self.ele,
            "yaw": round(self.yaw, 3),
            "pitch": round(self.pitch, 3),
            "hfov": round(self.hfov, 3),
            "height_m": round(self.height_m, 2),
        }


def enu(pose: Pose, lat: float, lon: float, ele: float) -> tuple[float, float, float]:
    """East, north and up in metres, from the camera."""
    east = (lon - pose.lon) * EARTH_M_PER_DEG_LON * math.cos(math.radians(pose.lat))
    north = (lat - pose.lat) * EARTH_M_PER_DEG_LAT
    up = ele - (pose.ele + pose.height_m)
    return east, north, up


def project(pose: Pose, lat: float, lon: float, ele: float) -> tuple[float, float] | None:
    """Normalized image coordinates, or None when the point is behind the camera."""
    east, north, up = enu(pose, lat, lon, ele)
    yaw = math.radians(pose.yaw)
    pitch = math.radians(pose.pitch)
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    flat = (math.sin(yaw), math.cos(yaw), 0.0)
    forward = (
        flat[0] * math.cos(pitch),
        flat[1] * math.cos(pitch),
        -math.sin(pitch),
    )
    upward = (
        flat[0] * math.sin(pitch),
        flat[1] * math.sin(pitch),
        math.cos(pitch),
    )
    point = (east, north, up)
    depth = _dot(point, forward)
    if depth <= 1e-6:
        return None
    half = math.tan(math.radians(pose.hfov) / 2)
    x = (_dot(point, right) / depth) / half
    y = (_dot(point, upward) / depth) / half * pose.aspect
    return 0.5 + x / 2, 0.5 - y / 2


def distance_m(pose: Pose, lat: float, lon: float) -> float:
    east, north, _up = enu(pose, lat, lon, pose.ele)
    return math.hypot(east, north)


def fit(pose: Pose, marks: list[dict], rounds: int = 5) -> tuple[Pose, float]:
    """Nudge yaw, pitch and field of view until the landmarks fall in place.

    `marks` are dicts with lat, lon, ele and the normalized x, y read off the
    picture. A coarse-to-fine sweep is enough: three unknowns, and the start is
    already the direction declared in OpenStreetMap.
    """
    best = pose
    step = {"yaw": 12.0, "pitch": 12.0, "hfov": 20.0}
    score = _residual(best, marks)
    for _ in range(rounds):
        for _sweep in range(24):
            moved = False
            for key in ("yaw", "pitch", "hfov"):
                for way in (1, -1):
                    trial = _shift(best, key, way * step[key])
                    value = _residual(trial, marks)
                    if value < score - 1e-9:
                        best, score, moved = trial, value, True
            if not moved:
                break
        step = {key: value / 3 for key, value in step.items()}
    return best, score


def _shift(pose: Pose, key: str, delta: float) -> Pose:
    values = {
        "lat": pose.lat,
        "lon": pose.lon,
        "ele": pose.ele,
        "yaw": pose.yaw,
        "pitch": pose.pitch,
        "hfov": pose.hfov,
        "height_m": pose.height_m,
        "aspect": pose.aspect,
    }
    values[key] = values[key] + delta
    if key == "hfov":
        values[key] = max(15.0, min(150.0, values[key]))
    if key == "pitch":
        values[key] = max(-45.0, min(45.0, values[key]))
    return Pose(**values)


def _residual(pose: Pose, marks: list[dict]) -> float:
    total = 0.0
    for mark in marks:
        seen = project(pose, mark["lat"], mark["lon"], mark["ele"])
        if seen is None:
            return 1e6
        total += (seen[0] - mark["x"]) ** 2 + (seen[1] - mark["y"]) ** 2
    return total


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
