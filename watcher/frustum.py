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


def axes(pose: Pose) -> tuple[tuple, tuple, tuple]:
    """Right, forward and up of the camera, in the east-north-up frame."""
    yaw = math.radians(pose.yaw)
    pitch = math.radians(pose.pitch)
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    flat = (math.sin(yaw), math.cos(yaw), 0.0)
    forward = (flat[0] * math.cos(pitch), flat[1] * math.cos(pitch), -math.sin(pitch))
    upward = (flat[0] * math.sin(pitch), flat[1] * math.sin(pitch), math.cos(pitch))
    return right, forward, upward


def ray(pose: Pose, sx: float, sy: float) -> tuple[float, float, float]:
    """The direction the camera looks at this point of the picture."""
    right, forward, upward = axes(pose)
    half = math.tan(math.radians(pose.hfov) / 2)
    nx = (sx - 0.5) * 2 * half
    ny = (0.5 - sy) * 2 * half / pose.aspect
    vector = tuple(forward[i] + nx * right[i] + ny * upward[i] for i in range(3))
    length = math.sqrt(sum(value * value for value in vector))
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def march(pose: Pose, sx: float, sy: float, terrain, reach_m: float, near_m: float = 3.0):
    """Follow the line of sight until it meets the ground.

    Returns the spot in metres east and north of the camera, and how far it is.
    None means the line of sight leaves over the skyline: that is sky.
    """
    east_d, north_d, up_d = ray(pose, sx, sy)
    eye = pose.ele + pose.height_m
    distance = near_m
    previous = None
    while distance < reach_m:
        east, north = east_d * distance, north_d * distance
        gap = (eye + up_d * distance) - terrain.height(east, north)
        if gap <= 0:
            if previous is not None:
                low, high = previous, distance
                for _ in range(24):
                    middle = (low + high) / 2
                    probe = (eye + up_d * middle) - terrain.height(east_d * middle, north_d * middle)
                    if probe <= 0:
                        high = middle
                    else:
                        low = middle
                distance = high
            return east_d * distance, north_d * distance, distance
        previous = distance
        distance *= 1.02
        distance += 0.5
    return None


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
