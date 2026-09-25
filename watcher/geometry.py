"""Point-in-polygon helpers. Coordinates are normalized between 0 and 1."""

from __future__ import annotations


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def assign_zone(x: float, y: float, zones: dict) -> str:
    for name in zones["priority"]:
        if point_in_polygon(x, y, zones["polygons"][name]):
            return name
    return "other"


def load_zones(path: str) -> dict:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
