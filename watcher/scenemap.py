"""The picture, told as surfaces.

`scene.json` holds a small grid: for every cell of the frame, what is on the
ground there — roadway, roundabout, car park, path, meadow, forest, building,
sky. Landmarks that never move, a wooden statue for instance, are listed apart.
The grid is built from OpenStreetMap by `scripts/build_scene.py`, so a new
camera only needs its position and its direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

LETTERS = {
    ".": "",
    "r": "road",
    "o": "roundabout",
    "p": "parking",
    "t": "path",
    "m": "meadow",
    "f": "forest",
    "b": "building",
    "s": "sky",
    "e": "scree",
}
CODES = {name: letter for letter, name in LETTERS.items() if name}

DRIVABLE = {"road", "roundabout", "parking"}
WALKABLE = DRIVABLE | {"path"}
FLAMMABLE = {"forest", "meadow", "scree"}


class SceneMap:
    def __init__(self, payload: dict | None = None):
        payload = payload or {}
        self.rows: list[str] = list(payload.get("grid") or [])
        self.landmarks: list[dict] = list(payload.get("landmarks") or [])
        self.pose: dict = dict(payload.get("pose") or {})
        self.reach: list[list[int]] = list(payload.get("reach") or [])
        self.height = len(self.rows)
        self.width = len(self.rows[0]) if self.rows else 0

    @classmethod
    def load(cls, path: Path) -> "SceneMap":
        if not path.is_file():
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return cls()

    @property
    def ready(self) -> bool:
        return self.width > 0 and self.height > 0

    def surface_at(self, x: float, y: float) -> str:
        """Normalized coordinates in, surface name out."""
        if not self.ready:
            return ""
        column = min(self.width - 1, max(0, int(x * self.width)))
        row = min(self.height - 1, max(0, int(y * self.height)))
        return LETTERS.get(self.rows[row][column], "")

    def surface_under(self, box: tuple[float, float, float, float]) -> str:
        """What a detection stands on: the bottom edge of its box, not its middle.

        A car is read at its wheels, a walker at their feet. Taking the centre
        would put a tall subject in the trees behind it.
        """
        if not self.ready:
            return ""
        x, y, w, h = box
        foot = min(0.999, y + h)
        votes: dict[str, int] = {}
        for step in range(5):
            sample = x + w * (step / 4 if w else 0)
            name = self.surface_at(min(0.999, max(0.0, sample)), foot)
            if name:
                votes[name] = votes.get(name, 0) + 1
        if not votes:
            return ""
        return max(votes, key=lambda key: (votes[key], key in DRIVABLE))

    def distance_at(self, x: float, y: float) -> float:
        """How far the ground is at this point of the picture, in metres."""
        if not self.reach:
            return 0.0
        rows, columns = len(self.reach), len(self.reach[0])
        row = min(rows - 1, max(0, int(y * rows)))
        column = min(columns - 1, max(0, int(x * columns)))
        return float(self.reach[row][column])

    def metres_across(self, box: tuple[float, float, float, float]) -> float:
        """The width of this box on the ground, in metres.

        A hundred pixels are a metre at the roundabout and thirty metres on the
        far slope. Without this, a plume and a parked van look the same size.
        """
        x, y, w, h = box
        span = self.distance_at(min(0.999, x + w / 2), min(0.999, y + h))
        hfov = float(self.pose.get("hfov") or 0)
        if span <= 0 or hfov <= 0:
            return 0.0
        return w * 2 * span * math.tan(math.radians(hfov) / 2)

    def landmark_at(self, box: tuple[float, float, float, float]) -> dict | None:
        """A fixed thing of the map that this box is drawn around.

        The box has to be about the size of the landmark. A wide box that
        happens to contain the statue is a passage in front of it, not the
        statue being mistaken for someone.
        """
        x, y, w, h = box
        for mark in self.landmarks:
            mx, my = mark.get("x", -1), mark.get("y", -1)
            reach = float(mark.get("r") or 0.02)
            if w > 5 * reach or h > 7 * reach:
                continue
            if x - reach <= mx <= x + w + reach and y - reach <= my <= y + h + reach:
                return mark
        return None
