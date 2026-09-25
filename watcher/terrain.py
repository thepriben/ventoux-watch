"""The shape of the ground around a camera.

A regular grid of elevations, asked once to a public service and kept on disk.
It is coarse — about ninety metres between samples — which is useless for
placing a kerb but right for knowing where a mountainside stands a kilometre
away. Reading it along the line of sight, rather than projecting map points
into the frame, keeps that coarseness harmless: an error of a few metres moves
the answer along the ray instead of throwing it across the picture.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

AGENT = "ventoux-watch/0.1 (github.com/thepriben/ventoux-watch)"
# Twenty-five metres across Europe, then thirty worldwide, then a ninety-metre
# grid as a last resort. The first that answers wins.
SOURCES = (
    "https://api.opentopodata.org/v1/eudem25m",
    "https://api.opentopodata.org/v1/srtm30m",
    "https://api.open-meteo.com/v1/elevation",
)
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0


class Terrain:
    def __init__(self, lat: float, lon: float, reach_m: float, step_m: float = 40.0, cache: Path | None = None):
        self.lat = lat
        self.lon = lon
        self.reach_m = reach_m
        self.step_m = step_m
        self.cache = cache
        self.side = int(round(2 * reach_m / step_m)) + 1
        self.grid = np.full((self.side, self.side), np.nan, dtype=np.float64)
        self.apron_m = 0.0
        self.apron_level = 0.0
        self.bias = 0.0
        # The grid is anchored on the position declared for the camera. Fitting
        # the view moves the camera a few metres; the ground does not move, so
        # the offset is carried here and the cache keeps its meaning.
        self.shift = (0.0, 0.0)
        if cache is not None and cache.is_file():
            self._read(cache)

    def anchor(self, lat: float, lon: float) -> None:
        """Say where the eye really is, relative to the grid."""
        east = (lon - self.lon) * M_PER_DEG_LON * math.cos(math.radians(self.lat))
        north = (lat - self.lat) * M_PER_DEG_LAT
        self.shift = (east, north)

    def level(self, radius_m: float, height: float) -> None:
        """Hold the ground flat close in, where the grid is far too coarse."""
        self.apron_m = radius_m
        self.apron_level = height

    def settle(self, marks: list[dict]) -> float:
        """Slide the whole grid until it agrees with the surveyed landmarks.

        A public elevation model is faithful about the shape of a slope and
        careless about its absolute height; a few metres of offset at the foot
        of the camera is enough to make every near ray hit at once. The marks
        used to fit the view are the one thing known for certain here, so they
        set the level and the model keeps only its relief.
        """
        self.bias = 0.0
        gaps = []
        for mark in marks:
            east = (mark["lon"] - self.lon) * M_PER_DEG_LON * math.cos(math.radians(self.lat))
            north = (mark["lat"] - self.lat) * M_PER_DEG_LAT
            gaps.append(float(mark["ele"]) - self.height(east - self.shift[0], north - self.shift[1]))
        gaps.sort()
        self.bias = gaps[len(gaps) // 2] if gaps else 0.0
        return self.bias

    def coords(self, row: int, col: int) -> tuple[float, float]:
        north = self.reach_m - row * self.step_m
        east = col * self.step_m - self.reach_m
        return self.lat + north / M_PER_DEG_LAT, self.lon + east / (M_PER_DEG_LON * math.cos(math.radians(self.lat)))

    def missing(self) -> list[tuple[int, int]]:
        rows, cols = np.nonzero(np.isnan(self.grid))
        return list(zip(rows.tolist(), cols.tolist()))

    def build(self, batch: int = 100, pause_s: float = 1.2, say=None, budget_s: float = 0.0) -> int:
        """Fill the holes. The service throttles, so a refusal is waited out and
        the batch tried again; what was already learnt is written either way.
        With a budget, it stops on time and the next call picks up where it
        left off."""
        holes = self.missing()
        added = 0
        until = time.time() + budget_s if budget_s else 0.0
        for start in range(0, len(holes), batch):
            if until and time.time() > until:
                break
            chunk = holes[start : start + batch]
            spots = [self.coords(row, col) for row, col in chunk]
            values = None
            for attempt in range(5):
                values = _ask(spots)
                if values is not None:
                    break
                time.sleep(10 * (attempt + 1))
            if values is None:
                break
            for (row, col), value in zip(chunk, values):
                self.grid[row, col] = value
                added += 1
            if say is not None:
                say(added, len(holes))
            if self.cache is not None and added % (batch * 10) == 0:
                self._write(self.cache)
            time.sleep(pause_s)
        if added and self.cache is not None:
            self._write(self.cache)
        return added

    def height(self, east: float, north: float) -> float:
        """Ground elevation at a point given in metres from the camera."""
        if self.apron_m and math.hypot(east, north) <= self.apron_m:
            return self.apron_level
        east, north = east + self.shift[0], north + self.shift[1]
        col = (east + self.reach_m) / self.step_m
        row = (self.reach_m - north) / self.step_m
        c0 = min(self.side - 2, max(0, int(col)))
        r0 = min(self.side - 2, max(0, int(row)))
        fc, fr = col - c0, row - r0
        patch = self.grid[r0 : r0 + 2, c0 : c0 + 2]
        if np.isnan(patch).any():
            return float(np.nanmean(self.grid)) if not np.isnan(self.grid).all() else 0.0
        top = patch[0, 0] * (1 - fc) + patch[0, 1] * fc
        bottom = patch[1, 0] * (1 - fc) + patch[1, 1] * fc
        return float(top * (1 - fr) + bottom * fr) + self.bias

    def _read(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        same = (
            abs(payload.get("lat", 0) - self.lat) < 1e-6
            and abs(payload.get("lon", 0) - self.lon) < 1e-6
            and payload.get("reach_m") == self.reach_m
            and payload.get("step_m") == self.step_m
        )
        if not same:
            return
        grid = np.array(
            [[np.nan if value is None else float(value) for value in row] for row in payload["grid"]],
            dtype=np.float64,
        )
        if grid.shape == self.grid.shape:
            self.grid = grid

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "lat": self.lat,
                    "lon": self.lon,
                    "reach_m": self.reach_m,
                    "step_m": self.step_m,
                    "grid": [[None if math.isnan(value) else round(value, 1) for value in row] for row in self.grid],
                }
            ),
            encoding="utf-8",
        )


def _ask(spots: list[tuple[float, float]]) -> list[float] | None:
    for source in SOURCES:
        if "opentopodata" in source:
            query = f"{source}?locations={'|'.join(f'{lat:.5f},{lon:.5f}' for lat, lon in spots)}"
            read = lambda payload: [point["elevation"] for point in payload["results"]]
        else:
            query = (
                f"{source}?latitude={','.join(f'{lat:.5f}' for lat, _lon in spots)}"
                f"&longitude={','.join(f'{lon:.5f}' for _lat, lon in spots)}"
            )
            read = lambda payload: payload["elevation"]
        try:
            request = urllib.request.Request(query, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=45) as response:
                values = read(json.loads(response.read().decode()))
            if values and all(value is not None for value in values):
                return [float(value) for value in values]
        except Exception:
            continue
    return None
