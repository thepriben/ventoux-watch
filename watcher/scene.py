"""Day, night and weather for the Mont Serein frame.

The same pixels are not the same event at noon, at dusk, or in fog.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("ventoux.scene")


@dataclass
class Scene:
    period: str
    weather: str
    temperature_c: float | None = None
    clouds: int | None = None
    luminance: float = 0.0

    @property
    def context(self) -> str:
        period = {"day": "de jour", "twilight": "au crépuscule", "night": "de nuit"}.get(self.period, "")
        parts = [part for part in (period, self.weather) if part]
        return ", ".join(parts)


class SceneReader:
    def __init__(self, lat: float, lon: float, refresh_s: int = 600):
        self.lat = lat
        self.lon = lon
        self.refresh_s = refresh_s
        self._weather = ""
        self._temperature: float | None = None
        self._clouds: int | None = None
        self._fetched = 0.0

    def read(self, frame: np.ndarray | None, when: datetime) -> Scene:
        moment = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        if time.time() - self._fetched > self.refresh_s:
            self._refresh()
        return Scene(
            period=solar_period(moment, self.lat, self.lon),
            weather=self._weather,
            temperature_c=self._temperature,
            clouds=self._clouds,
            luminance=luminance(frame),
        )

    def _refresh(self) -> None:
        query = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={self.lat}&longitude={self.lon}"
            "&current=temperature_2m,weather_code,cloud_cover"
            "&timezone=Europe%2FParis"
        )
        try:
            request = urllib.request.Request(query, headers={"User-Agent": "ventoux-watch/0.1"})
            with urllib.request.urlopen(request, timeout=15) as response:
                current = json.loads(response.read().decode()).get("current") or {}
            code = int(current.get("weather_code", -1))
            self._weather = weather_label(code)
            self._temperature = current.get("temperature_2m")
            self._clouds = current.get("cloud_cover")
        except Exception as exc:
            log.warning("Météo indisponible: %s", exc)
        self._fetched = time.time()


def solar_period(when: datetime, lat: float, lon: float) -> str:
    elevation = solar_elevation(when.astimezone(timezone.utc), lat, lon)
    if elevation >= 6:
        return "day"
    if elevation >= -6:
        return "twilight"
    return "night"


def solar_azimuth(when: datetime, lat: float, lon: float) -> float:
    """Which way the sun is, in degrees clockwise from north. `when` is UTC.

    Worth having because a low sun through the trees is warm, wide and growing,
    which is every sign of a fire but one: it is exactly where the sun is.
    """
    moment = when.astimezone(timezone.utc)
    day = moment.timetuple().tm_yday
    hour = moment.hour + moment.minute / 60 + moment.second / 3600
    gamma = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    hour_angle = math.radians((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    lat_r = math.radians(lat)
    elevation = math.radians(solar_elevation(moment, lat, lon))
    east = -math.sin(hour_angle) * math.cos(decl)
    north = math.sin(decl) * math.cos(lat_r) - math.cos(hour_angle) * math.cos(decl) * math.sin(lat_r)
    if abs(math.cos(elevation)) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(east, north)) % 360


def solar_elevation(when: datetime, lat: float, lon: float) -> float:
    """NOAA solar elevation in degrees. `when` is UTC."""
    moment = when.astimezone(timezone.utc)
    day = moment.timetuple().tm_yday
    hour = moment.hour + moment.minute / 60 + moment.second / 3600
    gamma = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    true_solar = hour * 60 + eqtime + 4 * lon
    hour_angle = math.radians(true_solar / 4 - 180)
    lat_r = math.radians(lat)
    sine = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sine))))


def weather_label(code: int) -> str:
    if code == 0:
        return "ciel dégagé"
    if code in (1, 2):
        return "peu nuageux"
    if code == 3:
        return "couvert"
    if code in (45, 48):
        return "brouillard"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "pluie"
    if 71 <= code <= 77 or 85 <= code <= 86:
        return "neige"
    if code >= 95:
        return "orage"
    return ""


def moon_in_sky(frame: np.ndarray | None, exclude: list[dict] | None = None) -> bool:
    """A small bright disc, round and far brighter than the sky around it.

    The summit beacon and the valley lamps are excluded by the same circles
    the motion mask uses.
    """
    if frame is None or frame.size == 0:
        return False
    height, width = frame.shape[:2]
    band = cv2.cvtColor(frame[: max(1, int(height * 0.45))], cv2.COLOR_BGR2GRAY)
    level = max(190, int(band.mean()) + 60)
    count, _labels, stats, centers = cv2.connectedComponentsWithStats((band >= level).astype(np.uint8))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        small = max(3, int(round(width * 0.006)))
        large = max(small + 2, int(round(width * 0.06)))
        if not small <= w <= large or not small <= h <= large:
            continue
        if not 0.55 <= w / h <= 1.8 or area / float(w * h) < 0.55:
            continue
        cx, cy = centers[index][0] / width, centers[index][1] / height
        if _inside_circle(cx, cy, exclude or []):
            continue
        spot = band[y : y + h, x : x + w]
        ring = band[max(0, y - 2 * h) : y + 3 * h, max(0, x - 2 * w) : x + 3 * w]
        if ring.size and float(spot.mean()) - float(ring.mean()) >= 45:
            return True
    return False


def _inside_circle(x: float, y: float, circles: list[dict]) -> bool:
    for circle in circles:
        dx = x - float(circle.get("cx", 0))
        dy = y - float(circle.get("cy", 0))
        reach = float(circle.get("r", 0)) * 1.6
        if dx * dx + dy * dy <= reach * reach:
            return True
    return False


def read_sky(frame: np.ndarray | None) -> str:
    """Weather read from the sky band of this camera, not from a station."""
    if frame is None or frame.size == 0:
        return ""
    height = frame.shape[0]
    band = frame[: max(1, int(height * 0.16))]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mean_v = float(val.mean())
    mean_s = float(sat.mean())
    spread = float(val.std())
    blue = float(((hue >= 90) & (hue <= 130) & (sat >= 40)).mean())
    if mean_v < 45:
        return "nuit"
    if mean_s < 28 and spread < 16:
        return "brouillard"
    if blue >= 0.45 and mean_s >= 50:
        return "ciel dégagé"
    if blue >= 0.18 or spread >= 28:
        return "peu nuageux"
    return "couvert"


class ViewLog:
    """The last webcam bulletin, with the frame it was read from."""

    def __init__(self, path: Path, every_s: int = 900, change_s: int = 120, exclude: list[dict] | None = None):
        self.path = path
        self.photo_path = path.parent / "view.jpg"
        self.every_s = every_s
        self.change_s = change_s
        self.exclude = exclude or []
        self.last: dict = {}
        self.dirty = False
        self._votes: list[str] = []
        self._last_commit = 0.0
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.last = dict(payload.get("last") or {})
            except json.JSONDecodeError:
                self.last = {}
        if self.last:
            self._last_commit = _stamp(self.last.get("t", ""))

    def note(self, frame: np.ndarray | None, api_label: str, temp_c: float | None, when: datetime, period: str = "") -> None:
        label = read_sky(frame)
        if not label:
            return
        self._votes.append(label)
        self._votes = self._votes[-8:]
        chosen = max(set(self._votes), key=self._votes.count)
        moment = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        stamp = moment.timestamp()
        changed = self.last.get("webcam") != chosen
        wait = self.change_s if changed else self.every_s
        if self.last and stamp - self._last_commit < wait:
            return
        moon = period in {"twilight", "night"} and moon_in_sky(frame, self.exclude)
        self.last = {
            "t": moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "webcam": chosen,
            "period": period,
            "moon": bool(moon),
            "api": api_label or "",
            "temp_c": None if temp_c is None else round(float(temp_c)),
            "photo": "data/view.jpg",
        }
        self._last_commit = stamp
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if frame is not None and frame.size:
            ok, encoded = cv2.imencode(".jpg", _narrow(frame, 480), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                self.photo_path.write_bytes(encoded.tobytes())
        self.path.write_text(json.dumps({"last": self.last}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.dirty = True


def _narrow(frame: np.ndarray, width: int) -> np.ndarray:
    height, frame_width = frame.shape[:2]
    if frame_width <= width:
        return frame
    scale = width / float(frame_width)
    return cv2.resize(frame, (width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


def _stamp(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def luminance(frame: np.ndarray | None) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) / 255.0
