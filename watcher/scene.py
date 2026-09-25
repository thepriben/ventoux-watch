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


def luminance(frame: np.ndarray | None) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) / 255.0
