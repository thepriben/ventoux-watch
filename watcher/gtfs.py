"""Theoretical buses near the camera, from GTFS kept on the Pi."""

from __future__ import annotations

import csv
import io
import logging
import math
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from watcher.naming import Trip

log = logging.getLogger("ventoux.gtfs")
PARIS = ZoneInfo("Europe/Paris")


class GtfsIndex:
    def __init__(self, root: Path, feeds: list[dict], lat: float, lon: float, radius_m: float = 3000):
        self.root = root
        self.feeds = feeds
        self.lat = lat
        self.lon = lon
        self.radius_m = radius_m
        self.rows: list[dict] = []
        self.refreshed_at = 0.0

    def refresh(self, force: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = []
        for feed in self.feeds:
            folder = self.root / feed["name"]
            archive = folder.with_suffix(".zip")
            fresh = archive.is_file() and time.time() - archive.stat().st_mtime < 20 * 3600
            if force or not fresh:
                self._download(feed["url"], archive, folder)
            if folder.is_dir():
                self.rows.extend(load_feed(folder, feed["name"], self.lat, self.lon, self.radius_m))
        self.refreshed_at = time.time()
        log.info("GTFS: %s passages indexés près de la caméra", len(self.rows))

    def trips_at(self, when: datetime, window_min: int = 15) -> list[Trip]:
        if when.tzinfo is None:
            when = when.replace(tzinfo=PARIS)
        local = when.astimezone(PARIS)
        found: list[Trip] = []
        seen: set[tuple] = set()
        for offset in (0, -1):
            day = local.date() + timedelta(days=offset)
            seconds = local.hour * 3600 + local.minute * 60 + local.second - offset * 86400
            for row in self.rows:
                if not service_runs(row["service"], day):
                    continue
                if abs(row["seconds"] - seconds) > window_min * 60:
                    continue
                key = (row["source"], row["trip_id"], row["stop_id"])
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    Trip(
                        route=row["route"],
                        headsign=row["headsign"],
                        stop_name=row["stop_name"],
                        scheduled=_clock(row["seconds"]),
                        source=row["source"],
                    )
                )
        return found

    def _download(self, url: str, archive: Path, folder: Path) -> None:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ventoux-watch/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            archive.write_bytes(payload)
            if folder.exists():
                for child in folder.iterdir():
                    if child.is_file():
                        child.unlink()
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
                for name in bundle.namelist():
                    if name.endswith("/"):
                        continue
                    target = folder / Path(name).name
                    target.write_bytes(bundle.read(name))
        except Exception as exc:
            log.warning("GTFS %s indisponible: %s", url, exc)


def load_feed(folder: Path, source: str, lat: float, lon: float, radius_m: float) -> list[dict]:
    if not (folder / "stops.txt").is_file() or not (folder / "stop_times.txt").is_file() or not (folder / "trips.txt").is_file():
        return []
    stops = {}
    stops_path = folder / "stops.txt"
    for row in _csv(stops_path):
        try:
            stop_lat = float(row["stop_lat"])
            stop_lon = float(row["stop_lon"])
        except (KeyError, ValueError):
            continue
        if haversine(lat, lon, stop_lat, stop_lon) <= radius_m:
            stops[row["stop_id"]] = row.get("stop_name") or row["stop_id"]
    if not stops:
        return []
    services = _services(folder)
    trips = {}
    for row in _csv(folder / "trips.txt"):
        trips[row["trip_id"]] = row
    routes = {row["route_id"]: row for row in _csv(folder / "routes.txt")} if (folder / "routes.txt").is_file() else {}
    indexed = []
    for row in _csv(folder / "stop_times.txt"):
        if row.get("stop_id") not in stops:
            continue
        trip = trips.get(row.get("trip_id", ""))
        if not trip:
            continue
        route = routes.get(trip.get("route_id", ""), {})
        indexed.append(
            {
                "source": source,
                "trip_id": row["trip_id"],
                "stop_id": row["stop_id"],
                "stop_name": stops[row["stop_id"]],
                "seconds": _seconds(row.get("departure_time") or row.get("arrival_time") or ""),
                "route": route.get("route_short_name") or route.get("route_long_name") or trip.get("route_id", ""),
                "headsign": trip.get("trip_headsign") or "",
                "service": services.get(trip.get("service_id", ""), {"days": set(), "start": "", "end": "", "added": set(), "removed": set()}),
            }
        )
    return indexed


def service_runs(service: dict, day) -> bool:
    stamp = day.strftime("%Y%m%d")
    if stamp in service["removed"]:
        return False
    if stamp in service["added"]:
        return True
    if service["start"] and stamp < service["start"]:
        return False
    if service["end"] and stamp > service["end"]:
        return False
    return day.weekday() in service["days"]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _services(folder: Path) -> dict:
    services: dict[str, dict] = {}
    calendar = folder / "calendar.txt"
    if calendar.is_file():
        names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for row in _csv(calendar):
            days = {index for index, name in enumerate(names) if row.get(name) == "1"}
            services[row["service_id"]] = {
                "days": days,
                "start": row.get("start_date", ""),
                "end": row.get("end_date", ""),
                "added": set(),
                "removed": set(),
            }
    dates = folder / "calendar_dates.txt"
    if dates.is_file():
        for row in _csv(dates):
            service = services.setdefault(
                row["service_id"],
                {"days": set(), "start": "", "end": "", "added": set(), "removed": set()},
            )
            if row.get("exception_type") == "1":
                service["added"].add(row["date"])
            elif row.get("exception_type") == "2":
                service["removed"].add(row["date"])
    return services


def _csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        return -10**9
    hour, minute, second = (int(part) for part in parts)
    return hour * 3600 + minute * 60 + second


def _clock(seconds: int) -> str:
    hour, rem = divmod(seconds, 3600)
    minute, second = divmod(rem, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"
