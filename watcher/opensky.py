"""Rolling archive of aircraft over Mont Serein. The file stays on the Pi."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from base64 import b64encode
from pathlib import Path

log = logging.getLogger("ventoux.opensky")


class SkyArchive:
    def __init__(self, path: Path, bbox: list[float], retain_days: int = 14, username: str = "", password: str = "", quiet_s: float = 60.0):
        self.path = path
        self.bbox = bbox
        self.retain_days = retain_days
        self.username = username
        self.password = password
        self.quiet_s = quiet_s
        self.asked = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, when: float, window_s: float = 120) -> list[dict]:
        """Who was flying over, at the moment something crossed the sky.

        OpenSky counts the questions, so only a crossing asks one. A reading
        already in the archive answers for free, and two crossings in the same
        minute share a single call.
        """
        known = self.around(when, window_s)
        if known:
            return known
        now = time.time()
        if now - self.asked < self.quiet_s:
            return []
        self.asked = now
        self.poll(now)
        return self.around(when, window_s)

    def poll(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        lamin, lomin, lamax, lomax = self.bbox
        query = urllib.parse.urlencode(
            {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
        )
        request = urllib.request.Request(
            f"https://opensky-network.org/api/states/all?{query}",
            headers={"User-Agent": "ventoux-watch/0.1"},
        )
        if self.username and self.password:
            token = b64encode(f"{self.username}:{self.password}".encode()).decode()
            request.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode())
        except Exception as exc:
            log.warning("OpenSky indisponible: %s", exc)
            return 0
        aircraft = []
        for state in payload.get("states") or []:
            if state[5] is None or state[6] is None:
                continue
            aircraft.append(
                {
                    "icao24": state[0],
                    "callsign": (state[1] or "").strip(),
                    "lon": state[5],
                    "lat": state[6],
                    "altitude_m": state[7] if state[7] is not None else state[13],
                }
            )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": now, "aircraft": aircraft}) + "\n")
        self._prune(now)
        return len(aircraft)

    def around(self, when: float, window_s: float = 120) -> list[dict]:
        if not self.path.is_file():
            return []
        closest: dict[str, tuple[float, dict]] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if abs(row["t"] - when) > window_s:
                    continue
                for aircraft in row["aircraft"]:
                    icao = aircraft.get("icao24") or ""
                    gap = abs(row["t"] - when)
                    previous = closest.get(icao)
                    if previous is None or gap < previous[0]:
                        closest[icao] = (gap, aircraft)
        return [item[1] for item in closest.values()]

    def _prune(self, now: float) -> None:
        if not self.path.is_file():
            return
        cutoff = now - self.retain_days * 86400
        kept = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("t", 0) >= cutoff:
                    kept.append(line if line.endswith("\n") else line + "\n")
        self.path.write_text("".join(kept), encoding="utf-8")
