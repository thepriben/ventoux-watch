"""Rolling archive of aircraft over Mont Serein. The file stays on the Pi."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from base64 import b64encode
from pathlib import Path

AGENT = "ventoux-watch/0.3 (github.com/thepriben/ventoux-watch)"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

log = logging.getLogger("ventoux.opensky")


class SkyArchive:
    def __init__(self, path: Path, bbox: list[float], retain_days: int = 14, username: str = "", password: str = "",
                 quiet_s: float = 60.0, client_id: str = "", client_secret: str = ""):
        self.path = path
        self.bbox = bbox
        self.retain_days = retain_days
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.quiet_s = quiet_s
        self.asked = 0.0
        self.token = ""
        self.token_until = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _bearer(self) -> str:
        """A fresh access token, kept until shortly before it expires.

        OpenSky closed the door on name and password in 2025: an account now
        issues a client identifier and a secret, and those buy a token that
        lasts half an hour. Without one the archive still works, on the handful
        of anonymous calls a day the service allows before it says no.
        """
        if not (self.client_id and self.client_secret):
            return ""
        if self.token and time.time() < self.token_until:
            return self.token
        body = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
        ).encode()
        request = urllib.request.Request(TOKEN_URL, data=body, headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode())
        except Exception as exc:
            log.warning("OpenSky refuse le jeton: %s", exc)
            return ""
        self.token = payload.get("access_token") or ""
        self.token_until = time.time() + float(payload.get("expires_in") or 1800) - 60
        return self.token

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
            headers={"User-Agent": AGENT},
        )
        bearer = self._bearer()
        if bearer:
            request.add_header("Authorization", f"Bearer {bearer}")
        elif self.username and self.password:
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
