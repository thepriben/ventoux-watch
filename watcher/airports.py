"""Turning an airport code into the name of a place.

OpenSky answers a route in four-letter ICAO codes. KPHL is exact and says
nothing; Philadelphia says everything. The table is built once by
scripts/build_airports.py from the public OurAirports list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("ventoux.airports")

_TABLE: dict[str, dict] | None = None


def _table() -> dict[str, dict]:
    global _TABLE
    if _TABLE is None:
        path = Path(__file__).resolve().parent.parent / "config" / "airports.json"
        try:
            _TABLE = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # A missing table costs a route on an event, nothing more. The
            # watcher has never needed it to see anything.
            log.warning("Table des aéroports illisible: %s", exc)
            _TABLE = {}
    return _TABLE


def town(code: str) -> str:
    """The town an airport serves, or the bare code when it is not in the table."""
    code = str(code or "").strip().upper()
    if not code:
        return ""
    found = _table().get(code)
    return (found or {}).get("town") or code


def describe_route(route: dict) -> dict:
    """A route with its two ends named, ready to be written into an event."""
    out = {}
    for end in ("from", "to"):
        code = str(route.get(end) or "").strip().upper()
        if code:
            out[end] = code
            out[f"{end}_town"] = town(code)
    return out
