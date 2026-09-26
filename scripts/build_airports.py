"""Build the airport table the history uses to read a route out loud.

OpenSky answers a route in ICAO codes: KPHL, LFMN, EGLL. Those are exact and
unreadable. This turns the public OurAirports list into the small part of it
this project needs — the airports an airliner over Mont Serein could plausibly
have left or be heading for — so the history can say Philadelphia instead.

Only airports with scheduled service are kept. The rest are airstrips, heliports
and private fields that no callsign in this sky will ever have come from, and
carrying eighty thousand of them to name a few hundred would be silly.

    python scripts/build_airports.py
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

SOURCE = "https://davidmegginson.github.io/ourairports-data/airports.csv"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "config" / "airports.json"


def fetch() -> list[dict]:
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        text = response.read().decode("utf-8", "replace")
    return list(csv.DictReader(text.splitlines()))


def main() -> int:
    rows = fetch()
    table = {}
    for row in rows:
        code = (row.get("icao_code") or row.get("ident") or "").strip().upper()
        # Four letters, scheduled flights, and a town to name it by. An airport
        # missing any of those cannot be turned into a sentence anybody reads.
        if len(code) != 4 or not code.isalpha():
            continue
        if row.get("scheduled_service") != "yes":
            continue
        town = (row.get("municipality") or "").strip()
        name = (row.get("name") or "").strip()
        if not (town or name):
            continue
        table[code] = {
            "town": town or name,
            "name": name,
            "country": (row.get("iso_country") or "").strip(),
        }
    OUT.write_text(json.dumps(table, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"{len(table)} aéroports desservis écrits dans {OUT.relative_to(ROOT)} ({size:.0f} ko)")
    for probe in ("KPHL", "LFMN", "LFML", "EGLL", "LFPG"):
        found = table.get(probe)
        print(f"   {probe} -> {found['town'] if found else '(absent)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
