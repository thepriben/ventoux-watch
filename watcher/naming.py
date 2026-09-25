"""Name a motion before it is allowed into the public history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    cls: str
    conf: float


@dataclass
class Trip:
    route: str
    headsign: str
    stop_name: str
    scheduled: str
    source: str


@dataclass
class Observation:
    zone: str = ""
    detections: list[Detection] = field(default_factory=list)
    aircraft: list[dict] = field(default_factory=list)
    trips: list[Trip] = field(default_factory=list)
    travel: float = 0.0
    area_ratio: float = 0.0
    duration_s: float = 0.0
    warm_ratio: float = 0.0
    area_grow: float = 1.0
    person_count: int = 0
    kind: str = "track"
    min_travel: float = 0.03
    max_sky_area: float = 0.02
    min_conf: dict | None = None
    crowd_min: int = 4
    fire_sustain_s: float = 20.0
    fire_grow: float = 1.5
    fire_warm: float = 0.08


@dataclass
class Decision:
    action: str
    type: str = ""
    label: str = ""
    reason: str = ""
    detail: dict = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def publish(self) -> bool:
        return self.action == "publish"


def choose_aircraft(aircraft: list[dict]) -> tuple[dict | None, str]:
    """Name one aircraft, or none when the sky is empty or ambiguous."""
    by_icao: dict[str, dict] = {}
    for item in aircraft:
        icao = str(item.get("icao24") or "").strip().lower()
        if not icao:
            continue
        altitude = item.get("altitude_m")
        if not isinstance(altitude, (int, float)):
            altitude = None
        by_icao[icao] = {
            "icao24": icao,
            "callsign": str(item.get("callsign") or "").strip(),
            "altitude_m": altitude,
        }
    items = list(by_icao.values())
    if not items:
        return None, "none"
    if len(items) == 1:
        return items[0], "unique"
    measured = [item for item in items if item["altitude_m"] is not None]
    if len(measured) == len(items) and len(measured) >= 2:
        measured.sort(key=lambda item: item["altitude_m"])
        lowest, nxt = measured[0], measured[1]
        if nxt["altitude_m"] > 0 and lowest["altitude_m"] < 0.5 * nxt["altitude_m"]:
            return lowest, "much_lower"
    return None, "ambiguous"


def _best(detections: list[Detection], names: set[str]) -> Detection | None:
    found = [item for item in detections if item.cls in names]
    if not found:
        return None
    return max(found, key=lambda item: item.conf)


def _aircraft_label(aircraft: dict) -> str:
    return aircraft["callsign"] or aircraft["icao24"].upper()


def decide(obs: Observation) -> Decision:
    conf = obs.min_conf or {"bus": 0.45, "bus_unnamed": 0.6, "car": 0.4}
    if obs.kind == "crowd":
        if obs.person_count >= obs.crowd_min:
            return Decision(
                "publish",
                "crowd",
                "Attroupement",
                reason="persons",
                detail={"persons": obs.person_count},
                confidence=1.0,
            )
        return Decision("hold", reason="crowd_below_threshold", detail={"persons": obs.person_count})

    if (
        obs.zone == "slope"
        and obs.duration_s >= obs.fire_sustain_s
        and obs.area_grow >= obs.fire_grow
        and obs.warm_ratio >= obs.fire_warm
    ):
        return Decision(
            "publish",
            "fire",
            "Incendie",
            reason="warm_growing",
            detail={"warm_ratio": round(obs.warm_ratio, 3), "grow": round(obs.area_grow, 2)},
            confidence=min(0.99, obs.warm_ratio),
        )

    if obs.zone == "sky":
        if obs.travel < obs.min_travel or obs.area_ratio > obs.max_sky_area:
            return Decision("hold", reason="sky_not_a_transit", detail={"travel": obs.travel, "area": obs.area_ratio})
        chosen, why = choose_aircraft(obs.aircraft)
        if chosen is None:
            return Decision("hold", reason=why, detail={"aircraft": obs.aircraft})
        return Decision(
            "publish",
            "plane",
            _aircraft_label(chosen),
            reason=why,
            detail=chosen,
            confidence=0.9 if why == "unique" else 0.7,
        )

    if obs.zone in {"road", "roundabout"}:
        if obs.travel < obs.min_travel:
            return Decision("hold", reason="static", detail={"travel": obs.travel})
        bus = _best(obs.detections, {"bus"})
        vehicle = _best(obs.detections, {"car", "truck"})
        if bus is not None and bus.conf >= conf["bus"]:
            if len(obs.trips) == 1:
                trip = obs.trips[0]
                return Decision(
                    "publish",
                    "bus",
                    f"Bus {trip.route}",
                    reason="schedule",
                    detail={
                        "route": trip.route,
                        "headsign": trip.headsign,
                        "stop": trip.stop_name,
                        "scheduled": trip.scheduled,
                        "source": trip.source,
                    },
                    confidence=bus.conf,
                )
            if bus.conf >= conf["bus_unnamed"]:
                return Decision(
                    "publish",
                    "bus",
                    "Bus",
                    reason="model_only",
                    detail={"trips": [trip.__dict__ for trip in obs.trips]},
                    confidence=bus.conf,
                )
            return Decision("hold", reason="bus_uncertain", confidence=bus.conf)
        if vehicle is not None and vehicle.conf >= conf["car"]:
            label = "Camion" if vehicle.cls == "truck" else "Voiture"
            return Decision("publish", "car", label, reason=vehicle.cls, detail={}, confidence=vehicle.conf)
        return Decision("hold", reason="unnamed_vehicle")

    return Decision("hold", reason="unclassified", detail={"zone": obs.zone})
