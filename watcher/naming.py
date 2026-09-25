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
    period: str = "day"
    weather: str = ""


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
            return _stamp(
                Decision(
                    "publish",
                    "crowd",
                    "Attroupement",
                    reason="persons",
                    detail={"persons": obs.person_count},
                    confidence=1.0,
                ),
                obs,
            )
        return Decision("hold", reason="crowd_below_threshold", detail={"persons": obs.person_count})

    if (
        obs.zone == "slope"
        and obs.duration_s >= obs.fire_sustain_s
        and obs.area_grow >= obs.fire_grow
        and obs.warm_ratio >= obs.fire_warm
    ):
        if obs.period == "twilight":
            return _motion(obs, "sunset", "Lueur du soir", "La pente rougit au crépuscule. Ce n'est pas retenu comme un incendie.")
        if obs.weather in {"brouillard", "neige", "pluie"} and obs.warm_ratio < 0.2:
            return _motion(obs, "weather_glow", "Lueur dans la météo", "La tache chaude reste ambiguë par ce temps.")
        return _stamp(
            Decision(
                "publish",
                "fire",
                "Incendie",
                reason="warm_growing",
                detail={"warm_ratio": round(obs.warm_ratio, 3), "grow": round(obs.area_grow, 2)},
                confidence=min(0.99, obs.warm_ratio),
            ),
            obs,
        )

    if obs.zone == "sky":
        if obs.area_ratio > obs.max_sky_area:
            return _motion(obs, "sky_mass", "Masse dans le ciel", "Trop large pour un avion. Nuage, ou changement de lumière.")
        if obs.travel < obs.min_travel:
            return _motion(obs, "sky_still", "Point dans le ciel", "Ça n'a pas traversé le ciel. La balise et les étoiles fixes sont déjà écartées.")
        chosen, why = choose_aircraft(obs.aircraft)
        if chosen is None:
            return _motion(obs, why, "Mouvement dans le ciel", "Aucun avion unique dans le créneau. Le passage est gardé sans indicatif.")
        return _stamp(
            Decision(
                "publish",
                "plane",
                _aircraft_label(chosen),
                reason=why,
                detail=chosen,
                confidence=0.9 if why == "unique" else 0.7,
            ),
            obs,
        )

    if obs.zone in {"road", "roundabout"}:
        if obs.travel < obs.min_travel:
            return _motion(obs, "static", "Presque immobile", "Le mouvement est trop court pour une voiture ou un bus.")
        bus = _best(obs.detections, {"bus"})
        vehicle = _best(obs.detections, {"car", "truck"})
        if bus is not None and bus.conf >= conf["bus"]:
            if len(obs.trips) == 1:
                trip = obs.trips[0]
                return _stamp(
                    Decision(
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
                    ),
                    obs,
                )
            if bus.conf >= conf["bus_unnamed"]:
                return _stamp(
                    Decision(
                        "publish",
                        "bus",
                        "Bus",
                        reason="model_only",
                        detail={"trips": [trip.__dict__ for trip in obs.trips]},
                        confidence=bus.conf,
                    ),
                    obs,
                )
            return _motion(obs, "bus_uncertain", "Véhicule incertain", "La forme rappelle un bus, sans assez de certitude ni une seule course à l'horaire.")
        if vehicle is not None and vehicle.conf >= conf["car"]:
            label = "Camion" if vehicle.cls == "truck" else "Voiture"
            return _stamp(Decision("publish", "car", label, reason=vehicle.cls, confidence=vehicle.conf), obs)
        return _motion(obs, "unnamed_vehicle", "Mouvement sur la route", "Quelque chose a traversé la chaussée ou le rond-point, sans classe sûre.")

    return _motion(obs, "unclassified", "Mouvement", "Un passage a été vu. La classe viendra quand cet endroit aura été revu.")


def _context(obs: Observation) -> str:
    period = {"day": "de jour", "twilight": "au crépuscule", "night": "de nuit"}.get(obs.period, "")
    return ", ".join(part for part in (period, obs.weather) if part)


def _stamp(decision: Decision, obs: Observation) -> Decision:
    decision.detail = {**decision.detail, "period": obs.period, "weather": obs.weather, "context": _context(obs)}
    return decision


def _motion(obs: Observation, reason: str, label: str, reading: str) -> Decision:
    return _stamp(
        Decision("publish", "motion", label, reason=reason, detail={"reading": reading}, confidence=0.3),
        obs,
    )
