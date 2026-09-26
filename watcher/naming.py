"""Name a motion before it is allowed into the public history."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from watcher.scenemap import DRIVABLE, FLAMMABLE

NOT_DRIVABLE = {"forest", "meadow", "building", "sky", "scree", "island", "playground", "pool"}

# The widest a thing of that kind can be where it stands, in metres. The scene
# map turns a box into ground metres, so a walker eight metres across is light
# or shadow whatever the model reads into it.
BIGGEST_M = {"person": 2.5, "car": 8.0, "truck": 20.0, "bus": 20.0}
# Nothing that drives or walks stands lower than this. Below it, on the
# roadway, what moved is the tarmac itself catching the light.
# Read the other way round it would not hold: a patch of light lying on the
# tarmac close to the camera measures as tall as a house, because the height
# is read as if the thing stood upright. Only the low end is trustworthy.
SKY_REACH_M = 120_000
# How far an aircraft can be and still be worth matching. Beyond that a jet is
# under two pixels wide and its contrail is indistinguishable from cloud, so a
# name put to it would be a guess dressed up as a reading.

LOWEST_M = 0.6
# And the narrowest. Every vehicle ever confirmed here has measured at least
# two metres and a half across the ground, a bus eleven. Below two metres there
# is nothing on wheels: a walker is that wide, and so is a patch of light.
SMALLEST_M = {"car": 2.0, "truck": 2.0, "bus": 2.0}
# How long something has to burn before the word "incendie" is used. The width
# of a plume says nothing: smoke spreads over a hundred metres in a minute
# above a fire the size of a car. How long it has held does say something.
BLAZE_S = 600.0


@dataclass
class Detection:
    cls: str
    conf: float
    cx: float | None = None
    cy: float | None = None


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
    smoke_ratio: float = 0.0
    rise: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
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
    fire_smoke: float = 0.35
    fire_rise: float = 0.008
    period: str = "day"
    weather: str = ""
    surface: str = ""
    near_road: bool = True
    colour: str = ""
    landmark: str = ""
    lit_ratio: float = 0.0
    camera_lat: float = 44.183501
    camera_lon: float = 5.2621281
    camera_ele: float = 1390.0
    camera_bearing: float = 140.0
    camera_fov: float = 90.0
    camera_pitch: float = 0.0
    camera_vfov: float = 50.0
    fixtures: list[dict] = field(default_factory=list)


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


# Adjectives, feminine then masculine, so the colour agrees with the word it
# follows.
TINTS = {
    "blanc": ("blanche", "blanc"),
    "noir": ("noire", "noir"),
    "gris": ("grise", "gris"),
    "rouge": ("rouge", "rouge"),
    "orange": ("orange", "orange"),
    "jaune": ("jaune", "jaune"),
    "vert": ("verte", "vert"),
    "bleu": ("bleue", "bleu"),
    "marron": ("marron", "marron"),
}
FEMININE = {"Voiture", "Camionnette"}


def _tinted(word: str, colour: str) -> str:
    """Put the colour after the word, spelt to agree with it."""
    pair = TINTS.get(colour)
    if not pair:
        return word
    return f"{word} {pair[0] if word in FEMININE else pair[1]}"


def _vehicle_word(obs: Observation, vehicle: Detection | None, bus: Detection | None) -> str:
    """A lorry only when the model says so and the thing really is that wide.

    The model calls half the cars lorries. The ground width settles it: at this
    place a car covers about two and a half metres, a lorry more than five.
    """
    if vehicle is None and bus is None:
        return "Véhicule"
    heavy = (vehicle is not None and vehicle.cls == "truck") or bus is not None
    if heavy and obs.width_m > 5.5:
        return "Camion"
    return "Voiture"


def _fits(obs: Observation, cls: str) -> bool:
    """Could a thing of that kind really be that wide, where it stands?"""
    limit = BIGGEST_M.get(cls)
    if limit and obs.width_m > limit:
        return False
    floor = SMALLEST_M.get(cls)
    return not (floor and 0 < obs.width_m < floor)


def _best(detections: list[Detection], names: set[str]) -> Detection | None:
    found = [item for item in detections if item.cls in names]
    if not found:
        return None
    return max(found, key=lambda item: item.conf)


def _aircraft_label(aircraft: dict) -> str:
    return aircraft["callsign"] or aircraft["icao24"].upper()


def in_camera_view(aircraft: dict, obs: Observation) -> bool:
    """True when this aircraft would show in the camera sky, not merely nearby."""
    lat = aircraft.get("lat")
    lon = aircraft.get("lon")
    altitude = aircraft.get("altitude_m")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False
    if not isinstance(altitude, (int, float)):
        return False
    distance = _distance_m(obs.camera_lat, obs.camera_lon, float(lat), float(lon))
    if not 0 < distance <= SKY_REACH_M:
        return False
    if altitude < obs.camera_ele:
        return False
    azimuth = _azimuth(obs.camera_lat, obs.camera_lon, float(lat), float(lon))
    relative = (azimuth - obs.camera_bearing + 540) % 360 - 180
    if abs(relative) > obs.camera_fov / 2:
        return False
    # The camera is tilted down, so its highest line of sight is barely twenty
    # degrees up. An airliner directly overhead is not in the picture; the same
    # airliner is, sixty kilometres out, near the top edge. Height alone tells
    # nothing — only the angle it is seen at does.
    climb = altitude - obs.camera_ele - _earth_drop_m(distance)
    rise = math.degrees(math.atan2(climb, distance))
    return obs.camera_pitch - obs.camera_vfov / 2 <= rise <= obs.camera_pitch + obs.camera_vfov / 2


def _earth_drop_m(distance: float) -> float:
    """How far the earth has curved away underfoot, eased by refraction.

    Nothing at two kilometres, but ninety metres at forty and four hundred at
    ninety, which is the difference between an aircraft inside the frame and one
    below its lower edge.
    """
    return distance * distance / (2 * 6_371_000 * 7 / 6)


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


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

    if (obs.surface in FLAMMABLE or (obs.zone == "slope" and not obs.surface)) and obs.duration_s >= obs.fire_sustain_s:
        flame = obs.warm_ratio >= obs.fire_warm
        # A fire that has just caught shows as a pale plume climbing out of the
        # trees, minutes before any flame is large enough to colour a pixel.
        plume = obs.smoke_ratio >= obs.fire_smoke and obs.rise >= obs.fire_rise
        if (flame or plume) and obs.area_grow >= obs.fire_grow:
            if obs.landmark and obs.travel < obs.min_travel:
                # The red lamp on the summit mast blinks in place all night.
                # It grows and it is warm, and it is not a fire.
                return _motion(
                    obs,
                    "beacon",
                    "Feu de balisage",
                    f"{obs.landmark} porte une lampe. Elle clignote sans bouger.",
                )
            if obs.period == "twilight" and not plume:
                return _motion(obs, "sunset", "Lueur du soir", "La pente rougit au crépuscule. Ce n'est pas retenu comme un incendie.")
            if obs.weather in {"brouillard", "neige", "pluie"} and obs.warm_ratio < 0.2:
                return _motion(obs, "weather_glow", "Lueur dans la météo", "La tache chaude reste ambiguë par ce temps.")
            if obs.period == "night" and plume and not flame:
                return _motion(obs, "night_plume", "Masse sur la pente", "Une masse pâle monte, mais de nuit une fumée ne se distingue pas d'un nuage bas.")
            return _stamp(
                Decision(
                    "publish",
                    "fire",
                    # Caught within seconds, it is a start. "Incendie" is kept
                    # for something that has held, so the word still means
                    # something the day it is used.
                    "Incendie" if obs.duration_s >= BLAZE_S else "Départ de feu",
                    reason="warm_growing" if flame else "plume_rising",
                    detail={
                        "warm_ratio": round(obs.warm_ratio, 3),
                        "smoke_ratio": round(obs.smoke_ratio, 3),
                        "rise": round(obs.rise, 4),
                        "grow": round(obs.area_grow, 2),
                        "width_m": round(obs.width_m, 1),
                        "surface": obs.surface,
                    },
                    confidence=min(0.99, max(obs.warm_ratio, obs.smoke_ratio * 0.8)),
                ),
                obs,
            )

    if obs.zone == "sky":
        if obs.area_ratio > obs.max_sky_area:
            return _motion(obs, "sky_mass", "Masse dans le ciel", "Trop large pour un avion. Nuage, ou changement de lumière.")
        if obs.travel < obs.min_travel:
            return _motion(obs, "sky_still", "Point dans le ciel", "Ça n'a pas traversé le ciel. La balise et les étoiles fixes sont déjà écartées.")
        visible = [item for item in obs.aircraft if in_camera_view(item, obs)]
        chosen, why = choose_aircraft(visible)
        if chosen is None:
            return _motion(obs, why, "Mouvement dans le ciel", "Aucun avion visible dans l'image. Le secteur OpenSky ne suffit pas.")
        return _stamp(
            Decision(
                "publish",
                "plane",
                _aircraft_label(chosen),
                reason=why,
                detail={**chosen, "seen": True},
                confidence=0.9 if why == "unique" else 0.7,
            ),
            obs,
        )

    if obs.zone in {"road", "roundabout", "other"}:
        bus = _best(obs.detections, {"bus"})
        vehicle = _best(obs.detections, {"car", "truck"})
        person = _best(obs.detections, {"person"})
        if obs.width_m > BIGGEST_M["truck"]:
            return _motion(
                obs,
                "oversized",
                "Tache trop large",
                f"Environ {obs.width_m:.0f} m au sol. Rien ne roule et ne marche à cette taille : de la lumière ou de l'ombre.",
            )
        if obs.surface in DRIVABLE and 0 < obs.height_m < LOWEST_M:
            return _motion(
                obs,
                "tarmac",
                "Motif sur la chaussée",
                f"Environ {obs.height_m * 100:.0f} cm de haut au sol. C'est le revêtement qui prend la lumière, pas un véhicule.",
            )
        if not _fits(obs, "person"):
            person = None
        if vehicle is not None and not _fits(obs, vehicle.cls):
            vehicle = None
        if bus is not None and not _fits(obs, "bus"):
            bus = None
        if obs.landmark and obs.travel < obs.min_travel:
            return _motion(
                obs,
                "landmark",
                "Repère éclairé",
                f"{obs.landmark} n'a pas bougé. Une lumière est passée dessus.",
            )
        if obs.surface in NOT_DRIVABLE and not obs.near_road and vehicle is not None and person is None:
            return _motion(obs, "off_road", "Mouvement hors chaussée", "Aucune voiture ne roule là.")
        if obs.surface == "island" and obs.travel < obs.min_travel:
            return _motion(
                obs,
                "island",
                "Décor de l'îlot",
                "Sur l'îlot central du rond-point, et ça n'a pas bougé. Les pierres et les figures y sont plantées.",
            )
        if obs.surface == "parking" and obs.travel < obs.min_travel:
            return _motion(obs, "parked", "Voiture garée", "Sur une aire de stationnement, et ça n'a pas bougé.")
        if obs.period in {"night", "twilight"} and obs.surface in DRIVABLE and bus is None and vehicle is None:
            # After dark the model reads a car body as a walker. On the roadway
            # at that hour, what moves is a vehicle unless the shape is plain.
            if person is not None and person.conf < 0.6:
                person = None
            lit = obs.lit_ratio >= 0.03
            if person is None and _fits(obs, "car") and (lit or obs.travel >= obs.min_travel):
                return _stamp(
                    Decision(
                        "publish",
                        "vehicle",
                        _tinted("Véhicule", obs.colour),
                        reason="car_lights" if lit else "night_road",
                        confidence=min(0.85, 0.5 + obs.lit_ratio),
                    ),
                    obs,
                )
        if obs.travel < obs.min_travel and not (person is not None and person.conf >= 0.4):
            return _motion(obs, "static", "Presque immobile", "Le mouvement est trop court pour une voiture ou un bus.")
        if (
            vehicle is not None
            and person is not None
            and vehicle.conf >= conf["car"]
            and person.conf >= 0.4
        ):
            return _stamp(
                Decision(
                    "publish",
                    "vehicle",
                    "Voiture et piéton",
                    reason="car_and_person",
                    confidence=min(vehicle.conf, person.conf),
                ),
                obs,
            )
        if bus is not None and bus.conf >= conf["bus"] and len(obs.trips) == 1:
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
        if vehicle is not None and vehicle.conf >= conf["car"]:
            return _stamp(
                Decision(
                    "publish",
                    "vehicle",
                    _tinted(_vehicle_word(obs, vehicle, bus), obs.colour),
                    reason=vehicle.cls,
                    detail={"width_m": round(obs.width_m, 1)} if obs.width_m else {},
                    confidence=vehicle.conf,
                ),
                obs,
            )
        if person is not None and person.conf >= 0.4:
            return _stamp(Decision("publish", "person", "Piéton", reason="person", confidence=person.conf), obs)
        if obs.travel < obs.min_travel:
            return _motion(obs, "static", "Presque immobile", "Le mouvement est trop court pour une voiture ou un bus.")
        if 0 < obs.width_m < SMALLEST_M["car"]:
            # Asked after the walker has had its say: a walker really is that
            # narrow. What is left is neither, and on tarmac it is light.
            return _motion(
                obs,
                "too_small",
                "Trop petit pour un véhicule",
                f"Environ {obs.width_m * 100:.0f} cm au sol. Une voiture en couvre deux mètres et demi ici.",
            )
        return _motion(obs, "unnamed_vehicle", "Mouvement sur la route", "Quelque chose a traversé la chaussée ou le rond-point, sans classe sûre.")

    if obs.zone == "slope" and obs.travel < max(obs.min_travel, 0.02):
        return Decision("hold", reason="slope_still")

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
