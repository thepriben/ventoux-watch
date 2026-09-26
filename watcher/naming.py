"""Name a motion before it is allowed into the public history."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from watcher.scenemap import DRIVABLE, FLAMMABLE

NOT_DRIVABLE = {"forest", "meadow", "building", "sky", "scree", "island", "playground", "pool"}

# The widest a thing of that kind can be where it stands, in metres. The scene
# map turns a box into ground metres, so a walker eight metres across is light
# or shadow whatever the model reads into it.
BIGGEST_M = {"person": 2.5, "car": 8.0, "truck": 20.0, "bus": 20.0,
             "bicycle": 3.0, "motorcycle": 3.5, "dog": 2.0, "horse": 3.5}
CYCLES = {"bicycle", "motorcycle"}
BEASTS = {"dog", "horse"}
CYCLE_WORD = {"bicycle": "Vélo", "motorcycle": "Moto"}
# A scooter and a motorbike are one class to the model and one word here. The
# difference matters to whoever rides it and to nobody reading this page.
BEAST_WORD = {"dog": "Chien", "horse": "Cheval"}
SURFACE_WORD = {"forest": "la forêt", "meadow": "la prairie", "scree": "la pierraille",
                "building": "un bâtiment", "road": "la route", "roundabout": "le rond-point",
                "parking": "le parking", "island": "l'îlot", "playground": "l'aire de jeux",
                "pool": "la piscine", "path": "le chemin"}


def _place_word(surface: str) -> str:
    return SURFACE_WORD.get(surface, "le relief")


# Nothing that drives or walks stands lower than this. Below it, on the
# roadway, what moved is the tarmac itself catching the light. Read the other
# way round it would not hold: a patch of light lying on the tarmac close to
# the camera measures as tall as a house, because the height is read as if the
# thing stood upright. Only the low end is trustworthy.
LOWEST_M = 0.6
# How far across the picture an aircraft may sit from what moved and still be
# called the same thing. Wide on purpose: the sky is read once every few
# minutes and an airliner covers fifteen kilometres between two readings, which
# at a hundred kilometres out is most of this budget on its own.
MAX_GAP = 0.12
PLANE_SPAN_M = 40.0
FRAME_PX = 1920.0
# Below two pixels across there is nothing to see: no shape, no motion that is
# not noise. A forty-metre airliner reaches two pixels of this frame at thirty
# kilometres, and half a pixel at a hundred. Every aircraft this watcher named
# in its first day was between fifty and a hundred kilometres out, which is to
# say none of them was in the picture at all; the white patch that moved was a
# cloud, and the callsign that fitted it was arithmetic, not sight.
SKY_REACH_M = 30_000
# How much larger than the aircraft the moving patch may be and still be called
# that aircraft. Generous: the blob is the union of a track, the jpeg smears a
# bright thing on blue, and a contrail belongs to the aircraft that made it.
# Twenty-five times the area is five times across. Beyond that it is weather.
BLOAT = 25.0
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
    frames: int = 0
    clipped: bool = False
    box_w: float = 0.0
    camera_lat: float = 44.183501
    camera_lon: float = 5.2621281
    camera_ele: float = 1390.0
    camera_bearing: float = 140.0
    camera_fov: float = 90.0
    camera_pitch: float = 0.0
    camera_vfov: float = 50.0
    fixtures: list[dict] = field(default_factory=list)
    at_x: float = -1.0
    at_y: float = -1.0
    sun_bearing: float = -1.0
    sun_elevation: float = 90.0


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


OPERATORS = {
    # The three letters that open a callsign are the operator's ICAO code. Only
    # the ones actually seen over Mont Serein are listed: a guessed airline is
    # worse than none, and an unknown code is simply left as it came.
    "AAL": "American Airlines", "AFR": "Air France", "BAW": "British Airways",
    "CCM": "Air Corsica", "DLH": "Lufthansa", "EJU": "easyJet Europe",
    "EWG": "Eurowings", "EZS": "easyJet Switzerland", "EZY": "easyJet",
    "KLM": "KLM", "RAM": "Royal Air Maroc", "RYR": "Ryanair",
    "SWR": "Swiss", "TAP": "TAP Air Portugal", "TVF": "Transavia France",
    "TUI": "TUI fly", "UAE": "Emirates", "UPS": "UPS Airlines",
    "VLG": "Vueling", "VOE": "Volotea", "WZZ": "Wizz Air",
}


def operator_of(callsign: str) -> str:
    """The airline behind a callsign, when its code is one we have met."""
    code = callsign[:3].upper()
    return OPERATORS.get(code, "") if len(callsign) > 3 and code.isalpha() else ""


def flight_of(callsign: str) -> str:
    """The flight number, which is what a passenger would recognise."""
    rest = callsign[3:].strip()
    return rest if operator_of(callsign) and rest else ""


def describe(item: dict) -> dict:
    """Everything worth keeping about one aircraft, from one state vector.

    One place, because the sky is read down two paths now and a flight named by
    one of them must not come out fuller than the same flight named by the other.
    """
    callsign = str(item.get("callsign") or "").strip()
    altitude = item.get("altitude_m")
    kept = {
        "icao24": str(item.get("icao24") or "").strip().lower(),
        "callsign": callsign,
        "altitude_m": altitude if isinstance(altitude, (int, float)) else None,
        "operator": operator_of(callsign),
        "flight": flight_of(callsign),
        "country": str(item.get("country") or "").strip(),
        "speed_ms": item.get("speed_ms"),
        "heading": item.get("heading"),
        "climb_ms": item.get("climb_ms"),
    }
    for extra in ("distance_km", "gap"):
        if item.get(extra) is not None:
            kept[extra] = item[extra]
    return kept


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
        by_icao[icao] = describe(item)
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


SHAPE_CONF = 0.2
CAR_TALL_M = 2.6


def _car_shaped(obs: Observation) -> bool:
    """Does the ground it covers have the footprint of a car?

    Only the width is trusted far from the camera and only the low end of the
    height is trusted near it, so both are asked loosely and the pair of them
    is what decides. A walker is under a metre across; a car is two and a half
    to eight, wider than it is tall, and never as tall as a house.
    """
    if not (SMALLEST_M["car"] <= obs.width_m <= BIGGEST_M["car"]):
        return False
    if not 0 < obs.height_m <= CAR_TALL_M:
        return False
    return obs.width_m > obs.height_m


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
    """What to call it: the airline and flight when known, the callsign if not.

    "American Airlines 746" is the same aircraft as "AAL746", but only one of
    them means anything to somebody reading the history.
    """
    if aircraft.get("operator") and aircraft.get("flight"):
        return f"{aircraft['operator']} {aircraft['flight']}"
    return aircraft.get("callsign") or str(aircraft.get("icao24") or "").upper()


# What it takes to call a vehicle after dark without seeing one. A car crosses
# this frame over several seconds and several frames; a beam sweeping the grass
# or the lit edge of the chalet terrace does not. Before these two lines the
# rule asked only that the patch fit a car and be bright, and it filed sixteen
# events in the first hour of one night, fourteen with nothing detected at all.
# What a weak reading needs when the blob runs off the edge of the picture.
# A clipped track has no size: half of it is outside the frame, so the surveyed
# footprint, the car test and the walker test are all measuring a fragment. Two
# events on one night were named from nothing but a person read at 0.43, both
# starting at exactly x = 0, both in fact the rear lights of a car.
EDGE_CONF = 0.6
# The widest a walker has ever been in this picture, with room to spare. Sixty
# three of them by day fill six hundredths of the width on average and a quarter
# at the very most; the four ever named after dark filled nearly half, wider
# than the widest vehicle of any day, and three were confirmed cars head-on or
# going away. Two headlights are read as a person, and no measurement of the
# ground can say otherwise where the ground is not surveyed. This one can.
PERSON_WIDEST = 0.28
# The least the model must see before the night rule may call a vehicle. That
# rule was written because a car body is dark after sunset and only its lights
# show, so it named on the shape of the ground alone. Counted over one evening
# it had named sixteen vehicles with nothing detected at all against four with
# something, and every one of the sixteen that was checked turned out to be a
# headlight crossing the grass or the lit edge of the terrace. A car read
# faintly, even as a person, is still a car; a car read as nothing is a lamp.
NIGHT_CONF = 0.25
NIGHT_FRAMES = 6
NIGHT_SECONDS = 3.0
SUN_LOW = 15.0
SUN_NEAR = 8.0
# A sun higher than fifteen degrees no longer shines through the trees into the
# lens, and eight degrees of bearing is about the width of the glare it makes.


def _facing_the_sun(obs: Observation) -> bool:
    """True when the warm patch sits in the direction the sun is coming from."""
    if obs.sun_bearing < 0 or obs.sun_elevation > SUN_LOW or not 0 <= obs.at_x <= 1:
        return False
    wide = math.tan(math.radians(obs.camera_fov / 2))
    across = math.degrees(math.atan((obs.at_x - 0.5) * 2 * wide))
    return abs((obs.camera_bearing + across - obs.sun_bearing + 540) % 360 - 180) <= SUN_NEAR


def _apparent_area(away: float, obs: Observation) -> float:
    """What fraction of the frame an airliner covers at this distance.

    Tiny, and that is the point: the number is what tells a jet from the cloud
    it was hiding behind, and nothing else in this file can.
    """
    if away <= 0:
        return 0.0
    angle = math.degrees(2 * math.atan(PLANE_SPAN_M / (2 * away)))
    return (angle / obs.camera_fov) * (angle / max(obs.camera_vfov, 1e-6))


def _gap_in_frame(aircraft: dict, obs: Observation, away: float) -> float | None:
    """How far, across the picture, this aircraft sits from what moved.

    A pinhole reading of the camera: good near the middle of the frame, a little
    optimistic at the edges. It is used to rank a handful of aircraft that are
    tens of degrees apart, never to place anything, so the slack does not
    matter and the number is written into the history to be argued with.
    """
    if not 0 <= obs.at_x <= 1 or not 0 <= obs.at_y <= 1:
        return None
    altitude = aircraft.get("altitude_m")
    if not isinstance(altitude, (int, float)) or away <= 0:
        return None
    azimuth = _azimuth(obs.camera_lat, obs.camera_lon, float(aircraft["lat"]), float(aircraft["lon"]))
    across = (azimuth - obs.camera_bearing + 540) % 360 - 180
    climb = altitude - obs.camera_ele - _earth_drop_m(away)
    rise = math.degrees(math.atan2(climb, away))
    wide = math.tan(math.radians(obs.camera_fov / 2))
    tall = math.tan(math.radians(obs.camera_vfov / 2))
    sx = 0.5 + math.tan(math.radians(across)) / (2 * wide)
    sy = 0.5 - math.tan(math.radians(rise - obs.camera_pitch)) / (2 * tall)
    return math.hypot(sx - obs.at_x, sy - obs.at_y)


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
            if not plume and _facing_the_sun(obs):
                # Warm, wide and growing, with no smoke and no plume rising, in
                # the exact direction of a sun that has just cleared the ridge.
                # The hour label was not enough: this one arrived six minutes
                # after the sun passed six degrees and was filed as daylight.
                return _motion(
                    obs,
                    "low_sun",
                    "Soleil bas dans les arbres",
                    f"Le soleil est à {obs.sun_elevation:.0f}° de hauteur, juste dans cette direction. "
                    "C'est sa lumière dans les branches, et il n'y a pas de fumée.",
                )
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
        if obs.surface and obs.surface != "sky":
            # The sky zone is one polygon drawn once; the scene map is surveyed
            # square by square. Where they disagree the map wins: a patch of
            # white sitting against the tree line is a cloud on the ridge, not
            # an airliner eleven kilometres up, however well a flight happens
            # to line up with it.
            return _motion(
                obs,
                "against_the_ground",
                "Mouvement devant le relief",
                f"Ça se détache sur {_place_word(obs.surface)}, pas sur le ciel. Un avion ne passe pas devant.",
            )
        if obs.area_ratio > obs.max_sky_area:
            return _motion(obs, "sky_mass", "Masse dans le ciel", "Trop large pour un avion. Nuage, ou changement de lumière.")
        if obs.travel < obs.min_travel:
            return _motion(obs, "sky_still", "Point dans le ciel", "Ça n'a pas traversé le ciel. La balise et les étoiles fixes sont déjà écartées.")
        visible = []
        for item in obs.aircraft:
            if not in_camera_view(item, obs):
                continue
            # How far it was, kept with it: an aircraft named in the history
            # without its distance cannot be checked against the picture, where
            # ninety kilometres is a hair and nine is a shape.
            away = _distance_m(obs.camera_lat, obs.camera_lon, float(item["lat"]), float(item["lon"]))
            if obs.area_ratio > BLOAT * _apparent_area(away, obs):
                # Too big to be this aircraft, whatever the sky says is there.
                continue
            seen = {**item, "distance_km": round(away / 1000, 1)}
            gap = _gap_in_frame(item, obs, away)
            if gap is not None:
                seen["gap"] = round(gap, 3)
            visible.append(seen)
        # When the blob's place in the picture is known, the aircraft that lands
        # nearest it wins. Height and distance were only ever standing in for
        # this: they guessed which aircraft one would see, where this checks.
        near = [item for item in visible if item.get("gap") is not None]
        if near:
            near.sort(key=lambda item: item["gap"])
            if near[0]["gap"] > MAX_GAP:
                # Aircraft in the sky, but none of them where the thing is. That
                # is the useful answer: what moved was a bird, a cloud edge or
                # an insect, and the nearest jet would have been a coincidence
                # dressed up as an identification.
                return _motion(obs, "none_at_that_spot", "Mouvement dans le ciel",
                               "Aucun avion à cet endroit de l'image. Ce n'est pas un avion.")
            if len(near) == 1 or near[0]["gap"] <= 0.5 * near[1]["gap"]:
                found = describe(near[0])
                return _stamp(
                    Decision("publish", "plane", _aircraft_label(found), reason="in_frame",
                             detail={**found, "seen": True}, confidence=0.9),
                    obs,
                )
            # Several aircraft near the same spot and no way to tell them apart.
            # The answer is that we do not know, and it must be given here: left
            # to fall through, the older rule below would pick the lowest of
            # them, which is how a contrail at eleven thousand metres came to be
            # signed by an aircraft flying at four.
            return _motion(obs, "several_at_that_spot", "Avion non identifié",
                           "Plusieurs avions à cet endroit de l'image, aucun moyen de les départager.")
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
        cycle = _best(obs.detections, {"bicycle", "motorcycle"})
        animal = _best(obs.detections, {"dog", "horse"})
        cycle = _best(obs.detections, CYCLES)
        beast = _best(obs.detections, BEASTS)
        if person is not None and obs.box_w > PERSON_WIDEST:
            person = None
        if obs.clipped and not obs.surface:
            best = max((hit.conf for hit in obs.detections), default=0.0)
            if best < EDGE_CONF:
                return _motion(
                    obs,
                    "edge_of_frame",
                    "Mouvement au bord de l'image",
                    "La tache sort du cadre et le sol n'est pas relevé là : sa taille est inconnue, "
                    "et rien n'a été reconnu assez sûrement pour la nommer sans elle.",
                )
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
        if person is not None and _car_shaped(obs):
            # The model reads a car on this roundabout at about a quarter
            # confidence and the people beside it at half, so the patch that
            # moved kept coming out as somebody on foot. The ground says
            # otherwise: nobody walking is four metres wide and a metre and a
            # half tall. A measurement beats a weak guess.
            person = None
        if cycle is not None and not _fits(obs, cycle.cls):
            cycle = None
        if beast is not None and not _fits(obs, beast.cls):
            beast = None
        if vehicle is not None and not _fits(obs, vehicle.cls):
            vehicle = None
        if bus is not None and not _fits(obs, "bus"):
            bus = None
        if person is not None and _car_shaped(obs):
            # The model reads a car on this roundabout at about a quarter
            # confidence and the people beside it at half, so the patch that
            # moved kept coming out as somebody on foot. The ground says
            # otherwise: nobody walking is four metres wide and a metre and a
            # half tall. A measurement beats a weak guess.
            person = None
        if cycle is not None and not _fits(obs, cycle.cls):
            cycle = None
        if animal is not None and not _fits(obs, animal.cls):
            animal = None
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
            # Brightness used to stand in for movement here. It cannot: the
            # brightest thing on this hillside at night is a patch of grass a
            # headlight is crossing, and it fits a car exactly.
            steady = obs.frames >= NIGHT_FRAMES and obs.duration_s >= NIGHT_SECONDS
            glimpsed = max((hit.conf for hit in obs.detections), default=0.0) >= NIGHT_CONF
            if person is None and glimpsed and _fits(obs, "car") and steady and obs.travel >= obs.min_travel:
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
        if vehicle is not None and vehicle.conf >= SHAPE_CONF and _car_shaped(obs):
            # Below the usual threshold, but the footprint settles it. Asked
            # only of something already shaped like a car: this is not a lower
            # bar, it is a second kind of evidence.
            return _stamp(
                Decision(
                    "publish",
                    "vehicle",
                    _tinted(_vehicle_word(obs, vehicle, bus), obs.colour),
                    reason="shape",
                    detail={"width_m": round(obs.width_m, 1), "height_m": round(obs.height_m, 1)},
                    confidence=max(vehicle.conf, 0.5),
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
        if cycle is not None and cycle.conf >= 0.35:
            # Before the walker, never after. The rider is a person too, and
            # whichever of the two is read first is what the event is called:
            # asked in the other order, every scooter on this roundabout came
            # out as somebody on foot.
            return _stamp(
                Decision("publish", "cycle", CYCLE_WORD[cycle.cls], reason=cycle.cls, confidence=cycle.conf),
                obs,
            )
        if person is not None and person.conf >= 0.4:
            # A dog is walked, not met: when both are in the same patch of
            # movement they are one event, and naming only the end of the lead
            # leaves out the half of it that was asked for.
            word = f"Piéton et {BEAST_WORD[beast.cls].lower()}" if beast is not None else "Piéton"
            return _stamp(Decision("publish", "person", word, reason="person", confidence=person.conf), obs)
        if beast is not None and beast.conf >= 0.4:
            return _stamp(
                Decision("publish", "animal", BEAST_WORD[beast.cls], reason=beast.cls, confidence=beast.conf),
                obs,
            )
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
