"""Where the sun and the moon are, seen from the camera.

A bright disc in the picture used to be called the moon because it was round,
bright and the hour was late. That is a guess, and it was bound to be wrong one
evening. Both bodies can be worked out to a fraction of a degree from the date
alone, so the disc can be named instead of guessed: if the sky says the moon is
at ninety-two degrees and the disc is at ninety-two degrees, it is the moon, and
if the sun is below the horizon and behind the camera it is not the sun.

The moon follows Paul Schlyter's abridged theory with the usual perturbation
terms. Its parallax is close to a degree, which is larger than the agreement we
are looking for, so the position is corrected to the observer rather than left
at the centre of the Earth. Checked against the IMCCE ephemeris service; see
tests/test_bodies.py for the epochs and the error.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from watcher.scene import solar_azimuth, solar_elevation

EARTH_RADII_KM = 6371.0


def _days(when: datetime) -> float:
    """Days since 1999-12-31 00:00 UT, the epoch Schlyter's elements use."""
    moment = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    year, month, day = moment.year, moment.month, moment.day
    hours = moment.hour + moment.minute / 60 + moment.second / 3600
    count = (367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9
             + day - 730530)
    return count + hours / 24.0


def _sun_longitude(day: float) -> tuple[float, float]:
    """Mean anomaly and true longitude of the sun, both in degrees.

    The moon's largest wobbles are pulls by the sun, so its place cannot be had
    without the sun's.
    """
    anomaly = (356.0470 + 0.9856002585 * day) % 360
    perihelion = (282.9404 + 4.70935e-5 * day) % 360
    eccentricity = 0.016709 - 1.151e-9 * day
    ecc_anom = anomaly + math.degrees(eccentricity) * math.sin(math.radians(anomaly)) * (
        1 + eccentricity * math.cos(math.radians(anomaly))
    )
    x = math.cos(math.radians(ecc_anom)) - eccentricity
    y = math.sin(math.radians(ecc_anom)) * math.sqrt(1 - eccentricity * eccentricity)
    true_anom = math.degrees(math.atan2(y, x))
    return anomaly, (true_anom + perihelion) % 360


def moon_radec(when: datetime) -> tuple[float, float, float]:
    """Right ascension in degrees, declination in degrees, distance in km.

    Geocentric: seen from the middle of the Earth, not from the hillside.
    """
    day = _days(when)
    node = (125.1228 - 0.0529538083 * day) % 360
    incl = 5.1454
    perigee = (318.0634 + 0.1643573223 * day) % 360
    axis = 60.2666
    ecc = 0.054900
    anomaly = (115.3654 + 13.0649929509 * day) % 360

    ecc_anom = anomaly + math.degrees(ecc) * math.sin(math.radians(anomaly)) * (
        1 + ecc * math.cos(math.radians(anomaly))
    )
    for _ in range(4):
        ecc_anom -= (ecc_anom - math.degrees(ecc) * math.sin(math.radians(ecc_anom)) - anomaly) / (
            1 - ecc * math.cos(math.radians(ecc_anom))
        )

    x = axis * (math.cos(math.radians(ecc_anom)) - ecc)
    y = axis * math.sqrt(1 - ecc * ecc) * math.sin(math.radians(ecc_anom))
    distance = math.hypot(x, y)
    true_anom = math.degrees(math.atan2(y, x))

    node_r, incl_r, arg_r = map(math.radians, (node, incl, true_anom + perigee))
    ex = distance * (math.cos(node_r) * math.cos(arg_r)
                     - math.sin(node_r) * math.sin(arg_r) * math.cos(incl_r))
    ey = distance * (math.sin(node_r) * math.cos(arg_r)
                     + math.cos(node_r) * math.sin(arg_r) * math.cos(incl_r))
    ez = distance * math.sin(arg_r) * math.sin(incl_r)

    longitude = math.degrees(math.atan2(ey, ex)) % 360
    latitude = math.degrees(math.atan2(ez, math.hypot(ex, ey)))

    sun_anomaly, sun_longitude = _sun_longitude(day)
    mean_long = (node + perigee + anomaly) % 360
    elong = (mean_long - sun_longitude) % 360
    arg_lat = (mean_long - node) % 360

    # The dozen terms worth keeping. The first two, the evection and the
    # variation, are more than a degree each; below a hundredth of a degree the
    # rest is smaller than the moon looks.
    d, m, mm, f = (math.radians(elong), math.radians(sun_anomaly),
                   math.radians(anomaly), math.radians(arg_lat))
    longitude += (-1.274 * math.sin(mm - 2 * d) + 0.658 * math.sin(2 * d)
                  - 0.186 * math.sin(m) - 0.059 * math.sin(2 * mm - 2 * d)
                  - 0.057 * math.sin(mm - 2 * d + m) + 0.053 * math.sin(mm + 2 * d)
                  + 0.046 * math.sin(2 * d - m) + 0.041 * math.sin(mm - m)
                  - 0.035 * math.sin(d) - 0.031 * math.sin(mm + m)
                  - 0.015 * math.sin(2 * f - 2 * d) + 0.011 * math.sin(mm - 4 * d))
    latitude += (-0.173 * math.sin(f - 2 * d) - 0.055 * math.sin(mm - f - 2 * d)
                 - 0.046 * math.sin(mm + f - 2 * d) + 0.033 * math.sin(f + 2 * d)
                 + 0.017 * math.sin(2 * mm + f))
    distance += -0.58 * math.cos(mm - 2 * d) - 0.46 * math.cos(2 * d)

    obliquity = math.radians(23.4393 - 3.563e-7 * day)
    lon_r, lat_r = math.radians(longitude % 360), math.radians(latitude)
    gx = math.cos(lat_r) * math.cos(lon_r)
    gy = math.cos(lat_r) * math.sin(lon_r)
    gz = math.sin(lat_r)
    ry = gy * math.cos(obliquity) - gz * math.sin(obliquity)
    rz = gy * math.sin(obliquity) + gz * math.cos(obliquity)
    ra = math.degrees(math.atan2(ry, gx)) % 360
    dec = math.degrees(math.atan2(rz, math.hypot(gx, ry)))
    return ra, dec, distance * EARTH_RADII_KM


def _sidereal(when: datetime, lon: float) -> float:
    """Local sidereal time in degrees: which way the sky has turned.

    Built on the sun's *mean* longitude, not its true one. The difference
    between the two is the equation of the centre, which reaches two degrees,
    and putting the whole sky two degrees out would defeat the point.
    """
    day = _days(when)
    moment = (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    hours = moment.hour + moment.minute / 60 + moment.second / 3600
    anomaly = (356.0470 + 0.9856002585 * day) % 360
    perihelion = (282.9404 + 4.70935e-5 * day) % 360
    mean_longitude = (anomaly + perihelion) % 360
    return (mean_longitude + 180.0 + hours * 15.0 + lon) % 360


def moon_position(when: datetime, lat: float, lon: float, ele_m: float = 0.0) -> tuple[float, float]:
    """The moon's bearing and height above the horizon, in degrees.

    Corrected to this hillside rather than to the middle of the Earth: the moon
    shifts by nearly a degree between the two, which is more than the agreement
    we are trying to prove.
    """
    ra, dec, distance_km = moon_radec(when)
    hour_angle = (_sidereal(when, lon) - ra) % 360

    # Parallax, by shifting to the observer in rectangular coordinates.
    lat_r, ha_r, dec_r = math.radians(lat), math.radians(hour_angle), math.radians(dec)
    reach = 1.0 + ele_m / (EARTH_RADII_KM * 1000.0)
    px = distance_km * math.cos(dec_r) * math.cos(ha_r) - EARTH_RADII_KM * reach * math.cos(lat_r)
    py = distance_km * math.cos(dec_r) * math.sin(ha_r)
    pz = distance_km * math.sin(dec_r) - EARTH_RADII_KM * reach * math.sin(lat_r)

    # Rotate to the horizon: south, east, up.
    south = px * math.sin(lat_r) - pz * math.cos(lat_r)
    east = py
    up = px * math.cos(lat_r) + pz * math.sin(lat_r)
    azimuth = (math.degrees(math.atan2(east, south)) + 180.0) % 360
    altitude = math.degrees(math.atan2(up, math.hypot(south, east)))
    return azimuth, altitude


def moon_phase(when: datetime) -> float:
    """How full the moon is, from 0 at new to 1 at full."""
    day = _days(when)
    _anomaly, sun_long = _sun_longitude(day)
    ra, dec, _distance = moon_radec(when)
    obliquity = math.radians(23.4393 - 3.563e-7 * day)
    ra_r, dec_r = math.radians(ra), math.radians(dec)
    gx = math.cos(dec_r) * math.cos(ra_r)
    gy = math.cos(dec_r) * math.sin(ra_r)
    gz = math.sin(dec_r)
    ey = gy * math.cos(obliquity) + gz * math.sin(obliquity)
    moon_long = math.degrees(math.atan2(ey, gx)) % 360
    return (1 - math.cos(math.radians(moon_long - sun_long))) / 2


def sun_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """The sun's bearing and height above the horizon, in degrees."""
    return solar_azimuth(when, lat, lon), solar_elevation(when, lat, lon)
