import json
import math
import time
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from watcher.geometry import assign_zone
from watcher.gtfs import GtfsIndex, load_feed
from watcher.main import _crossed_sky, _might_be_bus
from watcher.motion import MotionDetector, Track
from watcher.naming import Detection, Observation, Trip, choose_aircraft, decide
from watcher.review import apply_review, parse_review
from watcher.opensky import SkyArchive
from watcher.scene import ViewLog, moon_in_sky, read_sky, solar_period, weather_label
from watcher.store import Store, fold_events, small_jpeg

ROOT = Path(__file__).resolve().parents[1]
ZONES = json.loads((ROOT / "config" / "zones.json").read_text())


def _ahead(lat: float, lon: float, bearing: float, meters: float) -> tuple[float, float]:
    radius = 6_371_000
    br = math.radians(bearing)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(meters / radius) + math.cos(lat1) * math.sin(meters / radius) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(meters / radius) * math.cos(lat1),
        math.cos(meters / radius) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


class NamingTests(unittest.TestCase):
    def test_unique_aircraft_is_named(self):
        chosen, reason = choose_aircraft([{"icao24": "abc123", "callsign": "AFR123", "altitude_m": 8500}])
        self.assertEqual(reason, "unique")
        self.assertEqual(chosen["callsign"], "AFR123")

    def test_three_aircraft_stay_unnamed(self):
        aircraft = [
            {"icao24": "a", "callsign": "A", "altitude_m": 9000},
            {"icao24": "b", "callsign": "B", "altitude_m": 10000},
            {"icao24": "c", "callsign": "C", "altitude_m": 11000},
        ]
        chosen, reason = choose_aircraft(aircraft)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "ambiguous")

    def test_much_lower_aircraft_is_named(self):
        aircraft = [
            {"icao24": "high", "callsign": "HIGH", "altitude_m": 10000},
            {"icao24": "low", "callsign": "LOW1", "altitude_m": 3000},
        ]
        chosen, reason = choose_aircraft(aircraft)
        self.assertEqual(reason, "much_lower")
        self.assertEqual(chosen["callsign"], "LOW1")

    def test_sky_motion_without_a_single_plane_is_kept(self):
        decision = decide(Observation(zone="sky", travel=0.05, area_ratio=0.001, aircraft=[], period="night"))
        self.assertTrue(decision.publish)
        self.assertEqual(decision.type, "motion")
        self.assertEqual(decision.reason, "none")
        self.assertEqual(decision.detail["period"], "night")

    def test_cloud_is_not_called_a_plane(self):
        decision = decide(Observation(zone="sky", travel=0.2, area_ratio=0.2, aircraft=[{"icao24": "a", "callsign": "X", "altitude_m": 1000}]))
        self.assertEqual(decision.type, "motion")
        self.assertEqual(decision.label, "Masse dans le ciel")

    def test_a_person_on_the_road_is_a_pedestrian(self):
        decision = decide(Observation(zone="road", travel=0.0, detections=[Detection("person", 0.62)]))
        self.assertEqual(decision.type, "person")
        self.assertEqual(decision.label, "Piéton")

    def test_static_blob_is_not_a_car(self):
        decision = decide(Observation(zone="roundabout", travel=0.0, detections=[Detection("car", 0.9)]))
        self.assertEqual(decision.type, "motion")
        self.assertNotEqual(decision.label, "Voiture")

    def test_dusk_glow_is_not_a_fire(self):
        decision = decide(Observation(zone="slope", duration_s=25, area_grow=2.0, warm_ratio=0.2, period="twilight"))
        self.assertEqual(decision.type, "motion")
        self.assertEqual(decision.label, "Lueur du soir")

    def test_sky_motion_publishes_the_callsign(self):
        lat, lon = _ahead(44.183501, 5.2621281, 140, 2000)
        decision = decide(
            Observation(
                zone="sky",
                travel=0.05,
                area_ratio=0.001,
                aircraft=[{"icao24": "394c12", "callsign": "AFR472", "altitude_m": 1800, "lat": lat, "lon": lon}],
            )
        )
        self.assertTrue(decision.publish)
        self.assertEqual(decision.type, "plane")
        self.assertEqual(decision.label, "AFR472")
        self.assertTrue(decision.detail["seen"])

    def test_a_high_aircraft_outside_the_picture_is_not_named(self):
        decision = decide(
            Observation(
                zone="sky",
                travel=0.05,
                area_ratio=0.001,
                aircraft=[{"icao24": "47a039", "callsign": "NSZ5525", "altitude_m": 10836, "lat": 44.25, "lon": 5.45}],
            )
        )
        self.assertEqual(decision.type, "motion")
        self.assertNotEqual(decision.label, "NSZ5525")

    def test_car_on_the_road_is_published(self):
        decision = decide(Observation(zone="road", travel=0.08, detections=[Detection("car", 0.8)]))
        self.assertEqual(decision.type, "vehicle")
        self.assertEqual(decision.label, "Voiture")

    def test_bus_with_one_trip_uses_the_line(self):
        trip = Trip("Navette", "Mont Serein", "Chalet", "10:00:00", "transcove")
        decision = decide(Observation(zone="roundabout", travel=0.05, detections=[Detection("bus", 0.7)], trips=[trip]))
        self.assertEqual(decision.label, "Bus Navette")
        self.assertEqual(decision.detail["scheduled"], "10:00:00")

    def test_a_car_beside_a_person_is_named_as_both(self):
        decision = decide(
            Observation(zone="road", travel=0.08, detections=[Detection("car", 0.7), Detection("person", 0.6)])
        )
        self.assertEqual(decision.label, "Voiture et piéton")
        self.assertEqual(decision.type, "vehicle")

    def test_bus_with_two_trips_is_not_given_a_line(self):
        trips = [
            Trip("A", "Un", "Stop", "10:00:00", "zou"),
            Trip("B", "Deux", "Stop", "10:05:00", "zou"),
        ]
        decision = decide(Observation(zone="road", travel=0.05, detections=[Detection("bus", 0.8)], trips=trips))
        self.assertNotEqual(decision.type, "bus")
        self.assertNotEqual(decision.label, "Bus")

    def test_a_lit_landmark_is_not_a_walker(self):
        decision = decide(
            Observation(zone="other", travel=0.0, landmark="statue", detections=[Detection("person", 0.6)])
        )
        self.assertEqual(decision.type, "motion")
        self.assertNotEqual(decision.label, "Piéton")

    def test_a_car_cannot_drive_through_the_forest(self):
        decision = decide(
            Observation(zone="other", travel=0.08, surface="forest", near_road=False,
                        detections=[Detection("car", 0.8)])
        )
        self.assertEqual(decision.type, "motion")

    def test_a_car_on_the_verge_is_still_a_car(self):
        decision = decide(
            Observation(zone="road", travel=0.08, surface="meadow", near_road=True,
                        detections=[Detection("car", 0.8)])
        )
        self.assertEqual(decision.type, "vehicle")

    def test_car_lights_name_a_vehicle_at_night(self):
        decision = decide(
            Observation(zone="road", travel=0.05, period="night", lit_ratio=0.06, surface="road")
        )
        self.assertEqual(decision.type, "vehicle")
        self.assertEqual(decision.reason, "car_lights")

    def test_lights_off_the_road_stay_unnamed(self):
        decision = decide(
            Observation(zone="other", travel=0.05, period="night", lit_ratio=0.06, surface="meadow")
        )
        self.assertEqual(decision.type, "motion")

    def test_a_faint_walker_on_the_night_road_is_a_car(self):
        decision = decide(
            Observation(
                zone="roundabout",
                travel=0.05,
                period="twilight",
                surface="roundabout",
                detections=[Detection("person", 0.42)],
            )
        )
        self.assertEqual(decision.type, "vehicle")

    def test_a_clear_walker_at_night_is_still_a_walker(self):
        decision = decide(
            Observation(
                zone="roundabout",
                travel=0.05,
                period="night",
                surface="roundabout",
                detections=[Detection("person", 0.72)],
            )
        )
        self.assertEqual(decision.type, "person")

    def test_a_named_bus_survives_the_night_rule(self):
        decision = decide(
            Observation(
                zone="road",
                travel=0.05,
                period="night",
                surface="road",
                lit_ratio=0.1,
                detections=[Detection("bus", 0.7)],
                trips=[Trip("5", "Sault", "Mont Serein", "20:10", "gtfs")],
            )
        )
        self.assertEqual(decision.type, "bus")

    def test_a_lorry_needs_the_width_of_a_lorry(self):
        narrow = decide(Observation(zone="road", travel=0.08, surface="road", width_m=2.6,
                                    detections=[Detection("truck", 0.8)]))
        wide = decide(Observation(zone="road", travel=0.08, surface="road", width_m=7.0,
                                  detections=[Detection("truck", 0.8)]))
        self.assertEqual(narrow.label, "Voiture")
        self.assertEqual(wide.label, "Camion")

    def test_the_colour_agrees_with_the_word(self):
        car = decide(Observation(zone="road", travel=0.08, surface="road", width_m=2.5, colour="blanc",
                                 detections=[Detection("car", 0.8)]))
        lorry = decide(Observation(zone="road", travel=0.08, surface="road", width_m=7.0, colour="blanc",
                                   detections=[Detection("truck", 0.8)]))
        self.assertEqual(car.label, "Voiture blanche")
        self.assertEqual(lorry.label, "Camion blanc")

    def test_a_still_car_on_a_car_park_is_parked(self):
        decision = decide(Observation(zone="other", travel=0.0, surface="parking",
                                      detections=[Detection("car", 0.8)]))
        self.assertEqual(decision.type, "motion")
        self.assertEqual(decision.reason, "parked")

    def test_a_walker_the_size_of_a_bus_is_not_a_walker(self):
        decision = decide(
            Observation(zone="roundabout", travel=0.05, surface="roundabout", width_m=15.8,
                        detections=[Detection("person", 0.7)])
        )
        self.assertEqual(decision.type, "motion")

    def test_a_blob_wider_than_a_lorry_is_light(self):
        decision = decide(Observation(zone="road", travel=0.2, surface="road", width_m=40.0))
        self.assertEqual(decision.reason, "oversized")

    def test_a_plume_over_the_forest_is_a_fire_starting(self):
        decision = decide(
            Observation(surface="forest", duration_s=40, area_grow=2.0, smoke_ratio=0.5, rise=0.02, width_m=12)
        )
        self.assertEqual(decision.type, "fire")
        self.assertEqual(decision.reason, "plume_rising")

    def test_a_plume_that_does_not_climb_is_not_a_fire(self):
        decision = decide(
            Observation(surface="forest", duration_s=40, area_grow=2.0, smoke_ratio=0.5, rise=0.0)
        )
        self.assertEqual(decision.type, "motion")

    def test_a_plume_over_the_roadway_is_not_a_fire(self):
        decision = decide(
            Observation(surface="road", zone="road", duration_s=40, area_grow=2.0, smoke_ratio=0.5, rise=0.02)
        )
        self.assertNotEqual(decision.type, "fire")

    def test_crowd_needs_enough_people(self):
        self.assertFalse(decide(Observation(kind="crowd", person_count=2)).publish)
        decision = decide(Observation(kind="crowd", person_count=5))
        self.assertEqual(decision.type, "crowd")

    def test_growing_warm_patch_on_the_slope_is_fire(self):
        decision = decide(Observation(zone="slope", duration_s=25, area_grow=2.0, warm_ratio=0.2))
        self.assertEqual(decision.type, "fire")

    def test_roundabout_point_is_not_sky(self):
        self.assertEqual(assign_zone(0.12, 0.9, ZONES), "roundabout")
        self.assertEqual(assign_zone(0.5, 0.05, ZONES), "sky")


class MotionTests(unittest.TestCase):
    def test_a_moving_blob_ends_as_one_track(self):
        zones = {"priority": ["road"], "polygons": {"road": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "exclude": []}
        detector = MotionDetector(zones, motion_width=160, min_track_frames=3, warmup_frames=4)
        base = np.full((90, 160, 3), 40, dtype=np.uint8)
        for index in range(4):
            detector.step(base, index)
        ended = []
        for index in range(6):
            frame = base.copy()
            x = 20 + index * 12
            frame[30:50, x : x + 16] = 255
            ended.extend(detector.step(frame, 10 + index).ended)
        for index in range(3):
            ended.extend(detector.step(base, 20 + index).ended)
        self.assertEqual(len(ended), 1)
        self.assertGreater(ended[0].travel, 0.03)

    def test_a_full_frame_flash_is_ignored(self):
        detector = MotionDetector(ZONES, motion_width=160, warmup_frames=4)
        base = np.full((90, 160, 3), 30, dtype=np.uint8)
        for index in range(4):
            detector.step(base, index)
        white = np.full_like(base, 255)
        step = detector.step(white, 10)
        self.assertTrue(step.global_change)
        self.assertEqual(step.ended, [])


class GtfsTests(unittest.TestCase):
    def test_one_nearby_departure(self):
        folder = ROOT / "tests" / "fixtures" / "gtfs"
        rows = load_feed(folder, "fixture", 44.179, 5.2663, 3000)
        index = GtfsIndex(folder, [], 44.179, 5.2663)
        index.rows = rows
        when = datetime(2026, 9, 24, 10, 5, tzinfo=ZoneInfo("Europe/Paris"))
        trips = index.trips_at(when, 15)
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0].route, "Navette")

    def test_far_stop_is_ignored(self):
        folder = ROOT / "tests" / "fixtures" / "gtfs"
        rows = load_feed(folder, "fixture", 44.179, 5.2663, 3000)
        self.assertTrue(all(row["stop_name"] != "Avignon" for row in rows))


class SceneTests(unittest.TestCase):
    def test_noon_is_day_and_deep_night_is_night(self):
        paris = ZoneInfo("Europe/Paris")
        self.assertEqual(solar_period(datetime(2026, 6, 21, 13, 0, tzinfo=paris), 44.179, 5.2663), "day")
        self.assertEqual(solar_period(datetime(2026, 6, 21, 2, 0, tzinfo=paris), 44.179, 5.2663), "night")

    def test_weather_words(self):
        self.assertEqual(weather_label(0), "ciel dégagé")
        self.assertEqual(weather_label(45), "brouillard")
        self.assertEqual(weather_label(95), "orage")

    def test_blue_sky_is_clear_and_a_dark_frame_is_night(self):
        blue = np.zeros((80, 160, 3), dtype=np.uint8)
        blue[:] = (210, 120, 30)
        dark = np.zeros((80, 160, 3), dtype=np.uint8)
        dark[:] = (8, 8, 8)
        gray = np.zeros((80, 160, 3), dtype=np.uint8)
        gray[:] = (150, 150, 150)
        self.assertEqual(read_sky(blue), "ciel dégagé")
        self.assertEqual(read_sky(dark), "nuit")
        self.assertEqual(read_sky(gray), "brouillard")

    def test_only_the_last_bulletin_is_kept(self):
        folder = ROOT / "data" / "view-test"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "view.json"
        path.unlink(missing_ok=True)
        log = ViewLog(path, every_s=900, change_s=120)
        blue = np.zeros((40, 80, 3), dtype=np.uint8)
        blue[:] = (210, 120, 30)
        start = datetime(2026, 9, 25, 16, 0, tzinfo=ZoneInfo("UTC"))
        log.note(blue, "peu nuageux", 21.2, start, "day")
        log.note(blue, "peu nuageux", 21, start.replace(minute=5), "day")
        self.assertEqual(log.last["webcam"], "ciel dégagé")
        self.assertEqual(log.last["api"], "peu nuageux")
        self.assertEqual(log.last["temp_c"], 21)
        self.assertEqual(log.last["period"], "day")
        self.assertFalse(log.last["moon"])
        self.assertEqual(json.loads(path.read_text())["last"]["webcam"], "ciel dégagé")
        path.unlink(missing_ok=True)
        (folder / "view.jpg").unlink(missing_ok=True)

    def test_the_moon_is_found_over_a_dark_sky(self):
        night = np.zeros((200, 400, 3), dtype=np.uint8)
        night[:] = (30, 28, 25)
        cv2.circle(night, (120, 40), 8, (245, 245, 240), -1)
        self.assertTrue(moon_in_sky(night))
        self.assertFalse(moon_in_sky(np.full((200, 400, 3), 40, dtype=np.uint8)))

    def test_the_summit_beacon_is_not_the_moon(self):
        night = np.zeros((200, 400, 3), dtype=np.uint8)
        night[:] = (30, 28, 25)
        cv2.circle(night, (202, 55), 8, (245, 245, 240), -1)
        beacon = [{"cx": 0.505, "cy": 0.275, "r": 0.035}]
        self.assertFalse(moon_in_sky(night, beacon))
    def test_onnx_accepts_a_frame(self):
        path = ROOT / "models" / "yolo11n.onnx"
        if not path.is_file():
            self.skipTest("models/yolo11n.onnx absent")
        from watcher.detect import YoloDetector

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.assertEqual(YoloDetector(str(path)).detect(frame), [])


class SkyTests(unittest.TestCase):
    def test_archive_returns_the_aircraft_in_the_window(self):
        path = ROOT / "data" / "sky-test.jsonl"
        archive = SkyArchive(path, [44.0, 5.0, 44.3, 5.5])
        path.write_text(
            "\n".join(
                [
                    json.dumps({"t": 1000, "aircraft": [{"icao24": "aaa", "callsign": "OLD", "altitude_m": 1000}]}),
                    json.dumps({"t": 1500, "aircraft": [{"icao24": "bbb", "callsign": "NOW", "altitude_m": 2000}]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        found = archive.around(1505, 30)
        path.unlink()
        self.assertEqual([item["callsign"] for item in found], ["NOW"])

    def test_a_reading_already_held_spends_no_question(self):
        path = ROOT / "data" / "sky-test.jsonl"
        archive = SkyArchive(path, [44.0, 5.0, 44.3, 5.5])
        archive.poll = lambda now=None: self.fail("OpenSky ne devrait pas être appelé")
        path.write_text(
            json.dumps({"t": 1500, "aircraft": [{"icao24": "bbb", "callsign": "NOW"}]}) + "\n",
            encoding="utf-8",
        )
        found = archive.ask(1505, 30)
        path.unlink()
        self.assertEqual([item["callsign"] for item in found], ["NOW"])

    def test_two_crossings_in_the_same_quiet_period_share_one_question(self):
        path = ROOT / "data" / "sky-test.jsonl"
        archive = SkyArchive(path, [44.0, 5.0, 44.3, 5.5], quiet_s=600)
        asked = []
        archive.poll = lambda now=None: asked.append(now)
        archive.ask(time.time(), 30)
        archive.ask(time.time(), 30)
        if path.exists():
            path.unlink()
        self.assertEqual(len(asked), 1)

    def test_a_band_too_low_on_the_roadway_is_the_tarmac(self):
        flat = Observation(zone="roundabout", surface="roundabout", travel=0.2, width_m=2.2, height_m=0.29,
                           detections=[Detection(cls="car", conf=0.8)])
        self.assertEqual(decide(flat).type, "motion")
        self.assertEqual(decide(flat).reason, "tarmac")
        car = Observation(zone="roundabout", surface="roundabout", travel=0.2, width_m=2.5, height_m=0.79,
                          detections=[Detection(cls="car", conf=0.8)], min_conf={"car": 0.4})
        self.assertEqual(decide(car).type, "vehicle")

    def test_a_metre_wide_on_the_roundabout_is_not_a_vehicle(self):
        speck = Observation(zone="other", surface="roundabout", period="night", travel=0.2,
                            width_m=1.13, height_m=0.98)
        self.assertEqual(decide(speck).type, "motion")
        self.assertEqual(decide(speck).reason, "too_small")
        car = Observation(zone="other", surface="roundabout", period="night", travel=0.2,
                          width_m=2.6, height_m=0.98)
        self.assertEqual(decide(car).type, "vehicle")

    def test_what_stands_still_on_the_island_is_the_furniture(self):
        planted = Observation(zone="roundabout", surface="island", travel=0.0, width_m=1.6, height_m=1.26,
                              detections=[Detection(cls="person", conf=0.73)])
        self.assertEqual(decide(planted).type, "motion")
        self.assertEqual(decide(planted).reason, "island")
        crossing = Observation(zone="roundabout", surface="island", travel=0.2, width_m=3.0, height_m=1.4,
                               detections=[Detection(cls="car", conf=0.8)], min_conf={"car": 0.4})
        self.assertEqual(decide(crossing).type, "vehicle")

    def test_the_lamp_of_the_summit_mast_is_not_a_fire(self):
        common = dict(zone="slope", surface="forest", duration_s=30, warm_ratio=0.5,
                      area_grow=4.0, travel=0.0, period="night", width_m=6.0)
        lamp = Observation(landmark="Émetteur du mont Ventoux", **common)
        self.assertEqual(decide(lamp).type, "motion")
        self.assertEqual(decide(lamp).reason, "beacon")
        self.assertEqual(decide(Observation(**common)).type, "fire")

    def test_what_has_just_caught_is_a_start_not_a_blaze(self):
        common = dict(zone="slope", surface="forest", duration_s=30, warm_ratio=0.5,
                      area_grow=4.0, travel=0.0, period="night", width_m=6.0)
        self.assertEqual(decide(Observation(**common)).label, "Départ de feu")
        held = dict(common, duration_s=900)
        self.assertEqual(decide(Observation(**held)).label, "Incendie")

    def test_a_plume_that_climbs_into_the_sky_keeps_its_track(self):
        detector = MotionDetector(ZONES, motion_width=640, min_track_frames=1)
        detector.tracks = [Track(id=1, zone="slope", frames=4, centroid=(0.5, 0.5), first_centroid=(0.5, 0.62))]
        self.assertEqual(detector._match(0.5, 0.42, "sky", {0}), 0)
        self.assertEqual(detector.tracks[0].zone, "slope")

    def test_only_something_longer_than_a_car_opens_the_timetable(self):
        cfg = {"min_conf": 0.35}
        road = Track(id=1, zone="road")
        sky = Track(id=2, zone="sky")
        car = [Detection(cls="car", conf=0.9)]
        coach = [Detection(cls="bus", conf=0.6)]
        self.assertFalse(_might_be_bus(road, car, 3.2, cfg))
        self.assertTrue(_might_be_bus(road, car, 11.0, cfg))
        self.assertTrue(_might_be_bus(road, coach, 3.2, cfg))
        self.assertFalse(_might_be_bus(sky, coach, 11.0, cfg))

    def test_a_point_that_did_not_move_asks_nothing(self):
        cfg = {"min_travel": 0.01, "max_sky_area": 0.02}
        still = Track(id=1, zone="sky", area_ratio=0.001, first_centroid=(0.5, 0.5), centroid=(0.5, 0.5))
        cloud = Track(id=2, zone="sky", area_ratio=0.5, first_centroid=(0.1, 0.5), centroid=(0.9, 0.5))
        plane = Track(id=3, zone="sky", area_ratio=0.001, first_centroid=(0.1, 0.5), centroid=(0.9, 0.5))
        self.assertFalse(_crossed_sky(still, cfg))
        self.assertFalse(_crossed_sky(cloud, cfg))
        self.assertTrue(_crossed_sky(plane, cfg))


class StoreTests(unittest.TestCase):
    def test_a_burst_becomes_one_passage_and_keeps_the_correction(self):
        events = [
            {"id": "a", "t": "2026-09-25T08:52:04Z", "type": "motion", "label": "Mouvement", "zone": "other", "confidence": 0.3, "thumb": "data/thumbs/a.jpg", "detail": {}},
            {"id": "b", "t": "2026-09-25T08:52:21Z", "type": "car", "label": "Voiture", "zone": "roundabout", "confidence": 0.75, "thumb": "data/thumbs/b.jpg", "review": "accepted", "detail": {"correction": "Estafette", "context": "de jour, ciel dégagé"}},
            {"id": "c", "t": "2026-09-25T08:52:21Z", "type": "motion", "label": "Mouvement sur la route", "zone": "road", "confidence": 0.3, "thumb": "data/thumbs/c.jpg", "detail": {}},
        ]
        folded = fold_events(events)
        self.assertEqual(len(folded), 1)
        self.assertEqual(folded[0]["label"], "Estafette")
        self.assertEqual(folded[0]["review"], "accepted")
        self.assertGreaterEqual(folded[0]["detail"]["count"], 3)

    def test_old_events_are_dropped(self):
        root = ROOT / "data" / "store-test"
        root.mkdir(parents=True, exist_ok=True)
        store = Store(root, history_days=30)
        old = datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC"))
        store.add_event(old, "car", "Voiture", "road", 0.9, b"", {})
        store.prune(datetime(2026, 9, 24, tzinfo=ZoneInfo("UTC")))
        self.assertEqual(store.events, [])
        (root / "events.json").unlink(missing_ok=True)


class ThumbTests(unittest.TestCase):
    def test_motion_box_is_drawn_on_the_photo(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        image[:] = (30, 70, 30)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        marked = cv2.imdecode(np.frombuffer(small_jpeg(encoded.tobytes(), box=[0.4, 0.35, 0.1, 0.12]), dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = marked.shape[:2]
        subject = marked[int(0.41 * height), int(0.45 * width)]
        self.assertLess(int(subject[2]), 80)
        self.assertGreater(int(marked[:, :, 2].max()), 120)


class ReviewTests(unittest.TestCase):
    def test_motion_can_be_named_car_bus_or_wrong(self):
        body = "event_id: m1\nverdict: accepted\nlecture: Mouvement\nclasse: voiture\n"
        self.assertEqual(parse_review(body, "valide"), ("m1", "accepted", "voiture"))
        events = [{"id": "m1", "type": "motion", "label": "Mouvement", "detail": {}}]
        learning = {}
        self.assertTrue(apply_review(events, learning, "m1", "accepted", "voiture"))
        self.assertEqual(events[0]["type"], "vehicle")
        self.assertEqual(events[0]["label"], "Voiture")
        self.assertEqual(events[0]["detail"]["correction"], "Voiture")
        self.assertTrue(apply_review(events, learning, "m1", "accepted", "bus"))
        self.assertEqual(events[0]["type"], "bus")
        self.assertEqual(events[0]["label"], "Bus")
        self.assertTrue(apply_review(events, learning, "m1", "rejected", ""))
        self.assertEqual(events[0]["review"], "rejected")


if __name__ == "__main__":
    unittest.main()
