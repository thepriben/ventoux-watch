import json
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from watcher.geometry import assign_zone
from watcher.gtfs import GtfsIndex, load_feed
from watcher.motion import MotionDetector
from watcher.naming import Detection, Observation, Trip, choose_aircraft, decide
from watcher.opensky import SkyArchive
from watcher.store import Store

ROOT = Path(__file__).resolve().parents[1]
ZONES = json.loads((ROOT / "config" / "zones.json").read_text())


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

    def test_sky_motion_without_a_single_plane_is_held(self):
        decision = decide(Observation(zone="sky", travel=0.05, area_ratio=0.001, aircraft=[]))
        self.assertFalse(decision.publish)
        self.assertEqual(decision.reason, "none")

    def test_sky_motion_publishes_the_callsign(self):
        decision = decide(
            Observation(
                zone="sky",
                travel=0.05,
                area_ratio=0.001,
                aircraft=[{"icao24": "394c12", "callsign": "AFR472", "altitude_m": 4200}],
            )
        )
        self.assertTrue(decision.publish)
        self.assertEqual(decision.type, "plane")
        self.assertEqual(decision.label, "AFR472")

    def test_cloud_sized_sky_blob_is_held(self):
        decision = decide(Observation(zone="sky", travel=0.2, area_ratio=0.2, aircraft=[{"icao24": "a", "callsign": "X", "altitude_m": 1000}]))
        self.assertFalse(decision.publish)

    def test_car_on_the_road_is_published(self):
        decision = decide(Observation(zone="road", travel=0.08, detections=[Detection("car", 0.8)]))
        self.assertEqual(decision.type, "car")
        self.assertEqual(decision.label, "Voiture")

    def test_static_blob_is_not_a_car(self):
        decision = decide(Observation(zone="roundabout", travel=0.0, detections=[Detection("car", 0.9)]))
        self.assertFalse(decision.publish)

    def test_bus_with_one_trip_uses_the_line(self):
        trip = Trip("Navette", "Mont Serein", "Chalet", "10:00:00", "transcove")
        decision = decide(Observation(zone="roundabout", travel=0.05, detections=[Detection("bus", 0.7)], trips=[trip]))
        self.assertEqual(decision.label, "Bus Navette")
        self.assertEqual(decision.detail["scheduled"], "10:00:00")

    def test_bus_with_two_trips_is_not_given_a_line(self):
        trips = [
            Trip("A", "Un", "Stop", "10:00:00", "zou"),
            Trip("B", "Deux", "Stop", "10:05:00", "zou"),
        ]
        decision = decide(Observation(zone="road", travel=0.05, detections=[Detection("bus", 0.8)], trips=trips))
        self.assertEqual(decision.label, "Bus")
        self.assertEqual(decision.reason, "model_only")

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


class ModelTests(unittest.TestCase):
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


class StoreTests(unittest.TestCase):
    def test_old_events_are_dropped(self):
        root = ROOT / "data" / "store-test"
        root.mkdir(parents=True, exist_ok=True)
        store = Store(root, history_days=30)
        old = datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC"))
        store.add_event(old, "car", "Voiture", "road", 0.9, b"", {})
        store.prune(datetime(2026, 9, 24, tzinfo=ZoneInfo("UTC")))
        self.assertEqual(store.events, [])
        (root / "events.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
