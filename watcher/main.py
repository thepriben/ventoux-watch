"""Watch the Mont Serein stream: motion, then a name, then the history."""

from __future__ import annotations

import fcntl
import logging
import math
import os
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from watcher.airports import describe_route
from watcher.config import load_config
from watcher.detect import YoloDetector, body_colour, car_lights, count_persons
from watcher.drive import DriveUploader
from watcher.geometry import load_zones
from watcher.gtfs import GtfsIndex, PARIS
from watcher.memory import Memory
from watcher.motion import MotionDetector, smoke_ratio, warm_ratio
from watcher.naming import Observation, decide
from watcher.opensky import SkyArchive
from watcher.publish import publish
from watcher.scene import SceneReader, ViewLog, solar_azimuth, solar_elevation
from watcher.scenemap import FLAMMABLE, SceneMap
from watcher.store import Store

log = logging.getLogger("ventoux")
# Above this, on the ground, the thing is longer than a car and the timetable
# is worth opening.
BUS_LENGTH_M = 5.5
CLIP_TYPES = {"plane", "bus", "fire", "crowd"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    root = Path(cfg["_root"])
    if not _only_one(root / "data" / "watch.lock"):
        log.error("Un veilleur tourne déjà. Celui-ci s'arrête.")
        return
    zones = load_zones(root / cfg["zones"])
    motion = MotionDetector(
        zones,
        motion_width=cfg["motion_width"],
        min_track_frames=cfg["min_track_frames"],
        max_foreground_ratio=cfg["max_foreground_ratio"],
    )
    yolo = YoloDetector(str(root / cfg["model_path"]))
    if not yolo.ready:
        log.warning("Modèle absent (%s) : les voitures et bus attendront l'export ONNX", cfg["model_path"])
    sky = SkyArchive(
        root / "data" / "sky.jsonl",
        cfg["opensky"]["bbox"],
        cfg["opensky"]["retain_days"],
        cfg["opensky"].get("username") or "",
        cfg["opensky"].get("password") or "",
        cfg["opensky"].get("quiet_s", 240),
        cfg["opensky"].get("client_id") or "",
        cfg["opensky"].get("client_secret") or "",
    )
    camera = cfg["camera"]
    gtfs = GtfsIndex(root / "data" / "gtfs", cfg["gtfs"], camera["lat"], camera["lon"], cfg["gtfs_radius_m"])
    store = Store(root / "data", cfg["history_days"])
    scene = SceneReader(camera["lat"], camera["lon"])
    view = ViewLog(root / "data" / "view.json", exclude=zones.get("exclude") or [])
    memory = Memory(root / "data" / "learning.json")
    scene_map = SceneMap.load(root / "config" / "scene.json")
    if scene_map.ready:
        log.info("Carte de la scène : %d repères, calage %s", len(scene_map.landmarks), scene_map.pose.get("rms"))
    else:
        log.warning("Pas de config/scene.json : lance scripts/build_scene.py pour lire les surfaces")
    drive = DriveUploader(str(root / cfg["drive"]["credentials"]), cfg["drive"].get("folder_id") or "")
    ring: deque[tuple[float, bytes]] = deque(maxlen=14)
    pending: list[dict] = []
    crowd_hits: deque[tuple[float, int]] = deque()
    last_crowd = 0.0
    last_fire: dict[str, float] = {}
    alerted: set[int] = set()
    last_gtfs = 0.0
    last_publish = 0.0
    last_view = 0.0

    while True:
        try:
            for frame in _frames(cfg["stream_url"]):
                now = time.time()
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    ring.append((now, encoded.tobytes()))
                if now - last_gtfs >= cfg["gtfs_refresh_s"]:
                    gtfs.refresh()
                    last_gtfs = now
                if now - last_view >= 30:
                    moment = datetime.fromtimestamp(now, timezone.utc)
                    current = scene.read(frame, moment)
                    view.note(frame, current.weather, current.temperature_c, moment, current.period)
                    last_view = now
                step = motion.step(frame, now)
                for track in step.ended:
                    _on_track(track, now, cfg, yolo, sky, gtfs, store, last_fire, pending, scene, memory, scene_map)
                for track in _burning(motion.tracks, now, cfg, scene_map, alerted):
                    _on_track(track, now, cfg, yolo, sky, gtfs, store, last_fire, pending, scene, memory, scene_map)
                if step.roundabout_motion:
                    last_crowd = _crowd(frame, now, cfg, yolo, zones, crowd_hits, last_crowd, store, pending)
                _flush_clips(pending, ring, now, drive, store)
                if (store.dirty or view.dirty) and now - last_publish >= cfg["publish_interval_s"]:
                    publish(root)
                    store.dirty = False
                    view.dirty = False
                    last_publish = now
        except Exception:
            log.exception("Flux interrompu, nouvel essai dans 10 s")
            time.sleep(10)


def _on_track(track, now, cfg, yolo, sky, gtfs, store, last_fire, pending, scene, memory, scene_map=None) -> None:
    frame = cv2.imdecode(np.frombuffer(track.best_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if track.best_jpeg else None
    detections = yolo.detect(frame, track.bbox) if frame is not None else []
    when = datetime.fromtimestamp(track.updated, timezone.utc)
    current = scene.read(frame, when)
    scene_map = scene_map or SceneMap()
    box = _norm_box(frame, track)
    surface = scene_map.surface_under(box) if box else ""
    landmark = scene_map.landmark_at(box) if box else None
    lit = car_lights(frame, track.bbox) if frame is not None and current.period != "day" else 0.0
    aircraft = sky.ask(track.updated, cfg["opensky"]["match_window_s"]) if _crossed_sky(track, cfg) else []
    width_m = scene_map.metres_across(box) if box else 0.0
    trips = gtfs.trips_at(when.astimezone(PARIS), cfg["gtfs_window_min"]) if _might_be_bus(track, detections, width_m, cfg) else []
    duration = max(0.0, track.updated - track.started)
    on_fuel = surface in FLAMMABLE or (track.zone == "slope" and not surface)
    fire_ready = on_fuel and now - last_fire.get("fire", 0.0) >= cfg["fire"]["cooldown_s"]
    obs = Observation(
        zone=track.zone,
        detections=detections,
        aircraft=aircraft,
        trips=trips,
        travel=track.travel,
        area_ratio=track.area_ratio,
        duration_s=duration if fire_ready else 0.0,
        warm_ratio=warm_ratio(track.best_jpeg, track.best_bbox) if fire_ready else 0.0,
        smoke_ratio=smoke_ratio(track.best_jpeg, track.best_bbox) if fire_ready else 0.0,
        rise=track.rise,
        width_m=width_m,
        height_m=scene_map.metres_tall(box) if box else 0.0,
        area_grow=track.area_grow,
        min_travel=cfg["min_travel"],
        max_sky_area=cfg["max_sky_area"],
        min_conf=cfg["min_conf"],
        fire_sustain_s=cfg["fire"]["sustain_s"],
        fire_grow=cfg["fire"]["grow_ratio"],
        fire_warm=cfg["fire"]["warm_ratio"],
        fire_smoke=float(cfg["fire"].get("smoke_ratio") or 0.35),
        fire_rise=float(cfg["fire"].get("rise") or 0.008),
        period=current.period,
        weather=current.weather,
        surface=surface,
        near_road=scene_map.drivable_near(box) if box else True,
        colour=body_colour(frame, track.best_bbox) if frame is not None and current.period == "day" else "",
        landmark=(landmark or {}).get("name", ""),
        lit_ratio=lit,
        # The middle of the blob, not its corner: an aircraft is matched against
        # where the thing is, and a box records where it begins.
        at_x=box[0] + box[2] / 2 if box else -1.0,
        at_y=box[1] + box[3] / 2 if box else -1.0,
        sun_bearing=solar_azimuth(when, cfg["camera"]["lat"], cfg["camera"]["lon"]),
        sun_elevation=solar_elevation(when, cfg["camera"]["lat"], cfg["camera"]["lon"]),
        **_eye(cfg, scene_map),
    )
    decision = decide(obs)
    measured = {
        # What the rule actually weighed. Written down because a refusal with no
        # number behind it cannot be argued with later: the sky has turned down
        # thousands of things and left no way to tell a jet from a cloud edge.
        "travel": round(obs.travel, 4),
        "area_ratio": round(obs.area_ratio, 5),
        "duration_s": round(max(0.0, track.updated - track.started), 1),
        "frames": track.frames,
    }
    decision.detail.setdefault("measured", measured)
    if decision.type == "plane" and decision.detail.get("icao24"):
        # Asked now and not before: a route costs a call to OpenSky, and until
        # the rule has settled on one aircraft there is nothing to ask about.
        decision.detail.update(describe_route(sky.route(decision.detail["icao24"], track.updated)))
    if decision.type == "motion" and track.zone == "sky":
        store.add_candidate(when, track.zone, decision.reason, decision.detail)
        return
    if not decision.publish:
        store.add_candidate(when, track.zone, decision.reason, decision.detail)
        log.info("Candidat %s %s", track.zone, decision.reason)
        return
    if memory.observe(track.zone, track.centroid, decision) != "record":
        log.info("Compté sans nouvelle carte %s", decision.label)
        return
    if decision.type in {"motion", "habit"}:
        store.add_candidate(when, track.zone, decision.reason, decision.detail)
        return
    if decision.type == "fire":
        last_fire["fire"] = now
    if surface:
        decision.detail["surface"] = surface
    if box:
        decision.detail["box"] = [round(value, 4) for value in box]
    event = store.add_event(when, decision.type, decision.label, track.zone, decision.confidence, track.best_jpeg, decision.detail)
    if width_m >= BUS_LENGTH_M:
        close = store.keep_closeup(event, frame, track.best_bbox)
        if close:
            log.info("Recadrage gardé pour %s : %s", decision.label, close)
    log.info("Publié %s %s", decision.type, decision.label)
    if decision.type in CLIP_TYPES:
        pending.append({"id": event["id"], "after": now + 4, "started": track.started - 8})


_LOCK = None


def _only_one(path: Path) -> bool:
    """True when no other watcher holds the lock.

    Two watchers on one camera read the same frames and write the same history
    twice, and the second copy of an event is indistinguishable from a real one.
    The handle is kept in a module global on purpose: closed, the lock would be
    released and the guard would protect nothing.
    """
    global _LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOCK = path.open("w")
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _LOCK.close()
        _LOCK = None
        return False
    _LOCK.write(f"{os.getpid()}\n")
    _LOCK.flush()
    return True


def _eye(cfg, scene_map) -> dict:
    """Where the camera is and where it points, preferring the fitted pose.

    The figures typed into the config were a first guess; the pose in the scene
    file was fitted against surveyed marks and sits fourteen degrees off it in
    bearing. Asking whether an aircraft is in frame with the guess put it in the
    wrong part of the sky.
    """
    camera = cfg["camera"]
    pose = getattr(scene_map, "pose", None) or {}
    hfov = float(pose.get("hfov") or camera.get("fov") or 90)
    aspect = float(pose.get("aspect") or (16 / 9))
    vfov = 2 * math.degrees(math.atan(math.tan(math.radians(hfov / 2)) / aspect))
    return {
        "camera_lat": float(pose.get("lat") or camera["lat"]),
        "camera_lon": float(pose.get("lon") or camera["lon"]),
        "camera_ele": float(pose.get("ele") or camera.get("ele") or 1390),
        "camera_bearing": float(pose.get("yaw") if pose.get("yaw") is not None else (camera.get("bearing") or 140)),
        "camera_fov": hfov,
        "camera_pitch": float(pose.get("pitch") or 0.0),
        "camera_vfov": vfov,
    }


def _burning(tracks, now, cfg, scene_map, alerted: set) -> list:
    """Tracks that must be judged now, without waiting for them to end.

    A car is read when it has gone, which is soon enough. A fire never goes:
    the plume keeps growing and the track stays open, so waiting for the end
    would mean waiting for the fire to burn out. Once a shape has held on
    flammable ground for the sustain time, it is judged where it stands.
    """
    sustain = float(cfg["fire"]["sustain_s"])
    due = []
    live = {track.id for track in tracks}
    alerted.intersection_update(live)
    for track in tracks:
        if track.id in alerted or now - track.started < sustain:
            continue
        surface = scene_map.surface_under(_norm_box_of(track)) if scene_map.ready else ""
        if surface in FLAMMABLE or (track.zone == "slope" and not surface):
            alerted.add(track.id)
            due.append(track)
    return due


def _norm_box_of(track) -> tuple[float, float, float, float] | None:
    box = track.best_bbox if any(track.best_bbox) else track.bbox
    if not any(box) or not track.best_jpeg:
        return None
    frame = cv2.imdecode(np.frombuffer(track.best_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return _norm_box(frame, track)


def _might_be_bus(track, detections, width_m, cfg) -> bool:
    """Worth opening the timetable.

    A car has no departure time, so it is not worth reading the timetable for
    every one that passes. Either the model saw something long, or the ground
    width says the thing is longer than a car.
    """
    if track.zone not in {"road", "roundabout"}:
        return False
    if width_m >= BUS_LENGTH_M:
        return True
    return any(item.cls in {"bus", "truck"} and item.conf >= cfg["min_conf"] for item in detections)


def _crossed_sky(track, cfg) -> bool:
    """Worth asking OpenSky who was up there.

    A point that stayed put is a star or the mast beacon, and a wide patch is
    a cloud. Neither has a flight number, so neither spends a question.
    """
    if track.zone != "sky":
        return False
    return track.travel >= cfg["min_travel"] and track.area_ratio <= cfg["max_sky_area"]


def _norm_box(frame, track) -> tuple[float, float, float, float] | None:
    bbox = track.best_bbox if any(track.best_bbox) else track.bbox
    if frame is None or not any(bbox):
        return None
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    return x / width, y / height, max(w, 1) / width, max(h, 1) / height


def _crowd(frame, now, cfg, yolo, zones, hits, last_crowd, store, pending) -> float:
    if now - last_crowd < cfg["crowd"]["cooldown_s"]:
        return last_crowd
    polygon = zones["polygons"]["roundabout"]
    height, width = frame.shape[:2]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    bbox = (int(min(xs) * width), int(min(ys) * height), int((max(xs) - min(xs)) * width), int((max(ys) - min(ys)) * height))
    persons = count_persons(yolo.detect(frame, bbox))
    hits.append((now, persons))
    sustain = cfg["crowd"]["sustain_s"]
    while hits and now - hits[0][0] > sustain:
        hits.popleft()
    if hits and now - hits[0][0] >= sustain and all(count >= cfg["crowd"]["min_persons"] for _, count in hits):
        when = datetime.fromtimestamp(now, timezone.utc)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg = encoded.tobytes() if ok else b""
        event = store.add_event(when, "crowd", "Attroupement", "roundabout", 1.0, jpeg, {"persons": persons})
        pending.append({"id": event["id"], "after": now + 4, "started": now - 8})
        hits.clear()
        log.info("Publié attroupement (%s personnes)", persons)
        return now
    return last_crowd


def _flush_clips(pending, ring, now, drive, store) -> None:
    ready = [item for item in pending if now >= item["after"]]
    for item in ready:
        pending.remove(item)
        frames = [jpeg for stamp, jpeg in ring if item["started"] <= stamp <= now]
        if len(frames) < 2 or not drive.enabled:
            continue
        url = _encode_and_upload(frames, item["id"], drive)
        if url:
            store.set_clip(item["id"], url)


def _encode_and_upload(frames: list[bytes], event_id: str, drive: DriveUploader) -> str:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{event_id}.mp4"
        process = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-framerate", "1", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame)
        process.stdin.close()
        process.wait(timeout=60)
        if process.returncode != 0 or not path.is_file():
            return ""
        return drive.upload(path, path.name)


def _frames(url: str):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", url, "-an", "-vf", "fps=1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    buffer = b""
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
                if start < 0 or end < 0:
                    if start > 0:
                        buffer = buffer[start:]
                    elif start < 0 and len(buffer) > 1_000_000:
                        buffer = b""
                    break
                payload = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    finally:
        process.kill()


if __name__ == "__main__":
    main()
