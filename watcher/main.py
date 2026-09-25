"""Watch the Mont Serein stream: motion, then a name, then the history."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from watcher.config import load_config
from watcher.detect import YoloDetector, count_persons
from watcher.drive import DriveUploader
from watcher.geometry import load_zones
from watcher.gtfs import GtfsIndex, PARIS
from watcher.motion import MotionDetector, warm_ratio
from watcher.naming import Observation, decide
from watcher.opensky import SkyArchive
from watcher.publish import publish
from watcher.store import Store

log = logging.getLogger("ventoux")
CLIP_TYPES = {"plane", "bus", "fire", "crowd"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    root = Path(cfg["_root"])
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
    )
    camera = cfg["camera"]
    gtfs = GtfsIndex(root / "data" / "gtfs", cfg["gtfs"], camera["lat"], camera["lon"], cfg["gtfs_radius_m"])
    store = Store(root / "data", cfg["history_days"])
    drive = DriveUploader(str(root / cfg["drive"]["credentials"]), cfg["drive"].get("folder_id") or "")
    ring: deque[tuple[float, bytes]] = deque(maxlen=14)
    pending: list[dict] = []
    crowd_hits: deque[tuple[float, int]] = deque()
    last_crowd = 0.0
    last_fire: dict[str, float] = {}
    last_sky = 0.0
    last_gtfs = 0.0
    last_publish = 0.0

    while True:
        try:
            for frame in _frames(cfg["stream_url"]):
                now = time.time()
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    ring.append((now, encoded.tobytes()))
                if now - last_sky >= cfg["opensky"]["poll_s"]:
                    sky.poll(now)
                    last_sky = now
                if now - last_gtfs >= cfg["gtfs_refresh_s"]:
                    gtfs.refresh()
                    last_gtfs = now
                step = motion.step(frame, now)
                for track in step.ended:
                    _on_track(track, now, cfg, yolo, sky, gtfs, store, last_fire, pending)
                if step.roundabout_motion:
                    last_crowd = _crowd(frame, now, cfg, yolo, zones, crowd_hits, last_crowd, store, pending)
                _flush_clips(pending, ring, now, drive, store)
                if store.dirty and now - last_publish >= cfg["publish_interval_s"]:
                    publish(root)
                    store.dirty = False
                    last_publish = now
        except Exception:
            log.exception("Flux interrompu, nouvel essai dans 10 s")
            time.sleep(10)


def _on_track(track, now, cfg, yolo, sky, gtfs, store, last_fire, pending) -> None:
    frame = cv2.imdecode(np.frombuffer(track.best_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if track.best_jpeg else None
    detections = yolo.detect(frame, track.bbox) if frame is not None else []
    when = datetime.fromtimestamp(track.updated, timezone.utc)
    aircraft = sky.around(track.updated, cfg["opensky"]["match_window_s"]) if track.zone == "sky" else []
    trips = gtfs.trips_at(when.astimezone(PARIS), cfg["gtfs_window_min"]) if track.zone in {"road", "roundabout"} else []
    duration = max(0.0, track.updated - track.started)
    fire_ready = track.zone == "slope" and now - last_fire.get("fire", 0.0) >= cfg["fire"]["cooldown_s"]
    obs = Observation(
        zone=track.zone,
        detections=detections,
        aircraft=aircraft,
        trips=trips,
        travel=track.travel,
        area_ratio=track.area_ratio,
        duration_s=duration if fire_ready else 0.0,
        warm_ratio=warm_ratio(track.best_jpeg) if fire_ready else 0.0,
        area_grow=track.area_grow,
        min_travel=cfg["min_travel"],
        max_sky_area=cfg["max_sky_area"],
        min_conf=cfg["min_conf"],
        fire_sustain_s=cfg["fire"]["sustain_s"],
        fire_grow=cfg["fire"]["grow_ratio"],
        fire_warm=cfg["fire"]["warm_ratio"],
    )
    decision = decide(obs)
    if not decision.publish:
        store.add_candidate(when, track.zone, decision.reason, decision.detail)
        log.info("Candidat %s %s", track.zone, decision.reason)
        return
    if decision.type == "fire":
        last_fire["fire"] = now
    event = store.add_event(when, decision.type, decision.label, track.zone, decision.confidence, track.best_jpeg, decision.detail)
    log.info("Publié %s %s", decision.type, decision.label)
    if decision.type in CLIP_TYPES:
        pending.append({"id": event["id"], "after": now + 4, "started": track.started - 8})


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
