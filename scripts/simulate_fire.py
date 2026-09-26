"""Put a fire in front of the watcher and see when it calls it.

The plume is drawn; everything that judges it is the live code. The frames go
through the real background subtractor, the real tracker, the real smoke and
flame measurements and the real naming, so the answer this prints is the
answer the watcher would give on a real fire in that spot.

    .venv/bin/python -m scripts.simulate_fire --night --publish
    .venv/bin/python -m scripts.simulate_fire --frame data/thumbs/x.jpg --publish

Published events say so: the label begins with "Simulation", and the entry
carries simulation: true. Nothing can mistake one for a fire that happened.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from watcher.motion import MotionDetector, smoke_ratio, warm_ratio
from watcher.naming import Observation, decide
from watcher.scenemap import FLAMMABLE, SceneMap
from watcher.simulate import plume, sensor_noise
from watcher.store import Store

ROOT = Path(__file__).resolve().parents[1]
WARMUP_S = 12
STREAM_FRAME = "live"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default=STREAM_FRAME, help="photo de départ, ou 'live'")
    parser.add_argument("--spot", default="", help="foyer, en x,y normalisés")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--night", action="store_true", help="feu de nuit : la flamme, pas la fumée")
    parser.add_argument("--ember", type=float, default=0.0, metavar="S",
                        help="départ par la couleur : la flamme seule pendant S secondes, la fumée ensuite")
    parser.add_argument("--publish", action="store_true", help="met la simulation dans le flux")
    parser.add_argument("--at", default="", help="horodatage ISO, par défaut celui de la photo ou maintenant")
    args = parser.parse_args()

    cfg = json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    zones = json.loads((ROOT / "config" / "zones.json").read_text(encoding="utf-8"))
    scene_map = SceneMap.load(ROOT / "config" / "scene.json")
    base = _base_frame(args.frame, cfg)
    if base is None:
        print("Pas de photo de départ")
        return 1
    spot = _spot(args.spot, scene_map)
    when = _when(args.at, args.frame)
    period = "night" if args.night else "day"
    print(f"Foyer en {spot[0]:.3f}, {spot[1]:.3f} sur {scene_map.surface_at(*spot) or 'pente'}"
          f" à {scene_map.distance_at(*spot):.0f} m, de {period}"
          + (f", couleur seule pendant {args.ember:.0f} s" if args.ember else ""))

    motion = MotionDetector(
        zones,
        motion_width=cfg["motion_width"],
        min_track_frames=cfg["min_track_frames"],
        max_foreground_ratio=cfg["max_foreground_ratio"],
    )
    start = when.timestamp()
    for second in range(WARMUP_S):
        motion.step(sensor_noise(base, seed=second), start - WARMUP_S + second)

    raised = None
    for second in range(1, args.seconds + 1):
        frame = plume(sensor_noise(base, seed=100 + second), spot, second,
                      flame=args.night or args.ember > 0, seed=7, smoke_after_s=args.ember)
        now = start + second
        motion.step(frame, now)
        track = _widest(motion.tracks)
        if track is None:
            continue
        decision, obs = _judge(track, frame, now, cfg, scene_map, period)
        alert = decision.type == "fire"
        mark = f"ALERTE {decision.label}" if alert else decision.reason
        print(f"  {second:3d} s  âge {obs.duration_s:4.0f} s  fumée {obs.smoke_ratio:.2f}  flamme {obs.warm_ratio:.2f}"
              f"  montée {obs.rise:.3f}  croissance {obs.area_grow:.1f}  -> {mark}")
        if alert and raised is None:
            raised = (second, decision, frame, track)

    if raised is None:
        print("Aucune alerte. Le feu n'a pas été vu.")
        return 1
    second, decision, frame, track = raised
    print(f"\nAlerte à la {second}e seconde : {decision.label}")
    if args.publish:
        kind = (f"couleur seule pendant {args.ember:.0f} s puis fumée"
                if args.ember else "panache")
        _publish(when, second, decision, frame, track, scene_map, period,
                 f"Simulation : {kind} dessiné sur la vue réelle, alerte à la {second}e seconde.")
    return 0


def _judge(track, frame, now, cfg, scene_map, period):
    """The same reading the watcher makes, on the same numbers."""
    height, width = frame.shape[:2]
    x, y, w, h = track.bbox
    box = (x / width, y / height, max(w, 1) / width, max(h, 1) / height)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    jpeg = encoded.tobytes() if ok else b""
    obs = Observation(
        zone=track.zone,
        duration_s=max(0.0, now - track.started),
        warm_ratio=warm_ratio(jpeg, track.bbox),
        smoke_ratio=smoke_ratio(jpeg, track.bbox),
        rise=track.rise,
        area_grow=track.area_grow,
        area_ratio=track.area_ratio,
        travel=track.travel,
        width_m=scene_map.metres_across(box),
        height_m=scene_map.metres_tall(box),
        surface=scene_map.surface_under(box),
        near_road=scene_map.drivable_near(box),
        min_travel=cfg["min_travel"],
        max_sky_area=cfg["max_sky_area"],
        min_conf=cfg["min_conf"],
        fire_sustain_s=cfg["fire"]["sustain_s"],
        fire_grow=cfg["fire"]["grow_ratio"],
        fire_warm=cfg["fire"]["warm_ratio"],
        fire_smoke=float(cfg["fire"].get("smoke_ratio") or 0.35),
        fire_rise=float(cfg["fire"].get("rise") or 0.008),
        period=period,
        weather="ciel dégagé",
    )
    return decide(obs), obs


def _publish(when, second, decision, frame, track, scene_map, period, reading) -> None:
    store = Store(ROOT / "data", 30)
    height, width = frame.shape[:2]
    x, y, w, h = track.bbox
    box = [round(x / width, 4), round(y / height, 4), round(max(w, 1) / width, 4), round(max(h, 1) / height, 4)]
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    detail = dict(decision.detail)
    detail.update(
        {
            "simulation": True,
            "reading": reading,
            "period": period,
            "box": box,
        }
    )
    event = store.add_event(
        when,
        "fire",
        f"Simulation : {decision.label.lower()}",
        track.zone,
        decision.confidence,
        encoded.tobytes() if ok else b"",
        detail,
    )
    print(f"Publié dans le flux : {event['id']}  {event['label']}")


def _base_frame(source: str, cfg):
    if source != STREAM_FRAME:
        return cv2.imread(str(ROOT / source) if not Path(source).is_absolute() else source)
    import subprocess

    target = Path("/tmp/ventoux-sim.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", cfg["stream_url"], "-frames:v", "1", str(target)],
        check=False,
        timeout=90,
    )
    return cv2.imread(str(target)) if target.is_file() else None


def _spot(given: str, scene_map: SceneMap) -> tuple[float, float]:
    if given:
        x, y = given.split(",")
        return float(x), float(y)
    best = None
    for row, line in enumerate(scene_map.rows):
        for column, letter in enumerate(line):
            x, y = (column + 0.5) / len(line), (row + 0.5) / len(scene_map.rows)
            if scene_map.surface_at(x, y) not in FLAMMABLE:
                continue
            span = scene_map.distance_at(x, y)
            if not 150 <= span <= 700:
                continue
            if best is None or span > best[0]:
                best = (span, x, y)
    if best is None:
        return 0.5, 0.55
    return best[1], best[2]


def _when(given: str, source: str) -> datetime:
    if given:
        return datetime.fromisoformat(given.replace("Z", "+00:00")).astimezone(timezone.utc)
    stamp = Path(source).name[:20]
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _widest(tracks):
    return max(tracks, key=lambda item: item.area_ratio, default=None)


if __name__ == "__main__":
    raise SystemExit(main())
