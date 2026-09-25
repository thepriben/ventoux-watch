"""Public history and private candidates. Only named events are published."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

THUMB_WIDTH = 480
THUMB_QUALITY = 52
PASSAGE_ZONES = {"road", "roundabout", "other"}
RANK = {"fire": 6, "crowd": 5, "bus": 4, "vehicle": 3, "car": 3, "person": 3, "plane": 2, "motion": 1, "habit": 0}


class Store:
    def __init__(self, root: Path, history_days: int = 30):
        self.root = root
        self.history_days = history_days
        self.events_path = root / "events.json"
        self.thumbs = root / "thumbs"
        self.candidates_path = root / "candidates.jsonl"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.dirty = False
        self._seq = 0
        self.events = self._load()

    def add_event(self, when: datetime, type_: str, label: str, zone: str, confidence: float, jpeg: bytes, detail: dict) -> dict:
        stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self._seq += 1
        event = {
            "id": f"{stamp}-{type_}-{self._seq}",
            "t": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": type_,
            "label": label,
            "zone": zone,
            "confidence": round(confidence, 3),
            "thumb": "",
            "clip_url": "",
            "detail": detail,
        }
        host = open_passage(self.events, event)
        if host is not None:
            if (host.get("detail") or {}).get("correction") or not better_reading(event, host):
                _bump(host)
            else:
                _copy_reading(host, event)
                if jpeg:
                    _write_thumb(self.thumbs, host, small_jpeg(jpeg, box=(detail or {}).get("box")))
            self._write()
            self.dirty = True
            return host
        if jpeg:
            _write_thumb(self.thumbs, event, small_jpeg(jpeg, box=(detail or {}).get("box")))
        self.events.append(event)
        self.events.sort(key=lambda item: item["t"], reverse=True)
        self.prune(when)
        self._write()
        self.dirty = True
        return event

    def add_candidate(self, when: datetime, zone: str, reason: str, detail: dict) -> None:
        row = {
            "t": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zone": zone,
            "reason": reason,
            "detail": detail,
        }
        with self.candidates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def set_clip(self, event_id: str, url: str) -> None:
        for event in self.events:
            if event["id"] == event_id:
                event["clip_url"] = url
                self._write()
                self.dirty = True
                return

    def prune(self, now: datetime) -> None:
        cutoff = now.timestamp() - self.history_days * 86400
        kept = []
        for event in self.events:
            moment = datetime.strptime(event["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if moment.timestamp() >= cutoff:
                kept.append(event)
                continue
            thumb = self.root / "thumbs" / Path(event.get("thumb") or "").name
            if thumb.is_file():
                thumb.unlink()
        self.events = kept

    def _load(self) -> list[dict]:
        if not self.events_path.is_file():
            return []
        payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        return list(payload.get("events") or [])

    def _write(self) -> None:
        payload = {"events": self.events}
        self.events_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def passage_group(zone: str) -> str:
    if zone in PASSAGE_ZONES:
        return "passage"
    return zone or "other"


def gap_seconds(group: str, same_label: bool) -> int:
    if group == "sky":
        return 600 if same_label else 60
    return 60


def event_time(event: dict) -> datetime:
    return datetime.strptime(event["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def open_passage(events: list[dict], event: dict) -> dict | None:
    group = passage_group(event.get("zone", ""))
    when = event_time(event)
    newest = None
    for item in events:
        if passage_group(item.get("zone", "")) != group:
            continue
        if newest is None or event_time(item) > event_time(newest):
            newest = item
    if newest is None:
        return None
    same = newest.get("label") == event.get("label")
    if abs((when - event_time(newest)).total_seconds()) <= gap_seconds(group, same):
        return newest
    return None


def better_reading(new: dict, old: dict) -> bool:
    if (old.get("detail") or {}).get("correction"):
        return False
    if (new.get("detail") or {}).get("correction"):
        return True
    new_rank = RANK.get(new.get("type"), 1)
    old_rank = RANK.get(old.get("type"), 1)
    if new_rank != old_rank:
        return new_rank > old_rank
    return float(new.get("confidence") or 0) > float(old.get("confidence") or 0) + 0.05


def _bump(host: dict) -> None:
    detail = dict(host.get("detail") or {})
    detail["count"] = int(detail.get("count") or 1) + 1
    host["detail"] = detail


def _copy_reading(host: dict, event: dict) -> None:
    count = int((host.get("detail") or {}).get("count") or 1) + 1
    review = host.get("review")
    clip = host.get("clip_url")
    host["type"] = event.get("type")
    host["label"] = event.get("label")
    host["zone"] = event.get("zone")
    host["confidence"] = event.get("confidence")
    detail = dict(event.get("detail") or {})
    detail["count"] = count
    host["detail"] = detail
    if event.get("thumb"):
        host["thumb"] = event["thumb"]
    if review:
        host["review"] = review
    if clip:
        host["clip_url"] = clip


def _write_thumb(folder: Path, event: dict, jpeg: bytes) -> None:
    name = f"{event['id']}.jpg"
    (folder / name).write_bytes(jpeg)
    event["thumb"] = f"data/thumbs/{name}"


def fold_events(events: list[dict]) -> list[dict]:
    """One card per passage. Photos of the folded lines are left on disk."""
    kept: list[dict] = []
    for source in sorted(events, key=event_time):
        event = dict(source)
        event["detail"] = dict(source.get("detail") or {})
        host = open_passage(kept, event)
        if host is None:
            event["detail"]["count"] = int(event["detail"].get("count") or 1)
            kept.append(event)
            continue
        if (host.get("detail") or {}).get("correction") or not better_reading(event, host):
            _bump(host)
            continue
        _copy_reading(host, event)
    kept.sort(key=event_time, reverse=True)
    return kept


def passage_group(zone: str) -> str:
    if zone in PASSAGE_ZONES:
        return "passage"
    return zone or "other"


def gap_seconds(group: str, same_label: bool) -> int:
    if group == "sky":
        return 600 if same_label else 60
    return 60


def event_time(event: dict) -> datetime:
    return datetime.strptime(event["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def open_passage(events: list[dict], event: dict) -> dict | None:
    group = passage_group(event.get("zone", ""))
    when = event_time(event)
    newest = None
    for item in events:
        if passage_group(item.get("zone", "")) != group:
            continue
        if newest is None or event_time(item) > event_time(newest):
            newest = item
    if newest is None:
        return None
    same = newest.get("label") == event.get("label")
    if abs((when - event_time(newest)).total_seconds()) <= gap_seconds(group, same):
        return newest
    return None


def better_reading(new: dict, old: dict) -> bool:
    if (old.get("detail") or {}).get("correction"):
        return False
    if (new.get("detail") or {}).get("correction"):
        return True
    new_rank = RANK.get(new.get("type"), 1)
    old_rank = RANK.get(old.get("type"), 1)
    if new_rank != old_rank:
        return new_rank > old_rank
    return float(new.get("confidence") or 0) > float(old.get("confidence") or 0) + 0.05


def _bump(host: dict) -> None:
    detail = dict(host.get("detail") or {})
    detail["count"] = int(detail.get("count") or 1) + 1
    host["detail"] = detail


def _copy_reading(host: dict, event: dict) -> None:
    count = int((host.get("detail") or {}).get("count") or 1) + 1
    correction = (event.get("detail") or {}).get("correction") or (host.get("detail") or {}).get("correction")
    review = event.get("review") or host.get("review")
    clip = host.get("clip_url") or event.get("clip_url") or ""
    host["type"] = event.get("type") or host.get("type")
    host["label"] = event.get("label") or host.get("label")
    host["zone"] = event.get("zone") or host.get("zone")
    host["confidence"] = event.get("confidence", host.get("confidence"))
    detail = dict(event.get("detail") or {})
    if correction:
        detail["correction"] = correction
        if (event.get("detail") or {}).get("correction"):
            host["label"] = correction
    detail["count"] = count
    host["detail"] = detail
    if event.get("thumb"):
        host["thumb"] = event["thumb"]
    if review:
        host["review"] = review
    if clip:
        host["clip_url"] = clip


def _write_thumb(folder: Path, event: dict, jpeg: bytes) -> None:
    name = f"{event['id']}.jpg"
    (folder / name).write_bytes(jpeg)
    event["thumb"] = f"data/thumbs/{name}"


def fold_events(events: list[dict]) -> list[dict]:
    """One card per passage. Photos of the folded lines are left on disk."""
    kept: list[dict] = []
    for event in sorted(events, key=event_time):
        item = dict(event)
        item["detail"] = dict(event.get("detail") or {})
        host = open_passage(kept, item)
        if host is None:
            item["detail"].setdefault("count", 1)
            kept.append(item)
            continue
        if (host.get("detail") or {}).get("correction") or not better_reading(item, host):
            _bump(host)
            continue
        _copy_reading(host, item)
    kept.sort(key=event_time, reverse=True)
    return kept


def _outline(image: np.ndarray, box) -> None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    height, width = image.shape[:2]
    try:
        x = int(round(float(box[0]) * width))
        y = int(round(float(box[1]) * height))
        w = int(round(float(box[2]) * width))
        h = int(round(float(box[3]) * height))
    except (TypeError, ValueError):
        return
    pad_x = max(10, int(round(w * 0.45)))
    pad_y = max(10, int(round(h * 0.45)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width - 1, x + max(w, 1) + pad_x)
    y1 = min(height - 1, y + max(h, 1) + pad_y)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 210), 1)


def small_jpeg(jpeg: bytes, width: int = THUMB_WIDTH, quality: int = THUMB_QUALITY, box=None) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return jpeg
    height, frame_width = image.shape[:2]
    if frame_width > width:
        scale = width / float(frame_width)
        image = cv2.resize(image, (width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    _outline(image, box)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return jpeg
    return encoded.tobytes()
