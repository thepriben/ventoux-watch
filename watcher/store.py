"""Public history and private candidates. Only named events are published."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

THUMB_WIDTH = 480
THUMB_QUALITY = 52


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
        event_id = f"{stamp}-{type_}-{self._seq}"
        thumb_name = f"{event_id}.jpg"
        if jpeg:
            (self.thumbs / thumb_name).write_bytes(small_jpeg(jpeg))
        event = {
            "id": event_id,
            "t": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": type_,
            "label": label,
            "zone": zone,
            "confidence": round(confidence, 3),
            "thumb": f"data/thumbs/{thumb_name}" if jpeg else "",
            "clip_url": "",
            "detail": detail,
        }
        previous = next((item for item in self.events if item.get("id") == event_id), None)
        if previous:
            if previous.get("review"):
                event["review"] = previous["review"]
            if previous.get("clip_url"):
                event["clip_url"] = previous["clip_url"]
        self.events = [item for item in self.events if item["id"] != event_id]
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


def small_jpeg(jpeg: bytes, width: int = THUMB_WIDTH, quality: int = THUMB_QUALITY) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return jpeg
    height, frame_width = image.shape[:2]
    if frame_width > width:
        scale = width / float(frame_width)
        image = cv2.resize(image, (width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return jpeg
    return encoded.tobytes()
