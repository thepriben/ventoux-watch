"""Background subtraction, then one track per compact moving blob."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from watcher.geometry import assign_zone


@dataclass
class Track:
    id: int
    zone: str
    frames: int = 0
    misses: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    first_centroid: tuple[float, float] = (0.0, 0.0)
    centroid: tuple[float, float] = (0.0, 0.0)
    area_ratio: float = 0.0
    first_area: float = 0.0
    best_area: float = 0.0
    best_jpeg: bytes = b""
    started: float = 0.0
    updated: float = 0.0

    @property
    def travel(self) -> float:
        dx = self.centroid[0] - self.first_centroid[0]
        dy = self.centroid[1] - self.first_centroid[1]
        return (dx * dx + dy * dy) ** 0.5

    @property
    def area_grow(self) -> float:
        if self.first_area <= 0:
            return 1.0
        return self.area_ratio / self.first_area


@dataclass
class MotionStep:
    ended: list[Track] = field(default_factory=list)
    roundabout_motion: bool = False
    global_change: bool = False


class MotionDetector:
    def __init__(
        self,
        zones: dict,
        motion_width: int = 640,
        min_track_frames: int = 3,
        max_foreground_ratio: float = 0.35,
        warmup_frames: int = 8,
    ):
        self.zones = zones
        self.motion_width = motion_width
        self.min_track_frames = min_track_frames
        self.max_foreground_ratio = max_foreground_ratio
        self.warmup_frames = warmup_frames
        self.bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=False)
        self.tracks: list[Track] = []
        self._next_id = 1
        self._seen = 0

    def step(self, frame: np.ndarray, now: float) -> MotionStep:
        small, scale = _resize_width(frame, self.motion_width)
        self._seen += 1
        if self._seen <= self.warmup_frames:
            self.bg.apply(small, learningRate=-1)
            return MotionStep()
        mask = self.bg.apply(small, learningRate=0)
        mask = _prepare_mask(mask, self.zones, small.shape[1], small.shape[0])
        ratio = float(cv2.countNonZero(mask)) / float(mask.size)
        if ratio > self.max_foreground_ratio:
            return MotionStep(global_change=True)
        self.bg.apply(small, learningRate=-1)
        blobs = _blobs(mask, scale)
        return self._update_tracks(blobs, frame, now)

    def _update_tracks(self, blobs: list[dict], frame: np.ndarray, now: float) -> MotionStep:
        unused = set(range(len(self.tracks)))
        roundabout = False
        for blob in blobs:
            zone = assign_zone(blob["cx"], blob["cy"], self.zones)
            if zone == "roundabout":
                roundabout = True
            if zone == "sky" and blob["area_ratio"] < 0.00005:
                continue
            if zone != "sky" and blob["area_ratio"] < 0.0004:
                continue
            match = self._match(blob["cx"], blob["cy"], zone, unused)
            if match is None:
                track = Track(
                    id=self._next_id,
                    zone=zone,
                    bbox=blob["bbox"],
                    first_centroid=(blob["cx"], blob["cy"]),
                    centroid=(blob["cx"], blob["cy"]),
                    area_ratio=blob["area_ratio"],
                    first_area=blob["area_ratio"],
                    best_area=blob["area_ratio"],
                    started=now,
                    updated=now,
                    frames=1,
                )
                self._next_id += 1
                track.best_jpeg = _jpeg(frame)
                self.tracks.append(track)
                continue
            unused.discard(match)
            track = self.tracks[match]
            track.frames += 1
            track.misses = 0
            track.centroid = (blob["cx"], blob["cy"])
            track.bbox = blob["bbox"]
            track.area_ratio = blob["area_ratio"]
            track.updated = now
            track.zone = zone
            if blob["area_ratio"] >= track.best_area:
                track.best_area = blob["area_ratio"]
                track.best_jpeg = _jpeg(frame)

        ended: list[Track] = []
        kept: list[Track] = []
        for index, track in enumerate(self.tracks):
            if index in unused and track.frames > 0 and track.updated != now:
                track.misses += 1
            if track.misses >= 2:
                needed = 3 if track.zone == "sky" else self.min_track_frames
                if track.frames >= needed:
                    ended.append(track)
                continue
            kept.append(track)
        self.tracks = kept
        return MotionStep(ended=ended, roundabout_motion=roundabout)

    def _match(self, cx: float, cy: float, zone: str, unused: set[int]) -> int | None:
        limit = 0.28 if zone == "sky" else 0.18
        best_index = None
        best_distance = limit
        for index in unused:
            track = self.tracks[index]
            if track.zone != zone:
                continue
            distance = ((track.centroid[0] - cx) ** 2 + (track.centroid[1] - cy) ** 2) ** 0.5
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index


def _resize_width(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    height, frame_width = frame.shape[:2]
    if frame_width <= width:
        return frame, 1.0
    scale = width / float(frame_width)
    resized = cv2.resize(frame, (width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _prepare_mask(mask: np.ndarray, zones: dict, width: int, height: int) -> np.ndarray:
    binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    for circle in zones.get("exclude", []):
        center = (int(circle["cx"] * width), int(circle["cy"] * height))
        radius = int(circle["r"] * width)
        cv2.circle(binary, center, radius, 0, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _blobs(mask: np.ndarray, scale: float) -> list[dict]:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    height, width = mask.shape[:2]
    blobs = []
    for contour in contours:
        if cv2.contourArea(contour) < 8:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blobs.append(
            {
                "cx": (x + w / 2) / width,
                "cy": (y + h / 2) / height,
                "bbox": (int(x / scale), int(y / scale), max(int(w / scale), 1), max(int(h / scale), 1)),
                "area_ratio": (w * h) / float(width * height),
            }
        )
    return blobs


def _jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return encoded.tobytes()


def warm_ratio(jpeg: bytes) -> float:
    if not jpeg:
        return 0.0
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return 0.0
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    warm = (red > 140) & (red > green + 25) & (red > blue + 25)
    return float(np.count_nonzero(warm)) / float(warm.size)
