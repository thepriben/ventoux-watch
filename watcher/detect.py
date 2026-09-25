"""YOLO nano via ONNX Runtime. Missing model means no class labels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from watcher.naming import Detection

COCO = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck",
    4: "airplane",
}
KEEP = set(COCO)


class YoloDetector:
    def __init__(self, model_path: str):
        self.session = None
        self.input_name = ""
        path = Path(model_path)
        if not path.is_file():
            return
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @property
    def ready(self) -> bool:
        return self.session is not None

    def detect(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> list[Detection]:
        if self.session is None:
            return []
        crop = frame if bbox is None else _crop(frame, bbox, margin=0.35)
        if crop.size == 0:
            return []
        blob, _gain, _pad = _letterbox(crop, 640)
        raw = self.session.run(None, {self.input_name: blob})[0]
        boxes = _parse(raw)
        return _nms(boxes)


def count_persons(detections: list[Detection], min_conf: float = 0.35) -> int:
    return sum(1 for item in detections if item.cls == "person" and item.conf >= min_conf)


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int], margin: float) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    mx = int(w * margin)
    my = int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(width, x + w + mx)
    y1 = min(height, y + h + my)
    return frame[y0:y1, x0:x1]


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_w, new_h = int(round(width * gain)), int(round(height * gain))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    canvas[int(pad_y) : int(pad_y) + new_h, int(pad_x) : int(pad_x) + new_w] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, gain, (pad_x, pad_y)


def _parse(raw: np.ndarray) -> list[tuple[float, float, float, float, float, str]]:
    output = np.squeeze(raw)
    if output.ndim != 2:
        return []
    if output.shape[0] < output.shape[1]:
        output = output.T
    boxes = []
    for row in output:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        if class_id not in KEEP:
            continue
        conf = float(scores[class_id])
        if conf < 0.25:
            continue
        cx, cy, w, h = (float(v) for v in row[:4])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, COCO[class_id]))
    return boxes


def _nms(boxes: list[tuple[float, float, float, float, float, str]], iou_limit: float = 0.5) -> list[Detection]:
    boxes = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept: list[Detection] = []
    used: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1, conf, name in boxes:
        if any(_iou((x0, y0, x1, y1), previous) > iou_limit for previous in used):
            continue
        used.append((x0, y0, x1, y1))
        kept.append(Detection(name, conf))
    return kept


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)
