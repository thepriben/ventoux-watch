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
        return [Detection(name, conf) for _x0, _y0, _x1, _y1, conf, name in _nms(boxes)]

    def locate(self, frame: np.ndarray) -> list[tuple[str, float, int, int, int, int]]:
        """Class boxes in the frame's own pixels. The box is the find, not a mask."""
        if self.session is None or frame.size == 0:
            return []
        blob, gain, (pad_x, pad_y) = _letterbox(frame, 640)
        raw = self.session.run(None, {self.input_name: blob})[0]
        height, width = frame.shape[:2]
        found = []
        for x0, y0, x1, y1, conf, name in _nms(_parse(raw)):
            left = int(round((x0 - pad_x) / gain))
            top = int(round((y0 - pad_y) / gain))
            right = int(round((x1 - pad_x) / gain))
            bottom = int(round((y1 - pad_y) / gain))
            left, top = max(0, left), max(0, top)
            right, bottom = min(width - 1, right), min(height - 1, bottom)
            if right - left < 2 or bottom - top < 2:
                continue
            found.append((name, conf, left, top, right - left, bottom - top))
        return found


def count_persons(detections: list[Detection], min_conf: float = 0.35) -> int:
    return sum(1 for item in detections if item.cls == "person" and item.conf >= min_conf)


def car_lights(frame: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> float:
    """How much of a blob is car lighting, between 0 and 1.

    After dark the model sees almost nothing, but a car carries its own marks:
    red tail lights, white headlights, and the pool of light they throw on the
    road. A walker carries none of that.
    """
    crop = frame if bbox is None else _crop(frame, bbox, margin=0.25)
    if crop is None or crop.size == 0:
        return 0.0
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(crop))
    tail = (red > 110) & (red - green > 45) & (red - blue > 35)
    head = (blue > 210) & (green > 210) & (red > 210)
    lit = float(np.count_nonzero(tail | head)) / float(crop.shape[0] * crop.shape[1])
    return min(1.0, lit)


def body_colour(frame: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> str:
    """The colour of a body, or nothing when it is not plain enough to say.

    Only the middle of the blob is read, because its edges are road and grass.
    Grey and black are held to a higher bar than the rest: tarmac and shadow
    are grey and black too, and a wrong colour is worse than no colour.
    """
    crop = frame if bbox is None else _crop(frame, bbox, margin=-0.22)
    if crop is None or crop.size < 24:
        return ""
    hue, saturation, value = (channel.astype(np.int16) for channel in cv2.split(cv2.cvtColor(_balanced(frame, crop), cv2.COLOR_BGR2HSV)))
    names = np.full(hue.shape, "", dtype=object)
    names[:] = "rouge"
    # A camera warms what it sees: the yellow of a post van reads near hue 17,
    # where a colour chart would call it amber. The boundary follows the camera,
    # not the chart.
    names[(hue >= 8) & (hue < 15)] = "orange"
    names[(hue >= 15) & (hue < 33)] = "jaune"
    names[(hue >= 33) & (hue < 85)] = "vert"
    names[(hue >= 85) & (hue < 130)] = "bleu"
    names[(hue >= 8) & (hue < 20) & (value < 130)] = "marron"
    names[saturation < 58] = "gris"
    names[(saturation < 58) & (value > 160)] = "blanc"
    names[value < 55] = "noir"
    counts: dict[str, int] = {}
    for name in names.ravel():
        counts[name] = counts.get(name, 0) + 1
    winner = max(counts, key=lambda key: counts[key])
    share = counts[winner] / float(names.size)
    floor = 0.60 if winner in {"gris", "noir"} else 0.40
    return winner if share >= floor else ""


def _balanced(frame: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Undo the colour of the light before naming the colour of the paint.

    At dusk the whole scene turns blue and a white van reads as a blue one.
    Taking the frame as a whole to be grey on average, and scaling the channels
    until it is, leaves the paint and drops the hour of the day.
    """
    means = frame.reshape(-1, 3).mean(axis=0)
    if float(means.min()) < 1:
        return crop
    gain = means.mean() / means
    return np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)


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


def _nms(boxes: list[tuple[float, float, float, float, float, str]], iou_limit: float = 0.5):
    boxes = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    used: list[tuple[float, float, float, float]] = []
    for item in boxes:
        x0, y0, x1, y1, _conf, _name = item
        if any(_iou((x0, y0, x1, y1), previous) > iou_limit for previous in used):
            continue
        used.append((x0, y0, x1, y1))
        kept.append(item)
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
