"""Apply the corrections given by hand on the history of 25 September.

Each entry says what the frame really shows. When the red rectangle was drawn
on the wrong thing, the old one is erased from the thumbnail and a new one is
drawn around the right one. Photos of dropped events are kept on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Rectangles are given as they should appear on the 480x270 thumbnail. The
# last field is the box the photo was first drawn with, so the old stroke can
# be found even after the entry has been corrected once.
RENAME = {
    "2026-09-25T18:28:24Z": ("vehicle", "Voiture", (166, 214, 226, 244), [0.5859, 0.7667, 0.0391, 0.0583]),
    "2026-09-25T17:54:36Z": ("vehicle", "Camionnette blanche", (173, 213, 236, 248), [0.2547, 0.7389, 0.2516, 0.2222]),
    "2026-09-25T17:53:09Z": ("vehicle", "Voiture avec carriole", (178, 213, 259, 249), [0.2547, 0.7472, 0.3312, 0.2306]),
    "2026-09-25T17:50:59Z": ("vehicle", "Voiture", None, None),
    "2026-09-25T15:14:37Z": ("car", "Voiture blanche", (5, 188, 42, 211), None),
    "2026-09-25T12:33:22Z": ("vehicle", "Voiture et piéton", (32, 186, 53, 207), None),
}
# Strokes drawn on the wrong thing, given straight in thumbnail pixels because
# they were never stored as a box.
REFRAME = {
    "2026-09-25T18:58:22Z": {
        "type": "vehicle",
        "label": "Voiture",
        "erase": [(0, 169, 176, 270)],
        "draw": (298, 250, 378, 269),
    },
    "2026-09-25T15:56:28Z": {"erase": [(345, 205, 390, 270)]},
    "2026-09-25T12:42:12Z": {
        "type": "car",
        "label": "Voiture blanche",
        "erase": [(115, 147, 179, 192)],  # la cabane, building=kiosk
        "keep": (25, 187, 103, 231),
    },
    "2026-09-25T10:59:48Z": {
        "type": "car",
        "label": "Camionnette jaune",
        "erase": [(0, 203, 137, 270)],
        "draw": (28, 221, 103, 262),
    },
    "2026-09-25T12:19:07Z": {
        "type": "car",
        "label": "Voiture",
        "erase": [(33, 160, 69, 191)],  # une voiture garée dans la découpe
        "keep": (32, 197, 121, 270),
    },
    "2026-09-25T22:51:50Z": {
        "type": "car",
        "label": "Voiture blanche",
        "erase": [(0, 145, 182, 269)],  # le halo des phares sur le rond-point
        "draw": (283, 245, 369, 269),
    },
    "2026-09-25T08:23:34Z": {
        "type": "bus",
        "label": "Bus",
        "draw": (53, 169, 121, 209),
    },
}
DROP = {
    "2026-09-25T22:11:35Z": "l'îlot central du rond-point, ses pierres et ses figures",
    "2026-09-25T21:39:29Z": "le revêtement de la chaussée qui prend la lumière",
    "2026-09-25T19:31:07Z": "le halo des phares sur la chaussée, rien dedans",
    "2026-09-25T17:08:06Z": "ombre et soleil à la lisière",
    "2026-09-25T17:02:48Z": "la statue en bois",
    "2026-09-25T17:01:45Z": "la statue en bois",
}


def _raw_box(rect: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    """Undo the margin that the drawing adds, so a redraw lands on this rect."""
    box = []
    for low, high, span in ((rect[0], rect[2], width), (rect[1], rect[3], height)):
        room = high - low
        side = room - 20 if room <= 42 else room / 1.9
        pad = max(10.0, side * 0.45)
        box.append(((low + pad), max(side, 1.0)))
    (x, w), (y, h) = box
    return [round(x / width, 4), round(y / height, 4), round(w / width, 4), round(h / height, 4)]


def _drawn_rect(box, width: int, height: int) -> tuple[int, int, int, int]:
    """Where the stored box ends up once the margin is added, as in the store."""
    x, y, w, h = (float(box[0]) * width, float(box[1]) * height, float(box[2]) * width, float(box[3]) * height)
    pad_x, pad_y = max(10.0, w * 0.45), max(10.0, h * 0.45)
    return (
        int(max(0, x - pad_x)),
        int(max(0, y - pad_y)),
        int(min(width - 1, x + max(w, 1) + pad_x)),
        int(min(height - 1, y + max(h, 1) + pad_y)),
    )


def _red_ink(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image.astype(np.int16))
    return ((red - green > 40) & (red - blue > 40) & (red > 90)).astype(np.uint8) * 255


def _erase(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Take the old stroke out.

    Around the place where it was drawn, a line of the frame is red almost all
    the way across. That is the stroke. A red roof or a tail light is red only
    here and there, so it stays. The stroke is pure red, with green and blue
    both far below: a yellow van is red and green together, and survives.
    """
    height, width = image.shape[:2]
    x0, y0 = max(0, rect[0] - 12), max(0, rect[1] - 12)
    x1, y1 = min(width, rect[2] + 13), min(height, rect[3] + 13)
    blue, green, red = cv2.split(image[y0:y1, x0:x1].astype(np.int16))
    ink = ((red - green > 30) & (red - blue > 30) & (red > 70)).astype(np.uint8)
    line = np.zeros(ink.shape, dtype=np.uint8)
    line[ink.sum(axis=1) >= ink.shape[1] * 0.5, :] = 1
    line[:, ink.sum(axis=0) >= ink.shape[0] * 0.5] = 1
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.dilate(line * ink, np.ones((3, 3), np.uint8)) * 255
    return cv2.inpaint(image, mask, 4, cv2.INPAINT_TELEA)


def _find_rect(image: np.ndarray, near: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """The red rectangle already burnt into the photo, if there is one."""
    found = cv2.findContours(_red_ink(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    best, score = None, 60.0
    for shape in found:
        x, y, w, h = cv2.boundingRect(shape)
        if w < 12 or h < 8:
            continue
        gap = abs(x - near[0]) + abs(y - near[1])
        if gap < score:
            best, score = (x, y, x + w, y + h), gap
    return best


def _reframe(event: dict, order: dict) -> None:
    thumb = ROOT / event["thumb"]
    image = cv2.imread(str(thumb))
    if image is None:
        raise SystemExit(f"photo manquante {thumb}")
    detail = dict(event.get("detail") or {})
    rect = order.get("draw")
    done = detail.get("drawn")
    if done == list(rect or []):
        return
    for stale in (order.get("erase") or []) + ([done] if done else []):
        # A stroke this script drew on an earlier run is erased like any other:
        # without it, correcting a correction leaves two rectangles.
        image = _erase(image, tuple(stale))
    if rect is not None:
        cv2.rectangle(image, rect[:2], rect[2:], (0, 0, 210), 1)
        detail["drawn"] = list(rect)
    kept = rect or order.get("keep")
    if kept is not None:
        detail["box"] = _raw_box(kept, image.shape[1], image.shape[0])
    else:
        detail.pop("box", None)
    cv2.imwrite(str(thumb), image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    label = order.get("label")
    if label:
        event["type"] = order.get("type") or event["type"]
        event["label"] = label
        detail["correction"] = label
        detail["reading"] = f"{label}, indiqué à la main."
    event["detail"] = detail
    event["review"] = "accepted"
    print(f"recadré {event['t']}  {event['label']}")


def main() -> int:
    path = ROOT / "data" / "events.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    kept, touched = [], 0
    for event in payload.get("events") or []:
        when = event.get("t")
        if (event.get("detail") or {}).get("simulation"):
            # A simulation can land on the same second as a real event. It is
            # never a thing seen, so no hand correction applies to it.
            kept.append(event)
            continue
        if when in DROP:
            print(f"retiré  {when}  {DROP[when]}")
            touched += 1
            continue
        kept.append(event)
        if when in REFRAME:
            _reframe(event, REFRAME[when])
            touched += 1
            continue
        if when not in RENAME:
            continue
        kind, label, rect, was = RENAME[when]
        detail = dict(event.get("detail") or {})
        event["type"] = kind
        event["label"] = label
        detail["correction"] = label
        detail["reading"] = f"{label}, indiqué à la main."
        if rect is not None:
            thumb = ROOT / event["thumb"]
            image = cv2.imread(str(thumb))
            if image is None:
                print(f"photo manquante {thumb}")
                return 1
            height, width = image.shape[:2]
            stale = _drawn_rect(was, width, height) if was else _find_rect(image, rect)
            if stale is not None:
                image = _erase(image, stale)
            cv2.rectangle(image, rect[:2], rect[2:], (0, 0, 210), 1)
            cv2.imwrite(str(thumb), image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            detail["box"] = _raw_box(rect, width, height)
        event["detail"] = detail
        event["review"] = "accepted"
        touched += 1
        print(f"corrigé {when}  {label}")
    payload["events"] = kept
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{touched} entrées revues, {len(kept)} gardées")
    return 0


if __name__ == "__main__":
    sys.exit(main())
