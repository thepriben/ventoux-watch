"""Draw the zone polygons on the reference frame."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "sky": (255, 180, 40),
    "road": (40, 140, 220),
    "roundabout": (40, 180, 90),
    "slope": (180, 90, 40),
}


def main() -> None:
    zones = json.loads((ROOT / "config" / "zones.json").read_text())
    image = cv2.imread(str(ROOT / "data" / "reference.jpg"))
    if image is None:
        raise SystemExit("data/reference.jpg manquant")
    height, width = image.shape[:2]
    overlay = image.copy()
    for name, polygon in zones["polygons"].items():
        points = np.array([[int(x * width), int(y * height)] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [points], COLORS[name])
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
    for circle in zones.get("exclude", []):
        center = (int(circle["cx"] * width), int(circle["cy"] * height))
        cv2.circle(image, center, int(circle["r"] * width), (0, 0, 255), 2)
    target = ROOT / "data" / "zones-preview.jpg"
    cv2.imwrite(str(target), image)
    print(target)

if __name__ == "__main__":
    sys.exit(main())
