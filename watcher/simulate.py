"""A fire that never happened, painted on the real view.

The watcher can only be trusted on a fire if a fire has been put in front of
it. Mont Serein does not burn to order, so the plume is drawn instead: pale,
turbulent, climbing and widening with every second, the way a fire that has
just caught looks from a kilometre away. Everything downstream — background
subtraction, tracking, the smoke and flame measurements, the naming — is the
real code, untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

# How fast the plume climbs and spreads, as a share of the frame height per
# second. A fire in scrub throws a visible column within seconds; these numbers
# are deliberately modest so the test is not made easy.
CLIMB_PER_S = 0.020
SPREAD_PER_S = 0.004
DRIFT_PER_S = 0.004
SMOKE_BGR = (196, 196, 196)
FLAME_BGR = (30, 95, 235)


def plume(
    frame: np.ndarray,
    spot: tuple[float, float],
    age_s: float,
    flame: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """The view as it would look age_s seconds after the fire caught at spot.

    spot is given in the normalised frame, the base of the fire. The drawing
    is turbulent and seeded on the second, so two frames never match and the
    background subtractor sees the plume move.
    """
    if age_s <= 0:
        return frame.copy()
    height, width = frame.shape[:2]
    base_x, base_y = spot[0] * width, spot[1] * height
    column = CLIMB_PER_S * age_s * height
    smoke = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(seed + int(age_s * 4))
    puffs = max(8, int(column / 4))
    for index in range(puffs):
        along = (index + 1) / puffs
        radius = (0.010 + SPREAD_PER_S * age_s * along) * height
        x = base_x + DRIFT_PER_S * age_s * along * width + rng.normal(0, radius * 0.5)
        y = base_y - column * along + rng.normal(0, radius * 0.3)
        weight = (1.0 - 0.55 * along) * rng.uniform(0.7, 1.0)
        cv2.circle(smoke, (int(x), int(y)), max(2, int(radius)), float(weight), -1)
    blur = max(3, int(column / 6) | 1)
    smoke = cv2.GaussianBlur(smoke, (blur, blur), 0)
    smoke = np.clip(smoke, 0, 1)[:, :, None]
    paint = np.full(frame.shape, SMOKE_BGR, dtype=np.float32)
    out = frame.astype(np.float32) * (1 - smoke) + paint * smoke
    if flame:
        _flame(out, base_x, base_y, column, rng)
    return np.clip(out, 0, 255).astype(np.uint8)


def _flame(out: np.ndarray, base_x: float, base_y: float, column: float, rng) -> None:
    """The hot core at the foot of the plume.

    At night this is all a camera sees of a fire: the smoke disappears into the
    dark and only the flame carries any colour.
    """
    core = np.zeros(out.shape[:2], dtype=np.float32)
    reach = max(4.0, column * 0.22)
    for _ in range(9):
        x = base_x + rng.normal(0, reach * 0.4)
        y = base_y - abs(rng.normal(0, reach * 0.5))
        cv2.circle(core, (int(x), int(y)), max(2, int(reach * rng.uniform(0.3, 0.7))), float(rng.uniform(0.6, 1.0)), -1)
    core = cv2.GaussianBlur(core, (5, 5), 0)
    core = np.clip(core, 0, 1)[:, :, None]
    paint = np.full(out.shape, FLAME_BGR, dtype=np.float32)
    out[:] = out * (1 - core) + paint * core


def sensor_noise(frame: np.ndarray, seed: int = 0) -> np.ndarray:
    """The grain a real webcam has, so the still frames are not identical.

    Without it the background subtractor learns a perfect background and the
    test is kinder than reality.
    """
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1.6, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)
