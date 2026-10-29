"""A 40' high-cube container flattens the person; a crane lifts it and they spring back."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .squash_effect import blend, draw_person, person_top, shake
except ImportError:
    from squash_effect import blend, draw_person, person_top, shake


ASSETS = Path(__file__).resolve().parent / "assets"
CONTAINER_PATH = ASSETS / "container_40hc.png"
DROP_SOUND_PATH = ASSETS / "container_drop.wav"
CRANE_SOUND_PATH = ASSETS / "container_crane.wav"

CONTAINER_ASPECT = 12.192 / 2.896  # Length over height of a 40' high cube.
LENGTH_RATIO = 0.94  # Container length relative to the frame width.

FALL_STARTS = 0.25  # The scene darkens a moment before the box shows up.
DROP_LANDS = 0.90
DROP_SECONDS = 1.70
LIFT_LOCKS = 0.70  # The spreader sits on the box.
LIFT_STARTS = 0.90
LIFT_RELEASES = 1.50  # The person comes unstuck and springs back.
LIFT_FLIGHT = 0.55
LIFT_SECONDS = 2.20
STRETCH = 1.15  # How tall the person gets while stuck under the rising box.
BULGE = 0.35  # Squashed people get wider, stretched ones thinner.

DUST_COLOR = np.array((138, 152, 164), dtype=np.float32)  # BGR, grey-brown
YELLOW = (18, 180, 240)
CABLE = (52, 52, 56)


def pointing_up_score(points: np.ndarray) -> float:
    """Index straight and pointing up in image pixels (y down), the other fingers folded."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0
    index = points[5:9]
    length = float(np.linalg.norm(np.diff(index, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0
    direction = index[3] - index[0]
    straightness = float(np.linalg.norm(direction)) / length
    # Straight and within about 37 degrees of vertical.
    if straightness < 0.85 or -direction[1] < float(np.linalg.norm(direction[:2])) * 0.8:
        return 0.0
    if np.linalg.norm(index[3] - points[0]) <= np.linalg.norm(index[1] - points[0]) * 1.1:
        return 0.0
    # Folded fingers stay short, even when seen from the side.
    for base in (9, 13, 17):
        if np.linalg.norm(points[base + 3] - points[base]) > length * 0.6:
            return 0.0
    return float(np.clip(straightness, 0.0, 1.0))


def load_container() -> np.ndarray:
    sprite = cv2.imread(str(CONTAINER_PATH), cv2.IMREAD_COLOR)
    if sprite is None:
        raise RuntimeError(f"Sprite del container non disponibile: {CONTAINER_PATH}")
    return sprite


def container_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Centre column over the person, floor row and container height in pixels."""
    height, width = mask.shape
    size = max(4, round(width * LENGTH_RATIO / CONTAINER_ASPECT))
    columns = mask.sum(axis=0)
    total = float(columns.sum())
    center = width / 2 if total < 1.0 else float((columns * np.arange(width)).sum() / total)
    return round(min(max(center, width * 0.3), width * 0.7)), height, size


def _sized(sprite: np.ndarray, size: int, cache: dict[int, np.ndarray] | None) -> np.ndarray:
    sized = cache.get(size) if cache is not None else None
    if sized is None:
        length = max(1, round(size * CONTAINER_ASPECT))
        sized = cv2.resize(sprite, (length, size), interpolation=cv2.INTER_AREA)
        if cache is not None:
            cache[size] = sized
    return sized


def _paste(frame: np.ndarray, sized: np.ndarray, x1: int, bottom: float) -> None:
    """The container is opaque: copy it with its lower edge on row `bottom`."""
    y1 = round(bottom) - sized.shape[0]
    left, top = max(0, x1), max(0, y1)
    right = min(frame.shape[1], x1 + sized.shape[1])
    lower = min(frame.shape[0], y1 + sized.shape[0])
    if left < right and top < lower:
        frame[top:lower, left:right] = sized[top - y1:lower - y1, left - x1:right - x1]


def place_container(frame: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int, int],
                    bottom: float, cache: dict[int, np.ndarray] | None = None) -> None:
    sized = _sized(sprite, anchor[2], cache)
    _paste(frame, sized, anchor[0] - sized.shape[1] // 2, bottom)


def render_rest(background: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int, int],
                cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Only the container is left, resting on the floor of the frame."""
    output = background.copy()
    place_container(output, sprite, anchor, anchor[1], cache)
    return output


def _shadow(output: np.ndarray, x1: int, x2: int, bottom: float, size: int) -> None:
    """Soft shadow cast right below the container while it hangs in the frame."""
    strength = 0.45 * min(1.0, max(0.0, bottom / (size * 0.5)))
    height, width = output.shape[:2]
    top, end = max(0, math.ceil(bottom)), min(height, math.ceil(bottom + size * 0.9))
    left, right = max(0, x1), min(width, x2)
    if strength <= 0.0 or top >= end or left >= right:
        return
    rows = np.exp(-(np.arange(top, end) - bottom) / (size * 0.3))
    xs = np.arange(left, right)
    columns = np.clip(np.minimum(xs - x1, x2 - xs) / (size * 0.4), 0.0, 1.0)
    gain = (1.0 - strength * rows[:, None] * columns[None, :]).astype(np.float32)
    area = output[top:end, left:right]
    area[:] = (area.astype(np.float32) * gain[:, :, None]).astype(np.uint8)


def _dust(output: np.ndarray, x1: int, x2: int, size: int, age: float) -> None:
    """Grey dust squirts out from under the container after it lands."""
    life = DROP_SECONDS - DROP_LANDS
    if not 0.0 < age < life:
        return
    height, width = output.shape[:2]
    rng = np.random.default_rng(3307)
    count = 64
    side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    at_ends = rng.random(count) < 0.45
    start = np.where(at_ends, np.where(side < 0, x1, x2) + side * rng.uniform(-0.4, 0.1, count) * size,
                     rng.uniform(x1, x2, count))
    # Air squeezed out from under the box: fast sideways, low over the floor.
    speed_x = np.where(at_ends, side * rng.uniform(1.0, 3.0, count), side * rng.uniform(0.0, 1.2, count))
    speed_y = rng.uniform(0.1, 0.8, count)
    radius = rng.uniform(0.07, 0.16, count) * size
    level = rng.uniform(0.5, 0.9, count)
    drift = 0.3 * (1.0 - math.exp(-age / 0.3)) * size  # Air drag slows the puffs down.
    grow = 1.0 + 2.2 * (1.0 - math.exp(-age / 0.35))
    fade = (1.0 - age / life) ** 1.6 * min(1.0, age / 0.04)
    top = max(0, int(height - size * 1.6))
    layer = np.zeros((height - top, width), np.float32)
    for x, vx, vy, r, a in zip(start, speed_x, speed_y, radius, level):
        center = (round(x + vx * drift), round(height - r * 0.6 - vy * drift) - top)
        cv2.circle(layer, center, max(1, round(r * grow)), float(a * fade), -1, cv2.LINE_AA)
    alpha = np.clip(cv2.GaussianBlur(layer, (0, 0), max(1.0, size * 0.05)), 0.0, 1.0)[:, :, None]
    # Darker where the dust is thick near the floor, lighter on the rising tops.
    shade = np.linspace(1.12, 0.82, height - top, dtype=np.float32)[:, None, None]
    area = output[top:]
    area[:] = np.clip(area.astype(np.float32) * (1.0 - alpha) + DUST_COLOR * shade * alpha,
                      0, 255).astype(np.uint8)


def render_drop(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                sprite: np.ndarray, anchor: tuple[int, int, int], age: float,
                cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """The container falls on the person, squashes them flat and stays."""
    if age >= DROP_SECONDS:
        return render_rest(background, sprite, anchor, cache)
    center_x, floor, size = anchor
    if age < DROP_LANDS:
        bottom = floor * (max(0.0, age - FALL_STARTS) / (DROP_LANDS - FALL_STARTS)) ** 2
        dim = 0.16 * min(1.0, age / DROP_LANDS) ** 2
    else:
        after = age - DROP_LANDS
        bottom = floor - size * 0.08 * math.exp(-8.0 * after) * abs(math.sin(14.0 * after))
        dim = 0.16 * max(0.0, 1.0 - after / 0.12)

    top = person_top(mask)
    scale = 0.0 if top is None or age >= DROP_LANDS else (floor - bottom) / max(1, floor - top)
    if scale >= 1.0:
        output = blend(frame, background, mask)
    else:
        output = background.copy()
        if scale > 0.0:
            draw_person(output, frame, mask, top, 1.0 + BULGE * (1.0 - scale), scale)
    if dim > 0.0:
        # Something huge is about to block the light from above.
        height = output.shape[0]
        rows = (1.0 - dim * (1.0 - 0.5 * np.arange(height) / height)).astype(np.float32)
        output[:] = (output.astype(np.float32) * rows[:, None, None]).astype(np.uint8)

    sized = _sized(sprite, size, cache)
    x1 = center_x - sized.shape[1] // 2
    x2 = x1 + sized.shape[1]
    _shadow(output, x1, x2, bottom, size)
    _paste(output, sized, x1, bottom)
    if age >= DROP_LANDS:
        _dust(output, x1, x2, size, age - DROP_LANDS)
        output = shake(output, age - DROP_LANDS, size * 0.06)
    return output


def _tint(color, factor):
    return tuple(min(255, round(channel * factor)) for channel in color)


def _box(image: np.ndarray, x1: float, y1: float, x2: float, y2: float, color) -> None:
    """Painted steel box: lit top edge, darker underside, thin outline."""
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    if x2 <= x1 or y2 <= y1:
        return
    lip = max(1, (y2 - y1) // 5)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, -1)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y1 + lip - 1), _tint(color, 1.25), -1)
    cv2.rectangle(image, (x1, y2 - lip), (x2 - 1, y2 - 1), _tint(color, 0.7), -1)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 34), 1)


def _hazard(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> None:
    """Black diagonal stripes over a yellow box."""
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(image.shape[1], x2), min(image.shape[0], y2)
    if left >= right or top >= bottom:
        return
    roi = image[top:bottom, left:right]
    rise = y2 - y1
    step = max(2, rise // 2)
    for start in range(x1 - rise, x2, 2 * step):
        polygon = np.array(((start, y2), (start + step, y2), (start + step + rise, y1),
                            (start + rise, y1))) - (left, top)
        cv2.fillConvexPoly(roi, polygon.astype(np.int32), (24, 24, 26), cv2.LINE_AA)


def _draw_spreader(image: np.ndarray, x1: int, length: int, base: float, size: int,
                   locked: bool) -> None:
    """Quay-crane spreader sitting on (or above) the container top at row `base`."""
    center = x1 + length / 2
    arm, frame, head = size * 0.08, size * 0.14, size * 0.24
    sheave = max(1, round(size * 0.045))
    for side in (-1, 1):
        # Steel ropes run up to the trolley, outside the top of the frame.
        for offset in (-0.006, 0.006):
            bottom_x = center + (side * 0.045 + offset) * length
            top_x = center + (side * 0.03 + offset) * length
            cv2.line(image, (round(bottom_x), round(base - head)), (round(top_x), round(-size)),
                     CABLE, max(1, round(size * 0.012)), cv2.LINE_AA)
    for end, side in ((x1, -1), (x1 + length, 1)):
        # Flippers guide the spreader onto the corner castings.
        flipper = np.array((
            (end, base - arm), (end + side * 0.007 * length, base - arm),
            (end + side * 0.012 * length, base + size * 0.13), (end, base + size * 0.08),
        ))
        cv2.fillConvexPoly(image, np.round(flipper).astype(np.int32), _tint(YELLOW, 0.75), cv2.LINE_AA)
    _box(image, x1 + 0.004 * length, base - arm, x1 + 0.996 * length, base, YELLOW)
    _box(image, center - 0.17 * length, base - frame, center + 0.17 * length, base, YELLOW)
    _hazard(image, center - 0.05 * length, base - frame * 0.8, center + 0.05 * length, base - frame * 0.3)
    _box(image, center - 0.07 * length, base - head, center + 0.07 * length, base - frame,
         _tint(YELLOW, 0.85))
    light = (70, 230, 90) if locked else (0, 170, 255)
    for side in (-1, 1):
        cv2.circle(image, (round(center + side * 0.045 * length), round(base - head)), sheave,
                   (70, 72, 78), -1, cv2.LINE_AA)
        cv2.circle(image, (round(center + side * 0.045 * length), round(base - head)), sheave,
                   (150, 152, 158), max(1, sheave // 3), cv2.LINE_AA)
        cv2.circle(image, (round(center + side * 0.12 * length), round(base - frame * 0.55)),
                   max(1, round(size * 0.02)), light, -1, cv2.LINE_AA)


def render_lift(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                sprite: np.ndarray, anchor: tuple[int, int, int], age: float,
                cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """A crane spreader grabs the container and lifts it; the person stretches back up."""
    if age >= LIFT_SECONDS:
        return frame.copy() if mask is None else blend(frame, background, mask)
    center_x, floor, size = anchor
    top = person_top(mask)
    natural = floor - top if top is not None else floor * 0.75
    if age < LIFT_STARTS:
        scale = 0.0
    elif age < LIFT_RELEASES:
        progress = (age - LIFT_STARTS) / (LIFT_RELEASES - LIFT_STARTS)
        scale = STRETCH * progress * progress * (3.0 - 2.0 * progress)
    else:
        after = age - LIFT_RELEASES
        scale = 1.0 + (STRETCH - 1.0) * math.exp(-after / 0.13) * math.cos(2 * math.pi * 4.2 * after)
    if age < LIFT_RELEASES:
        bottom = floor - scale * natural  # The head sticks to the underside of the box.
    else:
        released = floor - STRETCH * natural
        flight = min(1.0, (age - LIFT_RELEASES) / LIFT_FLIGHT)
        bottom = released - (released + size * 1.4) * flight * flight

    output = background.copy()
    if scale > 0.0 and top is not None:
        draw_person(output, frame, mask, top, 1.0 + BULGE * (1.0 - scale), scale)
    sized = _sized(sprite, size, cache)
    x1 = center_x - sized.shape[1] // 2
    _shadow(output, x1, x1 + sized.shape[1], bottom, size)
    _paste(output, sized, x1, bottom)
    if age < LIFT_LOCKS:
        progress = age / LIFT_LOCKS
        start, landed = -size * 0.25 - 2, floor - size
        base = start + (landed - start) * (1.0 - (1.0 - progress) ** 3)
    else:
        base = bottom - size
    _draw_spreader(output, x1, sized.shape[1], base, size, locked=age >= LIFT_LOCKS)
    return output
