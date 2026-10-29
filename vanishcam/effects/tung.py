"""Tung Tung Tung Sahur bats the person smaller; Bombardiro Crocodilo winches them back up."""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from ..paths import IMAGES, SOUNDS, VOICES_DIR
from .squash import blend, draw_person, person_top, shake


BODY_PATH = IMAGES / "tung_body.png"
BAT_PATH = IMAGES / "tung_bat.png"
BURST_PATH = IMAGES / "tung_burst.png"
PLANE_PATH = IMAGES / "bombardiro.png"
BEATING_SOUND_PATH = SOUNDS / "tung_beating.wav"
RESCUE_SOUND_PATH = SOUNDS / "bombardiro_rescue.wav"
TUNG_VOICE_PATH = SOUNDS / "tung_voice.wav"
BOMBARDIRO_VOICE_PATH = SOUNDS / "bombardiro_voice.wav"
# Clips in assets/voci (tung.mp3, bombardiro.wav, ...) replace the synthesized voices.
VOICE_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")

# Sprite geometry, in pixels of the PNG files.
BODY_FLOOR = (210, 975)  # Between the feet.
BODY_SHOULDER = (332, 450)  # Where the bat arm is attached.
BAT_PIVOT = (30, 60)  # Shoulder end of the arm-and-bat sprite, which points right.
BAT_SWEET = 650  # Middle of the barrel.
PLANE_HOOK = (590, 303)  # Ring under the bomb bay.

CHARACTER_HEIGHT = 0.9  # Of the frame height.
PLANE_LENGTH = 0.42  # Of the frame width.

ENTER_SECONDS = 0.45
HIT_SECONDS = 0.40
STRIKE_AT = 0.18  # Within each hit: wind-up first, then the swing.
IMPACT_AT = 0.26
HITS = 4
EXIT_STARTS = 2.15
BEAT_SECONDS = 2.8
# Person size (x, y) before the first hit and after each one; the last hit squashes flat.
SIZES = ((1.0, 1.0), (0.62, 0.62), (0.38, 0.38), (0.2, 0.2), (0.14, 0.05))
PUSH = 0.06  # The bat drives the head down before bouncing off.
REST = math.radians(75)  # Bat hanging down in front.
RAISED = math.radians(-140)  # Bat lifted over the shoulder.
RECOIL = math.radians(12)

FLY_IN = 0.9
HOOK_LANDS = 1.45
PULL_STARTS = 1.6
RELEASE_AT = 2.5
RESCUE_SECONDS = 3.2
STRETCH = 1.12  # How tall the person gets on the cable before springing back.

STAR = (40, 220, 255)  # BGR
STAR_EDGE = (20, 110, 230)
CABLE = (40, 40, 44)


class TungSprites(NamedTuple):
    body: np.ndarray
    bat: np.ndarray
    burst: np.ndarray
    plane: np.ndarray


def thumb_down_score(points: np.ndarray) -> float:
    """Thumb straight and pointing down in image pixels (y down), the other fingers folded."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0
    thumb = points[2:5]
    length = float(np.linalg.norm(np.diff(thumb, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0
    direction = thumb[2] - thumb[0]
    straightness = float(np.linalg.norm(direction)) / length
    # Straight and within about 40 degrees of vertical.
    if straightness < 0.85 or direction[1] < float(np.linalg.norm(direction[:2])) * 0.75:
        return 0.0
    # The fist sits above the thumb.
    if thumb[2][1] <= points[[5, 9, 13, 17], 1].max():
        return 0.0
    palm = float(np.linalg.norm(points[9] - points[0]))
    for base in (5, 9, 13, 17):
        if np.linalg.norm(points[base + 3] - points[base]) > palm * 0.65:
            return 0.0
    return float(np.clip(straightness, 0.0, 1.0))


def voice_path(name: str, default: Path) -> Path:
    """The user's own clip for `name` in voci/, otherwise the synthesized voice."""
    clips = sorted(VOICES_DIR.glob(f"{name}.*")) if VOICES_DIR.is_dir() else []
    clips = [clip for clip in clips if clip.suffix.lower() in VOICE_EXTENSIONS]
    return clips[0] if clips else default


def _premultiplied(path: Path) -> np.ndarray:
    sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise RuntimeError(f"Sprite non disponibile: {path}")
    sprite = sprite.astype(np.float32)
    sprite[:, :, :3] *= sprite[:, :, 3:4] / 255.0
    return sprite


def load_tung() -> TungSprites:
    return TungSprites(*(_premultiplied(path) for path in (BODY_PATH, BAT_PATH, BURST_PATH, PLANE_PATH)))


def beating_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Person's centre column, head row and the side the character comes from (-1 left)."""
    height, width = mask.shape
    columns = mask.sum(axis=0)
    total = float(columns.sum())
    center = width / 2 if total < 1.0 else float((columns * np.arange(width)).sum() / total)
    top = person_top(mask)
    top = round(height * 0.15) if top is None else top
    return round(center), top, -1 if center >= width / 2 else 1


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _sprite(sprite: np.ndarray, scale: float, cache: dict | None, key: str,
            flip: int | None = None) -> np.ndarray:
    """Resized (and optionally mirrored) copy of a sprite, cached per size."""
    size = (max(1, round(sprite.shape[1] * scale)), max(1, round(sprite.shape[0] * scale)))
    cache_key = (key, size, flip)
    sized = cache.get(cache_key) if cache is not None else None
    if sized is None:
        sized = cv2.resize(sprite, size, interpolation=cv2.INTER_AREA)
        if flip is not None:
            sized = cv2.flip(sized, flip)
        if cache is not None:
            cache[cache_key] = sized
    return sized


def _place(output: np.ndarray, sprite: np.ndarray, anchor, at, angle: float = 0.0,
           zoom: float = 1.0, opacity: float = 1.0) -> None:
    """Composite a premultiplied BGRA sprite so that its `anchor` pixel lands on `at`,
    turned clockwise by `angle` radians."""
    height, width = sprite.shape[:2]
    c, s = math.cos(angle) * zoom, math.sin(angle) * zoom
    matrix = np.array(((c, -s, at[0] - c * anchor[0] + s * anchor[1]),
                       (s, c, at[1] - s * anchor[0] - c * anchor[1])))
    corners = matrix @ np.array(((0, width, width, 0), (0, 0, height, height), (1, 1, 1, 1)), float)
    x0, y0 = max(0, math.floor(corners[0].min())), max(0, math.floor(corners[1].min()))
    x1 = min(output.shape[1], math.ceil(corners[0].max()) + 1)
    y1 = min(output.shape[0], math.ceil(corners[1].max()) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    matrix[:, 2] -= (x0, y0)
    patch = cv2.warpAffine(sprite, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = patch[:, :, 3:4] * (opacity / 255.0)
    area = output[y0:y1, x0:x1]
    area[:] = np.clip(patch[:, :, :3] * opacity + area * (1.0 - alpha), 0, 255).astype(np.uint8)


def _size_at(age: float) -> tuple[float, float]:
    """Each hit knocks the person down a size with a squash that settles."""
    impacts = [ENTER_SECONDS + index * HIT_SECONDS + IMPACT_AT for index in range(HITS)]
    done = sum(1 for impact in impacts if age >= impact)
    size_x, size_y = SIZES[done]
    if done == 0:
        return size_x, size_y
    since = age - impacts[done - 1]
    if since < PUSH:
        before_x, before_y = SIZES[done - 1]
        push = _ease(since / PUSH)
        return _lerp(before_x, size_x * 1.25, push), _lerp(before_y, size_y * 0.72, push)
    since -= PUSH
    wobble = math.exp(-since / 0.07) * math.cos(2 * math.pi * 5.5 * since)
    return size_x * (1.0 + 0.25 * wobble), size_y * (1.0 - 0.28 * wobble)


def _turn(start: float, end: float, amount: float) -> float:
    """Interpolate an angle the short way round."""
    return start + ((end - start + math.pi) % (2 * math.pi) - math.pi) * amount


def _reach_angle(target_y: float, shoulder_y: float, reach: float) -> float:
    return math.asin(min(0.95, max(-0.8, (target_y - shoulder_y) / reach)))


def _hit_plan(anchor: tuple[int, int, int], height: int, scale: float) -> list[tuple]:
    """For every hit: where the character stands, the bat angles and the head it aims at."""
    center_x, top, side = anchor
    floor_y = height * 0.988
    reach = (BAT_SWEET - BAT_PIVOT[0]) * scale
    shoulder_y = floor_y - (BODY_FLOOR[1] - BODY_SHOULDER[1]) * scale
    shoulder_out = (BODY_SHOULDER[0] - BODY_FLOOR[0]) * scale
    plan = []
    for index in range(HITS):
        target_y = height - SIZES[index][1] * (height - top) - 12 * scale
        angle = _reach_angle(target_y, shoulder_y, reach)
        # Jump when the head is out of reach, otherwise stand on the floor.
        lift = target_y - reach * math.sin(angle) - shoulder_y
        shoulder_x = center_x + side * reach * math.cos(angle)
        # The bat follows the head down to its squashed size.
        squashed_y = height - SIZES[index + 1][1] * 0.72 * (height - top) - 12 * scale
        push = _reach_angle(squashed_y, shoulder_y + lift, reach)
        plan.append((shoulder_x + side * shoulder_out, floor_y + lift, angle, center_x, target_y, push))
    return plan


def _pose(age: float, plan: list[tuple], away: float, floor_y: float, height: int):
    """Floor point and bat angle (as seen from the left) of the character at `age`."""
    hop_height = height * 0.035
    if age < ENTER_SECONDS:
        progress = age / ENTER_SECONDS
        eased = 1.0 - (1.0 - progress) ** 3
        hop = abs(math.sin(2 * math.pi * progress)) * hop_height
        return _lerp(away, plan[0][0], eased), _lerp(floor_y, plan[0][1], eased) - hop, REST
    finished = ENTER_SECONDS + HITS * HIT_SECONDS
    if age < finished:
        index = int((age - ENTER_SECONDS) // HIT_SECONDS)
        local = age - ENTER_SECONDS - index * HIT_SECONDS
        x, y, angle = plan[index][:3]
        if index == 0:
            previous = (x, y, REST)
        else:
            previous = (plan[index - 1][0], plan[index - 1][1], plan[index - 1][5] - RECOIL)
        if local < STRIKE_AT:
            # Wind-up over the shoulder while stepping to the next spot.
            progress = _ease(local / STRIKE_AT)
            hop = math.sin(math.pi * progress) * hop_height if abs(x - previous[0]) > 1 else 0.0
            return (_lerp(previous[0], x, progress), _lerp(previous[1], y, progress) - hop,
                    _turn(previous[2], RAISED, progress))
        if local < IMPACT_AT:
            swing = (local - STRIKE_AT) / (IMPACT_AT - STRIKE_AT)
            return x, y, _lerp(RAISED, angle, swing * swing)
        if local < IMPACT_AT + PUSH:
            return x, y, _lerp(angle, plan[index][5], _ease((local - IMPACT_AT) / PUSH))
        return x, y, plan[index][5] - RECOIL * min(1.0, (local - IMPACT_AT - PUSH) / 0.06)
    x, y = plan[-1][:2]
    if age < EXIT_STARTS:
        return x, y, _lerp(plan[-1][5] - RECOIL, REST, _ease((age - finished) / (EXIT_STARTS - finished)))
    progress = min(1.0, (age - EXIT_STARTS) / (BEAT_SECONDS - 0.05 - EXIT_STARTS))
    hop = abs(math.sin(2 * math.pi * progress)) * hop_height
    return _lerp(x, away, progress * progress), _lerp(y, floor_y, progress) - hop, REST


def _draw_tung(output: np.ndarray, sprites: TungSprites, cache: dict | None, scale: float,
               floor, angle: float, side: int) -> None:
    """Body standing on `floor`; from the right everything is mirrored."""
    mirrored = side > 0
    body = _sprite(sprites.body, scale, cache, "body", 1 if mirrored else None)
    bat = _sprite(sprites.bat, scale, cache, "bat", 0 if mirrored else None)
    width = sprites.body.shape[1]
    foot_x = (width - BODY_FLOOR[0]) if mirrored else BODY_FLOOR[0]
    shoulder_x = (width - BODY_SHOULDER[0]) if mirrored else BODY_SHOULDER[0]
    _place(output, body, (foot_x * scale, BODY_FLOOR[1] * scale), floor)
    shoulder = (floor[0] + (shoulder_x - foot_x) * scale,
                floor[1] + (BODY_SHOULDER[1] - BODY_FLOOR[1]) * scale)
    pivot_y = (sprites.bat.shape[0] - BAT_PIVOT[1]) if mirrored else BAT_PIVOT[1]
    # A mirrored swing is the vertically flipped bat turned the other way round.
    _place(output, bat, (BAT_PIVOT[0] * scale, pivot_y * scale), shoulder,
           math.pi - angle if mirrored else angle)


def _draw_star(output: np.ndarray, center, radius: float) -> None:
    if radius < 1.0:
        return
    angles = -math.pi / 2 + np.arange(10) * math.pi / 5
    radii = np.where(np.arange(10) % 2 == 0, radius, radius * 0.45)
    points = np.stack((center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)), axis=1)
    points = np.round(points).astype(np.int32)
    cv2.fillPoly(output, [points], STAR, cv2.LINE_AA)
    cv2.polylines(output, [points], True, STAR_EDGE, max(1, round(radius * 0.15)), cv2.LINE_AA)


def _draw_impacts(output: np.ndarray, sprites: TungSprites, cache: dict | None,
                  plan: list[tuple], age: float, height: int) -> float | None:
    """TUNG! burst and flying stars; returns the seconds since the latest impact."""
    latest = None
    burst = _sprite(sprites.burst, 0.3 * height / sprites.burst.shape[0], cache, "burst")
    side = 1 if plan[0][0] > plan[0][3] else -1
    for index, (_, _, _, target_x, target_y, _) in enumerate(plan):
        since = age - (ENTER_SECONDS + index * HIT_SECONDS + IMPACT_AT)
        if not 0.0 <= since < 0.32:
            continue
        latest = since
        rng = np.random.default_rng(8800 + index)
        for direction in rng.uniform(-math.pi * 0.95, -math.pi * 0.05, 5):
            distance = height * 0.13 * (1.0 - math.exp(-since / 0.12))
            _draw_star(output, (target_x + math.cos(direction) * distance,
                                target_y + math.sin(direction) * distance),
                       height * 0.024 * (1.0 - since / 0.32))
        # Just above the head, on the side away from the character.
        center = (target_x - side * height * 0.05, max(burst.shape[0] * 0.5, target_y - height * 0.07))
        _place(output, burst, (burst.shape[1] / 2, burst.shape[0] / 2), center,
               angle=0.12 if index % 2 else -0.12, zoom=0.55 + 0.45 * min(1.0, since / 0.05),
               opacity=1.0 - max(0.0, (since - 0.2) / 0.12))
    return latest


def render_squashed(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Only a tiny squashed person is left on the floor."""
    output = background.copy()
    top = person_top(mask)
    if top is not None:
        draw_person(output, frame, mask, top, *SIZES[-1])
    return output


def render_beating(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                   sprites: TungSprites, anchor: tuple[int, int, int], age: float,
                   cache: dict | None = None) -> np.ndarray:
    """Tung Tung Tung Sahur walks in and every hit of the bat makes the person smaller."""
    if age >= BEAT_SECONDS:
        return render_squashed(frame, background, mask)
    height = background.shape[0]
    size_x, size_y = _size_at(age)
    top = person_top(mask)
    if (size_x, size_y) == (1.0, 1.0) and mask is not None:
        output = blend(frame, background, mask)
    else:
        output = background.copy()
        if top is not None:
            draw_person(output, frame, mask, top, size_x, size_y)
    scale = CHARACTER_HEIGHT * height / sprites.body.shape[0]
    side = anchor[2]
    plan = _hit_plan(anchor, height, scale)
    floor_y = height * 0.988
    away = -600 * scale if side < 0 else background.shape[1] + 600 * scale
    x, y, angle = _pose(age, plan, away, floor_y, height)
    _draw_tung(output, sprites, cache, scale, (x, y), angle, side)
    since = _draw_impacts(output, sprites, cache, plan, age, height)
    if since is not None:
        output = shake(output, since, height * 0.02, decay=10.0)
    return output


def _draw_hook(output: np.ndarray, x: float, tip_y: float, size: float) -> tuple[int, int]:
    """Steel hook hanging with its bottom at `tip_y`; returns where the cable is tied."""
    x, tip_y = round(x), round(tip_y)
    radius = max(2, round(size * 0.35))
    thickness = max(2, round(size * 0.13))
    eye = (x + radius, tip_y - round(size * 1.25))
    cv2.line(output, (x + radius, tip_y - radius), eye, (150, 150, 156), thickness, cv2.LINE_AA)
    cv2.ellipse(output, (x, tip_y - radius), (radius, radius), 0, -20, 200, (150, 150, 156), thickness, cv2.LINE_AA)
    cv2.circle(output, eye, max(2, round(size * 0.18)), (90, 90, 96), thickness, cv2.LINE_AA)
    return eye


def render_rescue(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                  sprites: TungSprites, anchor: tuple[int, int, int], age: float,
                  cache: dict | None = None) -> np.ndarray:
    """Bombardiro Crocodilo hooks the squashed person and pulls them back to full size."""
    if age >= RESCUE_SECONDS:
        return frame.copy() if mask is None else blend(frame, background, mask)
    height, width = background.shape[:2]
    center_x, anchor_top, _ = anchor
    top = person_top(mask)
    natural = height - (anchor_top if top is None else top)
    flat_x, flat_y = SIZES[-1]
    pull = _ease((age - PULL_STARTS) / (RELEASE_AT - PULL_STARTS))
    if age < PULL_STARTS:
        size_x, size_y = flat_x, flat_y
    elif age < RELEASE_AT:
        # The cable pulls them tall and thin first; the width catches up.
        size_x, size_y = _lerp(flat_x, 0.94, pull ** 0.6), _lerp(flat_y, STRETCH, pull)
    else:
        wobble = math.exp(-(age - RELEASE_AT) / 0.13) * math.cos(2 * math.pi * 4.2 * (age - RELEASE_AT))
        size_x, size_y = 1.0 - 0.06 * wobble, 1.0 + (STRETCH - 1.0) * wobble
    output = background.copy()
    if top is not None:
        draw_person(output, frame, mask, top, size_x, size_y)

    scale = PLANE_LENGTH * width / sprites.plane.shape[1]
    hover = height * 0.04 + PLANE_HOOK[1] * scale + height * 0.012 * math.sin(2 * math.pi * 1.1 * age)
    landed = height - flat_y * natural
    cable = landed - height * 0.04 - PLANE_HOOK[1] * scale
    hook = None
    tilt = 0.0
    if age < FLY_IN:
        progress = 1.0 - (1.0 - age / FLY_IN) ** 3
        ring = (_lerp(width + 600 * scale, center_x, progress), hover)
    elif age < PULL_STARTS:
        ring = (center_x, hover)
        hook = _lerp(hover + height * 0.03, landed, _ease((age - FLY_IN) / (HOOK_LANDS - FLY_IN)))
    elif age < RELEASE_AT:
        hook = height - size_y * natural
        ring = (center_x - width * 0.08 * pull, hook - cable)
        tilt = 0.35 * pull  # Nose up while climbing away.
    else:
        flight = (age - RELEASE_AT) / 0.35
        hook = height - STRETCH * natural - height * flight * flight
        ring = (center_x - width * (0.08 + 0.25 * flight), hook - cable)
        tilt = 0.35
    if hook is not None:
        # Until the release the hook stays on the person's head, then it flies off with the plane.
        tied = _draw_hook(output, center_x if age < RELEASE_AT else ring[0], hook, height * 0.04)
        cv2.line(output, (round(ring[0]), round(ring[1])), tied, CABLE,
                 max(2, round(height * 0.005)), cv2.LINE_AA)
    plane = _sprite(sprites.plane, scale, cache, "plane")
    _place(output, plane, (PLANE_HOOK[0] * scale, PLANE_HOOK[1] * scale), ring, angle=tilt)
    return output
