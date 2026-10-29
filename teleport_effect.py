"""Two fingers on the forehead, comic-style speed lines and a teleport squeeze."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


HOLD_SECONDS = 1.0
POSE_GAP_SECONDS = 0.25
TELEPORT_SECONDS = 0.85
SQUEEZE_STARTS = 0.30
TELEPORT_GONE_AT = 0.42
BEAM_SECONDS = 0.16
TELEPORT_SOUND_PATH = Path(__file__).resolve().parent / "assets" / "teleport.wav"
LINE_COLOR = np.array((255, 244, 214), dtype=np.float32)  # BGR: white with a cyan tint.


def two_fingers_score(points: np.ndarray) -> float:
    """Index and middle straight and side by side, ring and little finger folded."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0
    straightness, lengths, extended, folded = [], [], [], []
    for base in (5, 9, 13, 17):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if base < 13:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.80 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)
    reference_length = min(lengths[:2])
    # Seen from the side, folded fingers can look straight but short.
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # Unlike the V sign, the two fingertips stay close together.
    if float(np.linalg.norm(points[8] - points[12])) > reference_length * 0.45:
        return 0.0
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def forehead_from_keypoints(keypoints, width: int, height: int):
    """Forehead centre, upward direction and eye distance from BlazeFace keypoints."""
    if len(keypoints) < 4:
        return None
    first_eye, second_eye, _, mouth = (
        np.array((point.x * width, point.y * height), dtype=np.float64)
        for point in keypoints[:4]
    )
    eyes = (first_eye + second_eye) * 0.5
    scale = float(np.linalg.norm(second_eye - first_eye))
    upward = eyes - mouth
    norm = float(np.linalg.norm(upward))
    if scale < 4.0 or norm < 1e-6:
        return None
    upward /= norm
    return eyes + upward * scale * 0.7, upward, scale


def fingers_on_forehead(points: np.ndarray, forehead) -> bool:
    """Pads of index and middle (image pixels) resting on the forehead."""
    center, upward, scale = forehead
    pads = points[[7, 8, 11, 12], :2].mean(axis=0)
    offset = pads - center
    along = float(np.dot(offset, upward))
    across = abs(float(offset[0] * upward[1] - offset[1] * upward[0]))
    return across < scale * 1.1 and -scale * 0.6 < along < scale * 1.2


@dataclass
class PoseTimer:
    """Seconds the pose has been held; short detection gaps do not reset it."""

    started: float | None = None
    last_seen: float = float("-inf")

    def update(self, active: bool, now: float) -> float:
        if active:
            if self.started is None or now - self.last_seen > POSE_GAP_SECONDS:
                self.started = now
            self.last_seen = now
        elif now - self.last_seen > POSE_GAP_SECONDS:
            self.started = None
        return 0.0 if self.started is None else now - self.started


def make_speed_lines(count: int = 28, seed: int = 4471) -> np.ndarray:
    """Columns: x in the silhouette box, length, speed, phase, thickness, brightness."""
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(0.04, 0.96, count),
        rng.uniform(0.25, 0.85, count),
        rng.uniform(2.2, 4.5, count),
        rng.uniform(0.0, 1.0, count),
        rng.integers(1, 4, count),
        rng.uniform(0.55, 1.0, count),
    )).astype(np.float32)


def _smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                    lines: np.ndarray, age: float, arriving: bool = False) -> np.ndarray:
    """Lines run over the silhouette, which squeezes into a beam and is gone.

    Arriving plays the same timeline backwards on the live silhouette.
    """
    t = min(TELEPORT_SECONDS, max(0.0, TELEPORT_SECONDS - age if arriving else age))
    output = background.copy()
    points = cv2.findNonZero((mask > 0.05).astype(np.uint8))
    if points is None:
        return output
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 24), max(0, y - 24)
    right = min(mask.shape[1], x + width + 24)
    bottom = min(mask.shape[0], y + height + 24)
    area = np.s_[top:bottom, left:right]
    silhouette = mask[area]
    columns = np.arange(left, right, dtype=np.float32)
    center_x = float((silhouette.sum(axis=0) * columns).sum() / max(float(silhouette.sum()), 1e-6))
    canvas = output[area].astype(np.float32)

    if t < TELEPORT_GONE_AT:
        squeeze = _smooth((t - SQUEEZE_STARTS) / (TELEPORT_GONE_AT - SQUEEZE_STARTS))
        tint = 0.3 * _smooth(t / SQUEEZE_STARTS) + 0.6 * squeeze
        person = frame[area].astype(np.float32) * (1.0 - tint) + LINE_COLOR * tint
        alpha = silhouette
        start = left
        if squeeze > 0.0:
            # Squeeze the person towards its own centre column.
            scale = 1.0 - 0.95 * squeeze
            squeezed = max(1, round((right - left) * scale))
            person = cv2.resize(person, (squeezed, bottom - top), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (squeezed, bottom - top), interpolation=cv2.INTER_AREA)
            start = min(max(left, round(center_x - (center_x - left) * scale)), right - squeezed)
        target = np.s_[:, start - left : start - left + person.shape[1]]
        weight = alpha[:, :, None]
        canvas[target] = person * weight + canvas[target] * (1.0 - weight)

    if t < SQUEEZE_STARTS:
        strength = _smooth(t / SQUEEZE_STARTS)
    elif t < TELEPORT_GONE_AT:
        strength = 1.0
    else:
        strength = 1.0 - _smooth((t - TELEPORT_GONE_AT) / (TELEPORT_SECONDS - TELEPORT_GONE_AT))
    span = bottom - top
    layer = np.zeros(silhouette.shape, dtype=np.float32)
    visible = int(np.ceil(len(lines) * min(1.0, 0.3 + strength)))
    for position, length, speed, phase, thickness, brightness in lines[:visible]:
        line_x = round(position * (right - left - 1))
        middle = span - ((phase + speed * t) % 1.4 - 0.2) * span  # Lines run upwards.
        half = length * span * 0.5
        cv2.line(layer, (line_x, round(middle - half)), (line_x, round(middle + half)),
                 float(brightness), int(thickness), cv2.LINE_AA)
    beam = (t - TELEPORT_GONE_AT) / BEAM_SECONDS
    if 0.0 <= beam < 1.0:
        # A thin bright column marks the spot where the person vanished.
        cv2.line(layer, (round(center_x) - left, 0), (round(center_x) - left, span - 1),
                 1.0 - beam, max(1, round(7 * (1.0 - beam))), cv2.LINE_AA)
    region = cv2.dilate((silhouette > 0.3).astype(np.uint8), np.ones((17, 17), np.uint8))
    region = cv2.GaussianBlur(region.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = cv2.GaussianBlur(layer, (0, 0), sigmaX=3.0, sigmaY=3.0)
    weight = (np.clip(layer + glow * 1.2, 0.0, 1.0) * region * strength * 0.9)[:, :, None]
    canvas = canvas * (1.0 - weight) + LINE_COLOR * weight
    output[area] = np.clip(canvas, 0, 255).astype(np.uint8)
    return output
