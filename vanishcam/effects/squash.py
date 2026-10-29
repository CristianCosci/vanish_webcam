"""Squash, stretch and shake helpers shared by the cartoon effects."""

from __future__ import annotations

import math

import cv2
import numpy as np


def blend(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = mask[:, :, None]
    return (frame.astype(np.float32) * alpha
            + background.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def person_top(mask: np.ndarray | None) -> int | None:
    if mask is None:
        return None
    rows = np.flatnonzero((mask > 0.05).any(axis=1))
    return int(rows[0]) if rows.size else None


def draw_person(output: np.ndarray, frame: np.ndarray, mask: np.ndarray, top: int,
                scale_x: float, scale_y: float) -> None:
    """The person standing on the floor of the frame, resized around their centre column."""
    height, width = mask.shape
    tall = round((height - top) * scale_y)
    wide = round(width * scale_x)
    if tall < 1 or wide < 1:
        return
    columns = mask[top:].sum(axis=0)
    center = float((columns * np.arange(width)).sum() / max(float(columns.sum()), 1e-6))
    interpolation = cv2.INTER_AREA if scale_y < 1.0 else cv2.INTER_LINEAR
    person = cv2.resize(frame[top:], (wide, tall), interpolation=interpolation)
    alpha = cv2.resize(mask[top:], (wide, tall), interpolation=interpolation)
    # Feet stay on the floor and the centre column stays put.
    x1, y1 = round(center - center * wide / width), height - tall
    left, right, upper = max(0, x1), min(width, x1 + wide), max(0, y1)
    if left >= right:
        return
    area = np.s_[upper - y1:, left - x1:right - x1]
    weight = alpha[area][:, :, None]
    destination = output[upper:, left:right]
    destination[:] = (person[area].astype(np.float32) * weight
                      + destination.astype(np.float32) * (1.0 - weight)).astype(np.uint8)


def shake(output: np.ndarray, age: float, amplitude: float, decay: float = 6.0) -> np.ndarray:
    """Decaying camera jolt `age` seconds after an impact."""
    amplitude *= math.exp(-decay * age)
    if amplitude < 0.5:
        return output
    matrix = np.float32((
        (1, 0, 0.6 * amplitude * math.sin(2 * math.pi * 7 * age + 1.3)),
        (0, 1, amplitude * math.sin(2 * math.pi * 11 * age)),
    ))
    return cv2.warpAffine(output, matrix, (output.shape[1], output.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)
