"""Synthesize the container drop and crane sounds (44.1 kHz mono 16-bit WAV).

python tools/container_sounds.py assets/sounds
"""

import sys
import wave
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
SR = 44100
rng = np.random.default_rng(2207)


def silence(seconds):
    return np.zeros(int(SR * seconds), np.float64)


def add(track, start, sound):
    begin = int(SR * start)
    end = min(len(track), begin + len(sound))
    track[begin:end] += sound[: end - begin]


def times(seconds):
    return np.arange(int(SR * seconds)) / SR


def sweep(frequency):
    """Sine whose instantaneous frequency follows the given array."""
    return np.sin(2 * np.pi * np.cumsum(frequency) / SR)


def lowpass(signal, cutoff):
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff / SR)
    out = np.empty_like(signal)
    state = 0.0
    for index, value in enumerate(signal):
        state += alpha * (value - state)
        out[index] = state
    return out


def highpass(signal, cutoff):
    return signal - lowpass(signal, cutoff)


def ramp(t, duration):
    return np.clip(t / duration, 0.0, 1.0)


def metal(t, partials, decay_low, decay_high, seed):
    """Inharmonic steel resonance: low partials ring longer than high ones."""
    local = np.random.default_rng(seed)
    out = np.zeros_like(t)
    for index, frequency in enumerate(partials):
        mix = index / max(1, len(partials) - 1)
        decay = decay_low * (1 - mix) + decay_high * mix
        for detune in (0.997, 1.004):  # Pairs beat slowly like a real box.
            out += (np.sin(2 * np.pi * frequency * detune * t + local.uniform(0, 2 * np.pi))
                    * np.exp(-t / decay) / (index + 1) ** 0.6)
    return out / len(partials)


def thud(seconds, start_hz, end_hz, decay):
    t = times(seconds)
    frequency = end_hz + (start_hz - end_hz) * np.exp(-t / 0.12)
    return sweep(frequency) * np.exp(-t / decay) * ramp(t, 0.003)


def noise_burst(seconds, decay, low=None, high=None):
    t = times(seconds)
    burst = rng.normal(0, 1, len(t))
    if high is not None:
        burst = lowpass(burst, high)
    if low is not None:
        burst = highpass(burst, low)
    return burst / max(1e-9, np.abs(burst).max()) * np.exp(-t / decay)


def clink(frequency, decay, level):
    t = times(decay * 8)
    return np.sin(2 * np.pi * frequency * t) * np.exp(-t / decay) * level * ramp(t, 0.001)


def finish(track, name, peak=0.8):
    track = np.tanh(1.5 * track) / np.tanh(1.5)
    fade = int(SR * 0.08)
    track[-fade:] *= np.linspace(1, 0, fade)
    track *= peak / np.abs(track).max()
    data = (track * 32767).astype("<i2")
    with wave.open(str(OUT / name), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SR)
        output.writeframes(data.tobytes())
    print(name, f"{len(track) / SR:.2f}s")


# Drop: falling whistle, then the box hits the floor at 0.84 s (ffplay starts ~60 ms late).
IMPACT = 0.84
drop = silence(2.3)
t = times(IMPACT)
pitch = 1800 * (450 / 1800) ** (t / IMPACT) * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
whistle = sweep(pitch) + 0.12 * sweep(2 * pitch)
whistle += 0.03 * noise_burst(IMPACT, 10.0, low=900, high=4000)
whistle *= ramp(t, 0.06) * (0.15 + 0.07 * t / IMPACT) * np.clip((IMPACT - t) / 0.015, 0, 1)
add(drop, 0.0, whistle)
add(drop, IMPACT, 1.0 * thud(1.6, 75, 38, 0.45))
add(drop, IMPACT, 0.8 * lowpass(noise_burst(0.5, 0.07), 220) * 4)
add(drop, IMPACT, 0.9 * metal(times(1.4), (92, 143, 211, 287, 389, 523, 701, 947, 1279, 1733),
                              0.9, 0.22, 11))
add(drop, IMPACT, 0.35 * noise_burst(0.6, 0.12, low=1200))
for start, frequency, level in ((0.98, 2350, 0.16), (1.07, 3100, 0.12),
                                (1.13, 1950, 0.10), (1.24, 2800, 0.07)):
    add(drop, start, clink(frequency, 0.03, level))
    add(drop, start, level * 0.5 * noise_burst(0.02, 0.004, low=2000))
finish(drop, "container_drop.wav")


def motor(seconds, start_hz, end_hz, level):
    """Winch motor: buzzy harmonics plus a quiet gear whine."""
    t = times(seconds)
    shape = t / seconds
    frequency = start_hz + (end_hz - start_hz) * shape
    out = sum(sweep(frequency * k) / k for k in range(1, 12))
    out = lowpass(out, 900) * 0.9 + 0.12 * sweep(frequency * 7.5)
    rattle = (rng.random(len(t)) < 0.0016) * rng.normal(0, 1, len(t))
    out += 0.6 * lowpass(highpass(rattle, 1500), 6000) * 6
    return out * level * ramp(t, 0.05) * np.clip((seconds - t) / 0.06, 0, 1)


# Crane, shifted 60 ms earlier like the drop: lower, clunk, twistlocks, hoist, boing.
crane = silence(2.4)
add(crane, 0.0, motor(0.64, 95, 78, 0.13))
add(crane, 0.64, 0.6 * thud(0.6, 120, 80, 0.12))
add(crane, 0.64, 0.35 * metal(times(0.5), (310, 470, 690, 1010, 1470), 0.18, 0.06, 5))
for start in (0.72, 0.79):
    add(crane, start, 0.45 * noise_burst(0.06, 0.012, low=2500, high=6000))
    add(crane, start, clink(1200, 0.04, 0.25))
hoist = motor(1.26, 60, 105, 0.17)
hoist *= np.clip(1.0 - (times(1.26) - 0.95) / 0.31, 0.25, 1.0)  # Fades as the box flies away.
add(crane, 0.84, hoist)
t = times(0.8)
boing = 260 + 140 * np.exp(-t / 0.3) * np.sin(2 * np.pi * 13 * t) + 120 * (1 - np.exp(-t / 0.08))
twang = sweep(boing) + 0.4 * sweep(2 * boing) + 0.15 * sweep(3 * boing)
add(crane, 1.44, 0.55 * twang * np.exp(-t / 0.32) * ramp(t, 0.005))
finish(crane, "container_crane.wav")
