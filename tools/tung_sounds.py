"""Mix the Tung Tung Tung Sahur and Bombardiro Crocodilo sounds (44.1 kHz mono 16-bit WAV).

Voices come from the macOS speech synthesizer (`say`): Damayanti for "tung" and
"sahur", Rocko (Italian) for "Bombardiro Crocodilo". Everything else is synthesized.

    mkdir -p /tmp/voices && cd /tmp/voices
    say -v Damayanti --file-format=WAVE --data-format=LEI16@44100 -o tung.wav tung
    say -v Damayanti --file-format=WAVE --data-format=LEI16@44100 -o sahur.wav sahur
    say -v "Rocko (Italian (Italy))" --file-format=WAVE --data-format=LEI16@44100 \
        -o bombardiro_Rocko.wav "Bombardiro Crocodilo!"
    python tools/tung_sounds.py /tmp/voices assets/sounds
"""

import sys
import wave
from pathlib import Path

import numpy as np

VOICE_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SR = 44100
LEAD = 0.06  # ffplay starts about 60 ms late.
rng = np.random.default_rng(5150)


def load(name):
    with wave.open(str(VOICE_DIR / name)) as source:
        assert source.getframerate() == SR and source.getnchannels() == 1
        data = np.frombuffer(source.readframes(source.getnframes()), "<i2").astype(np.float64) / 32767
    loud = np.flatnonzero(np.abs(data) > 0.02)
    data = data[max(0, loud[0] - 40):loud[-1] + 400]  # Trim the silence around the word.
    return data / np.abs(data).max()


def deeper(voice, factor):
    """Lower the pitch (and slow down) by resampling."""
    positions = np.arange(0, len(voice) - 1, 1 / factor)
    return np.interp(positions, np.arange(len(voice)), voice)


def times(seconds):
    return np.arange(int(SR * seconds)) / SR


def sweep(frequency):
    return np.sin(2 * np.pi * np.cumsum(frequency) / SR)


def lowpass(signal, cutoff):
    """One-pole low-pass; the cutoff may change over time."""
    alpha = np.broadcast_to(1.0 - np.exp(-2 * np.pi * np.asarray(cutoff, float) / SR), signal.shape)
    out = np.empty_like(signal)
    state = 0.0
    for index, value in enumerate(signal):
        state += alpha[index] * (value - state)
        out[index] = state
    return out


def highpass(signal, cutoff):
    return signal - lowpass(signal, cutoff)


def add(track, start, sound, level=1.0):
    begin = max(0, int(SR * start))
    end = min(len(track), begin + len(sound))
    track[begin:end] += level * sound[: end - begin]


def knock(f0, decay=0.09):
    """Hollow wooden slit drum: a short pitched tok with a bright click."""
    t = times(decay * 7)
    pitch = f0 * (1 + 0.25 * np.exp(-t / 0.01))
    body = sweep(pitch) + 0.45 * sweep(2.1 * pitch) * np.exp(-t / 0.03) + 0.2 * sweep(3.9 * pitch) * np.exp(-t / 0.015)
    click = highpass(rng.normal(0, 1, len(t)), 1800) * np.exp(-t / 0.004) * 0.5
    return (body * np.exp(-t / decay) + click) * np.clip(t / 0.001, 0, 1)


def thud(f0, decay):
    t = times(decay * 6)
    return sweep(f0 * (1 + 0.8 * np.exp(-t / 0.02))) * np.exp(-t / decay)


def whoosh(seconds):
    t = times(seconds)
    noise = rng.normal(0, 1, len(t))
    band = highpass(lowpass(noise, 900 + 2600 * t / seconds), 300)
    return band / np.abs(band).max() * np.sin(np.pi * t / seconds) ** 2


def boing(seconds=0.8):
    t = times(seconds)
    pitch = 260 + 140 * np.exp(-t / 0.3) * np.sin(2 * np.pi * 13 * t) + 120 * (1 - np.exp(-t / 0.08))
    return (sweep(pitch) + 0.4 * sweep(2 * pitch) + 0.15 * sweep(3 * pitch)) * np.exp(-t / 0.32) * np.clip(t / 0.005, 0, 1)


def finish(tracks, names, peak=0.85):
    """Tracks that play together share one gain, taken from their sum."""
    shaped = []
    for track in tracks:
        track = np.tanh(1.4 * track) / np.tanh(1.4)
        fade = int(SR * 0.06)
        track[-fade:] *= np.linspace(1, 0, fade)
        shaped.append(track)
    gain = peak / np.abs(sum(shaped)).max()
    for track, name in zip(shaped, names):
        with wave.open(str(OUT / name), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SR)
            output.writeframes((track * gain * 32767).astype("<i2").tobytes())
        print(name, f"{len(track) / SR:.2f}s", f"picco {np.abs(track * gain).max():.2f}")


# Beating: matches tung_effect (impacts at 0.71, 1.11, 1.51 and 1.91 s).
tung = deeper(load("tung.wav"), 1.4)
sahur = deeper(load("sahur.wav"), 1.3)
beating = np.zeros(int(SR * 2.95))
beating_voice = np.zeros_like(beating)
for step in (0.08, 0.22, 0.36, 2.2, 2.38, 2.56):
    add(beating, step - LEAD, knock(520, 0.03), 0.22)
for index, impact in enumerate((0.71, 1.11, 1.51, 1.91)):
    add(beating, impact - 0.12 - LEAD, whoosh(0.12), 0.25)
    add(beating, impact - LEAD, knock(240 * 1.2 ** index), 0.9)
    add(beating, impact - LEAD, thud(110, 0.05), 0.5)
    add(beating_voice, impact - LEAD - 0.01, tung, 0.75)
add(beating_voice, 2.02 - LEAD, sahur, 0.85)
finish((beating, beating_voice), ("tung_beating.wav", "tung_voice.wav"))

# Rescue: matches the plane timeline (hook lands at 1.45 s, release at 2.5 s).
voice = load("bombardiro_Rocko.wav")
rescue = np.zeros(int(SR * 3.3))
rescue_voice = np.zeros_like(rescue)
t = times(2.9)
doppler = np.interp(t, (0.0, 0.9, 1.6, 2.5, 2.9), (1.08, 1.0, 1.0, 1.07, 1.1))
engines = sum(sweep(f * k * doppler) / k ** 1.2 for f in (68, 71.5) for k in range(1, 9))
engines *= 0.55 + 0.45 * np.sin(2 * np.pi * 24 * t) ** 2  # Propeller beat.
engines = lowpass(engines, 1100) * np.interp(t, (0.0, 0.45, 2.2, 2.9), (0.0, 1.0, 1.0, 0.0))
add(rescue, 0.0, engines / np.abs(engines).max(), 0.3)
add(rescue_voice, 0.15, voice, 0.9)
winch = sweep(np.full(len(times(0.55)), 180.0) + 30 * np.sin(2 * np.pi * 40 * times(0.55)))
add(rescue, 0.9 - LEAD, lowpass(winch, 700) * np.sin(np.pi * times(0.55) / 0.55), 0.18)
add(rescue, 1.45 - LEAD, knock(1650, 0.04), 0.5)
add(rescue, 1.49 - LEAD, knock(2100, 0.03), 0.35)
stretch = times(0.9)
squeak = sweep(300 + 900 * (stretch / 0.9) ** 1.5) * (0.6 + 0.4 * np.sin(2 * np.pi * 17 * stretch))
add(rescue, 1.6 - LEAD, lowpass(squeak, 2500) * np.sin(np.pi * stretch / 0.9) ** 0.5, 0.22)
add(rescue, 2.5 - LEAD, boing(), 0.6)
finish((rescue, rescue_voice), ("bombardiro_rescue.wav", "bombardiro_voice.wav"))
