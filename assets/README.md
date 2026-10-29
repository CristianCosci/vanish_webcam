# Images and sounds

## `images/`

| File | What it is |
|---|---|
| `substitution_log.png` | The Naruto mode log, an illustration with a transparent background. |
| `container_40hc.png` | The side of a light-blue 40' high-cube container (12.192 × 2.896 m): corrugated panels, rails, corner castings, yellow and black high-cube stripes, the star and MAERSK lettering, ISO 6346 number `MSKU 471225 1` with a valid check digit, `45G1` and `9'6"`. |
| `tung_body.png` | Tung Tung Tung Sahur without the bat arm. Feet at (210, 975), shoulder at (332, 450). |
| `tung_bat.png` | Arm and bat, horizontal, with the shoulder at (30, 60): the app rotates them around the shoulder. |
| `tung_burst.png` | The "TUNG!" burst shown on every hit. |
| `bombardiro.png` | Bombardiro Crocodilo facing left; the cable hangs from the ring at (590, 303). |

The container, Tung Tung Tung Sahur, the burst and Bombardiro Crocodilo are drawn in code by the scripts in [`tools/`](../tools). Smoke, teleport lines, crane, dust, hook and stars are drawn live by the app, so they follow the person and fit any resolution.

## `sounds/`

| File | When it plays |
|---|---|
| `vanish.MP3` | Naruto, when you vanish and when you come back. |
| `teleport.wav` | Teleport: a sweep from 300 to 1800 Hz, a whoosh and a click when you vanish. |
| `container_drop.wav` | Container: a falling whistle, then a metallic thud and small rattles. |
| `container_crane.wav` | Crane: winch, spreader landing, two twistlock clicks, winch going up and a "boing". |
| `tung_beating.wav` | Tung Tung Sahur: wooden footsteps, wooden drum knocks and the swishes of the bat. |
| `bombardiro_rescue.wav` | Bombardiro Crocodilo: propellers, winch, hook, a stretching squeak and a "boing". |
| `tung_voice.wav`, `bombardiro_voice.wav` | Fallback voices from the macOS speech synthesizer: Damayanti, pitched down, for "tung" and "sahur", Rocko for "Bombardiro Crocodilo!". |

Sounds start about 60 ms ahead of the animation to make up for the start-up delay of `ffplay`. The Tung Tung Sahur effects and voices play together and share one gain, so they do not clip when mixed.

## `voci/`

The original meme voices: `tung.wav` (nine "tung" and a "sahur") and `bombardiro.wav` (just "Bombardiro Crocodilo"). Any file in this folder named `tung` or `bombardiro` (MP3, WAV, M4A, OGG, AAC or FLAC) replaces the fallback voice in `sounds/`.
