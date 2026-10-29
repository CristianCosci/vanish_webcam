# Substitution log sprite

`substitution_log.png` was generated with the built-in ImageGen tool. The user
image was used as a visual reference for object shape and mood, not as an edit
target. The sprite is an original transparent-background illustration.

Final prompt:

> Create a single ORIGINAL 2D hand-painted anime-style game sprite for live video compositing, using the user's attached image ONLY as a visual reference for the object and mood. Subject: one short thick wooden substitution log floating diagonally (bottom-left to top-right), soft desaturated pale sage-green bark with visible wood grain, dark ink cracks and small dark notches, round cut end with concentric wood rings visible at lower-left. Dramatic clean silhouette, strong cel shading and highlights, polished Japanese animation look. Center the entire log, leave a modest transparent margin around all edges. NO smoke, NO particles, NO background, NO scenery, NO shadow outside the object, NO people, NO text, NO symbols or logos. True transparent alpha background; isolated cutout. Asset should be immediately usable as a sprite over live webcam video.

For smoke, the app uses its own procedurally animated shapes in
`naruto_effect.py`. A [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet)
was reviewed but is not included because the procedural smoke scales and follows
the detected person dynamically.

# Teleport sound

`teleport.wav` is an original sound synthesized with NumPy for Teleport mode: an
exponential sine sweep from 300 to 1800 Hz with a soft noise whoosh, a short
2.4 kHz click at the moment the person vanishes and a decaying high shimmer. No
recorded or third-party audio is included.

# Container

`container_40hc.png` is an original procedural drawing (NumPy, OpenCV and Pillow)
of the side of a light-blue 40' high-cube container (12.192 x 2.896 m) in Maersk
livery: trapezoidal corrugations lit from the upper left, rails, corner posts and
castings, high-cube warning stripes, a seven-pointed star with the MAERSK wordmark
set in Avenir Next Bold, and an ISO 6346 identification (MSKU 471225 1, 45G1,
9'6") with a valid check digit. Maersk and the seven-pointed star are trademarks
of A.P. Møller - Mærsk A/S; the drawing is not affiliated with or endorsed by
Maersk. The crane spreader, dust and shadows are drawn in real time by
`container_effect.py`.

`container_drop.wav` and `container_crane.wav` are original sounds synthesized
with NumPy: a falling slide whistle, then a low boom with inharmonic steel
resonances and small rattles for the drop; a winch motor, the spreader's clunk,
two twistlock clacks and a cartoon spring "boing" for the crane. Both run about
60 ms ahead of the animation to make up for the start-up delay of `ffplay`. No
recorded or third-party audio is included.

# Tung Tung Sahur

`tung_body.png`, `tung_bat.png`, `tung_burst.png` and `bombardiro.png` are
original fan-art drawings of the "Italian brainrot" internet memes Tung Tung
Tung Sahur and Bombardiro Crocodilo, drawn procedurally with NumPy, OpenCV and
Pillow; no existing image was copied or traced. The bat is a separate sprite
rotated around the shoulder at run time, the burst is set in Impact.

`tung_beating.wav` and `bombardiro_rescue.wav` are synthesized: wooden slit-drum
knocks, swishes, a twin-engine propeller drone, a winch, a hook clink, a
stretching squeak and a cartoon "boing". They play together with
`tung_voice.wav` and `bombardiro_voice.wav`, words spoken by the macOS speech
synthesizer: "tung" and "sahur" by the Damayanti voice, pitched down, and
"Bombardiro Crocodilo!" by the Italian Rocko voice. Each pair shares one gain so
that they do not clip when mixed. Clips in `../voci` take the place of the voices.
