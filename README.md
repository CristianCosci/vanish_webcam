# VanishCam

Vanish (and reappear) from your webcam with a hand gesture, live, during video calls. VanishCam recognizes the gesture, separates your silhouette from the background and sends the result to the OBS virtual camera, which Teams, Meet, Zoom and other apps see as a regular webcam.

Video is processed on your computer: it is never recorded or sent to a server. The output contains only the video and the effects, with no menus, text or instructions on top. The app's menu and terminal messages are in Italian.

## Modes

| | Mode | Gesture | What happens |
|---|---|---|---|
| 1 | **Normal Vanish** | V sign | you dissolve from the bottom up behind a glowing edge |
| 2 | **Naruto Vanish** | palms together in front of the webcam | a puff of smoke, and a wooden log is left in your place |
| 3 | **Teleport** (*Teletrasporto*) | index and middle fingers together on your forehead, for 1 second | comic-style speed lines, you squeeze into a beam and vanish |
| 4 | **Container** | index finger pointing up, for 1 second | a Maersk 40' HC container falls on your head and flattens you, with a thud, dust and a camera shake |
| 5 | **Tung Tung Sahur** | thumbs down, for 1 second | Tung Tung Tung Sahur walks in with his bat and every hit makes you smaller, until you are squashed flat on the floor |

**To reappear**, lower your hand for half a second and repeat the mode's gesture (holding it for 1 second again where needed):

- the dissolve, the smoke and the teleport play in reverse;
- in Container mode a crane lifts the container away and you stretch back up from the floor with a springy bounce;
- in Tung Tung Sahur mode Bombardiro Crocodilo flies in, hooks you with a cable and pulls you back up to full size.

Every mode has its own sounds. The terminal tells you when a gesture is recognized; nothing appears on the video.

## Installation

### macOS

You need [Homebrew](https://brew.sh), OBS (only for its virtual camera) and `ffmpeg`, which plays the sounds:

```sh
brew install --cask obs
brew install ffmpeg uv
git clone https://github.com/CristianCosci/vanish_webcam.git
cd vanish_webcam
uv venv --python 3.13
uv pip install -r requirements.txt
```

The first time, open OBS and click **Start Virtual Camera**: macOS asks you to allow a system extension. Go to **System Settings → General → Login Items & Extensions**, scroll down to **Extensions**, open **Camera Extensions** (ⓘ) and turn on **OBS Virtual Camera**. Then stop the virtual camera and quit OBS: VanishCam produces the output itself.

To start:

```sh
.venv/bin/python vanish_cam.py --virtual-camera --no-mirror
```

### Windows

Double-click `windows\avvia.bat`: the first time it creates the Python environment in `.venv` and installs everything, then it opens the menu. There are two variants:

- `windows\avvia_virtuale.bat` requires the virtual camera and stops with an error if it is missing;
- `windows\avvia_obs.bat` only shows the preview window, to be captured in OBS with a **Window Capture** source on **Webcam - Anteprima**.

Instead of OBS you can use [Unity Capture](https://github.com/schellingb/UnityCapture) with `--virtual-backend unitycapture`.

`windows\build_exe.bat` builds `dist\VanishCam.exe`, a single executable with the models, images and sounds inside, which does not need Python.

## Usage

1. Start VanishCam and pick a mode from the menu with the mouse or the `1` to `5` keys. To skip the menu, use `--mode normal`, `naruto`, `teleport`, `container` or `tung`.
2. Step out of the frame for 4 seconds while VanishCam captures the background; the terminal tells you when to step back in.
3. Make the gesture.

### In a video call

- Select **OBS Virtual Camera** as your camera.
- Turn off background blur and virtual backgrounds in the call app: VanishCam handles the background.
- Start with `--no-mirror`, so that people watching you read text the right way round (for example on the container). Teams mirrors only your own preview; everyone else sees you unmirrored.
- Only you hear the sounds: the virtual camera carries video only.

### Keys in the preview window

| Key | Action |
|---|---|
| `SPACE` | capture the background again |
| `R` | reappear instantly |
| `V` | start the effect without the gesture |
| `F` | toggle fullscreen |
| `Q` or `ESC` | quit and release the webcam |

### Options

| Option | |
|---|---|
| `--mode MODE` | skip the start menu |
| `--camera N` | webcam to use (default: 0) |
| `--width`, `--height`, `--fps` | output resolution and maximum FPS (defaults: 960×540, 30) |
| `--no-mirror` | do not mirror the image |
| `--virtual-camera`, `--no-virtual-camera` | require or disable the virtual camera; by default it is used when available |
| `--virtual-backend`, `--virtual-device` | virtual camera driver (`obs`, `unitycapture`, `v4l2loopback`) and device |
| `--fullscreen` | open the preview in fullscreen |
| `--gesture-debug` | print what gesture recognition sees in the terminal, useful when a gesture does not trigger |

### Tung Tung Sahur voices

The original meme voices are in `assets/voci` (`tung.wav` and `bombardiro.wav`). To change them, replace them with a file named `tung` or `bombardiro` (MP3, WAV, M4A, OGG, AAC or FLAC). When the folder is empty, VanishCam uses the fallback voices in `assets/sounds`, made with the macOS speech synthesizer. Knocks, propellers and the "boing" always stay in sync with the animation.

## Project layout

```text
vanish_cam.py          entry point (same as python -m vanishcam)
vanishcam/
  app.py               menu, webcam, gesture recognition and video loop
  paths.py             where assets and models live
  sound.py             sound playback (pygame on Windows, ffplay elsewhere)
  effects/
    naruto.py          smoke and log
    teleport.py        teleport lines and beam
    container.py       container, crane, dust
    tung.py            Tung Tung Tung Sahur and Bombardiro Crocodilo
    squash.py          squashing, stretching and shaking the silhouette
assets/                images, sounds and voices (details in assets/README.md)
models/                MediaPipe models, downloaded on first run
tests/                 automated tests
tools/                 scripts that draw the images and synthesize the sounds
windows/               Windows launchers, executable build and icon
```

## Development

The tests run without a webcam:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The container, Tung Tung Tung Sahur, Bombardiro Crocodilo and their sounds can be regenerated with the scripts in `tools/`; usage is at the top of each script.

MediaPipe is pinned to 1.2.0: with 1.0.1 on Apple Silicon the hand model crashes the app as soon as it loads.

## Credits

The Normal and Naruto modes and the base of the app come from [vanish_webcam](https://github.com/Nicovic17/vanish_webcam) by Nicovic17.
