# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

For step-by-step instructions in English and Italian, see the [user guide](USER_GUIDE_EN_IT.md).

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python.

The published EXE does not include the Teleport mode. To use it now, run the local scripts or rebuild the EXE from these sources.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`.
2. In the start menu, choose **Normal Vanish**, **Naruto Vanish**, or **Teleport** with the mouse or the `1`, `2`, and `3` keys.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you reappear. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`, and again when you reappear.

**Teleport:** Hold your index and middle fingers straight and together against your forehead for 1 second. Animated vertical comic-style lines run across your silhouette, which squeezes into a thin beam and vanishes; the lines linger briefly afterward. If you break the pose, the countdown starts over. `assets/teleport.wav` plays on both disappearance and reappearance. The terminal shows when the pose is recognized. No text is overlaid on the video.

**Reappearing:** In every mode, lower your hand for half a second, then repeat the mode's gesture (in Teleport mode, hold it for 1 second again). The effect plays in reverse: the dissolve rebuilds you, the smoke swallows the log, or the beam widens back into you. `R` still makes you reappear instantly.

To start a mode without the menu:

```powershell
local_app\AVVIA_LOCALE.bat --mode naruto
local_app\AVVIA_LOCALE.bat --mode normal
local_app\AVVIA_LOCALE.bat --mode teleport
```

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **Webcam - Anteprima**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
local_app\AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear instantly (or lower your hand and repeat the gesture).
- `V`: manually start the selected mode's effect.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
local_app\AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## macOS

The `.bat` launchers are Windows-only: on macOS, create a virtual environment and run `vanish_cam.py` directly. Without `pygame`, the Naruto and Teleport sounds play through `ffplay`. MediaPipe 1.0.1 aborts on Apple Silicon as soon as it loads the hand models, so `requirements.txt` pins 1.2.0.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Teleport mode uses the [MediaPipe BlazeFace short-range model](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md) to estimate the forehead position and `hand_landmarker.task` to track the fingers. The teleport's visual trace is drawn in code, without anime frames, and its sound is synthesized (see [assets/README.md](assets/README.md)).
