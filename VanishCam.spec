"""PyInstaller one-file build for the Windows webcam app."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


root = Path(SPECPATH).resolve()
icon_file = root / "assets" / "vanish_icon.ico"
resources = [
    (root / "hand_landmarker.task", "."),
    (root / "models" / "gesture_recognizer.task", "models"),
    (root / "models" / "selfie_segmenter_landscape.tflite", "models"),
    (root / "assets" / "substitution_log.png", "assets"),
    (root / "assets" / "vanish.MP3", "assets"),
    (root / "assets" / "teleport.wav", "assets"),
    (root / "assets" / "container_40hc.png", "assets"),
    (root / "assets" / "container_drop.wav", "assets"),
    (root / "assets" / "container_crane.wav", "assets"),
    (root / "assets" / "tung_body.png", "assets"),
    (root / "assets" / "tung_bat.png", "assets"),
    (root / "assets" / "tung_burst.png", "assets"),
    (root / "assets" / "bombardiro.png", "assets"),
    (root / "assets" / "tung_beating.wav", "assets"),
    (root / "assets" / "bombardiro_rescue.wav", "assets"),
    (root / "assets" / "tung_voice.wav", "assets"),
    (root / "assets" / "bombardiro_voice.wav", "assets"),
]
missing = [str(path) for path, _ in resources if not path.is_file()]
if not icon_file.is_file():
    missing.append(str(icon_file))
if missing:
    raise SystemExit("File necessari alla build mancanti:\n" + "\n".join(missing))

a = Analysis(
    [str(root / "vanish_cam.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(path), destination) for path, destination in resources],
    hiddenimports=["pygame.mixer", "mediapipe.tasks.c"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VanishCam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(icon_file),
)
