"""PyInstaller one-file build for Windows: python -m PyInstaller windows/VanishCam.spec"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


here = Path(SPECPATH).resolve()
root = here.parent
icon_file = here / "vanish_icon.ico"
models = ("selfie_segmenter_landscape.tflite", "gesture_recognizer.task", "hand_landmarker.task",
          "blaze_face_short_range.tflite")
resources = [(root / "models" / name, "models") for name in models]
for folder in ("assets/images", "assets/sounds", "assets/voci"):
    resources += [(path, folder) for path in sorted((root / folder).iterdir()) if path.is_file()]
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
