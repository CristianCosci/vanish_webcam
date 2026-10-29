"""Where the app finds its assets and models, also inside the Windows build."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
IMAGES = ASSETS / "images"
SOUNDS = ASSETS / "sounds"
# Clips here (tung.mp3, bombardiro.wav, ...) replace the synthesized voices.
VOICES_DIR = ASSETS / "voci"
# MediaPipe models, downloaded on first use.
MODELS = ROOT / "models"
